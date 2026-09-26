# %% [markdown]
# # 03 · Multi-GPU topology and P2P
#
# **Tier:** T0 — read `nvidia-smi` output (bundled **sample output in the documented format,
# illustrative**, or your machine's if it has a GPU), make placement decisions from it, and predict
# GPU-to-GPU bandwidth with a model. **T2** — with two or more GPUs, measure the P2P bandwidth matrix
# and fit α and β of the link: Kaggle's free 2×T4 (PCIe), a 2–8 GPU NVLink pod on RunPod/Vast/Lambda,
# or GCP `a2-highgpu-2g` (see `deploy/`). Concepts: primer §5 "Fabrics quantitatively"
# ([`../../PRIMER.md`](../../PRIMER.md)).
#
# **Predicted first in** [roofline-core notebook 03](../../roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb):
# there you derived the ring all-reduce by running one, priced TP against an ITL target and read a
# topology matrix. Here you read real `nvidia-smi` output, explain a measured P2P number against the
# topology model, and price TP with the α you *measured* — once you know which α that is.
#
# ## The one-minute version
#
# Inside one server, "GPU to GPU" can mean 450 GB/s each way over NVLink or a few GB/s staged
# through host memory across CPU sockets. `nvidia-smi topo -m` tells you which before you measure,
# and a P2P bandwidth matrix confirms it. Three decisions follow directly: a tensor-parallel group
# goes where every pair is on NVLink; each GPU uses the NIC on its own PCIe switch (GPUDirect RDMA,
# rail alignment); and the process feeding a GPU runs on that GPU's NUMA node. Then the α-β model
# prices the all-reduce that tensor parallelism performs twice per layer — and shows that at decode
# batch sizes it is the latency α, not the bandwidth, that you pay.

# %%
from IPython.display import Markdown, display

from gpubench import inventory, p2p, specs, topo, transfer
from gpubench.backends import get_backend, gpu_count
from gpubench.measure import si
from gpubench.report import measurements_markdown

QUICK = True
ngpu = gpu_count()
local_topo = topo.run_nvidia_smi()
local_rows, local_raw = inventory.query_gpus()
print(f"GPUs visible to PyTorch: {ngpu} | nvidia-smi on this machine: {'yes' if local_topo else 'no'}")
SAMPLE = "SAMPLE OUTPUT in the documented format (illustrative) — not a measurement of any machine"

# %% [markdown]
# ## 1 · What is in the box
#
# Start with the inventory: model, PCIe link, power limit, memory in use. Many "slow GPU" reports
# end here — a link that trained at x8, a power cap, a stray process holding memory.

# %%
inv = inventory.parse_csv(inventory.load_fixture("hgx-h100-8gpu"))
print(f"{SAMPLE}: an 8×H100 server\n")
print(f"{'gpu':>3} {'PCIe now/max':>14} {'power W':>12} {'used MiB':>9}")
for r in inv:
    print(f"{r['index']:>3} {'Gen%s x%s / Gen%s x%s' % (r['pcie.link.gen.current'], r['pcie.link.width.current'], r['pcie.link.gen.max'], r['pcie.link.width.max']):>14}"
          f" {'%g/%g' % (r['power.limit'], r['power.default_limit']):>12} {r['memory.used']:>9}")
if local_rows:
    print("\nthis machine:")
    for f in inventory.health(local_rows) or ["no findings"]:
        print(" ", f)

# %% [markdown]
# ## Exercise 3.1 — find the GPU with a degraded link
#
# Return the indices of GPUs whose PCIe link trained narrower than its maximum
# (`pcie.link.width.current < pcie.link.width.max`). Width, not generation: an idle link drops to a
# lower generation to save power and comes back under load; a missing lane never does.

# %% exercise
def degraded_links(rows):
    ### BEGIN SOLUTION
    return [r["index"] for r in rows
            if r.get("pcie.link.width.current") is not None and r.get("pcie.link.width.max") is not None
            and r["pcie.link.width.current"] < r["pcie.link.width.max"]]
    ### END SOLUTION

# %% check
assert degraded_links(inv) == [5]
assert degraded_links(inventory.parse_csv(inventory.load_fixture("colab-t4"))) == []   # Gen1 at idle is not a fault
for f in inventory.health(inv):
    print(f)
print("✅ GPU 5 runs at x8: half the host↔device bandwidth of its neighbours")

# %% [markdown]
# ## 2 · Reading nvidia-smi topo -m
#
# Every cell names the path between two devices, best to worst:
#
# | code | path | typical GPU↔GPU rate |
# |---|---|---|
# | `NV#` | a bonded set of # NVLinks (through NVSwitch on HGX boards) | 25–50 GB/s per link per direction |
# | `PIX` | at most one PCIe bridge — the same PCIe switch | the PCIe link rate |
# | `PXB` | several PCIe bridges, not the host bridge | the PCIe link rate |
# | `PHB` | through the CPU's PCIe host bridge (root complex) | the PCIe link rate *if* peer access works there; many platforms and most VMs disable it, and then copies are staged (~half the link or less) |
# | `NODE` | across host bridges inside one NUMA node | usually no direct P2P: staged |
# | `SYS` | across the socket interconnect (UPI/Infinity Fabric) | no direct P2P: staged through host memory |

# %%
text = topo.load_fixture("hgx-h100-8gpu")
print(f"{SAMPLE}: HGX H100 node\n")
print("\n".join(text.splitlines()[:17]))
hgx = topo.parse(text)
print("\n" + hgx.summary())

# %% [markdown]
# ## Exercise 3.2 — rank the paths
#
# Write `my_rank(code)`, a sortable key where a better path is larger: any `NV#` beats every PCIe
# path, and more NVLinks beat fewer; then `PIX > PXB > PHB > NODE > SYS`.

# %% exercise
def my_rank(code):
    ### BEGIN SOLUTION
    if code.startswith("NV"):
        return (5, int(code[2:]))
    return ({"PIX": 4, "PXB": 3, "PHB": 2, "NODE": 1, "SYS": 0}[code], 0)
    ### END SOLUTION

# %% check
best_to_worst = ["NV18", "NV12", "NV4", "PIX", "PXB", "PHB", "NODE", "SYS"]
assert sorted(reversed(best_to_worst), key=my_rank, reverse=True) == best_to_worst
assert all((my_rank(a) > my_rank(b)) == (topo.link_rank(a) > topo.link_rank(b)) for a in best_to_worst for b in best_to_worst)
print("✅ NV# > PIX > PXB > PHB > NODE > SYS")

# %% [markdown]
# ## 3 · Where does a tensor-parallel group go?
#
# Tensor parallelism all-reduces activations inside every layer, so a TP group is only as fast as its
# **worst** pair. Here is a two-socket PCIe server (sample output): two GPUs under each socket's PCIe
# switch, one NIC.

# %%
pcie_text = topo.load_fixture("pcie-4gpu-2socket")
print(f"{SAMPLE}: 4 GPUs, 2 sockets, no NVLink\n")
print("\n".join(pcie_text.splitlines()[:6]))
pcie_box = topo.parse(pcie_text)

# %% [markdown]
# ## Exercise 3.3 — choose the group
#
# Write `my_best_group(t, k)`: among all `k`-GPU subsets of `t.gpus`, return the one whose worst
# pairwise path (by `my_rank`) is best; break ties by the sorted list of all its pair ranks, then by
# the lowest GPU ids (the order `itertools.combinations` produces). `t.link(a, b)` gives a path code.

# %% exercise
from itertools import combinations


def my_best_group(t, k):
    ### BEGIN SOLUTION
    def score(group):
        ranks = sorted(my_rank(t.link(a, b)) for a, b in combinations(group, 2))
        return (ranks[0], ranks)
    return max(combinations(t.gpus, k), key=score)
    ### END SOLUTION

# %% check
assert my_best_group(pcie_box, 2) == ("GPU0", "GPU1")                     # the PIX pair, not a SYS pair
assert my_best_group(hgx, 4) == topo.best_group(hgx, 4) == ("GPU0", "GPU1", "GPU2", "GPU3")
assert my_best_group(topo.parse(topo.load_fixture("kaggle-2xt4")), 2) == ("GPU0", "GPU1")
print("✅ TP=2 on the PCIe box: GPU0+GPU1 (same switch). On NVSwitch every group is equal — the fabric is flat")

# %% [markdown]
# ## 4 · Which NIC, which CPU cores?
#
# For multi-node traffic (GPUDirect RDMA), each GPU should use the NIC with the best path to it —
# on its own PCIe switch tree (`PIX`/`PXB`, never through the host bridge) — so the NIC reads GPU
# memory without crossing the CPU; one NIC per GPU is a **rail**. And the process that feeds a GPU
# (tokenising, staging batches) belongs on that GPU's NUMA node, so its host buffers do not cross the
# socket link on every copy. roofline-core's Ex 3.5 picked these from a matrix; the lab's
# `topo.nearest_nic` and `topo.cpu_list` do it on real `nvidia-smi` output:

# %%
for g in ("GPU0", "GPU5"):
    nic, code = topo.nearest_nic(hgx, g)
    print(f"HGX {g} → {nic} ({hgx.nic_names[nic]}, {code}); feed it from cores {hgx.cpu_affinity[g]} "
          f"(NUMA node {hgx.numa_affinity[g]}, {len(topo.cpu_list(hgx, g))} CPUs)")
for g in pcie_box.gpus:
    print(f"PCIe box {g} → {' via '.join(topo.nearest_nic(pcie_box, g)[::-1])}")
print("→ on the PCIe box GPU2/GPU3 reach the only NIC across the sockets (SYS): a bottleneck for multi-node traffic")

# %% [markdown]
# ## 5 · From topology to predicted bandwidth
#
# A path code plus the link generation gives a theoretical per-direction rate (`p2p.predict`):
# NVLink links × the per-link rate of the generation; the PCIe link rate `GT/s × lanes × 128/130 ÷ 8`
# when the two GPUs have **peer access**; and about half of it when they do not and the driver
# stages the copy through host memory — two PCIe crossings plus a host copy (measured staged copies
# often land lower still). Whether peer access works is a property of the platform, not of the
# topology code: through a switch (`PIX`/`PXB`) it normally does, across host bridges or sockets
# (`NODE`/`SYS`) it rarely does, and through the root complex (`PHB`) it depends — many platforms
# and most VMs turn it off. So a `PHB` estimate made from a saved topology, without asking the
# driver, is an **upper bound**. This is a **model**; the measurement in section 6 is what you trust.

# %%
for name, arch, gen in (("hgx-h100-8gpu", "hopper", 5), ("a2-highgpu-2g", "ampere", 4),
                        ("pcie-4gpu-2socket", None, 4), ("kaggle-2xt4", "turing", 3)):
    t = topo.parse(topo.load_fixture(name))
    pred = p2p.predict(t, arch=arch, pcie_gen=gen)
    print(f"{name} ({SAMPLE.split(' —')[0].lower()}), PCIe Gen{gen}:")
    for a, b2, code in t.gpu_pairs()[:3]:
        e = pred[(a, b2)]
        print(f"   {a}→{b2}  {code:>5}  ≈ {e.gbs:6.1f} GB/s per direction (model, {e.mode}): {e.why}")
kaggle = topo.parse(topo.load_fixture("kaggle-2xt4"))
off = p2p.predict(kaggle, arch="turing", pcie_gen=3, peer_access={("GPU0", "GPU1"): False})[("GPU0", "GPU1")]
print(f"kaggle-2xt4 if the driver reports no peer access: ≈ {off.gbs:.1f} GB/s ({off.mode}): {off.why}")

# %% [markdown]
# ## Exercise 3.4 — explain a measured P2P number
#
# A measurement means little until you say what explains it. Write `p2p_verdict(measured_gbs,
# direct_gbs, peer_access)` returning one of:
#
# * `"staged"` — the driver reports no peer access, so the copy went through host memory; a number
#   near or below half the link is expected, not a fault;
# * `"direct"` — peer access and at least 60% of the direct-path model (`direct_gbs`, what the
#   link would give with peer access): healthy;
# * `"degraded"` — peer access, yet below 60% of the model: something between the GPUs is wrong
#   (ACS or an IOMMU forcing P2P through the root complex, a link trained narrower, a busy GPU).

# %% exercise
def p2p_verdict(measured_gbs, direct_gbs, peer_access):
    ### BEGIN SOLUTION
    if not peer_access:
        return "staged"
    return "direct" if measured_gbs >= 0.6 * direct_gbs else "degraded"
    ### END SOLUTION

# %% check
nv = p2p.path_bandwidth("NV18", "hopper")
assert p2p_verdict(390, nv.gbs, True) == "direct"                 # 87% of 450 GB/s over NVLink
assert p2p_verdict(120, nv.gbs, True) == "degraded"               # NV18 on paper, a quarter of it measured
direct_t4 = p2p.path_bandwidth("PHB", pcie_gen=3, peer_access=True).gbs
assert p2p_verdict(6.0, direct_t4, False) == "staged"             # e.g. a 2×T4 VM without peer access
assert p2p_verdict(13.0, direct_t4, True) == "direct"
print("✅ a P2P number is explained by the path *and* by whether the driver granted peer access")

# %% [markdown]
# ## 6 · Measure it (T2)
#
# With two or more GPUs, the lab copies a buffer between every ordered pair (and both directions at
# once), then sweeps sizes on one pair to fit α and β of the path — the same experiment as NVIDIA's
# `p2pBandwidthLatencyTest` and `nvbandwidth`, which remain the reference tools.

# %%
pipelined_ab = latency_ab = None
if ngpu >= 2:
    be = get_backend("torch")
    size = (64 << 20) if QUICK else (256 << 20)
    uni = p2p.measure_matrix(be, size)
    bi = p2p.measure_matrix(be, size, bidirectional=True)
    display(Markdown(measurements_markdown(uni + bi)))
    pipelined_ab = transfer.fit(p2p.size_sweep(be, 0, 1, transfer.sizes(4 << 10, 64 << 20, 4)))
    latency_ab = transfer.fit(p2p.size_sweep(be, 0, 1, transfer.sizes(4 << 10, 4 << 20, 4), repeats=20,
                                             min_time=0.0, latency=True))
    for label, ab in (("back to back (pipelined)", pipelined_ab), ("one synchronised copy (latency)", latency_ab)):
        print(f"GPU0→GPU1 {label:<32}: α = {si(ab.alpha, 's')}, β = {si(ab.beta, 'B/s')}, n½ = {si(ab.n_half, 'B')}")
    peer = uni[0].extras.get("peer_access")
    print(f"peer access GPU0→GPU1: {peer} ({uni[0].note})")
    if local_topo:
        t = topo.parse(local_topo)
        spec = specs.lookup(be.describe()["name"])
        link = transfer.host_link(be.describe()["name"])
        code = t.link("GPU0", "GPU1")
        e = p2p.path_bandwidth(code, spec.arch if spec else None, link["gen"], link["width"], peer)
        direct = p2p.path_bandwidth(code, spec.arch if spec else None, link["gen"], link["width"], True)
        got = uni[0].bytes_per_s() / 1e9
        print(f"model for GPU0→GPU1 ({code}, {e.mode}): {e.gbs:.1f} GB/s per direction — measured {got:.1f} GB/s "
              f"({got / e.gbs:.0%}); verdict: {p2p_verdict(got, direct.gbs, peer)}")
else:
    print(f"{ngpu} GPU(s) visible: P2P needs two. To run this section for real:\n"
          "  • Kaggle (free): Notebook → Settings → Accelerator → 'GPU T4 x2'; clone the repo, pip install -e the lab,\n"
          "    and re-run this notebook (PCIe through the root complex: direct if the VM grants peer access, else staged).\n"
          "  • RunPod / Vast / Lambda: a 2–8 GPU SXM (NVLink) machine for an hour; or GCP a2-highgpu-2g (deploy/gcp).\n"
          "  • Anywhere: python -m gpubench run --suite inventory,p2p --out results\n"
          "  • Reference tools: nvidia-smi topo -m, nvidia-smi nvlink --status, nvbandwidth.")

# %% [markdown]
# ## 7 · What the link is worth: the TP all-reduce per token
#
# Megatron-style tensor parallelism all-reduces the `batch × hidden` activations twice per layer.
# A ring all-reduce of `S` bytes over `p` GPUs takes `2(p−1)` steps, each paying a fixed cost α and
# moving `S/p` bytes: **`t = 2(p−1)·α + 2·(p−1)/p · S/B`** (primer §5.2, `p2p.ring_allreduce_time`;
# roofline-core Ex 3.1 derives it by running a ring). The table prices a 70B-class model's decode
# step with the primer's §5.1 link table — the α values there are illustrative per-step latencies,
# for the model; measure yours with [nccl-tests](../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/any-gpu/README.md)
# ([layer 02 §5](../../../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md)). The "2 nodes" row runs one
# *flat* ring through the NICs; NCCL's hierarchical all-reduce over rails does better (primer §5.3).

# %%
hidden, layers, p = 8192, 80, 8
hbm_step = 140e9 / p / specs.get("h100-sxm").mem_bw          # each GPU streams its 1/8 of the weights per step
fabrics = [("NVLink 4 via NVSwitch", 2e-6, 450e9),              # primer §5.1
           ("PCIe Gen4 x16, peer access", 4e-6, 31.5e9),        # primer §5.1
           ("PCIe Gen4 x16, staged (no P2P)", 10e-6, 15.75e9),  # ILLUSTRATIVE: ~half the link, a slower α
           ("2 nodes, 400 Gb/s NICs, flat ring", 5e-6, 50e9)]   # primer §5.1 (IB NDR)
print(f"HBM time per decode step (weights only, H100): {si(hbm_step, 's')}\n")
print(f"{'fabric':<34} {'batch':>5} {'per all-reduce':>15} {'per step (160)':>15} {'α share':>8}")
for name, alpha, bw in fabrics:
    for batch in (1, 64):
        one = p2p.ring_allreduce_time(batch * hidden * 2, p, alpha, bw)
        print(f"{name:<34} {batch:>5} {si(one, 's'):>15} {si(layers * 2 * one, 's'):>15} {2 * (p - 1) * alpha / one:>8.0%}")

# %% [markdown]
# At decode the α term is nearly everything, so the α you plug in *is* the answer — and section 6
# measured two. Each step of a ring waits for the previous step's data to arrive before it can
# forward it: a dependent chain, which pays the full **latency** of a copy. Back-to-back copies of
# independent buffers overlap their fixed costs, so the **pipelined** fit's α is smaller and would
# price the all-reduce too low. β is the other way round: it is a large-copy property, pinned best by
# the pipelined sweep, which reaches the large sizes (the latency sweep stops at a few MB).
#
# ## Exercise 3.5 — price TP with the right measured α
#
# Write `tp_allreduce_from_fits(nbytes, p, latency_fit, pipelined_fit)`: the time of one ring
# all-reduce of `nbytes` over `p` GPUs, taking α and β each from the fit that measures it. Use
# `p2p.ring_allreduce_time(nbytes, p, alpha, bw)`; the fits are `AlphaBeta(alpha, beta)` records.

# %% exercise
def tp_allreduce_from_fits(nbytes, p, latency_fit, pipelined_fit):
    ### BEGIN SOLUTION
    return p2p.ring_allreduce_time(nbytes, p, latency_fit.alpha, pipelined_fit.beta)
    ### END SOLUTION

# %% check
from gpubench.timing import AlphaBeta

lat_fit, pipe_fit = AlphaBeta(alpha=12e-6, beta=9e9), AlphaBeta(alpha=3e-6, beta=11e9)   # illustrative fits
t = tp_allreduce_from_fits(16384, 2, lat_fit, pipe_fit)
assert abs(t - (2 * 12e-6 + 16384 / 11e9)) < 1e-15                   # latency α, large-copy β
assert t > p2p.ring_allreduce_time(16384, 2, pipe_fit.alpha, pipe_fit.beta)   # the pipelined α would under-price it
if latency_ab is not None:
    tp2 = layers * 2 * tp_allreduce_from_fits(hidden * 2, 2, latency_ab, pipelined_ab)
    print(f"your GPU0↔GPU1 link, TP=2, batch 1: {si(tp2, 's')} of all-reduce per decode step "
          f"(with the pipelined α it would read {si(layers * 2 * p2p.ring_allreduce_time(hidden * 2, 2, pipelined_ab.alpha, pipelined_ab.beta), 's')})")
print("✅ a dependent chain pays the latency α; a bandwidth term takes the large-copy β")

# %% [markdown]
# At decode batch sizes an all-reduce moves a few kilobytes, so the bandwidth term is microseconds
# or less and the **α term is almost everything** — and it is paid 160 times per token. That is why
# TP stays inside the NVLink domain (lowest α, highest B), why engines ship custom one-shot
# all-reduce kernels and capture decode in CUDA graphs, and why TP across nodes is a last resort
# (the [parallelism menu](../../../gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu)
# puts pipeline and data parallelism there instead).
#
# ## In a design review
#
# **The two-minute version.** "Before choosing a parallelism layout I read `nvidia-smi topo -m`.
# NV# pairs get ~450 GB/s each way on H100; PIX/PXB pairs get the PCIe link rate; PHB gets it only
# if the platform grants peer access through the root complex — many do not, VMs especially — and
# NODE and SYS usually mean P2P is staged through host memory at half the link or less. Tensor
# parallelism goes where every pair is NV#, because its all-reduce runs twice per layer and at
# decode it is latency-bound: at batch 1 the bandwidth term is nanoseconds and α is the cost — the
# latency of a copy you wait for, not the issue cost of back-to-back copies. Each GPU uses the NIC
# on its own PCIe switch tree for GPUDirect RDMA, and its host process runs on its NUMA node. Then I
# measure the P2P matrix and check peer access: staged is expected without it; a pair with peer
# access far below the model means something between the GPUs is misconfigured."
#
# **Drills**
#
# 1. *Two nodes of 8×H100 — TP=16 for a 405B model?* — No: TP's per-layer all-reduces would cross
#    the NICs (~50 GB/s per NIC and a higher α instead of 450 GB/s NVLink). TP=8 inside each node,
#    pipeline or data parallelism across nodes (primer §5.3).
# 2. *A 2×T4 VM (Kaggle-style, path PHB) measures a third of its Gen3 x16 link GPU-to-GPU —
#    illustrative numbers. Broken?* — Not necessarily. The model's 15.8 GB/s for PHB assumes direct
#    peer access through the root complex; check `torch.cuda.can_device_access_peer` (the lab records
#    it). Without peer access the copy is staged through host memory — two PCIe crossings and a host
#    copy, about half the link at best, often less — which is expected, not a fault. With peer access
#    granted, a third of the link *is* a finding (ACS/IOMMU settings, link width). Either way: fine
#    for pipeline or data parallelism, poor for TP.
# 3. *Why pin GPU5's feeding process to cores 48–95 on the HGX box?* — GPU5 hangs off socket 1: host
#    buffers on socket 1's memory avoid the inter-socket link on every DMA, and its RDMA NIC (NIC5,
#    PXB) sits on the same socket.
