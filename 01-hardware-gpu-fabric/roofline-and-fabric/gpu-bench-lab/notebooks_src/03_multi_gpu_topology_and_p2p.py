# %% [markdown]
# # 03 · Multi-GPU topology and P2P
#
# **Tier:** T0. Read `nvidia-smi` output, decide the placement from it, and predict GPU-to-GPU bandwidth with a
# model. The output is the bundled **sample output in the documented format, illustrative**, or the output of your
# machine if it has a GPU. **T2**: with two or more GPUs, measure the P2P bandwidth matrix and fit $\alpha$ and $\beta$
# of the link. Use Kaggle's free 2×T4 (PCIe), a 2–8 GPU NVLink pod on RunPod/Vast/Lambda,
# or GCP `a2-highgpu-2g` (see `deploy/`). Concepts: primer §5 "Fabrics quantitatively"
# ([`../../PRIMER.md`](../../PRIMER.md)).
#
# **Predicted first in** [roofline-core notebook 03](../../roofline-core/notebooks/03_fabrics_and_collective_cost.ipynb):
# in that notebook, you ran a ring all-reduce and derived the ring all-reduce from it. You also calculated the cost of
# TP against an ITL target, and you read a topology matrix. In this notebook, you read real `nvidia-smi` output. You
# explain a measured P2P number with the topology model. Then you calculate the cost of TP with the $\alpha$ that you
# *measured*, when you know which $\alpha$ that is.
#
# ## The one-minute version
#
# Inside one server, "GPU to GPU" can mean 450 GB/s each way over NVLink. It can also mean a few GB/s, staged
# through host memory across CPU sockets. `nvidia-smi topo -m` tells you which before you measure, and a P2P bandwidth
# matrix then shows the same result. Three decisions come directly from this:
#
# - A tensor-parallel group goes where every pair is on NVLink.
# - Each GPU uses the NIC on its own PCIe switch (GPUDirect RDMA, rail alignment).
# - The process that feeds a GPU runs on the NUMA node of that GPU.
#
# Then the α-β model calculates the cost of the all-reduce that tensor parallelism does two times for each layer. The
# model also shows that at decode batch sizes, you pay the latency $\alpha$, not the bandwidth.

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
# Start with the inventory: model, PCIe link, power limit, memory in use. Many "slow GPU" reports end at this
# step. Examples are a link that trained at x8, a power cap, and an unexpected process that holds memory.

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
# Return the indices of the GPUs whose PCIe link trained narrower than its maximum
# (`pcie.link.width.current < pcie.link.width.max`). Use the width, not the generation. An idle link goes down to a
# lower generation to save power, and it comes back under load. A lane that is not there never comes back.

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
# Each cell gives the name of the path between two devices. The table shows the paths from best to worst:
#
# | code | path | typical GPU-to-GPU rate |
# |---|---|---|
# | `NV#` | a bonded set of # NVLinks (through NVSwitch on HGX boards) | 25–50 GB/s per link per direction |
# | `PIX` | at most one PCIe bridge, that is, the same PCIe switch | the PCIe link rate |
# | `PXB` | several PCIe bridges, not the host bridge | the PCIe link rate |
# | `PHB` | through the CPU's PCIe host bridge (root complex) | the PCIe link rate *if* peer access works there. Many platforms and most VMs disable it, and then the driver stages the copies (~half the link or less) |
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
# Write `my_rank(code)`. It is a key for a sort, and a better path has a larger key. Any `NV#` is better than every
# PCIe path, and more NVLinks are better than fewer. After these, the order is `PIX > PXB > PHB > NODE > SYS`.

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
# Tensor parallelism does an all-reduce of the activations inside each layer. Thus a TP group is only as fast as its
# **worst** pair. The next cell shows a two-socket PCIe server (sample output). It has two GPUs under the PCIe switch
# of each socket, and one NIC.

# %%
pcie_text = topo.load_fixture("pcie-4gpu-2socket")
print(f"{SAMPLE}: 4 GPUs, 2 sockets, no NVLink\n")
print("\n".join(pcie_text.splitlines()[:6]))
pcie_box = topo.parse(pcie_text)

# %% [markdown]
# ## Exercise 3.3 — choose the group
#
# Write `my_best_group(t, k)`. From all the $k$-GPU subsets of `t.gpus`, return the subset whose worst pairwise path
# (by `my_rank`) is best. If two subsets have the same score, compare the sorted lists of all their pair ranks. If they
# are still equal, use the lowest GPU ids (the order that `itertools.combinations` produces). `t.link(a, b)` gives a
# path code.

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
# For multi-node traffic (GPUDirect RDMA), the correct NIC for a GPU is the NIC with the best path to it. That NIC is
# on the PCIe switch tree of the GPU (`PIX`/`PXB`, never through the host bridge). Thus the NIC reads GPU memory and
# does not go through the CPU. One NIC for each GPU is a **rail**.
#
# Also, the correct place for the process that feeds a GPU is the NUMA node of that GPU. This process tokenizes the
# input and stages the batches. On that node, its host buffers do not go across the socket link on each copy.
# roofline-core's Ex 3.5 selected these from a matrix. The lab's `topo.nearest_nic` and `topo.cpu_list` do the same
# on real `nvidia-smi` output:

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
# A path code and the link generation give a theoretical rate for each direction (`p2p.predict`):
#
# - For NVLink: the number of NVLink links × the per-link rate of the generation.
# - For PCIe, when the two GPUs have **peer access**: the PCIe link rate
#   $\text{GT/s} \times \text{lanes} \times 128/130 \div 8$.
# - When the two GPUs do not have peer access: about half of that rate. The driver stages the copy through host
#   memory, with two PCIe crossings and a host copy. A measured staged copy often gets an even lower rate.
#
# The platform, not the topology code, decides if peer access works. Through a switch (`PIX`/`PXB`), peer access
# normally works. Across host bridges or sockets (`NODE`/`SYS`), it rarely works. Through the root complex (`PHB`), the
# result changes with the platform, and many platforms and most VMs turn it off.
#
# Thus a `PHB` estimate from a saved topology is an **upper bound** if you do not ask the driver. This is a
# **model**. The measurement in section 6 is the number that you trust.

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
# A measurement has almost no value until you say what explains it. Write `p2p_verdict(measured_gbs,
# direct_gbs, peer_access)`. It returns one of these values:
#
# * `"staged"`: the driver reports no peer access, thus the copy went through host memory. You can expect a number
#   near or below half the link. It is not a fault.
# * `"direct"`: there is peer access, and the measurement is at least 60% of the direct-path model (`direct_gbs`).
#   The direct-path model is the rate that the link gives with peer access. The link is healthy.
# * `"degraded"`: there is peer access, but the measurement is below 60% of the model. Something between the GPUs is
#   incorrect. Examples are ACS or an IOMMU that forces P2P through the root complex, a link that trained narrower,
#   or a busy GPU.

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
# With two or more GPUs, the lab copies a buffer between each ordered pair, and also in the two directions at the
# same time. Then it does a sweep of sizes on one pair to fit $\alpha$ and $\beta$ of the path. This is the same
# experiment as NVIDIA's `p2pBandwidthLatencyTest` and `nvbandwidth`. These two tools are still the reference tools.

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
# Megatron-style tensor parallelism does an all-reduce of the $\text{batch} \times \text{hidden}$ activations two times
# for each layer. A ring all-reduce of $S$ bytes over $p$ GPUs takes ${2(p-1)}$ steps. Each step pays a fixed cost
# $\alpha$ and moves $S/p$ bytes:
#
# $$
# t = 2(p-1)\,\alpha + \frac{2(p-1)}{p} \cdot \frac{S}{B}
# $$
#
# The formula is in primer §5.2 and in `p2p.ring_allreduce_time`. roofline-core Ex 3.1 runs a ring and derives the
# formula from it.
#
# The table calculates the cost of the decode step of a 70B-class model with the link table of primer §5.1. The
# $\alpha$ values in that link table are illustrative per-step latencies for the model. Measure your values with
# [nccl-tests](../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/any-gpu/README.md)
# ([layer 02 §5](../../../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md)). The "2 nodes" row runs one *flat* ring
# through the NICs. NCCL's hierarchical all-reduce over rails does better (primer §5.3).

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
# At decode, the $\alpha$ term is nearly all of the time. Thus the $\alpha$ that you put in the formula *is* the
# answer, and section 6 measured two values of it. Each step of a ring waits for the data of the step before it to
# arrive, and only then can it send that data on. This is a dependent chain, and it pays the full **latency** of a
# copy. Back-to-back copies of independent buffers overlap their fixed costs. Thus the $\alpha$ of the **pipelined**
# fit is smaller, and it gives a cost for the all-reduce that is too low.
#
# For $\beta$, the opposite is true. $\beta$ is a large-copy property. The pipelined sweep gives the best value for
# it, because that sweep gets to the large sizes. The latency sweep stops at a few MB.
#
# ## Exercise 3.5 — price TP with the right measured α
#
# Write `tp_allreduce_from_fits(nbytes, p, latency_fit, pipelined_fit)`. It returns the time of one ring
# all-reduce of `nbytes` over `p` GPUs. Take $\alpha$ and $\beta$ each from the fit that measures it. Use
# `p2p.ring_allreduce_time(nbytes, p, alpha, bw)`. The fits are `AlphaBeta(alpha, beta)` records.

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
# At decode batch sizes, an all-reduce moves a few kilobytes. Thus the bandwidth term is microseconds or less, and
# the **$\alpha$ term is almost everything**. Each token pays it 160 times. This is the reason for three things:
#
# - TP stays inside the NVLink domain (lowest $\alpha$, highest $B$).
# - Engines supply custom one-shot all-reduce kernels and capture the decode step in CUDA graphs.
# - TP across nodes is the last choice. The
#   [parallelism menu](../../../gpu-deployment/gpu-deployment-primer.md#4-when-one-gpu-isnt-enough-the-parallelism-menu)
#   puts pipeline and data parallelism there instead.
#
# ## In a design review
#
# **The two-minute version.** "Before I select a parallelism layout, I read `nvidia-smi topo -m`. NV# pairs get
# ~450 GB/s each way on H100, and PIX/PXB pairs get the PCIe link rate. PHB gets that rate only if the platform gives
# peer access through the root complex. Many platforms do not give it, and VMs especially do not. NODE and SYS usually
# mean that the driver stages P2P through host memory, at half the link or less.
#
# "Tensor parallelism goes where every pair is NV#. The reason is that its all-reduce runs two times for each layer,
# and at decode it is latency-bound. At batch 1, the bandwidth term is nanoseconds and $\alpha$ is the cost. That
# $\alpha$ is the latency of a copy that you wait for, not the issue cost of back-to-back copies. Each GPU uses the NIC
# on its own PCIe switch tree for GPUDirect RDMA, and its host process runs on its NUMA node.
#
# "Then I measure the P2P matrix and examine peer access. Without peer access, I expect a staged copy. If a pair has
# peer access and its rate is far below the model, something between the GPUs has an incorrect configuration."
#
# **Drills**
#
# 1. *Two nodes of 8×H100: TP=16 for a 405B model?* The answer is no. With TP=16, the per-layer all-reduces of TP
#    go through the NICs. Each NIC gives ~50 GB/s and has a higher $\alpha$, against 450 GB/s for NVLink. Use TP=8 inside each node, and
#    pipeline or data parallelism across nodes (primer §5.3).
# 2. *A 2×T4 VM (Kaggle-style, path PHB) measures a third of its Gen3 x16 link GPU-to-GPU (illustrative
#    numbers). Is it broken?* It is not necessarily broken. The model gives 15.8 GB/s for PHB only for direct peer
#    access through the root complex. Examine `torch.cuda.can_device_access_peer` (the lab records it).
#
#    Without peer access, the driver stages the copy through host memory. That is two PCIe crossings and a host
#    copy, at best about half the link and often less. You can expect this result. It is not a fault. When the platform gives
#    peer access, a third of the link *is* a result that you must examine (ACS/IOMMU settings, link width). In both cases, the link is good
#    for pipeline or data parallelism and poor for TP.
# 3. *Why pin the process that feeds GPU5 to cores 48–95 on the HGX box?* GPU5 connects to socket 1. Host buffers
#    in the memory of socket 1 do not use the inter-socket link on each DMA. Also, the RDMA NIC of GPU5 (NIC5, PXB) is
#    on the same socket.
