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
# ## 2 · Reading `nvidia-smi topo -m`
#
# Every cell names the path between two devices, best to worst:
#
# | code | path | typical GPU↔GPU rate |
# |---|---|---|
# | `NV#` | a bonded set of # NVLinks (through NVSwitch on HGX boards) | 25–50 GB/s per link per direction |
# | `PIX` | at most one PCIe bridge — the same PCIe switch | the PCIe link rate |
# | `PXB` | several PCIe bridges, not the host bridge | the PCIe link rate |
# | `PHB` | through the CPU's PCIe host bridge (root complex) | often lower, sometimes no P2P |
# | `NODE` | across host bridges inside one NUMA node | usually no direct P2P |
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
# ideally on the same PCIe switch (`PIX`/`PXB`) — so the NIC reads GPU memory without crossing the
# CPU; one NIC per GPU is a **rail**. And the process that feeds a GPU (tokenising, staging batches)
# belongs on that GPU's NUMA node, so its host buffers do not cross the socket link on every copy.
#
# ## Exercise 3.4 — the NIC and the cores for a GPU
#
# Write `my_nearest_nic(t, gpu)` → `(nic, code)` (best-ranked path from `gpu` to any of `t.nics`)
# and `my_cpu_list(spec)` that expands `"0-3,8-11"` into `[0, 1, 2, 3, 8, 9, 10, 11]`.

# %% exercise
def my_nearest_nic(t, gpu):
    ### BEGIN SOLUTION
    nic = max(t.nics, key=lambda n: my_rank(t.link(gpu, n)))
    return nic, t.link(gpu, nic)
    ### END SOLUTION


def my_cpu_list(spec):
    ### BEGIN SOLUTION
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out
    ### END SOLUTION

# %% check
for g in hgx.gpus:
    assert my_nearest_nic(hgx, g) == topo.nearest_nic(hgx, g)
assert my_nearest_nic(hgx, "GPU5") == ("NIC5", "PXB") and my_nearest_nic(pcie_box, "GPU3") == ("NIC0", "SYS")
assert my_cpu_list("0-3,8-11") == [0, 1, 2, 3, 8, 9, 10, 11] and my_cpu_list("5") == [5]
assert my_cpu_list(hgx.cpu_affinity["GPU5"]) == topo.cpu_list(hgx, "GPU5")
print(f"✅ GPU5 → {hgx.nic_names['NIC5']} (PXB), cores {hgx.cpu_affinity['GPU5']}; "
      "on the PCIe box GPU2/GPU3 reach the only NIC across the sockets (SYS) — a bottleneck for multi-node traffic")

# %% [markdown]
# ## 5 · From topology to predicted bandwidth
#
# A path code plus the link generation gives a theoretical per-direction rate: NVLink links × the
# per-link rate of the generation; PCIe `GT/s × lanes × 128/130 ÷ 8`; and roughly half of that when
# P2P is unavailable and the copy is staged through host memory. This is a **model** — the
# measurement in section 6 is what you trust.

# %%
for name, arch, gen in (("hgx-h100-8gpu", "hopper", 5), ("a2-highgpu-2g", "ampere", 4),
                        ("pcie-4gpu-2socket", None, 4), ("kaggle-2xt4", "turing", 3)):
    t = topo.parse(topo.load_fixture(name))
    pred = p2p.predict(t, arch=arch, pcie_gen=gen)
    print(f"{name} ({SAMPLE.split(' —')[0].lower()}), PCIe Gen{gen}:")
    for a, b2, code in t.gpu_pairs()[:3]:
        e = pred[(a, b2)]
        print(f"   {a}→{b2}  {code:>5}  ≈ {e.gbs:6.1f} GB/s per direction (model): {e.why}")

# %% [markdown]
# ## 6 · Measure it (T2)
#
# With two or more GPUs, the lab copies a buffer between every ordered pair (and both directions at
# once), then sweeps sizes on one pair to fit α and β of the path — the same experiment as NVIDIA's
# `p2pBandwidthLatencyTest` and `nvbandwidth`, which remain the reference tools.

# %%
measured_ab = None
if ngpu >= 2:
    be = get_backend("torch")
    size = (64 << 20) if QUICK else (256 << 20)
    uni = p2p.measure_matrix(be, size)
    bi = p2p.measure_matrix(be, size, bidirectional=True)
    display(Markdown(measurements_markdown(uni + bi)))
    sweep = p2p.size_sweep(be, 0, 1, transfer.sizes(4 << 10, 64 << 20, 4))
    measured_ab = transfer.fit(sweep)
    print(f"GPU0→GPU1: α = {si(measured_ab.alpha, 's')}, β = {si(measured_ab.beta, 'B/s')}, "
          f"n½ = {si(measured_ab.n_half, 'B')}  ({uni[0].note})")
    if local_topo:
        t = topo.parse(local_topo)
        spec = specs.lookup(be.describe()["name"])
        gen = (local_rows or [{}])[0].get("pcie.link.gen.max") or 4
        e = p2p.predict(t, arch=spec.arch if spec else None, pcie_gen=gen)[("GPU0", "GPU1")]
        print(f"model for GPU0→GPU1 ({t.link('GPU0', 'GPU1')}): {e.gbs:.1f} GB/s per direction — measured "
              f"{si(uni[0].bytes_per_s(), 'B/s')} ({uni[0].bytes_per_s() / 1e9 / e.gbs:.0%} of the model)")
else:
    print(f"{ngpu} GPU(s) visible: P2P needs two. To run this section for real:\n"
          "  • Kaggle (free): Notebook → Settings → Accelerator → 'GPU T4 x2'; clone the repo, pip install -e the lab,\n"
          "    and re-run this notebook (PCIe, no NVLink — expect the 'staged' regime).\n"
          "  • RunPod / Vast / Lambda: a 2–8 GPU SXM (NVLink) machine for an hour; or GCP a2-highgpu-2g (deploy/gcp).\n"
          "  • Anywhere: python -m gpubench run --suite inventory,p2p --out results\n"
          "  • Reference tools: nvidia-smi topo -m, nvidia-smi nvlink --status, nvbandwidth.")

# %% [markdown]
# ## 7 · What the link is worth: the TP all-reduce per token
#
# Megatron-style tensor parallelism all-reduces the `batch × hidden` activations twice per layer.
# A ring all-reduce of `S` bytes over `p` GPUs takes `2(p−1)` steps, each paying a fixed cost α and
# moving `S/p` bytes: **`t = 2(p−1)·α + 2·(p−1)/p · S/B`** (primer §5).
#
# ## Exercise 3.5 — price one all-reduce
#
# Write `my_ring_allreduce(nbytes, p, alpha, bw)` (seconds; zero for a single GPU).

# %% exercise
def my_ring_allreduce(nbytes, p, alpha, bw):
    ### BEGIN SOLUTION
    if p < 2:
        return 0.0
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * nbytes / bw
    ### END SOLUTION

# %% check
assert abs(my_ring_allreduce(1e9, 8, 0.0, 100e9) - 0.0175) < 1e-12      # pure bandwidth: 2·7/8 · 1 GB / 100 GB/s
assert abs(my_ring_allreduce(0, 8, 10e-6, 1e9) - 140e-6) < 1e-12        # pure latency: 14 steps × 10 µs
assert my_ring_allreduce(123, 1, 1.0, 1.0) == 0.0
assert abs(my_ring_allreduce(4e6, 4, 3e-6, 200e9) - p2p.ring_allreduce_time(4e6, 4, 3e-6, 200e9)) < 1e-15
print("✅ ring all-reduce: 2(p−1)·α + 2(p−1)/p · S/B")

# %%
# A 70B-class dense model (hidden 8192, 80 layers, bf16) with TP=8. The α values are ILLUSTRATIVE
# assumptions per ring step (measure yours with nccl-tests — layer 02); the bandwidths are link rates.
hidden, layers, p = 8192, 80, 8
hbm_step = 140e9 / p / specs.get("h100-sxm").mem_bw          # each GPU streams its 1/8 of the weights per step
fabrics = [("NVLink4 via NVSwitch", 2e-6, 450e9), ("PCIe Gen4 x16, staged", 10e-6, 12e9), ("2 nodes, 400 Gb/s NICs", 8e-6, 50e9)]
print(f"HBM time per decode step (weights only, H100): {si(hbm_step, 's')}\n")
print(f"{'fabric':<24} {'batch':>5} {'per all-reduce':>15} {'per step (160 of them)':>24} {'α share':>8}")
for name, alpha, bw in fabrics:
    for batch in (1, 64):
        one = my_ring_allreduce(batch * hidden * 2, p, alpha, bw)
        step = layers * 2 * one
        print(f"{name:<24} {batch:>5} {si(one, 's'):>15} {si(step, 's'):>24} {2 * (p - 1) * alpha / one:>8.0%}")
if measured_ab is not None:
    tp2 = layers * 2 * my_ring_allreduce(hidden * 2, 2, measured_ab.alpha, measured_ab.beta)
    print(f"\nyour measured GPU0↔GPU1 link, TP=2, batch 1: {si(tp2, 's')} of all-reduce per decode step")

# %% [markdown]
# At decode batch sizes an all-reduce moves a few kilobytes, so the bandwidth term is microseconds
# or less and the **α term is almost everything** — and it is paid 160 times per token. That is why
# TP stays inside the NVLink domain (lowest α, highest B), why engines ship custom one-shot
# all-reduce kernels and capture decode in CUDA graphs, and why TP across nodes is a last resort.
#
# ## In a design review
#
# **The two-minute version.** "Before choosing a parallelism layout I read `nvidia-smi topo -m`.
# NV# pairs get ~450 GB/s each way on H100; PIX/PXB pairs get the PCIe link rate; PHB, NODE and SYS
# usually mean P2P is staged through host memory. Tensor parallelism goes where every pair is NV#,
# because its all-reduce runs twice per layer and at decode it is latency-bound: at batch 1 the
# bandwidth term is nanoseconds and α is the cost. Each GPU uses the NIC on its own PCIe switch for
# GPUDirect RDMA, and its host process runs on its NUMA node. Then I measure the P2P matrix: a pair
# far below the model's rate means peer access is off or the copy is being staged."
#
# **Drills**
#
# 1. *Two nodes of 8×H100 — TP=16 for a 405B model?* — No: TP's per-layer all-reduces would cross
#    the NICs (~50 GB/s and higher α instead of 450 GB/s NVLink). TP=8 inside each node, pipeline or
#    data parallelism across nodes.
# 2. *Kaggle's 2×T4 measures ~5 GB/s GPU-to-GPU on a Gen3 x16 link. Broken?* — No: the path is
#    PHB, and in a VM direct P2P is often unavailable, so the copy is staged through host memory —
#    the model's "about half the link" regime. Fine for pipeline or data parallelism; poor for TP.
# 3. *Why pin GPU5's feeding process to cores 48–95 on the HGX box?* — GPU5 hangs off socket 1: host
#    buffers on socket 1's memory avoid the inter-socket link on every DMA, and its RDMA NIC (NIC5,
#    PXB) sits on the same socket.
