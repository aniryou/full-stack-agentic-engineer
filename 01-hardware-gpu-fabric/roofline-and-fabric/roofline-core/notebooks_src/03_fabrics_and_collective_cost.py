# %% [markdown]
# # 03 · Fabrics and collective cost
#
# **Tier:** T0 — the α-β model and topology arithmetic; no GPUs. To measure P2P and all-reduce
# bandwidth on real links, use `gpu-bench-lab` notebook `03_multi_gpu_topology_and_p2p` (T2; free on a
# Kaggle 2×T4 box over PCIe) and layer 02's `cuda-nccl-lab` (nccl-tests, busbw).
#
# ## The one-minute version
# Moving `n` bytes costs `t = α + n/β`: a fixed latency plus a bandwidth term. A ring all-reduce
# over `p` ranks costs `2(p−1)α + 2(p−1)/p · n/β`. Two regimes follow. A decode step's
# tensor-parallel all-reduces are tiny (one token × d_model × 2 bytes = 16 KiB for a 70B model),
# so they are **latency-bound**: what matters is how many collectives a step makes and the
# algorithm's α-count. A prefill step's all-reduces are tens of MB, so they are
# **bandwidth-bound**: what matters is β, which falls ~9× from NVLink to a 400 Gb/s NIC. That is the
# quantitative reason tensor parallelism stays inside the NVLink domain, why clusters give every
# GPU its own NIC (rails), and what `nvidia-smi topo -m` is telling you. Primer: `../PRIMER.md` §5.

# %%
from roofline import fabric, llm, specs

L = fabric.LINKS
print(f"{'link':42s} {'GB/s/dir':>8s} {'alpha us':>8s}  tier")
for k, lk in L.items():
    print(f"{lk.name:42s} {lk.gbs:8.2f} {lk.alpha_us:8.0f}  {lk.tier}")
print("\nBandwidths are theoretical, per direction. Alphas are illustrative per-step latencies incl. software.")

# %% [markdown]
# ## α-β: small messages pay latency, large ones pay bandwidth
# The ring's two terms are equal at `n = p·α·β` — about 7 MB for 8 GPUs on NVLink 4 with α = 2 µs.
# Below that, a collective is latency-bound on this model; above it, bandwidth-bound.

# %%
for key in ("nvlink4", "ib-ndr"):
    lk = L[key]
    print(f"{lk.name}: ring crossover for p=8 at {fabric.allreduce_crossover_bytes(8, lk) / 1e6:.1f} MB")
    for n in (16 * 1024, 1 << 20, 64 << 20, 1 << 30):
        print(f"   {n / 2**20:8.3f} MiB all-reduce: {fabric.ring_allreduce_time(n, 8, lk) * 1e6:10.1f} us")

# %% [markdown]
# ## algbw and busbw, as nccl-tests reports them
# `algbw = size / time`. `busbw = algbw × 2(p−1)/p` for all-reduce, so that a perfect ring reads
# as the per-direction link bandwidth, independent of p. A large all-reduce on NVLink 4 should show
# busbw ≈ 450 GB/s on this model (real systems land below it; layer 02 measures it).

# %%
GIB = 1 << 30
t = fabric.ring_allreduce_time(GIB, 8, L["nvlink4"])
print(f"1 GiB, 8 ranks: {t * 1e3:.2f} ms, algbw {fabric.algbw(GIB, t) / 1e9:.0f} GB/s, busbw {fabric.busbw(GIB, t, 8) / 1e9:.0f} GB/s")

# %% [markdown]
# ## Tensor parallelism: two all-reduces per layer, every step
# Megatron-style TP all-reduces a `[tokens × d_model]` activation after attention's output
# projection and after the MLP's down projection: `2 × layers` collectives per forward step.

# %%
m70 = llm.PRESETS["llama-3.1-70b"]
h100 = specs.get("h100-sxm")
dec = llm.decode(m70, h100, 1, 1024)
print(f"Llama-3.1-70B: {fabric.tp_allreduces_per_step(m70)} all-reduces/step, "
      f"{fabric.tp_allreduce_bytes(m70, 1) / 1024:.0f} KiB each at batch 1")
print(f"TP=8 decode: each GPU streams its weight shard in {dec.t_memory / 8 * 1e3:.2f} ms;")
for algo in ("ring", "recursive-doubling"):
    print(f"   all-reduce time per step ({algo:18s}): {fabric.tp_comm_time(m70, 1, 8, L['nvlink4'], algo=algo) * 1e3:.2f} ms")
pf = llm.prefill(m70, h100, 4096)
print(f"\nTP=8 prefill of 4,096 tokens: per-GPU compute {pf.t_compute / 8 * 1e3:.1f} ms; "
      f"all-reduce {fabric.tp_allreduce_bytes(m70, 4096) / 1e6:.0f} MB each")
for key in ("nvlink4", "ib-ndr"):
    print(f"   communication per step over {L[key].name}: {fabric.tp_comm_time(m70, 4096, 8, L[key]) * 1e3:.1f} ms")

# %% [markdown]
# ## Rails: every GPU gets its own NIC
# A hierarchical all-reduce reduce-scatters inside the node over NVLink, all-reduces `n/g` per GPU
# across nodes, and all-gathers inside the node. With one NIC per GPU (a rail-optimized design)
# each GPU's share rides its own NIC; with one NIC per node, eight GPUs queue behind it.

# %%
for nics in (8, 1):
    t = fabric.hierarchical_allreduce_time(GIB, 8, 4, L["nvlink4"], L["ib-ndr"], nics_per_node=nics)
    print(f"1 GiB all-reduce over 4 nodes x 8 GPUs, {nics} NIC(s) per node: {t * 1e3:.1f} ms")
print("switch hops (32 nodes per rail leaf):",
      {"same rail": fabric.switch_hops((0, 3), (17, 3), 32), "cross-rail": fabric.switch_hops((0, 3), (17, 5), 32),
       "cross-rail with PXN": fabric.switch_hops((0, 3), (17, 5), 32, pxn=True)})

# %% [markdown]
# ## Leaf-spine: oversubscription and bisection
# A two-tier folded Clos of radix-R switches is non-blocking (1:1) when every leaf splits its
# ports half down, half up; it tops out at `R²/2` endpoints. Giving more ports to hosts saves
# switches and cuts bisection bandwidth by the oversubscription ratio.

# %%
for down in (32, 48):
    ls = fabric.leaf_spine(3072 if down == 48 else 2048, 64, down)
    print(f"{ls.endpoints} x 400G endpoints, radix 64, {down} down: {ls.leaves} leaves, {ls.spines} spines, "
          f"{ls.oversubscription:.0f}:1, bisection {fabric.bisection_gbs(ls.endpoints, 400, ls.oversubscription):,.0f} GB/s")

# %% [markdown]
# ## GPUDirect: remove the host bounce
# Without GPUDirect RDMA a GPU→remote-GPU message is copied to host memory, sent, and copied up
# again. Store-and-forward pays every hop in sequence; a pipelined direct path pays only the
# slowest hop (plus one chunk per other hop) and keeps the CPU and host memory out of the data path.

# %%
print(f"1 GiB, host-staged store-and-forward [PCIe5, IB NDR, PCIe5]: {fabric.staged_transfer_time(GIB, [63, 50, 63]) * 1e3:.1f} ms")
print(f"1 GiB, GPUDirect RDMA pipelined in 1 MiB chunks:        {fabric.staged_transfer_time(GIB, [63, 50, 63], 1 << 20) * 1e3:.1f} ms")

# %% [markdown]
# ## Exercise 3.1 — the ring all-reduce
# Write `ring_allreduce(n, p, alpha, beta)` in seconds (α in s, β in bytes/s). A ring does a
# reduce-scatter then an all-gather: `2(p−1)` steps, each moving `n/p` bytes per rank.

# %% exercise
def ring_allreduce(n, p, alpha, beta):
    ### BEGIN SOLUTION
    if p == 1:
        return 0.0
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * n / beta
    ### END SOLUTION

# %% check
for key in ("nvlink4", "ib-ndr", "pcie4x16"):
    lk = L[key]
    for n, p in [(16384, 8), (64 << 20, 8), (1 << 30, 16), (1000, 1)]:
        assert abs(ring_allreduce(n, p, lk.alpha, lk.beta) - fabric.ring_allreduce_time(n, p, lk)) < 1e-12
print("✅ ring all-reduce: 2(p−1)α + 2(p−1)/p · n/β")

# %% [markdown]
# ## Exercise 3.2 — read a busbw sweep
# Below is a **simulated** all-reduce sweep (generated with the α-β model for 8 ranks on NVLink 4 —
# not a measurement). Write `busbw(n, t, p)` (bytes/s) and `latency_bound(results, p, link_bps)`
# returning the message sizes whose busbw is below half the link bandwidth.

# %%
sweep = [(n, fabric.ring_allreduce_time(n, 8, L["nvlink4"])) for n in (2 ** k for k in range(3, 31))]
print("simulated sweep (size B, time us):", [(n, round(t * 1e6, 1)) for n, t in sweep[::6]])

# %% exercise
def busbw(n, t, p):
    ### BEGIN SOLUTION
    return n / t * 2 * (p - 1) / p
    ### END SOLUTION


def latency_bound(results, p, link_bps):
    ### BEGIN SOLUTION
    return [n for n, t in results if busbw(n, t, p) < 0.5 * link_bps]
    ### END SOLUTION

# %% check
lb = latency_bound(sweep, 8, L["nvlink4"].beta)
assert lb == [2 ** k for k in range(3, 23)], lb
assert abs(busbw(*sweep[-1], 8) / 1e9 - 447.8) < 1.0
print(f"✅ latency-bound up to {lb[-1] / 2**20:.0f} MiB — just below p·α·β = 7.2 MB; "
      "a 16 KiB TP message is ~440x smaller than that")

# %% [markdown]
# ## Exercise 3.3 — choose a TP degree for latency
# Batch-1 decode latency with TP = p is roughly: each GPU streams `1/p` of the bytes, plus the
# step's all-reduces. Write `decode_latency(model, device, tp, link, algo)` using
# `llm.decode(...).t_memory` (batch 1, context 1024) and `fabric.tp_comm_time`, then
# `best_tp(model, device, link, algo, choices)` returning the degree with the lowest latency.

# %% exercise
def decode_latency(model, device, tp, link, algo="ring"):
    ### BEGIN SOLUTION
    return llm.decode(model, device, 1, 1024).t_memory / tp + fabric.tp_comm_time(model, 1, tp, link, algo=algo)
    ### END SOLUTION


def best_tp(model, device, link, algo="ring", choices=(2, 4, 8)):
    ### BEGIN SOLUTION
    return min(choices, key=lambda p: decode_latency(model, device, p, link, algo))
    ### END SOLUTION

# %% check
assert abs(decode_latency(m70, h100, 8, L["nvlink4"]) * 1e3 - 9.69) < 0.01
assert abs(decode_latency(m70, h100, 8, L["nvlink4"], "recursive-doubling") * 1e3 - 6.18) < 0.01
assert best_tp(m70, h100, L["nvlink4"]) == 8 and best_tp(m70, h100, L["nvlink4"], "recursive-doubling") == 8
per_gpu = {p: 1 / (decode_latency(m70, h100, p, L["nvlink4"]) * p) for p in (2, 4, 8)}
assert per_gpu[2] > per_gpu[4] > per_gpu[8]
print(f"✅ TP=8 is fastest per token (9.7 ms ring, 6.2 ms latency-optimal), but tokens per GPU-second "
      f"fall from {per_gpu[2]:.1f} (TP=2) to {per_gpu[8]:.1f} (TP=8): latency costs efficiency")

# %% [markdown]
# ## Exercise 3.4 — size a two-tier fabric
# 1,024 GPUs, one 400 Gb/s NIC each, radix-64 switches. Write
# `size_fabric(endpoints, radix, down, gbps)` returning `(leaves, spines, oversubscription, bisection_GBps)`.
# Each leaf gives `down` ports to hosts and the rest to spines; each spine port takes one leaf uplink.

# %% exercise
import math

def size_fabric(endpoints, radix, down, gbps=400):
    ### BEGIN SOLUTION
    up = radix - down
    leaves = math.ceil(endpoints / down)
    spines = math.ceil(leaves * up / radix)
    oversub = down / up
    bisection = endpoints / 2 * gbps / 8 / max(1.0, oversub)
    return leaves, spines, oversub, bisection
    ### END SOLUTION

# %% check
assert size_fabric(1024, 64, 32) == (32, 16, 1.0, 25_600)
leaves, spines, over, bis = size_fabric(1024, 64, 48)
assert (leaves, spines, over) == (22, 6, 3.0) and abs(bis - 25_600 / 3) < 1e-6
ref = fabric.leaf_spine(1024, 64, 48)
assert (ref.leaves, ref.spines) == (leaves, spines)
print("✅ 1:1 needs 48 switches (32 leaves + 16 spines); 3:1 needs 28 but keeps a third of the bisection")

# %% [markdown]
# ## Exercise 3.5 — read `nvidia-smi topo -m`
# Here is a topology matrix in the documented format for a hypothetical 2-socket, 4-GPU,
# 2-NIC server (**illustrative, not captured from a machine**). Write `best_pair(matrix)` — the GPU pair with
# the best connection (use `fabric.topo_rank`, lower is better) — and `closest_nic(matrix, gpu)`.

# %%
matrix = {
    "GPU0": {"GPU1": "NV12", "GPU2": "SYS", "GPU3": "SYS", "NIC0": "PIX", "NIC1": "SYS"},
    "GPU1": {"GPU0": "NV12", "GPU2": "SYS", "GPU3": "SYS", "NIC0": "PXB", "NIC1": "SYS"},
    "GPU2": {"GPU0": "SYS", "GPU1": "SYS", "GPU3": "NV4", "NIC0": "SYS", "NIC1": "PIX"},
    "GPU3": {"GPU0": "SYS", "GPU1": "SYS", "GPU2": "NV4", "NIC0": "SYS", "NIC1": "PXB"},
}
for code, meaning in fabric.TOPO_LEGEND.items():
    print(f"{code:5s} {meaning}")

# %% exercise
def best_pair(matrix):
    ### BEGIN SOLUTION
    pairs = [(a, b) for a in matrix for b in matrix[a] if b.startswith("GPU") and a < b]
    return min(pairs, key=lambda ab: fabric.topo_rank(matrix[ab[0]][ab[1]]))
    ### END SOLUTION


def closest_nic(matrix, gpu):
    ### BEGIN SOLUTION
    nics = [k for k in matrix[gpu] if k.startswith("NIC")]
    return min(nics, key=lambda n: fabric.topo_rank(matrix[gpu][n]))
    ### END SOLUTION

# %% check
assert best_pair(matrix) == ("GPU0", "GPU1")
assert closest_nic(matrix, "GPU3") == "NIC1" and closest_nic(matrix, "GPU0") == "NIC0"
print("✅ put a 2-GPU TP job on GPU0+GPU1 (NV12), never GPU1+GPU2 (SYS: across the sockets); "
      "pin each GPU's process to its socket and its nearest NIC")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Every transfer is α + n/β. Tensor parallelism makes two
# all-reduces per layer per step — 160 for a 70B model. At decode they are 16 KiB each, so they are
# latency-bound: the count × the algorithm's latency is the cost, which is why engines use
# latency-optimal all-reduce kernels. At prefill they are tens of MB and bandwidth-bound: 46 ms
# per 4K-token step over NVLink versus 387 ms over a 400 Gb/s NIC, five times the compute. So TP
# stays inside the NVLink domain; across nodes I use pipeline or data parallelism, one NIC per GPU
# on a rail-optimized, non-blocking fabric, and I check `nvidia-smi topo -m` before placing ranks."
#
# **Drill.**
# 1. *Why not TP=16 across two 8-GPU H100 nodes?* — the ring then runs at NIC speed (50 GB/s vs 450):
#    a 4K-token prefill step spends ~427 ms in all-reduce against ~37 ms of compute per GPU
#    (TP=8 on NVLink: 46 ms vs 74 ms). Use PP=2 × TP=8, FP8 to fit in one node, or a bigger NVLink domain.
# 2. *nccl-tests shows busbw of 20 GB/s at 64 KB. Is the fabric broken?* — no: at 64 KB the collective is
#    latency-bound (below p·α·β); judge the fabric on the large-message plateau.
# 3. *Is a 3:1 oversubscribed fabric fine for inference?* — often, for independent replicas (mostly
#    north-south traffic); not for training all-reduces or disaggregated KV transfer, which need bisection.
