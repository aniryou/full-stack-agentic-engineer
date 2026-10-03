# %% [markdown]
# # 03 · Fabrics and collective cost
#
# **Tier:** T0. It uses the α-β model and topology arithmetic, with no GPUs. To measure P2P and all-reduce
# bandwidth on real links, use `gpu-bench-lab` notebook `03_multi_gpu_topology_and_p2p` (T2). It is free on a
# Kaggle 2×T4 box over PCIe. Also use the `cuda-nccl-lab` of layer 02 (nccl-tests, busbw).
#
# ## The one-minute version
# A transfer of $n$ bytes costs $t = \alpha + n/\beta$: a constant latency plus a bandwidth term. A ring all-reduce
# over $p$ ranks costs
#
# $$
# 2(p-1)\,\alpha + \frac{2(p-1)}{p} \cdot \frac{n}{\beta}.
# $$
#
# Two regimes follow from this. The tensor-parallel all-reduces of a decode step are small (one token × $d_{\text{model}}$ ×
# 2 bytes = 16 KiB for a 70B model). Thus they are **latency-bound**: what matters is the number of collectives that a
# step makes and the $\alpha$-count of the algorithm. The all-reduces of a prefill step are tens of MB, thus they are
# **bandwidth-bound**: what matters is $\beta$. The bandwidth term of the ring (~$2n/\beta$) does not decrease as
# the TP degree grows, but the compute of each GPU does decrease.
#
# Past the node, $\beta$ per GPU decreases ~9× from NVLink to a 400 Gb/s NIC. That is the quantitative reason
# for two things:
#
# - Tensor parallelism stays inside the NVLink domain.
# - Clusters give every GPU its own NIC (rails) and run collectives hierarchically.
#
# It is also what `nvidia-smi topo -m` tells you. Primer: `../PRIMER.md` §5.

# %%
from roofline import fabric, llm, specs

L = fabric.LINKS
print(f"{'link':42s} {'GB/s/dir':>8s} {'alpha us':>8s}  tier")
for k, lk in L.items():
    print(f"{lk.name:42s} {lk.gbs:8.2f} {lk.alpha_us:8.0f}  {lk.tier}")
print("\nBandwidths are theoretical, per direction. Alphas are illustrative per-step latencies incl. software.")

# %% [markdown]
# ## α-β: small messages pay latency, large ones pay bandwidth
# The two terms of the ring are equal at $n = p \cdot \alpha \cdot \beta$. That is about 7 MB for 8 GPUs on NVLink 4
# with $\alpha = 2$ µs. Below that size, a collective is latency-bound on this model. Above it, the collective is bandwidth-bound.

# %%
for key in ("nvlink4", "ib-ndr"):
    lk = L[key]
    print(f"{lk.name}: ring crossover for p=8 at {fabric.allreduce_crossover_bytes(8, lk) / 1e6:.1f} MB")
    for n in (16 * 1024, 1 << 20, 64 << 20, 1 << 30):
        print(f"   {n / 2**20:8.3f} MiB all-reduce: {fabric.ring_allreduce_time(n, 8, lk) * 1e6:10.1f} us")

# %% [markdown]
# ## algbw and busbw, as nccl-tests reports them
# $\mathrm{algbw} = \text{size}/\text{time}$. For all-reduce, $\mathrm{busbw} = \mathrm{algbw} \times 2(p-1)/p$.
# With this definition, a perfect ring shows the per-direction link bandwidth, independent of $p$. On this model, a large
# all-reduce on NVLink 4 shows busbw ≈ 450 GB/s. Real systems are below it, and layer 02 measures it.

# %%
GIB = 1 << 30
t = fabric.ring_allreduce_time(GIB, 8, L["nvlink4"])
print(f"1 GiB, 8 ranks: {t * 1e3:.2f} ms, algbw {fabric.algbw(GIB, t) / 1e9:.0f} GB/s, busbw {fabric.busbw(GIB, t, 8) / 1e9:.0f} GB/s")

# %% [markdown]
# ## Tensor parallelism: two all-reduces per layer, every step
# Megatron-style TP does an all-reduce of a $[\text{tokens} \times d_{\text{model}}]$ activation at two points. These are after the
# output projection of attention and after the down projection of the MLP. That is $2 \times \text{layers}$ collectives per forward step.

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
print(f"   communication per step over NVLink 4: {fabric.tp_comm_time(m70, 4096, 8, L['nvlink4']) * 1e3:.1f} ms")
print(f"   if every ring hop crossed a 400 Gb/s NIC (8 GPUs in 8 nodes): {fabric.tp_comm_time(m70, 4096, 8, L['ib-ndr']) * 1e3:.1f} ms")

# %% [markdown]
# ## TP=16 across two nodes
# NCCL does not push a cross-node all-reduce through one NIC. On a rail-optimized cluster, it spreads
# the traffic over every rail (a ring channel per NIC, or its tree algorithm). The core represents that as a
# hierarchical all-reduce (`fabric.tp_comm_time_across_nodes`):
#
# 1. A reduce-scatter inside each node over NVLink.
# 2. An all-reduce of the $n/8$ share of each GPU across nodes, over its own NIC.
# 3. An all-gather inside the node.
#
# Even so, communication becomes larger than compute. Without a NIC per GPU, it is far worse.

# %%
print(f"TP=16 prefill of 4,096 tokens: per-GPU compute {pf.t_compute / 16 * 1e3:.1f} ms")
for label, t in [("rails, 8 NICs per node", fabric.tp_comm_time_across_nodes(m70, 4096, 8, 2, L["nvlink4"], L["ib-ndr"])),
                 ("1 NIC per node", fabric.tp_comm_time_across_nodes(m70, 4096, 8, 2, L["nvlink4"], L["ib-ndr"], 1)),
                 ("one flat ring through the NICs", fabric.tp_comm_time(m70, 4096, 16, L["ib-ndr"]))]:
    print(f"   all-reduce per step, {label:32s}: {t * 1e3:6.1f} ms")

# %% [markdown]
# ## Rails: every GPU gets its own NIC
# A hierarchical all-reduce does a reduce-scatter inside the node over NVLink, an all-reduce of $n/g$ per GPU
# across nodes, and an all-gather inside the node. With one NIC per GPU (a rail-optimized design),
# the share of each GPU goes over its own NIC. With one NIC per node, eight GPUs wait in a queue behind it.

# %%
for nics in (8, 1):
    t = fabric.hierarchical_allreduce_time(GIB, 8, 4, L["nvlink4"], L["ib-ndr"], nics_per_node=nics)
    print(f"1 GiB all-reduce over 4 nodes x 8 GPUs, {nics} NIC(s) per node: {t * 1e3:.1f} ms")
print("switch hops (32 nodes per rail leaf):",
      {"same rail": fabric.switch_hops((0, 3), (17, 3), 32), "cross-rail": fabric.switch_hops((0, 3), (17, 5), 32),
       "cross-rail with PXN": fabric.switch_hops((0, 3), (17, 5), 32, pxn=True)})

# %% [markdown]
# ## Leaf-spine: oversubscription and bisection
# A two-tier folded Clos of radix-$R$ switches is non-blocking (1:1) when every leaf divides its
# ports into half down and half up. Its maximum is $R^2/2$ endpoints. If you give more ports to hosts, you need
# fewer switches, and the bisection bandwidth decreases by the oversubscription ratio.

# %%
for down in (32, 48):
    ls = fabric.leaf_spine(3072 if down == 48 else 2048, 64, down)
    print(f"{ls.endpoints} x 400G endpoints, radix 64, {down} down: {ls.leaves} leaves, {ls.spines} spines, "
          f"{ls.oversubscription:.0f}:1, bisection {fabric.bisection_gbs(ls.endpoints, 400, ls.oversubscription):,.0f} GB/s")

# %% [markdown]
# ## GPUDirect: remove the host bounce
# Without GPUDirect RDMA, a message from a GPU to a remote GPU goes through host memory. The system copies it to
# host memory, sends it, and copies it up again. Store-and-forward pays for each hop in sequence. A pipelined
# direct path pays only for the slowest hop (plus one chunk per other hop). It also keeps the CPU and host memory
# out of the data path.

# %%
print(f"1 GiB, host-staged store-and-forward [PCIe5, IB NDR, PCIe5]: {fabric.staged_transfer_time(GIB, [63, 50, 63]) * 1e3:.1f} ms")
print(f"1 GiB, GPUDirect RDMA pipelined in 1 MiB chunks:        {fabric.staged_transfer_time(GIB, [63, 50, 63], 1 << 20) * 1e3:.1f} ms")

# %% [markdown]
# ## Exercise 3.1 — the ring all-reduce, by running one
# Each of $p$ ranks holds a vector, cut into $p$ chunks (`vectors[r][j]` is chunk $j$ on rank $r$). In one
# synchronous **step**, every rank sends one chunk to its right neighbour `(r + 1) % p`. There are two phases:
#
# - Phase 1 (reduce-scatter): the receiver *adds* the chunk into its own copy. When the phase ends, each rank holds
#   one chunk that is the sum over all ranks.
# - Phase 2 (all-gather): the finished chunks go around the same ring, and the receiver *overwrites* its copy.
#
# Write `simulate_ring_allreduce(vectors)`. It returns `(result, steps)`. You must find yourself which chunk each
# rank sends at each step. Then write `ring_allreduce(n, p, alpha, beta)` in seconds ($\alpha$ in s, $\beta$ in bytes/s).
# Use what the simulation tells you: the number of steps, and the number of bytes in each message.

# %% exercise
def simulate_ring_allreduce(vectors):
    ### BEGIN SOLUTION
    p = len(vectors)
    buf = [list(v) for v in vectors]
    steps = 0
    for s in range(p - 1):                                   # reduce-scatter
        sends = [(r, (r - s) % p, buf[r][(r - s) % p]) for r in range(p)]
        for r, j, val in sends:
            buf[(r + 1) % p][j] += val
        steps += 1
    for s in range(p - 1):                                   # all-gather
        sends = [(r, (r + 1 - s) % p, buf[r][(r + 1 - s) % p]) for r in range(p)]
        for r, j, val in sends:
            buf[(r + 1) % p][j] = val
        steps += 1
    return buf, steps
    ### END SOLUTION


def ring_allreduce(n, p, alpha, beta):
    ### BEGIN SOLUTION
    if p == 1:
        return 0.0
    _, steps = simulate_ring_allreduce([[0] * p for _ in range(p)])
    return steps * (alpha + n / p / beta)                    # each message is one chunk: n/p bytes
    ### END SOLUTION

# %% check
import random
random.seed(0)
for p in (2, 3, 4, 8):
    vecs = [[random.randint(-9, 9) for _ in range(p)] for _ in range(p)]
    result, steps = simulate_ring_allreduce(vecs)
    total = [sum(v[j] for v in vecs) for j in range(p)]
    assert all(row == total for row in result), (p, result, total)
    assert steps == 2 * (p - 1), (p, steps)
for key in ("nvlink4", "ib-ndr", "pcie4x16"):
    lk = L[key]
    for n, p in [(16384, 8), (64 << 20, 8), (1 << 30, 16), (1000, 1)]:
        assert abs(ring_allreduce(n, p, lk.alpha, lk.beta) - fabric.ring_allreduce_time(n, p, lk)) < 1e-12
print("✅ every rank ends with the full sum after 2(p−1) steps of one n/p chunk: 2(p−1)α + 2(p−1)/p · n/β")

# %% [markdown]
# ## Exercise 3.2 — read a busbw sweep
# The next cell has a **simulated** all-reduce sweep. The α-β model generated it for 8 ranks on NVLink 4, and it
# is not a measurement. Write `busbw(n, t, p)` (bytes/s) and `latency_bound(results, p, link_bps)`.
# The second function returns the message sizes whose busbw is below half the link bandwidth.

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
# ## Exercise 3.3 — choose a TP degree against an ITL SLO
# With $\text{TP} = p$, the batch-1 decode latency is approximately this: each GPU streams $1/p$ of the bytes, plus
# the all-reduces of the step. Nodes hold 8 GPUs on NVLink 4, with one 400 Gb/s NIC per GPU.
#
# Write `decode_latency(model, device, tp, algo)`:
#
# - Use `llm.decode(...).t_memory` (batch 1, context 1024).
# - For `tp ≤ 8`, the all-reduces run inside a node (`fabric.tp_comm_time` over NVLink 4 with `algo`).
# - For `tp > 8`, they run hierarchically across `tp // 8` nodes (`fabric.tp_comm_time_across_nodes`).
#   That function is ring-based, whatever `algo` says.
#
# Then write `choose_tp(model, device, itl_s, algo, choices)`. It returns the **fewest GPUs** whose latency meets
# `itl_s`, or `None`. Make a prediction first: what does a faster all-reduce algorithm do to the answer for a tight SLO?

# %% exercise
def decode_latency(model, device, tp, algo="ring"):
    ### BEGIN SOLUTION
    weights = llm.decode(model, device, 1, 1024).t_memory / tp
    if tp <= 8:
        return weights + fabric.tp_comm_time(model, 1, tp, L["nvlink4"], algo=algo)
    return weights + fabric.tp_comm_time_across_nodes(model, 1, 8, tp // 8, L["nvlink4"], L["ib-ndr"])
    ### END SOLUTION


def choose_tp(model, device, itl_s, algo="ring", choices=(2, 4, 8, 16)):
    ### BEGIN SOLUTION
    ok = [p for p in sorted(choices) if decode_latency(model, device, p, algo) <= itl_s]
    return ok[0] if ok else None
    ### END SOLUTION

# %% check
assert abs(decode_latency(m70, h100, 8) * 1e3 - 9.69) < 0.01
assert abs(decode_latency(m70, h100, 8, "recursive-doubling") * 1e3 - 6.18) < 0.01
assert abs(decode_latency(m70, h100, 16) * 1e3 - 8.70) < 0.01
assert choose_tp(m70, h100, 0.015) == 4 and choose_tp(m70, h100, 0.010) == 8
assert choose_tp(m70, h100, 0.009) == 16                          # the ring forces a second node...
assert choose_tp(m70, h100, 0.009, "recursive-doubling") == 8     # ...a latency-optimal all-reduce does not
assert choose_tp(m70, h100, 0.005) is None
per_gpu = {p: 1 / (decode_latency(m70, h100, p) * p) for p in (2, 4, 8, 16)}
assert per_gpu[2] > per_gpu[4] > per_gpu[8] > per_gpu[16]
print(f"✅ 15 ms → TP=4, 10 ms → TP=8, 9 ms → TP=16 across nodes with a ring but TP=8 with a latency-optimal "
      f"all-reduce; tokens per GPU-second fall from {per_gpu[2]:.1f} (TP=2) to {per_gpu[16]:.1f} (TP=16)")

# %% [markdown]
# ## Exercise 3.4 — size a two-tier fabric
# The fabric has 1,024 GPUs, one 400 Gb/s NIC for each GPU, and radix-64 switches. Write
# `size_fabric(endpoints, radix, down, gbps)`. It returns `(leaves, spines, oversubscription, bisection_GBps)`.
# Each leaf gives `down` ports to hosts and the other ports to spines. Each spine port takes one leaf uplink.

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
# ## Exercise 3.5 — read nvidia-smi topo -m
# Here is a topology matrix in the documented format. It is for a hypothetical 2-socket, 4-GPU,
# 2-NIC server (**illustrative, not captured from a machine**). Write `best_pair(matrix)`. It returns the GPU pair
# with the best connection (use `fabric.topo_rank`, lower is better). Also write `closest_nic(matrix, gpu)`.

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
# **The two-minute version.** "Every transfer is $\alpha + n/\beta$. Tensor parallelism makes two
# all-reduces per layer per step, thus 160 for a 70B model. At decode, they are 16 KiB each, thus they are
# latency-bound. The cost is the count × the latency of the algorithm. This is why engines use
# latency-optimal all-reduce kernels.
#
# "At prefill, they are tens of MB and bandwidth-bound, and that cost does not decrease as TP grows. At TP=8 on
# NVLink, it is 46 ms per 4K-token step against 74 ms of compute per GPU. At TP=16 across two nodes, it is
# 75 ms against 37 ms, even with a NIC per GPU and the traffic spread over all NICs. Thus TP stays inside the
# NVLink domain. Across nodes, I use pipeline or data parallelism, and one NIC per GPU on a rail-optimized,
# non-blocking fabric. I examine `nvidia-smi topo -m` before I place ranks."
#
# **Drill.**
# 1. *Why not TP=16 across two 8-GPU H100 nodes?* Communication becomes larger than compute. With rails (the traffic
#    spread over every NIC), a 4K-token prefill step spends ~75 ms in all-reduce. Its compute per GPU is ~37 ms
#    (TP=8 on NVLink: 46 ms against 74 ms). Thus the step is barely faster for twice the GPUs. With one NIC per node,
#    it is ~263 ms, and as a flat ring through the NICs, ~427 ms. Use PP=2 × TP=8, FP8 to fit in one node, or a
#    larger NVLink domain.
# 2. *nccl-tests shows busbw of 20 GB/s at 64 KB. Is the fabric broken?* No. At 64 KB, the collective is
#    latency-bound (below $p \cdot \alpha \cdot \beta$). Use the large-message plateau to judge the fabric.
# 3. *Is a 3:1 oversubscribed fabric fine for inference?* Often it is, for independent replicas (mostly
#    north-south traffic). It is not fine for training all-reduces or disaggregated KV transfer, because these need bisection.
