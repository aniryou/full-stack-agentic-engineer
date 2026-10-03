# %% [markdown]
# # 04 · Expert parallelism and all-to-all
#
# **Tier:** T0. It needs numpy and a few seconds. Every printed time is an α-β or roofline model (simulated), with the
# illustrative link numbers of layer 01. To run EP = 2 on real GPUs, use
# `../moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb` (T2: the free 2×T4 of Kaggle). Without two GPUs, that
# notebook falls back to the model of this notebook.
#
# ## The one-minute version
# **Expert parallelism** (EP) puts $E/p$ whole experts on each of $p$ GPUs. Every MoE layer then moves *tokens*, not
# weights. A **dispatch** all-to-all sends the hidden state of each token to the GPUs that hold its $k$ experts. A
# **combine** all-to-all brings the $k$ weighted results back.
#
# With dedicated all-to-all kernels (DeepEP-class), each all-to-all is at most
# $\text{tokens} \times k \times \text{hidden} \times \text{bytes}$ per GPU per direction. Under uniform routing,
# $(p - 1)/p$ of it goes off the GPU. The layer 02 PRIMER §5.6 gives 4 MiB, 22 µs pairwise or 10 µs direct for a
# Mixtral-like layer at 256 tokens per GPU. The default exchange of vLLM moves the volume of tensor parallelism
# instead.
#
# Then all ranks wait for the **slowest rank**. The expert work of each rank is a roofline of its own. In prefill, the
# rank that holds the hot experts computes the most rows and sets the step. In memory-bound decode, every rank streams
# its touched experts, whatever the skew, and the skew goes to the link of the hot rank. That is why engines rebalance
# and replicate hot experts (the EPLB of vLLM).
#
# Large MoE deployments use EP together with **data-parallel attention** (wide-EP). The attention and its KV are per
# rank, and the experts are spread thin. This layout frees HBM for KV, and it collects the tokens of every rank at
# each expert.
#
# After this notebook, you can calculate the cost of the all-to-alls on NVLink, PCIe and InfiniBand. You can also find
# the slowest rank and select an EP degree.
#
# Primer: `../PRIMER.md` §6 *Running MoE on GPUs*. The collective itself: layer 02 PRIMER §5. Wide-EP in a fleet:
# layer 05 PRIMER §8.

# %%
import numpy as np

from moecore import ep as E
from moecore import sizing as S
from moecore import touched as T

M, L, D = S.MODELS, E.LINKS, T.DEVICES
MIX, Q3, DS = M["mixtral-8x7b"], M["qwen3-30b-a3b"], M["deepseek-v3"]
for name, link in L.items():
    print(f"{name:10s} {link.name:28s} {link.gbs:7.2f} GB/s per direction, alpha {link.alpha_us} us (illustrative)")

# %% [markdown]
# ## Worked example 1 — what one all-to-all costs
# Per GPU and direction, with every assignment remote (the upper bound), the volume is
# $\text{tokens} \times k \times \text{hidden} \times \text{bytes}$. DeepSeek-V3 dispatches in FP8 (1 byte, plus a
# 4-byte scale per 128 channels) and combines in BF16. The table is for eight GPUs and a direct all-to-all (every link
# at the same time), per MoE layer. A model has 32 (Mixtral) or 58 (DeepSeek-V3) of these layers per step.

# %%
print(f"{'tokens/GPU':>10s} {'link':>9s} | {'Mixtral dispatch':>20s} | {'DeepSeek dispatch':>20s} | {'DeepSeek combine':>20s}")
for tok in (8, 64, 4096):
    for lk in ("nvlink4", "pcie-l4", "ib-ndr"):
        d = E.dispatch_bytes(tok, 2, 4096)
        dd, dc = E.dispatch_bytes(tok, 8, 7168, 1, 128), E.dispatch_bytes(tok, 8, 7168, 2)
        cell = lambda b: f"{b / 2 ** 20:7.2f} MiB {E.a2a_time(b, 8, L[lk]) * 1e6:7.1f} us"
        print(f"{tok:10d} {lk:>9s} | {cell(d)} | {cell(dd)} | {cell(dc)}")

# %% [markdown]
# Decode moves a small quantity of data per layer (latency-bound: $\alpha$ is the largest term). Prefill moves a large
# quantity (bandwidth-bound). For DeepSeek at 4,096 tokens per GPU, one dispatch is 231 MiB. That is ~0.5 ms on NVLink
# and ~4 ms over one 400 Gb/s NIC per GPU. That is why DeepEP supplies two kernel families: low-latency (decode,
# CUDA-graph friendly) and high-throughput (prefill, which forwards over NVLink and then over RDMA).
#
# The formula explains the published EP8 low-latency figures of DeepEP. The dispatch is 128
# tokens × 8 × (7,168 + 7,168/128 × 4) B = 7,569,408 B in 77 µs = 98.3 GB/s (reported 98).
#
# ## Worked example 2 — the slowest rank sets the layer
# The setup puts the 128 experts of Qwen3-30B-A3B on 8 H100s (16 per GPU). It runs uniform routing, then a skewed
# workload. The skew is Zipf $s = 1.0$ over a random order of experts: a model of hot experts, simulated. Both run at a
# decode size (128 tokens per GPU) and at a prefill chunk (4,096 tokens per GPU). The expert work of each rank is a
# roofline of its own: its rows × 2 × expert params ÷ peak, against its touched experts × expert bytes ÷ bandwidth.
# The layer waits for the slowest rank, and each all-to-all waits for the busiest port.
#
# A port sends and receives at the same time. The combine is the dispatch in reverse.

# %%
rng = np.random.default_rng(0)
hot_pop = T.zipf_popularity(128, 1.0)[rng.permutation(128)]
h100, nv = D["h100-sxm"], L["nvlink4"]
layouts, results = {}, {}
for tpg in (128, 4096):
    origin_t = np.repeat(np.arange(8), tpg)
    for name, pop in (("uniform", None), ("skewed", hot_pop)):
        idx = T.sample_routes(128, 8, 8 * tpg, pop, np.random.default_rng(1))
        layouts[tpg, name] = idx
        for strat in ("linear", "round_robin"):
            lt = E.layer_time(idx, origin_t, E.placement(128, 8, strat), 8, Q3.expert_params(), 2048, h100, nv)
            results[tpg, name, strat] = lt
            print(f"{tpg:5d} tok/GPU {name:8s} {strat:12s} rows max/mean {lt.imbalance:.2f}, slowest rank "
                  f"{lt.bound:7s}-bound: layer {lt.total * 1e6:7.1f} us = 2 x {lt.comm * 1e6:5.1f} comm + "
                  f"{lt.rank.max() * 1e6:6.1f} experts ({lt.penalty:.2f}x balanced)")
origin = np.repeat(np.arange(8), 128)
d = results[128, "skewed", "linear"]
print(f"\ndecode, skewed: expert time per rank {np.round(d.rank * 1e6, 1).tolist()} us "
      f"(weight reads {d.memory.max() * 1e6:.1f} us; the hot rank's FLOPs {d.compute.max() * 1e6:.1f} us)")

# %% [markdown]
# At decode size, each expert sees ~64 rows, far below the ridge of the H100 (~295 FLOP/B). Every rank uses ~45 µs to
# stream the weights of its 16 experts, whatever the skew. The extra rows of the hot rank hide under that read. In
# decode, the cost of the skew is in the exchange. The port of the hot rank receives the dispatch and sends the
# combine for ~1.5× the mean traffic. At prefill size, the GEMMs are compute-bound, and the 1.65× rows of the busiest
# rank set the layer.
#
# Placement alone, for example `--expert-placement-strategy round_robin` of vLLM (expert $e$ on rank $e \bmod p$),
# moves the hot spot but does not remove it. Also, vLLM v0.30.0 applies round-robin only under these conditions
# (verify for your version):
# - the model has more than one expert group (DeepSeek-V3 has 8)
# - there are no redundant experts
# - EPLB is off
# - if the deployment uses all-to-all kernels, the backend is DeepEP low-latency or NIXL-EP
#
# For Qwen3, which has no groups, vLLM logs a warning and falls back to linear (verify for your version). EPLB
# measures the load over a window and moves the experts to new ranks. It replicates the hottest experts
# (`num_redundant_experts`):

# %%
load = np.bincount(layouts[128, "skewed"].ravel(), minlength=128)
for red in (0, 8, 16):
    r = E.rebalance(load, 8, red)
    print(f"EPLB-style packing with {red:2d} redundant experts: max/mean {r.max() / r.mean():.3f}")
extra = E.wide_ep_weights(DS, 32, 1, redundant=32) - E.wide_ep_weights(DS, 32, 1)
print(f"the price: one redundant DeepSeek-V3 expert per rank = 58 layers x {DS.expert_params():,} B (FP8) = "
      f"{extra / 2 ** 30:.2f} GiB of HBM per GPU (vLLM's doc: ~2.4 GB)")

# %% [markdown]
# ## Worked example 3 — TP or EP for the experts, and the wide-EP layout
# Without `--enable-expert-parallel`, vLLM shards every expert across the GPUs like any other MLP (tensor parallel).
# Every GPU holds the same tokens, and a ring all-reduce gives the full output again. With the flag and DP = 1
# (TP × EP), nothing changes on the wire. Every GPU already holds every token and runs its own experts, and the same
# all-reduce adds their outputs.
#
# With DP > 1, the default `allgather_reducescatter` of vLLM gathers the tokens of every rank and reduce-scatters the
# outputs. That is the volume of TP again. Only all-to-all kernels (DeepEP-class, Hopper with NVLink/RDMA) send only
# the assignments.

# %%
for p in (2, 8):
    tp = E.moe_comm(64, 2, 4096, p, nv, "tp")
    ag = E.moe_comm(64 // p, 2, 4096, p, nv, "agrs")
    a2 = E.moe_comm(64 // p, 2, 4096, p, nv, "a2a")
    print(f"Mixtral MoE layer, batch 64 on {p} GPUs (per GPU sent, simulated): TP or TP x EP all-reduce "
          f"{tp[0] / 1024:4.0f} KiB {tp[1] * 1e6:5.1f} us | DP + EP, allgather_reducescatter {ag[0] / 1024:4.0f} KiB "
          f"{ag[1] * 1e6:5.1f} us | DP + EP, all-to-all kernels {a2[0] / 1024:4.0f} KiB {a2[1] * 1e6:4.1f} us")
print()
for n in (8, 16, 32, 64):
    w = E.wide_ep_weights(DS, n, 1)
    print(f"DeepSeek-V3 FP8, wide-EP {n:2d}: {w / 1e9:5.1f} GB weights per H200 ({w / 141e9:4.0%} of HBM), "
          f"4K-token sequences that fit: {S.sessions(DS, D['h200'], n, 4096, 8, layout='ep'):6,}")

# %% [markdown]
# Every rank holds a replica of 17.1 B parameters (MLA, three dense layers, the shared experts, routers, embeddings).
# Only the routed experts become smaller with EP. Above EP 32, the replicated part is the larger part of the weights
# of each GPU.
#
# ## Exercise 4.1 — dispatch bytes
# Write `dispatch(tokens, k, hidden, elem_bytes, scale_block=0)`. It returns the bytes that one GPU sends per
# direction when every assignment is remote. When `scale_block` is not 0 (FP8), add a 4-byte scale per `scale_block`
# channels.

# %% exercise
def dispatch(tokens, k, hidden, elem_bytes, scale_block=0):
    ### BEGIN SOLUTION
    row = hidden * elem_bytes + (hidden // scale_block * 4 if scale_block else 0)
    return tokens * k * row
    ### END SOLUTION

# %% check
assert dispatch(256, 2, 4096, 2) == 4 * 2 ** 20                         # layer 02 PRIMER §5.6
assert dispatch(128, 8, 7168, 1, 128) == 7_569_408 and dispatch(128, 8, 7168, 2) == 14_680_064
print(f"✅ DeepEP EP8 low-latency: dispatch {dispatch(128, 8, 7168, 1, 128) / 77e-6 / 1e9:.1f} GB/s at 77 us "
      f"(reported 98), combine {dispatch(128, 8, 7168, 2) / 114e-6 / 1e9:.1f} GB/s at 114 us (reported 127)")

# %% [markdown]
# ## Exercise 4.2 — the α-β all-to-all
# Write `a2a(size, p, gbs, alpha_us, algo)`. Use these two algorithms:
# - pairwise = ${p - 1}$ steps, each of $\alpha + \text{size}/p \div B$.
# - direct = one step of $\alpha + (p - 1)/p \cdot \text{size} \div B$.
#
# Calculate again the 22 µs and 10 µs of layer 02 for 4 MiB on 8 GPUs (450 GB/s, $\alpha$ = 2 µs).

# %% exercise
def a2a(size, p, gbs, alpha_us, algo="direct"):
    ### BEGIN SOLUTION
    steps = p - 1 if algo == "pairwise" else 1
    return steps * alpha_us * 1e-6 + (p - 1) / p * size / (gbs * 1e9)
    ### END SOLUTION

# %% check
assert round(a2a(4 * 2 ** 20, 8, 450, 2, "pairwise") * 1e6) == 22 and round(a2a(4 * 2 ** 20, 8, 450, 2) * 1e6) == 10
for lk in L.values():
    assert np.isclose(a2a(1e6, 8, lk.gbs, lk.alpha_us), E.a2a_time(1e6, 8, lk))
print("✅ 22 us pairwise, 10 us direct: at decode sizes the alpha term is most of it - fewer steps win")

# %% [markdown]
# ## Exercise 4.3 — the exchange matrix
# Write `exchange_matrix(idx, origin, where, p)`. It returns a $p \times p$ array. Its [src, dst] entry counts the
# assignments of tokens that live on rank `origin[t]` to experts that live on rank `where[e]`.

# %% exercise
def exchange_matrix(idx, origin, where, p):
    ### BEGIN SOLUTION
    m = np.zeros((p, p), dtype=int)
    for t, experts in enumerate(idx):
        for e in experts:
            m[origin[t], where[e]] += 1
    return m
    ### END SOLUTION

# %% check
idx_u = layouts[128, "uniform"]
where = E.placement(128, 8)
mine = exchange_matrix(idx_u, origin, where, 8)
assert (mine == E.exchange(idx_u, origin, where, 8)).all()
print(f"✅ each rank sends {mine.sum(1)[0]} rows; {1 - np.trace(mine) / mine.sum():.1%} leave their GPU (uniform: 7/8)")

# %% [markdown]
# ## Exercise 4.4 — each rank is its own roofline
# Write `rank_time(rows, touched, expert_params, device, weight_bytes=2)`. Per rank, it returns the larger of two
# times:
# - its FLOP time (rows × 2 × `expert_params` ÷ `device.peak()`)
# - its weight-read time (touched experts × `expert_params` × `weight_bytes` ÷ `device.bandwidth()`)
#
# Then use the skewed routing with linear placement at 128 and at 4,096 tokens per GPU. Set `penalty[tpg]` = the time
# of the slowest rank ÷ the time of a mean rank. A mean rank has the mean rows and the mean touched experts. Count the
# expert work only. `E.exchange(...).sum(0)` gives the rows, and `E.touched_per_rank` gives the experts.

# %% exercise
def rank_time(rows, touched, expert_params, device, weight_bytes=2):
    ### BEGIN SOLUTION
    flops = np.asarray(rows, dtype=float) * 2 * expert_params / device.peak()
    reads = np.asarray(touched, dtype=float) * expert_params * weight_bytes / device.bandwidth()
    return np.maximum(flops, reads)
    ### END SOLUTION


### BEGIN SOLUTION
penalty = {}
where_lin = E.placement(128, 8)
for tpg in (128, 4096):
    idx_s = layouts[tpg, "skewed"]
    rows = E.exchange(idx_s, np.repeat(np.arange(8), tpg), where_lin, 8).sum(0)
    touched = E.touched_per_rank(idx_s, where_lin, 8)
    t = rank_time(rows, touched, Q3.expert_params(), h100)
    penalty[tpg] = t.max() / rank_time(rows.mean(), touched.mean(), Q3.expert_params(), h100)
### END SOLUTION

# %% check
for tpg in (128, 4096):
    r = results[tpg, "skewed", "linear"]
    assert np.allclose(rank_time(r.rows, E.touched_per_rank(layouts[tpg, "skewed"], E.placement(128, 8), 8),
                                 Q3.expert_params(), h100), r.rank)
assert penalty[128] < 1.02 and penalty[4096] > 1.5
print(f"✅ same skew, two regimes: in decode the hot rank's expert work is {penalty[128]:.2f}x a balanced rank's "
      f"(weight reads dominate), in prefill {penalty[4096]:.2f}x (FLOPs dominate) - on every one of 48 layers. "
      f"Decode still pays at the exchange: with the busiest port the layer is "
      f"{results[128, 'skewed', 'linear'].penalty:.2f}x slower (simulated)")

# %% [markdown]
# ## Exercise 4.5 — choose the EP degree
# The setup is DeepSeek-V3 in FP8 on H200s (141 GB), wide-EP. Select `ep_degree` from (8, 16, 32, 64). Select the
# smallest degree whose per-GPU weights (`E.wide_ep_weights(DS, n, 1)`) use at most half of HBM. The HBM that the
# weights do not use stays for KV and activations.

# %% exercise
### BEGIN SOLUTION
ep_degree = next(n for n in (8, 16, 32, 64) if E.wide_ep_weights(DS, n, 1) <= 0.5 * 141e9)
### END SOLUTION

# %% check
assert ep_degree == 16
print(f"✅ EP {ep_degree}: {E.wide_ep_weights(DS, 16, 1) / 1e9:.1f} GB per GPU - two 8-GPU nodes, so the "
      "all-to-all now crosses the scale-out network; llm-d's wide-EP guide runs DeepSeek-R1 at 16-way DP per role")

# %% [markdown]
# ## Exercise 4.6 — EP across the network
# The setup has these values:
# - two 8-GPU H200 nodes
# - DeepSeek-V3 FP8, batch 512 at 4K context
# - EP 16 with DP attention and all-to-all kernels
# - FP8 dispatch (`fp8 = dict(dispatch_elem=1, scale_block=128)`)
#
# Set `one_domain` to `E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", L["nvlink4"], 1, **fp8)`. This is as if all 16
# GPUs share one NVLink domain, an NVL72-class rack. Set `two_nodes` to the same over `L["ib-ndr"]` with
# `per_node=8, intra=L["nvlink4"]`. In this layout, each GPU has its 7 node peers on NVLink and the other 8 behind
# its 400 Gb/s NIC. Set `comm_share` to the fraction of the two-node step that goes to the all-to-alls.

# %% exercise
### BEGIN SOLUTION
fp8 = dict(dispatch_elem=1, scale_block=128)
one_domain = E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", L["nvlink4"], 1, **fp8)
two_nodes = E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", L["ib-ndr"], 1, **fp8, per_node=8, intra=L["nvlink4"])
comm_share = two_nodes["comm"] / two_nodes["time"]
### END SOLUTION

# %% check
assert one_domain["roofline"] == two_nodes["roofline"] and two_nodes["comm"] > 3 * one_domain["comm"]
assert 0.15 < comm_share < 0.3
upper = E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", L["ib-ndr"], 1, **fp8)
print(f"✅ same roofline ({two_nodes['roofline'] * 1e3:.1f} ms), all-to-alls {one_domain['comm'] * 1e3:.1f} ms in one NVLink "
      f"domain vs {two_nodes['comm'] * 1e3:.1f} ms across two nodes ({comm_share:.0%} of the step, no overlap; every peer "
      f"behind the NIC would be {upper['comm'] * 1e3:.1f} ms, simulated). Hence NVL72-class racks, DeepEP's RDMA kernels, "
      "and two-batch overlap (one micro-batch computes while the other communicates)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "With expert parallelism, each GPU owns $E/p$ whole experts. Each MoE layer does two
# all-to-alls: it dispatches the tokens to their experts, and it combines the results. With DeepEP-class kernels, each
# is at most $\text{tokens} \times k \times \text{hidden} \times \text{bytes}$ per GPU. The default exchange of vLLM
# moves the volume of TP, and TP × EP with DP = 1 only does an all-reduce.
#
# "At decode sizes, the all-to-all is latency-bound: tens of microseconds per layer on NVLink. At prefill sizes, it is
# bandwidth-bound: hundreds of microseconds on NVLink and milliseconds across 400 Gb/s NICs.
#
# "Then the step waits for the busiest GPU. In prefill, this is the GPU that computes the rows of the hot experts. In
# memory-bound decode, it is the GPU whose link carries the traffic of the hot experts. Placement moves the hot spot.
# An EPLB-style rebalance with a few redundant experts removes most of it, for a couple of GiB per GPU.
#
# "We use EP together with data-parallel attention (wide-EP). Each rank holds a replica of the attention weights, but
# the KV is per rank and the experts are spread thin. Thus each GPU has space for a large batch, and every expert sees
# the tokens of all ranks. When we can, we keep EP inside the NVLink domain. Across nodes, we put the all-to-all in the
# budget, or we overlap it."
#
# **Drill questions**
# 1. *TP or EP for Mixtral on 8 GPUs at decode?* EP with DP attention and all-to-all kernels. EP does two small
#    all-to-alls of the tokens that each GPU holds. TP does a ring all-reduce of the whole batch, and it also does
#    all-reduces in attention. With the default `allgather_reducescatter` of vLLM, the bytes are the same as for TP. Then the gain
#    of EP is the memory layout and larger expert GEMMs. TP is simpler and satisfactory at small scale. EP scales.
# 2. *Your EP=16 deployment is slower per token than EP=8. Name two causes.* The all-to-all now crosses the network
#    ($\alpha$ and $\beta$ are both worse). The replicated attention weights are a larger share of the bytes of each
#    GPU. There are also fewer tokens per expert per GPU.
# 3. *What does one redundant expert per rank cost for DeepSeek-V3 in FP8?* 58 × 44 MB ≈ 2.4 GiB of HBM per GPU.
