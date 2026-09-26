# %% [markdown]
# # 04 · Expert parallelism and all-to-all
#
# **Tier:** T0 — numpy, a few seconds; every time printed is an α-β or roofline model (simulated), with layer 01's
# illustrative link numbers. To run EP = 2 for real, use `../moe-lab/notebooks/04_expert_parallelism_on_two_gpus.ipynb`
# (T2: Kaggle's free 2×T4; it falls back to this notebook's model without two GPUs).
#
# ## The one-minute version
# **Expert parallelism** (EP) puts E/p whole experts on each of p GPUs. Every MoE layer then moves *tokens*, not
# weights: a **dispatch** all-to-all sends each token's hidden state to the GPUs holding its k experts, and a
# **combine** all-to-all brings the k weighted results back — each at most `tokens × k × hidden × bytes` per GPU
# per direction, (p − 1)/p of it off the GPU under uniform routing (layer 02 PRIMER §5.6: 4 MiB, 22 µs pairwise or
# 10 µs direct for a Mixtral-like layer at 256 tokens per GPU). Then everyone waits for the **slowest rank**: a hot
# expert makes its GPU the step time, which is why engines rebalance and replicate hot experts (vLLM's EPLB).
# Large MoE deployments pair EP with **data-parallel attention** (wide-EP): attention and its KV are per rank,
# experts are spread thin — which frees HBM for KV and pools every rank's tokens at each expert. After this notebook
# you can price the all-to-alls on NVLink, PCIe and InfiniBand, find the slowest rank, and choose an EP degree.
#
# Primer: `../PRIMER.md` §6 *Running MoE on GPUs*; the collective itself: layer 02 PRIMER §5; wide-EP in a fleet:
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
# Per GPU and direction, every assignment remote (the upper bound): `tokens × k × hidden × bytes`. DeepSeek-V3
# dispatches in FP8 (1 byte, plus a 4-byte scale per 128 channels) and combines in BF16. Eight GPUs, direct
# all-to-all (every link at once), per MoE layer; a model has 32 (Mixtral) or 58 (DeepSeek-V3) of them per step.

# %%
print(f"{'tokens/GPU':>10s} {'link':>9s} | {'Mixtral dispatch':>20s} | {'DeepSeek dispatch':>20s} | {'DeepSeek combine':>20s}")
for tok in (8, 64, 4096):
    for lk in ("nvlink4", "pcie4x16", "ib-ndr"):
        d = E.dispatch_bytes(tok, 2, 4096)
        dd, dc = E.dispatch_bytes(tok, 8, 7168, 1, 128), E.dispatch_bytes(tok, 8, 7168, 2)
        cell = lambda b: f"{b / 2 ** 20:7.2f} MiB {E.a2a_time(b, 8, L[lk]) * 1e6:7.1f} us"
        print(f"{tok:10d} {lk:>9s} | {cell(d)} | {cell(dd)} | {cell(dc)}")

# %% [markdown]
# Decode moves little data per layer (latency-bound: α dominates), prefill moves a lot (bandwidth-bound: 231 MiB
# per dispatch for DeepSeek at 4,096 tokens per GPU is ~0.5 ms on NVLink and ~4 ms over one 400 Gb/s NIC per GPU).
# That is why DeepEP ships two kernel families: low-latency (decode, CUDA-graph friendly) and high-throughput
# (prefill, NVLink-then-RDMA forwarding). The formula explains DeepEP's published EP8 low-latency figures: 128
# tokens × 8 × (7,168 + 7,168/128 × 4) B = 7,569,408 B in 77 µs = 98.3 GB/s (reported 98).
#
# ## Worked example 2 — the slowest rank sets the layer
# Qwen3-30B-A3B's 128 experts on 8 GPUs, 128 tokens per GPU. Uniform routing, then a skewed workload (Zipf s = 1.0
# over a random order of experts — a model of hot experts, simulated). Each rank computes the rows routed to its
# experts; the layer waits for the busiest.

# %%
rng = np.random.default_rng(0)
hot_pop = T.zipf_popularity(128, 1.0)[rng.permutation(128)]
origin = np.repeat(np.arange(8), 128)
h100, nv = D["h100-sxm"], L["nvlink4"]
layouts = {}
for name, pop in (("uniform", None), ("skewed", hot_pop)):
    idx = T.sample_routes(128, 8, 1024, pop, np.random.default_rng(1))
    layouts[name] = idx
    for strat in ("linear", "round_robin"):
        m = E.exchange(idx, origin, E.placement(128, 8, strat), 8)
        lt = E.layer_time(m, Q3.expert_params(), 2048, h100, nv)
        print(f"{name:8s} {strat:12s} rows per rank {m.sum(0).tolist()}  max/mean {lt.imbalance:.2f}  "
              f"layer {lt.total * 1e6:5.1f} us (2 x {lt.comm * 1e6:.1f} comm + {lt.compute.max() * 1e6:.1f} slowest compute)")

# %% [markdown]
# Under skew the busiest GPU does 1.6× the average work and the whole layer waits for it. Placement strategy alone
# (vLLM's `--expert-placement-strategy linear | round_robin`) moves the hot spot but does not remove it. EPLB
# measures the load over a window and re-places experts, replicating the hottest ones (`num_redundant_experts`):

# %%
load = np.bincount(layouts["skewed"].ravel(), minlength=128)
for red in (0, 8, 16):
    r = E.rebalance(load, 8, red)
    print(f"EPLB-style packing with {red:2d} redundant experts: max/mean {r.max() / r.mean():.3f}")
extra = E.wide_ep_weights(DS, 32, 1, redundant=32) - E.wide_ep_weights(DS, 32, 1)
print(f"the price: one redundant DeepSeek-V3 expert per rank = 58 layers x {DS.expert_params():,} B (FP8) = "
      f"{extra / 2 ** 30:.2f} GiB of HBM per GPU (vLLM's doc: ~2.4 GB)")

# %% [markdown]
# ## Worked example 3 — TP or EP for the experts, and the wide-EP layout
# Without `--enable-expert-parallel`, vLLM shards every expert across the GPUs like any other MLP (tensor
# parallel): every GPU holds the same tokens and a ring all-reduce restores the output. With EP and data-parallel
# attention, each GPU holds its own tokens and whole experts, and ships only the assignments.

# %%
for p in (2, 8):
    tp = E.moe_comm(64, 2, 4096, p, nv, "tp")
    ep = E.moe_comm(64 // p, 2, 4096, p, nv, "ep")
    print(f"Mixtral MoE layer, batch 64 on {p} GPUs: TP all-reduce {tp[0] / 1024:6.0f} KiB {tp[1] * 1e6:5.1f} us | "
          f"EP dispatch+combine {ep[0] / 1024:5.0f} KiB {ep[1] * 1e6:5.1f} us (per GPU sent, simulated)")
print()
for n in (8, 16, 32, 64):
    w = E.wide_ep_weights(DS, n, 1)
    print(f"DeepSeek-V3 FP8, wide-EP {n:2d}: {w / 1e9:5.1f} GB weights per H200 ({w / 141e9:4.0%} of HBM), "
          f"4K-token sequences that fit: {S.sessions(DS, D['h200'], n, 4096, 8, layout='ep'):6,}")

# %% [markdown]
# 17.1 B parameters (MLA, three dense layers, the shared experts, routers, embeddings) are replicated on every rank;
# only the routed experts shrink with EP. Past EP 32 the replicated part dominates each GPU's weights.
#
# ## Exercise 4.1 — dispatch bytes
# Write `dispatch(tokens, k, hidden, elem_bytes, scale_block=0)`: bytes one GPU sends per direction when every
# assignment is remote, adding a 4-byte scale per `scale_block` channels when `scale_block` is set (FP8).

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
# Write `a2a(size, p, gbs, alpha_us, algo)`: pairwise = p − 1 steps each of α + size/p ÷ B; direct = one step of
# α + (p − 1)/p · size ÷ B. Reproduce layer 02's 22 µs and 10 µs for 4 MiB on 8 GPUs (450 GB/s, α = 2 µs).

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
# Write `exchange_matrix(idx, origin, where, p)`: a p × p array whose [src, dst] entry counts the assignments of
# tokens that live on rank `origin[t]` to experts that live on rank `where[e]`.

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
idx_u = layouts["uniform"]
where = E.placement(128, 8)
mine = exchange_matrix(idx_u, origin, where, 8)
assert (mine == E.exchange(idx_u, origin, where, 8)).all()
print(f"✅ each rank sends {mine.sum(1)[0]} rows; {1 - np.trace(mine) / mine.sum():.1%} leave their GPU (uniform: 7/8)")

# %% [markdown]
# ## Exercise 4.4 — what does a hot rank cost?
# For the skewed routing with linear placement, set `slow` to the layer time (seconds) from `E.layer_time`, and
# `ideal` to the time if the same total rows were spread evenly (same comm, compute = mean of the ranks'). Then
# `penalty = slow / ideal`.

# %% exercise
### BEGIN SOLUTION
m_skew = E.exchange(layouts["skewed"], origin, E.placement(128, 8), 8)
lt = E.layer_time(m_skew, Q3.expert_params(), 2048, h100, nv)
slow = lt.total
ideal = 2 * lt.comm + lt.compute.mean()
penalty = slow / ideal
### END SOLUTION

# %% check
assert 1.1 < penalty < 1.5 and np.isclose(slow, 2 * lt.comm + lt.compute.max())
print(f"✅ the hot rank makes this layer {penalty:.2f}x slower than a balanced one ({lt.imbalance:.2f}x on the expert GEMMs alone; simulated) - on every one of "
      "48 layers, every step")

# %% [markdown]
# ## Exercise 4.5 — choose the EP degree
# DeepSeek-V3 in FP8 on H200s (141 GB), wide-EP. Pick `ep_degree` from (8, 16, 32, 64): the smallest whose
# per-GPU weights (`E.wide_ep_weights(DS, n, 1)`) use at most half of HBM, leaving the rest for KV and activations.

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
# Two 8-GPU H200 nodes, DeepSeek-V3 FP8, batch 512 at 4K context, EP 16 with DP attention. Set `nvlink` and `ib`
# to the step-time dicts from `E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", link, 1)` for `L["nvlink4"]` (as
# if all 16 shared one NVLink domain) and `L["ib-ndr"]`, and `comm_share_ib` to the fraction of the IB step spent
# in all-to-alls.

# %% exercise
### BEGIN SOLUTION
nvlink = E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", L["nvlink4"], 1)
ib = E.decode_on(DS, D["h200"], 512, 4096, 16, "ep", L["ib-ndr"], 1)
comm_share_ib = ib["comm"] / ib["time"]
### END SOLUTION

# %% check
assert nvlink["roofline"] == ib["roofline"] and ib["comm"] > 5 * nvlink["comm"]
assert 0.3 < comm_share_ib < 0.45
print(f"✅ same roofline ({ib['roofline'] * 1e3:.1f} ms), all-to-alls {nvlink['comm'] * 1e3:.1f} ms on NVLink vs "
      f"{ib['comm'] * 1e3:.1f} ms over 400 Gb/s ({comm_share_ib:.0%} of the step, no overlap, simulated). Hence NVL72-"
      "class racks, DeepEP's RDMA kernels, and two-batch overlap (one micro-batch computes while the other communicates)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "With expert parallelism each GPU owns E/p whole experts, and each MoE layer does two
# all-to-alls — dispatch the tokens to their experts, combine the results — each at most tokens × k × hidden ×
# bytes per GPU. At decode sizes that is latency-bound, tens of microseconds per layer on NVLink; at prefill sizes
# it is bandwidth-bound, hundreds of microseconds on NVLink and milliseconds across 400 Gb/s NICs. The step then
# waits for the busiest GPU, so hot experts cost everyone: placement moves the hot spot, EPLB-style rebalancing with
# a few redundant experts removes most of it for a couple of GiB per GPU. We pair EP with data-parallel attention
# (wide-EP): attention weights are replicated but the KV is per rank and the experts are spread thin, so each GPU
# has room for a big batch and every expert sees all ranks' tokens. We keep EP inside the NVLink domain when we
# can; across nodes we budget for the all-to-all or overlap it."
#
# **Drill questions**
# 1. *TP or EP for Mixtral on 8 GPUs at decode?* — EP with DP attention: two small all-to-alls of each GPU's own
#    tokens vs a ring all-reduce of the whole batch (and TP's all-reduces in attention too); TP is simpler and fine
#    at small scale, EP scales.
# 2. *Your EP=16 deployment is slower per token than EP=8. Name two causes.* — The all-to-all now crosses the
#    network (α and β both worse), and the replicated attention weights are a larger share of each GPU's bytes;
#    also fewer tokens per expert per GPU.
# 3. *What does one redundant expert per rank cost for DeepSeek-V3 in FP8?* — 58 × 44 MB ≈ 2.4 GiB of HBM per GPU.
