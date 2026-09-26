# %% [markdown]
# # 03 · Which experts a batch touches
#
# **Tier:** T0 — numpy and arithmetic, a few seconds; every step time here is a roofline bound (simulated). To
# measure decode step time against batch for a real MoE and a dense model, continue with
# `../moe-lab/notebooks/03_batch_vs_weight_stream.ipynb` (T1; it prints this notebook's predictions when no GPU is
# found).
#
# ## The one-minute version
# One token reads k of E experts per layer. A decode step reads the **union** of its tokens' choices: with uniform
# routing a layer touches `E (1 − (1 − k/E)^T)` distinct experts for T tokens — the closed form layer 01 uses in
# `roofline.llm.experts_touched` (its PRIMER §3.6). So the weight bytes a step streams climb from about *active*
# at batch 1 to about *total* by a batch of a few × E/k, while the FLOPs stay proportional to *active × batch*.
# Bytes → total, FLOPs ∝ active: the batch at which decode turns compute-bound grows by about total ÷ active —
# 754 for Mixtral and 2,055 for Qwen3-30B-A3B on an H200, against 207 for a dense 8B. That is why MoE serving is
# about big batches. The KV cache follows attention, not experts, so it is unchanged — which shifts the KV share
# of a step down. Skewed routing touches fewer experts and heats one.
#
# Primer: `../PRIMER.md` §5 *MoE at inference: which experts a step touches*.

# %%
import math

import numpy as np

from moecore import touched as T
from moecore.sizing import MODELS

H200 = T.DEVICES["h200"]
MIX, Q3, L8 = MODELS["mixtral-8x7b"], MODELS["qwen3-30b-a3b"], MODELS["llama-3.1-8b"]

# %% [markdown]
# ## Worked example 1 — the union grows with the batch
# Distinct experts per layer, uniform routing. Coarse experts saturate fast; fine-grained ones keep the saving
# to larger batches.

# %%
fams = [("Mixtral-8x7B", 8, 2), ("OLMoE-1B-7B", 64, 8), ("gpt-oss-120b", 128, 4), ("Qwen3-30B-A3B", 128, 8),
        ("Llama 4 Maverick", 128, 1), ("DeepSeek-V3", 256, 8)]
batches = (1, 4, 16, 64, 256, 1024)
print(f"{'model (E, k)':24s}" + "".join(f"{b:>9d}" for b in batches))
for name, e, k in fams:
    print(f"{name + f' ({e}, {k})':24s}" + "".join(f"{T.experts_touched(e, k, b):9.1f}" for b in batches))

# %% [markdown]
# The closed form is exact for independent tokens that each pick k *distinct* experts uniformly: one token misses a
# given expert with probability 1 − k/E, T tokens miss it with (1 − k/E)^T. A Monte Carlo draw agrees; a skewed
# (Zipf) popularity — a *model* of hot experts, not a measurement — touches fewer experts and loads the hottest
# one several times the mean.

# %%
print(f"{'Qwen3 (128, 8), T = 16':24s} {'touched':>8s} {'hottest / mean load':>20s}")
print(f"{'closed form':24s} {T.experts_touched(128, 8, 16):8.1f}")
for s in (0.0, 0.5, 1.0, 1.5):
    touched, hot = T.touched_mc(128, 8, 16, s=s)
    print(f"{'Zipf s = ' + str(s) + ' (simulated)':24s} {touched:8.1f} {hot:20.1f}")

# %% [markdown]
# ## Worked example 2 — the decode step on the roofline
# `touched.decode_step` is the same one-kernel model as layer 01's `roofline.llm.decode()`: weights streamed
# (attention, the touched experts, router, LM head, one embedding row per token) plus the KV read, against the
# FLOPs of active × batch. It reproduces layer 01's §3.6 table to the byte (a test pins it).

# %%
print(f"{'batch':>5s} {'Mixtral experts':>15s} {'step bytes':>11s} {'step ms':>8s} {'bound':>7s} {'tok/s':>8s} {'Qwen3 experts':>14s}")
for b in (1, 4, 16, 64, 256, 1024):
    s = T.decode_step(MIX, H200, b, 1024)
    print(f"{b:5d} {T.experts_touched(8, 2, b):15.2f} {s.bytes / 1e9:9.1f} GB {s.time * 1e3:8.2f} {s.bound:>7s} "
          f"{b / s.time:8.0f} {T.experts_touched(128, 8, b):14.1f}")
x = {m.name: T.decode_crossover_batch(m, H200, 0) for m in (L8, MIX, Q3)}
print("\ndecode turns compute-bound (c = 0, H200, ridge 206):", x)

# %% [markdown]
# Mixtral decodes like a 13B model at batch 1 and like a 47B model from batch 16. Throughput still climbs with the
# batch — the full weight stream is shared by more tokens — but the batch at which the step stops being
# memory-bound is 3.6 times the dense model's. Fine-grained Qwen3 is 9.9 times (its streamed ÷ multiplied weight
# ratio, without the input embedding, which is gathered, not streamed).
#
# ## Worked example 3 — the KV cache does not care
# An MoE caches exactly what its attention caches: Mixtral's GQA is Llama-3.1-8B's shape, so both store 128 KiB per
# token. The same KV bytes are a smaller share of an MoE step, so at equal context and batch the MoE stays
# weight-bound longer — and long context shifts the balance back (notebook 05).

# %%
for b, c in ((64, 1024), (64, 32_768)):
    print(f"batch {b}, context {c:>6,}: KV share of the step - Mixtral {T.kv_share(MIX, b, c):5.1%}, "
          f"Llama-3.1-8B {T.kv_share(L8, b, c):5.1%}, Qwen3-30B-A3B {T.kv_share(Q3, b, c):5.1%}")

# %% [markdown]
# ## Exercise 3.1 — derive the closed form
# Write `touched(e, k, t)`: the expected number of distinct experts one layer reads for t tokens, uniform routing,
# k distinct experts per token. Then check it against a simulation.

# %% exercise
def touched(e, k, t):
    ### BEGIN SOLUTION
    return e * (1 - (1 - k / e) ** t)
    ### END SOLUTION

# %% check
for e, k, t in ((8, 2, 1), (8, 2, 16), (128, 8, 16), (256, 8, 64)):
    assert math.isclose(touched(e, k, t), T.experts_touched(e, k, t))
mc, _ = T.touched_mc(256, 8, 32)
assert abs(touched(256, 8, 32) - mc) / mc < 0.02
print(f"✅ E(1-(1-k/E)^T): DeepSeek-V3 touches {touched(256, 8, 32):.1f} of 256 experts at 32 tokens "
      f"(Monte Carlo {mc:.1f}, simulated)")

# %% [markdown]
# ## Exercise 3.2 — when is the saving gone?
# Solve the closed form for T. Set `t_mixtral` to the smallest batch at which Mixtral touches at least 7.5 of 8
# experts per layer, and `t_qwen` to the smallest at which Qwen3-30B-A3B touches at least 120 of 128.

# %% exercise
### BEGIN SOLUTION
t_mixtral = math.ceil(math.log(1 - 7.5 / 8) / math.log(1 - 2 / 8))
t_qwen = math.ceil(math.log(1 - 120 / 128) / math.log(1 - 8 / 128))
### END SOLUTION

# %% check
assert T.experts_touched(8, 2, t_mixtral) >= 7.5 > T.experts_touched(8, 2, t_mixtral - 1)
assert T.experts_touched(128, 8, t_qwen) >= 120 > T.experts_touched(128, 8, t_qwen - 1)
print(f"✅ Mixtral streams ~94% of its experts from batch {t_mixtral}; Qwen3 from batch {t_qwen}. "
      "Both are small next to a serving batch: at decode batch 64 Mixtral, OLMoE and Qwen3 stream nearly every expert "
      "(top-1 Llama 4 Maverick about 40%)")

# %% [markdown]
# ## Exercise 3.3 — the bytes of a step
# Write `step_bytes(cfg, batch, context)` for bf16 weights and KV: every layer's attention, router and shared
# experts, the touched routed experts (`T.experts_touched`), the LM head, one embedding row per token, plus each
# sequence's KV read and one new entry (`(context + 1) × cfg.kv_bytes_per_token()`). Use the `MoEConfig` methods
# (`attn_params`, `expert_params`, `router_params`, `moe_layer_params`, `dense_layer_params`, ...).

# %% exercise
def step_bytes(cfg, batch, context):
    ### BEGIN SOLUTION
    e = T.experts_touched(cfg.n_experts, cfg.top_k, batch)
    layers = cfg.moe_layers * cfg.moe_layer_params(e) + (cfg.layers - cfg.moe_layers) * cfg.dense_layer_params()
    weights = (layers + cfg.vocab * cfg.d_model + batch * cfg.d_model) * 2
    return weights + batch * (context + 1) * cfg.kv_bytes_per_token()
    ### END SOLUTION

# %% check
for cfg in (MIX, Q3, L8, MODELS["deepseek-v3"]):
    for b, c in ((1, 1024), (64, 4096)):
        assert math.isclose(step_bytes(cfg, b, c), T.decode_step(cfg, H200, b, c).bytes, rel_tol=1e-12), cfg.name
print(f"✅ Mixtral, batch 1, 1K context: {step_bytes(MIX, 1, 1024):,.0f} B - the 25.6 GB of layer 01's table")

# %% [markdown]
# ## Exercise 3.4 — predict the crossover from the dense one
# Weights streamed at large batch ≈ total; FLOPs ∝ active. So crossover(MoE) ≈ crossover(dense) × (streamed ÷
# multiplied weights). Using only Llama-3.1-8B's crossover (207 on an H200) and Qwen3's config, predict Qwen3's.
# Leave the input embedding out of both counts (it is a gather: neither streamed nor multiplied).

# %% exercise
### BEGIN SOLUTION
emb = Q3.vocab * Q3.d_model
predicted = 207 * (Q3.total() - emb) / (Q3.active() - emb)
### END SOLUTION

# %% check
actual = T.decode_crossover_batch(Q3, H200, 0)
assert abs(predicted - actual) / actual < 0.01
print(f"✅ predicted {predicted:.0f}, bisection says {actual:,} - the ratio carries the whole effect")

# %% [markdown]
# ## Exercise 3.5 — rows per expert
# Inside the step, each touched expert runs a GEMM on the rows routed to it: on average `batch × k ÷ E` rows. That
# row count is the expert GEMM's arithmetic intensity (FLOP per weight byte at 2-byte weights). For batch 256, set
# `rows` to a dict of rows per expert for Mixtral (8, 2), Qwen3 (128, 8) and DeepSeek-V3 (256, 8), and
# `batch_for_ridge` to the batch each needs for the average expert to see 206 rows (the H200's ridge).

# %% exercise
### BEGIN SOLUTION
cfgs = {"mixtral": (8, 2), "qwen3": (128, 8), "deepseek-v3": (256, 8)}
rows = {n: 256 * k / e for n, (e, k) in cfgs.items()}
batch_for_ridge = {n: math.ceil(206 * e / k) for n, (e, k) in cfgs.items()}
### END SOLUTION

# %% check
assert rows == {"mixtral": 64.0, "qwen3": 16.0, "deepseek-v3": 8.0}
assert batch_for_ridge == {"mixtral": 824, "qwen3": 3296, "deepseek-v3": 6592}
print("✅ each expert sees only B·k/E of the batch: DeepSeek-V3 needs ~6,600 tokens per step for its experts' "
      "GEMMs to reach the ridge - more than one engine's decode batch; expert parallelism pools many GPUs' tokens")

# %% [markdown]
# ## Exercise 3.6 — skew changes the bytes
# Hot experts are a workload property: a code-heavy batch favours a few experts. Model it with Zipf popularity
# s = 1.0 (simulated) for Qwen3-30B-A3B at batch 16, 1K context: set `skew_touched` from `T.touched_mc`, then
# `speedup` = uniform step time ÷ skewed step time (`T.decode_step(..., touched=skew_touched)`).

# %% exercise
### BEGIN SOLUTION
skew_touched, hot = T.touched_mc(128, 8, 16, s=1.0)
speedup = T.decode_step(Q3, H200, 16, 1024).time / T.decode_step(Q3, H200, 16, 1024, touched=skew_touched).time
### END SOLUTION

# %% check
assert skew_touched < T.experts_touched(128, 8, 16) and 1.1 < speedup < 1.6
print(f"✅ skew: {skew_touched:.1f} experts touched instead of {T.experts_touched(128, 8, 16):.1f}, the step is "
      f"{speedup:.2f}x faster on one GPU (simulated) - and the hottest expert carries {hot:.1f}x the mean load, "
      "which is what hurts once experts live on different GPUs (notebook 04)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Per token an MoE computes like its active parameters, but a decode step streams
# every expert that *any* token in the batch picked. With uniform routing that is E(1 − (1 − k/E)^T) experts per
# layer, so Mixtral is 13B-like at batch 1 and streams all 93 GB by batch 16, and fine-grained models like
# Qwen3-30B-A3B hold the saving a bit longer. Throughput per step still improves with batch, but the batch at
# which decode becomes compute-bound scales with total ÷ active: about 750 for Mixtral and 2,000 for Qwen3 on an
# H200, versus 200 for a dense 8B, because each expert only sees B·k/E rows. So MoE wants big batches — which on
# one GPU run into KV memory, and across GPUs means expert parallelism. The KV cache is whatever the attention
# says; MoE does not change it, prefix caching works the same, and at long context the KV share climbs back."
#
# **Drill questions**
# 1. *Mixtral at batch 1 vs batch 64: how do the bytes per step compare?* — 25.6 GB vs 101.7 GB on the roofline:
#    2 of 8 experts per layer vs all 8, plus the KV.
# 2. *Why is 'decode is compute-bound from batch ≈ ridge' wrong for MoE?* — The weight stream approaches total
#    parameters while FLOPs follow active ones, so the crossover is ridge × (streamed ÷ multiplied).
# 3. *Does MoE change prefix caching or KV sizing?* — No: both follow attention. Only the share of KV in the step
#    changes.
