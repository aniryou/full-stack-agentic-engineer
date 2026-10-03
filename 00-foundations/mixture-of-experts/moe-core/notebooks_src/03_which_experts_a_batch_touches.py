# %% [markdown]
# # 03 · Which experts a batch touches
#
# **Tier:** T0. It needs numpy and arithmetic, and a few seconds. Every step time here is a roofline bound
# (simulated). To measure decode step time against batch for a real MoE and a dense model, continue with
# `../moe-lab/notebooks/03_batch_vs_weight_stream.ipynb` (T1). When that notebook finds no GPU, it prints the
# predictions of this notebook.
#
# ## The one-minute version
# One token reads $k$ of $E$ experts per layer. A decode step reads the **union** of the choices of its tokens. With
# uniform routing, a layer touches $E(1 - (1 - k/E)^T)$ different experts for $T$ tokens. This is the closed form that
# layer 01 uses in `roofline.llm.experts_touched` (its PRIMER §3.6). Thus the weight bytes that a step streams increase
# from approximately *active* at batch 1 to approximately *total* at a batch of a few × $E/k$. The FLOPs stay
# proportional to *active × batch*.
#
# The bytes go toward total, and the FLOPs are ∝ active. Thus the batch at which decode becomes compute-bound
# increases by approximately total ÷ active. On an H200, that batch is 754 for Mixtral and 2,055 for Qwen3-30B-A3B,
# against 207 for a dense 8B. That is why MoE serving is about large batches.
#
# The KV cache depends on the attention, not on the experts, so it does not change. Thus the KV share of a step moves
# down. Skewed routing touches fewer experts and makes one expert hot.
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
# The table shows the number of different experts per layer, with uniform routing. Coarse experts saturate fast.
# Fine-grained experts keep their smaller weight stream up to larger batches.

# %%
fams = [("Mixtral-8x7B", 8, 2), ("OLMoE-1B-7B", 64, 8), ("gpt-oss-120b", 128, 4), ("Qwen3-30B-A3B", 128, 8),
        ("Llama 4 Maverick", 128, 1), ("DeepSeek-V3", 256, 8)]
batches = (1, 4, 16, 64, 256, 1024)
print(f"{'model (E, k)':24s}" + "".join(f"{b:>9d}" for b in batches))
for name, e, k in fams:
    print(f"{name + f' ({e}, {k})':24s}" + "".join(f"{T.experts_touched(e, k, b):9.1f}" for b in batches))

# %% [markdown]
# The closed form is exact for independent tokens that each select $k$ *different* experts uniformly. One token misses
# a given expert with probability $1 - k/E$. $T$ tokens miss it with $(1 - k/E)^T$. A Monte Carlo draw agrees. A
# skewed (Zipf) popularity is a *model* of hot experts, not a measurement. It touches fewer experts and puts several
# times the mean load on the hottest expert.

# %%
print(f"{'Qwen3 (128, 8), T = 16':24s} {'touched':>8s} {'hottest / mean load':>20s}")
print(f"{'closed form':24s} {T.experts_touched(128, 8, 16):8.1f}")
for s in (0.0, 0.5, 1.0, 1.5):
    touched, hot = T.touched_mc(128, 8, 16, s=s)
    print(f"{'Zipf s = ' + str(s) + ' (simulated)':24s} {touched:8.1f} {hot:20.1f}")

# %% [markdown]
# ## Worked example 2 — the decode step on the roofline
# `touched.decode_step` is the same one-kernel model as `roofline.llm.decode()` of layer 01. It takes the streamed
# weights (attention, the touched experts, router, LM head, one embedding row per token) plus the KV read. It compares
# them against the FLOPs of active × batch. Its result agrees with the §3.6 table of layer 01 to the byte (a test
# makes sure of this).

# %%
print(f"{'batch':>5s} {'Mixtral experts':>15s} {'step bytes':>11s} {'step ms':>8s} {'bound':>7s} {'tok/s':>8s} {'Qwen3 experts':>14s}")
for b in (1, 4, 16, 64, 256, 1024):
    s = T.decode_step(MIX, H200, b, 1024)
    print(f"{b:5d} {T.experts_touched(8, 2, b):15.2f} {s.bytes / 1e9:9.1f} GB {s.time * 1e3:8.2f} {s.bound:>7s} "
          f"{b / s.time:8.0f} {T.experts_touched(128, 8, b):14.1f}")
x = {m.name: T.decode_crossover_batch(m, H200, 0) for m in (L8, MIX, Q3)}
print("\ndecode turns compute-bound (c = 0, H200, ridge 206):", x)

# %% [markdown]
# Mixtral decodes like a 13B model at batch 1 and like a 47B model from batch 16. Throughput still increases with the
# batch, because more tokens share the full weight stream. But the step is memory-bound until a batch 3.6 times that
# of the dense model. For fine-grained Qwen3, it is 9.9 times. This is its streamed ÷ multiplied weight ratio, without
# the input embedding. The model gathers the input embedding and does not stream it.
#
# ## Worked example 3 — the KV cache does not care
# An MoE caches exactly what its attention caches. The GQA of Mixtral has the shape of Llama-3.1-8B, so both store
# 128 KiB per token. The same KV bytes are a smaller share of an MoE step. Thus, at equal context and batch, the MoE
# stays weight-bound longer. But long context moves the balance back (notebook 05).

# %%
for b, c in ((64, 1024), (64, 32_768)):
    print(f"batch {b}, context {c:>6,}: KV share of the step - Mixtral {T.kv_share(MIX, b, c):5.1%}, "
          f"Llama-3.1-8B {T.kv_share(L8, b, c):5.1%}, Qwen3-30B-A3B {T.kv_share(Q3, b, c):5.1%}")

# %% [markdown]
# ## Exercise 3.1 — derive the closed form
# Write `touched(e, k, t)`. It returns the expected number of different experts that one layer reads for $t$ tokens.
# Use uniform routing and $k$ different experts per token. Then compare it with a simulation.

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
# Solve the closed form for $T$. Set `t_mixtral` to the smallest batch at which Mixtral touches at least 7.5 of 8
# experts per layer. Set `t_qwen` to the smallest batch at which Qwen3-30B-A3B touches at least 120 of 128.

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
# Write `step_bytes(cfg, batch, context)` for bf16 weights and KV. Count these bytes:
# - the attention, router and shared experts of every layer
# - the touched routed experts (`T.experts_touched`)
# - the LM head
# - one embedding row per token
# - the KV read of each sequence and one new entry (`(context + 1) × cfg.kv_bytes_per_token()`)
#
# Use the `MoEConfig` methods (`attn_params`, `expert_params`, `router_params`, `moe_layer_params`,
# `dense_layer_params`, ...).

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
# At large batch, the streamed weights ≈ total, and the FLOPs ∝ active. Thus crossover(MoE) ≈ crossover(dense) ×
# (streamed ÷ multiplied weights). Predict the crossover of Qwen3. Use only the crossover of Llama-3.1-8B (207 on an
# H200) and the config of Qwen3. Do not include the input embedding in the two counts. It is a gather, and the step
# does not stream or multiply it.

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
# In the step, each touched expert runs a GEMM on the rows routed to it. On average, this is
# $\text{batch} \times k \div E$ rows. That row count is the arithmetic intensity of the expert GEMM (FLOP per weight
# byte at 2-byte weights). For batch 256, set `rows` to a dict of rows per expert for Mixtral (8, 2), Qwen3 (128, 8)
# and DeepSeek-V3 (256, 8). Set `batch_for_ridge` to the batch that each model needs. At that batch, the average
# expert sees 206 rows (the ridge of the H200).

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
# Hot experts are a property of the workload: a code-heavy batch favours a few experts. Make a model of this with Zipf
# popularity $s = 1.0$ (simulated) for Qwen3-30B-A3B at batch 16, 1K context. Set `skew_touched` from
# `T.touched_mc`. Then set `speedup` = uniform step time ÷ skewed step time
# (`T.decode_step(..., touched=skew_touched)`).

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
# **The two-minute version.** "Per token, an MoE computes like its active parameters. But a decode step streams every
# expert that *any* token in the batch selected. With uniform routing, that is $E(1 - (1 - k/E)^T)$ experts per
# layer. Thus Mixtral is 13B-like at batch 1 and streams all 93 GB by batch 16. Fine-grained models like
# Qwen3-30B-A3B keep the smaller stream slightly longer.
#
# "Throughput per step still increases with batch. But the batch at which decode becomes compute-bound scales with
# total ÷ active. On an H200, that batch is approximately 750 for Mixtral and 2,000 for Qwen3, against 200 for a
# dense 8B. The cause is that each expert sees only $B \cdot k/E$ rows. Thus MoE wants large batches. On one GPU,
# large batches come to the limit of KV memory, and across GPUs they mean expert parallelism.
#
# "The attention alone sets the KV cache. MoE does not change it, prefix caching operates the same, and at long
# context the KV share increases again."
#
# **Drill questions**
# 1. *Mixtral at batch 1 and at batch 64: how do the bytes per step compare?* 25.6 GB against 101.7 GB on the
#    roofline. At batch 1, the step reads 2 of 8 experts per layer. At batch 64, it reads all 8. Both also read the
#    KV.
# 2. *Why is 'decode is compute-bound from batch ≈ ridge' incorrect for MoE?* The weight stream approaches the total
#    parameters, but the FLOPs change with the active parameters. Thus the crossover is ridge × (streamed ÷
#    multiplied).
# 3. *Does MoE change prefix caching or the KV size?* No. Both depend on the attention. Only the share of KV in the
#    step changes.
