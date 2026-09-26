# %% [markdown]
# # 06 · Quantization
#
# **Tier:** T0 — CPU only, no network, a few seconds. Everything is emulated in float64 ("fake quantization") so
# errors are visible; speeds on real GPUs are **SIMULATED** with the roofline model. Real quantized checkpoints
# (AWQ/GPTQ INT4, FP8) served by vLLM are `vllm-serving-lab` notebook `05_speculation_and_quantization_in_vllm` (T1).
#
# ## The one-minute version
# Quantization stores a number in fewer bits plus a **scale** that says what the bits mean: `w ≈ code × scale`.
# The design choice is **granularity** — how many weights share one scale. Per tensor is cheapest, but one outlier
# stretches the scale and wrecks everyone else's resolution; per output channel isolates outlier rows; groups of
# 32–128 inputs (the usual INT4 recipe, GPTQ/AWQ) isolate them further for a few extra bits per group. What it buys
# depends on the bottleneck: **decode is memory-bound**, so fewer weight bytes are faster tokens (weight-only INT4
# ≈ 3.5× on one request); **prefill is compute-bound**, so only formats the tensor cores compute in natively (FP8
# W8A8 on Ada/Hopper/Blackwell, INT8 W8A8) make it faster. Quantizing the **KV cache** (FP8) halves its bytes, which
# doubles how many sessions fit. And it always costs some accuracy — measure it on your own evals.
#
# Primer: §8 *Quantization* (`../../PRIMER.md`); memory sizing: `00-foundations/gpu-capacity-planning/PRIMER.md`.

# %%
from dataclasses import replace

import numpy as np

from minengine import Batch, Engine, PagedKVCache, SamplingParams, TinyLM, encode, perf, quant
from minengine.model import CORPUS

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — formats and granularity on one weight matrix

# %%
w = rng.standard_normal((256, 128)) * 0.02          # a (d_in, d_out) weight, used as x @ w
schemes = [("int8", "tensor", None, 8), ("int8", "channel", None, 8), ("fp8", "tensor", None, 8),
           ("int4", "channel", None, 4), ("int4", "group", 128, 4), ("int4", "group", 32, 4)]
print(f"{'scheme':18} {'bits/weight':>11} {'rel error':>10} {'SQNR dB':>8}")
for fmt, gran, g, bits in schemes:
    e = quant.error(w, quant.fake_quant(w, fmt=fmt, granularity=gran, **({"group_size": g} if g else {})))
    bpw = quant.bits_per_weight(bits, g if g else None)
    print(f"{fmt + ' ' + gran + (str(g) if g else ''):18} {bpw:11.3f} {e['rel']:10.4f} {e['sqnr_db']:8.1f}")

# %% [markdown]
# Each bit is worth about 6 dB of signal-to-noise; INT4 loses ~24 dB against INT8. Groups win back a little at a
# small storage cost (a 16-bit scale per 128 weights adds 0.125 bits). On well-behaved weights INT8 per channel is
# actually more precise than FP8 (43 vs 32 dB): FP8-E4M3 has only 3 mantissa bits. FP8's strength is *range* — a
# floating grid keeps the same relative precision from 2⁻⁶ to 448 — which matters for activations, whose scale
# varies token to token, and it is what the tensor cores compute in.
#
# ## Worked example 2 — the FP8-E4M3 grid

# %%
print("values in [1, 2):  ", quant.fp8_e4m3(np.linspace(1, 1.99, 200)).round(4).tolist()[::25])
print("values in [16, 32):", sorted(set(quant.fp8_e4m3(np.linspace(16, 30.9, 400)).tolist())))
print("largest finite:", quant.fp8_e4m3(1e6), " smallest subnormal:", quant.fp8_e4m3(2 ** -9), "=", 2 ** -9)
x = rng.standard_normal(10000)
rel = np.abs(quant.fp8_e4m3(x) - x) / np.abs(x)
print(f"relative rounding error on N(0,1) values: median {np.median(rel):.3%}, max {rel[np.abs(x) > 2 ** -6].max():.3%}")

# %% [markdown]
# Eight values per power of two, so the relative step is 1/8 and the worst rounding error is 1/16 = 6.25%,
# whatever the magnitude — until values fall below 2⁻⁶ (subnormals) or above 448 (saturation). That is why FP8 is
# used with a per-tensor or per-channel scale that maps the data's range onto the grid.
#
# ## Worked example 3 — one outlier channel

# %%
wo = rng.standard_normal((64, 32)) * 0.02
wo[:, 3] *= 100                                      # one output channel 100x larger
rest = [c for c in range(32) if c != 3]
for gran in ["tensor", "channel"]:
    wq = quant.fake_quant(wo, fmt="int8", granularity=gran)
    print(f"int8 per {gran:7}: whole-matrix error {quant.error(wo, wq)['rel']:.3f}, "
          f"error on the 31 normal channels {quant.error(wo[:, rest], wq[:, rest])['rel']:.3f}")

# %% [markdown]
# The aggregate error of per-tensor INT8 looks fine (3%), because the outlier channel dominates the norm; the 31
# ordinary channels are almost erased (their values round to a handful of levels). **Aggregate metrics hide
# per-channel damage** — the reason per-channel scales are the default for INT8 weights.
#
# ## Worked example 4 — activations have outliers too: SmoothQuant for W8A8
# W8A8 quantizes activations as well, per token, at run time. LLM activations have a few channels that are
# consistently huge. SmoothQuant divides those channels of X by a factor `s` and multiplies the matching rows of W
# by it: `X W = (X / s)(s W)`, moving the difficulty into the weights, which tolerate it.

# %%
X, W = rng.standard_normal((32, 64)), rng.standard_normal((64, 16)) * 0.05
X[:, 5] *= 60                                       # an outlier activation channel
s = quant.smoothquant_scales(np.abs(X).max(0), np.abs(W).max(1), alpha=0.5)
naive = quant.error(X @ W, quant.quantize_activations(X) @ quant.fake_quant(W))["rel"]
smooth = quant.error(X @ W, quant.quantize_activations(X / s) @ quant.fake_quant(W * s[:, None]))["rel"]
print(f"W8A8 output error: naive {naive:.4f}, smoothed {smooth:.4f} ({naive / smooth:.1f}x smaller)")

# %% [markdown]
# ## Worked example 5 — what it does to a model
# The same prompt text through the original model and quantized copies (linear layers quantized; embeddings kept,
# as is usual). KL divergence and top-1 agreement over 400 positions, and how many greedy tokens match.

# %%
model = TinyLM()
ids = encode(CORPUS[:400])
ref = model.forward_dense(ids)
greedy_ref = model.generate_dense(encode("The engine "), 30)
for label, kw in [("int8 per-channel", dict(fmt="int8", granularity="channel")),
                  ("fp8 per-tensor", dict(fmt="fp8", granularity="tensor")),
                  ("int4 group 32", dict(fmt="int4", granularity="group", group_size=32)),
                  ("int4 per-channel", dict(fmt="int4", granularity="channel"))]:
    qm = model.quantized(**kw)
    c = quant.compare_logits(ref, qm.forward_dense(ids))
    g = qm.generate_dense(encode("The engine "), 30)
    same = next((i for i, (a, b) in enumerate(zip(greedy_ref, g)) if a != b), 30)
    print(f"{label:17} KL {c['kl']:.5f} nats  top-1 {c['top1']:.1%}  greedy text identical for {same} tokens")

# %% [markdown]
# Small per-token damage, yet greedy text diverges within a few tokens: one flipped near-tie changes everything
# after it. Exact-match on generated text is a brittle metric; distribution distance (KL, top-1) and **task-level
# evals** on your own data are the ones to trust.
#
# ## Worked example 6 — an FP8 KV cache
# Same tokens, fed one at a time (like decode), with the cache stored in FP8 vs full precision.

# %%
def teacher_forced(kv_dtype, ids, B=4):
    cache = PagedKVCache(model.cfg, num_blocks=len(ids) // B + 1, block_size=B, kv_dtype=kv_dtype)
    table = list(range(len(ids) // B + 1))
    rows = [model.forward(Batch(np.array([t]), np.array([i]), np.array([table[i // B] * B + i % B]),
                                [0, 1], [table], [i + 1]), cache)[0] for i, t in enumerate(ids)]
    return np.array(rows)


c = quant.compare_logits(teacher_forced("float64", ids[:200]), teacher_forced("fp8", ids[:200]))
print(f"FP8 KV cache: KL {c['kl']:.5f} nats, top-1 {c['top1']:.1%}, and half the bytes of a bf16 cache")

# %% [markdown]
# ## Worked example 7 — what each scheme buys on an L4 (SIMULATED)
# Llama-3.1-8B on a 24 GB L4; decode at 1,000 tokens of context; an 1,800-token prefill; sessions of 1,800 + 200
# tokens. "Weight-only" formats are dequantized to bf16 inside the GEMM, so they save bytes but not FLOPs.

# %%
G, base = perf.GPUS["L4"], perf.LLMS["llama-3.1-8b"]
INT4 = 4.125 / 8                                     # bytes per param, INT4 with a 16-bit scale per 128
options = {"bf16": base,
           "int8 weight-only": replace(base, bytes_per_param=1.0),
           "int4 weight-only (g128)": replace(base, bytes_per_param=INT4),
           "fp8 W8A8": replace(base, bytes_per_param=1.0, compute_scale=2.0),
           "fp8 W8A8 + fp8 KV": replace(base, bytes_per_param=1.0, compute_scale=2.0, kv_bytes_per_value=1.0),
           "int4 weight-only + fp8 KV": replace(base, bytes_per_param=INT4, kv_bytes_per_value=1.0)}
per_session = -(-(1800 + 200 - 1) // 16)
print(f"{'':27}{'weights':>8} {'decode b=1':>11} {'decode b=32':>12} {'prefill 1.8k':>13} {'sessions':>9}")
for name, L in options.items():
    print(f"{name:27}{L.weight_bytes / 1e9:6.1f}GB {1e3 * perf.step_time(G, L, [(1000, 1)]):9.1f}ms "
          f"{1e3 * perf.step_time(G, L, [(1000, 1)] * 32):10.1f}ms {1e3 * perf.step_time(G, L, [(0, 1800)]):11.0f}ms "
          f"{perf.kv_cache_blocks(G, L) // per_session:9d}")
print("(SIMULATED: roofline model, 80% bandwidth / 60% FLOPs, 2 ms per step overhead)")

# %% [markdown]
# Read the columns: weight-only INT4 makes single-stream decode ~3.5× faster and frees memory for KV, but its
# prefill is no faster. FP8 W8A8 halves prefill time (FP8 tensor cores) and halves weight bytes. FP8 KV doubles the
# sessions that fit. On a small GPU, quantization is a **concurrency** lever before it is a speed lever.
#
# ## Exercise 6.1 — symmetric per-channel INT8
# For `w` of shape `(d_in, d_out)`, one scale per output column: `scale = max|w[:, j]| / 127`,
# `codes = round(w / scale)` clipped to ±127. Return `(codes, scale)` with `scale` of shape `(d_out,)`.

# %% exercise
def int8_per_channel(w):
    ### BEGIN SOLUTION
    scale = np.maximum(np.abs(w).max(axis=0), 1e-12) / 127
    codes = np.clip(np.round(w / scale), -127, 127)
    return codes, scale
    ### END SOLUTION

# %% check
codes, scale = int8_per_channel(w)
assert np.abs(codes).max() <= 127 and scale.shape == (128,) and np.all(codes == np.round(codes))
assert np.all(np.abs(codes * scale - w) <= scale / 2 + 1e-15)          # at most half a step
assert np.allclose(codes * scale, quant.fake_quant(w, fmt="int8", granularity="channel"))
print(f"✅ int8 per-channel: max error {np.abs(codes * scale - w).max():.2e} <= half a step")

# %% [markdown]
# ## Exercise 6.2 — group-wise INT4
# Now one scale per group of `g` consecutive **input** rows of each output column: reshape to `(d_in // g, g, d_out)`,
# `scale = max|group| / 7`, `codes = round(w / scale)` clipped to ±7. Return the dequantized weight `w_hat`.

# %% exercise
def int4_groupwise(w, g):
    ### BEGIN SOLUTION
    d_in, d_out = w.shape
    wg = w.reshape(d_in // g, g, d_out)
    scale = np.maximum(np.abs(wg).max(axis=1, keepdims=True), 1e-12) / 7
    return (np.clip(np.round(wg / scale), -7, 7) * scale).reshape(d_in, d_out)
    ### END SOLUTION

# %% check
for g in [32, 128]:
    assert np.allclose(int4_groupwise(w, g), quant.fake_quant(w, fmt="int4", granularity="group", group_size=g))
print(f"✅ int4 g32 error {quant.error(w, int4_groupwise(w, 32))['rel']:.3f} < g128 "
      f"{quant.error(w, int4_groupwise(w, 128))['rel']:.3f}: smaller groups, finer scales")

# %% [markdown]
# ## Exercise 6.3 — what the weights weigh
# Write `weights_gb(params, bits, group_size=None, scale_bits=16)` including the scales, then decide which of these
# fit on **one 80 GB GPU** with at least 20% of it left for the KV cache: Llama-3.1-8B (8.03B), a 70B model, both
# in bf16 and INT4 g128. Fill `fits`.

# %% exercise
def weights_gb(params, bits, group_size=None, scale_bits=16):
    ### BEGIN SOLUTION
    return params * (bits + (scale_bits / group_size if group_size else 0)) / 8 / 1e9
    ### END SOLUTION


fits = {("8B", "bf16"): None, ("8B", "int4"): None, ("70B", "bf16"): None, ("70B", "int4"): None}
### BEGIN SOLUTION
for size, params in [("8B", 8.03e9), ("70B", 70.6e9)]:
    fits[(size, "bf16")] = weights_gb(params, 16) <= 0.8 * 80
    fits[(size, "int4")] = weights_gb(params, 4, 128) <= 0.8 * 80
### END SOLUTION

# %% check
assert abs(weights_gb(8e9, 4, 128) - 4.125) < 1e-9 and abs(weights_gb(8e9, 16) - 16) < 1e-9
assert fits == {("8B", "bf16"): True, ("8B", "int4"): True, ("70B", "bf16"): False, ("70B", "int4"): True}
print(f"✅ 70B: {weights_gb(70.6e9, 16):.0f} GB in bf16 (two GPUs, tensor parallel) vs "
      f"{weights_gb(70.6e9, 4, 128):.1f} GB in INT4 g128 (one GPU)")

# %% [markdown]
# ## Exercise 6.4 — predict the decode speed-up
# Single-stream decode on an L4 is the weight read. Predict the INT4-g128 vs bf16 speed-up from bytes alone
# (`bytes_ratio`), then compute what the roofline model says (`modelled`, decode at batch 1, context 1,000).
# Why are they different?

# %% exercise
### BEGIN SOLUTION
bytes_ratio = 2.0 / INT4
modelled = perf.step_time(G, base, [(1000, 1)]) / perf.step_time(G, options["int4 weight-only (g128)"], [(1000, 1)])
### END SOLUTION

# %% check
assert abs(bytes_ratio - 3.88) < 0.01 and 3.3 < modelled < 3.7
print(f"✅ bytes say {bytes_ratio:.2f}x, the model says {modelled:.2f}x - the KV read and the fixed per-step "
      "overhead do not shrink (SIMULATED)")

# %% [markdown]
# ## Exercise 6.5 — pick a scheme
# An L4 serving Llama-3.1-8B must hold **at least 48 concurrent sessions** of 1,800 + 200 tokens **and** prefill an
# 1,800-token prompt in **at most 300 ms**. From `options`, set `choice` to the name that meets both.

# %% exercise
### BEGIN SOLUTION
ok = [name for name, L in options.items()
      if perf.kv_cache_blocks(G, L) // per_session >= 48 and perf.step_time(G, L, [(0, 1800)]) <= 0.300]
choice = ok[0]
### END SOLUTION

# %% check
assert choice == "fp8 W8A8 + fp8 KV"
print("✅", choice, "- INT4 wins memory and decode but not prefill; FP8 W8A8 needs FP8 tensor cores (Ada/Hopper+),"
      " and the FP8 KV cache supplies the sessions")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Quantization trades bits for accuracy, and what it buys depends on the bottleneck.
# Decode is memory-bound: weight-only INT4 with group-wise scales cuts the bytes per token by almost 4× and single-
# stream latency by ~3.5×, but prefill is compute-bound, so only W8A8 formats that run natively on the tensor
# cores — FP8 on Ada, Hopper and Blackwell — make prefill faster. Separately, an FP8 KV cache halves the KV bytes
# and doubles the sessions per GPU, which on a 24 GB card matters more than speed. Granularity is where accuracy
# is won: per-channel for INT8, groups of 32–128 for INT4, and outlier handling like SmoothQuant or AWQ for
# activations. We quantize, then gate on task-level evals against the bf16 baseline, not on perplexity alone."
#
# **Drill questions**
# 1. *Why doesn't weight-only INT4 speed up prefill?* — Prefill is compute-bound, and weight-only kernels
#    dequantize to bf16 before the matmul: same FLOPs.
# 2. *What does INT4 with a 16-bit scale per 128 weights cost per weight?* — 4 + 16/128 = 4.125 bits.
# 3. *Per-tensor INT8 of a matrix with one outlier channel: why is the aggregate error misleading?* — The outlier
#    dominates the norm; the other channels lose almost all resolution while the total barely moves.
