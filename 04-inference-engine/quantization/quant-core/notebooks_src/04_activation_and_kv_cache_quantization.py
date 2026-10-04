# %% [markdown]
# # 04 · Activation and KV-cache quantization
#
# **Tier:** T0. It needs numpy and runs in a few seconds. The lab's notebook 04 (T1, Ada or newer) serves an FP8 KV
# cache in vLLM (`--kv-cache-dtype fp8`). The size calculation here is the same arithmetic. Where the notebook prints
# times, it labels them SIMULATED.
#
# ## The one-minute version
# When you quantize the **activations**, a GEMM can run on INT8 or FP8 tensor cores. Both operands become 8-bit
# codes. The products accumulate in a wide register (INT32 or FP32). The kernel applies the two scales once per
# output, in the **epilogue**: $y = s_x[\text{token}] \cdot s_w[\text{channel}] \cdot \sum q_x \cdot q_w$.
#
# Because the scales factor out in this way, activation scales can be per token and weight scales can be per output
# channel. But neither scale can change along the reduction axis. For the same reason, block formats (DeepSeek-V3's
# 128×128 FP8) rescale one partial sum per 128-wide block.
#
# Some layers stay in 16-bit: the LM head, embeddings, norms and the attention softmax. The reason is that their
# errors are high-cost, or that their quantization saves nothing.
#
# The **KV cache** is an activation that the engine stores for later. The engine quantizes it once and reads it at
# every decode step. FP8 halves its bytes (twice the sessions), with 3 mantissa bits of error. This is true *if* its
# scale fits the data. But vLLM's default scale is 1.0. Keys have outlier channels and values do not, thus 2–4-bit
# schemes (KIVI) quantize keys per channel and values per token.
#
# After this notebook, you can:
# - implement a W8A8 GEMM epilogue,
# - say which layers to leave alone,
# - calculate the size of a quantized KV cache, and judge it.
#
# Primer: `../PRIMER.md` §5 *Weight-and-activation quantization* and §6 *KV-cache quantization*. The FlashAttention
# deep dive §9.4 gives the FP8 error sources on the kernel side. vllm-internals §6.3 gives the backend conditions.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

import numpy as np

from quantcore import TinyModel, cost, eval as E, formats as F, granularity as G, kvquant as K, quantize_model, smoothquant as S, w8a8

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — a W8A8 INT8 GEMM, exactly
# Quantize $X$ per token and $W$ per output channel. Multiply the integer codes with integer accumulation. Then apply
# $s_x \cdot s_w$. The result is equal to the fake-quantized product, to within rounding. This shows that the scales
# do factor out.

# %%
Xa, Wa = rng.standard_normal((16, 4096)), rng.standard_normal((1024, 4096)) * 0.02
Y, _ = w8a8.w8a8_matmul(Xa, Wa, "int8")
fake = G.quantize_activations(Xa, "int8") @ G.fake_quant(Wa, fmt="int8").T
print(f"integer-accumulate GEMM vs fake quant: max difference {np.abs(Y - fake).max():.1e}")
print(f"relative error vs the fp64 product: {np.linalg.norm(Y - Xa @ Wa.T) / np.linalg.norm(Xa @ Wa.T):.4f}")
print(f"worst-case |accumulator| for K = 4,096: 127 x 127 x 4096 = {127 * 127 * 4096:,} < 2^31 = {2 ** 31:,}")

# %% [markdown]
# ## Worked example 2 — static vs dynamic activation scales on a model
# Every hidden linear is in W8A8, with weights per channel. The activation scales are either per token at run time,
# or one static scale per layer from 256 calibration samples.

# %%
m = TinyModel()
X, y = m.sample(4000, "test")
Xc, _ = m.sample(256, "calib")
ref = m.forward(X)
static = {k: np.abs(v).max() for k, v in m.calibration_inputs(Xc).items()}


def w8a8_model(model, fmt="int8", per="token", targets=None):
    targets = targets or model.linears()
    wq = {n: G.fake_quant(model.weights[n], fmt=fmt, granularity="channel") for n in targets}
    aq = lambda name, x: (G.quantize_activations(x, fmt, per, static[name] if per == "tensor" else None)
                          if name in targets else x)
    return model.forward(X, act_quant=aq, weights=wq)


for fmt in ("int8", "fp8"):
    for per in ("token", "tensor"):
        r = E.compare(ref, w8a8_model(m, fmt, per), y)
        print(f"{fmt.upper()} W8A8, {'dynamic per-token' if per == 'token' else 'static per-tensor'} activations: "
              f"KL {r['kl']:.4f}, top-1 {r['top1']:.1%}, acc {r['acc']:.1%}")

# %% [markdown]
# Dynamic per-token scales win for both formats. For INT8, they give 4× less KL. For FP8, the decrease is small,
# because the float grid of FP8 already absorbs most of the range. The cost is one max-reduction per token, and
# kernels fuse it into the quantization step. Dynamic per-token scales are the default for `FP8_DYNAMIC` and INT8
# `W8A8` recipes. Static activation scales (the `FP8` preset) save that reduction, and they need calibration data.
#
# ## Worked example 3 — block-scaled FP8, as DeepSeek-V3 stores it
# Weights have one scale per 128×128 tile. Activations have one scale per token per 128 channels, which the kernel
# calculates at run time. The GEMM does a loop over 128-wide $k$ blocks. It rescales each partial sum by its own two
# scales, before it adds that sum.

# %%
Xb, Wb = rng.standard_normal((16, 512)), rng.standard_normal((512, 512))
Wb[:128, :128] *= 30
Xb[:, :128] *= 20
refb = Xb @ Wb.T
for label, Yb in (("block FP8 (1x128 act, 128x128 weight)", w8a8.block_fp8_matmul(Xb, Wb)),
                  ("per-token / per-channel FP8", w8a8.w8a8_matmul(Xb, Wb, "fp8")[0]),
                  ("per-tensor FP8", w8a8.w8a8_matmul(Xb, Wb, "fp8", act="tensor", weight="tensor")[0]),
                  ("per-token / per-channel INT8", w8a8.w8a8_matmul(Xb, Wb, "int8")[0]),
                  ("per-tensor INT8", w8a8.w8a8_matmul(Xb, Wb, "int8", act="tensor", weight="tensor")[0])):
    print(f"{label:38} relative output error {np.linalg.norm(Yb - refb) / np.linalg.norm(refb):.4f}")

# %% [markdown]
# For FP8, the granularity has almost no effect at these ranges (a 30× tile is far inside E4M3's $2^{14.8}$).
# Per-token INT8 is more precise than any FP8 (7 bits against 3).
#
# Block scales are worth their cost in two cases.
# The first case is extreme ranges: in training, and in models whose weights and activations span more than the
# range of FP8. The second case is a kernel that already tiles by 128. That kernel can apply a per-tile scale at no
# cost (DeepGEMM, CUTLASS block-scaled GEMMs on SM90+). There is no CUTLASS block-FP8 kernel for SM89 in vLLM's
# `scaled_mm_entry.cu` (verify).
#
# ## Worked example 4 — FP4 W4A4: sixteen activations share one scale
# On Blackwell, NVFP4 W4A4 also quantizes activations. It uses E2M1 values with one E4M3 scale per **16** channels
# of each token. An outlier channel sets the scale of its block at $\mathrm{amax}/6$.
#
# Consider an outlier that is 30× the typical value. A typical neighbour then goes to 6/30 = 0.2 on the E2M1 grid.
# This is below 0.25, the value that rounds up to the smallest step of E2M1, 0.5. Thus most of the 15 neighbours
# become **zero**. The input of the first up-projection of the tiny model has its four outlier channels in three of
# its four 16-channel blocks:

# %%
Au = m.calibration_inputs(Xc)["blocks.0.up"]
Wu = m.weights["blocks.0.up"]
au = np.abs(Au).max(0)
hot = np.argsort(-au)[:4]
ordinary = np.setdiff1d(np.arange(64), hot)
nv = F.nvfp4(Au)[3]
print(f"outlier channels {sorted(hot.tolist())} -> blocks {sorted(set((hot // 16).tolist()))}")
free = np.array([c for c in range(64) if c // 16 not in set((hot // 16).tolist())])    # blocks with no outlier
print(f"NVFP4 activations: ordinary channels {G.error(Au[:, ordinary], nv[:, ordinary])['rel']:.1%} error, "
      f"{np.mean(nv[:, ordinary] == 0):.0%} of them flushed to zero; in the outlier-free block "
      f"{G.error(Au[:, free], nv[:, free])['rel']:.1%}")
s_up = S.smooth_scales(au, np.abs(Wu).max(0), 0.5)
nvs = F.nvfp4(Au / s_up)[3]
print(f"after SmoothQuant (alpha 0.5): ordinary channels {G.error(Au[:, ordinary] / s_up[ordinary], nvs[:, ordinary])['rel']:.1%}")
refu = Au @ Wu.T
w4a4 = lambda A_, W_: F.nvfp4(A_)[3] @ F.nvfp4(W_)[3].T
print(f"layer output error, NVFP4 W4A4: {np.linalg.norm(w4a4(Au, Wu) - refu) / np.linalg.norm(refu):.1%}; "
      f"smoothed first: {np.linalg.norm(w4a4(Au / s_up, Wu * s_up) - refu) / np.linalg.norm(refu):.1%}")


def nvfp4_model(model, acts=True):
    """Every hidden linear in NVFP4: weights always, activations too for W4A4; the head in 16-bit."""
    wq = {n: F.nvfp4(model.weights[n])[3] for n in model.linears()}
    aq = (lambda name, x: x if name == "head" else F.nvfp4(x)[3]) if acts else None
    return model.forward(X, act_quant=aq, weights=wq)


smooth = m
for name in ("blocks.0.up", "blocks.1.up"):
    A_ = smooth.calibration_inputs(Xc)[name]
    smooth = smooth.with_weights(smooth.fold(name, S.smooth_scales(np.abs(A_).max(0), np.abs(smooth.weights[name]).max(0), 0.5)))
for label, logits in (("NVFP4 weight-only (W4A16)", nvfp4_model(m, acts=False)), ("NVFP4 W4A4", nvfp4_model(m)),
                      ("NVFP4 W4A4 + SmoothQuant 0.5", nvfp4_model(smooth))):
    r = E.compare(ref, logits, y)
    print(f"{label:29} KL {r['kl']:.3f}  top-1 {r['top1']:.1%}  acc {r['acc']:.1%} (fp {r['acc_ref']:.1%})")

# %% [markdown]
# Per-16 blocks help the blocks with no outlier (10% error there). They cannot help the blocks that hold an outlier.
# About half of all ordinary activations become zero. The model loses 4 points more than with the 4-bit weights
# alone. If you move the range into the weights first (SmoothQuant), you get most of that loss back. The
# AWQ-style scale method works the same way.
#
# The mitigations in production are these:
# - that same move of the range into the weights,
# - Hadamard rotations (llm-compressor's SpinQuant and QuIP transforms spread an outlier over every channel of the
#   block),
# - quantization-aware distillation (primer §7).
#
# The lab's notebook 05 does this again on its bundled model. There, the gap is even larger.
#
# ## Worked example 5 — which layers stay in high precision
# Recipes quantize the linears of the transformer blocks, and set `ignore=["lm_head"]`. Quantize the head of the tiny
# model too:

# %%
for bits in (8, 4):
    r = E.compare(ref, quantize_model(m, "rtn", bits, None, targets=["head"]).forward(X), y)
    print(f"only the head, INT{bits} per channel: KL {r['kl']:.4f}, acc {r['acc']:.1%} (±{r['stderr']:.1%}), "
          f"{r['lost']} right answers lost")
r = E.compare(ref, quantize_model(m, "rtn", 4, None).forward(X), y)
print(f"all four hidden linears, INT4 per channel: KL {r['kl']:.4f}, acc {r['acc']:.1%}")

# %% [markdown]
# One INT4 layer at the output costs 4.0 points. All four hidden linears together cost 6.0 points. Thus the head is the
# most sensitive single layer. Its errors go directly to the logits, and no later layer averages them. In an LLM, the
# head is also one large GEMM per sampled token (the vocabulary is 128K–152K rows). That is why a 16-bit head costs
# bytes at decode (serving-engine §8: 1.05 GB of Llama-3.1-8B's 5.70 GB INT4 weights).
#
# The input **embedding** is a gather, not a GEMM. Its quantization saves memory but no compute. **Norms** and the
# **attention softmax** are small, and their range is exactly the range that low precision handles worst. Recipes
# also ignore MoE **routers** (`mlp.gate`), because a flipped top-k choice is a discrete error.
#
# ## Worked example 6 — an FP8 KV cache and its scale
# This is the decode attention of one head over 512 cached tokens. The keys have four outlier channels (magnitude
# ~12, as KIVI reports for real keys). The values have a shared mean. The error is the relative error of the
# attention output.

# %%
Q, Kc, V = K.synthetic_qkv()
for scale in (1.0, "tensor", "token"):
    print(f"FP8 E4M3 K and V, scale {scale!s:6}: attention output error {K.attention_error(Q, Kc, V, K.fp8_kv(Kc, scale), K.fp8_kv(V, scale)):.4f}")
print(f"  of which keys only {K.attention_error(Q, Kc, V, K.fp8_kv(Kc, 'tensor'), V):.4f}, values only "
      f"{K.attention_error(Q, Kc, V, Kc, K.fp8_kv(V, 'tensor')):.4f}")
for f in (1e-3, 1e-2, 1.0, 1e2, 1e3):
    Vf = V * f
    print(f"values x {f:g}: scale 1.0 -> {K.attention_error(Q, Kc, Vf, Kc, K.fp8_kv(Vf, 1.0)):.4f}, "
          f"calibrated v_scale -> {K.attention_error(Q, Kc, Vf, Kc, K.fp8_kv(Vf, 'tensor')):.4f}")

# %% [markdown]
# Here, a well-scaled FP8 KV cache costs much less than 1% of attention-output error. Key errors are more important
# than value errors, because key errors go through the exponential of the softmax. The attention averages the value
# errors. The scale is important only at the edges. With vLLM's uncalibrated default of 1.0, values far below 1 go
# into the subnormals of E4M3 (below 2⁻⁶). Values above 448 saturate.
#
# The `kv_cache_scheme` calibration of llm-compressor writes a scale that fits the data into the checkpoint, as
# `k_scale`/`v_scale`.
# Because the scale is important at the edges, the FlashAttention deep dive's §9.4 lists "scale choice" as an error
# source.
#
# ## Worked example 7 — below 8 bits: KIVI
# A 2–4-bit KV cache needs finer scales. Keys have one scale and one minimum per **channel** per group of 32 tokens
# (thus the outlier channels get their own scales). Values have them per **token** per group of 32 channels. The
# newest tokens stay 16-bit until a group is full (the residual window).

# %%
for bits in (4, 2):
    ch = K.attention_error(Q, Kc, V, *K.kivi(Kc, V, bits, 32, 32))
    tok = K.attention_error(Q, Kc, V, *K.kivi(Kc, V, bits, 32, 32, key_axis=1))
    print(f"INT{bits} KV, groups of 32: keys per channel {ch:.4f} vs keys per token {tok:.4f}; "
          f"{K.kv_bits_per_element(bits, 32, 16, 16):.0f} bits per element with a 16-bit scale and minimum")
m8 = cost.MODELS["llama-3.1-8b"]
L4 = cost.GPUS["L4"]
print("Llama-3.1-8B on an L4 with FP8 weights, sessions of 2,000 tokens (core's round memory inputs):")
for label, b in (("bf16 KV", 16), ("FP8 KV", 8), ("4-bit KIVI (5 bits/elem)", 5), ("2-bit KIVI (3 bits/elem)", 3)):
    print(f"  {label:26} {m8.kv_bytes_per_token(b):>9,.0f} B/token  {cost.sessions(L4, m8, 'w8a8-fp8', 2000, b):4d} sessions")

# %% [markdown]
# FP8 KV doubles the sessions. 4-bit KIVI gives 3.2×, and 2-bit KIVI gives 5.3×. On this head, they cost 1.2% and
# 7% attention-output error. At this snapshot, vLLM's sub-8-bit KV dtypes are per-token-head
# (`int4_per_token_head`, dynamic scales), not KIVI (vLLM's `CacheDType`, verify). The per-channel key method is the
# reason to find out which of the two a system implements.
#
# **Prefix caching with a quantized cache.** The engine uses a cached block again, in its stored form. The engine hashes
# the tokens of the block as usual (serving-engine §5), and the block holds FP8 values. This works because the scales are static
# per layer (or, for per-token scales, the engine stores them with the block). Thus a later request that reads the
# block dequantizes it exactly as the request that wrote it did. If the scales of a scheme depend on the request that
# *reads* the block, two requests cannot share the block.
#
# ## Exercise 4.1 — the epilogue
# Write `int8_w8a8(X, W)`. Use these steps:
# - per-token activation scales $s_x = \max|x_t|/127$,
# - per-output-channel weight scales $s_w = \max|w_j|/127$,
# - codes $\operatorname{round}(x/s)$, clipped to ±127,
# - an `int64` matrix product of the codes,
# - then the epilogue.
#
# Return the float result.

# %% exercise
def int8_w8a8(X, W):
    ### BEGIN SOLUTION
    sx = np.abs(X).max(1, keepdims=True) / 127
    sw = np.abs(W).max(1, keepdims=True) / 127
    qx = np.clip(np.round(X / sx), -127, 127).astype(np.int64)
    qw = np.clip(np.round(W / sw), -127, 127).astype(np.int64)
    return (qx @ qw.T) * sx * sw.T
    ### END SOLUTION

# %% check
assert np.allclose(int8_w8a8(Xa, Wa), w8a8.w8a8_matmul(Xa, Wa, "int8")[0], atol=1e-9)
print("✅ codes times codes, then s_x[t] * s_w[j]: the scales never enter the inner loop")

# %% [markdown]
# ## Exercise 4.2 — how long can the reduction be?
# In the worst case, every product is 127 × 127 (or 128 × 128 with the full −128…127 range), and all products have
# the same sign. Calculate `k_max_restricted` and `k_max_full`. Each one is the largest $K$ for which the accumulator
# stays below 2³¹.

# %% exercise
### BEGIN SOLUTION
k_max_restricted = (2 ** 31 - 1) // (127 * 127)
k_max_full = (2 ** 31 - 1) // (128 * 128)
### END SOLUTION

# %% check
assert k_max_restricted == 133144 and k_max_full == 131071
print(f"✅ {k_max_restricted:,} and {k_max_full:,}: far above any hidden size (8,192-28,672), so INT32 accumulation "
      "never overflows in practice; FP8 accumulates in FP32 registers (on Hopper the tensor core's own FP8 "
      "accumulation is narrower, so long-k kernels promote partial sums every 128 - primer §5)")

# %% [markdown]
# ## Exercise 4.3 — KIVI's keys: per channel, grouped over tokens
# Write `keys_per_channel(Kmat, bits, group)` for `Kmat[tokens, channels]`. For each channel and each group of
# `group` consecutive tokens, do asymmetric min-max quantization:
# - $\mathrm{scale} = (\mathrm{max} - \mathrm{min})/(2^{\mathrm{bits}} - 1)$, after you widen the range to include 0,
# - $\mathrm{zero} = \operatorname{round}(-\mathrm{min}/\mathrm{scale})$,
# - codes clipped to $0 \ldots 2^{\mathrm{bits}} - 1$.
#
# Return the dequantized keys.

# %% exercise
def keys_per_channel(Kmat, bits, group):
    ### BEGIN SOLUTION
    t, c = Kmat.shape
    G_ = Kmat.T.reshape(c, t // group, group)
    lo = np.minimum(G_.min(-1, keepdims=True), 0)
    hi = np.maximum(G_.max(-1, keepdims=True), 0)
    scale = np.maximum(hi - lo, 1e-12) / (2 ** bits - 1)
    zero = np.round(-lo / scale)
    q = np.clip(np.round(G_ / scale) + zero, 0, 2 ** bits - 1)
    return ((q - zero) * scale).reshape(c, t).T
    ### END SOLUTION

# %% check
mine = keys_per_channel(Kc[:480], 2, 32)
assert np.allclose(mine, K.quant_groups(Kc[:480], 0, 2, 32))
print(f"✅ per-channel keys: attention error {K.attention_error(Q, Kc[:480], V[:480], mine, V[:480]):.4f} with 2-bit keys "
      "and exact values")

# %% [markdown]
# ## Exercise 4.4 — size the cache
# The model is Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128). Fill `bytes_per_token` for bf16, FP8, 4-bit KIVI
# (5 bits per element) and 2-bit KIVI (3 bits). Then fill `sessions_fp8_weights` for each format, with
# `cost.sessions`. `sessions_fp8_weights` is the number of 2,000-token sessions on an L4 with FP8 weights.

# %% exercise
bytes_per_token, sessions_fp8_weights = {}, {}
### BEGIN SOLUTION
for label, b in (("bf16", 16), ("fp8", 8), ("kivi4", 5), ("kivi2", 3)):
    bytes_per_token[label] = 2 * 32 * 8 * 128 * b / 8
    sessions_fp8_weights[label] = cost.sessions(L4, m8, "w8a8-fp8", 2000, b)
### END SOLUTION

# %% check
assert bytes_per_token == {"bf16": 131072, "fp8": 65536, "kivi4": 40960, "kivi2": 24576}
assert sessions_fp8_weights == {"bf16": 43, "fp8": 87, "kivi4": 140, "kivi2": 234}
print("✅ 43 -> 87 -> 140 -> 234 sessions: the KV format, not the weight format, sets concurrency once the weights are small")

# %% [markdown]
# ## Exercise 4.5 — calibrate a v_scale
# The values of a model are small (`Vs = V * 2e-3`). With vLLM's default scale 1.0, they go into the subnormals of
# FP8. Calculate the per-tensor `v_scale` that llm-compressor writes ($\mathrm{amax}/448$). Then calculate the
# attention-output error with it.

# %% exercise
Vs = V * 2e-3
### BEGIN SOLUTION
v_scale = np.abs(Vs).max() / 448
err_calibrated = K.attention_error(Q, Kc, Vs, Kc, K.fp8_kv(Vs, v_scale))
### END SOLUTION

# %% check
err_default = K.attention_error(Q, Kc, Vs, Kc, K.fp8_kv(Vs, 1.0))
assert np.isclose(v_scale, np.abs(Vs).max() / 448) and err_calibrated < 0.005 and err_default > 5 * err_calibrated
print(f"✅ v_scale {v_scale:.2e}: error {err_calibrated:.4f} vs {err_default:.4f} with the default 1.0")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "In W8A8, the tensor cores multiply 8-bit codes, and the kernel applies the two scales
# once per output in the epilogue. Thus we can use per-token activation scales and per-channel weight scales at no
# cost. We select dynamic activation scales because they never saturate on inputs that calibration did not see. We
# leave the LM head, embeddings, norms, the attention softmax and MoE routers in 16-bit. The errors of the head go to
# the logits, and the errors of a router are discrete.
#
# "The KV cache is a separate decision. FP8 halves it, which gives twice the sessions. It costs less than 1% of
# attention-output error when its scales fit the data. Because vLLM's default scale is 1.0, we calibrate k_scale and
# v_scale for any model whose K or V are far from order one. If we go below 8 bits, we want to quantize keys per
# channel, KIVI-style, because keys carry outlier channels. We also want to make sure that prefix-cached blocks keep
# their scales."
#
# **Drills**
# 1. *Why can activation scales be per token but not per input channel in a W8A8 GEMM?* A per-input-channel scale
#    changes along the reduction axis. Thus you cannot factor it out of $\sum q_x \cdot q_w$ into the epilogue.
#    SmoothQuant moves that variation into the weights instead.
# 2. *FP8 KV on a model whose values have magnitude 1e-3, no calibration: what occurs?* With the default scale 1.0,
#    most values are E4M3 subnormals or zero. The attention output error increases from ~0.3% to ~5% (worked
#    example 6).
# 3. *Keys per channel, values per token: why the asymmetry?* Keys have outlier channels that stay the same, thus
#    these channels set a per-token scale. Values have no such channels, and the attention uses the values per token (a
#    weighted sum over tokens).
