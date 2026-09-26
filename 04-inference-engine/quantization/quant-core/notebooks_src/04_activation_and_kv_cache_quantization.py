# %% [markdown]
# # 04 · Activation and KV-cache quantization
#
# **Tier:** T0 — numpy, a few seconds. Serving an FP8 KV cache in vLLM (`--kv-cache-dtype fp8`) is the lab's
# notebook 04 (T1, Ada or newer); the sizing here is the same arithmetic, SIMULATED where it prints times.
#
# ## The one-minute version
# Quantizing **activations** is what lets a GEMM run on INT8 or FP8 tensor cores: both operands become 8-bit
# codes, the products accumulate in a wide register (INT32 or FP32), and the two scales are applied once per
# output in the **epilogue** — `y = s_x[token] · s_w[channel] · Σ qx·qw`. That factoring is why activation
# scales can be per token and weight scales per output channel but neither can vary along the reduction axis,
# and why block formats (DeepSeek-V3's 128×128 FP8) rescale one partial sum per 128-wide block. Some layers
# stay in 16-bit — the LM head, embeddings, norms, the attention softmax — because their errors are expensive
# or they save nothing. The **KV cache** is an activation stored for later: quantized once, read at every
# decode step. FP8 halves its bytes (twice the sessions) with 3 mantissa bits of error, *if* its scale fits the
# data — vLLM's default scale is 1.0. Keys have outlier channels and values do not, so 2–4-bit schemes (KIVI)
# quantize keys per channel and values per token. After this notebook you can implement a W8A8 GEMM epilogue,
# say which layers to leave alone, and size and judge a quantized KV cache.
#
# Primer: `../PRIMER.md` §5 *Weight-and-activation quantization* and §6 *KV-cache quantization*; the kernel-side
# FP8 error sources are the FlashAttention deep dive §9.4, the backend conditions vllm-internals §6.3.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

import numpy as np

from quantcore import TinyModel, cost, eval as E, granularity as G, kvquant as K, quantize_model, w8a8

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — a W8A8 INT8 GEMM, exactly
# Quantize X per token and W per output channel, multiply the integer codes with integer accumulation, then
# apply `s_x · s_w`. The result equals the fake-quantized product to rounding — the scales really do factor out.

# %%
Xa, Wa = rng.standard_normal((16, 4096)), rng.standard_normal((1024, 4096)) * 0.02
Y, _ = w8a8.w8a8_matmul(Xa, Wa, "int8")
fake = G.quantize_activations(Xa, "int8") @ G.fake_quant(Wa, fmt="int8").T
print(f"integer-accumulate GEMM vs fake quant: max difference {np.abs(Y - fake).max():.1e}")
print(f"relative error vs the fp64 product: {np.linalg.norm(Y - Xa @ Wa.T) / np.linalg.norm(Xa @ Wa.T):.4f}")
print(f"worst-case |accumulator| for K = 4,096: 127 x 127 x 4096 = {127 * 127 * 4096:,} < 2^31 = {2 ** 31:,}")

# %% [markdown]
# ## Worked example 2 — static vs dynamic activation scales on a model
# Every hidden linear in W8A8 (weights per channel), activations either per token at run time or with one
# static scale per layer from 256 calibration samples.

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
# Dynamic per-token scales win for both formats — 4× less KL for INT8, a little for FP8, whose float grid
# already absorbs most of the range — at the price of one max-reduction per token, which kernels fuse into the quantization step. This is the default for `FP8_DYNAMIC` and INT8 `W8A8`
# recipes; static activation scales (the `FP8` preset) save that reduction and need calibration data.
#
# ## Worked example 3 — block-scaled FP8, as DeepSeek-V3 stores it
# Weights: one scale per 128×128 tile. Activations: one scale per token per 128 channels, computed on the fly.
# The GEMM loops over 128-wide k blocks and rescales each partial sum by its own two scales before adding it.

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
# For FP8 the granularity hardly matters at these ranges (a 30× tile is far inside E4M3's 2^14.8), and per-token
# INT8 is more precise than any FP8 (7 bits against 3). Block scales earn their place when ranges are extreme — in
# training, and for models whose weights and activations span more than FP8's range — and because a kernel that
# already tiles by 128 can apply a per-tile scale for free (DeepGEMM, CUTLASS block-scaled GEMMs on SM90+; there
# is no CUTLASS block-FP8 kernel for SM89, facts sheet §2).
#
# ## Worked example 4 — which layers stay in high precision
# Recipes quantize the transformer blocks' linears and `ignore=["lm_head"]`. Quantize the tiny model's head too:

# %%
for bits in (8, 4):
    r = E.compare(ref, quantize_model(m, "rtn", bits, None, targets=["head"]).forward(X), y)
    print(f"only the head, INT{bits} per channel: KL {r['kl']:.4f}, acc {r['acc']:.1%} (±{r['stderr']:.1%}), "
          f"{r['lost']} right answers lost")
r = E.compare(ref, quantize_model(m, "rtn", 4, None).forward(X), y)
print(f"all four hidden linears, INT4 per channel: KL {r['kl']:.4f}, acc {r['acc']:.1%}")

# %% [markdown]
# One INT4 layer at the output costs 3.7 points, against 5.8 for all four hidden linears together: the head is
# the most sensitive single layer. Its errors land directly on the logits, with no later layer to average them;
# in an LLM it is also one large GEMM per sampled token (the vocabulary is 128K–152K rows), which is why
# keeping it 16-bit costs bytes at decode (serving-engine §8: 1.05 GB of Llama-3.1-8B's 5.70 GB INT4 weights).
# The input **embedding** is a gather, not a GEMM: quantizing it saves memory but no compute. **Norms** and the
# **attention softmax** are tiny, and their range is exactly what low precision handles worst. MoE **routers**
# (`mlp.gate`) are ignored too: a flipped top-k choice is a discrete error.
#
# ## Worked example 5 — an FP8 KV cache and its scale
# One head's decode attention over 512 cached tokens: keys with four outlier channels (magnitude ~12, as KIVI
# reports for real keys), values with a shared mean. Error = relative error of the attention output.

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
# Well-scaled FP8 KV costs well under 1% of attention-output error here, and key errors matter more than value
# errors: they pass through the softmax's exponential, value errors are averaged. The scale matters only at the
# edges: with vLLM's uncalibrated default of 1.0, values far below 1 fall into E4M3's subnormals (below 2⁻⁶)
# and values above 448 saturate. That is what llm-compressor's `kv_cache_scheme` calibration writes into the
# checkpoint as `k_scale`/`v_scale` — and why the FlashAttention deep dive's §9.4 lists "scale choice" as an
# error source.
#
# ## Worked example 6 — below 8 bits: KIVI
# 2–4-bit KV needs finer scales. Keys: one scale and minimum per **channel** per group of 32 tokens (the outlier
# channels get their own). Values: per **token** per group of 32 channels. The newest tokens stay 16-bit until a
# group fills (the residual window).

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
# FP8 KV doubles the sessions; 4-bit KIVI gives 3.2×, 2-bit 5.3× — at 1.2% and 7% attention-output error on this
# head. vLLM's sub-8-bit KV dtypes at this snapshot are per-token-head (`int4_per_token_head`, dynamic scales),
# not KIVI (facts sheet §2, verify): the per-channel key trick is the reason to check which one a system implements.
#
# **Prefix caching with a quantized cache.** A cached block is reused as stored: its tokens are hashed as usual
# (serving-engine §5) and the block holds FP8 values. That works because the scales are static per layer (or
# stored with the block, for per-token scales): a later request reading the block dequantizes it exactly as the
# request that wrote it did. A scheme whose scales depended on the *reading* request would break sharing.
#
# ## Exercise 4.1 — the epilogue
# Write `int8_w8a8(X, W)`: per-token activation scales `s_x = max|x_t| / 127`, per-output-channel weight scales
# `s_w = max|w_j| / 127`, codes `round(x / s)` clipped to ±127, an `int64` matrix product of the codes, then the
# epilogue. Return the float result.

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
# In the worst case every product is 127 × 127 (or 128 × 128 with the full −128…127 range) with the same sign.
# Compute `k_max_restricted` and `k_max_full`: the largest K for which the accumulator stays below 2³¹.

# %% exercise
### BEGIN SOLUTION
k_max_restricted = (2 ** 31 - 1) // (127 * 127)
k_max_full = (2 ** 31 - 1) // (128 * 128)
### END SOLUTION

# %% check
assert k_max_restricted == 133144 and k_max_full == 131071
print(f"✅ {k_max_restricted:,} and {k_max_full:,}: far above any hidden size (8,192-28,672), so INT32 accumulation "
      "never overflows in practice; FP8 accumulates in FP32 for the same reason")

# %% [markdown]
# ## Exercise 4.3 — KIVI's keys: per channel, grouped over tokens
# Write `keys_per_channel(Kmat, bits, group)` for `Kmat[tokens, channels]`: for each channel and each run of
# `group` consecutive tokens, asymmetric min-max quantization (`scale = (max − min) / (2^bits − 1)` after
# widening to include 0, `zero = round(−min / scale)`, codes clipped to 0…2^bits − 1). Return the dequantized keys.

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
# For Llama-3.1-8B (32 layers, 8 KV heads, head_dim 128), fill `bytes_per_token` for bf16, FP8, 4-bit KIVI (5 bits
# per element) and 2-bit KIVI (3 bits), and `sessions_fp8_weights`: 2,000-token sessions on an L4 with FP8 weights
# for each, using `cost.sessions`.

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
# A model's values are small (`Vs = V * 2e-3`). With vLLM's default scale 1.0 they fall into FP8's subnormals.
# Compute the per-tensor `v_scale` llm-compressor would write (amax / 448) and the attention-output error with it.

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
# **The two-minute version.** "W8A8 means the tensor cores multiply 8-bit codes and the two scales are applied
# once per output in the epilogue, so we can use per-token activation scales and per-channel weight scales for
# free; we choose dynamic activation scales because they never saturate on inputs calibration did not see. We
# leave the LM head, embeddings, norms, the attention softmax and MoE routers in 16-bit: the head's errors land on
# the logits and a router's are discrete. The KV cache is separate: FP8 halves it — twice the sessions — and costs
# under 1% of attention-output error when its scales fit the data; vLLM's default scale is 1.0, so we calibrate
# k_scale and v_scale for any model whose K or V are far from order one. Below 8 bits we would want keys quantized
# per channel, KIVI-style, because keys carry outlier channels; and we would check that prefix-cached blocks keep
# their scales."
#
# **Drills**
# 1. *Why can activation scales be per token but not per input channel in a W8A8 GEMM?* — A per-input-channel
#    scale varies along the reduction axis, so it cannot be factored out of Σ qx·qw into the epilogue (SmoothQuant
#    moves that variation into the weights instead).
# 2. *FP8 KV on a model whose values have magnitude 1e-3, no calibration: what happens?* — With the default scale
#    1.0 most values are E4M3 subnormals or zero; attention output error jumps from ~0.3% to ~5% (worked example 5).
# 3. *Keys per channel, values per token — why the asymmetry?* — Keys have fixed outlier channels, so a per-token
#    scale is set by them; values have no such channels and are consumed per token (a weighted sum over tokens).
