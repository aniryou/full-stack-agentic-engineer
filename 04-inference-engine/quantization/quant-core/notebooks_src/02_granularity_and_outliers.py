# %% [markdown]
# # 02 · Granularity and outliers: who shares a scale
#
# **Tier:** T0 — numpy only, a few seconds.
#
# ## The one-minute version
# A scale is set by the largest magnitude among the values that share it, so an outlier coarsens the grid for
# its whole group. That makes **granularity** — how many values share one scale — the main accuracy lever you
# control. Per tensor, one value hurts everything. Per **output channel** (a row of `W`), an outlier *row* is
# isolated, but an outlier *input column* is in every row, so per-channel scales cannot isolate it; **groups**
# of 32–128 along the input dimension confine it to one group per row, for 16/g extra bits. Activations are
# quantized at run time: **dynamic per-token** scales follow each token, **static per-tensor** scales come
# from calibration and saturate anything larger than calibration saw. LLM activations have a few channels
# 20–40× larger than the rest in every token, so a per-token INT8 scale is set by them and the other
# channels keep only a few levels — the problem SmoothQuant (notebook 03) and FP8's float grid address. After
# this notebook you can predict which granularity survives which outlier, and say why aggregate error
# metrics hide the damage.
#
# Primer: `../PRIMER.md` §2 (outliers) and §3 *Granularity and the bits-per-weight budget*. The layout here is
# `W[out, in]` (torch and checkpoints); `minengine` stores `w[d_in, d_out]`, so `W = w.T`.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

import numpy as np

from quantcore import TinyModel, formats as F, granularity as G, w8a8

# %% [markdown]
# ## Worked example 1 — the survey table, reproduced
# serving-engine PRIMER §8 quotes six schemes on one 256×128 Gaussian weight from `minengine.quant`. The same
# seed through quantcore's own code gives the same numbers (pinned in `tests/test_repo_numbers.py`) — the
# starting point this notebook goes beyond.

# %%
w = np.random.default_rng(0).standard_normal((256, 128)) * 0.02      # minengine's (d_in, d_out) weight
W = w.T                                                               # quantcore's (out, in)
for fmt, gran, g, bits in [("int8", "tensor", None, 8), ("int8", "channel", None, 8), ("fp8", "tensor", None, 8),
                           ("int4", "channel", None, 4), ("int4", "group", 128, 4), ("int4", "group", 32, 4)]:
    q = G.quantize(W, fmt=fmt, granularity=gran, group_size=g or 128)
    e = G.error(W, q.w_hat)
    print(f"{fmt} {gran}{g or '':<4} {F.bits_per_weight(bits, g):6.3f} bits  rel {e['rel']:.4f}  SQNR {e['sqnr_db']:4.1f} dB"
          f"  scales {q.scale.shape}")

# %% [markdown]
# The scale shapes are the ones a compressed-tensors checkpoint stores: `(1,)` per tensor, `(out, 1)` per
# channel, `(out, in/g)` per group.
#
# ## Worked example 2 — an outlier row, and an outlier column
# First serving-engine §8's case: one output channel 100× larger. Then the case that matters more for LLM
# weights: one **input** column 20× larger (the column that meets a large activation channel, notebook 03).

# %%
rng = np.random.default_rng(1)
Wr = rng.standard_normal((32, 64)) * 0.02
Wr[3] *= 100                                                          # one output row
for gran in ("tensor", "channel"):
    Wh = G.fake_quant(Wr, fmt="int8", granularity=gran)
    print(f"outlier ROW, INT8 per {gran:7}: aggregate error {G.error(Wr, Wh)['rel']:.3f}, "
          f"median row {np.median(G.row_errors(Wr, Wh)):.3f}")
Wc = rng.standard_normal((256, 512)) * 0.02
Wc[:, 7] *= 20                                                        # one input column
rest = [c for c in range(512) if c != 7]
for gran, g in (("tensor", 0), ("channel", 0), ("group", 128), ("group", 64), ("group", 32)):
    Wh = G.fake_quant(Wc, fmt="int4", granularity=gran, group_size=g or 128, convention="full")
    print(f"outlier COLUMN, INT4 {gran:7}{g or '':<4}: other columns {G.error(Wc[:, rest], Wh[:, rest])['rel']:.3f}, "
          f"the outlier column {G.error(Wc[:, [7]], Wh[:, [7]])['rel']:.3f}")

# %% [markdown]
# Aggregate metrics are dominated by the outlier and look fine while the ordinary values are wrecked. Per-channel
# scales fix an outlier row completely and an outlier column not at all (it sets every row's scale); groups
# shrink the damage to one group of each row — 61% error on the other columns per channel, 32% with groups of
# 128, 18% with groups of 32.
#
# ## Worked example 3 — activations: the outlier channels are in every token
# The tiny model's first up-projection reads `RMSNorm(x) × gain`, and four gains are 25–40×. That is how real
# LLMs get "massive activations": a few fixed channels, large in every token.

# %%
m = TinyModel()
Xc, _ = m.sample(256, "calib")
A = m.calibration_inputs(Xc)["blocks.0.up"]
amax = np.abs(A).max(0)
out = np.argsort(-amax)[:4]
normal = np.setdiff1d(np.arange(A.shape[1]), out)
print(f"outlier channels {sorted(out.tolist())}: absmax {np.round(np.sort(amax[out]), 1).tolist()}; "
      f"other channels' median absmax {np.median(amax[normal]):.2f}")
for label, Ah in (("INT8 per token (dynamic)", G.quantize_activations(A, "int8", "token")),
                  ("INT8 per tensor", G.quantize_activations(A, "int8", "tensor")),
                  ("FP8 per token (dynamic)", G.quantize_activations(A, "fp8", "token"))):
    print(f"{label:25}: whole tensor {G.error(A, Ah)['rel']:.4f}, the 60 ordinary channels {G.error(A[:, normal], Ah[:, normal])['rel']:.4f}")

# %% [markdown]
# A per-token scale is set by the outlier channel, so the ordinary channels of that token round with a step of
# ~amax/127 — an 11% error on them, invisible in the 1.4% aggregate. FP8 per token keeps them at 2.7%: a float
# grid's precision is *relative*, so small values keep their 3 mantissa bits. This is the argument for FP8
# activations, and for SmoothQuant when the format is INT8.
#
# ## Worked example 4 — static vs dynamic activation scales
# A static scale is one number per tensor, fixed at calibration — no max-reduction at run time, and the only
# option for some kernels — but anything above it saturates. Calibrate on 256 samples, test on 2,000 others.

# %%
Xt, _ = m.sample(2000, "test")
At = m.calibration_inputs(Xt)["blocks.0.up"]
Wu = m.weights["blocks.0.up"]
ref = At @ Wu.T
for label, a in (("calibration max", np.abs(A).max()), ("99.99th percentile", np.percentile(np.abs(A), 99.99)),
                 ("99.9th percentile", np.percentile(np.abs(A), 99.9)), ("99th percentile", np.percentile(np.abs(A), 99))):
    Y, info = w8a8.w8a8_matmul(At, Wu, "int8", act="tensor", static_amax=a)
    print(f"static amax = {label:19} ({a:6.1f}): output error {np.linalg.norm(Y - ref) / np.linalg.norm(ref):.4f}, "
          f"saturated {info['saturated']:.3%}")
Y, _ = w8a8.w8a8_matmul(At, Wu, "int8")
print(f"dynamic per token                  : output error {np.linalg.norm(Y - ref) / np.linalg.norm(ref):.4f}")

# %% [markdown]
# Clipping buys resolution for everyone else only while the clipped values do not matter. The 99.99th
# percentile clips 0.02% of values and edges out the max; the 99th clips 1% — the outlier channels themselves —
# and the error jumps to 24%. Dynamic per-token scales cost a reduction per token and win by 2× over the best
# static choice — why vLLM's FP8 and INT8 W8A8 paths default to them, and why `FP8_DYNAMIC` needs no
# calibration data at all.
#
# ## Worked example 5 — block scales (DeepSeek-V3's FP8)
# DeepSeek-V3's FP8 checkpoints store one FP32 scale per 128×128 weight tile and quantize activations per token
# per 128 channels on the fly. A block is a 2-D group: finer than per-tensor, and still aligned with the
# tiles a GEMM kernel works on.

# %%
Wb = rng.standard_normal((512, 512))
Wb[:128, :128] *= 50                                                  # one hot tile
cold = slice(128, None)
for fmt in ("int8", "fp8"):
    for gran in ("tensor", "block"):
        Wh = G.fake_quant(Wb, fmt=fmt, granularity=gran)
        print(f"{fmt} per {gran:6}: error on the rows away from the hot tile {G.error(Wb[cold], Wh[cold])['rel']:.4f}")
print(f"bits per weight with FP32 block scales: {F.bits_per_weight(8, 128 * 128, scale_bits=32):.3f}; "
      f"scales stored: {G.quantize(Wb, 'fp8', 'block').scale.shape}")

# %% [markdown]
# INT8 needs the local scale (50× less error away from the hot tile); FP8 barely notices, because a 50× range
# is well inside E4M3's 2^14.8. Block scales matter for FP8 when the range exceeds that — and they are what
# make an FP8 GEMM's epilogue need one rescale per 128-wide k block (notebook 04).
#
# ## Exercise 2.1 — group-wise INT4, checkpoint layout
# Write `group_int4(W, g)` for `W[out, in]`: symmetric, **full** convention (codes −8…7, scale = group amax /
# 7.5), groups of `g` consecutive inputs in each row. Return `(codes, scales)` with codes shaped like `W` and
# scales shaped `(out, in // g)`.

# %% exercise
def group_int4(W, g):
    ### BEGIN SOLUTION
    o, i = W.shape
    Wg = W.reshape(o, i // g, g)
    scales = np.maximum(np.abs(Wg).max(-1), 1e-12) / 7.5
    codes = np.clip(np.round(Wg / scales[..., None]), -8, 7)
    return codes.reshape(o, i), scales
    ### END SOLUTION

# %% check
codes, scales = group_int4(Wc, 64)
q = G.quantize(Wc, "int4", "group", group_size=64, convention="full")
assert scales.shape == (256, 8) and codes.min() >= -8 and codes.max() <= 7
assert np.allclose(scales, q.scale) and np.allclose((codes.reshape(256, 8, 64) * scales[..., None]).reshape(256, 512), q.w_hat)
print(f"✅ codes {codes.shape}, scales {scales.shape}: {F.bits_per_weight(4, 64):.3f} bits per weight with fp16 scales")

# %% [markdown]
# ## Exercise 2.2 — the cheapest granularity that survives an outlier column
# For `Wc` (one input column 20× larger), pick the granularity with the **fewest bits per weight** that keeps the
# other columns' INT4 error (full convention) below 20%. Candidates: `"channel"`, and groups of 128, 64, 32.
# Set `choice` to `"channel"` or the group size as an int.

# %% exercise
### BEGIN SOLUTION
cands = [("channel", F.bits_per_weight(4, 512))] + [(g, F.bits_per_weight(4, g)) for g in (128, 64, 32)]
ok = [(c, b) for c, b in cands
      if G.error(Wc[:, rest], G.fake_quant(Wc, fmt="int4", granularity="channel" if c == "channel" else "group",
                                            group_size=128 if c == "channel" else c, convention="full")[:, rest])["rel"] < 0.2]
choice = min(ok, key=lambda cb: cb[1])[0]
### END SOLUTION

# %% check
assert choice == 32
print("✅ g32 (4.5 bits): only groups this small keep one outlier column from coarsening its neighbours - or move the outlier "
      "out of the weight's way first (AWQ scaling, rotations), which is notebook 03")

# %% [markdown]
# ## Exercise 2.3 — predict the per-token damage
# A token has 63 ordinary channels of rms 1 and one channel at ±60. Under per-token INT8 (restricted, ±127) the
# step is `60 / 127` for the whole token. Using the rounding-noise model (noise rms = step / √12), predict the
# relative error of the ordinary channels, `predicted`, then check it by measurement.

# %% exercise
rng3 = np.random.default_rng(3)
Xo = rng3.standard_normal((1000, 64))
Xo[:, 9] = 60 * np.sign(rng3.standard_normal(1000))
### BEGIN SOLUTION
predicted = (60 / 127) / np.sqrt(12) / 1.0
### END SOLUTION

# %% check
ordinary = [c for c in range(64) if c != 9]
measured = G.error(Xo[:, ordinary], G.quantize_activations(Xo, "int8", "token")[:, ordinary])["rel"]
assert abs(predicted - measured) < 0.01, (predicted, measured)
print(f"✅ predicted {predicted:.3f}, measured {measured:.3f}: one channel at 60x sets a step that leaves the others ~4 bits")

# %% [markdown]
# ## Exercise 2.4 — choose a static scale on calibration data only
# Choose `static_amax` for the up-projection's INT8 static per-tensor input scale among the four candidates of
# worked example 4, scoring each by the output error on the **calibration** activations `A` (never the test set:
# that is what calibration means). The check measures your choice on the held-out `At` and requires it within 10%
# of the best candidate there.

# %% exercise
### BEGIN SOLUTION
cal_ref = A @ Wu.T
cands = [np.abs(A).max(), *(np.percentile(np.abs(A), p) for p in (99.99, 99.9, 99))]
cal_err = [np.linalg.norm(w8a8.w8a8_matmul(A, Wu, "int8", act="tensor", static_amax=a)[0] - cal_ref) for a in cands]
static_amax = cands[int(np.argmin(cal_err))]
### END SOLUTION

# %% check
err = lambda a: np.linalg.norm(w8a8.w8a8_matmul(At, Wu, "int8", act="tensor", static_amax=a)[0] - ref) / np.linalg.norm(ref)
best = min(err(a) for a in (np.abs(A).max(), *(np.percentile(np.abs(A), p) for p in (99.99, 99.9, 99))))
assert err(static_amax) <= 1.1 * best
print(f"✅ static amax {static_amax:.1f}: held-out error {err(static_amax):.4f} (best candidate {best:.4f}) - "
      "a choice made on calibration data holds on new inputs when calibration looks like traffic")

# %% [markdown]
# ## Exercise 2.5 — FP8 block scales
# Write `fp8_block_scales(W, b=128)`: one scale per `b × b` tile, `tile amax / 448`, shaped `(out // b, in // b)`.

# %% exercise
def fp8_block_scales(W, b=128):
    ### BEGIN SOLUTION
    o, i = W.shape
    return np.abs(W.reshape(o // b, b, i // b, b)).max(axis=(1, 3)) / 448
    ### END SOLUTION

# %% check
s = fp8_block_scales(Wb)
assert s.shape == (4, 4) and np.allclose(s, G.quantize(Wb, "fp8", "block").scale)
assert s[0, 0] > 20 * s[1, 1]
print(f"✅ 16 scales for a 512x512 weight; the hot tile's scale is {s[0, 0] / np.median(s):.0f}x the median")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Granularity decides who pays for an outlier. We use per-channel scales for INT8
# weights and groups of 128 for INT4, and we know what each does *not* protect against: per-channel scales
# isolate an outlier row but not an outlier input column, which is in every row — groups confine it to one
# group, and at 4 bits one bad column still leaves its neighbours at ~30% error with groups of 128, so we pair
# INT4 with AWQ or GPTQ rather than shrink the groups. Activations are worse: LLMs have a few channels 20–40×
# larger in every token, so a per-token INT8 scale leaves the other channels ~4 bits. Dynamic per-token FP8
# keeps them at 3 mantissa bits, which is why FP8 W8A8 usually needs no smoothing and INT8 W8A8 does. We
# prefer dynamic activation scales; a static scale saturates whatever calibration missed. And we never trust an
# aggregate error number: we look per channel."
#
# **Drills**
# 1. *Why can't per-channel weight scales fix an outlier input column?* — The column is in every output row, so
#    it sets every row's scale; only groups along the input dimension (or moving the outlier, AWQ) confine it.
# 2. *Your per-token INT8 activations show 1.4% error. Should you relax?* — Not yet: look at the ordinary
#    channels. With an outlier channel at 30× they carry ~11% error while the aggregate hides it.
# 3. *Static or dynamic activation scales for FP8 W8A8?* — Dynamic per token unless the kernel requires static:
#    it costs one reduction, needs no calibration data (`FP8_DYNAMIC`), and never saturates on unseen inputs.
