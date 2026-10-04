# %% [markdown]
# # 02 · Granularity and outliers: who shares a scale
#
# **Tier:** T0. It needs numpy only and runs in a few seconds.
#
# ## The one-minute version
# The largest magnitude among the values that share a scale sets that scale. Thus an outlier makes the grid
# coarser for its whole group. This makes **granularity** the main accuracy lever that you control. Granularity is
# how many values share one scale.
#
# Per tensor, one value hurts everything. With one scale per **output channel** (a row of `W`), an outlier *row*
# stays isolated. But an outlier *input column* is in every row, thus per-channel scales cannot isolate it.
# **Groups** of 32–128 along the input dimension confine it to one group per row, for ${16/g}$ extra bits.
#
# The engine quantizes activations at run time. **Dynamic per-token** scales change with each token. **Static
# per-tensor** scales come from calibration, and they saturate any value larger than the values that calibration
# saw. LLM activations have a few channels that are 20–40× larger than the rest, in every token. Thus these
# channels set a per-token INT8 scale, and the other channels keep only a few levels. SmoothQuant (notebook 03) and
# the float grid of FP8 address this problem.
#
# After this notebook, you can predict which granularity survives which outlier. You can also say why aggregate
# error metrics hide the damage.
#
# Primer: `../PRIMER.md` §2 (outliers) and §3 *Granularity and the bits-per-weight budget*. The layout here is
# `W[out, in]` (torch and checkpoints). `minengine` stores `w[d_in, d_out]`, thus `W = w.T`.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

import numpy as np

from quantcore import TinyModel, formats as F, granularity as G, w8a8

# %% [markdown]
# ## Worked example 1 — the survey table, reproduced
# serving-engine PRIMER §8 quotes six schemes on one 256×128 Gaussian weight from `minengine.quant`. The same seed
# through the code of quantcore itself gives the same numbers (pinned in `tests/test_repo_numbers.py`). This is the
# start point, and this notebook goes beyond it.

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
# The scale shapes are the shapes that a compressed-tensors checkpoint stores: `(1,)` per tensor, `(out, 1)` per
# channel, `(out, in/g)` per group.
#
# ## Worked example 2 — an outlier row, and an outlier column
# First, the case of serving-engine §8: one output channel is 100× larger. Then, the harder case for INT4 weights:
# one weight **input column** is 20× larger than the rest. This is a *weight* outlier. Its cousin, an
# *activation* outlier channel, is worked example 3. There, the weight column is ordinary, and the problem is a
# different one.

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
# The outlier dominates the aggregate metrics. Thus these metrics look good, but the ordinary values get large
# errors. Per-channel scales correct an outlier row completely. They do not correct an outlier column at all,
# because the column sets the scale of every row. Groups decrease the damage to one group of each row. The error
# on the other columns is 61% per channel, 32% with groups of 128 and 18% with groups of 32.
#
# For a weight outlier, the remedies work on `W`:
#
# - Groups confine it.
# - GPTQ lets the other columns compensate for its rounding (notebook 03).
# - A rotation spreads it over every column (primer §4).
#
# AWQ does not apply. Its scales come from activation magnitudes, and the inputs of this column are ordinary.
#
# ## Worked example 3 — activations: the outlier channels are in every token
# The first up-projection of the tiny model reads $\operatorname{RMSNorm}(x) \times \mathrm{gain}$, and four gains
# are 25–40×. That is how real LLMs get "massive activations": a few channels, always the same, that are large in
# every token.

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
# The outlier channel sets the per-token scale. Thus the ordinary channels of that token round with a step of
# ~$\mathrm{amax}/127$. This gives them an 11% error, which the 1.4% aggregate does not show. FP8 per token keeps
# them at 2.7%. The precision of a float grid is *relative*, thus small values keep their 3 mantissa bits. This is
# the argument for FP8 activations, and for SmoothQuant when the format is INT8.
#
# What about the **weights** that those channels meet? Look at the columns of the up-projection for the four
# outlier channels. Also look at the source of its INT4 output error (`G.output_error_by_input`: channel $c$
# contributes $\lVert X_{:,c} \rVert^2 \cdot \lVert W_{:,c} - \hat{W}_{:,c} \rVert^2$).

# %%
Wu0 = m.weights["blocks.0.up"]
col = np.abs(Wu0).max(0)
share = G.output_error_by_input(A, Wu0, G.fake_quant(Wu0, fmt="int4", granularity="group", group_size=32, convention="full"))
print(f"weight-column absmax of the outlier channels {np.round(col[np.sort(out)], 3).tolist()}, median column "
      f"{np.median(col):.3f}; share of the INT4 g32 output error from those 4 channels: {share[out].sum() / share.sum():.1%}")

# %% [markdown]
# The columns are ordinary, and no weight granularity singles them out. But they are the source of 98% of the
# output error of the layer. The reason is that the rounding error of a column reaches the output multiplied by
# its input. That is the *activation* outlier case, and its remedies act on the activation side:
#
# - AWQ scales those exact columns up before it rounds them (a finer grid relative to them, folded back into the
#   norm).
# - SmoothQuant moves the activation range into them for W8A8 (notebook 03).
#
# Weight outliers (worked example 2) and activation outliers are different problems.
#
# ## Worked example 4 — static vs dynamic activation scales
# A static scale is one number per tensor, and calibration sets it. It needs no max-reduction at run time, and it
# is the only option for some kernels. But any value above it saturates. Calibrate on 256 samples. Then measure on
# 2,000 other samples.

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
# A clip gives more resolution to all other values, but only while the clipped values are not important. The
# 99.99th percentile clips 0.02% of values and is slightly better than the max. The 99th percentile clips 1%,
# which is the outlier channels themselves, and the error jumps to 24%. Dynamic per-token scales cost a reduction
# per token, and they win by 2× over the best static choice. This is why the FP8 and INT8 W8A8 paths of vLLM use
# them by default. It is also why `FP8_DYNAMIC` needs no calibration data at all.
#
# ## Worked example 5 — block scales (DeepSeek-V3's FP8)
# The FP8 checkpoints of DeepSeek-V3 store one FP32 scale per 128×128 weight tile. The scheme quantizes
# activations per token per 128 channels, at run time. A block is a 2-D group. It is finer than per-tensor, and it
# is still aligned with the tiles that a GEMM kernel works on.

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
# INT8 needs the local scale (50× less error away from the hot tile). FP8 shows almost no difference, because a
# 50× range is well inside the $2^{14.8}$ of E4M3. Block scales are important for FP8 when the range is larger
# than that. Also, block scales are the reason that the epilogue of an FP8 GEMM needs one rescale per 128-wide k
# block (notebook 04).
#
# ## Exercise 2.1 — group-wise INT4, checkpoint layout
# Write `group_int4(W, g)` for `W[out, in]`. Use the symmetric **full** convention (codes −8…7,
# $\mathrm{scale} = \text{group amax}/7.5$). Use groups of `g` consecutive inputs in each row. Return
# `(codes, scales)`. The codes have the same shape as `W`, and the scales have the shape `(out, in // g)`.

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
assert scales.shape == (256, 8) and codes.min() >= -8 and codes.max() <= 7 and np.allclose(scales, q.scale)
# -amax / scale is exactly -7.5 in exact arithmetic, a tie; in floating point it can land one ulp either side, so
# np.round may say -7 where quantcore (which resolves the tie the exact way, formats.quantize_int) says -8.
tie = np.isclose(np.abs(Wc / np.repeat(scales, 64, axis=1)), 7.5, rtol=0, atol=1e-9)
assert np.all((codes == q.codes.reshape(256, 512)) | tie)
print(f"✅ codes {codes.shape}, scales {scales.shape}: {F.bits_per_weight(4, 64):.3f} bits per weight with fp16 scales")

# %% [markdown]
# ## Exercise 2.2 — the cheapest granularity that survives an outlier column
# `Wc` has one input column that is 20× larger. For `Wc`, select the granularity with the **fewest bits per
# weight** that meets one condition. The INT4 error of the other columns (full convention) must stay below 20%.
# The candidates are `"channel"`, and groups of 128, 64, 32. Set `choice` to `"channel"`, or to the group size as
# an int.

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
print("✅ g32 (4.5 bits): only groups this small keep one outlier weight column from coarsening its neighbours - "
      "the other levers are GPTQ, which lets the other columns absorb its rounding (notebook 03), and rotations")

# %% [markdown]
# ## Exercise 2.3 — predict the per-token damage
# A token has 63 ordinary channels of rms 1 and one channel at ±60. Under per-token INT8 (restricted, ±127), the
# step is `60 / 127` for the whole token. Use the rounding-noise model
# ($\text{noise rms} = \mathrm{step}/\sqrt{12}$). Predict the relative error of the ordinary channels, and put it
# in `predicted`. Then measure the error to make sure that the prediction is correct.

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
# Select `static_amax` for the INT8 static per-tensor input scale of the up-projection. Select it from the four
# candidates of worked example 4. Give each candidate a score: the output error on the **calibration**
# activations `A`. Never use the test set, because that is what calibration means. The check measures your choice
# on the held-out `At`. Your choice must be within 10% of the best candidate there.

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
# Write `fp8_block_scales(W, b=128)`. It returns one scale per $b \times b$ tile, $\text{tile amax}/448$, with
# the shape `(out // b, in // b)`.

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
# weights and groups of 128 for INT4. We also know what each one does *not* protect against.
#
# "Per-channel scales isolate an outlier row, but not an outlier input column, which is in every row. Groups
# confine that column to one group. But at 4 bits, one bad column still leaves its neighbours at ~30% error with
# groups of 128. Thus we use INT4 together with GPTQ, and we do not make the groups smaller.
#
# "Activation outliers are a different problem: LLMs have a few channels that are 20–40× larger in every token.
# The weight columns that they meet are ordinary. But those inputs multiply the rounding error of these columns
# (98% of the error of our toy up-projection), and AWQ corrects that multiplied error. Also, these channels set a
# per-token INT8 activation scale that leaves the other channels ~4 bits. Dynamic per-token FP8 keeps them at 3 mantissa bits.
# This is why FP8 W8A8 usually needs no smoothing, and INT8 W8A8 does.
#
# "We prefer dynamic activation scales, because a static scale saturates any value that calibration did not see.
# Also, we never trust an aggregate error number. We look per channel."
#
# **Drills**
# 1. *Why can per-channel weight scales not correct an outlier input column?* The column is in every output row, thus
#    it sets the scale of every row. Only groups along the input dimension confine it (or GPTQ compensates for its
#    rounding, or a rotation spreads it). AWQ is for the other kind of outlier: a large *activation* channel.
# 2. *Your per-token INT8 activations show 1.4% error. Can you stop there?* Not yet. Look at the ordinary
#    channels. With an outlier channel at 30×, they carry ~11% error, and the aggregate hides it.
# 3. *Static or dynamic activation scales for FP8 W8A8?* Dynamic per token, unless the kernel must have static
#    scales. It costs one reduction, needs no calibration data (`FP8_DYNAMIC`), and never saturates on inputs that
#    calibration did not see.
