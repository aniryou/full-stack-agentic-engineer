# %% [markdown]
# # 01 · Number formats and the error they cost
#
# **Tier:** T0 — numpy only, a few seconds on a laptop or Colab CPU. If torch is installed, one cell checks
# the FP8 grids against torch's own `float8_e4m3fn` / `float8_e5m2` casts; without torch it says so and moves on.
#
# ## The one-minute version
# Every low-precision format is a **grid** of representable values plus a **scale** that stretches it over
# the data: `x ≈ code × scale`. Integer grids (INT8, INT4) are evenly spaced, so the absolute error is at
# most half a step everywhere and each extra bit halves it — about **6 dB of signal-to-noise per bit**.
# Floating grids (FP8 E4M3 and E5M2, FP4 E2M1) are spaced by powers of two, so the *relative* error is
# the same at every magnitude and the exponent bits buy **range** instead of precision: E4M3 has 3
# mantissa bits (≤ 6.25% rounding error) over 2^14.8 of normal range, E5M2 2 bits (≤ 12.5%) over 2^29.8.
# Block formats — **MXFP4** (a power-of-two scale per 32) and **NVFP4** (an FP8 scale per 16 plus one per
# tensor) — give a 4-bit float grid a local scale. Every scale costs bits: INT4 with a 16-bit scale per
# 128 weights is **4.125 bits per weight**, 4.156 with a 4-bit zero point, MXFP4 4.25, NVFP4 4.5. After
# this notebook you can say which grid a format is, predict its error from its bits and the data's
# crest factor, and count what a checkpoint really stores.
#
# Primer: `../PRIMER.md` §2 *Number formats* and §3 *Granularity and the bits-per-weight budget*. The
# survey this deepens — formats and granularity on one weight matrix — is serving-engine PRIMER §8 and
# `mini-engine-core` notebook 06; this notebook reproduces its table and goes further.

# %%
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # small matrices: one BLAS thread is fastest, and safe on busy machines

import numpy as np

from quantcore import cost, formats as F, granularity as G

rng = np.random.default_rng(0)

# %% [markdown]
# ## Worked example 1 — integer grids: symmetric, the extra code, and the zero point
# Symmetric INT4 has 16 codes, −8…7. Two conventions share it out:
# **restricted** (±7, scale = amax/7: `minengine.quant`, SmoothQuant's fake-quant) wastes the −8 code;
# **full** (−8…7, scale = amax/7.5: GPTQ's `quant.py`, compressed-tensors' `_calculate_range`) uses all 16,
# so its step is 7/7.5 of the restricted one. **Asymmetric** (AWQ, KIVI) shifts an unsigned 0…15 grid by a
# zero point so it covers [min, max] instead of [−amax, amax] — worth it for one-sided data.

# %%
for bits in (8, 4):
    print(f"INT{bits}: restricted {F.int_range(bits)}, full {F.int_range(bits, convention='full')}, "
          f"asymmetric {F.int_range(bits, symmetric=False)}")
w = rng.standard_normal((128, 256)) * 0.02                    # (out, in), as in torch.nn.Linear and checkpoints
for conv in ("restricted", "full"):
    e = G.error(w, G.fake_quant(w, fmt="int4", granularity="group", group_size=32, convention=conv))
    print(f"INT4 g32 {conv:10}: rel error {e['rel']:.4f}, SQNR {e['sqnr_db']:.2f} dB")
relu = np.maximum(rng.standard_normal((128, 256)), 0)          # one-sided data: ReLU outputs
for sym in (True, False):
    e = G.error(relu, G.fake_quant(relu, fmt="int4", granularity="channel", symmetric=sym))
    print(f"ReLU output, INT4 per channel, {'symmetric ' if sym else 'asymmetric'}: rel error {e['rel']:.4f}")

# %% [markdown]
# The full convention buys ~0.2 dB for free, which is why checkpoints use it. For one-sided data a
# symmetric grid spends half its codes on values that never occur — asymmetric halves the step.
#
# ## Worked example 2 — FP8: E4M3 vs E5M2, from the bit patterns

# %%
for f in (F.E4M3, F.E5M2, F.E2M1):
    print(f"{f.name:9} max {f.max_value:>8g}  min normal {f.min_normal:.3g}  min subnormal {f.min_subnormal:.3g}  "
          f"values >= 0: {len(f.grid()):3d}  normal range 2^{np.log2(f.max_value / f.min_normal):.1f}")
x = rng.standard_normal(10000)
for f in (F.E4M3, F.E5M2):
    rel = np.abs(F.to_float(x, f) - x) / np.abs(x)
    print(f"{f.name}: rounding error on N(0,1) values — median {np.median(rel):.2%}, "
          f"max {rel[np.abs(x) > f.min_normal].max():.2%} (bound 2^-(mantissa+1) = {2.0 ** -(f.man_bits + 1):.2%})")

# %% [markdown]
# E4M3 (1 sign, 4 exponent, 3 mantissa bits, bias 7) has no infinities: the pattern S.1111.111 is NaN, so
# its largest value is 1.75 × 2⁸ = **448**; 254 finite codes, 253 distinct values. E5M2 keeps IEEE's
# infinities, reaches **57,344**, and has half the precision. That is why inference uses E4M3 for weights,
# activations and the KV cache (vLLM's FP8 linear methods accept only `float8_e4m3fn`), and E5M2 is mostly a
# training format for gradients.
#
# **Check it against torch (optional).** torch 2.1+ has both dtypes; a cast must land on the same grid.

# %%
try:
    import torch
    xs = np.clip(rng.standard_normal(100000) * rng.uniform(0, 1, 100000) ** 3 * 400, -448, 448)
    for dt, f in ((torch.float8_e4m3fn, F.E4M3), (torch.float8_e5m2, F.E5M2)):
        same = np.array_equal(torch.tensor(xs).to(dt).double().numpy(), F.to_float(xs, f))
        print(f"torch {dt}: identical to formats.to_float on 100,000 values: {same}")
    print("overflow:", torch.tensor([500.0]).to(torch.float8_e4m3fn).item(), "(E4M3, this torch build)",
          torch.tensor([70000.0]).to(torch.float8_e5m2).item(), "(E5M2 has inf)")
except (ImportError, AttributeError):
    print("torch (2.1+) not installed - skipping the cross-check (the numpy grid is the reference here)")

# %% [markdown]
# A cast is not a quantizer: kernels divide by a scale and clamp to ±448 before casting (vLLM's
# `scaled_fp8_quant`), because what an overflowing cast does depends on the dtype and the library.
#
# ## Worked example 3 — FP4 and the block formats
# E2M1 has eight magnitudes, {0, 0.5, 1, 1.5, 2, 3, 4, 6}: too few to use without a local scale. MXFP4 gives
# every 32 values a power-of-two (E8M0) scale, NVFP4 every 16 an E4M3 scale plus one FP32 scale per tensor.
# Compare them with INT4 at similar bits on Gaussian weights and on heavy-tailed ones (Student-t, 3 degrees
# of freedom — closer to what real weight rows look like in the tails).

# %%
Wg, Wt = rng.standard_normal((256, 512)) * 0.02, rng.standard_t(3, (256, 512)) * 0.02
schemes = {
    "INT4 g32, fp16 scale (4.5 b)": lambda W: G.fake_quant(W, fmt="int4", granularity="group", group_size=32, convention="full"),
    "INT4 g16, fp16 scale (5.0 b)": lambda W: G.fake_quant(W, fmt="int4", granularity="group", group_size=16, convention="full"),
    "FP4 g32, fp16 scale (4.5 b)": lambda W: G.fake_quant(W, fmt="fp4", granularity="group", group_size=32),
    "MXFP4 (4.25 b)": lambda W: F.mxfp4(W)[2],
    "NVFP4 (4.5 b)": lambda W: F.nvfp4(W)[3],
}
print(f"{'':30}{'Gaussian':>10}{'heavy-tailed':>14}   (relative error)")
for name, fn in schemes.items():
    print(f"{name:30}{G.error(Wg, fn(Wg))['rel']:10.4f}{G.error(Wt, fn(Wt))['rel']:14.4f}")

# %% [markdown]
# Three lessons. On Gaussian data an evenly spaced grid is as good as a float grid at the same bits. On
# heavy tails the float grid wins: most values are small and E2M1 spends its codes near zero. And MXFP4's
# power-of-two scale is coarse — the block max lands anywhere in [4, 8) on the E2M1 grid: above 6 it is
# clipped, near 4 the top codes go unused — which is what NVFP4's E4M3 scale per 16 fixes, for 0.25 more bits.
#
# ## Worked example 4 — about 6 dB per bit, and what outliers cost
# The uniform-quantizer model: rounding noise has power step²/12, and with `step = 2·amax / 2^b`,
#
# `SQNR ≈ 6.02·b + 4.77 − 20·log10(amax / rms)`  (`formats.sqnr_rule_db`)
#
# The last term is the **crest factor**. A sine wave (crest √2) gives the textbook 6.02·b + 1.76; every 10×
# of amax over rms — one outlier — costs 20 dB, more than three bits.

# %%
w = rng.standard_normal((128, 256))
crest = np.mean(np.abs(w).max(1) / np.sqrt((w ** 2).mean(1)))
print(f"per-channel crest factor of 256 Gaussian values: {crest:.2f}")
for b in (2, 3, 4, 5, 6, 8):
    e = G.error(w, G.fake_quant(w, fmt=f"int{b}", granularity="channel", convention="full"))
    print(f"INT{b}: measured {e['sqnr_db']:5.1f} dB, rule {F.sqnr_rule_db(b, crest):5.1f} dB")

# %% [markdown]
# The rule holds from 4 bits up; at 2–3 bits the noise is no longer "busy" (most values round to one of a
# few codes) and the model over-predicts. The slope, ~6 dB per bit, is why INT8 is near-lossless and INT4
# needs care (serving-engine PRIMER §8: 43.0 vs 17.9 dB for the same weight).
#
# ## Worked example 5 — the bits a checkpoint really stores

# %%
rows = [("INT8 per channel (16-bit scale per 4,096)", F.bits_per_weight(8, 4096)),
        ("INT4 g128 symmetric (compressed-tensors W4A16)", F.bits_per_weight(4, 128)),
        ("INT4 g128 + 4-bit zero point (AWQ, GPTQ-format)", F.bits_per_weight(4, 128, zero_point_bits=4)),
        ("INT4 g32 symmetric", F.bits_per_weight(4, 32)),
        ("MXFP4 (E8M0 per 32)", F.bits_per_weight(4, 32, scale_bits=8)),
        ("NVFP4 (E4M3 per 16, + FP32 per tensor)", F.bits_per_weight(4, 16, scale_bits=8, tensor_scale_bits=32, numel=4096 * 4096)),
        ("FP8, FP32 scale per 128x128 block (DeepSeek-V3)", F.bits_per_weight(8, 128 * 128, scale_bits=32))]
for name, b in rows:
    print(f"{name:52} {b:8.5f} bits")
for name in ("qwen2.5-0.5b", "llama-3.1-8b"):
    m = cost.MODELS[name]
    print(f"{name}: bf16 {m.weight_bytes(16) / 1e9:.3f} GB -> INT4 g128 {m.weight_bytes(4.125) / 1e9:.3f} GB "
          f"({m.weight_bytes(16) / m.weight_bytes(4.125):.2f}x; embedding + LM head, "
          f"{m.embed_params * (1 if m.tied else 2) / m.params:.1%} of the parameters, stay 16-bit)")

# %% [markdown]
# Two INT4 conventions live in this repo and both are right: `minengine.quant` counts 4.125 bits (symmetric,
# a 16-bit scale per 128, as compressed-tensors' symmetric W4A16 stores it), `servelab.sizing` 4.156 (plus a
# 4-bit zero point, as AWQ and every GPTQ-format checkpoint — AutoGPTQ/GPTQModel's packed `qzeros` — store). And the whole-model ratio is
# far from 4×, because recipes leave the embedding and LM head in 16-bit: for Qwen2.5-0.5B, whose tied table
# is 27.6% of its parameters, INT4 buys 2.16×.
#
# ## Worked example 6 — how the codes are laid out
# compressed-tensors packs signed INT4 codes offset by +8, eight per int32, element 0 in the lowest bits;
# FP4 codes go two per byte, the first in the low nibble.

# %%
p = F.pack_int4(np.array([[-8, -7, 0, 1, 2, 3, 4, 7]]))
print("INT4 [-8,-7,0,1,2,3,4,7] ->", hex(int(p.view(np.uint32)[0, 0])), "| a [4096, 896] weight packs to",
      F.pack_int4(np.zeros((4096, 896), int)).shape)
print("FP4 [0.5, -6, 1.5, 0] ->", [hex(int(b)) for b in F.pack_fp4(np.array([0.5, -6.0, 1.5, 0.0]))])

# %% [markdown]
# ## Exercise 1.1 — round to any float format
# Write `round_float(x, exp_bits, man_bits, bias, max_value)`: saturate to ±max_value, find each value's
# binade `e = floor(log2 |x|)` but never below the smallest normal exponent `1 − bias` (below it the spacing
# stays fixed: subnormals), round `x / 2^(e − man_bits)` to the nearest integer (numpy's `np.round` ties to
# even), and scale back.

# %% exercise
def round_float(x, exp_bits, man_bits, bias, max_value):
    ### BEGIN SOLUTION
    x = np.clip(np.asarray(x, float), -max_value, max_value)
    e = np.floor(np.log2(np.maximum(np.abs(x), 2.0 ** (1 - bias))))
    step = 2.0 ** (e - man_bits)
    return np.clip(np.round(x / step) * step, -max_value, max_value)
    ### END SOLUTION

# %% check
xs = rng.standard_normal(5000) * 50
for f in (F.E4M3, F.E5M2, F.E2M1):
    assert np.array_equal(round_float(xs, f.exp_bits, f.man_bits, f.bias, f.max_value), F.to_float(xs, f)), f.name
ties = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])            # vLLM's cast_to_fp4 thresholds
assert list(round_float(ties, 2, 1, 1, 6.0)) == [0, 1, 1, 2, 2, 4, 4]
print("✅ one rounding rule gives E4M3, E5M2 and E2M1; FP4 ties go to the even mantissa, as vLLM's reference does")

# %% [markdown]
# ## Exercise 1.2 — asymmetric INT4 with a zero point
# For a vector `v`, compute `scale = (max − min) / 15` and `zero = round(−min / scale)` after widening the
# range to include 0, then codes `clip(round(v / scale) + zero, 0, 15)` and the dequantized `(codes − zero) × scale`.
# Return `(codes, scale, zero, v_hat)`.

# %% exercise
def asym_int4(v):
    ### BEGIN SOLUTION
    lo, hi = min(v.min(), 0.0), max(v.max(), 0.0)
    scale = (hi - lo) / 15
    zero = np.round(-lo / scale)
    codes = np.clip(np.round(v / scale) + zero, 0, 15)
    return codes, scale, zero, (codes - zero) * scale
    ### END SOLUTION

# %% check
v = np.array([-0.3, 0.0, 0.05, 0.4, 1.2])
codes, scale, zero, v_hat = asym_int4(v)
assert scale == 0.1 and zero == 3 and v_hat[1] == 0.0 and codes.min() >= 0 and codes.max() <= 15
s2, z2 = F.asym_params(v.min(), v.max(), 4)
assert np.isclose(s2, scale) and z2 == zero
print(f"✅ codes {codes.astype(int).tolist()}, scale {scale:.3f}, zero point {int(zero)}: 0 is exact, error <= {scale / 2:.3f}")

# %% [markdown]
# ## Exercise 1.3 — predict the SQNR of an outlier
# A weight row has 4,096 Gaussian values of rms 0.02 and one outlier of magnitude 0.8. Predict its SQNR under
# per-channel INT8 (full convention) from the rule — `predicted` — before measuring it: the rms barely
# moves, the amax is the outlier. Then say how many bits the outlier costs compared with the same row without
# it (`bits_lost`, using 6.02 dB per bit).

# %% exercise
row = np.random.default_rng(3).standard_normal(4096) * 0.02
row_out = row.copy()
row_out[100] = 0.8
### BEGIN SOLUTION
rms = np.sqrt(np.mean(row_out ** 2))
predicted = F.sqnr_rule_db(8, 0.8 / rms)
bits_lost = 20 * np.log10((0.8 / rms) / (np.abs(row).max() / np.sqrt(np.mean(row ** 2)))) / 6.02
### END SOLUTION

# %% check
measured = G.error(row_out[None], G.fake_quant(row_out[None], fmt="int8", convention="full"))["sqnr_db"]
assert abs(predicted - measured) < 1.0, (predicted, measured)
assert 2.8 < bits_lost < 3.4
print(f"✅ predicted {predicted:.1f} dB, measured {measured:.1f} dB: one value 40x the rms costs {bits_lost:.1f} bits of the row's 8")

# %% [markdown]
# ## Exercise 1.4 — what the weights weigh
# Fill `gb`: the weight memory of Llama-3.1-8B (`cost.MODELS["llama-3.1-8b"]`, embedding and LM head kept in
# 16-bit) for INT4 g128 **with** a 4-bit zero point, NVFP4, and MXFP4. Use `F.bits_per_weight` and
# `Model.weight_bytes(bits)`; report GB (10⁹ bytes).

# %% exercise
m8 = cost.MODELS["llama-3.1-8b"]
gb = {"int4-g128-asym": None, "nvfp4": None, "mxfp4": None}
### BEGIN SOLUTION
gb["int4-g128-asym"] = m8.weight_bytes(F.bits_per_weight(4, 128, zero_point_bits=4)) / 1e9
gb["nvfp4"] = m8.weight_bytes(F.bits_per_weight(4, 16, scale_bits=8)) / 1e9
gb["mxfp4"] = m8.weight_bytes(F.bits_per_weight(4, 32, scale_bits=8)) / 1e9
### END SOLUTION

# %% check
assert abs(gb["int4-g128-asym"] - 5.727) < 0.002          # servelab.sizing: 5,727,854,592 B with 8.03e9 params
assert abs(gb["nvfp4"] - 6.027) < 0.002 and abs(gb["mxfp4"] - 5.809) < 0.002
print(f"✅ Llama-3.1-8B: INT4 asym {gb['int4-g128-asym']:.2f} GB, MXFP4 {gb['mxfp4']:.2f} GB, NVFP4 {gb['nvfp4']:.2f} GB "
      "- the 2.1 GB of 16-bit embedding and LM head is a third of each")

# %% [markdown]
# ## Exercise 1.5 — the MXFP4 shared exponent
# For each block of 32, the OCP MX rule is `exp = floor(log2(block amax)) − 2` (2 = E2M1's largest exponent,
# since 6 = 1.5 × 2²), stored as the byte `exp + 127`. Write `mx_scale_codes(x)` returning those bytes for a
# 1-D array whose length is a multiple of 32.

# %% exercise
def mx_scale_codes(x):
    ### BEGIN SOLUTION
    amax = np.abs(x.reshape(-1, 32)).max(1)
    return (np.floor(np.log2(amax)) - 2 + 127).astype(np.uint8)
    ### END SOLUTION

# %% check
xb = rng.standard_normal(32 * 64) * np.repeat(2.0 ** rng.integers(-6, 6, 64), 32)
assert np.array_equal(mx_scale_codes(xb), F.mxfp4(xb)[1].ravel())
print("✅ E8M0 codes match; under the OCP rule a block whose amax is 7.5 keeps exponent 0 and saturates to",
      F.mxfp4(np.full(32, 7.5))[2][0])

# %% [markdown]
# That is the OCP spec's rule. llm-compressor writes MXFP4 checkpoints with compressed-tensors' variant
# (`round_to_power_2`): amax is rounded to a power of two first — up when its mantissa is ≥ 1.75 — so 7.5 = 1.875 × 2²
# gets exponent 1, becomes 3.75 → 4 on the grid, and is stored as 8 instead of clipping to 6. The block max lands in
# [3.5, 7) rather than [4, 8): less clipping, at the price of one step of scale for blocks just under a power of
# two. `F.mxfp4(x, rule="compressed-tensors")` implements it, and the lab's `quantlab.fp4.mxfp4_scale_exponent`
# (notebook 05) is the same rule.

# %%
Wmx = np.random.default_rng(0).standard_normal((256, 512)) * 0.02     # the primer §2 table's Gaussian weight
for rule in ("ocp", "compressed-tensors"):
    e, codes, xh = F.mxfp4(np.full(32, 7.5), rule=rule)
    print(f"{rule:18}: E8M0 byte {int(codes.ravel()[0])}, 7.5 -> {xh[0]}; Gaussian weight relative error "
          f"{G.error(Wmx, F.mxfp4(Wmx, rule=rule)[2])['rel']:.4f}")

# %% [markdown]
# ## Exercise 1.6 — pack INT4 codes the way a checkpoint does
# Write `pack8(codes)` for a 1-D array of eight signed codes in −8…7: add 8, then OR code *i* into bits
# `4i … 4i+3` of a uint32. Return a Python int.

# %% exercise
def pack8(codes):
    ### BEGIN SOLUTION
    out = 0
    for i, c in enumerate(codes):
        out |= (int(c) + 8) << (4 * i)
    return out
    ### END SOLUTION

# %% check
assert pack8([-8, -7, 0, 1, 2, 3, 4, 7]) == 0xFCBA9810
c = rng.integers(-8, 8, 8)
assert pack8(c) == int(F.pack_int4(c[None]).view(np.uint32)[0, 0])
print("✅ 0xfcba9810 - the value compressed-tensors' pack_to_int32 produces for [-8,-7,0,1,2,3,4,7]")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "A format is a grid and a scale. INT grids are evenly spaced, so their error is
# absolute and each bit is ~6 dB; FP grids are spaced by powers of two, so their error is relative — E4M3
# rounds within 6.25% anywhere in a 2^15 range, which is why FP8 is the activation and KV format and why a
# per-tensor FP8 scale is often enough. At 4 bits the grid alone is too coarse, so every 4-bit format carries
# small scales: INT4 with a 16-bit scale per 128 costs 4.125 bits, MXFP4 a power-of-two scale per 32 (4.25),
# NVFP4 an FP8 scale per 16 (4.5) — and the finer NVFP4 scale measurably beats MXFP4's. Outliers are the
# enemy of every integer grid: the SQNR loses 20 dB for every 10× of amax over rms. And the checkpoint is
# never 4× smaller: embeddings and the LM head stay 16-bit, 2.16× for Qwen2.5-0.5B."
#
# **Drills**
# 1. *Why E4M3 and not E5M2 for inference?* — Inference needs precision more than range (a scale handles
#    range): E4M3 has 3 mantissa bits (≤ 6.25% error) vs 2 (≤ 12.5%); E5M2's extra range suits gradients.
# 2. *What is 4.156 bits?* — INT4 plus a 16-bit scale and a 4-bit zero point per 128 weights: 4 + 20/128.
# 3. *A 4,096-value row gains one value 40× its rms. What does INT8 per-channel lose?* — about 3 bits: the
#    crest factor rises from ~3.9 to ~34, and 20·log10(34 / 3.9) ≈ 19 dB ≈ 3.1 bits (exercise 1.3).
