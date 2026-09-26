# %% [markdown]
# # 05 · FP4 and the Blackwell path: E2M1, block scales, and when four bits are fast
#
# **Tier:** T0 — calculators and emulation only: the E2M1 grid, NVFP4 and MXFP4 quantizers written
# the way compressed-tensors and vLLM write them, checkpoint layouts, a roofline throughput model for
# Blackwell (**simulated**; peak numbers from datasheets, verify) and the accuracy of FP4 weights and
# activations on the bundled tiny model. T1/T2 needs a Blackwell GPU (B200, RTX PRO 6000, GB200/GB300;
# Cloud Run offers the RTX PRO 6000 — verify) with CUDA 12.8+; everything product-specific here is
# `(verify)`.
#
# ## The one-minute version
#
# Four bits hold 15 values: E2M1 is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and their negatives. A usable
# format attaches a scale to every small block:
#
# | Format | Block | Scale | Bits per weight | Where it multiplies natively |
# |---|---|---|---|---|
# | **MXFP4** (OCP) | 32 | E8M0 (a power of two) | 4.25 | Blackwell, sm_100+ (weight-only in vLLM from sm_80) |
# | **NVFP4** | 16 | FP8 E4M3 + one FP32 per tensor | 4.5 | Blackwell, sm_100+ (W4A4; weight-only from sm_75) |
# | INT4 g128 (for contrast) | 128 | bf16 | 4.125 | nowhere as 4-bit math: always W4A16 |
#
# On Blackwell, NVFP4 **W4A4** runs FP4 tensor cores at 2x the FP8 rate — weights *and* activations
# in four bits. Everywhere else vLLM serves NVFP4 weights **weight-only** (Marlin: dequantize, multiply
# in 16-bit), which saves memory and decode bandwidth but not FLOPs — and a dequantizing kernel that
# runs below the BF16 GEMM's efficiency is *slower* than BF16 at prefill sizes. FP4 *activations* are
# the accuracy risk: one outlier channel sets the scale of its whole 16-block.
#
# Concepts: PRIMER §2 "Number formats" (FP4, MXFP4, NVFP4) and §5 "Weight-and-activation
# quantization" (FP4 W4A4 on Blackwell) ([`PRIMER.md`](../../PRIMER.md)); the device table is layer 01's
# ([`../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md`](../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md) §9).

# %%
import math
from quantlab import bench as B, compress as C, env, evalharness as E, fp4, kv, serve, tinymodel as tm
import numpy as np

print(env.describe())
print("E2M1 grid:", fp4.E2M1_GRID.tolist(), "| codes of 0.5, 6, -6:", fp4.fp4_encode([0.5, 6, -6]).tolist())
print("bits per weight:", {k: round(v, 4) for k, v in fp4.BPW.items()})
for fmt in ("bf16", "fp8", "int4-g128", "mxfp4", "nvfp4"):
    lay = fp4.layout(4096, 4096, fmt)
    print(f"  {fmt:10s} 4096x4096 Linear: " + ", ".join(f"{k} {dt}{list(s)}" for k, (dt, s, _) in lay.items()),
          f"= {fp4.layer_bytes(4096, 4096, fmt) / 2**20:.2f} MiB")

# %% [markdown]
# ## Exercise 5.1 — round to E2M1
#
# The spacing of E2M1 is 0.5 below 2, 1 between 2 and 4, and 2 between 4 and 6; values above 6
# saturate. Write `e2m1(x)`: round to the nearest grid value, **ties to the even mantissa** (vLLM's
# `cast_to_fp4`: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4), keeping the sign.

# %% exercise
def e2m1(x):
    ### BEGIN SOLUTION
    x = np.asarray(x, dtype=np.float64)
    a = np.minimum(np.abs(x), 6.0)
    step = np.where(a < 2, 0.5, np.where(a < 4, 1.0, 2.0))
    return np.copysign(np.minimum(np.round(a / step) * step, 6.0), x)
    ### END SOLUTION

# %% check
xs = np.array([0.25, 0.26, 0.74, 0.75, 1.25, 1.75, 2.5, 2.51, 3.5, 5.0, 5.01, 7.0, -1.3, -0.1])
assert e2m1(xs).tolist() == fp4.fp4_round(xs).tolist() == [0, 0.5, 0.5, 1, 1, 2, 2, 3, 4, 4, 6, 6, -1.5, -0.0]
r = np.random.default_rng(0).uniform(-8, 8, 10000)
assert np.array_equal(e2m1(r), fp4.fp4_round(r))
print("✅ E2M1 rounding matches vLLM's reference thresholds")

# %% [markdown]
# ## Exercise 5.2 — the MXFP4 scale: a power of two per 32
#
# compressed-tensors rounds each block's `amax` to a power of two `2^e` — down, unless the mantissa
# of `amax` is at least 1.75, then up — and stores `127 + e - 2` as an unsigned byte (E8M0; the `2`
# puts the block max in E2M1's top binade). Write `mx_code(amax)` for an array of block maxima.

# %% exercise
def mx_code(amax):
    ### BEGIN SOLUTION
    a = np.asarray(amax, dtype=np.float64)
    e = np.floor(np.log2(a))
    e = e + (a / 2.0 ** e >= 1.75)
    return (127 + e - 2).astype(np.uint8)
    ### END SOLUTION

# %% check
blocks = np.array([1.0, 1.7, 1.75, 6.0, 7.9, 0.01, 300.0])
assert mx_code(blocks).tolist() == fp4.mxfp4_scale_exponent(blocks).tolist()
ratio = blocks / 2.0 ** (mx_code(blocks).astype(int) - 127)
print(f"✅ codes {mx_code(blocks).tolist()}; block max / scale lands in [3.5, 7): {np.round(ratio, 2).tolist()} "
      "(above 6 saturates: MX trades a little clipping for scales that are pure exponents)")

# %% [markdown]
# ## Exercise 5.3 — NVFP4's two-level scale
#
# compressed-tensors (and vLLM's reference) quantize a tensor `w` with
# `global = 448 x 6 / amax(w)` (a multiplier), then per block of 16: `local = E4M3(global x block_amax / 6)`,
# `value = E2M1(w / (local / global))`, and dequantize `value x local / global`. Write
# `nvfp4_fake_quant(w, group=16)` returning the dequantized weight (use `e2m1` and
# `quantlab.numerics.minifloat_round(x, "e4m3")` for the E4M3 rounding).

# %% exercise
from quantlab import numerics as N

def nvfp4_fake_quant(w, group=16):
    ### BEGIN SOLUTION
    w = np.asarray(w, dtype=np.float64)
    gs = 448 * 6 / np.abs(w).max()
    b = w.reshape(w.shape[0], -1, group)
    local = N.minifloat_round(gs * np.abs(b).max(-1) / 6, "e4m3")
    local = np.where(local == 0, 2.0 ** -9, local)
    return (e2m1(b / (local[..., None] / gs)) * local[..., None] / gs).reshape(w.shape)
    ### END SOLUTION

# %% check
rng = np.random.default_rng(1)
w = rng.standard_normal((64, 256)) * 0.02
np.testing.assert_allclose(nvfp4_fake_quant(w), fp4.nvfp4_quantize(w).dequantize(), rtol=1e-12, atol=1e-15)
w_out = w.copy()
w_out[rng.random(w.shape) < 0.01] *= 20                                          # 1% outlier weights
table = {name: (fp4.sqnr_db(x, f(x)), fp4.sqnr_db(w_out, f(w_out))) for name, f in {
    "NVFP4 (16, E4M3)": lambda x: nvfp4_fake_quant(x), "MXFP4 (32, E8M0)": lambda x: fp4.mxfp4_quantize(x).dequantize(),
    "INT4 g128": lambda x: fp4.int4_group_fake_quant(x, 128), "INT4 g32": lambda x: fp4.int4_group_fake_quant(x, 32)}.items()
         for x in [w]}
for k, (a, b) in table.items():
    print(f"  {k:18s} SQNR Gaussian {a:5.1f} dB   with 1% outliers {b:5.1f} dB")
assert table["NVFP4 (16, E4M3)"][1] > table["MXFP4 (32, E8M0)"][1] and table["NVFP4 (16, E4M3)"][1] > table["INT4 g128"][1]
print("✅ your NVFP4 = the library's; small blocks with fine scales hold up best when outliers appear")

# %% [markdown]
# ## Worked example: FP4 weights versus FP4 activations on the tiny model
#
# Weight-only FP4 (NVFP4A16, MXFP4) is close to INT4 g128. FP4 *activations* (NVFP4 W4A4) hit the
# planted massive-activation channels: each 16-value block of a token that contains one gets a scale
# 24x too coarse for its other 15 values. SmoothQuant (migrate the outlier into the weights first)
# rescues most of it — the same reason Blackwell W4A4 recipes lean on smoothing, rotations or
# quantization-aware training.

# %%
ref = tm.load()
calib = C.calibration_inputs(ref, 128)
cache, rows = {}, {}
mx = ref.with_weights({n + ".weight": fp4.mxfp4_quantize(ref.weights[n + ".weight"]).dequantize() for n in ref.linear_names()})
rows["MXFP4 weight-only"] = E.mini_eval(mx, ref, n=500, ref_cache=cache)
for name, r in {"NVFP4A16 (weight-only)": C.Recipe("NVFP4A16"), "NVFP4 W4A4": C.Recipe("NVFP4"),
                "NVFP4 W4A4 + SmoothQuant 0.5": C.Recipe("NVFP4", ("smoothquant",), smoothing_strength=0.5),
                "NVFP4 W4A4 + SmoothQuant 0.8": C.Recipe("NVFP4", ("smoothquant",))}.items():
    q = C.quantize_model(ref, r, calib)
    rows[name] = E.mini_eval(q.model(), ref, n=500, act_quant=q.act_quant(), ref_cache=cache)
print("measured on the bundled tiny model (T0)")
print(E.table(rows))

# %% [markdown]
# ## Exercise 5.4 — how big is an NVFP4 model?
#
# Every recipe keeps the embeddings, `lm_head` and norms in bf16. Write `model_gb(shape, bits)`: the
# linear-layer parameters (`kv.params(shape).linear`) at `bits` per weight plus everything else at 16
# bits, in GB (1e9 bytes).

# %% exercise
def model_gb(shape, bits):
    ### BEGIN SOLUTION
    p = kv.params(shape)
    return (p.linear * bits / 8 + (p.total - p.linear) * 2) / 1e9
    ### END SOLUTION

# %% check
llama = kv.load_shape("llama-3.1-8b-instruct")
sizes = {f: model_gb(llama, fp4.BPW[f]) for f in ("bf16", "fp8", "int4-g128-asym", "mxfp4", "nvfp4")}
assert abs(sizes["bf16"] - 16.06) < 0.01 and abs(sizes["fp8"] - 9.08) < 0.01 and abs(sizes["int4-g128-asym"] - 5.73) < 0.01
assert abs(sizes["nvfp4"] - kv.weight_bytes(llama, "nvfp4") / 1e9) < 1e-6
print("✅ Llama-3.1-8B:", {k: f"{v:.2f} GB" for k, v in sizes.items()},
      "— NVFP4 is *larger* than INT4 g128: it pays 0.34 more bits for finer blocks, and buys FP4 math on Blackwell")

# %% [markdown]
# ## Worked example: the Blackwell roofline for one GEMM
#
# `down_proj` of Llama-3.1-8B on a B200 (dense peaks from layer 01's `specs.py`: 2,250 BF16, 4,500 FP8,
# 9,000 FP4 TFLOP/s; 8 TB/s — verify). `w4a16-nvfp4` is what the NVFP4 checkpoint becomes when its
# activations are not quantized: FP4 bytes, BF16 math.

# %%
for M in (1, 64, 512, 4096):
    t = {s: B.gemm_time(M, 14336, 4096, "B200", s) * 1e6 for s in ("bf16", "fp8", "w4a4-nvfp4", "w4a16-nvfp4")}
    print(f"  B200, M={M:5d}: " + "  ".join(f"{s} {v:7.1f} us" for s, v in t.items()) + "   [SIMULATED, 100% of peak]")
print(f"B200 BF16 ridge: {serve.gpu('B200').peak('bf16') / 8e12:.0f} FLOP/byte; RTX PRO 6000: "
      f"{serve.gpu('RTXPRO6000').peak('bf16') / 1.6e12:.0f}")

# %% [markdown]
# At 100% efficiency every kernel is at least as fast as BF16. Real dequantizing kernels are not:
# NVIDIA's Model Optimizer team reported NVFP4 weight-only (W4A16) *slower* than BF16 in 10 of 12
# GEMM shapes on Blackwell, and W4A4 faster in 9 of 12 (announcement dated 2026-09-16, verify) — the
# missing term is the mixed-input kernel's efficiency relative to the BF16 GEMM.
#
# ## Exercise 5.5 — from which step size is weight-only FP4 slower than BF16?
#
# Model the W4A16-NVFP4 kernel as reaching a fraction `eff` of the BF16 GEMM's compute rate (its memory
# side unchanged) and the BF16 GEMM at 100%. Write `slower_from(K, N, gpu, eff)`: the smallest token
# count `M` at which the W4A16 time exceeds the BF16 time (use `B.GEMM_PATH` for the bytes and
# `serve.gpu(gpu).peak("bf16")`, `mem_bw_gbs`).

# %% exercise
def slower_from(K, N, gpu, eff):
    ### BEGIN SOLUTION
    g = serve.gpu(gpu)
    w4, a4, _ = B.GEMM_PATH["w4a16-nvfp4"]
    bw, peak = g.mem_bw_gbs * 1e9, g.peak("bf16")
    for M in range(1, 1 << 15):
        t4 = max(2 * M * K * N / (peak * eff), (K * N * w4 + M * K * a4 + M * N * 2) / bw)
        t16 = max(2 * M * K * N / peak, (K * N * 2 + M * K * 2 + M * N * 2) / bw)
        if t4 > t16:
            return M
    return None
    ### END SOLUTION

# %% check
m70 = slower_from(14336, 4096, "B200", 0.7)
m90 = slower_from(14336, 4096, "B200", 0.9)
assert slower_from(14336, 4096, "B200", 1.0) is None and 150 < m70 < m90 < 300
print(f"✅ on a B200 a W4A16 kernel at 70% of the BF16 GEMM's efficiency loses from {m70} tokens per step "
      f"({m90} at 90%) — every prefill chunk; W4A4 on FP4 tensor cores has no such cliff [SIMULATED]")

# %% [markdown]
# ## On Blackwell (T1/T2, verify)
#
# llm-compressor's `NVFP4` recipe needs 20 calibration samples (only the activations' global scales use
# data); vLLM serves the result with no flag. Below sm_100 the same checkpoint loads weight-only.

# %%
print(C.llmcompressor_script(C.Recipe("NVFP4"), "Qwen/Qwen2.5-1.5B-Instruct"))
for g in ("B200", "RTXPRO6000", "H100-80GB", "L4"):
    p = serve.plan("nvfp4", g)
    print(f"{g:11s} {p.compute}  [{p.kernel}]")
print("NVFP4 KV cache (--kv-cache-dtype nvfp4):", {g: kv.attention_backend(g, "nvfp4").backend or "refused"
                                                   for g in ("B200", "RTXPRO6000", "H100-80GB")})

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "FP4 is a Blackwell feature, not a checkpoint format. NVFP4 stores E2M1 values with
# an FP8 scale per 16 and one FP32 scale per tensor, 4.5 bits per weight — slightly more than INT4 g128
# — and on B200s it runs W4A4 on FP4 tensor cores at twice the FP8 rate. On our H100s and L4s the same
# checkpoint runs weight-only through Marlin: a memory win only, and a dequantizing kernel below the
# BF16 GEMM's efficiency is slower than BF16 from a couple of hundred tokens per step, i.e. every
# prefill. The accuracy risk is the activations: in the lab's model, FP4 activations drop task accuracy
# from 100% to about half until SmoothQuant moves the outlier channels into the weights. So: FP4 W4A4 on
# Blackwell with a smoothing or rotation recipe and an eval gate; INT4 or FP8 elsewhere."
#
# **Drill 1.** *Why does NVFP4 use 16-element blocks and an E4M3 scale when MXFP4 uses 32 and E8M0?* —
# Finer blocks isolate outliers better and an E4M3 scale has mantissa bits (MX scales are powers of two,
# up to ~2x too coarse); the price is 0.25 more bits per weight and a per-tensor FP32 scale.
#
# **Drill 2.** *We serve an NVFP4 checkpoint on H100s and prefill got slower than BF16. Bug?* — No: below
# sm_100 vLLM runs it weight-only (Marlin), BF16 math plus dequantization; prefill is compute-bound, so
# the dequantization is pure overhead. Use FP8 on Hopper.
#
# **Drill 3.** *Does NVFP4's global scale divide or multiply?* — In compressed-tensors it is a multiplier,
# `448 x 6 / amax`; local scales are E4M3 of `global x block_amax / 6`, and dequantization divides by the
# global again. Inverting it is a classic loader bug.
