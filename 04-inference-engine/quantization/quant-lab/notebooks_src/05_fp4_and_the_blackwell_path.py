# %% [markdown]
# # 05 · FP4 and the Blackwell path: E2M1, block scales, and when four bits are fast
#
# **Tier:** T0. This notebook uses only calculators and emulation. It has the E2M1 grid, and NVFP4 and
# MXFP4 quantizers that use the same steps as the compressed-tensors and vLLM code. It also has
# checkpoint layouts and a roofline throughput model for Blackwell (**simulated**, with peak numbers
# from datasheets, verify). It also shows the accuracy of FP4 weights and activations on the bundled
# tiny model.
#
# **Tier T1:** This tier needs one Blackwell GPU with CUDA 12.8+. Rent the GPU. Do not use a small one
# (for example, use a B200 or an RTX PRO 6000). Cloud Run offers the RTX PRO 6000 (verify). Everything
# product-specific here is `(verify)`.
#
# ## The one-minute version
#
# Four bits hold 15 values. E2M1 is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and their negatives. A format that
# you can use in practice gives every small block its own scale:
#
# | Format | Block | Scale | Bits per weight | Where it multiplies natively |
# |---|---|---|---|---|
# | **MXFP4** (OCP) | 32 | E8M0 (a power of two) | 4.25 | Blackwell, sm_100+ (weight-only in vLLM from sm_80) |
# | **NVFP4** | 16 | FP8 E4M3 + one FP32 per tensor | 4.5 | Blackwell, sm_100+ (W4A4, and weight-only from sm_75) |
# | INT4 g128 (for contrast) | 128 | bf16 | 4.125 | nowhere as 4-bit math: always W4A16 |
#
# On Blackwell, NVFP4 **W4A4** runs FP4 tensor cores at 2x the FP8 rate. The weights *and* the
# activations are in four bits. On all other GPUs, vLLM serves NVFP4 weights **weight-only** (Marlin:
# dequantize, then multiply in 16-bit). Weight-only serving saves memory and decode bandwidth, but not
# FLOPs. Also, a kernel that dequantizes and runs below the efficiency of the BF16 GEMM is *slower* than BF16 at
# prefill sizes. FP4 *activations* are the accuracy risk: one outlier channel sets the scale of its
# whole 16-block.
#
# Concepts: PRIMER §2 "Number formats" (FP4, MXFP4, NVFP4) and §5 "Weight-and-activation
# quantization" (FP4 W4A4 on Blackwell) ([`PRIMER.md`](../../PRIMER.md)). The device table is in layer 01
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
# The interval between E2M1 grid values is 0.5 below 2, 1 between 2 and 4, and 2 between 4 and 6.
# Values above 6 saturate. Write `e2m1(x)`. The function rounds to the nearest grid value, with **ties
# to the even mantissa**, and keeps the sign. The vLLM function `cast_to_fp4` uses the same rule. It
# rounds 0.25 to 0, 0.75 to 1, 1.25 to 1, 1.75 to 2, 2.5 to 2, 3.5 to 4 and 5 to 4.

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
# compressed-tensors rounds the `amax` of each block to a power of two $2^e$. It rounds down, but if
# the mantissa of `amax` is at least 1.75, it rounds up. Then it stores ${127 + e - 2}$ as an unsigned
# byte (E8M0). The `2` puts the block max in the top binade of E2M1. Write `mx_code(amax)` for an array
# of block maxima.
#
# An llm-compressor MXFP4 checkpoint stores this code. The reference rule of the OCP MX spec has no
# round-up: it is $\lfloor \log_2(\mathrm{amax}) \rfloor - 2$. This rule puts the block max in [4, 8).
# Under this rule, a block with a max of 7.5 clips to 6. But the rounding of compressed-tensors gives that block the next exponent (PRIMER §2).
# `quantcore.formats.mxfp4` implements both rules, `rule="ocp"` and `rule="compressed-tensors"`.

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
ocp = np.floor(np.log2(blocks)) - 2
print(f"✅ codes {mx_code(blocks).tolist()}; block max / scale lands in [3.5, 7): {np.round(ratio, 2).tolist()} "
      "(above 6 saturates: MX trades a little clipping for scales that are pure exponents). The OCP floor rule "
      f"would give {np.round(blocks / 2.0 ** ocp, 2).tolist()}: never below 4, up to 8")

# %% [markdown]
# ## Exercise 5.3 — NVFP4's two-level scale
#
# compressed-tensors (and the reference code of vLLM) quantize a tensor `w` with
# $\mathrm{global} = 448 \times 6/\mathrm{amax}(w)$. This value is a multiplier. Then, for each block of 16,
# they calculate $\mathrm{local} = \operatorname{E4M3}(\mathrm{global} \times \mathrm{block\_amax}/6)$ and
# $\mathrm{value} = \operatorname{E2M1}(w/(\mathrm{local}/\mathrm{global}))$. To dequantize, they calculate
# $\mathrm{value} \times \mathrm{local}/\mathrm{global}$.
#
# Write `nvfp4_fake_quant(w, group=16)`. It returns the dequantized weight. Use `e2m1`, and use
# `quantlab.numerics.minifloat_round(x, "e4m3")` to round to E4M3.

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
# Weight-only FP4 (NVFP4A16, MXFP4) is near INT4 g128. FP4 *activations* (NVFP4 W4A4) hit the planted
# massive-activation channels. In a token, each 16-value block that contains such a channel gets a scale
# that is 24x too coarse for its other 15 values. If SmoothQuant moves the outlier into the weights
# first, SmoothQuant recovers most of the lost accuracy. For the same reason, Blackwell W4A4 recipes
# depend on smoothing, rotations or quantization-aware training.

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
# Every recipe keeps the embeddings, `lm_head` and the norms in bf16. Write `model_gb(shape, bits)`. It
# returns the size in GB (1e9 bytes). Count the linear-layer parameters (`kv.params(shape).linear`) at
# `bits` per weight. Count all other parameters at 16 bits.

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
# This example uses the `down_proj` of Llama-3.1-8B on a B200. The dense peaks come from the `specs.py`
# of layer 01: 2,250 BF16, 4,500 FP8 and 9,000 FP4 TFLOP/s, and 8 TB/s (verify). `w4a16-nvfp4` is the
# NVFP4 checkpoint when you do not quantize its activations: FP4 bytes and BF16 math.

# %%
for M in (1, 64, 512, 4096):
    t = {s: v * 1e6 for s, v in fp4.gemm_times(M, 14336, 4096, "B200").items()}
    print(f"  B200, M={M:5d}: " + "  ".join(f"{s} {v:7.1f} us" for s, v in t.items()) + "   [SIMULATED, 100% of peak]")
print(f"B200 BF16 ridge: {serve.gpu('B200').peak('bf16') / 8e12:.0f} FLOP/byte; RTX PRO 6000: "
      f"{serve.gpu('RTXPRO6000').peak('bf16') / 1.6e12:.0f}")

# %% [markdown]
# At 100% efficiency, every kernel is at least as fast as BF16. Real kernels that dequantize are not.
# The Model Optimizer team of NVIDIA reported NVFP4 weight-only (W4A16) as *slower* than BF16 in 10 of
# 12 GEMM shapes on Blackwell. In the same announcement (dated 2026-09-16, verify), the team reported
# W4A4 as faster in 9 of 12. One term is not in the calculation at 100% efficiency: the efficiency of the
# mixed-input kernel relative to the BF16 GEMM.
#
# ## Exercise 5.5 — from which step size is weight-only FP4 slower than BF16?
#
# In your calculation, the W4A16-NVFP4 kernel runs at a fraction `eff` of the compute rate of the BF16
# GEMM. The memory side of this kernel does not change. The BF16 GEMM runs at 100%. Write
# `slower_from(K, N, gpu, eff)`. It returns the smallest token count `M` at which the W4A16 time is
# more than the BF16 time. Use `B.GEMM_PATH` for the bytes, and use `serve.gpu(gpu).peak("bf16")` and
# `mem_bw_gbs`.

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
assert m70 == fp4.weight_only_slower_from(14336, 4096, "B200", 0.7)
print(f"✅ on a B200 a W4A16 kernel at 70% of the BF16 GEMM's efficiency loses from {m70} tokens per step "
      f"({m90} at 90%) — every prefill chunk; W4A4 on FP4 tensor cores has no such cliff [SIMULATED]")

# %% [markdown]
# ## On Blackwell (T1, one rented GPU, verify)
#
# The `NVFP4` recipe of llm-compressor needs 20 calibration samples. Only the global scales of the
# activations use data. vLLM serves the result with no flag. Below sm_100, the same checkpoint loads
# weight-only.

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
# an FP8 scale per 16 and one FP32 scale per tensor. This format uses 4.5 bits per weight, slightly
# more than INT4 g128. On B200s, NVFP4 runs W4A4 on FP4 tensor cores at twice the FP8 rate.
#
# "On our H100s and L4s, the same checkpoint runs weight-only through Marlin. The weight-only path gives
# a memory win only. Also, a kernel that dequantizes can run below the efficiency of the BF16 GEMM.
# In that case, the kernel is slower than BF16 from a couple of hundred tokens per step. Every prefill
# is in that range of step sizes.
#
# "The accuracy risk is the activations. In the lab's model, FP4 activations decrease task accuracy
# from 100% to approximately half, until SmoothQuant moves the outlier channels into the weights. Thus,
# use FP4 W4A4 on Blackwell with a smoothing or rotation recipe and an eval gate. Use INT4 or FP8 on
# other GPUs."
#
# **Drill 1.** *Why does NVFP4 use 16-element blocks and an E4M3 scale when MXFP4 uses 32 and E8M0?*
# Finer blocks isolate outliers better. Also, an E4M3 scale has mantissa bits. But MX scales are powers
# of two, and they are up to ~2x too coarse. The cost is 0.25 more bits per weight and a per-tensor FP32
# scale.
#
# **Drill 2.** *We serve an NVFP4 checkpoint on H100s and prefill got slower than BF16. Bug?* No. Below
# sm_100, vLLM runs it weight-only (Marlin), with BF16 math plus dequantization. Because prefill is
# compute-bound, the dequantization is only overhead. Use FP8 on Hopper.
#
# **Drill 3.** *Does NVFP4's global scale divide or multiply?* In compressed-tensors, it is a
# multiplier, $448 \times 6/\mathrm{amax}$. The local scales are E4M3 of $\mathrm{global} \times \mathrm{block\_amax}/6$, and
# dequantization divides by the global again. An inverted global scale is a classic loader bug.
