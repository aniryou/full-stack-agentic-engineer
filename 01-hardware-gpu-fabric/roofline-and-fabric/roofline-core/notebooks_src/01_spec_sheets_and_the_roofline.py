# %% [markdown]
# # 01 · Spec sheets and the roofline
#
# **Tier:** T0 — runs on a laptop, Colab CPU or CI; no GPU, no network. The measured
# counterpart is `gpu-bench-lab` notebook `01_measure_your_roofline` (T0 on your CPU, T1 on a GPU).
#
# ## The one-minute version
# A GPU datasheet is a handful of numbers, and two of them decide almost everything:
# **peak FLOP/s** (for the precision you actually run, *dense*) and **memory bandwidth**.
# Their ratio is the **ridge point**: the arithmetic intensity (FLOPs per byte moved) a
# kernel needs before the math units, not memory, become the limit. After this notebook
# you can read a datasheet without falling into its traps, place any kernel on the
# roofline `min(peak, intensity × bandwidth)`, and say *why* a batch-1 LLM decode runs at
# a fraction of a percent of peak. Primer: `../PRIMER.md` §1–2.

# %%
from roofline import specs
from roofline import roofline as rl

h100 = specs.get("h100-sxm")
print(h100.name, "| catalogue as of", specs.AS_OF, "(every number: verify)")
print("dense peaks, TFLOP/s:", h100.tflops)
print("memory:", h100.memory_gb, "GB at", h100.memory_tbs, "TB/s | L2:", h100.l2_mb, "MB | TDP:", h100.tdp_w, "W")
print("NVLink as marketed:", h100.scaleup_gbs, "GB/s -> per direction:", h100.scaleup_gbs_per_dir, "GB/s")

# %% [markdown]
# ## Trap 1 — the asterisk
# Datasheets headline tensor throughput *with 2:4 structured sparsity* (the asterisk), which
# is exactly twice the dense rate. LLM inference runs dense, so the catalogue stores dense
# numbers and `specs.from_sparse()` converts. H100's "1,979 TFLOPS\*" bf16 is 989 dense.
#
# ## Where a peak comes from
# A peak is units × work per clock × clock. Tensor-core work per SM per clock doubled from
# Ampere to Hopper, which is most of the A100 → H100 jump. Sustained clocks under power
# and thermal limits sit below the boost clock, so a peak is a ceiling, not a promise.

# %%
print(f"{'device':10s} {'SMs':>4s} {'FLOP/clk/SM':>12s} {'GHz':>5s} {'computed':>9s} {'datasheet':>10s}")
for key, (sms, per_clk, ghz) in specs.CLOCKS.items():
    print(f"{key:10s} {sms:4d} {per_clk:12d} {ghz:5.2f} {specs.peak_from_clock(sms, per_clk, ghz):9.1f} "
          f"{specs.DEVICES[key].tflops['fp16']:10.1f}")
print("\nFP32 on CUDA cores vs bf16 on tensor cores (H100):", h100.tflops["fp32"], "vs", h100.tflops["bf16"],
      f"-> {h100.tflops['bf16'] / h100.tflops['fp32']:.1f}x: a kernel that avoids tensor cores idles most of the chip")

# %% [markdown]
# ## The ridge point of every device in the catalogue
# `ridge = peak / bandwidth`, in FLOP per byte. Compute-first generations push it up
# (A100 80GB 153 → H100 295 → GB200 312: FLOPs grew faster than bandwidth); memory
# refreshes pull it back down (H200 is an H100 with more bandwidth: 206). Every narrower
# precision doubles it again — fp8 needs twice the intensity of bf16 to pay off.

# %%
rows = []
for key, d in specs.DEVICES.items():
    p = "bf16" if d.supports("bf16") else "fp16"
    rows.append((d.ridge(p), d.name, p, d.ridge("fp8") if d.supports("fp8") else None))
for ridge, name, p, r8 in sorted(rows):
    print(f"{name:38s} {p}: {ridge:6.0f} FLOP/B" + (f"   fp8: {r8:6.0f}" if r8 else ""))

# %% [markdown]
# ## Kernels on the roofline
# Compulsory bytes: each operand read once, each result written once (what a perfectly
# fused and tiled kernel would move). A GEMM of `m×k` by `k×n` does `2mnk` FLOPs over
# `(mk + kn + mn)·b` bytes, so a square GEMM has intensity `2n/(3b)`; a GEMV (one token
# through a weight matrix, `m = 1`) has about `2/b` = 1 FLOP/B at bf16.

# %%
N = 1 << 26
kernels = [rl.elementwise(N, 2, name="vector add, bf16"), rl.reduction(N, 4, name="sum, fp32"),
           rl.gemm(1, 4096, 4096, name="GEMV 1x4096x4096"), rl.gemm(64, 4096, 4096, name="GEMM 64x4096x4096"),
           rl.gemm(4096, 4096, 4096, name="GEMM 4096^3")]
print(rl.ascii_roofline(h100, kernels))

# %%
# Optional: a real log-log chart if matplotlib is installed (pip install matplotlib).
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed - the text chart above is the same picture")
else:
    xs = [10 ** (i / 20) for i in range(-20, 81)]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for key in ["t4", "l4", "h100-sxm"]:
        d = specs.DEVICES[key]; p = "bf16" if d.supports("bf16") else "fp16"
        ax.loglog(xs, [rl.attainable(x, d, p) / 1e12 for x in xs], label=f"{d.name} ({p})")
    for k in kernels:
        ax.scatter([k.intensity], [rl.attainable(k.intensity, h100) / 1e12], s=12)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)"); ax.set_ylabel("attainable TFLOP/s"); ax.legend(fontsize=7)
    plt.show()

# %% [markdown]
# ## Exercise 1.1 — the roofline from scratch
# Write `ridge(peak_tflops, bw_tbs)` (FLOP/byte) and `attainable(intensity, peak_tflops, bw_tbs)`
# (TFLOP/s). Units: TFLOP/s is 1e12 FLOP/s, TB/s is 1e12 bytes/s — so they cancel neatly.

# %% exercise
def ridge(peak_tflops: float, bw_tbs: float) -> float:
    ### BEGIN SOLUTION
    return peak_tflops / bw_tbs
    ### END SOLUTION


def attainable(intensity: float, peak_tflops: float, bw_tbs: float) -> float:
    ### BEGIN SOLUTION
    return min(peak_tflops, intensity * bw_tbs)
    ### END SOLUTION

# %% check
assert abs(ridge(989.4, 3.35) - 295.34) < 0.01
assert attainable(1.0, 989.4, 3.35) == 3.35 and attainable(1e4, 989.4, 3.35) == 989.4
for d in specs.DEVICES.values():
    p = "bf16" if d.supports("bf16") else "fp16"
    assert abs(ridge(d.tflops[p], d.memory_tbs) - d.ridge(p)) < 1e-9
    assert abs(attainable(50, d.tflops[p], d.memory_tbs) * 1e12 - rl.attainable(50, d, p)) < 1
print("✅ ridge and attainable match the library for all", len(specs.DEVICES), "devices")

# %% [markdown]
# ## Exercise 1.2 — read a datasheet
# Below is a datasheet for an imaginary part, written the way vendors write them. Fill in
# `read_sheet` so it returns the numbers you would actually plan with: **dense** tensor peaks
# (the footnote tells you which lines are sparse), memory in GB and TB/s, the scale-up link
# **per direction**, the network in **GB/s** (it is quoted in gigabits), and the bf16 ridge.
# `num()` pulls the number out of a string for you.

# %%
import re

def num(s: str) -> float:
    """First number in a string, commas removed: num('1,600 TFLOPS*') -> 1600.0"""
    return float(re.search(r"[\d,.]+", s).group().replace(",", ""))

sheet = {
    "BF16 Tensor Core": "1,600 TFLOPS*",
    "FP8 Tensor Core": "3,200 TFLOPS*",
    "FP32": "60 TFLOPS",
    "GPU memory": "96 GB HBM3e",
    "Memory bandwidth": "4 TB/s",
    "NVLink": "800 GB/s (total bidirectional)",
    "Network": "4 x 400 Gb/s",
    "Footnote": "* with sparsity",
}

# %% exercise
def read_sheet(sheet: dict) -> dict:
    ### BEGIN SOLUTION
    bf16 = specs.from_sparse(num(sheet["BF16 Tensor Core"]))
    fp8 = specs.from_sparse(num(sheet["FP8 Tensor Core"]))
    nics, gbit = re.findall(r"[\d.]+", sheet["Network"])
    return {
        "bf16_tflops": bf16,
        "fp8_tflops": fp8,
        "fp32_tflops": num(sheet["FP32"]),
        "memory_gb": num(sheet["GPU memory"]),
        "bw_tbs": num(sheet["Memory bandwidth"]),
        "link_gbs_per_dir": num(sheet["NVLink"]) / 2,
        "network_gbs": float(nics) * float(gbit) / 8,
        "ridge_bf16": bf16 / num(sheet["Memory bandwidth"]),
    }
    ### END SOLUTION

# %% check
got = read_sheet(sheet)
want = {"bf16_tflops": 800, "fp8_tflops": 1600, "fp32_tflops": 60, "memory_gb": 96, "bw_tbs": 4,
        "link_gbs_per_dir": 400, "network_gbs": 200, "ridge_bf16": 200}
for k, v in want.items():
    assert abs(got[k] - v) < 1e-6, (k, got[k], v)
print("✅ read like a planner: dense peaks, per-direction links, bytes not bits")

# %% [markdown]
# ## Exercise 1.3 — when does a square GEMM become compute-bound?
# Write `gemm_intensity(m, n, k, b)` and `smallest_compute_bound_square(device, precision, b)`:
# the smallest integer `n` for which an `n×n×n` GEMM reaches the ridge. Predict first: the
# intensity is `2n/(3b)`, so `n ≥ 1.5 · b · ridge`. Which needs the larger matrix, a T4 or an H100?

# %% exercise
def gemm_intensity(m: int, n: int, k: int, b: float = 2) -> float:
    ### BEGIN SOLUTION
    return 2 * m * n * k / ((m * k + k * n + m * n) * b)
    ### END SOLUTION


def smallest_compute_bound_square(device, precision: str = "bf16", b: float = 2) -> int:
    ### BEGIN SOLUTION
    n = 1
    while gemm_intensity(n, n, n, b) < device.ridge(precision):
        n += 1
    return n
    ### END SOLUTION

# %% check
assert abs(gemm_intensity(4096, 4096, 4096) - rl.gemm(4096, 4096, 4096).intensity) < 1e-9
assert abs(gemm_intensity(1, 8192, 8192) - rl.gemm(1, 8192, 8192).intensity) < 1e-9
assert smallest_compute_bound_square(h100) == 887                    # 1.5 x 2 x 295.3 = 886.03
assert smallest_compute_bound_square(specs.get("h200")) == 619           # more bandwidth, lower ridge
assert smallest_compute_bound_square(specs.get("t4"), "fp16") == 610
print("✅ H100 needs n >= 887, H200 n >= 619, T4 n >= 610: the higher the ridge, the bigger the GEMM")

# %% [markdown]
# ## Exercise 1.4 — the ridge moved
# The same kernel can be compute-bound on an old GPU and memory-bound on a new one. Write
# `flips(kernels, old, new, p_old, p_new)` returning the names of kernels that are
# compute-bound on `old` but memory-bound on `new` (use `rl.time_kernel(...).bound`).

# %%
square = [rl.gemm(n, n, n, 2, name=f"GEMM {n}^3") for n in (256, 512, 768, 1024, 2048)]

# %% exercise
def flips(kernels, old, new, p_old="fp16", p_new="bf16") -> list:
    ### BEGIN SOLUTION
    return [k.name for k in kernels
            if rl.time_kernel(k, old, p_old).bound == "compute" and rl.time_kernel(k, new, p_new).bound == "memory"]
    ### END SOLUTION

# %% check
assert flips(square, specs.get("t4"), h100) == ["GEMM 768^3"]
assert flips(square, specs.get("t4"), specs.get("l4")) == ["GEMM 768^3", "GEMM 1024^3"]
print("✅ a 768^3 GEMM saturates a T4 (I = 256 > 203) but starves an H100 (256 < 295)")

# %% [markdown]
# ## Exercise 1.5 — how many tokens reach the ridge?
# During decode every weight matrix is multiplied by a `[tokens × d]` activation. Write
# `tokens_to_ridge(device, d)`: the smallest number of tokens `m` for which
# `gemm(m, d, d, b)` is compute-bound on `device` at `precision`. Predict first: roughly the
# ridge itself. Then predict what FP8 (1-byte operands *and* the fp8 peak) does to the answer.

# %% exercise
def tokens_to_ridge(device, d: int, precision: str = "bf16", b: float = 2) -> int:
    ### BEGIN SOLUTION
    m = 1
    while rl.time_kernel(rl.gemm(m, d, d, b), device, precision).bound == "memory":
        m += 1
    return m
    ### END SOLUTION

# %% check
assert tokens_to_ridge(h100, 4096) == 346
assert tokens_to_ridge(h100, 8192) == 319
assert tokens_to_ridge(h100, 8192, "fp8", b=1) == 319     # half the bytes AND twice the peak
assert tokens_to_ridge(specs.get("t4"), 4096, "fp16") == 226
print("✅ ~300 tokens per weight read on an H100 (d=8192: 319), unchanged by FP8; a T4 needs ~226")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Every kernel pays max(FLOPs / peak, bytes / bandwidth). The
# ratio peak / bandwidth — about 295 FLOP per byte for an H100 in bf16 — is the intensity
# you need before compute is the limit. Elementwise ops are near 0.2, a GEMV is 1, a big
# square GEMM is over 1,000. So I read a datasheet for dense peak at my precision and for
# bandwidth, compute the ridge, and place each kernel. If it is left of the ridge, faster
# math is wasted: I cut bytes — fuse, tile, batch, quantize."
#
# **Drill.**
# 1. *The datasheet says 3,958 TFLOPS of FP8. What do you plan with?* — 1,979 dense; the
#    headline assumes 2:4 sparsity, which inference does not use; and sustained clocks are lower still.
# 2. *Why is batch-1 decode at <1% of peak on an H100?* — each token multiplies every weight
#    once: ~2 FLOPs per 2-byte weight, 1 FLOP/B, 295× below the ridge; the GPU is a bandwidth machine here.
# 3. *We moved a workload from T4s to H100s and some kernels got no faster than bandwidth
#    alone predicts. Why?* — the ridge moved from ~203 to ~295 FLOP/B; kernels between those
#    intensities (a 768³ GEMM) went from compute-bound to memory-bound.
