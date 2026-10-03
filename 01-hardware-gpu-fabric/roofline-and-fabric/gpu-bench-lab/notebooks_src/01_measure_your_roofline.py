# %% [markdown]
# # 01 · Measure your roofline
#
# **Tier:** T0. The notebook runs on any CPU (laptop, Colab, CI) and measures *that* CPU with numpy. The same
# cells become **T1** when PyTorch can see a CUDA GPU (Colab/Kaggle T4, an L4, any rented GPU). Then the backend
# changes to torch, and every number is the GPU's. Concepts: primer §1 "Spec-sheet literacy" and §2 "The roofline
# model" ([`../../PRIMER.md`](../../PRIMER.md)).
#
# **Predicted first in** [roofline-core notebook 01](../../roofline-core/notebooks/01_spec_sheets_and_the_roofline.ipynb):
# in that notebook, you calculated ridges, attainable FLOP/s and the H100's decode crossover from datasheets. In this
# notebook, you measure the same quantities on the machine in front of you, and you compare the predictions with them.
#
# ## The one-minute version
#
# A roofline is two numbers that you can measure. The first is the most FLOP/s that any kernel gets (a large GEMM).
# The second is the most bytes/s that any kernel moves (a STREAM kernel). Their ratio is the **ridge point**. It is
# the number of FLOPs that an operation must do for each byte that it moves before arithmetic, not memory, is the
# limit.
#
# In this notebook, you measure the two numbers on the machine in front of you. You put GEMMs of different sizes and
# dtypes on *your* roofline. You explain what your measured peak tells you about the hardware. Then you predict and
# measure how the time of a decode projection changes with the batch. The time is flat while the projection is
# memory-bound, and this is the fact behind "batching is nearly free".

# %%
import math
import time

from IPython.display import Markdown, display

from gpubench import gemm, get_backend, inventory, measure, membw, roofline, specs
from gpubench.accounting import gemm_cost, gemm_intensity, stream_cost
from gpubench.measure import Measurement, si
from gpubench.report import measurements_markdown
from gpubench.timing import Timing

QUICK = True            # False: bigger sizes (minutes, not seconds) and numbers closer to the true peak
be = get_backend("auto")
info = be.describe()
for key in ("name", "backend", "blas", "usable_cpus", "simd_bits", "isa", "caches",
            "compute_capability", "sm_count", "memory_bytes", "torch"):
    if info.get(key) is not None:
        print(f"{key:>18}: {info[key]}")

# %% [markdown]
# ## 1 · What machine is this?
#
# Record the machine before you measure it, because a number without its hardware has no use. For a CPU, the flat
# roof is the product **cores × clock × vector lanes × 2 × FMA units**. The 2 is there because an FMA is a multiply
# and an add. In numpy, the BLAS library (OpenBLAS or MKL, one thread per core) gets to that roof. The slanted roof is the memory channels.
#
# For a GPU, the flat roof is SMs × clock × tensor-core rate for the dtype. The slanted roof is HBM (or GDDR on an L4/T4).
#
# ## 2 · Counting work: FLOPs and bytes
#
# A GEMM $C(m \times n) = A(m \times k) \cdot B(k \times n)$ does $m \cdot n \cdot k$ multiply-adds. That is
# **${2mnk}$ FLOPs**. Its **compulsory traffic** is one read of $A$ and $B$ and one write of $C$:
# $(mk + kn + mn) \cdot b$ bytes for $b$-byte elements. A kernel moves at least that quantity, and more if its
# tiles do not fit on the chip. Thus the ratio is the best-case **arithmetic intensity** (primer §2). Each shape in
# the next cell is an LLM matmul:

# %%
cases = [(4096, 4096, 4096, 2, "square 4096 GEMM, bf16"),
         (1, 8192, 8192, 2, "decode, batch 1: one token × a d=8192 projection"),
         (64, 8192, 8192, 2, "decode, batch 64"),
         (2048, 8192, 8192, 2, "prefill of a 2,048-token prompt")]
print(f"{'shape':<50} {'FLOPs':>11} {'bytes':>10} {'FLOP/B':>8}")
for m, n, k, b, what in cases:
    c = gemm_cost(m, n, k, b)
    print(f"{what:<50} {si(c.flops, 'FLOP'):>11} {si(c.bytes, 'B'):>10} {c.intensity:8.1f}")

# %% [markdown]
# Decode at batch 1 does about **one FLOP per byte** at bf16 (two FLOPs per weight, two bytes per weight). On all
# hardware, the rate at which the weights stream in is its limit. Batching uses each weight again for every sequence
# in the batch, so the intensity increases approximately with the batch size.
#
# ## Exercise 1.1 — count a GEMM yourself
#
# Write `my_gemm_cost(m, n, k, b)`. It returns `(flops, bytes)` with the conventions of §2.

# %% exercise
def my_gemm_cost(m, n, k, b):
    ### BEGIN SOLUTION
    flops = 2 * m * n * k
    nbytes = (m * k + k * n + m * n) * b
    return flops, nbytes
    ### END SOLUTION

# %% check
assert my_gemm_cost(2, 2, 2, 4) == (16, 48)
assert my_gemm_cost(1, 8192, 8192, 2) == (134_217_728, 134_250_496)
for shape in [(4096, 4096, 4096, 2), (7, 13, 29, 8), (128, 64, 32, 1)]:
    f, nb = my_gemm_cost(*shape)
    ref = gemm_cost(*shape)
    assert (f, nb) == (ref.flops, ref.bytes), shape
print("✅ my_gemm_cost matches the lab's accounting (and a square GEMM's intensity is n/(1.5·b))")

# %% [markdown]
# ## 3 · Timing honestly
#
# The first call of anything is slow. Fresh buffers page-fault, the BLAS thread pool (or cuBLAS) starts and selects
# a kernel, and the clocks increase. Thus a benchmark does a warm-up first. Then it repeats the call until each
# sample is sufficiently long that the timer resolution is small in comparison. It takes several samples and reports
# the **best** (what the machine can do) and the **median** (what you typically see). On a GPU, it must also wait
# until the asynchronous work is complete, and the torch backend uses CUDA events for this.

# %%
op = be.make_gemm(1024, 1024, 1024, "float32")
t0 = time.perf_counter()
op.fn()
be.sync()
first = time.perf_counter() - t0
m = measure(be, op, "gemm", {"m": 1024, "n": 1024, "k": 1024, "dtype": "float32"}, repeats=5, min_time=0.05)
print(f"first call  : {si(first, 's')}")
print(f"best sample : {si(m.timing.best, 's')}   ({m.timing.inner} calls per sample, {len(m.timing.samples)} samples)")
print(f"median      : {si(m.timing.median, 's')}   spread (cv) {m.timing.cv:.1%}")
print(f"→ {si(m.flops_per_s(), 'FLOP/s')} best, {si(m.flops_per_s('median'), 'FLOP/s')} median")

# %% [markdown]
# ## 4 · The GEMM sweep: the flat roof
#
# The sweep runs square GEMMs over a range of sizes, in every dtype that this backend supports. Expect FLOP/s to
# increase with size. Small GEMMs cannot keep every core/SM busy, and their fixed costs are large compared with their
# work. Also expect a speedup each time that the element width decreases by half. The cause is two times the SIMD lanes
# on a CPU, and the tensor-core rate on a GPU.
#
# The speedup of fp16/bf16 over IEEE fp32 **depends on the part**. It is about 15–16× on an A100 or H100, 8× on a
# T4 and 4× on an L4. It is 2× on an RTX 4090 (its fp16 figure with fp32 accumulate, which is what PyTorch uses).
# TF32 has a rate between the two, where it exists. The next cell prints the ratios from the spec table, so you know what to expect on
# your GPU.
#
# On the CPU, numpy's float16 has **no BLAS path** at all. It is an emulated loop, and its row shows the cost of "no
# hardware support for this dtype". (On a T4, bf16 is the same: it has no native support.)

# %%
print(f"{'GPU (spec, dense)':<26} {'fp32':>7} {'tf32':>7} {'fp16/bf16':>10} {'16-bit : fp32':>14}")
for g in specs.GPUS:
    f32, half = g.dense_tflops.get("float32"), g.dense_tflops.get("bfloat16") or g.dense_tflops.get("float16")
    if f32 and half:
        tf32 = g.dense_tflops.get("tf32")
        print(f"{g.name:<26} {f32:7.1f} {tf32 if tf32 else '—':>7} {half:10.1f} {half / f32:13.1f}×")

# %%
skipped = []
gemms = gemm.sweep(be, quick=QUICK, skipped=skipped)
display(Markdown(measurements_markdown(gemms)))
for s in skipped:
    print("skipped:", s)

# %% [markdown]
# ## 5 · The bandwidth roof
#
# The slanted roof is the fastest rate at which any kernel moves bytes to and from main memory. STREAM's kernels
# measure it over arrays much larger than the last-level cache (notebook 02 examines this in detail). On a CPU, we
# use every usable core, because one core cannot fill the memory bus.

# %%
threads = 1 if be.is_gpu else inventory.usable_cpus()
n = membw.stream_elems(be)
streams = membw.stream_suite(be, n, threads=threads, repeats=3)
display(Markdown(measurements_markdown(streams)))
bw = membw.peak(streams)
print(f"slanted roof: {si(bw.bytes_per_s(), 'B/s')} ({bw.op}, {threads} thread(s), {si(bw.cost.bytes, 'B')} per call)")

# %% [markdown]
# ## Exercise 1.2 — build the roofline from your measurements
#
# roofline-core calculated ridges from datasheet numbers. In this notebook, the two roofs come from the tables of §4
# and §5. Write `my_roofline(gemms, streams, dtype)`. It returns `(peak_flops, peak_bw, ridge)`. Use the **best**
# sample of each measurement (what the machine *can* do). Two decisions need your judgement:
#
# * The flat roof is a *per-dtype* number. fp32 and fp64 are different machines.
# * The slanted roof must use only kernels whose counted bytes are all memory traffic. numpy's triad is two passes
#   (`m.extras["passes"] == 2`). Its second pass reads again what the first pass wrote immediately before (partly from
#   cache), and updates it in place. This skips the write-allocate read (notebook 02, §4). Thus its *moved* rate is
#   not a memory rate, and it can be faster than every single-pass kernel. Do not include such rows.

# %% exercise
def my_roofline(gemms, streams, dtype):
    ### BEGIN SOLUTION
    peak = max(m.flops_per_s() for m in gemms if m.params["dtype"] == dtype)
    bw = max(m.bytes_per_s() for m in streams if m.extras.get("passes", 1) == 1)
    return peak, bw, peak / bw
    ### END SOLUTION

# %% check
main = [d for d in dict.fromkeys(m.params["dtype"] for m in gemms) if not (be.name == "numpy" and d == "float16")]
for d in main:
    ref = roofline.measured_roofline(gemms, streams, d)
    peak, bwr, rdg = my_roofline(gemms, streams, d)
    assert (peak, bwr) == (ref.peak_flops, ref.peak_bw) and abs(rdg - ref.ridge) < 1e-9, d
# a two-pass row that "moves" twice as fast must not become the roof
fake_g = [Measurement("gemm", {"dtype": "float64"}, gemm_cost(512, 512, 512, 8), Timing((1e-3,)), "numpy", "cpu")]
one = Measurement("stream.add", {}, stream_cost("add", 10**6, 8), Timing((2.4e-3,)), "numpy", "cpu",
                  extras={"passes": 1})                                          # 10 GB/s
two = Measurement("stream.triad", {}, stream_cost("triad", 10**6, 8).times(5 / 3), Timing((2e-3,)), "numpy",
                  "cpu", extras={"passes": 2})                                   # "moves" 20 GB/s
assert my_roofline(fake_g, [one, two], "float64")[1] == one.bytes_per_s()
print("✅ your roofline:", ", ".join(f"{d} ridge {my_roofline(gemms, streams, d)[2]:.1f} FLOP/B" for d in main))

# %% [markdown]
# ## 6 · Your roofline
#
# Flat roof: the fastest GEMM of a dtype. Slanted roof: the fastest byte-mover. The chart is log-log, and each `o` is
# one of your GEMMs at its compulsory intensity. A small GEMM can be *above* the slanted roof. Its operands stay in
# cache between calls, so it does not actually move the compulsory bytes from DRAM. The roofline counts only DRAM (or
# HBM) traffic.

# %%
roofs = [roofline.measured_roofline(gemms, streams, d) for d in main]
best_dtype = max(main, key=lambda d: gemm.best(gemms, d).flops_per_s())
points = [("o", m.intensity, m.flops_per_s()) for m in gemms if m.params["dtype"] == best_dtype]
print(roofline.ascii_plot(roofs, points))
print(f"\n'o' = {best_dtype} GEMMs of sizes {[m.params['m'] for m in gemms if m.params['dtype'] == best_dtype]}")

# %% [markdown]
# ## Exercise 1.3 — which of your GEMMs were compute-bound?
#
# For each GEMM of `best_dtype`, return `(size, bound, efficiency)`. `bound` is `"memory"` or `"compute"` on *your*
# roofline. A GEMM is compute-bound at the ridge or to the right of it. `efficiency` is the achieved FLOP/s (best
# sample) divided by the attainable FLOP/s at the intensity of that GEMM. The attainable FLOP/s is the lower of the
# two roofs at that intensity.

# %% exercise
def classify(measurements, peak_flops, peak_bw):
    out = []
    ### BEGIN SOLUTION
    for m in measurements:
        att = min(peak_flops, m.intensity * peak_bw)
        kind = "compute" if m.intensity >= peak_flops / peak_bw else "memory"
        out.append((m.params["m"], kind, m.flops_per_s() / att))
    ### END SOLUTION
    return out

# %% check
roof = roofline.measured_roofline(gemms, streams, best_dtype)
mine = [m for m in gemms if m.params["dtype"] == best_dtype]
rows = classify(mine, roof.peak_flops, roof.peak_bw)
for (size, kind, eff), m in zip(rows, mine):
    assert kind == roof.bound(m.intensity) and abs(eff - roof.efficiency(m.flops_per_s(), m.intensity)) < 1e-9
    print(f"  {size:>6}  {kind:<8} {eff:6.1%} of its roof")
print(f"✅ classified on your {best_dtype} roofline (ridge {roof.ridge:.1f} FLOP/B)")

# %% [markdown]
# Read the result in two ways. First, examine **where the ridge is**. A CPU has a small quantity of compute for each
# byte of bandwidth, so its ridge is at approximately 5–20 FLOP/B. Thus almost every GEMM is to the right of it. A GPU's ridge
# is 150–400, so on a GPU the same small GEMMs are to the left of it (memory-bound).
#
# Second, examine **how far below its roof** each GEMM runs. If a small GEMM is "compute-bound" but gets only a third
# of its roof, something that the roofline does not model limits it. That can be thread start-up, kernel launch, or
# too few tiles to occupy every core or SM. That fixed cost per call is the $\alpha$ of notebook 02.
#
# (On a GPU run, the flat roof of each dtype is only your fastest GEMM of that dtype. Compare it with the spec in §7
# before you trust it.)

# %% [markdown]
# ## 7 · Measured vs spec
#
# For a GPU, the spec sheet gives dense peaks (the lab's `specs` table, dated (verify)). For a CPU, you can
# *calculate* one from the instruction set:
# $\text{cores} \times \text{GHz} \times \text{lanes} \times 2 \times \text{FMA units}$, with
# $\text{lanes} = \text{vector bits} / (8 \times \text{bytes per element})$ (`specs.cpu_peak_flops`). It is an
# estimate. The FMA-unit count is an assumption. Heavy AVX-512 code often runs below the nominal clock. Also, a cloud
# vCPU can be a hyperthread, not a core.
#
# ## Exercise 1.4 — what does your measured peak imply?
#
# Invert the formula. For a *measured* GEMM rate, write
# `implied_fma_units(measured_flops, cores, ghz, simd_bits, dtype_bytes)`. It returns the number of FMA units per
# core that explains the rate at the nominal clock.
#
# A value of about 2 on most x86 server cores (1 on some) tells you
# that BLAS gets to the vector peak. A value well below 1 tells you that it does not. Possible causes are sizes that
# are too small (`QUICK`), or fewer physical cores than vCPUs. A value above 2 tells you that the clock or the core count
# that you assumed is incorrect (turbo, or hyperthreads counted as cores).

# %% exercise
def implied_fma_units(measured_flops, cores, ghz, simd_bits, dtype_bytes):
    ### BEGIN SOLUTION
    lanes = simd_bits // (8 * dtype_bytes)
    return measured_flops / (cores * ghz * 1e9 * lanes * 2)
    ### END SOLUTION

# %% check
assert abs(implied_fma_units(358.4e9, 4, 2.8, 512, 8) - 2.0) < 1e-12      # 4 AVX-512 cores at the fp64 peak
assert abs(implied_fma_units(358.4e9, 4, 2.8, 512, 4) - 1.0) < 1e-12      # the same rate in fp32 is half the peak
assert abs(implied_fma_units(specs.cpu_peak_flops(8, 3.2, 128, 4, 4), 8, 3.2, 128, 4) - 4) < 1e-12
print("✅ implied_fma_units inverts the ISA peak: halve the element width and the same FLOP/s is half as many units")

# %%
if be.is_gpu:
    spec = specs.lookup(info["name"])
    if spec is None:
        print(f"no spec entry for {info['name']!r}: add one to gpubench/specs.py to compare")
    for d in main:
        got = gemm.best(gemms, d).flops_per_s()
        peak = spec.peak_flops(d) if spec else None
        print(f"{d:>14}: measured {si(got, 'FLOP/s'):>13}   spec {si(peak, 'FLOP/s') if peak else 'n/a':>13}"
              + (f"   {got / peak:5.1%} of dense peak" if peak else ""))
    if spec:
        print(f"{'memory':>14}: measured {si(bw.bytes_per_s(), 'B/s'):>13}   spec {si(spec.mem_bw, 'B/s'):>13}"
              f"   {bw.bytes_per_s() / spec.mem_bw:5.1%}")
    half = next((d for d in ("float16", "bfloat16") if d in main), None)     # fp16: every tensor-core GPU has it
    if half and "float32" in main:
        ratio = gemm.best(gemms, half).flops_per_s() / gemm.best(gemms, "float32").flops_per_s()
        exp = spec.peak_flops(half) / spec.peak_flops("float32") if spec and spec.peak_flops(half) else None
        print(f"{half} : float32 measured {ratio:.1f}×" + (f", spec {exp:.1f}× for this part" if exp else ""))
else:
    cores, ghz = info.get("usable_cpus") or info.get("logical_cpus"), info.get("ghz_nominal")
    for d in main:
        got = gemm.best(gemms, d).flops_per_s()
        if ghz:
            b_el = 8 if d == "float64" else 4
            units = implied_fma_units(got, cores, ghz, info["simd_bits"], b_el)
            est = specs.cpu_peak_flops(cores, ghz, info["simd_bits"], 2, b_el)
            print(f"{d:>8}: measured {si(got, 'FLOP/s'):>13} = {units:.2f} FMA units/core at {ghz} GHz on "
                  f"{cores} cpus, {info['simd_bits']}-bit   (the 2-unit estimate {si(est, 'FLOP/s')}: {got / est:.0%})")
        else:
            print(f"{d:>8}: measured {si(got, 'FLOP/s')}; no clock in the CPU model name, so no estimate")
    print("\nNo GPU here. For the tensor-core roofs: on Colab choose Runtime → Change runtime type → T4 GPU and "
          "re-run this notebook; or on any GPU box run `python -m gpubench run --backend torch` "
          "(deploy/any-gpu/ for Docker and plain pip, deploy/gcp/ for a Spot L4 VM).")

# %% [markdown]
# A healthy GEMM gets 70–85% of the *dense* spec of a GPU. If you compare against a number with an asterisk (2:4
# sparsity), you conclude that you get 40%. Then you look for a bug that does not exist. Under tensor load, a 70 W
# card (T4, L4) keeps lower clocks than its boost-clock spec.
#
# ## 8 · Your ridge next to datacenter GPUs
#
# The next cell shows the spec rooflines (dense, dated (verify)) at the dtype that LLM serving uses, next to the
# roofline that you measured. Every one of them needs hundreds of FLOPs per byte before compute is the limit.

# %%
rows = []
for key, dtype in [("t4", "float16"), ("l4", "bfloat16"), ("a100-80gb", "bfloat16"), ("h100-sxm", "bfloat16"),
                   ("h200", "bfloat16"), ("b200", "bfloat16")]:
    r = roofline.spec_roofline(specs.get(key), dtype)
    rows.append((r.label, r.peak_flops, r.peak_bw, r.ridge))
mine_roof = roofline.measured_roofline(gemms, streams, best_dtype)
rows.append((mine_roof.label, mine_roof.peak_flops, mine_roof.peak_bw, mine_roof.ridge))
for label, p, b, rdg in rows:
    print(f"{label:<44} {si(p, 'FLOP/s'):>13} {si(b, 'B/s'):>11}   ridge {rdg:7.1f} FLOP/B")

# %% [markdown]
# ## Exercise 1.5 — batching on your machine: predict, then measure
#
# At decode, a projection is a $(\text{batch} \times d) \cdot (d \times d)$ GEMM. Each step reads the weights one
# time, for any batch size, so the intensity increases with the batch. In roofline-core notebook 01 (Ex 1.5), you found
# the batch where that intensity crosses the ridge of an H100.
#
# In this notebook, you use *your* roofline to predict the **time** of the projection at several batch sizes. Then
# you measure it. Write `predicted_seconds(m, n, k, b, peak_flops, peak_bw)`. It returns the roofline time of an
# $(m \times k) \cdot (k \times n)$ GEMM with $b$-byte elements. That time is the longer of its compute time and its
# memory time (use `my_gemm_cost`).

# %% exercise
def predicted_seconds(m, n, k, b, peak_flops, peak_bw):
    ### BEGIN SOLUTION
    flops, nbytes = my_gemm_cost(m, n, k, b)
    return max(flops / peak_flops, nbytes / peak_bw)
    ### END SOLUTION

# %% check
assert abs(predicted_seconds(1, 8192, 8192, 2, 989e12, 3.35e12) - 134_250_496 / 3.35e12) < 1e-15   # decode: bytes
assert abs(predicted_seconds(4096, 4096, 4096, 2, 989e12, 3.35e12) - 2 * 4096 ** 3 / 989e12) < 1e-15  # prefill: FLOPs
dt = "float32" if not be.is_gpu else "float16"
d = 4096 if not be.is_gpu else 8192                   # weights well past the last-level cache / L2
batches = [1, 4, 16, 64, 256] if not be.is_gpu else [1, 16, 64, 256, 1024]
roof = roofline.measured_roofline(gemms, streams, dt)
bsz = 4 if dt == "float32" else 2
table = []
for B in batches:
    got = gemm.run_gemm(be, B, d, d, dt, repeats=3, min_time=0.02)
    table.append((B, predicted_seconds(B, d, d, bsz, roof.peak_flops, roof.peak_bw), got.seconds(), got.flops_per_s()))
print(f"(batch × {d})·({d} × {d}) {dt} on this machine — model from your roofline (ridge {roof.ridge:.1f} FLOP/B):")
print(f"{'batch':>6} {'predicted':>11} {'measured':>11} {'measured FLOP/s':>16}  bound")
for B, pred, meas, fl in table:
    print(f"{B:>6} {si(pred, 's'):>11} {si(meas, 's'):>11} {si(fl, 'FLOP/s'):>16}  {roof.bound(gemm_intensity(B, d, d, bsz))}")
assert table[-1][3] > table[0][3]                       # batching raised the throughput
print(f"✅ {batches[-1]}× the rows for {table[-1][2] / table[0][2]:.1f}× the time: throughput ×{table[-1][3] / table[0][3]:.0f}")

# %% [markdown]
# Compare the table with the model. While the projection is memory-bound, the time of the model almost does not change.
# The step streams the same $d^2$ weights for 1 row or for 16, so throughput increases with the batch for almost no
# cost. After the batch passes your ridge, the FLOPs become the limit, and the time increases with the batch. Where
# the measurement is different from the model, the model tells you something:
#
# * **far below** it: the weights came from cache, not from memory (a CPU with a large last-level cache: the model
#   counts DRAM bytes).
# * **far above** it at a small batch: the library runs below the roof there. For a GEMM with a few rows, CPU BLAS
#   libraries typically copy ("pack") the weight matrix into their own tile layout. That is one more read and write
#   of every weight. But batch 1 uses a GEMV path that streams the weights one time. GPU libraries have the same kind
#   of sudden change in rate between GEMV and GEMM kernels. This is why engines supply their own small-batch decode
#   kernels.
#
# **The H100 numbers, reconciled.** For $d = 8192$ at bf16, a standalone GEMM like the one that you measured must
# also read its $B \cdot d$ input and write its $B \cdot d$ output. Thus its intensity is
# $2Bd^2/((2Bd + d^2) \cdot 2)$, and it gets to the H100's ridge (295.2) at **batch 319**. Inside a model, the
# activations stay on chip between fused kernels (primer §3.1, §3.4). Only the weights stream in, the intensity is
# $2Bd^2/(d^2 \cdot 2) = B$, and the weight GEMMs become compute-bound at the ridge itself: **batch 296**,
# roofline-core's `gemm_crossover_batch()`. Both values are correct, because they count different bytes.

# %% check
h100 = specs.get("h100-sxm")
ridge_h100 = h100.ridge("bf16")
standalone = next(B for B in range(1, 5000) if gemm_intensity(B, 8192, 8192, 2) >= ridge_h100)
on_chip = next(B for B in range(1, 5000) if 2 * B * 8192 ** 2 / (8192 ** 2 * 2) >= ridge_h100)
assert (standalone, on_chip) == (319, 296) and on_chip == math.ceil(ridge_h100)
print(f"✅ H100 bf16, d = 8192: {standalone} counting activation bytes, {on_chip} with activations on chip")

# %% [markdown]
# Below the crossover, a decode step costs approximately the same for 1 sequence or for 300. The time is the weight
# bytes divided by bandwidth. That is why continuous batching
# ([layer 04](../../../../04-inference-engine/serving-engine/PRIMER.md)) is the single change with the largest
# effect on throughput. It is also why quantised weights (fewer bytes) make decode faster, but a faster tensor core
# does not.
#
# A real decode step also reads the KV cache of each sequence. Batching does not share that read across sequences
# ([KV cache primer](../../../../04-inference-engine/kv-cache/kv-cache-primer.md), primer §3.4).
# [Capacity planning](../../../../00-foundations/gpu-capacity-planning/PRIMER.md) uses these numbers to size a fleet.
#
# ## In a design review
#
# **The two-minute version.** "Every accelerator has two roofs: peak FLOP/s and memory bandwidth. Their ratio, the
# ridge point, is about 300 FLOPs per byte on an H100 at bf16, and hundreds on any current part. The intensity of a
# GEMM is $2mnk / ((mk + kn + mn) \cdot b)$.
#
# "A large prefill GEMM is far to the right of the ridge (compute-bound). A decode step at batch 1 is at ~1 FLOP per
# byte (memory-bound, at well under 1% of peak FLOP/s). Thus decode time is weight bytes divided by bandwidth until
# the batch gets to a few hundred. Up to that point, batching gives throughput that is nearly free.
#
# "Quantisation decreases the bytes that decode streams, and that is what makes decode faster. FP8 schemes for the
# weights *and* the activations also double the tensor-core rate. That matters only where the math is the limit:
# prefill and large batches. Weight-only schemes (W4A16) still compute at the bf16 rate. I did not trust the
# datasheet, and I measured the roofs on our hardware: a healthy GEMM gets 70–85% of *dense* peak."
#
# **Drills**
#
# 1. *We moved batch-1 decode from an A100 to an H100 and it got 1.6× faster, not 3×. Why?* Decode at batch 1 is
#    bandwidth-bound. The H100 SXM has ~1.6× the HBM bandwidth of the A100 80GB (3.35 TB/s against 2.04 TB/s). The
#    3× FLOP/s has no effect at ~1 FLOP/byte.
# 2. *Our bf16 GEMM benchmark shows 50% of the spec sheet. Is the GPU broken?* First, make sure that you compared
#    against the dense number. Often, the headline number is for 2:4 sparsity,
#    which gives 2× the dense one. Then examine the power cap
#    and the sustained clocks (`nvidia-smi`). Then examine the size, because small GEMMs cannot fill the SMs. A result
#    of 70–85% of dense is healthy.
# 3. *What batch size makes a $d = 8192$ projection compute-bound on an H100?* About 300. Inside a model, it is batch
#    296. There, the activations stay on chip and the intensity is equal to the batch. Thus the crossover is the
#    ridge (~295 FLOP/B). For a standalone GEMM that also reads and writes its activations, it is 319
#    ($R \cdot b \cdot d / (2(d - R \cdot b))$). A real decode step stays memory-bound for even longer.
#    KV-cache reads add bytes per sequence, and batching does not share them.
