# %% [markdown]
# # 01 · Measure your roofline
#
# **Tier:** T0 — runs on any CPU (laptop, Colab, CI) and measures *that* CPU with numpy. The same
# cells become **T1** when PyTorch can see a CUDA GPU (Colab/Kaggle T4, an L4, any rented GPU):
# the backend switches to torch and every number is the GPU's. Concepts: primer §1 "Spec-sheet
# literacy" and §2 "The roofline model" ([`../../PRIMER.md`](../../PRIMER.md)).
#
# ## The one-minute version
#
# A roofline is two numbers you can measure: the most FLOP/s any kernel reaches (a big GEMM) and
# the most bytes/s any kernel moves (a STREAM kernel). Their ratio, the **ridge point**, is how
# many FLOPs an operation must perform per byte it moves before arithmetic, not memory, is what
# limits it. In this notebook you measure both on the machine in front of you, place GEMMs of
# different sizes and dtypes on *your* roofline, set your ridge beside a datacenter GPU's, and
# compute the decode batch size at which an H100 stops being memory-bound — the number behind
# "batching is nearly free".

# %%
import os
import time

from IPython.display import Markdown, display

from gpubench import gemm, get_backend, measure, membw, roofline, specs
from gpubench.accounting import gemm_cost, gemm_intensity
from gpubench.measure import si
from gpubench.report import measurements_markdown

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
# Write the machine down before measuring it: a number without its hardware is useless. For a
# CPU the flat roof is **cores × clock × vector lanes × 2 (an FMA is a multiply and an add) × FMA
# units**, and numpy reaches it through its BLAS library (OpenBLAS or MKL, one thread per core).
# The slanted roof is the memory channels. For a GPU the flat roof is SMs × clock × tensor-core
# rate for the dtype, and the slanted roof is HBM (or GDDR on an L4/T4).
#
# ## 2 · Counting work: FLOPs and bytes
#
# A GEMM `C(m×n) = A(m×k)·B(k×n)` does `m·n·k` multiply-adds, which is **`2mnk` FLOPs**. Its
# **compulsory traffic** is reading A and B once and writing C once: `(mk + kn + mn)·b` bytes for
# `b`-byte elements. A kernel moves at least that (more if its tiles do not fit on chip), so the
# ratio is the best-case **arithmetic intensity** (primer §2). Every shape below is an LLM matmul:

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
# Decode at batch 1 does about **one FLOP per byte** at bf16 (two FLOPs per weight, two bytes per
# weight): whatever the hardware, it is limited by how fast the weights stream in. Batching reuses
# each weight for every sequence in the batch, so intensity grows roughly with the batch size.
#
# ## Exercise 1.1 — count a GEMM yourself
#
# Write `my_gemm_cost(m, n, k, b)` returning `(flops, bytes)` with the conventions above.

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
# The first call of anything is slow: fresh buffers page-fault, the BLAS thread pool (or cuBLAS)
# starts up and picks a kernel, clocks ramp. So a benchmark warms up, then repeats the call until
# each sample lasts long enough to swamp timer resolution, takes several samples, and reports
# **best** (what the machine can do) and **median** (what you would typically see). On a GPU it
# must also wait for the asynchronous work to finish — the torch backend uses CUDA events.

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
# Square GEMMs over a range of sizes and every dtype this backend supports. Expect FLOP/s to rise
# with size (small GEMMs cannot keep every core/SM busy, and fixed costs are not amortised), and
# each halving of the element width to help: twice the SIMD lanes on a CPU, the next tensor-core
# rate on a GPU. numpy's float16 has **no BLAS path** at all — it is an emulated loop, and the
# number you are about to see is what "no hardware support for this dtype" means.

# %%
skipped = []
gemms = gemm.sweep(be, quick=QUICK, skipped=skipped)
display(Markdown(measurements_markdown(gemms)))
for s in skipped:
    print("skipped:", s)

# %% [markdown]
# ## 5 · The bandwidth roof
#
# The slanted roof is the fastest any kernel moves bytes to and from main memory. STREAM's
# kernels over arrays much larger than the last-level cache measure it (notebook 02 takes this
# apart); on a CPU we use every core, because one core cannot fill the memory bus.

# %%
threads = 1 if be.is_gpu else (os.cpu_count() or 1)
n = membw.stream_elems(be)
streams = membw.stream_suite(be, n, threads=threads, repeats=3)
display(Markdown(measurements_markdown(streams)))
bw = membw.peak(streams)
print(f"slanted roof: {si(bw.bytes_per_s(), 'B/s')} ({bw.op}, {threads} thread(s), {si(bw.cost.bytes, 'B')} per call)")

# %% [markdown]
# ## Exercise 1.2 — the roofline itself
#
# Write `my_attainable(intensity, peak_flops, peak_bw)` (the roofline) and
# `my_ridge(peak_flops, peak_bw)` (the intensity where the two roofs meet).

# %% exercise
def my_attainable(intensity, peak_flops, peak_bw):
    ### BEGIN SOLUTION
    return min(peak_flops, intensity * peak_bw)
    ### END SOLUTION


def my_ridge(peak_flops, peak_bw):
    ### BEGIN SOLUTION
    return peak_flops / peak_bw
    ### END SOLUTION

# %% check
assert my_attainable(1.0, 989e12, 3.35e12) == 3.35e12           # decode at batch 1 on an H100: bandwidth-bound
assert my_attainable(1000.0, 989e12, 3.35e12) == 989e12         # a big GEMM: compute-bound
assert abs(my_ridge(989e12, 3.35e12) - 295.2) < 0.1
for i in (0.5, 4, 30, 300, 3000):
    assert my_attainable(i, 2e12, 5e10) == roofline.attainable(i, 2e12, 5e10)
print("✅ roofline and ridge — an H100 needs ~295 FLOP per byte of HBM traffic to be compute-bound at bf16")

# %% [markdown]
# ## 6 · Your roofline
#
# Flat roof: the fastest GEMM of a dtype. Slanted roof: the fastest byte-mover. The chart is
# log-log; each `o` is one of your GEMMs at its compulsory intensity. A small GEMM can sit *above*
# the slanted roof: its operands stay in cache between calls, so it is not really moving the
# compulsory bytes from DRAM — the roofline counts DRAM (or HBM) traffic only.

# %%
main = [d for d in dict.fromkeys(m.params["dtype"] for m in gemms) if not (be.name == "numpy" and d == "float16")]
roofs = [roofline.measured_roofline(gemms, streams, d) for d in main]
best_dtype = max(main, key=lambda d: gemm.best(gemms, d).flops_per_s())
points = [("o", m.intensity, m.flops_per_s()) for m in gemms if m.params["dtype"] == best_dtype]
print(roofline.ascii_plot(roofs, points))
print(f"\n'o' = {best_dtype} GEMMs of sizes {[m.params['m'] for m in gemms if m.params['dtype'] == best_dtype]}")

# %% [markdown]
# ## Exercise 1.3 — which of your GEMMs were compute-bound?
#
# For each GEMM of `best_dtype`, return `(size, bound, efficiency)`: `bound` is `"memory"` or
# `"compute"` on *your* roofline, and `efficiency` is achieved FLOP/s (best sample) divided by the
# attainable FLOP/s at that GEMM's intensity. Use your functions from 1.2.

# %% exercise
def classify(measurements, peak_flops, peak_bw):
    out = []
    ### BEGIN SOLUTION
    for m in measurements:
        att = my_attainable(m.intensity, peak_flops, peak_bw)
        kind = "compute" if m.intensity >= my_ridge(peak_flops, peak_bw) else "memory"
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
# Read the result two ways. First, **where the ridge is**: a CPU has little compute per byte of
# bandwidth, so its ridge sits around 5–20 FLOP/B and almost every GEMM lands right of it; a GPU's
# ridge is 150–400, so the same small GEMMs land left of it (memory-bound) on a GPU. Second, **how
# far below its roof** each GEMM runs: a small GEMM that is "compute-bound" yet reaches a third of
# its roof is limited by something the roofline does not model — thread start-up, kernel launch,
# too few tiles to occupy every core or SM. That fixed cost per call is the α of notebook 02.
#
# (On a GPU run, the flat roof of each dtype is simply your fastest GEMM of it; compare it with the
# spec in the next section before trusting it.)

# %% [markdown]
# ## 7 · Measured vs spec
#
# For a GPU the spec sheet gives dense peaks (the lab's `specs` table, dated — verify). For a CPU
# you can *compute* one from the instruction set: `cores × GHz × lanes × 2 × FMA units`, with
# `lanes = vector bits / (8 × bytes per element)`. It is an estimate — the FMA-unit count is an
# assumption, heavy AVX-512 code often runs below the nominal clock, and a cloud vCPU may be a
# hyperthread, not a core.
#
# ## Exercise 1.4 — a CPU's peak from its ISA
#
# Write `cpu_peak(cores, ghz, simd_bits, fma_units, dtype_bytes)` in FLOP/s.

# %% exercise
def cpu_peak(cores, ghz, simd_bits, fma_units, dtype_bytes):
    ### BEGIN SOLUTION
    lanes = simd_bits // (8 * dtype_bytes)
    return cores * ghz * 1e9 * lanes * 2 * fma_units
    ### END SOLUTION

# %% check
assert cpu_peak(4, 2.8, 512, 2, 8) == 358.4e9           # 4 cores of AVX-512, fp64
assert cpu_peak(4, 2.8, 512, 2, 4) == 716.8e9           # fp32: twice the lanes
assert cpu_peak(8, 3.2, 128, 4, 4) == specs.cpu_peak_flops(8, 3.2, 128, 4, 4)   # an Arm core with 4 NEON FMA pipes
print("✅ cpu_peak — halving the element width doubles the lanes, and the peak")

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
else:
    cores, ghz = info.get("usable_cpus") or info.get("logical_cpus"), info.get("ghz_nominal")
    for d in main:
        got = gemm.best(gemms, d).flops_per_s()
        if ghz:
            est = cpu_peak(cores, ghz, info["simd_bits"], 2, 8 if d == "float64" else 4)
            print(f"{d:>8}: measured {si(got, 'FLOP/s'):>13}   ISA estimate {si(est, 'FLOP/s'):>13}"
                  f" ({cores} cpus × {ghz} GHz × {info['simd_bits']}-bit × 2 FMA units)   {got / est:5.1%}")
        else:
            print(f"{d:>8}: measured {si(got, 'FLOP/s')}; no clock in the CPU model name, so no estimate")

# %% [markdown]
# 70–85% of a GPU's *dense* spec is a healthy GEMM; if you compared against a number with an
# asterisk (2:4 sparsity) you would conclude you get 40% and go hunting for a bug that is not there.
# A 70 W card (T4, L4) sustains lower clocks than its boost-clock spec under tensor load.
#
# ## 8 · Your ridge next to datacenter GPUs
#
# The spec rooflines (dense, dated — verify) at the dtype an LLM serves in, next to what you
# measured. Every one of them needs hundreds of FLOPs per byte before compute is the limit.

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
# ## Exercise 1.5 — when does decode stop being memory-bound?
#
# At decode, a projection is a `(batch × d)·(d × d)` GEMM: the weights are read once per step
# whatever the batch, so intensity grows with the batch. Write `crossover_batch(d, peak_flops,
# peak_bw, b)`: the **smallest integer batch** whose intensity reaches the ridge (or `None` if no
# batch ever does). This ignores KV-cache reads, which grow with the batch — primer §3 adds them.

# %% exercise
def crossover_batch(d, peak_flops, peak_bw, b):
    ### BEGIN SOLUTION
    ridge_point = peak_flops / peak_bw
    if d <= ridge_point * b:            # intensity 2Bd/((2B + d)·b) approaches d/b from below
        return None
    batch = max(1, int(ridge_point * b * d / (2 * (d - ridge_point * b))))
    while gemm_intensity(batch, d, d, b) < ridge_point:
        batch += 1
    while batch > 1 and gemm_intensity(batch - 1, d, d, b) >= ridge_point:
        batch -= 1
    return batch
    ### END SOLUTION

# %% check
h100 = specs.get("h100-sxm")
assert crossover_batch(8192, h100.peak_flops("bf16"), h100.mem_bw, 2) == 319
l4 = specs.get("l4")
assert crossover_batch(8192, l4.peak_flops("bf16"), l4.mem_bw, 2) == 448
assert crossover_batch(256, h100.peak_flops("bf16"), h100.mem_bw, 2) is None      # a tiny model never gets there
yours = crossover_batch(8192, mine_roof.peak_flops, mine_roof.peak_bw, 2)
print(f"✅ H100 bf16: batch {crossover_batch(8192, h100.peak_flops('bf16'), h100.mem_bw, 2)}; "
      f"L4: {crossover_batch(8192, l4.peak_flops('bf16'), l4.mem_bw, 2)}; this machine: {yours}")

# %% [markdown]
# Below the crossover, a decode step costs about the same whether it carries 1 sequence or 300:
# the time is the weight bytes divided by bandwidth. That is why continuous batching (layer 04) is
# the biggest single throughput lever, and why quantising weights (fewer bytes) speeds decode up
# while a faster tensor core does not.
#
# ## In a design review
#
# **The two-minute version.** "Every accelerator has two roofs: peak FLOP/s and memory bandwidth.
# Their ratio — the ridge point — is about 300 FLOPs per byte on an H100 at bf16 and hundreds on
# anything current. A GEMM's intensity is `2mnk / ((mk + kn + mn)·b)`: a large prefill GEMM sits
# far right of the ridge (compute-bound), a decode step at batch 1 sits at ~1 FLOP per byte
# (memory-bound, running at well under 1% of peak FLOP/s). So decode time is weight bytes divided
# by bandwidth until the batch reaches a few hundred; batching is nearly free throughput up to
# there, and quantisation pays twice — fewer bytes, faster tensor cores. I measured the roofs on
# our hardware rather than trusting the datasheet: a healthy GEMM reaches 70–85% of *dense* peak."
#
# **Drills**
#
# 1. *We moved batch-1 decode from an A100 to an H100 and it got 1.6× faster, not 3×. Why?* —
#    Decode at batch 1 is bandwidth-bound; the H100 SXM has ~1.6× the A100 80GB's HBM bandwidth
#    (3.35 vs 2.04 TB/s). The 3× FLOP/s never comes into play at ~1 FLOP/byte.
# 2. *Our bf16 GEMM benchmark shows 50% of the spec sheet. Is the GPU broken?* — First check that
#    you compared against the dense number (the headline is often 2:4 sparse — 2× the dense one),
#    then the power cap and sustained clocks (`nvidia-smi`), then the size (small GEMMs cannot fill
#    the SMs). 70–85% of dense is healthy.
# 3. *What batch size makes a d=8192 projection compute-bound on an H100?* — About 320 (the
#    ridge, ~295 FLOP/B, is reached at batch ≈ `R·b·d / (2(d − R·b))`). A real decode step stays
#    memory-bound even longer: KV-cache reads add bytes per sequence that batching does not amortise.
