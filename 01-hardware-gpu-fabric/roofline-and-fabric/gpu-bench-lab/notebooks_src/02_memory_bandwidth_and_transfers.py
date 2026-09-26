# %% [markdown]
# # 02 · Memory bandwidth and transfers
#
# **Tier:** T0 — STREAM, threads, fusion, the cache ladder and an α-β fit of `memcpy`, all measured
# on this CPU. **T1** (a CUDA GPU with PyTorch) — the same STREAM on HBM/GDDR, the GPU's cache
# ladder, and host↔device copies from pinned and pageable memory. Concepts: primer §2 "The roofline
# model", §4 "The memory hierarchy and why tiling/fusion win", §5 "Fabrics quantitatively" (the α-β
# model) — [`../../PRIMER.md`](../../PRIMER.md).
#
# **Predicted first in** [roofline-core notebook 02](../../roofline-core/notebooks/02_llm_inference_on_the_roofline.ipynb)
# (decode is a weight stream; fusion removes passes, Ex 2.6). Here you measure the bandwidth those
# predictions divide by, and explain the numbers you get.
#
# ## The one-minute version
#
# Most LLM inference time is spent moving bytes, so bandwidth is the number to get right, and it
# is easy to get wrong. Count bytes by a stated convention (STREAM's), use arrays far bigger than
# the caches, and report the best of several runs. One core cannot fill a memory bus — Little's law
# says bandwidth = bytes in flight ÷ latency — so CPU bandwidth scales with threads, and a GPU keeps
# tens of thousands of threads in flight. Fusion wins by deleting whole passes over memory. And
# every copy costs `α + n/β`: small copies are latency-bound, which is why engines batch them and
# keep their host buffers pinned — and *which* α you measured depends on whether the copies waited
# for each other.

# %%
import math

from IPython.display import Markdown, display

from gpubench import get_backend, inventory, membw, transfer
from gpubench.accounting import (NUMPY_TRIAD_PASSES, STREAM, STREAM_FORMULA, chain_cost, dtype_bytes, passes_cost,
                                 stream_cost, write_allocate_bytes)
from gpubench.measure import measure, si
from gpubench.report import bar_chart, measurements_markdown
from gpubench.specs import pcie_gbs
from gpubench.timing import fit_alpha_beta

QUICK = True
be = get_backend("auto")
info = be.describe()
cpus = inventory.usable_cpus()          # the affinity mask: a container may allow fewer than the host has
b = dtype_bytes(be.stream_dtype)
print(info["name"], "|", "caches:", info.get("caches") or f"L2 {si(info.get('l2_cache_bytes') or 0, 'B')}",
      "| usable CPUs:", cpus)

# %% [markdown]
# ## 1 · STREAM's four kernels, and how it counts
#
# John McCalpin's STREAM benchmark defines four loops over arrays `a`, `b`, `c` and counts the bytes
# of every array element the loop names — read or written once — and nothing else. Its rules: each
# array at least 4× the last-level cache — the *sum* of every last-level cache the run can use, so
# 4× both sockets' L3 on a two-socket server (`inventory.llc_total_bytes`) — so you measure memory,
# not cache; and report the **best** of several runs (the machine's capability).
#
# Two ways this lab's numpy STREAM differs from the reference binary, so you do not over-read it:
# its worker threads are not pinned to cores, and its arrays are first touched by one thread — on a
# multi-socket (NUMA) host every page then lives on one socket's memory, and the all-core number
# undercounts the machine. (STREAM's OpenMP build touches each slice from its own pinned thread.)

# %%
print(f"{'kernel':<7} {'loop':<12} {'reads':>6} {'writes':>7} {'FLOPs/elem':>11} {'bytes/elem (fp64)':>18}")
for k, (reads, writes, f) in STREAM.items():
    print(f"{k:<7} {STREAM_FORMULA[k]:<12} {reads:>6} {writes:>7} {f:>11} {stream_cost(k, 1, 8).bytes:>18.0f}")

# %% [markdown]
# ## Exercise 2.1 — count like STREAM, then like numpy
#
# Write `my_stream_bytes(kernel, n, b)` with STREAM's convention. numpy has no single-pass
# `b + q·c`, so the lab's numpy triad runs **two** passes, `a = q·c` then `a = a + b`; write
# `numpy_triad_bytes(n, b)`: the bytes those two passes actually move.

# %% exercise
def my_stream_bytes(kernel, n, b):
    ### BEGIN SOLUTION
    arrays = {"copy": 2, "scale": 2, "add": 3, "triad": 3}[kernel]
    return arrays * n * b
    ### END SOLUTION


def numpy_triad_bytes(n, b):
    ### BEGIN SOLUTION
    return (1 + 1) * n * b + (2 + 1) * n * b     # pass 1 reads c, writes a; pass 2 reads a and b, writes a
    ### END SOLUTION

# %% check
for k in STREAM:
    assert my_stream_bytes(k, 1000, 8) == stream_cost(k, 1000, 8).bytes, k
assert my_stream_bytes("triad", 1, 8) == 24                      # STREAM's "24 bytes per iteration"
assert numpy_triad_bytes(1000, 8) == passes_cost(1000, 8, NUMPY_TRIAD_PASSES).bytes   # what the lab charges
assert numpy_triad_bytes(1, 8) == 40
print("✅ STREAM counts 24 B per fp64 triad element; numpy's two-pass triad really moves 40")

# %% [markdown]
# ## 2 · STREAM on this machine
#
# One thread first, then every usable core (on a GPU, one kernel already uses every SM). The
# **moved** columns divide the bytes the code actually moved by the time; **STREAM convention**
# divides STREAM's count — they differ only for numpy's two-pass triad. That triad's *moved* rate is
# not a clean memory rate either: its second pass re-reads what the first just wrote (partly from
# cache) and updates it in place, which skips a write-allocate read (section 4). The roofline's
# slanted roof is therefore built from the single-pass kernels only (`membw.peak`).

# %%
n = membw.stream_elems(be)
llc = info.get("llc_total_bytes") or max((info.get("caches") or {"x": info.get("l2_cache_bytes") or 0}).values())
print(f"{n:,} elements per array ({si(n * b, 'B')} each; last-level cache, all instances: {si(llc, 'B')})")
one = membw.stream_suite(be, n, threads=1, repeats=3)
full = [] if be.is_gpu else membw.stream_suite(be, n, threads=cpus, repeats=3)
display(Markdown(measurements_markdown(one + full)))
print(f"slanted roof (fastest single-pass kernel): {si(membw.peak(one + full).bytes_per_s(), 'B/s')}")

# %% [markdown]
# ## 3 · One core cannot fill the bus: Little's law
#
# A memory system is a pipeline: **bandwidth = bytes in flight ÷ latency**. DRAM latency is ~100 ns
# whatever you do, so the only way to more bandwidth is more requests outstanding. One CPU core can
# track only ~10–20 cache-line misses at a time (plus what its prefetchers add), which caps a single
# thread well below the memory channels. More threads, more misses in flight — until the channels
# saturate. A GPU is this idea taken to its limit.

# %%
if be.is_gpu:
    print("On a GPU a single kernel already spreads over every SM with tens of thousands of threads in "
          "flight — that is how it keeps ~2 MB of requests outstanding. Thread scaling is a CPU experiment.")
    scaling = []
else:
    scaling = membw.thread_scaling(be, "add", n, repeats=3)
    print(bar_chart([(f"{m.params['threads']} thread(s)", m.bytes_per_s()) for m in scaling], "B/s"))

# %% [markdown]
# ## Exercise 2.2 — read your scaling curve with Little's law
#
# Two functions that turn the curve above into an explanation. `lines_in_flight(bandwidth,
# latency, line=64)`: how many cache lines must be outstanding to sustain `bandwidth` at `latency`.
# `saturation_threads(curve, frac=0.9)`: given `[(threads, bytes_per_s), ...]`, the smallest thread
# count that reaches `frac` of the best bandwidth in the curve — where adding cores stops paying.

# %% exercise
def lines_in_flight(bandwidth, latency, line=64):
    ### BEGIN SOLUTION
    return bandwidth * latency / line
    ### END SOLUTION


def saturation_threads(curve, frac=0.9):
    ### BEGIN SOLUTION
    best = max(bw for _, bw in curve)
    return min(t for t, bw in curve if bw >= frac * best)
    ### END SOLUTION

# %% check
assert abs(lines_in_flight(300e9, 100e-9) - 468.75) < 1e-9              # a DDR5 server socket: ~470 lines
assert abs(lines_in_flight(3.35e12, 600e-9, 32) - 62_812.5) < 1e-6      # an H100 at an illustrative 600 ns, 32 B sectors
assert saturation_threads([(1, 10e9), (2, 19e9), (4, 30e9), (8, 31e9)]) == 4
assert saturation_threads([(1, 10e9), (2, 10.5e9)]) == 1                 # one thread already saturates
if scaling:
    curve = [(m.params["threads"], m.bytes_per_s()) for m in scaling]
    lat = 90e-9                                                           # an assumed DRAM latency — verify for your CPU
    one_t, sat = curve[0][1], saturation_threads(curve)
    best = max(bw for _, bw in curve)
    print(f"one thread: {si(one_t, 'B/s')} ≈ {lines_in_flight(one_t, lat):.0f} lines in flight at {lat * 1e9:.0f} ns; "
          f"the best, {si(best, 'B/s')}, needs ≈ {lines_in_flight(best, lat):.0f}, reached by {sat} thread(s)")
print("✅ Little's law: bandwidth is bought with outstanding requests — cores on a CPU, warps on a GPU")

# %% [markdown]
# ## 4 · Write-allocate: the read nobody asked for
#
# On most CPUs a store to a cache line that is not in cache first **reads** the line
# (read-for-ownership), then writes it back later. So `c = a` really moves three arrays, not two,
# unless the code uses non-temporal (streaming) stores that bypass the cache — which glibc's
# `memcpy` does for large copies, and a numpy ufunc loop does not. STREAM does not count this read.
#
# ## Exercise 2.3 — the real DRAM traffic
#
# Write `dram_bytes(kernel, n, b, write_allocate)`: STREAM's bytes plus, when `write_allocate`, one
# extra read of every array the kernel writes.

# %% exercise
def dram_bytes(kernel, n, b, write_allocate=True):
    ### BEGIN SOLUTION
    reads, writes, _ = STREAM[kernel]
    return (reads + writes + (writes if write_allocate else 0)) * n * b
    ### END SOLUTION

# %% check
assert dram_bytes("copy", 1, 8) == 24 and dram_bytes("copy", 1, 8, write_allocate=False) == 16
assert dram_bytes("triad", 1, 8) == 32 and dram_bytes("add", 1000, 4) == 16_000
for k in STREAM:
    assert dram_bytes(k, 10, 8) == stream_cost(k, 10, 8).bytes + write_allocate_bytes(k, 10, 8)
print("✅ with write-allocate, copy moves 1.5× and triad 1.33× what STREAM reports")

# %%
if not be.is_gpu:
    rate = {m.op.split(".")[1]: m.bytes_per_s() for m in full}
    print(f"all threads: copy (memcpy) {si(rate['copy'], 'B/s')} vs scale (ufunc) {si(rate['scale'], 'B/s')}"
          f" — same STREAM bytes; ratio scale/copy = {rate['scale'] / rate['copy']:.2f}")
    print("If scale runs near 2/3 of copy, its stores pay the write-allocate read and memcpy's streaming stores"
          " do not. Near 1.0, something else limits both (too few misses in flight, a noisy neighbour). "
          "The model tells you what to look for; the measurement decides.")

# %% [markdown]
# ## 5 · Fusion: delete whole passes over memory
#
# Apply `k` cheap elementwise ops to a large array. **Unfused**, each op is its own pass: read the
# array, write it back, `k` times. **Fused**, the array is processed in cache-sized blocks and all
# `k` ops run on a block while it sits in cache: one read and one write of DRAM. Same FLOPs, `k`×
# fewer DRAM bytes. It is exactly what a fused GPU kernel does with registers and shared memory,
# and what [FlashAttention](../../../../04-inference-engine/flash-attention/flash-attention-primer.md)
# does to attention (primer §4). The experiment runs on the CPU even when a GPU is present — GPU
# kernel fusion is layer 02's lab.
#
# ## Exercise 2.4 — predict it first
#
# Write `chain_dram_bytes(n, b, k, fused)` and `speedup_bound(k)`: the speedup fusion would give if
# both versions ran at the same memory bandwidth.

# %% exercise
def chain_dram_bytes(n, b, k, fused):
    ### BEGIN SOLUTION
    passes = 1 if fused else k
    return 2 * passes * n * b
    ### END SOLUTION


def speedup_bound(k):
    ### BEGIN SOLUTION
    return chain_dram_bytes(1, 1, k, False) / chain_dram_bytes(1, 1, k, True)
    ### END SOLUTION

# %% check
assert chain_dram_bytes(1000, 8, 8, fused=False) == 128_000 and chain_dram_bytes(1000, 8, 8, fused=True) == 16_000
assert chain_dram_bytes(10, 4, 3, False) == chain_cost(10, 4, 3, False).bytes
assert speedup_bound(8) == 8
print("✅ fusion keeps the FLOPs and removes (k−1) of k round trips to DRAM")

# %%
cpu = be if not be.is_gpu else get_backend("numpy", verbose=False)
k = 8
nf = membw.stream_elems(cpu)
unfused, fused = membw.fusion(cpu, n=nf, k=k, repeats=3)
display(Markdown(measurements_markdown([unfused, fused])))
print(f"measured speedup {unfused.seconds() / fused.seconds():.2f}× (bound {speedup_bound(k):.0f}×); "
      f"unfused ran at {si(unfused.bytes_per_s(), 'B/s')} of DRAM traffic")

# %% [markdown]
# The measured speedup falls short of `k` because the fused version is not free: each block pays
# numpy's per-call overhead `k` times, and its `k` passes still stream the block through L1/L2 —
# a faster memory, but a memory. A compiled fused kernel keeps the intermediate in *registers*
# and pays neither cost, which is why it gets much closer to the bound. The next section measures
# both costs, and then puts a number on the gap.
#
# ## 6 · The cache ladder
#
# Bandwidth as a function of working-set size, with an in-place `x *= 1` (one read and one write
# per element). Each plateau is a level of the hierarchy; the left end is not a cache at all but the
# fixed cost of a call (numpy dispatch on a CPU, a kernel launch on a GPU).

# %%
lad = membw.cache_ladder(be, repeats=3)
print(bar_chart([(si(m.params["working_set"], "B"), m.bytes_per_s()) for m in lad], "B/s"))
print("\ncache sizes reported by the system:", info.get("caches") or f"L2 {si(info.get('l2_cache_bytes') or 0, 'B')}")
print(f"smallest working set: {si(lad[0].seconds(), 's')} per call — that is mostly fixed cost, not bytes")

# %% [markdown]
# Read the knees against the cache sizes: the rate drops where the working set outgrows a level
# (an in-place update touches the set once per call, so a level holds a working set about its own
# size). **Back to the fusion gap**, with the ladder's numbers: the fused chain made `k` numpy calls
# per block, each on a block that sits in cache, so its time should be about
# `blocks × k × (time of one ladder call at the block size)`.

# %%
if not be.is_gpu:
    block_bytes = fused.extras["block_bytes"]
    rung = min(lad, key=lambda m: abs(m.params["working_set"] - block_bytes))
    blocks = math.ceil(nf * 8 / block_bytes)
    model = blocks * k * rung.seconds()
    print(f"{blocks} blocks × {k} calls × {si(rung.seconds(), 's')} per call on {si(rung.params['working_set'], 'B')} "
          f"= {si(model, 's')} predicted; fused measured {si(fused.seconds(), 's')}; "
          f"one DRAM pass alone would take {si(2 * nf * 8 / membw.peak(one + full).bytes_per_s(), 's')}")
    print("The fused version is bound by in-cache passes and per-call cost, not by DRAM — exactly what a "
          "compiled kernel that keeps the intermediates in registers removes.")

# %% [markdown]
# ## 7 · Every copy costs α + n/β
#
# Time a copy over a range of sizes and fit `t(n) = α + n/β`: α is the fixed cost of *any* copy
# (a call, a descriptor, a launch), β the bandwidth of the slowest link on the path, and `n½ = α·β`
# the size at which you get half of β (primer §5). On T0 we time host `memcpy` of cache-cold data;
# the method is the same one you would use on PCIe, NVLink or a network.
#
# **Which α?** It depends on how the copies were timed. The sweep issues copies back to back and
# synchronises only at the ends. A synchronous copy (numpy's `memcpy`, a pageable host→device copy)
# finishes before the next starts, so its α is the latency of one copy. An asynchronous one (a
# *pinned* host→device copy, `non_blocking=True`) is queued while the previous one runs, so its
# fixed cost overlaps the transfer and the fit's α is an **issue cost**, often several times smaller
# than the time until one copy's bytes have arrived. On a GPU the lab therefore also runs a *latency*
# sweep — one synchronised copy per sample — and reports both. Use the pipelined α for a stream of
# independent copies, the latency α when each copy waits for the last (every step of a ring
# all-reduce, notebook 03).
#
# ## Exercise 2.5 — fit α and β
#
# Write `my_fit(sizes, times)` returning `(alpha, beta)` by ordinary least squares on
# `t = α + s·n` (slope `s = 1/β`). No libraries needed: slope = cov(n, t) / var(n).

# %% exercise
def my_fit(sizes, times):
    ### BEGIN SOLUTION
    k = len(sizes)
    mn, mt = sum(sizes) / k, sum(times) / k
    cov = sum((x - mn) * (y - mt) for x, y in zip(sizes, times))
    var = sum((x - mn) ** 2 for x in sizes)
    slope = cov / var
    return mt - slope * mn, 1.0 / slope
    ### END SOLUTION

# %% check
sizes = [2 ** e for e in range(12, 28, 2)]
exact = [5e-6 + s / 20e9 for s in sizes]                 # α = 5 µs, β = 20 GB/s
alpha, beta = my_fit(sizes, exact)
assert abs(alpha - 5e-6) < 1e-9 and abs(beta - 20e9) / 20e9 < 1e-6
lib = fit_alpha_beta(sizes, exact)
assert abs(lib.alpha - alpha) < 1e-9 and abs(lib.n_half - 100_000) < 1
print(f"✅ α = {si(alpha, 's')}, β = {si(beta, 'B/s')}, n½ = α·β = {si(alpha * beta, 'B')}")

# %%
sweep = transfer.sizes(4 << 10, (64 << 20) if QUICK else (1 << 30), 4)
if be.is_gpu:
    copies = transfer.hostdevice_sweep(be, sweep, repeats=3)
    copies += transfer.hostdevice_sweep(be, transfer.sizes(4 << 10, 4 << 20, 4), pinned=(True,), latency=True,
                                        repeats=20, min_time=0.0)
else:
    copies = transfer.memcpy_sweep(be, sweep, repeats=3)
fits = {}
for key, series in transfer.series(copies).items():
    ab = fits[key] = transfer.fit(series)
    ols_alpha, ols_beta = my_fit([m.params["nbytes"] for m in series], [m.seconds("median") for m in series])
    equiv = f" (STREAM-convention equivalent {si(2 * ab.beta, 'B/s')})" if key[0] == "memcpy" else ""
    print(f"{transfer.series_label(key):<22} α = {si(ab.alpha, 's'):>9}   β = {si(ab.beta, 'B/s'):>10}{equiv}"
          f"   n½ = {si(ab.n_half, 'B'):>9}   r² {ab.r2:.3f}")
    print(f"{'':<22} your plain least squares: α = {si(ols_alpha, 's')}, β = {si(ols_beta, 'B/s')}")
    for m in series:
        print(f"   {si(m.params['nbytes'], 'B'):>9}: measured {si(m.bytes_per_s('median'), 'B/s'):>10}"
              f"   model {si(ab.bandwidth(m.params['nbytes']), 'B/s'):>10}")

# %% [markdown]
# Three things to read here. **Your fit vs the library's.** Plain least squares minimises absolute
# error, so the largest copies — thousands of times longer than the smallest — decide everything
# and α comes out as noise (sometimes negative). The library weights each point by `1/t`, so the
# small sizes that pin α count as much as the large ones that pin β. **β vs STREAM copy.** A transfer
# counts each delivered byte once; STREAM's copy counts the read and the write. So a memcpy β of
# 5 GB/s is the same memory traffic as a STREAM copy of 10 GB/s — compare the "STREAM-convention
# equivalent" with the one-thread copy row of section 2, not β itself. **Model vs measured.** Each
# copy reads bytes that are not in cache (the op cycles through a pool 4× the last-level cache), like
# a real transfer. α-β assumes **one** bottleneck, and a CPU copy has several regimes: glibc's
# `memcpy` switches to non-temporal stores once a copy is a sizeable fraction of the last-level
# cache, which skips the write-allocate read of section 4 — so big copies can run *faster* than the
# fit. Over PCIe the link is the bottleneck at every size above a few KB, and the fit is much tighter.
#
# ## Exercise 2.6 — predict a copy, then measure it
#
# How big must a copy be to get a fraction `f` of the link? Solve `n / t(n) = f·β` for `n` with
# `t(n) = α + n/β` and write `size_for_fraction(alpha, beta, f)`. (At `f = ½` it is `n½`.) The check
# then predicts, from *your* fit, the size that reaches 80% of β, measures a copy of that size, and
# compares.

# %% exercise
def size_for_fraction(alpha, beta, f):
    ### BEGIN SOLUTION
    return f * alpha * beta / (1 - f)
    ### END SOLUTION

# %% check
assert abs(size_for_fraction(5e-6, 20e9, 0.5) - 100_000) < 1e-6          # n½ = α·β
assert abs(size_for_fraction(10e-6, 25e9, 0.9) - 2.25e6) < 1e-3          # 90% of a PCIe link: 9·α·β
key = next(kk for kk in fits if kk[0] in ("memcpy", "h2d") and kk[1] in (None, True) and kk[2] != "latency")
ab = fits[key]
n80 = int(min(max(size_for_fraction(ab.alpha, ab.beta, 0.8), 4096), 256 << 20))
op = be.make_memcpy(n80) if not be.is_gpu else be.make_transfer(n80, "h2d", True)
got = measure(be, op, "copy", {"nbytes": n80}, repeats=5, min_time=0.01)
pred = ab.time(n80)
print(f"{transfer.series_label(key)}: predicted {si(pred, 's')} for {si(n80, 'B')} "
      f"({si(n80 / pred, 'B/s')}, 80% of β); measured {si(got.seconds('median'), 's')} "
      f"({si(got.bytes_per_s('median'), 'B/s')})")
assert 0.2 < got.seconds("median") / pred < 5, "the α-β model is off by more than 5×: re-run on a quiet machine"
print("✅ the fit predicts a copy it never saw — within the noise of a shared machine")

# %% [markdown]
# ## 8 · Host ↔ device: pinned vs pageable (T1)
#
# A GPU's copy engine DMAs from host memory it can address directly: **pinned** (page-locked)
# memory. From ordinary **pageable** memory the driver first copies each chunk into a pinned
# bounce buffer, so the copy is slower and blocks the host — it cannot overlap with compute. The
# link's theoretical rate per direction is `GT/s × lanes × encoding ÷ 8` (`specs.pcie_gbs`): Gen3
# runs 8 GT/s per lane, Gen4 16, Gen5 32, all with 128b/130b encoding; packet headers and flow
# control take another 10–20%, so a good pinned copy lands at 80–90% of it.

# %%
print("PCIe per direction (theoretical, not a measurement) and 16 GB of weights at 85% of it:")
for g, w in ((3, 16), (4, 16), (4, 8), (5, 16)):
    print(f"   Gen{g} x{w:<2}: {pcie_gbs(g, w):5.1f} GB/s → {16e9 / (0.85 * pcie_gbs(g, w) * 1e9):5.2f} s")
if be.is_gpu:
    link = transfer.host_link(info["name"])
    exp = transfer.pcie_expectation(link["gen"], link["width"])
    best = transfer.best_bandwidth([m for m in copies if m.params.get("pinned") and m.params.get("mode") != "latency"])
    print(f"\nthis GPU: {exp['link']} ({link['source']}): theoretical {exp['theoretical_gbs']:.1f} GB/s per direction; "
          f"best pinned copy {si(best.bytes_per_s(), 'B/s')} ({best.bytes_per_s() / 1e9 / exp['theoretical_gbs']:.0%})")
    rows, _ = inventory.query_gpus()
    for f in inventory.health(rows or []):
        print(f)
else:
    print("\nNo GPU here, so no host↔device copies were measured. On a GPU runtime (Colab: Runtime → Change"
          " runtime type → T4 GPU) this notebook sweeps h2d/d2h × pinned/pageable, fits α-β to each, and adds a"
          " latency sweep of pinned copies. Anywhere: python -m gpubench run --backend torch --suite transfer")

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "Bandwidth is the number that decides decode speed, so we measure it
# with STREAM's rules: arrays 4× the last-level cache, bytes counted by STREAM's convention, best
# of N — and we say when an implementation moves more than STREAM counts. On a CPU one thread
# cannot fill the bus (Little's law: bandwidth = bytes in flight ÷ latency), so bandwidth scales
# with cores; a GPU keeps megabytes in flight across its SMs. Fusion is the biggest software lever
# on a memory-bound block: it keeps the FLOPs and deletes passes. For copies we fit `α + n/β` and
# design so transfers are large, batched, pinned and asynchronous; below `n½` you are paying
# latency, not bandwidth — and we say which α we measured: back-to-back async copies give an issue
# cost, a synchronised copy gives the latency a dependent step pays."
#
# **Drills**
#
# 1. *Our 4 KB host→device copies of token IDs run at a few hundred MB/s on a 32 GB/s link. Bug?* —
#    No: they are α-bound. If each copy is waited for, a latency of ~10 µs (illustrative) caps a 4 KB
#    copy at ≤ 0.4 GB/s, and `n½` is hundreds of KB. Batch them, keep the buffer pinned, issue them
#    asynchronously — then consecutive copies overlap their fixed costs and only the issue cost remains.
# 2. *Why does fusing bias + GELU + residual speed up a memory-bound layer?* — Each unfused op
#    reads and writes the activation from HBM; fused, it is read once and written once. Time
#    follows bytes, so `k` passes become one.
# 3. *A vendor's STREAM number is 20% above ours on the same CPU. Who is wrong?* — Possibly
#    nobody: check thread count and pinning (and NUMA placement on a two-socket box), array size vs
#    the *total* last-level cache, and whether their build uses non-temporal stores (no
#    write-allocate read). State the convention with the number.
