# %% [markdown]
# # 02 · Memory bandwidth and transfers
#
# **Tier:** T0 — STREAM, threads, fusion, the cache ladder and an α-β fit of `memcpy`, all measured
# on this CPU. **T1** (a CUDA GPU with PyTorch) — the same STREAM on HBM/GDDR, the GPU's cache
# ladder, and host↔device copies from pinned and pageable memory. Concepts: primer §2 "The roofline
# model", §4 "The memory hierarchy and why tiling/fusion win", §5 "Fabrics quantitatively" (the α-β
# model) — [`../../PRIMER.md`](../../PRIMER.md).
#
# ## The one-minute version
#
# Most LLM inference time is spent moving bytes, so bandwidth is the number to get right, and it
# is easy to get wrong. Count bytes by a stated convention (STREAM's), use arrays far bigger than
# the caches, and report the best of several runs. One core cannot fill a memory bus — Little's law
# says bandwidth = bytes in flight ÷ latency — so CPU bandwidth scales with threads, and a GPU keeps
# tens of thousands of threads in flight. Fusion wins by deleting whole passes over memory. And
# every copy costs `α + n/β`: small copies are latency-bound, which is why engines batch them and
# keep their host buffers pinned.

# %%
import math
import os

from IPython.display import Markdown, display

from gpubench import get_backend, inventory, membw, transfer
from gpubench.accounting import (NUMPY_TRIAD_PASSES, STREAM, STREAM_FORMULA, chain_cost, dtype_bytes, passes_cost,
                                 stream_cost, write_allocate_bytes)
from gpubench.measure import si
from gpubench.report import bar_chart, measurements_markdown
from gpubench.specs import pcie_gbs
from gpubench.timing import fit_alpha_beta

QUICK = True
be = get_backend("auto")
info = be.describe()
cpus = os.cpu_count() or 1
b = dtype_bytes(be.stream_dtype)
print(info["name"], "|", "caches:", info.get("caches") or f"L2 {si(info.get('l2_cache_bytes') or 0, 'B')}")

# %% [markdown]
# ## 1 · STREAM's four kernels, and how it counts
#
# John McCalpin's STREAM benchmark defines four loops over arrays `a`, `b`, `c` and counts the bytes
# of every array element the loop names — read or written once — and nothing else. Its rules: each
# array at least 4× the last-level cache (so you measure memory, not cache), and report the **best**
# of several runs (the machine's capability).

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
# One thread first, then every core (on a GPU, one kernel already uses every SM). The **moved**
# columns divide the bytes the code actually moved by the time; **STREAM convention** divides
# STREAM's count — they differ only for numpy's two-pass triad.

# %%
n = membw.stream_elems(be)
print(f"{n:,} elements per array ({si(n * b, 'B')} each; last-level cache "
      f"{si(max((info.get('caches') or {'x': info.get('l2_cache_bytes') or 0}).values()), 'B')})")
one = membw.stream_suite(be, n, threads=1, repeats=3)
full = [] if be.is_gpu else membw.stream_suite(be, n, threads=cpus, repeats=3)
display(Markdown(measurements_markdown(one + full)))

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
    one_add = scaling[0].bytes_per_s()
    print(f"\none thread moves {si(one_add, 'B/s')}; at an assumed ~90 ns DRAM latency (typical — verify for your CPU)"
          f" that is {one_add * 90e-9:,.0f} bytes ≈ {one_add * 90e-9 / 64:.0f} cache lines in flight")

# %% [markdown]
# ## Exercise 2.2 — how much must be in flight?
#
# Write `bytes_in_flight(bandwidth, latency)` (Little's law) and `cores_needed(bandwidth, latency,
# misses_per_core, line)`: the smallest whole number of cores that keeps enough cache lines in
# flight, if each core sustains `misses_per_core` outstanding lines of `line` bytes.

# %% exercise
def bytes_in_flight(bandwidth, latency):
    ### BEGIN SOLUTION
    return bandwidth * latency
    ### END SOLUTION


def cores_needed(bandwidth, latency, misses_per_core=16, line=64):
    ### BEGIN SOLUTION
    return math.ceil(bytes_in_flight(bandwidth, latency) / (misses_per_core * line))
    ### END SOLUTION

# %% check
assert abs(bytes_in_flight(300e9, 100e-9) - 30_000) < 1e-6        # a DDR5 server socket: 30 KB in flight
assert cores_needed(300e9, 100e-9) == 30
assert abs(bytes_in_flight(3.35e12, 600e-9) - 2.01e6) < 1          # an H100 at an illustrative 600 ns
assert abs(bytes_in_flight(3.35e12, 600e-9) / 32 - 62_812.5) < 1e-6  # ...in 32-byte sectors
print("✅ a socket needs ~30 KB in flight; an H100 ~2 MB — tens of thousands of outstanding requests")

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
# and what FlashAttention does to attention (primer §4). The experiment runs on the CPU even when a
# GPU is present — GPU kernel fusion is layer 02's lab.
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
# and pays neither cost, which is why it gets much closer to the bound.
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
# ## 7 · Every copy costs α + n/β
#
# Time a copy over a range of sizes and fit `t(n) = α + n/β`: α is the fixed cost of *any* copy
# (a call, a descriptor, a launch), β the bandwidth of the slowest link on the path, and `n½ = α·β`
# the size at which you get half of β (primer §5). On T0 we time host `memcpy` of cache-cold data;
# the method is the same one you would use on PCIe, NVLink or a network.
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
else:
    copies = transfer.memcpy_sweep(be, sweep, repeats=3)
for (op, pinned), series in transfer.series(copies).items():
    ab = transfer.fit(series)
    label = op if pinned is None else f"{op} {'pinned' if pinned else 'pageable'}"
    print(f"{label:<14} α = {si(ab.alpha, 's'):>9}   β = {si(ab.beta, 'B/s'):>10}   n½ = {si(ab.n_half, 'B'):>9}"
          f"   (fit r² in log space {ab.r2:.3f})")
    for m in series:
        print(f"   {si(m.params['nbytes'], 'B'):>9}: measured {si(m.bytes_per_s('median'), 'B/s'):>10}"
              f"   model {si(ab.bandwidth(m.params['nbytes']), 'B/s'):>10}")

# %% [markdown]
# Each copy here reads bytes that are not in cache (the op cycles through a pool 4× the last-level
# cache), like a real transfer. Where measured and model disagree, the model is telling you
# something: α-β assumes **one** bottleneck, and a CPU copy has several regimes. The usual one to
# spot is at the largest sizes: glibc's `memcpy` switches to non-temporal (streaming) stores once a
# copy is a sizeable fraction of the last-level cache, which skips the write-allocate read of
# section 4 — so big copies can run *faster* than the fit. Over PCIe the link is the bottleneck at
# every size above a few KB, and the fit is much tighter.
#
# ## 8 · Host ↔ device: pinned vs pageable (T1)
#
# A GPU's copy engine DMAs from host memory it can address directly: **pinned** (page-locked)
# memory. From ordinary **pageable** memory the driver first copies each chunk into a pinned
# bounce buffer, so the copy is slower and blocks the host — it cannot overlap with compute. The
# link's theoretical rate is `GT/s × lanes × encoding ÷ 8` per direction.
#
# ## Exercise 2.6 — PCIe arithmetic
#
# Write `my_pcie_gbs(gen, lanes)` (Gen3: 8 GT/s, Gen4: 16, Gen5: 32; 128b/130b encoding) and
# `copy_seconds(nbytes, gbs, efficiency)`: how long a copy takes at that fraction of the link.

# %% exercise
def my_pcie_gbs(gen, lanes):
    ### BEGIN SOLUTION
    gts = {3: 8.0, 4: 16.0, 5: 32.0}[gen]
    return gts * lanes * (128 / 130) / 8
    ### END SOLUTION


def copy_seconds(nbytes, gbs, efficiency=0.85):
    ### BEGIN SOLUTION
    return nbytes / (gbs * 1e9 * efficiency)
    ### END SOLUTION

# %% check
for g in (3, 4, 5):
    assert abs(my_pcie_gbs(g, 16) - pcie_gbs(g, 16)) < 1e-9
assert abs(my_pcie_gbs(4, 16) - 31.508) < 1e-3 and abs(my_pcie_gbs(5, 8) - 31.508) < 1e-3
t16 = copy_seconds(16e9, my_pcie_gbs(4, 16))            # an 8B-parameter bf16 model over Gen4 x16
assert abs(t16 - 0.5975) < 1e-3
print(f"✅ 16 GB of weights: {t16:.2f} s over Gen4 x16, {copy_seconds(16e9, my_pcie_gbs(3, 16)):.2f} s over "
      f"Gen3 x16 (a T4), {copy_seconds(16e9, my_pcie_gbs(5, 16)):.2f} s over Gen5 x16")

# %%
if be.is_gpu:
    rows, _ = inventory.query_gpus()
    if rows:
        r = rows[0]
        gen, width = r.get("pcie.link.gen.max"), r.get("pcie.link.width.current") or r.get("pcie.link.width.max")
        if gen and width:
            exp = transfer.pcie_expectation(gen, width)
            best = transfer.best_bandwidth([m for m in copies if m.params.get("pinned")])
            print(f"{exp['link']}: theoretical {exp['theoretical_gbs']:.1f} GB/s per direction; best pinned copy "
                  f"{si(best.bytes_per_s(), 'B/s')} ({best.bytes_per_s() / 1e9 / exp['theoretical_gbs']:.0%})")
        for f in inventory.health(rows):
            print(f)
else:
    print("No GPU here, so no host↔device copies were measured. On a GPU runtime (Colab: Runtime → Change"
          " runtime type → T4 GPU) this notebook sweeps h2d/d2h × pinned/pageable and fits α-β to each.\n"
          "Theoretical per-direction PCIe bandwidth (not a measurement):")
    for g, w in ((3, 16), (4, 16), (4, 8), (5, 16)):
        print(f"   Gen{g} x{w}: {my_pcie_gbs(g, w):5.1f} GB/s")

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
# latency, not bandwidth."
#
# **Drills**
#
# 1. *Our 4 KB host→device copies of token IDs run at a few hundred MB/s on a 32 GB/s link. Bug?* —
#    No: they are α-bound. With α ≈ 10 µs a 4 KB copy takes ≥ 10 µs, i.e. ≤ 0.4 GB/s, and `n½` is
#    hundreds of KB. Batch them, keep the buffer pinned, issue them asynchronously.
# 2. *Why does fusing bias + GELU + residual speed up a memory-bound layer?* — Each unfused op
#    reads and writes the activation from HBM; fused, it is read once and written once. Time
#    follows bytes, so `k` passes become one.
# 3. *A vendor's STREAM number is 20% above ours on the same CPU. Who is wrong?* — Possibly
#    nobody: check thread count and pinning, array size vs cache, and whether their build uses
#    non-temporal stores (no write-allocate read). State the convention with the number.
