# %% [markdown]
# # 02 · Memory-bound kernels on a real GPU: effective bandwidth, launch overhead and CUDA Graphs
#
# **Tier:** T1 — any NVIDIA GPU (Colab or Kaggle T4, an L4, a rented RTX 4090) with `numba-cuda`
# installed. Without a GPU the notebook takes its **T0 path**: the same kernels run at toy sizes in the
# simulator (correctness only), and every number printed is labelled *model prediction* — computed
# from byte counts and stated assumptions, never presented as a measurement.
#
# ## The one-minute version
#
# * A memory-bound kernel is timed in **bytes**: effective bandwidth = bytes the algorithm must move ÷
#   time, compared with the datasheet peak. 80–90 % of peak means the kernel is done; the only way
#   faster is to move fewer bytes (fusion, lower precision).
# * Measure the kernel, not the plumbing: data already on the device, CUDA events, a warm-up launch.
# * Small launches are **latency-bound**: `t = α + S/B` with α a few microseconds of launch overhead —
#   the same model notebook 04 fits to collectives.
# * Access patterns and extra passes cost bandwidth: naive transpose ≪ tiled ≈ copy; fused softmax
#   moves half the bytes of the unfused one.
# * A step made of many tiny kernels is **launch-bound**: the CPU cannot issue kernels as fast as the GPU
#   finishes them. CUDA Graphs replay the whole sequence with one launch.
#
# Concepts: [the primer](../../PRIMER.md) §2 *The execution model*, §3 *Memory access patterns*,
# §4 *Streams, launch overhead and CUDA Graphs*. The roofline itself is layer 01.

# %%
import sys

import numpy as np

from gpurt import env

from gpurt.kernels import SIMULATOR, cuda  # noqa: E402
from gpurt.kernels import bench, traffic  # noqa: E402
from gpurt.kernels import elementwise as ew  # noqa: E402
from gpurt.kernels import softmax as sm  # noqa: E402
from gpurt.kernels import transpose as tr  # noqa: E402
from gpurt.launch import LaunchModel  # noqa: E402

sys.setswitchinterval(1e-4)  # only matters for the simulator path
HAVE_GPU = (not SIMULATOR) and cuda.is_available()
print(env.describe())
if HAVE_GPU:
    DEV = bench.device_info()
    SPEC = traffic.spec_for(DEV["name"]) or traffic.GPUS["T4"]  # unknown GPU: compare with a T4, say so below
    print(f"T1 path: measuring on {DEV['name']} (cc {DEV['cc']}, {DEV['sms']} SMs); datasheet spec: {SPEC}")
else:
    SPEC = traffic.GPUS["T4"]
    print("T0 path: no GPU. Kernels run in the simulator for correctness; numbers below are MODEL PREDICTIONS "
          f"for a {SPEC.name} ({SPEC.mem_gbps} GB/s peak, verify), not measurements.")
    print("To measure: run this notebook on a GPU, or `python -m gpurt.kernels.bench` (see deploy/any-gpu).")

# %% [markdown]
# ## Bytes first
#
# Before timing anything, count what each kernel *must* move (`gpurt.kernels.traffic`):

# %%
n = 1 << 24
rows = [("copy", traffic.elementwise_bytes("copy", n)), ("vec_add", traffic.elementwise_bytes("vec_add", n)),
        ("reduction", traffic.reduction_bytes(n)), ("transpose 4096^2", traffic.transpose_bytes(4096, 4096)),
        ("softmax unfused 4096^2", traffic.softmax_bytes(4096, 4096, "unfused")),
        ("softmax fused 4096^2", traffic.softmax_bytes(4096, 4096, "fused"))]
for name, b in rows:
    print(f"{name:>24}: {b / 2**20:8.0f} MiB   ideal time at {SPEC.mem_gbps} GB/s: {b / (SPEC.mem_gbps * 1e9) * 1e3:6.3f} ms")

# %% [markdown]
# ## Exercise 2.1 — effective bandwidth
#
# Write `effective(n_elements, bytes_per_element, seconds, peak_gbps)` returning `(gbps, fraction_of_peak)`
# (GB = 1e9 bytes). Use it on a hypothetical run: a `vec_add` of 2²⁴ floats (12 bytes per element) that
# takes 0.70 ms on a T4 (320 GB/s).

# %% exercise
def effective(n_elements: int, bytes_per_element: int, seconds: float, peak_gbps: float) -> tuple[float, float]:
    ### BEGIN SOLUTION
    gbps = n_elements * bytes_per_element / seconds / 1e9
    return gbps, gbps / peak_gbps
    ### END SOLUTION

# %% check
gbps, frac = effective(1 << 24, 12, 0.70e-3, 320)
assert abs(gbps - 287.6) < 0.1 and abs(frac - 0.899) < 0.001
print(f"✅ {gbps:.1f} GB/s = {frac:.0%} of peak: a streaming kernel at ~90 % has nothing left to gain but fewer bytes")

# %% [markdown]
# ## A bandwidth sweep: two regimes
#
# Time `copy` and `vec_add` from 4 KB to 256 MB of traffic. Small launches cost a roughly fixed α (launch +
# latency); large ones approach the DRAM bandwidth B. On the T0 path we check correctness in the simulator
# and print what the α-β model *predicts* for the reference GPU.

# %%
from gpurt.dist.alphabeta import AlphaBeta, fit  # noqa: E402

if HAVE_GPU:
    sweep = bench.bandwidth_sweep("copy") + bench.bandwidth_sweep("vec_add")
    for r in sweep:
        print(f"{r['kind']:>8} n={r['n']:>10}  {r['seconds'] * 1e6:9.1f} µs  {r['gbps']:7.1f} GB/s  (measured)")
    MODEL = fit([r["bytes"] for r in sweep if r["kind"] == "copy"], [r["seconds"] for r in sweep if r["kind"] == "copy"])
    print("copy fit (measured):", MODEL)
else:
    x = np.arange(1000, dtype=np.float32)
    assert np.array_equal(ew.run_copy(x, blocks=2, threads=64), x)
    assert np.array_equal(ew.run_vec_add(x, x), 2 * x)
    MODEL = AlphaBeta(alpha_s=5e-6, bw_Bps=SPEC.mem_gbps * 1e9 * 0.8)  # ASSUMPTIONS: 5 µs, 80 % of peak
    print("simulator: copy and vec_add are correct. MODEL PREDICTION (α = 5 µs, B = 80 % of peak — assumptions):")
    for k in range(10, 27, 2):
        b = traffic.elementwise_bytes("copy", 2 ** k)
        print(f"    copy n=2^{k:<2}  {MODEL.time(b) * 1e6:9.1f} µs  {MODEL.algbw_gbps(b):7.1f} GB/s  ({MODEL.regime(b)})")

# %% [markdown]
# ## Exercise 2.2 — where launch overhead stops mattering
#
# The two terms of `t = α + S/B` are equal at `S½ = α·B`, where the kernel reaches half its asymptotic
# bandwidth. Write `crossover_elements(alpha_s, bw_Bps, bytes_per_element)`: the vector length at which a
# kernel moving `bytes_per_element` bytes per element reaches that point. Evaluate it for a copy-like
# kernel with α = 5 µs and B = 256 GB/s (80 % of a T4).

# %% exercise
def crossover_elements(alpha_s: float, bw_Bps: float, bytes_per_element: int) -> float:
    ### BEGIN SOLUTION
    return alpha_s * bw_Bps / bytes_per_element
    ### END SOLUTION

# %% check
assert abs(crossover_elements(5e-6, 256e9, 8) - 160_000) < 1e-6
mine = crossover_elements(MODEL.alpha_s, MODEL.bw_Bps, 8)
print(f"✅ below ~{mine:,.0f} floats a copy is launch-bound on {'this GPU' if HAVE_GPU else 'the model GPU'}: "
      "many small kernels waste the machine; fuse or batch them")

# %% [markdown]
# ## Transpose: the cost of strided stores
#
# Notebook 01 counted it: the naive transpose's stores touch 32 sectors per warp request instead of 4.
# On a GPU the effect is smaller than 8× because L2 merges some partial sectors before they reach DRAM —
# which is why you measure.

# %%
if HAVE_GPU:
    tb = bench.transpose_bench(4096)
    for name, r in tb.items():
        print(f"{name:>13}: {r['gbps']:7.1f} GB/s  (measured)")
    COPY_GBPS = tb["copy2d"]["gbps"]
else:
    X = np.random.default_rng(0).random((40, 70), dtype=np.float32)
    assert np.array_equal(tr.run_transpose(X, "naive"), X.T) and np.array_equal(tr.run_transpose(X, "tiled"), X.T)
    COPY_GBPS = MODEL.bw_Bps / 1e9
    print(f"simulator: both transposes are correct. Model copy bandwidth: {COPY_GBPS:.0f} GB/s (assumption)")

# %% [markdown]
# ## Exercise 2.3 — a no-cache model of the naive transpose
#
# Half the bytes are loads at full efficiency, half are stores at efficiency `e` (useful bytes ÷ bytes
# moved; 1/8 for the naive transpose). If moving bytes runs at the copy bandwidth `B`, what effective
# bandwidth does the transpose report? Write `naive_transpose_gbps(copy_gbps, store_efficiency)`.

# %% exercise
def naive_transpose_gbps(copy_gbps: float, store_efficiency: float) -> float:
    ### BEGIN SOLUTION
    # bytes/2 at B, plus bytes/2 at B*e  ->  t = bytes/(2B) * (1 + 1/e)
    return 2 * copy_gbps / (1 + 1 / store_efficiency)
    ### END SOLUTION

# %% check
assert abs(naive_transpose_gbps(300, 0.125) - 66.667) < 1e-3
assert naive_transpose_gbps(300, 1.0) == 300
pred = naive_transpose_gbps(COPY_GBPS, 0.125)
if HAVE_GPU:
    print(f"✅ model {pred:.0f} GB/s vs measured naive {tb['naive']['gbps']:.0f} GB/s — above the model means caches "
          "absorbed part of the waste; the tiled kernel should sit near copy")
else:
    print(f"✅ model prediction for the naive transpose: {pred:.0f} GB/s (22 % of copy) — a pessimistic bound; "
          "measure it on a GPU")

# %% [markdown]
# ## Fusion: softmax in one pass instead of four kernels
#
# The unfused softmax moves the matrix six times (24 B/element) in four launches; the fused *online*
# softmax moves it three times (12 B/element) in one.

# %%
if HAVE_GPU:
    sb = bench.softmax_bench(4096, 4096)
    for name, r in sb.items():
        print(f"{name:>8}: {r['seconds'] * 1e3:7.3f} ms  {r['gbps']:7.1f} GB/s effective  (measured)")
    print(f"speedup {sb['unfused']['seconds'] / sb['fused']['seconds']:.2f}x")
else:
    S = np.random.default_rng(1).standard_normal((6, 50)).astype(np.float32) * 3
    for fused in (True, False):
        np.testing.assert_allclose(sm.run_softmax(S, fused, threads=16), sm.softmax_reference(S), rtol=1e-5, atol=1e-6)
    print("simulator: fused and unfused softmax match NumPy")

# %% [markdown]
# ## Exercise 2.4 — predict the fusion speedup, including launches
#
# Write `softmax_times(rows, cols, bw_Bps, launch_s)` returning `(t_unfused, t_fused)`: 24 B/element and
# four launches versus 12 B/element and one launch. Compare a large matrix with a tiny one.

# %% exercise
def softmax_times(rows: int, cols: int, bw_Bps: float, launch_s: float) -> tuple[float, float]:
    ### BEGIN SOLUTION
    elems = rows * cols
    return 4 * launch_s + 24 * elems / bw_Bps, launch_s + 12 * elems / bw_Bps
    ### END SOLUTION

# %% check
tu, tf = softmax_times(4096, 4096, 300e9, 5e-6)
assert abs(tu / tf - 2.01) < 0.01
tu_small, tf_small = softmax_times(16, 128, 300e9, 5e-6)
assert 3.9 < tu_small / tf_small < 4.0
print(f"✅ large: {tu / tf:.2f}x (bytes decide); tiny: {tu_small / tf_small:.2f}x (launches decide)")

# %% [markdown]
# ## Determinism: atomics change the order of additions
#
# Floating-point addition is not associative. The atomic reduction adds block results in whatever order
# blocks finish, so repeated runs can differ in the last bits; the two-pass reduction fixes the order.

# %%
if HAVE_GPU:
    rb = bench.reduction_bench()
    print("distinct atomic results  :", rb["distinct_atomic_results"], "(measured, 5 runs)")
    print("distinct two-pass results:", rb["distinct_two_pass_results"])
    print("float64 reference        :", rb["float64_reference"])
else:
    v = np.random.default_rng(2).random(1 << 20, dtype=np.float32)
    orders = {float(np.add.reduce(v[p])) for p in (np.arange(v.size), np.random.default_rng(3).permutation(v.size),
                                                      np.random.default_rng(4).permutation(v.size))}
    print("NumPy, same numbers summed in three orders (float32):", sorted(orders))

# %% [markdown]
# ## Launch overhead and CUDA Graphs
#
# `gpurt.launch.LaunchModel`: eager step ≈ `n·max(k + g, L)`, graph step ≈ `G + n·(k + g)` for `n`
# kernels of `k` µs, a GPU-side gap `g`, CPU launch cost `L` and one graph launch `G`. On a GPU with
# torch, `measure_graph_vs_eager()` measures the real thing.

# %%
from gpurt.launch import measure_graph_vs_eager  # noqa: E402

if HAVE_GPU:
    print(f"Numba empty-kernel launch: {bench.launch_overhead(500):.1f} µs per launch (measured, host clock)")
    if env.has_module("torch"):
        g = measure_graph_vs_eager(n_kernels=200)
        print(f"torch, 200 tiny kernels: eager {g['eager_us']:.0f} µs, graph {g['graph_us']:.0f} µs, "
              f"{g['speedup']:.1f}x (measured; replay correct: {g['replay_correct']})")
else:
    m = LaunchModel()
    print(f"MODEL (assumptions: L = {m.launch_us} µs, G = {m.graph_launch_us} µs, g = {m.node_gap_us} µs):")
    for k in (2, 5, 20):
        print(f"    200 kernels of {k:>2} µs: eager {m.eager_us(200, k):6.0f} µs, graph {m.graph_us(200, k):6.0f} µs, "
              f"launch-bound: {m.launch_bound(k)}")

# %% [markdown]
# ## Exercise 2.5 — should this decode step be captured in a graph?
#
# A decode step runs `layers × kernels_per_layer` kernels. Write `decode_step(layers, kernels_per_layer,
# kernel_us, model)` returning `(eager_us, graph_us, launch_bound)` with a `LaunchModel`. Evaluate a
# 32-layer model with 12 kernels per layer at batch 1 (≈3 µs per kernel) and at a large batch (≈20 µs).

# %% exercise
def decode_step(layers: int, kernels_per_layer: int, kernel_us: float, model: LaunchModel):
    ### BEGIN SOLUTION
    n = layers * kernels_per_layer
    return model.eager_us(n, kernel_us), model.graph_us(n, kernel_us), model.launch_bound(kernel_us)
    ### END SOLUTION

# %% check
m = LaunchModel(launch_us=6.0, graph_launch_us=8.0, node_gap_us=1.0)
eager, graph, bound = decode_step(32, 12, 3.0, m)
assert (eager, graph, bound) == (2304.0, 1544.0, True)
eager_big, graph_big, bound_big = decode_step(32, 12, 20.0, m)
assert not bound_big and graph_big >= eager_big
print(f"✅ batch 1: {eager / graph:.2f}x faster as a graph; large batch: no gain — graphs remove CPU cost, "
      "not GPU work. Engines capture decode graphs per batch size, not prefill.")

# %% [markdown]
# ## In a design review
#
# **Two minutes.** For memory-bound kernels I report *effective bandwidth* — the bytes the algorithm
# must move divided by the time — next to the datasheet peak. I measure with data on the device, CUDA
# events around many back-to-back launches, after a warm-up. A size sweep shows two regimes: a latency
# floor of a few microseconds per launch and a bandwidth plateau; the α-β fit gives both numbers and the
# size where they cross. Anything near 80–90 % of peak is finished; below that, look at access patterns
# (the naive transpose's strided stores) or extra passes (unfused softmax moves twice the bytes). When a
# step is hundreds of tiny kernels, the CPU is the bottleneck, `GPU util` still reads high, and CUDA
# Graphs — or fewer, fused kernels — are the fix.
#
# **Drill questions**
#
# 1. *The first call of my kernel takes 300 ms, the rest 50 µs. Is the GPU slow?* — No: the first call
#    JIT-compiles (Numba, Triton, PTX JIT) and may create the CUDA context. Warm up before timing, and in
#    production pre-compile or cache (engines warm up and capture graphs at start-up).
# 2. *Our fused kernel reaches 88 % of peak DRAM bandwidth. How do we make it faster?* — Move fewer bytes:
#    lower precision (bf16/fp8), fuse the neighbouring op, or avoid re-reading inputs. More threads will
#    not help a saturated memory system.
# 3. *When do CUDA Graphs not help?* — When kernels are long enough that the CPU keeps ahead (large
#    batches, prefill), and when shapes change every step — a graph is captured per shape, so engines
#    pad to a set of captured batch sizes.
