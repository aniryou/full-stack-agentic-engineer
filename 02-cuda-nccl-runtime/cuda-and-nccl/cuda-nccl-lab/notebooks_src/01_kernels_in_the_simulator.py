# %% [markdown]
# # 01 · CUDA kernels in the simulator: threads, blocks, shared memory and barriers
#
# **Tier:** T0 — runs on any CPU. Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`) executes every CUDA
# thread as a Python thread, so the kernels below are *real CUDA kernels* that you can run, break and
# inspect on a laptop. Notebook 02 runs the same source on a GPU.
#
# ## The one-minute version
#
# * A kernel is **the body of a loop**. `kernel[blocks, threads](args)` runs one copy per thread; each copy
#   finds its index as `cuda.grid(1) = blockIdx.x * blockDim.x + threadIdx.x`.
# * The grid is rounded up to whole blocks, so the last block has idle threads: **the bounds check is
#   part of the algorithm**.
# * Threads of one block share fast on-chip **shared memory** and meet at `cuda.syncthreads()`. Blocks
#   cannot wait for each other inside a kernel — combine their results with a second launch or atomics.
# * Memory speed is decided per **warp request**: 32 lanes asking for 32 consecutive floats touch four
#   32-byte sectors (coalesced); 32 lanes striding across rows touch 32 sectors for the same useful bytes.
# * **Tiling** stages data in shared memory so each byte fetched from DRAM is used many times: a tiled
#   GEMM issues `tile`× fewer global loads than the naive one.
#
# Concepts: [the primer](../../PRIMER.md) §2 *The execution model* and §3 *Memory access patterns*.

# %%
import sys

import numpy as np

from gpurt import env

print(env.describe())
from gpurt.kernels import MODE, SIMULATOR, blocks_for, cuda  # noqa: E402  (decides simulator vs GPU first)

print("numba CUDA mode:", MODE)
# The simulator releases threads from cuda.syncthreads() by polling; with CPython's default 5 ms GIL
# slices every barrier costs milliseconds. A 0.1 ms slice makes barrier-heavy kernels ~5x faster here.
sys.setswitchinterval(1e-4)

# %% [markdown]
# ## Who am I? Indexing a grid
#
# Launch 3 blocks of 4 threads over 10 elements and let every thread write down its coordinates.

# %%
@cuda.jit
def whoami(block_ids, thread_ids, global_ids):
    i = cuda.grid(1)  # = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < global_ids.size:
        block_ids[i] = cuda.blockIdx.x
        thread_ids[i] = cuda.threadIdx.x
        global_ids[i] = i


n, threads = 10, 4
blocks = blocks_for(n, threads)
b_ids, t_ids, g_ids = (np.full(n, -1) for _ in range(3))
whoami[blocks, threads](b_ids, t_ids, g_ids)
print(f"{blocks} blocks x {threads} threads = {blocks * threads} threads for {n} elements")
print("element :", g_ids)
print("block   :", b_ids)
print("thread  :", t_ids)

# %% [markdown]
# Two threads of the last block had nothing to do. Remove the `if` and they index past the end. On a GPU
# that is silent memory corruption or *"an illegal memory access was encountered"* (and an XID in the
# node's logs — notebook 06). The simulator turns it into an `IndexError` you can read:

# %%
@cuda.jit
def add_without_bounds_check(a, b, out):
    i = cuda.grid(1)
    out[i] = a[i] + b[i]


a = np.arange(10, dtype=np.float32)
try:
    add_without_bounds_check[blocks, threads](a, a, np.zeros_like(a))
except IndexError as e:
    print("simulator caught it:", str(e).splitlines()[0][:120])

# %% [markdown]
# ## Exercise 1.1 — the launch configuration
#
# Write `launch_config(n, threads)` returning `(blocks, idle_threads)`: the number of blocks needed to
# cover `n` elements with `threads` per block (at least one block), and how many launched threads get
# no element.

# %% exercise
def launch_config(n: int, threads: int) -> tuple[int, int]:
    ### BEGIN SOLUTION
    blocks = max(1, (n + threads - 1) // threads)
    return blocks, blocks * threads - n
    ### END SOLUTION

# %% check
assert launch_config(1000, 256) == (4, 24)
assert launch_config(1024, 256) == (4, 0)
assert launch_config(1, 128) == (1, 127)
print("✅ launch_config: ceil(n / threads) blocks, the remainder idles")

# %% [markdown]
# ## Exercise 1.2 — a grid-stride loop
#
# A launch does not have to match the data. With a **grid-stride loop** each thread starts at
# `cuda.grid(1)` and jumps by `cuda.gridsize(1)` (the total number of threads) until it runs off the
# end. That lets you size the grid to the *GPU* (a few waves of blocks per SM) instead of to the data.
#
# Write the kernel `scale_add(alpha, x, y, out)` computing `out = alpha * x + y` with a grid-stride loop.
# The check launches only 2 blocks of 32 threads for 1,000 elements.

# %% exercise
@cuda.jit
def scale_add(alpha, x, y, out):
    ### BEGIN SOLUTION
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(start, out.size, stride):
        out[i] = alpha * x[i] + y[i]
    ### END SOLUTION

# %% check
rng = np.random.default_rng(0)
x, y = rng.random(1000, dtype=np.float32), rng.random(1000, dtype=np.float32)
out = np.zeros_like(x)
scale_add[2, 32](np.float32(3.0), x, y, out)  # np.float32: a Python float would be float64 inside the kernel
assert 2 * 32 < x.size
np.testing.assert_allclose(out, 3.0 * x + y, rtol=1e-6)
print("✅ 64 threads covered 1,000 elements")

# %% [markdown]
# ## Shared memory and barriers: a block reduction
#
# `gpurt.kernels.reduction` sums a vector. Each block loads its slice into **shared memory**, then halves
# the number of active threads at every step: `buf[tid] += buf[tid + stride]`. Between steps,
# `cuda.syncthreads()` guarantees every write of step *k* is visible before any read of step *k+1*.
# Blocks cannot synchronise with each other, so the per-block partial sums are combined by a second
# launch (deterministic) or by `cuda.atomic.add` (one launch, order-dependent rounding).

# %%
import inspect  # noqa: E402

from gpurt.kernels import reduction as red  # noqa: E402

src = inspect.getsource(red.make_block_sum)
print(src[src.index("    @cuda.jit"):])
v = rng.random(3000, dtype=np.float32)
print("two-pass :", red.run_sum(v, threads=64), f"({red.launches_for_two_pass(v.size, 64)} launches)")
print("atomic   :", red.run_sum(v, threads=64, atomic=True))
print("float64  :", float(v.astype(np.float64).sum()))

# %% [markdown]
# The two answers can differ in the last bits: floating-point addition is not associative, and the two
# schemes add in different orders. On a GPU the atomic order even changes from run to run.
#
# ## Exercise 1.3 — your own block reduction
#
# Write `block_max(x, out)`: every block of `T = 32` threads finds the maximum of its 32 elements with a
# shared-memory tree and writes it to `out[blockIdx.x]`; the host takes the max of the partials.
# Threads past the end of `x` must contribute something that cannot win (use `NEG_INF`). Keep
# `cuda.syncthreads()` **outside** any `if` — every thread of the block must reach every barrier.

# %% exercise
from numba import float32  # noqa: E402

T = 32
NEG_INF = np.float32(-np.inf)


@cuda.jit
def block_max(x, out):
    buf = cuda.shared.array(T, float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)
    ### BEGIN SOLUTION
    buf[tid] = x[i] if i < x.size else NEG_INF
    cuda.syncthreads()
    stride = T // 2
    while stride > 0:
        if tid < stride:
            buf[tid] = max(buf[tid], buf[tid + stride])
        cuda.syncthreads()
        stride //= 2
    if tid == 0:
        out[cuda.blockIdx.x] = buf[0]
    ### END SOLUTION

# %% check
v = rng.standard_normal(1000).astype(np.float32)
partials = np.zeros(blocks_for(v.size, T), np.float32)
block_max[partials.size, T](v, partials)
assert partials.max() == v.max()
assert np.all(partials == np.array([v[k * T:(k + 1) * T].max() for k in range(partials.size)]))
print(f"✅ {partials.size} blocks, each reduced its slice in log2({T}) = {int(np.log2(T))} barrier steps")

# %% [markdown]
# ## Watching the memory system: coalescing, measured from the kernel itself
#
# `gpurt.kernels.trace` hands a kernel arrays that record every access with the simulated thread that
# made it, then groups accesses into **warp requests** (same block, same warp of 32 lanes, same k-th
# access) and counts the distinct 32-byte **sectors** each request touches. The transpose is the classic
# case: the naive kernel reads rows (coalesced) and writes columns (strided).

# %%
from gpurt.kernels import transpose as tr  # noqa: E402
from gpurt.kernels.trace import bank_conflict_ways, trace_launch  # noqa: E402

N = 64
A = np.arange(N * N, dtype=np.float32).reshape(N, N)
grid, block = tr.launch_config(N, N)
traces = {}
for name, kernel in (("naive", tr.transpose_naive), ("tiled", tr.transpose_tiled)):
    out = np.zeros_like(A)
    traces[name] = trace_launch(kernel, grid, block, A, out, names=["in", "out"])
    assert np.array_equal(out, A.T)
    print(f"--- {name} transpose (block {block}, grid {grid})\n{traces[name].table()}")
for name, t in traces.items():
    print(f"{name}: lanes 0-3 of the first store request write elements {t.warp_request('out', 'W')[:4]}")

# %% [markdown]
# Naive: 32 sectors per store request, 12.5 % of the moved bytes useful. Tiled: the block reads a
# 32×32 tile with row loads into shared memory, synchronises, and writes the transposed tile back with
# row stores — 4 sectors per request on both sides. (The simulator counts requests to the memory system;
# on a GPU, L2 absorbs part of the naive kernel's waste — notebook 02 measures what is left.)
#
# ## Exercise 1.4 — predict sectors per request
#
# Write `sectors_per_request(stride_bytes, lanes=32, sector=32)`: lane `l` accesses byte address
# `l * stride_bytes` (4-byte elements, base aligned); return how many distinct sectors one warp request
# touches. Then predict the naive transpose's stores (the stride is one output row of `N` floats).

# %% exercise
def sectors_per_request(stride_bytes: int, lanes: int = 32, sector: int = 32) -> int:
    ### BEGIN SOLUTION
    return len({(lane * stride_bytes) // sector for lane in range(lanes)})
    ### END SOLUTION

# %% check
assert sectors_per_request(4) == 4  # consecutive floats: coalesced
assert sectors_per_request(8) == 8  # every other float: half the bytes wasted
assert sectors_per_request(0) == 1  # all lanes read one address: a broadcast
naive = trace_launch(tr.transpose_naive, grid, block, A, np.zeros_like(A), names=["in", "out"]).get("out", "W")
assert sectors_per_request(N * 4) == naive.sectors_per_request == 32
print("✅ stride 4 B -> 4 sectors, stride 8 B -> 8, stride of a row -> 32 (what the tracer measured)")

# %% [markdown]
# ## Exercise 1.5 — padding away bank conflicts
#
# Shared memory has 32 banks of 4-byte words; word `w` lives in bank `w % 32`. The tiled transpose
# reads a tile *column*: lane `l` reads word `l * width + c` of a tile `width` words wide. If two lanes hit
# different words of one bank, the accesses serialise (an n-way conflict). `bank_conflict_ways` computes
# that degree. Return the list of widths in `range(32, 41)` that are conflict-free for a column read —
# and notice the pattern.

# %% exercise
def conflict_free_widths(widths=range(32, 41)) -> list[int]:
    ### BEGIN SOLUTION
    return [w for w in widths if bank_conflict_ways([lane * w for lane in range(32)]) == 1]
    ### END SOLUTION

# %% check
assert bank_conflict_ways([lane * 32 for lane in range(32)]) == 32  # unpadded: one bank, 32 ways
assert conflict_free_widths() == [33, 35, 37, 39]
print("✅ any odd width works: it is coprime with 32 banks. TILE + 1 is the cheapest pad.")

# %% [markdown]
# ## Reuse: tiling a matrix multiply
#
# The naive GEMM thread for `C[row, col]` loads a full row of A and column of B from global memory. The
# tiled kernel loads one element of A and one of B per `tile`-wide step into shared memory, then every
# thread of the block reads them `tile` times on-chip. Count it from the kernels:

# %%
from gpurt.kernels import matmul as mm  # noqa: E402
from gpurt.kernels import traffic  # noqa: E402

M = Nn = K = 32
Am, Bm = rng.random((M, K), dtype=np.float32), rng.random((K, Nn), dtype=np.float32)
for name, kernel, tile in (("naive", mm.matmul_naive, 1), ("tiled 8", mm.make_matmul_tiled(8), 8),
                           ("tiled 16", mm.make_matmul_tiled(16), 16)):
    C = np.zeros((M, Nn), np.float32)
    g, b = mm.launch_config(M, Nn, 16 if tile == 1 else tile)
    acc = trace_launch(kernel, g, b, Am, Bm, C, names=["A", "B", "C"]).accesses()
    assert np.allclose(C, Am @ Bm, rtol=1e-5)
    loads = acc[("A", "R")] + acc[("B", "R")]
    print(f"{name:>8}: {loads:6d} global loads (model: {traffic.matmul_global_loads(M, Nn, K, tile):6d})")

# %% [markdown]
# ## Exercise 1.6 — what tiling buys at scale
#
# Write `loads_and_intensity(M, N, K, tile)` returning `(global_loads, flops_per_byte)`: the element
# loads the tiled kernel issues (the formula the tracer just confirmed) and the arithmetic intensity
# `2·M·N·K / (loads × 4 bytes)`, as if no cache helped. Compare with a T4's balance point
# (`traffic.machine_balance`): ~25 FLOP/byte.

# %% exercise
def loads_and_intensity(M: int, N: int, K: int, tile: int) -> tuple[int, float]:
    ### BEGIN SOLUTION
    loads = 2 * M * N * -(-K // tile)
    return loads, 2 * M * N * K / (loads * 4)
    ### END SOLUTION

# %% check
assert loads_and_intensity(1024, 1024, 1024, 1) == (2 * 1024 ** 3, 0.25)
assert loads_and_intensity(1024, 1024, 1024, 16) == (134_217_728, 4.0)
balance = traffic.machine_balance(traffic.GPUS["T4"])
print(f"✅ tile 16: 16x fewer loads, 4 FLOP/B — still below a T4's {balance:.0f} FLOP/B, which is why real GEMMs "
      "also tile in registers (and use tensor cores)")

# %% [markdown]
# ## In a design review
#
# **Two minutes.** A CUDA kernel is the body of a parallel loop: the launch picks how many blocks of
# how many threads, each thread computes its index and must check it against the data size. Threads in
# a block cooperate through shared memory and barriers; blocks are independent, so anything global needs
# a second launch or atomics — and atomics make float results order-dependent. Performance is decided by
# memory requests per warp: coalesced accesses fetch 4 sectors for 32 floats, strided ones 32. Tiling
# fixes both problems at once — it turns strided global accesses into row accesses (transpose) and turns
# repeated global loads into shared-memory reuse (GEMM: `tile`× fewer loads). Everything above ran in a
# simulator; the same source runs on a GPU in notebook 02.
#
# **Drill questions**
#
# 1. *A naive transpose runs at a fifth of copy bandwidth. Why, and what is the fix?* — Its stores are
#    strided: each warp store touches 32 sectors instead of 4. Stage a tile in shared memory so both the
#    loads and the stores are row-contiguous; pad the tile to `TILE + 1` to avoid 32-way bank conflicts.
# 2. *Why does a tiled GEMM need two barriers per tile?* — One after loading (nobody reads a half-written
#    tile) and one after computing (nobody overwrites a tile another thread is still reading).
# 3. *Why can't blocks synchronise inside a kernel?* — Blocks run in waves as SMs free up; a barrier
#    across blocks that are not resident would deadlock. Split the work into two launches (or use
#    atomics / cooperative launches, with their limits).
