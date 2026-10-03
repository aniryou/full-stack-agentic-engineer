# %% [markdown]
# # 01 · CUDA kernels in the simulator: threads, blocks, shared memory and barriers
#
# **Tier:** T0. It runs on any CPU. Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`) runs each CUDA
# thread as a Python thread. Thus the kernels in this notebook are *real CUDA kernels*. You can run them,
# break them and examine them on a laptop. Notebook 02 runs the same source on a GPU.
#
# ## The one-minute version
#
# * A kernel is **the body of a loop**. `kernel[blocks, threads](args)` runs one copy per thread. Each copy
#   finds its index as `cuda.grid(1) = blockIdx.x * blockDim.x + threadIdx.x`.
# * The launch rounds the grid up to whole blocks. Thus the last block has idle threads: **the bounds check is
#   part of the algorithm**.
# * Threads of one block share fast on-chip **shared memory** and meet at `cuda.syncthreads()`. Blocks
#   cannot wait for each other in a kernel. Combine their results with a second launch or with atomics.
# * The **warp request** decides the memory speed. When 32 lanes ask for 32 consecutive floats, they touch four
#   32-byte sectors (coalesced). When 32 lanes stride across rows, they touch 32 sectors for the same useful bytes.
# * **Tiling** puts data in shared memory first. Thus the kernel uses each byte that it fetches from DRAM many
#   times. A tiled GEMM issues `tile`× fewer global loads than the naive GEMM.
#
# Concepts: [the primer](../../PRIMER.md) §2 *The execution model* and §3 *Memory access patterns*.

# %%
import sys

import numpy as np

from gpurt import env

# This notebook is about the simulator (the tracer below needs it), so force it even on a GPU machine.
# It must happen before numba.cuda is imported; in a kernel that already imported it, restart the kernel.
env.ensure_numba_mode(simulator=True)
print(env.describe())
from gpurt.kernels import MODE, SIMULATOR, blocks_for, cuda  # noqa: E402

print("numba CUDA mode:", MODE, "- every CUDA thread is a Python thread" if SIMULATOR else "- compiled for this GPU")
# The simulator releases threads from cuda.syncthreads() by polling; with CPython's default 5 ms GIL
# slices every barrier costs milliseconds. A 0.1 ms slice makes barrier-heavy kernels ~5x faster here.
sys.setswitchinterval(1e-4)

# %% [markdown]
# ## Who am I? Indexing a grid
#
# Launch 3 blocks of 4 threads over 10 elements. Let each thread write its coordinates.

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
# Two threads of the last block had nothing to do. If you remove the `if`, these threads index past the end.
# On a GPU, the result is silent memory corruption or *"an illegal memory access was encountered"*. The
# node's logs also show an XID (notebook 06). The simulator changes the error into an `IndexError` that you
# can read:

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
# Write `launch_config(n, threads)`. It returns `(blocks, idle_threads)`. The first value is the number of
# blocks that are necessary to cover `n` elements with `threads` per block (at least one block). The second
# value is the number of launched threads that get no element.

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
# A launch does not need to match the data. In a **grid-stride loop**, each thread starts at
# `cuda.grid(1)`. It then jumps by `cuda.gridsize(1)` (the total number of threads) until it goes past the
# end. Thus you can set the size of the grid for the *GPU* (a few waves of blocks per SM), not for the data.
#
# Write the kernel `scale_add(alpha, x, y, out)`. It calculates `out = alpha * x + y` with a grid-stride loop.
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
# `gpurt.kernels.reduction` calculates the sum of a vector. Each block loads its slice into **shared memory**.
# Then, at each step, the block divides the number of active threads by two: `buf[tid] += buf[tid + stride]`.
# Between steps, `cuda.syncthreads()` makes sure that each write of step $k$ is visible before any read of
# step $k+1$. Blocks cannot sync with each other. Thus a second launch (deterministic) or `cuda.atomic.add`
# (one launch, order-dependent rounding) combines the partial sums of the blocks.

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
# The two answers can be different in the last bits. Floating-point addition is not associative, and the two
# methods add in different orders. On a GPU, the atomic order even changes from one run to the next.
#
# ## Exercise 1.3 — your own block reduction
#
# Write `block_max(x, out)`. Each block of `T = 32` threads finds the maximum of its 32 elements with a
# shared-memory tree. The block writes the maximum to `out[blockIdx.x]`. The host then takes the max of the
# partials. Threads past the end of `x` must supply a value that cannot win (use `NEG_INF`).
#
# Keep `cuda.syncthreads()` **outside** any `if`. Each thread of the block must get to each barrier.

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
# `gpurt.kernels.trace` gives a kernel arrays that record each access and the simulated thread that made it.
# It then puts the accesses into **warp requests** (same block, same warp of 32 lanes, same $k$-th access).
# It also counts the different 32-byte **sectors** that each request touches. The transpose is the standard
# example. The naive kernel reads rows (coalesced) and writes columns (strided).

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
# Naive: 32 sectors per store request, and 12.5 % of the moved bytes are useful. Tiled: the block reads a
# 32×32 tile into shared memory with row loads. It then syncs its threads and writes the transposed tile
# back with row stores. The result is 4 sectors per request on the two sides.
#
# (The simulator counts the
# requests to the memory system. On a GPU, L2 prevents a part of the waste of the naive kernel. Notebook 02
# measures the remainder.)
#
# ## Exercise 1.4 — predict sectors per request
#
# Write `sectors_per_request(stride_bytes, lanes=32, sector=32)`. Lane `l` accesses byte address
# `l * stride_bytes` (4-byte elements, base aligned). Return the number of different sectors that one warp
# request touches. Then predict the stores of the naive transpose (the stride is one output row of `N` floats).

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
# Shared memory has 32 banks of 4-byte words. Word `w` is in bank `w % 32`. The tiled transpose reads a tile
# *column*: lane `l` reads word `l * width + c` of a tile that is `width` words wide. If two lanes hit
# different words of one bank, the hardware serialises the accesses (an $n$-way conflict).
# `bank_conflict_ways` calculates that degree.
#
# Return the list of widths in `range(32, 41)` that are conflict-free for a column read. Then look at the
# pattern.

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
# In the naive GEMM, the thread for `C[row, col]` loads a full row of $A$ and a full column of $B$ from
# global memory. The tiled kernel loads one element of $A$ and one element of $B$ into shared memory per
# `tile`-wide step. Then each thread of the block reads them `tile` times on-chip. Count the loads from the
# kernels:

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
# Write `loads_and_intensity(M, N, K, tile)`. It returns `(global_loads, flops_per_byte)`. The first value is
# the number of element loads that the tiled kernel issues (the formula that the tracer confirmed in the
# last cell). The second value is the arithmetic intensity
# $2 \cdot M \cdot N \cdot K / (\text{loads} \times 4\ \text{bytes})$, with the assumption that no cache helps.
# Compare it with the balance point of a T4 (`traffic.machine_balance`): ~25 FLOP/byte.

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
# **Two minutes.** A CUDA kernel is the body of a parallel loop. The launch sets the number of blocks and
# the number of threads in each block. Each thread calculates its index and must compare it with the data
# size.
#
# Threads in a block work together through shared memory and barriers. Blocks are independent. Thus
# a global result needs a second launch or atomics, and atomics make float results order-dependent.
#
# The memory requests per warp decide the performance. Coalesced accesses fetch 4 sectors for 32 floats,
# and strided accesses fetch 32. Tiling repairs the two problems at the same time. It changes strided global
# accesses into row accesses (transpose). It also changes repeated global loads into shared-memory reuse
# (GEMM: `tile`× fewer loads). Everything in this notebook ran in a simulator, and the same source runs on
# a GPU in notebook 02.
#
# **Drill questions**
#
# 1. *A naive transpose runs at a fifth of copy bandwidth. Why, and what is the solution?* Its stores are
#    strided: each warp store touches 32 sectors, not 4. Put a tile in shared memory first, so that the
#    loads and the stores are row-contiguous. Pad the tile to `TILE + 1` to prevent 32-way bank conflicts.
# 2. *Why does a tiled GEMM need two barriers per tile?* One barrier comes after the load, so that no
#    thread reads a half-written tile. One barrier comes after the computation, so that no thread
#    overwrites a tile that another thread still reads.
# 3. *Why can blocks not sync in a kernel?* Blocks run in waves when SMs become free. A barrier across
#    blocks that are not resident causes a deadlock. Divide the work into two launches (or use atomics or
#    cooperative launches, with their limits).
