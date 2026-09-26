# %% [markdown]
# # 02 · Tiling, fusion, launches and occupancy
#
# **Tier:** T0 (CPU, simulated; numpy only). Real timings of the same kernels are in the lab's
# `02_memory_bound_kernels_on_a_real_gpu` (T1).
#
# ## The one-minute version
# A memory-bound kernel takes *(bytes it moves) / (memory bandwidth)*, plus its launch. There
# are four levers:
#
# * **Tiling.** A GEMM that stages `BM x BK` and `BK x BN` tiles in shared memory reloads A only
#   `ceil(N/BN)` times and B only `ceil(M/BM)` times, instead of N and M times. Traffic falls by
#   roughly the tile size.
# * **Fusion.** A softmax built from five kernels makes about eight matrix-sized passes through
#   HBM. A fused one makes two, or three with online softmax when a row does not fit on chip.
#   FlashAttention is this idea applied to attention.
# * **CUDA Graphs.** When kernels are shorter than their launch cost, the GPU waits on the CPU.
#   A graph replays the whole sequence with one launch, which is why engines capture decode steps.
# * **Occupancy.** Registers, shared memory, and warp and block slots cap how many warps an SM
#   keeps resident. You need enough warps, or enough loads in flight per warp, to cover memory
#   latency (Little's law). More is not always better.
#
# Primer: §3 *Memory access patterns*, §2 *The execution model* (occupancy) and §4 *Streams,
# launch overhead and CUDA Graphs* (`../../PRIMER.md`).

# %%
import math

import numpy as np

from gpusim import occupancy as occ
from gpusim import tiling

n = 4096                                               # C = A @ B, all 4096 x 4096, bf16
print(f"{'variant':>18}  {'global traffic':>14}  {'FLOP/byte':>9}  {'time if all from HBM on H100':>28}")
for label, (bm, bn) in [("naive (1x1)", (1, 1)), ("tile 32x32", (32, 32)), ("tile 128x128", (128, 128)),
                        ("tile 128x256", (128, 256)), ("compulsory", (None, None))]:
    t = tiling.gemm_traffic(n, n, n, bm, bn, dtype_bytes=2)
    print(f"{label:>18}  {t['bytes'] / 1e9:>11.2f} GB  {t['intensity']:>9.1f}  {t['bytes'] / 3.35e12 * 1e3:>25.2f} ms")

# %% [markdown]
# The 4096^3 GEMM is 137 GFLOP, about 0.14 ms at ~990 TFLOP/s of dense BF16. Without reuse it
# would stream 275 GB, 82 ms at H100 bandwidth. A 128x128 shared-memory tile cuts global
# traffic by about 126x. The rest of the gap to the compulsory 0.1 GB is closed by **L2**: many
# thread blocks reuse the same A and B panels while those panels are still in the 50 MB L2, so
# HBM sees far less than the "global traffic" column. (The roofline and ridge point are layer
# 01's topic; here we count bytes.)
#
# The simulator runs a real tiled GEMM on the CPU and counts its global loads:

# %%
rng = np.random.default_rng(0)
A = rng.integers(-4, 5, (96, 80)).astype(np.float64)
B = rng.integers(-4, 5, (80, 112)).astype(np.float64)
C, counted = tiling.tiled_matmul(A, B, BM=32, BN=32, BK=16)
print("exact:", np.array_equal(C, A @ B), "| counted:", counted,
      "| formula:", {k: v for k, v in tiling.gemm_traffic(96, 112, 80, 32, 32).items() if k.endswith("elems")})

# %% [markdown]
# ## Exercise 2.1: the tiled-GEMM traffic formula
#
# Write `tiled_traffic_bytes(M, N, K, BM, BN, dtype_bytes)`. Each output tile loads its full
# `BM x K` panel of A and `K x BN` panel of B, and writes its `BM x BN` block of C once. Tiles at
# the edges are partial, so count only real elements. Hint: each element of A is loaded once for
# every *column* of tiles.

# %% exercise
def tiled_traffic_bytes(M, N, K, BM, BN, dtype_bytes=2):
    ### BEGIN SOLUTION
    a_loads = M * K * math.ceil(N / BN)
    b_loads = K * N * math.ceil(M / BM)
    return (a_loads + b_loads + M * N) * dtype_bytes
    ### END SOLUTION

# %% check
for shape in [(4096, 4096, 4096, 128, 128), (96, 112, 80, 32, 32), (1000, 300, 70, 64, 128), (8, 4096, 4096, 64, 64)]:
    assert tiled_traffic_bytes(*shape) == tiling.gemm_traffic(*shape)["bytes"], shape
_, c = tiling.tiled_matmul(A, B, 32, 32, 16)
assert tiled_traffic_bytes(96, 112, 80, 32, 32, 8) == (c["loads"] + c["stores"]) * 8
print("✅ traffic = (M*K*ceil(N/BN) + K*N*ceil(M/BM) + M*N) x bytes, matching the simulated kernel")

# %% [markdown]
# Look at the last shape, `M = 8`: a decode-sized GEMM with 8 tokens against a 4096x4096
# weight. Its traffic is dominated by reading B (the weights) once. That is the batch-1 decode
# problem in one line: no tile size can raise the intensity above what M allows.
#
# ## Tiles cost shared memory, and shared memory caps occupancy
# A block computing a `BM x BN` tile stages `(BM*BK + BK*BN)` elements per pipeline stage:

# %%
smem = tiling.tile_smem_bytes(128, 128, 32, dtype_bytes=2, stages=3)
for cc in ("8.9", "9.0"):
    o = occ.occupancy(256, regs_per_thread=128, smem_per_block=smem, cc=cc)
    print(f"CC {cc} ({occ.SMS[cc].name}): 128x128x32 bf16, 3 stages = {smem // 1024} KB/block -> "
          f"{o.blocks_per_sm} blocks/SM, {o.active_warps} warps, occupancy {o.occupancy:.0%}, limits {o.limits}")

# %% [markdown]
# ## Exercise 2.2: the register limit, by hand
#
# Registers are the most common occupancy limiter. On CC 7.x to 10.x an SM has 65,536 32-bit
# registers split across **4 sub-partitions** of 16,384. They are allocated **per warp** in units
# of 256 registers. Write `blocks_by_registers(threads_per_block, regs_per_thread)` using these
# rules (assume regs_per_thread <= 255):
#
# 1. registers per warp = `regs_per_thread x 32`, rounded up to a multiple of 256
# 2. warps per sub-partition = `16384 // registers_per_warp`, and warps per SM = 4 x that
# 3. blocks = `warps per SM // ceil(threads_per_block / 32)`

# %% exercise
def blocks_by_registers(threads_per_block, regs_per_thread):
    ### BEGIN SOLUTION
    per_warp = math.ceil(regs_per_thread * 32 / 256) * 256
    warps_per_sm = (16384 // per_warp) * 4
    return warps_per_sm // math.ceil(threads_per_block / 32)
    ### END SOLUTION

# %% check
for threads in (32, 64, 96, 128, 256, 512, 1024):
    for regs in (16, 24, 32, 40, 64, 72, 80, 96, 128, 168, 255):
        want = occ.occupancy(threads, regs_per_thread=regs, cc="9.0").limits["registers"]
        assert blocks_by_registers(threads, regs) == want, (threads, regs, want)
print("✅ blocks_by_registers matches the occupancy calculator for 77 configurations")
print("   e.g. 80 regs, 32 threads:", blocks_by_registers(32, 80), "blocks (not 65536 // 2560 = 25)")

# %% [markdown]
# ## Exercise 2.3: pick a GEMM tile for an L4
#
# You are choosing a tile for a 4096^3 BF16 GEMM on an L4 (CC 8.9). A block has 256 threads
# using 128 registers each. You want the **highest arithmetic intensity** (from
# `tiling.gemm_traffic`) among the candidates that still fit **at least 2 blocks per SM**
# (use `occ.occupancy` with `tiling.tile_smem_bytes`). Two resident blocks let one block's loads
# overlap another's math. Write `choose_tile(candidates)` returning the winning
# `(BM, BN, BK, stages)`.

# %% exercise
candidates = [(64, 64, 32, 2), (128, 128, 32, 3), (128, 256, 32, 3), (256, 128, 64, 2),
              (128, 128, 64, 4), (128, 256, 16, 2)]

def choose_tile(candidates):
    ### BEGIN SOLUTION
    def fits(c):
        bm, bn, bk, st = c
        smem_bytes = tiling.tile_smem_bytes(bm, bn, bk, 2, st)
        return occ.occupancy(256, regs_per_thread=128, smem_per_block=smem_bytes, cc="8.9").blocks_per_sm >= 2
    ok = [c for c in candidates if fits(c)]
    return max(ok, key=lambda c: tiling.gemm_traffic(4096, 4096, 4096, c[0], c[1])["intensity"])
    ### END SOLUTION

# %% check
best = choose_tile(candidates)
assert best == (128, 256, 16, 2), best
print("✅", best, "wins: BM x BN sets the intensity, while BK and stages only set the shared memory,")
print("   so a thin BK buys the big output tile. (A real kernel also pays registers for the 128x256")
print("   accumulator, which is why production GEMMs often run 1 block/SM and hide latency by pipelining.)")

# %% [markdown]
# ## Fusion: the bytes between kernels
# A row-wise softmax over a 4096x4096 BF16 score matrix, the size of one attention head's scores
# at 4k context:

# %%
R = Cc = 4096
for v in ("unfused", "fused", "online"):
    t = tiling.softmax_traffic(R, Cc, 2, v)
    print(f"{v:>8}: {t['kernels']} kernel(s), {t['bytes'] / 2**20:>6.0f} MiB of HBM traffic, "
          f"{t['bytes'] / 3.35e12 * 1e6:>5.1f} us at 3.35 TB/s (simulated arithmetic)")

# %% [markdown]
# Unfused, every step (max, subtract, exp, sum, divide) reads its input from HBM and writes its
# output back, which is about 4x the traffic of a fused kernel, plus five launches instead of one.
# When a row is too long to hold on chip, the **online** version streams the row once to build a
# running max `m` and sum `s`, rescaling `s` by `exp(m_old - m_new)` when the max grows, then
# streams it again to write the output. FlashAttention goes one step further: it fuses the
# softmax *into* the QK^T and PV matmuls, so the score matrix never reaches HBM at all.
#
# ## Exercise 2.4: online softmax
#
# Write `online_softmax(x, block)` for a 2-D array. Keep only a running max and a running sum
# per row while you walk the columns `block` at a time (never the whole row), then produce the
# output. It must match `tiling.softmax_ref` and must not overflow on large inputs.

# %% exercise
def online_softmax(x, block=128):
    x = np.asarray(x, dtype=np.float64)
    ### BEGIN SOLUTION
    m = np.full((x.shape[0], 1), -np.inf)
    s = np.zeros((x.shape[0], 1))
    for c in range(0, x.shape[1], block):
        blk = x[:, c:c + block]
        m_new = np.maximum(m, blk.max(axis=1, keepdims=True))
        s = s * np.exp(m - m_new) + np.exp(blk - m_new).sum(axis=1, keepdims=True)
        m = m_new
    return np.exp(x - m) / s
    ### END SOLUTION

# %% check
x = rng.normal(size=(6, 1000)) * 400                    # exp(400) overflows even float64
y = online_softmax(x, block=64)
assert np.allclose(y, tiling.softmax_ref(x), rtol=1e-12, atol=0) and np.all(np.isfinite(y))
assert np.allclose(online_softmax(x, block=7), y, rtol=1e-12)
print("✅ online softmax is exact for any block size: rescaling by exp(m_old - m_new) keeps the books balanced")

# %% [markdown]
# ## Launch overhead and CUDA Graphs
# A decode step at small batch is hundreds of short kernels. Each eager launch costs CPU time
# (driver plus framework dispatch). If a kernel finishes before the next one is enqueued, the GPU
# idles. The numbers below are **assumptions** (5 us per launch, 10 us to launch a graph); the
# lab measures them on a real GPU.

# %%
for kernel_us in (2.0, 5.0, 20.0):
    ks = [kernel_us] * 384                               # e.g. 32 layers x 12 kernels
    e, g = tiling.step_time(ks, launch_us=5.0), tiling.step_time(ks, launch_us=5.0, graph=True)
    print(f"kernels of {kernel_us:>4} us: eager {e['time_us']:>6.0f} us (GPU idle {e['gpu_idle']:.0%}), "
          f"graph {g['time_us']:>6.0f} us -> {e['time_us'] / g['time_us']:.2f}x")

# %% [markdown]
# ## Exercise 2.5: is this decode step launch-bound?
#
# A decode step at batch 8 runs 40 layers x 10 kernels. Each kernel takes **3 us** on the GPU,
# and eager PyTorch spends **12 us** of CPU time per launch (assumed). A captured graph costs one
# **15 us** launch. Predict, *without* calling `step_time`:
#
# * `eager_us`: the eager step time
# * `graph_us`: the graph step time
# * `breakeven_kernel_us`: the kernel duration above which the eager step becomes GPU-bound,
#   meaning graphs would save at most one launch

# %% exercise
### BEGIN SOLUTION
n_kernels = 40 * 10
eager_us = n_kernels * 12 + 3          # the CPU is the bottleneck: the last launch plus one kernel
graph_us = 15 + n_kernels * 3          # one launch, then back to back
breakeven_kernel_us = 12               # kernels longer than a launch keep the GPU fed
### END SOLUTION

# %% check
ks = [3.0] * 400
assert eager_us == tiling.step_time(ks, launch_us=12)["time_us"]
assert graph_us == tiling.step_time(ks, launch_us=12, graph=True, graph_launch_us=15)["time_us"]
assert tiling.step_time([breakeven_kernel_us + 0.5] * 400, 12)["gpu_idle"] < 0.01
assert tiling.step_time([breakeven_kernel_us - 0.5] * 400, 12)["gpu_idle"] > 0.03
print(f"✅ eager {eager_us:.0f} us vs graph {graph_us:.0f} us ({eager_us / graph_us:.1f}x): "
      "graphs pay off when kernels are shorter than launches")

# %% [markdown]
# ## Occupancy is a means: Little's law
# To sustain bandwidth B with latency L, B x L bytes must be in flight at every instant. With an
# **assumed** 600 ns loaded DRAM latency, compare how many independent 4-byte loads each resident
# thread must keep outstanding:

# %%
for dev in ("L4", "A100-80GB", "H100-SXM"):
    d = occ.DEVICES[dev]
    per_sm = occ.bytes_in_flight(d["hbm_Bps"], 600e-9) / d["sms"]
    print(f"{dev:>9}: {per_sm / 1024:5.1f} KB in flight per SM -> "
          f"{occ.loads_in_flight_per_thread(dev, 600e-9):.2f} x 4-byte loads/thread at full occupancy, "
          f"{occ.loads_in_flight_per_thread(dev, 600e-9, bytes_per_load=16):.2f} x 16-byte loads")
print("tail effect:", occ.waves(140, n_sms=132, blocks_per_sm=1), "<- 140 blocks on 132 SMs")

# %% [markdown]
# An L4 saturates its 300 GB/s with half its threads each holding one load. An H100 needs almost
# two outstanding 4-byte loads per thread even at full occupancy, so fast kernels use 16-byte
# vector loads, several independent loads per thread (ILP), or asynchronous copies (cp.async,
# TMA) that do not tie up threads at all. That is why well-tuned GEMM and attention kernels run
# at *low* occupancy and still saturate the machine.
#
# ## In a design review
#
# **The two-minute version.** "Our kernels are memory-bound, so we count bytes. Tiling makes
# each byte fetched from HBM feed many FLOPs from shared memory and registers: a 128-wide tile
# cuts GEMM traffic about 128x, and L2 captures the reuse between tiles. Fusion removes the
# round trips between kernels. A fused softmax moves 2 bytes per element instead of about 8, and
# FlashAttention never writes the score matrix at all. At small batch, decode is hundreds of
# microsecond-scale kernels, so the CPU launch path becomes the bottleneck and we capture CUDA
# graphs per batch size. We size occupancy with Little's law, not as a target: enough bytes in
# flight to cover latency, whether from warps, ILP or async copies."
#
# **Drill questions**
#
# 1. *FlashAttention does the same FLOPs. Why is it 2-4x faster?* Attention was memory-bound on
#    the N x N score matrix going to HBM and back. Fusing QK^T, softmax and PV into one tiled
#    kernel removes that traffic, so time drops to near the matmuls' own cost.
# 2. *Profiling shows gaps between tiny kernels in decode. What do you try?* CUDA graphs (capture
#    per batch-size bucket), fusing elementwise ops (torch.compile), and removing host syncs
#    (`.item()`, prints) from the step.
# 3. *Should we push occupancy to 100%?* Only if latency is not covered. Big-tile GEMMs trade
#    occupancy for register reuse and hide latency with pipelined async copies. Check achieved
#    bandwidth and stall reasons, not the occupancy number.
