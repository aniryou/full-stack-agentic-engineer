"""Reduction: many values -> one, with shared memory and barriers.

The one idea: threads of one block cooperate through **shared memory** (on-chip, per block) and
**``cuda.syncthreads()``** (a barrier for the block). Each block reduces its slice with a tree —
half the threads add pairs, then a quarter, ... — in ``log2(threads)`` steps. Blocks cannot
synchronise with each other inside a kernel, so the per-block results are combined either by

* **a second pass** (launch again on the partial sums until one value is left): deterministic, or
* **atomics** (``cuda.atomic.add`` into one cell): one launch, but the order of the additions depends
  on which block finishes first, so the float result can differ in the last bits from run to run.

The same determinism question reappears one layer up: NCCL's all-reduce also fixes an order.

Design choices in the tree (primer §2-§3): each thread first adds *two* elements while loading
(halves the idle threads), and the tree uses *sequential addressing* (``buf[tid] += buf[tid+stride]``)
so active threads stay contiguous: whole warps retire together instead of diverging, and
consecutive lanes hit consecutive shared-memory banks.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numba import cuda, float32

from . import blocks_for


@lru_cache(maxsize=None)
def make_block_sum(threads: int = 256, atomic: bool = False):
    """Build the block-reduction kernel ``block_sum(x, out)`` for a power-of-two block size.

    ``threads`` and ``atomic`` are Python constants captured by the closure, so Numba compiles them
    as constants: the shared array gets a fixed size and the unused ``if atomic`` branch is removed.
    Non-atomic: ``out[blockIdx.x]`` receives the block's partial sum. Atomic: all blocks add into ``out[0]``.
    """
    if threads < 2 or threads & (threads - 1):
        raise ValueError("threads must be a power of two >= 2")

    @cuda.jit
    def block_sum(x, out):
        buf = cuda.shared.array(threads, float32)
        tid = cuda.threadIdx.x
        i = cuda.blockIdx.x * (threads * 2) + tid
        s = float32(0.0)
        if i < x.size:  # first add happens during the load
            s = x[i]
        if i + threads < x.size:
            s += x[i + threads]
        buf[tid] = s
        cuda.syncthreads()
        stride = threads // 2
        while stride > 0:
            if tid < stride:
                buf[tid] += buf[tid + stride]
            cuda.syncthreads()  # every thread, every step — outside the if
            stride //= 2
        if tid == 0:
            if atomic:
                cuda.atomic.add(out, 0, buf[0])
            else:
                out[cuda.blockIdx.x] = buf[0]

    return block_sum


def blocks_for_sum(n: int, threads: int) -> int:
    """Each block consumes ``2 * threads`` elements."""
    return blocks_for(n, 2 * threads)


def run_sum(x, threads: int = 256, atomic: bool = False) -> float:
    """Sum a float32 vector on the device.

    Two-pass (default): launch ``block_sum`` on the partials until one value is left — for n = 2**24
    and 256 threads that is 32768 -> 64 -> 1, three launches. Atomic: one launch.
    """
    d = cuda.to_device(np.ascontiguousarray(x, dtype=np.float32))
    if atomic:
        out = cuda.to_device(np.zeros(1, dtype=np.float32))
        make_block_sum(threads, True)[blocks_for_sum(d.size, threads), threads](d, out)
        return float(out.copy_to_host()[0])
    kernel = make_block_sum(threads, False)
    while d.size > 1:
        blocks = blocks_for_sum(d.size, threads)
        partial = cuda.device_array(blocks, dtype=np.float32)
        kernel[blocks, threads](d, partial)
        d = partial
    return float(d.copy_to_host()[0])


def launches_for_two_pass(n: int, threads: int = 256) -> int:
    """How many kernel launches the two-pass scheme needs for n elements."""
    launches = 0
    while n > 1:
        n = blocks_for_sum(n, threads)
        launches += 1
    return launches
