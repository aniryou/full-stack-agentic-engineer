"""Matrix multiply: naive vs shared-memory tiled — the kernel behind every GEMM lesson.

The one idea: **reuse**. C = A·B needs 2·M·N·K FLOPs but only (M·K + K·N + M·N) distinct elements.
The naive kernel (one thread per C element) asks the memory system for 2·K elements per thread —
2·M·N·K loads in total — and relies on caches to absorb the redundancy. The tiled kernel stages a
``tile x tile`` block of A and of B in **shared memory** once, then every thread of the block reads
it ``tile`` times from on-chip SRAM: global loads fall by a factor of ``tile`` (2·M·N·K / tile).

Two barriers per tile, both required:
  1. after loading — nobody may read the tile before every thread has written its element;
  2. after computing — nobody may overwrite the tile while a slower thread is still reading it.

Thread-to-data mapping: ``threadIdx.x`` (the fastest-varying index, consecutive lanes of a warp)
walks **columns**, so a warp's loads of B and stores of C touch consecutive addresses (coalesced),
while its loads of A hit one address (a broadcast). ``gpurt.kernels.trace`` measures exactly that.

Real GEMMs (cuBLAS, CUTLASS, Triton) add register tiling, vectorised loads, double buffering and
tensor cores on top; this is the first rung of that ladder (primer §3).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from numba import cuda, float32

from . import blocks_for


@cuda.jit
def matmul_naive(A, B, C):
    col, row = cuda.grid(2)  # x -> column (coalesced across a warp), y -> row
    if row < C.shape[0] and col < C.shape[1]:
        acc = float32(0.0)
        for k in range(A.shape[1]):
            acc += A[row, k] * B[k, col]
        C[row, col] = acc


@lru_cache(maxsize=None)
def make_matmul_tiled(tile: int = 16):
    """Build a tiled GEMM kernel for a given tile size (a compile-time constant, launch with
    ``block=(tile, tile)``). Handles any M, N, K: out-of-range tile elements are loaded as zero."""
    if tile < 1:
        raise ValueError("tile must be >= 1")

    @cuda.jit
    def matmul_tiled(A, B, C):
        sA = cuda.shared.array((tile, tile), float32)
        sB = cuda.shared.array((tile, tile), float32)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        row = cuda.blockIdx.y * tile + ty
        col = cuda.blockIdx.x * tile + tx
        M = A.shape[0]
        K = A.shape[1]
        N = B.shape[1]
        acc = float32(0.0)
        for t in range((K + tile - 1) // tile):
            k0 = t * tile
            if row < M and k0 + tx < K:
                sA[ty, tx] = A[row, k0 + tx]
            else:
                sA[ty, tx] = 0.0
            if k0 + ty < K and col < N:
                sB[ty, tx] = B[k0 + ty, col]
            else:
                sB[ty, tx] = 0.0
            cuda.syncthreads()  # 1: the whole tile is loaded
            for j in range(tile):
                acc += sA[ty, j] * sB[j, tx]
            cuda.syncthreads()  # 2: everyone has finished reading it
        if row < M and col < N:
            C[row, col] = acc

    return matmul_tiled


matmul_tiled = make_matmul_tiled(16)


def launch_config(M: int, N: int, tile: int = 16):
    """``(grid, block)`` covering an M x N output with ``tile x tile`` blocks (grid is (x=cols, y=rows))."""
    return (blocks_for(N, tile), blocks_for(M, tile)), (tile, tile)


def run_matmul(A, B, variant: str = "tiled", tile: int = 16) -> np.ndarray:
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shapes {A.shape} and {B.shape} do not multiply")
    M, N = A.shape[0], B.shape[1]
    kernel = matmul_naive if variant == "naive" else make_matmul_tiled(tile)
    grid, block = launch_config(M, N, tile)
    d_C = cuda.device_array((M, N), dtype=np.float32)
    kernel[grid, block](cuda.to_device(A), cuda.to_device(B), d_C)
    return d_C.copy_to_host()
