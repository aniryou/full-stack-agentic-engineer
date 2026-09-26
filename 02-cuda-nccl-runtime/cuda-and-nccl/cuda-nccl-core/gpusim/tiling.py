"""Tiling and fusion: a memory-bound kernel costs its bytes divided by bandwidth, plus its launch.
So the optimisations that matter either cut bytes (tiling reuses data from on-chip memory; fusion
skips round trips through HBM) or cut launches (fusion, and CUDA Graphs, which replay a whole
sequence of kernels as one launch).

Byte counts are global-memory traffic (what crosses L2/HBM): elements x dtype bytes.
"""
from __future__ import annotations

import math

import numpy as np


def gemm_traffic(M: int, N: int, K: int, BM: int | None = None, BN: int | None = None,
                 dtype_bytes: int = 2) -> dict:
    """Global traffic of C[M,N] = A[M,K] @ B[K,N].

    BM = BN = 1   naive: each output element streams its own row of A and column of B
    BM, BN        tiled: each A element is reloaded once per *column* of output tiles
                  (ceil(N/BN) times) and each B element once per *row* of tiles (ceil(M/BM))
    None          compulsory: every input read once (the ideal-cache bound the roofline uses)
    """
    if BM is None or BN is None:
        a, b = M * K, K * N
    else:
        a, b = M * K * math.ceil(N / BN), K * N * math.ceil(M / BM)
    c = M * N
    flops = 2 * M * N * K
    nbytes = (a + b + c) * dtype_bytes
    return {"flops": flops, "bytes": nbytes, "intensity": flops / nbytes,
            "a_elems": a, "b_elems": b, "c_elems": c}


def tile_smem_bytes(BM: int, BN: int, BK: int, dtype_bytes: int = 2, stages: int = 1) -> int:
    """Shared memory one block needs to stage an A tile (BM x BK) and a B tile (BK x BN);
    `stages` > 1 is multi-buffering, loading slab k+1 while computing on slab k."""
    return stages * (BM * BK + BK * BN) * dtype_bytes


def tiled_matmul(A: np.ndarray, B: np.ndarray, BM: int = 32, BN: int = 32, BK: int = 32):
    """A shared-memory-tiled GEMM run on the CPU. Each (i, j) output tile is one 'thread block';
    it walks K in BK-wide slabs, staging an A tile and a B tile through 'shared memory'.
    Returns C and the elements it loaded/stored from global memory, which equal
    gemm_traffic(M, N, K, BM, BN)'s a_elems + b_elems and c_elems."""
    M, K = A.shape
    N = B.shape[1]
    C = np.zeros((M, N), dtype=np.result_type(A, B))
    loads = stores = 0
    for i in range(0, M, BM):
        for j in range(0, N, BN):
            acc = np.zeros((min(BM, M - i), min(BN, N - j)), dtype=C.dtype)
            for k in range(0, K, BK):
                a, b = A[i:i + BM, k:k + BK], B[k:k + BK, j:j + BN]    # global -> shared
                loads += a.size + b.size
                acc += a @ b                                            # shared -> registers
            C[i:i + BM, j:j + BN] = acc                                 # registers -> global
            stores += acc.size
    return C, {"loads": loads, "stores": stores}


def softmax_ref(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def softmax_traffic(rows: int, cols: int, dtype_bytes: int = 2, variant: str = "unfused") -> dict:
    """HBM bytes and kernel launches of a row-wise softmax over a rows x cols matrix.

    unfused  5 kernels (max, subtract, exp, sum, divide), each a round trip through HBM:
             reads 5RC + 2R, writes 3RC + 2R
    fused    1 kernel, a row fits on chip: read x once, write y once (2RC)
    online   1 kernel for rows too long for the chip: pass 1 streams x keeping a running max and
             sum (online softmax), pass 2 re-reads x and writes y (3RC); FlashAttention's trick
    """
    R, C = rows, cols
    elems = {"unfused": 8 * R * C + 4 * R, "fused": 2 * R * C, "online": 3 * R * C}[variant]
    return {"bytes": elems * dtype_bytes, "kernels": 5 if variant == "unfused" else 1}


def softmax_online(x: np.ndarray, block: int = 128) -> np.ndarray:
    """Row-wise softmax that only ever looks at `block` columns at a time. The running max m and
    running sum l are rescaled by exp(m_old - m_new) whenever a bigger value arrives. Exact."""
    x = np.asarray(x, dtype=np.float64)
    m = np.full(x.shape[:-1] + (1,), -np.inf)
    s = np.zeros_like(m)
    for c in range(0, x.shape[-1], block):
        blk = x[..., c:c + block]
        m_new = np.maximum(m, blk.max(axis=-1, keepdims=True))
        s = s * np.exp(m - m_new) + np.exp(blk - m_new).sum(axis=-1, keepdims=True)
        m = m_new
    return np.exp(x - m) / s


def elementwise_traffic(n: int, ops: int, dtype_bytes: int = 2, fused: bool = False) -> dict:
    """A chain of `ops` unary elementwise ops on n elements: unfused, every op reads and writes
    all n; fused, the chain reads once and writes once in a single kernel."""
    return {"bytes": 2 * n * (1 if fused else ops) * dtype_bytes, "kernels": 1 if fused else ops}


def step_time(kernel_us, launch_us: float = 5.0, graph: bool = False,
              graph_launch_us: float = 10.0) -> dict:
    """Wall time of a sequence of dependent kernels on one stream.

    Eager: the CPU enqueues kernel i at (i+1) x launch_us; the GPU starts it once it is enqueued
    AND the previous kernel has finished. When kernels are shorter than the launch cost, the GPU
    waits on the CPU ("launch-bound").
    Graph: one launch for the whole captured sequence, then the kernels run back to back.
    """
    busy = float(sum(kernel_us))
    if graph:
        t = graph_launch_us + busy
    else:
        t = cpu = 0.0
        for k in kernel_us:
            cpu += launch_us
            t = max(t, cpu) + k
    return {"time_us": t, "busy_us": busy, "gpu_idle": 1 - busy / t if t else 0.0}
