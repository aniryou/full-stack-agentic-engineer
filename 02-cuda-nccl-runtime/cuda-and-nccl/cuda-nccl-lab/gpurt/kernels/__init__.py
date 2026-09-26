"""CUDA kernels written with Numba — one source, two targets.

The one idea: a CUDA kernel is the *body of a loop*; the launch ``kernel[blocks, threads](...)``
runs one copy of it per thread, and each copy finds its loop index from ``blockIdx``,
``blockDim`` and ``threadIdx``. Everything in this package is written against ``numba.cuda`` so
that the **same source** runs

* on the CPU, in Numba's CUDA simulator (``NUMBA_ENABLE_CUDASIM=1``): every CUDA thread becomes a
  Python thread, shared memory is a shared NumPy array, ``cuda.syncthreads()`` is a real barrier and
  out-of-bounds accesses raise ``IndexError``. Correct, instructive, slow — T0;
* on a real NVIDIA GPU (numba-cuda installed, a driver present): compiled to PTX and run — T1.

Which one you get is decided by :func:`gpurt.env.ensure_numba_mode`, called here before
``numba.cuda`` is imported. Modules:

    elementwise  vector add, SAXPY with a grid-stride loop, copy           (primer §2)
    reduction    shared-memory tree reduction; two-pass vs atomics          (primer §2-§3)
    matmul       naive vs shared-memory tiled GEMM                          (primer §3)
    transpose    naive vs shared-memory tiled transpose, bank-conflict pad (primer §3)
    softmax      unfused (4 kernels) vs fused online softmax                (primer §3)
    trace        record every global access in the simulator -> sectors per warp request
    traffic      the byte and FLOP counts each kernel should move (the model to check against)
    bench        timing on a real GPU with CUDA events (T1)
    triton_kernels  the same ideas in Triton (optional, T1, lazy)
"""

from __future__ import annotations

from gpurt.env import ensure_numba_mode, fast_simulator

MODE = ensure_numba_mode()
SIMULATOR = MODE == "simulator"

from numba import cuda  # noqa: E402  (must follow the mode decision)


def blocks_for(n: int, threads: int) -> int:
    """Blocks needed so that ``blocks * threads >= n`` (ceil division; at least one block)."""
    if threads <= 0:
        raise ValueError("threads must be positive")
    return max(1, (n + threads - 1) // threads)


__all__ = ["MODE", "SIMULATOR", "blocks_for", "cuda", "fast_simulator"]
