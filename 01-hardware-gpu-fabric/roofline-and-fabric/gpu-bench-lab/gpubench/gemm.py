"""GEMM throughput: how close to its FLOP/s roof this machine gets, by size, shape and dtype.

GEMM is the one LLM operation that can be compute-bound, so it is how you find the flat roof.
Three things decide how close you get:

* **size** — a small GEMM cannot keep every core (CPU) or SM (GPU) busy, and its fixed costs
  (thread wake-up, kernel launch) are not amortised over enough FLOPs;
* **shape** — a skinny ``m``-row GEMM (decode at batch ``m``) has intensity of about ``m``
  FLOPs per byte at 2-byte weights, whatever the hardware (``decode_shapes``);
* **dtype** — each halving of the element width roughly doubles the tensor-core rate *and*
  halves the bytes (and a dtype without a hardware path — numpy float16 — is catastrophically slow).

``sweep`` measures a grid of sizes × dtypes and keeps a note for anything it had to skip.
"""
from __future__ import annotations

from .accounting import canonical_dtype
from .backends import BackendUnavailable
from .measure import Measurement, measure

SIZES = {
    "numpy": {"quick": [128, 256, 512, 1024], "full": [64, 128, 256, 512, 1024, 2048, 4096]},
    "torch": {"quick": [1024, 2048, 4096], "full": [256, 512, 1024, 2048, 4096, 8192, 16384]},
}
NUMPY_FLOAT16_MAX = 256          # numpy float16 runs at ~1 GFLOP/s: keep it small


def run_gemm(be, m: int, n: int, k: int, dtype: str, *, repeats: int = 3, min_time: float = 0.05,
             check: bool = False) -> Measurement:
    """Time one ``(m×k)·(k×n)`` GEMM; ``check=True`` also verifies the product first."""
    dtype = canonical_dtype(dtype)
    op = be.make_gemm(m, n, k, dtype)
    if check and op.verify is not None:
        op.fn()
        if not op.verify():
            raise AssertionError(f"GEMM {m}x{n}x{k} {dtype} produced a wrong result")
    return measure(be, op, "gemm", {"m": m, "n": n, "k": k, "dtype": dtype}, repeats=repeats, min_time=min_time)


def sweep(be, sizes=None, dtypes=None, quick: bool = True, repeats: int = 3, min_time: float = 0.05,
          skipped: list | None = None) -> list:
    """Square GEMMs over ``sizes`` × ``dtypes``. Anything that cannot run (FP8 on an old GPU,
    out of memory, a dtype the library rejects) is appended to ``skipped`` with the reason."""
    sizes = sizes or SIZES[be.name]["quick" if quick else "full"]
    dtypes = [canonical_dtype(d) for d in (dtypes or be.gemm_dtypes())]
    out = []
    for dtype in dtypes:
        for s in sizes:
            if be.name == "numpy" and dtype == "float16" and s > NUMPY_FLOAT16_MAX:
                continue
            try:
                out.append(run_gemm(be, s, s, s, dtype, repeats=repeats, min_time=min_time))
            except (BackendUnavailable, RuntimeError, MemoryError, TypeError, ValueError) as e:
                if skipped is not None:
                    skipped.append({"dtype": dtype, "size": s, "reason": f"{type(e).__name__}: {e}"[:200]})
    return out


def decode_shapes(d_model: int, batches, d_out: int | None = None) -> list:
    """The decode projection ``(batch × d_model)·(d_model × d_out)`` for each batch size:
    the weights are read once per step whatever the batch, so intensity grows with batch."""
    d_out = d_out or d_model
    return [(b, d_out, d_model) for b in batches]


def best(measurements, dtype: str | None = None, stat: str = "best") -> Measurement:
    """The measurement with the highest FLOP/s (optionally of one dtype)."""
    ms = [m for m in measurements if dtype is None or m.params.get("dtype") == canonical_dtype(dtype)]
    if not ms:
        raise ValueError(f"no GEMM measurements{' for ' + dtype if dtype else ''}")
    return max(ms, key=lambda m: m.flops_per_s(stat))


def best_by_dtype(measurements, stat: str = "best") -> dict:
    return {d: best(measurements, d, stat) for d in dict.fromkeys(m.params["dtype"] for m in measurements)}
