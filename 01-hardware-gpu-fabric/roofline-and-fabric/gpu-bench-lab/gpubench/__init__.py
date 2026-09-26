"""gpubench — measure the machine you have.

GEMM throughput by size and dtype, memory bandwidth (STREAM), host↔device and GPU↔GPU
transfers, weights loading from disk, and the roofline those numbers imply — with a numpy
backend that measures any CPU (tier T0) and a PyTorch backend that measures CUDA GPUs (T1/T2).

Start with ``get_backend("auto")``; the notebooks walk through every module. The concepts are
in the topic primer (``../PRIMER.md``, sections 1–6).
"""
__version__ = "0.1.0"

from .accounting import OpCost, gemm_cost, stream_cost, transfer_cost  # noqa: E402
from .backends import BackendUnavailable, get_backend  # noqa: E402
from .measure import Measurement, Op, measure, si  # noqa: E402
from .roofline import Roofline, attainable, ridge  # noqa: E402

__all__ = ["__version__", "OpCost", "gemm_cost", "stream_cost", "transfer_cost", "BackendUnavailable",
           "get_backend", "Measurement", "Op", "measure", "si", "Roofline", "attainable", "ridge"]
