"""Backends: the same benchmarks on a CPU (numpy, tier T0) or a CUDA GPU (PyTorch, T1/T2).

A backend knows how to *build* an operation on its device (``make_gemm``, ``make_stream``,
...) and how to *time* it (wall clock on the CPU; CUDA events on the GPU). Everything else —
what to count, which sizes to sweep, how to report — is shared, so a CPU number and a GPU
number in the same table were produced by the same accounting.

``get_backend("auto")`` picks torch when PyTorch and a CUDA GPU are both present, and numpy
otherwise, and tells you why. PyTorch is only imported inside ``TorchBackend``, so nothing
here needs it installed.
"""
from __future__ import annotations


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run here (no PyTorch, no GPU, unsupported dtype...)."""


def get_backend(name: str = "auto", verbose: bool = True, **kwargs):
    """Return a backend instance: ``"numpy"``, ``"torch"``, or ``"auto"`` (torch if a CUDA GPU
    is usable, else numpy). ``"torch"`` raises ``BackendUnavailable`` with the reason."""
    from .numpy_backend import NumpyBackend

    if name == "numpy":
        return NumpyBackend(**kwargs)
    if name in ("torch", "cuda"):
        from .torch_backend import TorchBackend
        return TorchBackend(**kwargs)
    if name != "auto":
        raise ValueError("backend must be 'auto', 'numpy' or 'torch'")
    try:
        from .torch_backend import TorchBackend
        be = TorchBackend(**kwargs)
        if verbose:
            print(f"gpubench: using the torch backend on {be.device} ({be.describe()['name']})")
        return be
    except BackendUnavailable as why:
        if verbose:
            print(f"gpubench: using the numpy backend (CPU, tier T0) — {why}")
        return NumpyBackend()


def gpu_available() -> bool:
    """True when the torch backend would work (PyTorch installed and a CUDA device visible)."""
    try:
        import torch  # lazy: probed here, never required
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def gpu_count() -> int:
    """CUDA devices PyTorch can see (0 without PyTorch or a GPU)."""
    if not gpu_available():
        return 0
    import torch
    return torch.cuda.device_count()


__all__ = ["BackendUnavailable", "get_backend", "gpu_available", "gpu_count"]
