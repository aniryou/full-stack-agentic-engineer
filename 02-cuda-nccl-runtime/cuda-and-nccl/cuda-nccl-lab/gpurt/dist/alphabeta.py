"""The α-β model: every transfer costs a fixed latency plus bytes over bandwidth.

    t(S) = α + S / B

α (seconds) is the per-operation cost that does not depend on size — launching the collective,
ranks synchronising, protocol handshakes, the first byte crossing the link. B (bytes/s) is the
asymptotic rate. Two consequences worth being able to derive on a whiteboard:

* the achieved bandwidth ``S / t(S) = B · S / (S + α·B)`` reaches half of B at the
  **half-performance size S½ = α·B** (Hockney's n½). Below ~S½ an operation is *latency-bound*
  (a faster link barely helps; fewer steps do); above it, *bandwidth-bound*.
* a ring all-reduce over n ranks with per-hop latency α and per-link bandwidth B takes
  ``2(n-1)·α + 2(n-1)/n · S/B``: the bandwidth term is flat in n but the latency term grows
  linearly — why NCCL uses trees (log n steps), NVLink SHARP or one-shot kernels for small messages,
  and why a decode step's small tensor-parallel all-reduces are a latency problem (primer §5).

:func:`fit` recovers (α, B) from measured (size, time) pairs by least squares on *relative* error,
so a 10 µs point and a 10 ms point weigh the same. The same model fits a kernel bandwidth sweep
(α = launch overhead, B = DRAM bandwidth) — one mental model for both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .busbw import bus_factor


@dataclass(frozen=True)
class AlphaBeta:
    alpha_s: float  # latency term, seconds
    bw_Bps: float  # asymptotic (algorithm) bandwidth, bytes/s
    rel_rms: float = 0.0  # relative RMS error of the fit

    def time(self, size_bytes: float) -> float:
        return self.alpha_s + size_bytes / self.bw_Bps

    def algbw_gbps(self, size_bytes: float) -> float:
        return size_bytes / self.time(size_bytes) / 1e9

    @property
    def n_half(self) -> float:
        """Bytes at which half of the asymptotic bandwidth is reached (α·B)."""
        return self.alpha_s * self.bw_Bps

    def regime(self, size_bytes: float) -> str:
        return "latency-bound" if size_bytes < self.n_half else "bandwidth-bound"

    def peak_busbw_gbps(self, op: str, n: int) -> float:
        return self.bw_Bps * bus_factor(op, n) / 1e9

    def __str__(self) -> str:
        return (f"α = {self.alpha_s * 1e6:.1f} µs, B = {self.bw_Bps / 1e9:.2f} GB/s, "
                f"S½ = {self.n_half / 1024:.0f} KiB (fit error {self.rel_rms:.1%})")


def fit(sizes, times) -> AlphaBeta:
    """Fit ``t = α + S/B`` minimising Σ((α + βS_i - t_i)/t_i)², β = 1/B. α is clamped at 0."""
    S = np.asarray(sizes, dtype=float)
    t = np.asarray(times, dtype=float)
    if S.shape != t.shape or S.size < 2:
        raise ValueError("need at least two (size, time) points of equal length")
    if np.any(t <= 0) or np.any(S < 0):
        raise ValueError("times must be positive and sizes non-negative")
    w = 1.0 / t
    X = np.column_stack([w, S * w])
    (alpha, beta), *_ = np.linalg.lstsq(X, np.ones_like(t), rcond=None)
    if alpha < 0:  # every point is bandwidth-bound: fit through the origin
        alpha = 0.0
        u = S * w
        beta = float(np.sum(u) / np.sum(u * u))
    if beta <= 0:
        raise ValueError("time does not grow with size; no bandwidth to fit")
    rel = float(np.sqrt(np.mean(((alpha + beta * S - t) / t) ** 2)))
    return AlphaBeta(float(alpha), 1.0 / beta, rel)


def _field(row, key):
    return row[key] if isinstance(row, dict) else getattr(row, key)


def fit_rows(rows, min_size: int = 0) -> AlphaBeta:
    """Fit sweep rows — ``gpurt.dist`` Rows or dicts with ``size`` (bytes) and ``time_us`` — e.g. from
    ``gpurt.nccltests`` or a pipes/gloo/NCCL sweep. Rows smaller than ``min_size`` are ignored."""
    pts = [(_field(r, "size"), _field(r, "time_us") * 1e-6) for r in rows
           if _field(r, "size") >= max(min_size, 1)]
    return fit([p[0] for p in pts], [p[1] for p in pts])


def ring_allreduce_time(size_bytes: float, n: int, alpha_s: float, link_Bps: float) -> float:
    """2(n-1) latency hops + 2(n-1)/n of the buffer over each link."""
    return 2 * (n - 1) * alpha_s + 2 * (n - 1) / n * size_bytes / link_Bps


def ring_allgather_time(size_bytes: float, n: int, alpha_s: float, link_Bps: float) -> float:
    """(n-1) hops; ``size_bytes`` is the gathered output size, as in nccl-tests."""
    return (n - 1) * alpha_s + (n - 1) / n * size_bytes / link_Bps


def synthetic_times(sizes, alpha_s: float, bw_Bps: float, noise: float = 0.0, seed: int = 0) -> list[float]:
    """Times the α-β model *predicts* (optionally with multiplicative noise) — simulated data, for
    exercising a fit. Never present these as measurements."""
    rng = np.random.default_rng(seed)
    return [(alpha_s + s / bw_Bps) * math.exp(rng.normal(0.0, noise)) if noise else alpha_s + s / bw_Bps
            for s in sizes]
