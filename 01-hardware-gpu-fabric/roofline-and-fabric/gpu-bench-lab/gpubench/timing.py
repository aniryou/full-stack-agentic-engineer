"""Timing a benchmark honestly — and the α-β model that turns a size sweep into two numbers.

Four rules, enforced by ``bench``:

1. **Warm up.** The first call pays for things you are not trying to measure: page faults
   on fresh buffers, thread-pool start-up, library autotuning (cuBLAS picks a kernel),
   CPU/GPU clocks ramping up. Run the operation first, then start the clock.
2. **Make each sample long.** A sample repeats the call ``inner`` times, with ``inner``
   grown until the sample lasts ``min_time``. Timer resolution and per-call overhead then
   stop mattering (unless the overhead *is* the thing, which the α in α-β captures).
3. **Take several samples and say which statistic you report.** ``best`` (fastest sample)
   estimates what the machine *can* do — STREAM reports it; ``median`` is what you would
   *typically* see. Report both; never a single unlabelled run.
4. **Synchronise asynchronous devices.** A GPU call returns before the GPU finishes. Wall-
   clock timing must ``sync`` before starting and before stopping the clock; the torch
   backend instead brackets work with CUDA events.

``fit_alpha_beta`` fits ``t(n) = α + n/β`` to a size sweep: α is the fixed cost per call or
message (latency, launch, syscall), β the asymptotic bandwidth, and ``n½ = α·β`` the size at
which you reach half of β. It is the same model the primer uses for links in §5.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Timing:
    """Seconds per call, one entry per sample (each sample averaged over ``inner`` calls)."""
    samples: tuple
    inner: int = 1
    warmup: int = 0
    method: str = "wall"      # "wall" (perf_counter, synchronised) | "cuda-events"

    @property
    def best(self) -> float:
        return min(self.samples)

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    @property
    def worst(self) -> float:
        return max(self.samples)

    @property
    def cv(self) -> float:
        """Coefficient of variation (stdev / mean): above ~0.05, suspect a noisy machine."""
        if len(self.samples) < 2 or self.mean == 0:
            return 0.0
        return statistics.stdev(self.samples) / self.mean

    def stat(self, which: str = "best") -> float:
        if which not in ("best", "median", "mean", "worst"):
            raise ValueError("stat must be best, median, mean or worst")
        return getattr(self, which)

    def to_dict(self) -> dict:
        return {"samples": list(self.samples), "inner": self.inner, "warmup": self.warmup, "method": self.method,
                "best": self.best, "median": self.median, "cv": self.cv}

    @classmethod
    def from_dict(cls, d: dict) -> "Timing":
        return cls(tuple(d["samples"]), d.get("inner", 1), d.get("warmup", 0), d.get("method", "wall"))


def bench(fn: Callable[[], object], *, sync: Callable[[], object] | None = None, warmup: int = 1,
          repeats: int = 5, min_time: float = 0.02, max_inner: int = 1 << 16,
          setup: Callable[[], object] | None = None,
          clock: Callable[[], float] = time.perf_counter) -> Timing:
    """Time ``fn`` by the four rules above and return a ``Timing``.

    ``setup`` runs before every sample, outside the clock (e.g. dropping the page cache for
    a cold read); pass ``max_inner=1`` with it so each sample is exactly one call.
    ``clock`` is injectable so the tests can drive a fake one.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        if setup:
            setup()
        fn()
    sync()

    def sample(inner: int) -> float:
        if setup:
            setup()
        sync()
        t0 = clock()
        for _ in range(inner):
            fn()
        sync()
        return clock() - t0

    # calibrate: grow `inner` until one sample lasts min_time (this also warms up further)
    inner = 1
    while inner < max_inner:
        dt = sample(inner)
        if dt >= min_time:
            break
        grow = 2 * inner if dt <= 0 else math.ceil(inner * 1.2 * min_time / dt)
        inner = min(max_inner, max(2 * inner, grow))
    samples = tuple(sample(inner) / inner for _ in range(repeats))
    return Timing(samples=samples, inner=inner, warmup=warmup, method="wall")


# -- the α-β model -------------------------------------------------------------------------------
@dataclass(frozen=True)
class AlphaBeta:
    """``t(n) = alpha + n / beta``: fixed cost ``alpha`` (s) plus ``n`` bytes at ``beta`` (B/s)."""
    alpha: float
    beta: float
    r2: float = 1.0

    @property
    def n_half(self) -> float:
        """Hockney's n½ = α·β: the size at which achieved bandwidth is half of β."""
        return self.alpha * self.beta

    def time(self, n: float) -> float:
        return self.alpha + n / self.beta

    def bandwidth(self, n: float) -> float:
        """Achieved bandwidth at size ``n``: ``n / t(n)`` — approaches β only for n >> n½."""
        return n / self.time(n)

    def to_dict(self) -> dict:
        return {"alpha_s": self.alpha, "beta_Bps": self.beta, "n_half_bytes": self.n_half, "r2": self.r2}


def fit_alpha_beta(sizes: Sequence[float], times: Sequence[float], relative: bool = True) -> AlphaBeta:
    """Least-squares fit of ``t = α + n/β``.

    With ``relative=True`` each point is weighted by ``1/t`` so the fit minimises *relative*
    error: small sizes (which pin α) count as much as large ones (which pin β). Plain least
    squares would let the largest transfer decide everything.
    """
    import numpy as np

    n = np.asarray(sizes, dtype=float)
    t = np.asarray(times, dtype=float)
    if n.shape != t.shape or n.size < 2:
        raise ValueError("need at least two (size, time) pairs of equal length")
    w = 1.0 / t if relative else np.ones_like(t)
    design = np.stack([np.ones_like(n), n], axis=1) * w[:, None]
    (alpha, slope), *_ = np.linalg.lstsq(design, t * w, rcond=None)
    pred = alpha + slope * n
    # goodness of fit in log space: the sweep spans orders of magnitude in size and time
    if np.all(pred > 0) and np.all(t > 0):
        y, yhat = np.log(t), np.log(pred)
    else:
        y, yhat = t, pred
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    beta = 1.0 / slope if slope > 0 else float("inf")
    return AlphaBeta(alpha=float(alpha), beta=float(beta), r2=r2)
