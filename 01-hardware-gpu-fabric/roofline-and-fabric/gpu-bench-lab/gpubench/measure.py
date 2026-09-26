"""A measurement is a counted cost divided by a measured time — and a record that says which.

Every benchmark in the lab builds an ``Op``: the zero-argument function to time, plus the
``OpCost`` that accounting.py says one call computes and moves. ``measure`` times the op with
the backend's timer and returns a ``Measurement``, which keeps the counted FLOPs and bytes, the
whole timing distribution, the backend and device, and the basis of the byte count. Tables,
rooflines and reports are built only from these records, so every number can be traced back
to *what was counted* and *how it was timed* — the two questions to ask of any benchmark.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable

from .accounting import OpCost
from .timing import Timing, bench


@dataclass
class Op:
    """An operation ready to time. Built by a backend (``make_gemm``, ``make_stream``, ...)."""
    fn: Callable[[], object]
    cost: OpCost
    note: str = ""
    timer: str = "auto"                   # "auto": backend default; "wall": perf_counter + sync
    setup: Callable[[], object] | None = None   # before every sample, outside the clock
    max_inner: int | None = None          # 1 for cold-cache runs (each sample = one call)
    context: Callable[[], object] | None = None  # context manager factory active while timing
    verify: Callable[[], bool] | None = None     # does the op compute what it claims?
    extras: dict = field(default_factory=dict)


@dataclass
class Measurement:
    op: str                  # "gemm", "stream.triad", "h2d", "p2p", "load.pread", ...
    params: dict
    cost: OpCost
    timing: Timing
    backend: str
    device: str
    note: str = ""
    extras: dict = field(default_factory=dict)

    def seconds(self, stat: str = "best") -> float:
        return self.timing.stat(stat)

    def flops_per_s(self, stat: str = "best") -> float:
        return self.cost.flops / self.seconds(stat)

    def bytes_per_s(self, stat: str = "best") -> float:
        return self.cost.bytes / self.seconds(stat)

    @property
    def intensity(self) -> float:
        return self.cost.intensity

    def summary(self, stat: str = "best") -> str:
        parts = [f"{self.op:<14}", ", ".join(f"{k}={v}" for k, v in self.params.items())]
        t = self.seconds(stat)
        parts.append(f"{si(t, 's')} ({stat}, {len(self.timing.samples)}×{self.timing.inner})")
        if self.cost.flops:
            parts.append(si(self.flops_per_s(stat), "FLOP/s"))
        if self.cost.bytes:
            parts.append(si(self.bytes_per_s(stat), "B/s"))
        return "  ".join(parts)

    def to_dict(self) -> dict:
        return {"op": self.op, "params": self.params, "cost": self.cost.to_dict(), "timing": self.timing.to_dict(),
                "backend": self.backend, "device": self.device, "note": self.note, "extras": self.extras,
                "rates_best": {"flops_per_s": self.flops_per_s("best"), "bytes_per_s": self.bytes_per_s("best")},
                "rates_median": {"flops_per_s": self.flops_per_s("median"), "bytes_per_s": self.bytes_per_s("median")}}

    @classmethod
    def from_dict(cls, d: dict) -> "Measurement":
        c = d["cost"]
        cost = OpCost(c["flops"], c["bytes_read"], c["bytes_written"], c.get("basis", "memory"))
        return cls(d["op"], dict(d["params"]), cost, Timing.from_dict(d["timing"]), d["backend"], d["device"],
                   d.get("note", ""), dict(d.get("extras", {})))


def measure(backend, op: Op, name: str, params: dict, *, repeats: int = 5, min_time: float = 0.02,
            warmup: int = 1) -> Measurement:
    """Time ``op`` with ``backend``'s timer (or the host wall clock if ``backend`` is None)."""
    kw = dict(repeats=repeats, min_time=min_time, warmup=warmup, setup=op.setup)
    if op.max_inner is not None:
        kw["max_inner"] = op.max_inner
    ctx = op.context() if op.context else nullcontext()
    with ctx:
        if backend is None:
            timing = bench(op.fn, **kw)
        else:
            timing = backend.time(op.fn, method=op.timer, **kw)
    return Measurement(op=name, params=dict(params), cost=op.cost, timing=timing,
                       backend=getattr(backend, "name", "host"), device=getattr(backend, "device", "cpu"),
                       note=op.note, extras=dict(op.extras))


# -- units ------------------------------------------------------------------------------------------
_PREFIX = [(1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""),
           (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")]


def si(x: float, unit: str = "", digits: int = 3) -> str:
    """``si(3.35e12, "B/s")`` -> ``'3.35 TB/s'``; ``si(2.5e-6, "s")`` -> ``'2.50 µs'``."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if x == 0 or math.isinf(x):
        return f"{x:g} {unit}".strip()
    for scale, p in _PREFIX:
        if abs(x) >= scale:
            v = x / scale
            dec = max(0, digits - 1 - int(math.floor(math.log10(abs(v)))))
            return f"{v:.{dec}f} {p}{unit}".strip()
    return f"{x:.2e} {unit}".strip()
