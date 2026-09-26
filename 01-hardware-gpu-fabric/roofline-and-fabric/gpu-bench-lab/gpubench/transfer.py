"""Moving bytes between memories: the α-β model, pinned vs pageable, and PCIe arithmetic.

A copy of ``n`` bytes costs ``t(n) = α + n/β``. α is the fixed cost of *any* copy — a driver
call, a DMA descriptor, a kernel launch, a syscall — and β the rate of the slowest link on
the path. Small copies are latency-bound (α dominates), big ones bandwidth-bound, and the
crossover is ``n½ = α·β``: below it you get less than half the link. That is why engines batch
small transfers, and why "PCIe is 32 GB/s" is only true for large, pinned copies.

T0 measures host ``memcpy`` (RAM to RAM) with the same sweep and fit, so the method is the same
on a laptop. T1 measures the real thing: host↔device from **pinned** memory (DMA straight from
page-locked pages) and **pageable** memory (the driver first copies into a pinned bounce
buffer, one chunk at a time, so it is slower and cannot overlap with compute).
"""
from __future__ import annotations

from collections import defaultdict

from .measure import Measurement, measure
from .specs import pcie_gbs
from .timing import AlphaBeta, fit_alpha_beta


def sizes(min_bytes: int = 4 << 10, max_bytes: int = 256 << 20, factor: int = 4) -> list:
    out, s = [], min_bytes
    while s <= max_bytes:
        out.append(s)
        s *= factor
    return out


def memcpy_sweep(be, sweep=None, repeats: int = 5, min_time: float = 0.01) -> list:
    """T0: host-to-host copies over ``sweep`` sizes (numpy backend)."""
    out = []
    for s in sweep or sizes():
        out.append(measure(be, be.make_memcpy(s), "memcpy", {"nbytes": s}, repeats=repeats, min_time=min_time))
    return out


def hostdevice_sweep(be, sweep=None, directions=("h2d", "d2h"), pinned=(True, False),
                     repeats: int = 5, min_time: float = 0.02) -> list:
    """T1: host↔device copies (torch backend) for each direction × pinned/pageable × size."""
    if not be.is_gpu:
        raise ValueError("host↔device transfers need the torch backend and a GPU")
    out = []
    for d in directions:
        for p in pinned:
            for s in sweep or sizes():
                m = measure(be, be.make_transfer(s, d, p), d, {"nbytes": s, "pinned": p},
                            repeats=repeats, min_time=min_time)
                out.append(m)
    return out


def series(measurements) -> dict:
    """Group a sweep into series keyed by ``(op, pinned)``, each sorted by size."""
    groups = defaultdict(list)
    for m in measurements:
        groups[(m.op, m.params.get("pinned"))].append(m)
    return {k: sorted(v, key=lambda m: m.params["nbytes"]) for k, v in groups.items()}


def fit(measurements, stat: str = "median") -> AlphaBeta:
    """α-β fit of one series (sizes from ``params['nbytes']``, times from the timing ``stat``)."""
    ms = sorted(measurements, key=lambda m: m.params["nbytes"])
    return fit_alpha_beta([m.params["nbytes"] for m in ms], [m.seconds(stat) for m in ms])


def pcie_expectation(gen: int, width: int, efficiency: float = 0.85) -> dict:
    """What a large pinned copy should reach on this link: theoretical × a typical efficiency."""
    theo = pcie_gbs(gen, width)
    return {"link": f"PCIe Gen{gen} x{width}", "theoretical_gbs": theo, "expected_pinned_gbs": theo * efficiency}


def best_bandwidth(measurements: list, stat: str = "best") -> Measurement:
    return max(measurements, key=lambda m: m.bytes_per_s(stat))
