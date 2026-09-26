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

**Two α's.** How the sweep times copies decides what its α means. *Pipelined* (the default):
many copies are issued back to back and the device is synchronised only at the ends, so an
asynchronous (pinned, ``non_blocking``) copy's fixed cost overlaps the previous copy's transfer —
α is a per-copy *issue* cost, the right number for a stream of independent copies. *Latency*
(``latency=True``): one copy per sample, synchronised before and after, so α is the full time
until the bytes have arrived — the right number when each copy waits for the previous one, as
every step of a ring all-reduce does. A synchronous copy (pageable memory, a numpy ``memcpy``)
cannot overlap, so its two α's are the same thing. Large copies give the same β either way.
"""
from __future__ import annotations

import re
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
                     repeats: int = 5, min_time: float = 0.02, latency: bool = False) -> list:
    """T1: host↔device copies (torch backend) for each direction × pinned/pageable × size.

    ``latency=False``: copies back to back (α = pipelined issue cost). ``latency=True``: one
    synchronised copy per sample (α = the latency of one copy); use more ``repeats`` then.
    """
    if not be.is_gpu:
        raise ValueError("host↔device transfers need the torch backend and a GPU")
    mode = "latency" if latency else "pipelined"
    out = []
    for d in directions:
        for p in pinned:
            for s in sweep or sizes():
                m = measure(be, be.make_transfer(s, d, p, sync_each=latency), d,
                            {"nbytes": s, "pinned": p, "mode": mode}, repeats=repeats, min_time=min_time)
                out.append(m)
    return out


def series(measurements) -> dict:
    """Group a sweep into series keyed by ``(op, pinned, mode)``, each sorted by size."""
    groups = defaultdict(list)
    for m in measurements:
        groups[(m.op, m.params.get("pinned"), m.params.get("mode"))].append(m)
    return {k: sorted(v, key=lambda m: m.params["nbytes"]) for k, v in groups.items()}


def series_label(key) -> str:
    """``("h2d", True, "latency")`` -> ``"h2d pinned latency"``; ``("memcpy", None, None)`` -> ``"memcpy"``."""
    op, pinned, mode = key
    parts = [op]
    if pinned is not None:
        parts.append("pinned" if pinned else "pageable")
    if mode:
        parts.append(mode)
    return " ".join(parts)


def fit(measurements, stat: str = "median") -> AlphaBeta:
    """α-β fit of one series (sizes from ``params['nbytes']``, times from the timing ``stat``)."""
    ms = sorted(measurements, key=lambda m: m.params["nbytes"])
    return fit_alpha_beta([m.params["nbytes"] for m in ms], [m.seconds(stat) for m in ms])


def host_link(device_name: str | None = None, index: int = 0) -> dict:
    """The host↔GPU link to model with: ``{"gen", "width", "source"}``.

    First choice is what ``nvidia-smi`` reports for GPU ``index`` — the maximum generation (an
    idle link drops to Gen1 to save power) and the *current* width (a link that trained at x8
    stays at x8). Then the spec table's ``host_link`` for ``device_name``. Last, Gen4 x16,
    labelled an assumption — never silently.
    """
    from .inventory import query_gpus
    from .specs import lookup

    rows, _ = query_gpus()
    if rows and len(rows) > index:
        r = rows[index]
        gen, width = r.get("pcie.link.gen.max"), r.get("pcie.link.width.current") or r.get("pcie.link.width.max")
        if isinstance(gen, int) and isinstance(width, int):
            return {"gen": gen, "width": width, "source": "nvidia-smi"}
    spec = lookup(device_name) if device_name else None
    if spec is not None:
        m = re.search(r"Gen(\d)\s*x(\d+)", spec.host_link)
        if m:
            return {"gen": int(m.group(1)), "width": int(m.group(2)), "source": f"spec table ({spec.name})"}
    return {"gen": 4, "width": 16, "source": "assumed (no nvidia-smi, no spec entry)"}


def pcie_expectation(gen: int, width: int, efficiency: float = 0.85) -> dict:
    """What a large pinned copy should reach on this link: theoretical × a typical efficiency."""
    theo = pcie_gbs(gen, width)
    return {"link": f"PCIe Gen{gen} x{width}", "theoretical_gbs": theo, "expected_pinned_gbs": theo * efficiency}


def best_bandwidth(measurements: list, stat: str = "best") -> Measurement:
    return max(measurements, key=lambda m: m.bytes_per_s(stat))
