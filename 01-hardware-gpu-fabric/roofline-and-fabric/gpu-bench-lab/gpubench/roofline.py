"""The roofline: attainable FLOP/s = min(peak, intensity × bandwidth) — built from *your* numbers.

Two measured quantities define a machine's roofline: the highest FLOP/s any kernel reached
(the flat roof) and the highest bytes/s any kernel moved (the slanted roof). Their ratio,
the **ridge point**, is the arithmetic intensity (FLOPs per byte) an operation needs before
compute, rather than memory, limits it. Below the ridge, a faster ALU buys nothing.

``measured_roofline`` builds the roofline from Measurements (best GEMM for the flat roof,
best single-pass STREAM kernel for the slanted one); ``spec_roofline`` builds the datasheet version, so the
two can be compared; ``noise_warnings`` flags a roofline measured on a busy machine. Primer §2 derives
the model; this module only applies it.

One caution the notebooks repeat: the intensities here count *compulsory DRAM/HBM bytes*. A
small GEMM whose operands sit in cache after the first call is not really moving those bytes
from DRAM, so it can sit *above* the DRAM roofline — that is the cache doing its job, not an
error in the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .measure import si


def attainable(intensity: float, peak_flops: float, peak_bw: float) -> float:
    """``min(peak_flops, intensity × peak_bw)`` in FLOP/s."""
    return min(peak_flops, intensity * peak_bw)


def ridge(peak_flops: float, peak_bw: float) -> float:
    """The intensity (FLOP/byte) where the two roofs meet."""
    return peak_flops / peak_bw


def bound(intensity: float, peak_flops: float, peak_bw: float) -> str:
    """``"memory"`` below the ridge point, ``"compute"`` at or above it."""
    return "compute" if intensity >= ridge(peak_flops, peak_bw) else "memory"


@dataclass(frozen=True)
class Roofline:
    label: str
    peak_flops: float
    peak_bw: float
    source: str = "measured"        # "measured" | "spec" | "estimate"

    @property
    def ridge(self) -> float:
        return ridge(self.peak_flops, self.peak_bw)

    def attainable(self, intensity: float) -> float:
        return attainable(intensity, self.peak_flops, self.peak_bw)

    def bound(self, intensity: float) -> str:
        return bound(intensity, self.peak_flops, self.peak_bw)

    def efficiency(self, achieved_flops: float, intensity: float) -> float:
        """Achieved ÷ attainable at that intensity: how close to its own roof a kernel runs."""
        return achieved_flops / self.attainable(intensity)

    def describe(self) -> str:
        return (f"{self.label} [{self.source}]: peak {si(self.peak_flops, 'FLOP/s')}, "
                f"bandwidth {si(self.peak_bw, 'B/s')}, ridge {self.ridge:.1f} FLOP/B")

    def to_dict(self) -> dict:
        return {"label": self.label, "peak_flops": self.peak_flops, "peak_bw": self.peak_bw,
                "ridge": self.ridge, "source": self.source}


def measured_roofline(gemms, bandwidths, dtype: str | None = None, label: str | None = None,
                      stat: str = "best") -> Roofline:
    """Flat roof = fastest GEMM (of ``dtype`` if given); slanted roof = fastest *single-pass*
    byte-mover (a multi-pass implementation such as numpy's two-pass triad is left out: part of
    its counted traffic can be served from cache, so its rate is not a DRAM rate)."""
    g = [m for m in gemms if dtype is None or m.params.get("dtype") == dtype]
    bandwidths = [m for m in bandwidths if m.extras.get("passes", 1) == 1]
    if not g or not bandwidths:
        raise ValueError("need at least one GEMM and one single-pass bandwidth measurement")
    peak = max(m.flops_per_s(stat) for m in g)
    bw = max(m.bytes_per_s(stat) for m in bandwidths)
    dev = g[0].device
    return Roofline(label or f"{dev} {dtype or 'best dtype'} (measured)", peak, bw, "measured")


NOISY_CV = 0.15      # coefficient of variation of a roof-setting measurement's samples above which we warn


def noise_warnings(gemms, bandwidths, is_gpu: bool = False, cv_limit: float = NOISY_CV) -> list:
    """Sanity checks on a measured roofline; each returned string starts with a label a reader can grep.

    Two symptoms of a machine that was busy while it was measured (a shared vCPU, a laptop on battery,
    another job): the samples behind a roof disagree with each other (their coefficient of variation
    exceeds ``cv_limit``), or — on a CPU — float32 peaks *below* float64. A SIMD unit does twice as
    many float32 lanes as float64 ones, so a CPU's float32 roof should be about 2× its float64 roof; an
    inversion means the float32 sweep ran while something else had the cores. An empty list means
    neither symptom was seen, not that the numbers are right.
    """
    label = f"noisy measurement (shared {'GPU' if is_gpu else 'CPU'}?)"
    out, peaks = [], {}
    for dtype in dict.fromkeys(m.params.get("dtype") for m in gemms):
        top = max((m for m in gemms if m.params.get("dtype") == dtype), key=lambda m: m.flops_per_s())
        peaks[dtype] = top.flops_per_s()
        if top.timing.cv > cv_limit:
            out.append(f"{label}: the {dtype} flat roof ({si(peaks[dtype], 'FLOP/s')}) comes from samples that "
                       f"vary by {top.timing.cv:.0%} (coefficient of variation); re-run on an idle machine")
    single = [m for m in bandwidths if m.extras.get("passes", 1) == 1]
    if single:
        top = max(single, key=lambda m: m.bytes_per_s())
        if top.timing.cv > cv_limit:
            out.append(f"{label}: the bandwidth roof ({si(top.bytes_per_s(), 'B/s')}, {top.op}) comes from "
                       f"samples that vary by {top.timing.cv:.0%} (coefficient of variation); re-run on an idle machine")
    f32, f64 = peaks.get("float32"), peaks.get("float64")
    if not is_gpu and f32 is not None and f64 is not None and f32 < f64:
        out.append(f"{label}: float32 peaked at {si(f32, 'FLOP/s')}, below float64's {si(f64, 'FLOP/s')}; a CPU "
                   "does twice as many float32 lanes per SIMD instruction, so expect about 2× — this roofline "
                   "describes the load on the machine, not the machine; re-run on an idle machine")
    return out


def spec_roofline(spec, dtype: str) -> Roofline:
    """The datasheet roofline of a ``specs.GpuSpec`` at one dtype."""
    peak = spec.peak_flops(dtype)
    if peak is None:
        raise ValueError(f"{spec.name} has no {dtype} fast path in the spec table")
    return Roofline(f"{spec.name} {dtype} (spec)", peak, spec.mem_bw, "spec")


# -- an ASCII roofline, so the picture needs nothing but a terminal -----------------------------------
_MARKS = "#=~-+:"


def ascii_plot(rooflines, points=(), width: int = 64, height: int = 16,
               x_range: tuple | None = None, y_range: tuple | None = None) -> str:
    """Log-log roofline chart. ``points`` are ``(char, intensity, flop_per_s)`` tuples."""
    rooflines = list(rooflines)
    points = [p for p in points if p[1] > 0 and p[2] > 0]
    xs = [p[1] for p in points] + [r.ridge for r in rooflines]
    if x_range is None:
        lo = min(xs + [1.0]) / 4
        hi = max(xs + [1.0]) * 4
        x_range = (2.0 ** math.floor(math.log2(lo)), 2.0 ** math.ceil(math.log2(hi)))
    if y_range is None:
        ys = [p[2] for p in points] + [r.peak_flops for r in rooflines] + \
             [r.attainable(x_range[0]) for r in rooflines]
        y_range = (10.0 ** math.floor(math.log10(min(ys))), 10.0 ** math.ceil(math.log10(max(ys))))
    (x0, x1), (y0, y1) = x_range, y_range
    lx0, lx1, ly0, ly1 = math.log(x0), math.log(x1), math.log(y0), math.log(y1)

    def col(x: float) -> int:
        return round((math.log(x) - lx0) / (lx1 - lx0) * (width - 1))

    def row(y: float) -> int:
        return (height - 1) - round((math.log(y) - ly0) / (ly1 - ly0) * (height - 1))

    grid = [[" "] * width for _ in range(height)]
    for i, r in enumerate(rooflines):
        for c in range(width):
            x = math.exp(lx0 + c / (width - 1) * (lx1 - lx0))
            rr = row(r.attainable(x))
            if 0 <= rr < height and grid[rr][c] == " ":
                grid[rr][c] = _MARKS[i % len(_MARKS)]
    for ch, x, y in points:
        c, rr = col(x), row(y)
        if 0 <= c < width and 0 <= rr < height:
            grid[rr][c] = ch

    ylabels = {}
    for d in range(math.ceil(math.log10(y0)), math.floor(math.log10(y1)) + 1):
        ylabels[row(10.0 ** d)] = si(10.0 ** d, "")
    lines = [f"{ylabels.get(i, ''):>7} |" + "".join(grid[i]) for i in range(height)]
    lines.append(" " * 8 + "+" + "-" * width)
    ticks = [" "] * (width + 12)
    x = 2.0 ** math.ceil(math.log2(x0))
    last = -10
    while x <= x1:
        c = col(x)
        label = f"{x:g}"
        if c - last > len(label) and c + len(label) <= width:
            for j, chx in enumerate(label):
                ticks[9 + c + j] = chx
            last = c + len(label)
        x *= 2
    lines.append("".join(ticks).rstrip())
    lines.append(" " * 9 + "arithmetic intensity (FLOP/byte, log)   ↑ FLOP/s (log)")
    for i, r in enumerate(rooflines):
        lines.append(f"  {_MARKS[i % len(_MARKS)]}  {r.describe()}")
    return "\n".join(lines)
