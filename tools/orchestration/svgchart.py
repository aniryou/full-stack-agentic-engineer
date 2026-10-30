#!/usr/bin/env python3
"""svgchart.py: charts in the figure house style (tools/orchestration/FIGURE-STYLE.md §4 and §7), computed by a core.

A core's ``tools/make_figures.py`` builds a spec (a plain dict: the axes, the series, the marks, a note), renders it
with ``render()`` and writes the SVG into the topic's ``figures/`` folder with ``write_figure()``. The core's
``tests/test_figures.py`` calls ``check_figure()`` with the same spec builder, so the committed chart is pinned to the
core the way the primer's numbers are: the test fails when the core computes other values, or when the file is not
what ``render()`` makes of its own spec (the spec travels inside the SVG, in its ``<desc>``). Standard library only;
the output is deterministic (every coordinate is rounded to a tenth of a pixel, the spec is serialised with sorted
keys), so the same spec gives the same bytes on every machine.

The spec::

    {
      "kind": "line" | "bar",
      "title": "One sentence: what the chart shows.",     # the aria-label and the <title>
      "width": 880, "height": 360,                        # optional; the defaults; width 720 to 1000, height at most 560
      "x": {"label": "batch size (sequences)", "scale": "linear" | "log",
            "ticks": [1, 2, 4, 8], "tick_labels": ["1", "2", "4", "8"],    # optional; ticks are chosen when absent
            "min": 1, "max": 64},                                           # optional; the data decides when absent
      "y": {"label": "step time (ms)", ...},
      "categories": ["FP16", "INT8", "INT4"],             # bar charts only: one bar group per category
      "series": [
        {"name": "H100 SXM", "x": [...], "y": [...],      # a line series: x and y of equal length (bars: y only)
         "style": "accent" | "warm" | "plain" | "violet" | "yellow" | "muted",
         "dash": false, "markers": false, "step": false, "fill": false, "labels": false}
      ],
      "marks": [                                          # optional annotations, in data coordinates
        {"kind": "vline", "x": 230.6, "label": "break-even", "style": "warm"},
        {"kind": "hline", "y": 3.35, "label": "HBM roofline"},
        {"kind": "point", "x": 8, "y": 2.1, "label": "8 sequences: 2.1 ms", "style": "accent", "anchor": "start"},
        {"kind": "band", "x0": 0, "x1": 100, "label": "memory-bound", "style": "muted"},
        {"kind": "text", "x": 10, "y": 5, "text": "…", "style": "muted", "anchor": "middle"}
      ],
      "legend": "top-right" | "top-left" | "bottom-right" | "bottom-left" | "none",   # optional; top-right
      "note": "computed by roofline.llm.decode_step_ms (roofline-core)"               # optional; muted, bottom-left
    }

Command line: ``python3 tools/orchestration/svgchart.py demo <out.svg>`` writes a demonstration chart.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

CARD = "#161E1C"
TEXT = "#E4EAE7"
MUTED = "#97A59F"
GRID = "#27322E"
AXIS = "#55635E"
FONT = "Bricolage Grotesque, system-ui, sans-serif"
MONO = "JetBrains Mono, monospace"
STYLES = {            # the house palette: the accent for the subject, the warm colour for a cost or a limit
    "accent": "#4CC5B7",
    "warm": "#E39A62",
    "plain": "#E4EAE7",
    "violet": "#B7A3F0",
    "yellow": "#E6C85C",
    "muted": "#97A59F",
}
DEFAULT_W, DEFAULT_H = 880, 360
MIN_W, MAX_W, MAX_H = 720, 1000, 560
MARGIN = {"left": 66, "right": 22, "top": 22, "bottom": 54}
SVG_NS = "http://www.w3.org/2000/svg"


# ------------------------------------------------------------------ numbers and text

def _f(v: float) -> str:
    """A coordinate: one decimal, no negative zero."""
    s = f"{v:.1f}"
    return "0.0" if s == "-0.0" else s


def fmt_num(v: float) -> str:
    """A tick label: integers with thousands separators, other values with up to three significant digits."""
    if isinstance(v, bool):
        return str(v)
    if float(v).is_integer():
        return f"{int(v):,}"
    s = f"{v:.3g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}e{int(e)}"
    return s


def _text_w(s: str, size: float) -> float:
    """A rough width of a label in the sans-serif font, for boxes around text."""
    return len(s) * size * 0.56 + 4


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """About n ticks between lo and hi, on a 1-2-5 step (the usual choice for a linear axis)."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if raw <= step:
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = []
    t = first
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 10) + 0.0)
        t += step
    return ticks


def log_ticks(lo: float, hi: float) -> list[float]:
    """Powers of ten across the range; 1-2-5 within a decade when the range is shorter than two decades."""
    a, b = math.log10(lo), math.log10(hi)
    if b - a >= 2:
        return [10.0 ** k for k in range(math.ceil(a - 1e-9), math.floor(b + 1e-9) + 1)]
    ticks = []
    for k in range(math.floor(a) - 1, math.ceil(b) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** k
            if lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9):
                ticks.append(v)
    return ticks


# ------------------------------------------------------------------ scales

class _Scale:
    def __init__(self, lo: float, hi: float, px0: float, px1: float, log: bool):
        if log and (lo <= 0 or hi <= 0):
            raise ValueError("a log axis needs positive bounds")
        if hi <= lo:
            hi = lo + 1.0
        self.lo, self.hi, self.px0, self.px1, self.log = lo, hi, px0, px1, log

    def _t(self, v: float) -> float:
        if self.log:
            if v <= 0:
                raise ValueError(f"a log axis cannot place {v}")
            return (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def px(self, v: float) -> float:
        return self.px0 + (self.px1 - self.px0) * self._t(v)


def _domain(axis: dict, values: list[float], is_y: bool) -> tuple[float, float]:
    log = axis.get("scale") == "log"
    vals = [float(v) for v in values if v is not None]
    for key in ("min", "max"):
        if key in axis:
            vals.append(float(axis[key]))
    if not vals:
        return (1.0, 10.0) if log else (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    if "min" in axis:
        lo = float(axis["min"])
    elif is_y and not log and lo >= 0:
        lo = 0.0
    elif not log and lo != hi:
        lo = lo - (hi - lo) * (0.05 if is_y else 0.0)
    if "max" in axis:
        hi = float(axis["max"])
    elif is_y and not log:
        hi = hi + (hi - lo) * 0.08 if hi > lo else hi + 1.0
    if log:
        # given ticks bound the axis; otherwise the nearest powers of ten (an exact power stays where it is)
        ticks = [float(v) for v in axis.get("ticks") or []]
        if "min" not in axis:
            lo = min(ticks) if ticks and min(ticks) <= lo else 10.0 ** math.floor(math.log10(lo) + 1e-9)
        if "max" not in axis:
            hi = max(ticks) if ticks and max(ticks) >= hi else 10.0 ** math.ceil(math.log10(hi) - 1e-9)
    elif axis.get("ticks"):
        ticks = [float(v) for v in axis["ticks"]]
        if "min" not in axis:
            lo = min(lo, min(ticks))
        if "max" not in axis:
            hi = max(hi, max(ticks))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


# ------------------------------------------------------------------ rendering

def _validate(spec: dict) -> None:
    kind = spec.get("kind")
    if kind not in ("line", "bar"):
        raise ValueError("kind must be 'line' or 'bar'")
    if not spec.get("title"):
        raise ValueError("a title (one sentence) is required")
    w, h = spec.get("width", DEFAULT_W), spec.get("height", DEFAULT_H)
    if not (MIN_W <= w <= MAX_W) or not (120 <= h <= MAX_H):
        raise ValueError(f"width must be {MIN_W}..{MAX_W} and height at most {MAX_H} (FIGURE-STYLE.md §3, §4)")
    series = spec.get("series") or []
    if not series:
        raise ValueError("at least one series")
    for s in series:
        if s.get("style", "accent") not in STYLES:
            raise ValueError(f"unknown style {s.get('style')!r}")
        if kind == "line":
            if len(s.get("x", [])) != len(s.get("y", [])) or not s.get("y"):
                raise ValueError(f"series {s.get('name')!r}: x and y of equal, non-zero length")
        else:
            if len(s.get("y", [])) != len(spec.get("categories") or []):
                raise ValueError(f"series {s.get('name')!r}: one value per category")
    for m in spec.get("marks") or []:
        if m.get("kind") not in ("vline", "hline", "point", "band", "text"):
            raise ValueError(f"unknown mark kind {m.get('kind')!r}")
        if m.get("style", "muted") not in STYLES:
            raise ValueError(f"unknown mark style {m.get('style')!r}")


def _canonical(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def render(spec: dict) -> str:
    """The SVG text of a chart (one trailing newline). Deterministic for a given spec."""
    _validate(spec)
    W, H = spec.get("width", DEFAULT_W), spec.get("height", DEFAULT_H)
    kind = spec["kind"]
    xa, ya = spec.get("x", {}) or {}, spec.get("y", {}) or {}
    series = spec["series"]
    marks = spec.get("marks") or []
    note = spec.get("note")
    legend_pos = spec.get("legend", "top-right" if len(series) > 1 else "none")
    L, R, T, B = MARGIN["left"], MARGIN["right"], MARGIN["top"], MARGIN["bottom"]
    px0, px1 = L, W - R
    py0, py1 = H - B, T         # y grows downward: py0 is the axis line

    out = []
    title = spec["title"]
    out.append(f'<svg xmlns="{SVG_NS}" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label={quoteattr(title)} style="color: {TEXT}">')
    out.append(f"  <title>{escape(title)}</title>")
    out.append(f"  <desc>{escape(_canonical(spec))}</desc>")
    out.append(f'  <rect x="0" y="0" width="{W}" height="{H}" rx="8" fill="{CARD}"/>')
    out.append(f'  <g font-family="{FONT}" font-size="12" fill="currentColor">')

    # --- y scale (both kinds)
    ys = [float(v) for s in series for v in s["y"]]
    for m in marks:
        if m["kind"] in ("hline", "point", "text"):
            ys.append(float(m["y"]))
    ylog = ya.get("scale") == "log"
    ylo, yhi = _domain(ya, ys, is_y=True)
    Y = _Scale(ylo, yhi, py0, py1, ylog)
    yticks = [float(t) for t in ya["ticks"]] if ya.get("ticks") else (log_ticks(ylo, yhi) if ylog else nice_ticks(ylo, yhi))
    ylabels = ya.get("tick_labels") or [fmt_num(t) for t in yticks]

    # --- x scale
    if kind == "line":
        xs = [float(v) for s in series for v in s["x"]]
        for m in marks:
            if m["kind"] in ("vline", "point", "text"):
                xs.append(float(m["x"]))
            if m["kind"] == "band":
                xs += [float(m["x0"]), float(m["x1"])]
        xlog = xa.get("scale") == "log"
        xlo, xhi = _domain(xa, xs, is_y=False)
        X = _Scale(xlo, xhi, px0, px1, xlog)
        xticks = [float(t) for t in xa["ticks"]] if xa.get("ticks") else (log_ticks(xlo, xhi) if xlog else nice_ticks(xlo, xhi))
        xlabels = xa.get("tick_labels") or [fmt_num(t) for t in xticks]
        xpos = [X.px(t) for t in xticks]
    else:
        cats = spec["categories"]
        n = len(cats)
        slot = (px1 - px0) / max(1, n)
        xpos = [px0 + slot * (i + 0.5) for i in range(n)]
        xlabels = xa.get("tick_labels") or [str(c) for c in cats]
        X = None

    # --- grid and axes
    out.append(f'    <g stroke="{GRID}" stroke-width="1">')
    for t in yticks:
        y = Y.px(t)
        if py1 - 0.5 <= y <= py0 + 0.5:
            out.append(f'      <line x1="{_f(px0)}" y1="{_f(y)}" x2="{_f(px1)}" y2="{_f(y)}"/>')
    out.append("    </g>")
    out.append(f'    <g stroke="{AXIS}" stroke-width="1.2">')
    out.append(f'      <line x1="{_f(px0)}" y1="{_f(py0)}" x2="{_f(px1)}" y2="{_f(py0)}"/>')
    out.append(f'      <line x1="{_f(px0)}" y1="{_f(py0)}" x2="{_f(px0)}" y2="{_f(py1)}"/>')
    out.append("    </g>")

    # --- tick labels
    out.append(f'    <g font-size="11" fill="{MUTED}">')
    for t, lab in zip(yticks, ylabels):
        y = Y.px(t)
        if py1 - 0.5 <= y <= py0 + 0.5:
            out.append(f'      <text x="{_f(px0 - 8)}" y="{_f(y + 3.5)}" text-anchor="end">{escape(str(lab))}</text>')
    for x, lab in zip(xpos, xlabels):
        if px0 - 0.5 <= x <= px1 + 0.5:
            lines = str(lab).split("\n")
            if len(lines) == 1:
                out.append(f'      <text x="{_f(x)}" y="{_f(py0 + 16)}" text-anchor="middle">{escape(lines[0])}</text>')
            else:
                spans = "".join(f'<tspan x="{_f(x)}" dy="{"0" if i == 0 else "12"}">{escape(l)}</tspan>' for i, l in enumerate(lines))
                out.append(f'      <text x="{_f(x)}" y="{_f(py0 + 16)}" text-anchor="middle">{spans}</text>')
    out.append("    </g>")

    # --- axis titles
    if xa.get("label"):
        out.append(f'    <text x="{_f((px0 + px1) / 2)}" y="{_f(H - 12)}" text-anchor="middle" fill="{MUTED}">{escape(xa["label"])}</text>')
    if ya.get("label"):
        cy = (py0 + py1) / 2
        out.append(f'    <text x="16" y="{_f(cy)}" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 16 {_f(cy)})">{escape(ya["label"])}</text>')

    # --- clip for the plot area
    out.append(f'    <clipPath id="plot"><rect x="{_f(px0 - 1)}" y="{_f(py1 - 6)}" width="{_f(px1 - px0 + 2)}" height="{_f(py0 - py1 + 7)}"/></clipPath>')

    # --- bands first (under the data)
    for m in marks:
        if m["kind"] != "band" or kind != "line":
            continue
        col = STYLES[m.get("style", "muted")]
        x0, x1 = X.px(float(m["x0"])), X.px(float(m["x1"]))
        out.append(f'    <rect x="{_f(min(x0, x1))}" y="{_f(py1)}" width="{_f(abs(x1 - x0))}" height="{_f(py0 - py1)}" fill="{col}" fill-opacity="0.12"/>')
        if m.get("label"):
            out.append(f'    <text x="{_f((x0 + x1) / 2)}" y="{_f(py1 + 14)}" text-anchor="middle" font-size="11" fill="{col}">{escape(m["label"])}</text>')

    # --- the data
    out.append('    <g clip-path="url(#plot)">')
    if kind == "line":
        for s in series:
            col = STYLES[s.get("style", "accent")]
            pts = [(X.px(float(x)), Y.px(float(y))) for x, y in zip(s["x"], s["y"])]
            if s.get("step"):
                d = [f"M{_f(pts[0][0])} {_f(pts[0][1])}"]
                for (xa_, ya_), (xb, yb) in zip(pts, pts[1:]):
                    d.append(f"L{_f(xb)} {_f(ya_)} L{_f(xb)} {_f(yb)}")
                path = " ".join(d)
            else:
                path = "M" + " L".join(f"{_f(x)} {_f(y)}" for x, y in pts)
            if s.get("fill"):
                area = path + f" L{_f(pts[-1][0])} {_f(py0)} L{_f(pts[0][0])} {_f(py0)} Z"
                out.append(f'      <path d="{area}" fill="{col}" fill-opacity="0.14" stroke="none"/>')
            dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
            out.append(f'      <path d="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"{dash}/>')
            if s.get("markers"):
                for x, y in pts:
                    out.append(f'      <circle cx="{_f(x)}" cy="{_f(y)}" r="3" fill="{CARD}" stroke="{col}" stroke-width="1.6"/>')
            if s.get("labels"):
                for (x, y), v in zip(pts, s["y"]):
                    out.append(f'      <text x="{_f(x)}" y="{_f(y - 8)}" text-anchor="middle" font-size="10.5" fill="{col}">{escape(fmt_num(float(v)))}</text>')
    else:
        n_series = len(series)
        slot = (px1 - px0) / max(1, len(spec["categories"]))
        group_w = slot * 0.72
        bar_w = group_w / n_series
        base = Y.px(ylo)
        for j, s in enumerate(series):
            col = STYLES[s.get("style", "accent")]
            for i, v in enumerate(s["y"]):
                x = xpos[i] - group_w / 2 + bar_w * j
                y = Y.px(float(v))
                h = max(0.0, base - y)
                out.append(f'      <rect x="{_f(x + 1)}" y="{_f(y)}" width="{_f(max(1.0, bar_w - 2))}" height="{_f(h)}" rx="2" fill="{col}" fill-opacity="0.85"/>')
                if s.get("labels", True):
                    out.append(f'      <text x="{_f(x + bar_w / 2)}" y="{_f(y - 5)}" text-anchor="middle" font-size="10.5" fill="{col}">{escape(fmt_num(float(v)))}</text>')
    out.append("    </g>")

    # --- lines and points over the data
    for m in marks:
        col = STYLES[m.get("style", "muted")]
        k = m["kind"]
        if k == "vline" and kind == "line":
            x = X.px(float(m["x"]))
            out.append(f'    <line x1="{_f(x)}" y1="{_f(py1)}" x2="{_f(x)}" y2="{_f(py0)}" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 3"/>')
            if m.get("label"):
                anchor = m.get("anchor", "start")
                dx = 5 if anchor == "start" else -5
                out.append(f'    <text x="{_f(x + dx)}" y="{_f(py1 + 12)}" text-anchor="{anchor}" font-size="11" fill="{col}">{escape(m["label"])}</text>')
        elif k == "hline":
            y = Y.px(float(m["y"]))
            out.append(f'    <line x1="{_f(px0)}" y1="{_f(y)}" x2="{_f(px1)}" y2="{_f(y)}" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 3"/>')
            if m.get("label"):
                anchor = m.get("anchor", "end")
                x = px1 - 4 if anchor == "end" else px0 + 6
                out.append(f'    <text x="{_f(x)}" y="{_f(y - 5)}" text-anchor="{anchor}" font-size="11" fill="{col}">{escape(m["label"])}</text>')
        elif k == "point" and kind == "line":
            x, y = X.px(float(m["x"])), Y.px(float(m["y"]))
            out.append(f'    <circle cx="{_f(x)}" cy="{_f(y)}" r="4.5" fill="{col}" stroke="{CARD}" stroke-width="1.5"/>')
            if m.get("label"):
                anchor = m.get("anchor", "start")
                dx = {"start": 9, "end": -9, "middle": 0}[anchor]
                dy = float(m.get("dy", -9))
                out.append(f'    <text x="{_f(x + dx)}" y="{_f(y + dy)}" text-anchor="{anchor}" font-size="11" fill="{col}">{escape(m["label"])}</text>')
        elif k == "text":
            if kind == "line":
                x = X.px(float(m["x"]))
            else:
                x = xpos[int(m["x"])] if isinstance(m["x"], int) and 0 <= int(m["x"]) < len(xpos) else float(m["x"])
            y = Y.px(float(m["y"]))
            anchor = m.get("anchor", "middle")
            out.append(f'    <text x="{_f(x)}" y="{_f(y)}" text-anchor="{anchor}" font-size="11" fill="{col}">{escape(m["text"])}</text>')

    # --- legend
    if legend_pos != "none" and series:
        rows = [(s.get("name") or f"series {i + 1}", STYLES[s.get("style", "accent")], bool(s.get("dash"))) for i, s in enumerate(series)]
        box_w = max(_text_w(r[0], 11.5) for r in rows) + 40
        box_h = 10 + 17 * len(rows)
        if "right" in legend_pos:
            bx = px1 - box_w - 8
        else:
            bx = px0 + 10
        if "top" in legend_pos:
            by = py1 + 8
        else:
            by = py0 - box_h - 8
        out.append(f'    <g font-size="11.5">')
        out.append(f'      <rect x="{_f(bx)}" y="{_f(by)}" width="{_f(box_w)}" height="{_f(box_h)}" rx="5" fill="{CARD}" fill-opacity="0.9" stroke="{GRID}"/>')
        for i, (name, col, dash) in enumerate(rows):
            yy = by + 14 + 17 * i
            dash_attr = ' stroke-dasharray="6 4"' if dash else ""
            if kind == "bar":
                out.append(f'      <rect x="{_f(bx + 10)}" y="{_f(yy - 6)}" width="16" height="9" rx="2" fill="{col}" fill-opacity="0.85"/>')
            else:
                out.append(f'      <line x1="{_f(bx + 10)}" y1="{_f(yy - 1)}" x2="{_f(bx + 28)}" y2="{_f(yy - 1)}" stroke="{col}" stroke-width="2"{dash_attr}/>')
            out.append(f'      <text x="{_f(bx + 34)}" y="{_f(yy + 3)}" fill="currentColor">{escape(name)}</text>')
        out.append("    </g>")

    # --- note
    if note:
        out.append(f'    <text x="10" y="{_f(H - 7)}" font-size="10.5" fill="{MUTED}" font-family="{MONO}">{escape(note)}</text>')

    out.append("  </g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ reading back and checking

def read_spec(svg_text: str) -> dict:
    """The spec a chart was rendered from (its <desc>)."""
    root = ET.fromstring(svg_text)
    desc = root.find(f"{{{SVG_NS}}}desc")
    if desc is None or not desc.text:
        raise ValueError("no <desc> with a spec: not a chart made by svgchart.render()")
    return json.loads(desc.text)


def write_figure(path, spec: dict) -> pathlib.Path:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render(spec), encoding="utf-8")
    return p


def _diff(a, b, path: str, rel_tol: float, abs_tol: float, out: list) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            out.append(f"{path}: {a!r} != {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            out.append(f"{path}: computed {a!r}, stored {b!r}")
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'computed' if k in a else 'stored'}")
            else:
                _diff(a[k], b[k], f"{path}.{k}", rel_tol, abs_tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items computed, {len(b)} stored")
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", rel_tol, abs_tol, out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def check_figure(path, make_spec, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> None:
    """For a test: the committed chart at ``path`` is what render() makes of its own spec, and that spec is what
    ``make_spec()`` computes now (numbers within the tolerances, everything else exactly). Raises AssertionError."""
    p = pathlib.Path(path)
    text = p.read_text(encoding="utf-8")
    stored = read_spec(text)
    if render(stored) != text:
        raise AssertionError(f"{p.name}: the file is not what svgchart.render() makes of its own spec; regenerate it with the core's tools/make_figures.py")
    fresh = make_spec()
    diffs: list[str] = []
    _diff(fresh, stored, "spec", rel_tol, abs_tol, diffs)
    if diffs:
        shown = "\n  ".join(diffs[:12]) + ("\n  …" if len(diffs) > 12 else "")
        raise AssertionError(f"{p.name}: the chart's data differs from what the core computes now ({len(diffs)} differences); regenerate it with the core's tools/make_figures.py:\n  {shown}")


# ------------------------------------------------------------------ demo

def demo_spec() -> dict:
    xs = [1, 2, 4, 8, 16, 32, 64, 128]
    weights_ms = [2.0] * len(xs)                       # a fixed read of the weights per step
    kv_ms = [0.05 * x for x in xs]                     # a KV read that grows with the batch
    return {
        "kind": "line",
        "title": "Decode step time against batch size: the weights cost a fixed read, the KV cache read grows with the batch (demonstration data).",
        "x": {"label": "batch size (sequences)", "scale": "log", "ticks": xs},
        "y": {"label": "step time (ms)"},
        "series": [
            {"name": "weights + KV", "x": xs, "y": [w + k for w, k in zip(weights_ms, kv_ms)], "style": "accent", "markers": True},
            {"name": "weights only", "x": xs, "y": weights_ms, "style": "muted", "dash": True},
        ],
        "marks": [
            {"kind": "vline", "x": 40, "label": "KV read = weights read", "style": "warm"},
            {"kind": "point", "x": 8, "y": 2.4, "label": "8 sequences: 2.4 ms", "style": "accent"},
        ],
        "legend": "top-left",
        "note": "demonstration data: svgchart.demo_spec()",
    }


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "demo":
        out = write_figure(sys.argv[2], demo_spec())
        print(f"wrote {out}")
    else:
        print(__doc__)
        sys.exit(2)
