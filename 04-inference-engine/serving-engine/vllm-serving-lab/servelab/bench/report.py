"""report.py — tables and text charts for benchmark results (no plotting library needed).

One idea: a benchmark result is a *comparison* — this config against that one, this rate against
the next — so the output is a table with one row per run and the same columns every time, plus a
text curve for the one relationship that matters most: latency against load.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def fmt(v) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        if 0 < abs(v) < 1:
            return f"{v:.2f}"
        return f"{v:,.1f}"
    return str(v)


def table(rows: list, cols: list | None = None) -> str:
    """Align a list of dicts as a text table (column order from ``cols`` or the first row)."""
    if not rows:
        return "(no rows)"
    cols = cols or list(rows[0])
    cells = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda vals: "  ".join(v.rjust(w) for v, w in zip(vals, widths))  # noqa: E731
    return "\n".join([line(cols), line(["-" * w for w in widths])] + [line(r) for r in cells])


def compare(summaries: dict, percentile_: int = 99) -> str:
    """One row per labelled :class:`~servelab.bench.summary.Summary`."""
    rows = []
    for label, s in summaries.items():
        row = s.row(percentile_)
        row["label"] = label
        rows.append(row)
    return table(rows)


def bar(value: float, max_value: float, width: int = 30) -> str:
    if not max_value or math.isnan(value):
        return ""
    return "#" * max(0, min(width, int(round(width * value / max_value))))


def curve(xs: list, ys: list, xlabel: str = "x", ylabel: str = "y", width: int = 36) -> str:
    """A sideways bar chart: one line per x, bar length proportional to y."""
    finite = [y for y in ys if not math.isnan(y)]
    top = max(finite) if finite else 0
    lines = [f"{xlabel:>10} | {ylabel}"]
    lines += [f"{fmt(x):>10} | {bar(y, top, width):<{width}} {fmt(y)}" for x, y in zip(xs, ys)]
    return "\n".join(lines)


def save_json(run, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(run.to_dict(), indent=1, default=str))
    return p
