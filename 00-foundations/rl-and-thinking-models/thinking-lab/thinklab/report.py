"""report.py — tables, text charts and a JSON + Markdown report that always says where numbers came from.

One idea: a number without its provenance is a liability in a design review. Every table printed by
this lab carries a label — MEASURED (a real server or a real training run on this machine),
SIMULATED (the engine emulator or the simulated model) or ILLUSTRATIVE (recorded sample output in a
documented format) — and :func:`write` stores the same label next to the data.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

BARS = " ▁▂▃▄▅▆▇█"


def fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
    return str(v)


def table(rows: list, cols: list | None = None, title: str = "") -> str:
    """A plain-text table from a list of dicts."""
    if not rows:
        return f"{title}\n(no rows)" if title else "(no rows)"
    cols = cols or list(rows[0])
    cells = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    w = [max(len(str(c)), *(len(row[i]) for row in cells)) for i, c in enumerate(cols)]
    line = lambda xs: "  ".join(str(x).rjust(n) for x, n in zip(xs, w))       # noqa: E731
    out = [line(cols), line("-" * n for n in w)] + [line(r) for r in cells]
    return (title + "\n" if title else "") + "\n".join(out)


def sparkline(values: list) -> str:
    vals = [v for v in values if v == v]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    return "".join(BARS[1 + int((v - lo) / span * (len(BARS) - 2))] if v == v else " " for v in values)


def histogram(values: list, bins: int = 12, width: int = 40, log: bool = False, label: str = "") -> str:
    """A horizontal text histogram (log-spaced bins for heavy-tailed lengths)."""
    vals = [v for v in values if v > 0] if log else list(values)
    if not vals:
        return "(empty)"
    lo, hi = min(vals), max(vals)
    if log:
        edges = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / bins) for i in range(bins + 1)]
    else:
        edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    counts = [0] * bins
    for v in vals:
        i = min(bins - 1, max(0, next((k for k in range(bins) if v <= edges[k + 1]), bins - 1)))
        counts[i] += 1
    top = max(counts) or 1
    lines = [label] if label else []
    for i, c in enumerate(counts):
        lines.append(f"{edges[i]:>8.0f}-{edges[i + 1]:<8.0f} {'#' * round(c / top * width):<{width}} {c}")
    return "\n".join(lines)


def curve(rows: list, x: str, ys: list, title: str = "") -> str:
    """Several series against one x, as a table plus a sparkline per series."""
    out = [table(rows, [x] + ys, title)]
    for y in ys:
        out.append(f"{y:>16}: {sparkline([r[y] for r in rows])}")
    return "\n".join(out)


def plot(rows: list, x: str, ys: list, title: str = "", path: str | None = None) -> bool:
    """A matplotlib chart when matplotlib is available (returns False otherwise; text output stands)."""
    try:
        import matplotlib
        matplotlib.use("Agg" if path else matplotlib.get_backend())
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    fig, axes = plt.subplots(1, len(ys), figsize=(4 * len(ys), 3))
    axes = axes if len(ys) > 1 else [axes]
    for ax, y in zip(axes, ys):
        ax.plot([r[x] for r in rows], [r[y] for r in rows], marker="o", ms=3)
        ax.set_xlabel(x)
        ax.set_title(y)
        ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.close(fig) if path else plt.show()
    return True


def write(data: dict, label: str, stem: str, out_dir: str = "_run_outputs") -> tuple:
    """Write ``<stem>.json`` and ``<stem>.md`` with the provenance label and a timestamp."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"label": label, "written": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    (d / f"{stem}.json").write_text(json.dumps(payload, indent=2, default=str))
    md = [f"# {stem}", "", f"**{label}** — written {payload['written']}", ""]
    for k, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            cols = list(v[0])
            md += [f"## {k}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            md += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in v] + [""]
        else:
            md += [f"- **{k}**: {v}"]
    (d / f"{stem}.md").write_text("\n".join(md) + "\n")
    return d / f"{stem}.json", d / f"{stem}.md"
