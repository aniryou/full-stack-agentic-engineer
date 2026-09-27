"""report.py — tables, text charts and a JSON + Markdown report that always say where the numbers came from.

One idea: a number without its provenance is a liability in a design review. Every table this lab prints
carries one of four labels, and :func:`write` stores it next to the data:

    MEASURED       a real run on this machine (the tiny transformers with torch) or a real server (vLLM at T1)
    SIMULATED      this lab's fake teacher or a model's output (its timings and answers are made up by design)
    PREDICTED      a calculator (the roofline cost model, the training-memory model): arithmetic, not a run
    ILLUSTRATIVE   bundled sample output in a documented format, or a recorded run from another machine
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

BARS = " ▁▂▃▄▅▆▇█"
LABELS = ("MEASURED", "SIMULATED", "PREDICTED", "ILLUSTRATIVE")


def fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return str(v)
        if v != 0 and abs(v) < 0.001:
            return f"{v:.2e}"
        return f"{v:,.4f}" if abs(v) < 1 else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
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


def histogram(values: list, bins: int = 10, width: int = 40, label: str = "") -> str:
    """A horizontal text histogram."""
    vals = list(values)
    if not vals:
        return "(empty)"
    lo, hi = min(vals), max(vals)
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)] if hi > lo else [lo - 0.5, lo + 0.5]
    counts = [0] * (len(edges) - 1)
    for v in vals:
        i = next((k for k in range(len(counts)) if v <= edges[k + 1]), len(counts) - 1)
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
        out.append(f"{y:>14}: {sparkline([r[y] for r in rows])}")
    return "\n".join(out)


def plot(series: dict, x: str, ys: list, title: str = "", path: str | None = None) -> bool:
    """One panel per metric in ``ys``, one line per named series (``{name: rows}``). Returns False when
    matplotlib is missing (the text tables stand on their own)."""
    try:
        import matplotlib
        if path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    fig, axes = plt.subplots(1, len(ys), figsize=(3.6 * len(ys), 3))
    axes = axes if len(ys) > 1 else [axes]
    for ax, y in zip(axes, ys):
        for name, rows in series.items():
            ax.plot([r[x] for r in rows], [r[y] for r in rows], marker="o", ms=2, label=name)
        ax.set_xlabel(x)
        ax.set_title(y)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
    plt.close(fig) if path else plt.show()
    return True


def write(data: dict, label: str, stem: str, out_dir: str = "_run_outputs") -> tuple:
    """Write ``<stem>.json`` and ``<stem>.md`` with the provenance label and a timestamp."""
    if label.split()[0].strip("[]:") not in LABELS:
        raise ValueError(f"label must start with one of {LABELS}")
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
