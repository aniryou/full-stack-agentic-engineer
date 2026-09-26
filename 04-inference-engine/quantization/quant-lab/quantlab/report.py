"""report.py — one table per decision, every number labelled with where it came from.

One idea: a quantization decision mixes numbers of very different standing — bytes counted from a
file (exact), step times from a roofline emulator (simulated), task accuracy from an eval run
(measured, with an error bar), numbers copied from a tool's documented output (sample). A report
that prints them side by side must say which is which, or a simulated 2x reads like a measured
one. ``Report`` collects rows with a ``source`` per column group and renders markdown and JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SOURCES = {
    "exact": "counted from the checkpoint files",
    "simulated": "SIMULATED (roofline emulator; efficiencies are assumptions)",
    "measured": "MEASURED on real hardware",
    "sample": "sample output in the documented format (illustrative)",
    "t0-eval": "measured on the bundled tiny model (T0), not on your model",
}


def fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        if abs(v) >= 0.01 or v == 0:
            return f"{v:.3f}"
        return f"{v:.2e}"
    if isinstance(v, tuple):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def markdown(rows: list, columns: list | None = None) -> str:
    if not rows:
        return "(no rows)"
    columns = columns or list(rows[0])
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    out += ["| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join(out)


@dataclass
class Report:
    title: str
    sections: list = field(default_factory=list)      # (heading, source key, rows, columns)

    def add(self, heading: str, source: str, rows: list, columns: list | None = None) -> "Report":
        if source not in SOURCES:
            raise KeyError(f"source must be one of {', '.join(SOURCES)}")
        self.sections.append((heading, source, rows, columns))
        return self

    def to_markdown(self) -> str:
        parts = [f"# {self.title}"]
        for heading, source, rows, cols in self.sections:
            parts += [f"\n## {heading}", f"*Source: {SOURCES[source]}.*\n", markdown(rows, cols)]
        return "\n".join(parts) + "\n"

    def to_json(self) -> dict:
        return {"title": self.title, "sections": [{"heading": h, "source": s, "source_label": SOURCES[s], "rows": r}
                                                  for h, s, r, _ in self.sections]}

    def save(self, stem) -> tuple:
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        md, js = stem.with_suffix(".md"), stem.with_suffix(".json")
        md.write_text(self.to_markdown())
        js.write_text(json.dumps(self.to_json(), indent=2, default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
        return md, js
