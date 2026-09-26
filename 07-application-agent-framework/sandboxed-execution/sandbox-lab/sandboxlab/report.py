"""report.py — one JSON document and one Markdown page per run: verdicts, latencies, provenance.

One idea: a sandbox claim is only as good as its evidence, so every report says where each row
came from — measured on this machine, simulated, or sample output in the documented format
(illustrative) — and which machine it was (``env.capabilities()``).
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path

from . import env
from .bench import Measurement
from .probes import PROBES, ProbeResult


def build(verdicts: dict[str, list[ProbeResult]] | None = None, latency: list[Measurement] | None = None) -> dict:
    caps = env.capabilities()
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "machine": asdict(caps), "levels_measurable_here": caps.levels(),
            "verdicts": {k: [r.to_dict() for r in v] for k, v in (verdicts or {}).items()},
            "latency": [asdict(m) for m in latency or []]}


def markdown(doc: dict) -> str:
    lines = [f"# Sandbox report ({doc['generated']})", "",
             f"Machine: `{doc['machine']['os']}`, root: {doc['machine']['root']}, netns: {doc['machine']['netns']}, "
             f"docker: {doc['machine']['docker']}, runsc: {doc['machine']['runsc']}", ""]
    if doc["verdicts"]:
        cols = list(doc["verdicts"])
        lines += ["## Probe verdicts", "", "| probe | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        by = {c: {r["probe"]: r for r in doc["verdicts"][c]} for c in cols}
        for p in PROBES:
            if any(p.name in by[c] for c in cols):
                lines.append(f"| {p.name} | " + " | ".join(by[c].get(p.name, {}).get("verdict", "-") for c in cols) + " |")
        labels = sorted({r["label"] for c in cols for r in doc["verdicts"][c]})
        lines += ["", "Provenance: " + "; ".join(labels), ""]
    if doc["latency"]:
        lines += ["## Start-up and round-trip latency", "", "| level | p50 ms | p95 ms | n | provenance |", "|---|---:|---:|---:|---|"]
        lines += [f"| {m['level']} | {m['p50_ms']:.1f} | {m['p95_ms']:.1f} | {m['n']} | {m['label']} |" for m in doc["latency"]]
    return "\n".join(lines) + "\n"


def write(doc: dict, out_dir: str | Path = "reports") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = doc["generated"].replace(":", "").replace("-", "")
    j, m = out / f"sandbox-report-{stamp}.json", out / f"sandbox-report-{stamp}.md"
    j.write_text(json.dumps(doc, indent=1) + "\n")
    m.write_text(markdown(doc))
    return j, m
