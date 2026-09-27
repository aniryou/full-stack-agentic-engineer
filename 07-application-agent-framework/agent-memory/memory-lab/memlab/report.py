"""report.py — every run leaves a JSON file and a Markdown table that says where its numbers came from.

One idea: a number without its provenance is a rumour. Each report records the tier, whether the
numbers are *simulated* (fake server, roofline), *measured* (a real server) or *computed* (the harness
on the scripted model, exactly reproducible), and the settings that produced them.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .harness import CATEGORIES, RunResult, wilson_interval


def _plain(x):
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def harness_markdown(results: dict[str, RunResult], provenance: str) -> str:
    lines = [f"**Planted-facts harness** — {provenance}", "",
             "| mode | correct | 95% Wilson | stale | hallucinated | paraphrase | memory tokens | input tokens | calls |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        k, n = r.rate()
        lo, hi = wilson_interval(k, n)
        kp, np_ = r.rate("correct", paraphrase=True)
        lines.append(f"| {name} | {k}/{n} | [{lo:.2f}, {hi:.2f}] | {r.rate('stale')[0]} | {r.rate('hallucinated')[0]} | "
                     f"{kp}/{np_} | {r.mean('memory_tokens'):.0f} | {r.mean('input_tokens'):.0f} | {r.mean('model_calls'):.2f} |")
    lines += ["", "| category | " + " | ".join(results) + " |", "|---|" + "---:|" * len(results)]
    for c in CATEGORIES:
        cells = []
        for r in results.values():
            k, n = r.rate("correct", c)
            cells.append(f"{k}/{n}" if n else "-")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cachebench_markdown(runs: dict) -> str:
    lines = ["| layout | hit rate | prefill ms (simulated roofline) | $ / session | source |", "|---|---:|---:|---:|---|"]
    for name, r in runs.items():
        lines.append(f"| {name} | {r.hit_rate:.1%} | {r.prefill_ms:.0f} | {r.cost_usd:.5f} | {r.source} |")
    return "\n".join(lines)


def save(name: str, payload, markdown: str, out_dir: str | Path = "reports", provenance: str = "") -> Path:
    """Write ``<out_dir>/<name>.json`` and ``.md``; returns the JSON path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (out / f"{name}.json").write_text(json.dumps({"name": name, "created": stamp, "provenance": provenance,
                                                  "data": _plain(payload)}, indent=2, default=str))
    (out / f"{name}.md").write_text(f"# {name}\n\n_{provenance} — {stamp}_\n\n{markdown}\n")
    return out / f"{name}.json"
