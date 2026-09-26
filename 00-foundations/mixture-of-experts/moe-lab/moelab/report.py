"""report.py — write what a run found as JSON and Markdown, with every number's provenance on it.

One idea: a number without its source misleads the next reader. Every section carries one of three
labels — **measured** (a real GPU or server), **simulated** (a model in this lab), **illustrative**
(bundled sample data) — plus the environment it came from, so a report pasted into a design review
says what it can and cannot support.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__, env

LABELS = (env.MEASURED, env.SIMULATED, env.ILLUSTRATIVE)


@dataclass
class Section:
    heading: str
    label: str
    body: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Report:
    title: str
    sections: list = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    environment: str = field(default_factory=env.describe)

    def add(self, heading: str, label: str, body: str = "", **data) -> "Report":
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}, not {label!r}")
        self.sections.append(Section(heading, label, body, data))
        return self

    def to_markdown(self) -> str:
        out = [f"# {self.title}", "",
               f"*{self.created} · moelab {__version__} · Python {platform.python_version()} · {self.environment}*", ""]
        for s in self.sections:
            out += [f"## {s.heading} [{s.label.upper()}]", ""]
            if s.body:
                out += ["```", s.body.rstrip(), "```", ""]
            for k, v in s.data.items():
                out.append(f"- **{k}**: {v}")
            if s.data:
                out.append("")
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    def save(self, directory: str | Path = "results", stem: str = "moelab-report") -> tuple[Path, Path]:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        md, js = d / f"{stem}.md", d / f"{stem}.json"
        md.write_text(self.to_markdown())
        js.write_text(self.to_json())
        return md, js
