"""One JSON file and one Markdown page per run, so a laptop, a Colab T4 and a GCP L4 line up.

The JSON keeps everything (every timing sample, the counted cost, the machine description,
what was skipped and why) so results can be re-analysed later; the Markdown is the page you
read or paste into a design doc. Tables show both the best and the median sample and name
the byte convention, because "12 GB/s" without those two facts is not a result.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import socket
from dataclasses import dataclass, field
from pathlib import Path

from .measure import Measurement, si

SCHEMA = "gpubench.report/v1"


def table(rows, columns) -> str:
    """Markdown table. ``columns`` = ``[(header, function(row) -> str), ...]``."""
    head = "| " + " | ".join(h for h, _ in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(f(r)) for _, f in columns) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def bar_chart(pairs, unit: str = "", width: int = 40) -> str:
    """Horizontal ASCII bars for ``[(label, value), ...]`` — enough to see a plateau or a knee."""
    pairs = list(pairs)
    top = max((v for _, v in pairs), default=0) or 1
    lw = max((len(str(label)) for label, _ in pairs), default=0)
    return "\n".join(f"{str(label):>{lw}} | {'█' * max(1, round(width * v / top)):<{width}} {si(v, unit)}"
                     for label, v in pairs)


def _gemm_cols(stat):
    return [("dtype", lambda m: m.params["dtype"]),
            ("m×n×k", lambda m: f"{m.params['m']}×{m.params['n']}×{m.params['k']}"),
            ("time", lambda m: si(m.seconds(stat), "s")),
            ("FLOP/s (best)", lambda m: si(m.flops_per_s("best"), "FLOP/s")),
            ("FLOP/s (median)", lambda m: si(m.flops_per_s("median"), "FLOP/s")),
            ("FLOP/B", lambda m: f"{m.intensity:.1f}"),
            ("note", lambda m: m.note)]


def _stream_cols(stat):
    return [("kernel", lambda m: m.op.split(".", 1)[1]),
            ("elements", lambda m: f"{m.params['n']:,}"),
            ("threads", lambda m: m.params.get("threads", 1)),
            ("bytes/call", lambda m: si(m.cost.bytes, "B")),
            ("moved (best)", lambda m: si(m.bytes_per_s("best"), "B/s")),
            ("moved (median)", lambda m: si(m.bytes_per_s("median"), "B/s")),
            ("STREAM convention", lambda m: si(m.extras.get("stream_bytes", m.cost.bytes) / m.seconds(stat), "B/s")),
            ("note", lambda m: m.note)]


def _xfer_cols(stat):
    return [("op", lambda m: m.op),
            ("size", lambda m: si(m.params.get("nbytes", m.params.get("working_set", 0)), "B")),
            ("detail", lambda m: ", ".join(f"{k}={v}" for k, v in m.params.items() if k not in ("nbytes", "working_set"))),
            ("time", lambda m: si(m.seconds(stat), "s")),
            ("rate (best)", lambda m: si(m.bytes_per_s("best"), "B/s")),
            ("rate (median)", lambda m: si(m.bytes_per_s("median"), "B/s")),
            ("note", lambda m: m.note)]


def measurements_markdown(ms, stat: str = "best") -> str:
    """Group measurements by family and render one table per family."""
    families = {"GEMM": [m for m in ms if m.op == "gemm"],
                "Memory bandwidth (STREAM)": [m for m in ms if m.op.startswith("stream.")],
                "Other memory experiments": [m for m in ms if m.op.startswith(("chain.", "ladder"))],
                "Transfers and copies": [m for m in ms if m.op in ("memcpy", "h2d", "d2h") or m.op.startswith("p2p")],
                "Weights loading": [m for m in ms if m.op.startswith(("load.", "file_to_device"))]}
    cols = {"GEMM": _gemm_cols, "Memory bandwidth (STREAM)": _stream_cols}
    parts = []
    for title, rows in families.items():
        if rows:
            parts.append(f"### {title}\n\n" + table(rows, cols.get(title, _xfer_cols)(stat)))
    return "\n\n".join(parts)


@dataclass
class Report:
    meta: dict = field(default_factory=dict)
    measurements: list = field(default_factory=list)
    analyses: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)

    @classmethod
    def new(cls, backend=None, **extra) -> "Report":
        meta = {"schema": SCHEMA, "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "host": socket.gethostname(), "python": platform.python_version(), "platform": platform.platform()}
        if backend is not None:
            meta["backend"] = backend.name
            meta["device"] = backend.describe()
        meta.update(extra)
        return cls(meta=meta)

    def extend(self, ms) -> "Report":
        self.measurements.extend(ms)
        return self

    def to_dict(self) -> dict:
        return {"meta": self.meta, "measurements": [m.to_dict() for m in self.measurements],
                "analyses": self.analyses, "skipped": self.skipped}

    @classmethod
    def from_dict(cls, d: dict) -> "Report":
        return cls(d.get("meta", {}), [Measurement.from_dict(m) for m in d.get("measurements", [])],
                   d.get("analyses", {}), d.get("skipped", []))

    def save(self, out_dir, stem: str | None = None) -> tuple:
        """Write ``<stem>.json`` and ``<stem>.md`` into ``out_dir``; returns both paths."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if stem is None:
            stamp = self.meta.get("created_utc", "run").replace(":", "").replace("-", "")[:15]
            stem = f"gpubench-{self.meta.get('host', 'host')}-{self.meta.get('backend', 'x')}-{stamp}"
        jp, mp = out / f"{stem}.json", out / f"{stem}.md"
        jp.write_text(json.dumps(self.to_dict(), indent=1, default=str))
        mp.write_text(self.to_markdown())
        return jp, mp

    @classmethod
    def load(cls, path) -> "Report":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_markdown(self, stat: str = "best") -> str:
        dev = self.meta.get("device", {})
        lines = [f"# gpubench report — {dev.get('name', self.meta.get('host', ''))}", "",
                 f"*{self.meta.get('created_utc', '')} · backend `{self.meta.get('backend', '?')}` · "
                 f"host `{self.meta.get('host', '?')}` · every number below was measured on this machine "
                 f"by this run, except rows explicitly marked spec/model.*", ""]
        if dev:
            lines += ["## Machine", "", table(sorted(dev.items()), [("field", lambda kv: kv[0]),
                                                                  ("value", lambda kv: kv[1])]), ""]
        if self.measurements:
            lines += ["## Measurements", "", f"Time and rates use the **{stat}** sample unless a column says "
                      "otherwise; *moved* counts the bytes the code actually read and wrote.", "",
                      measurements_markdown(self.measurements, stat), ""]
        if self.analyses:
            lines += ["## Analyses", ""]
            for name, value in self.analyses.items():
                lines += [f"### {name}", "", "```", json.dumps(value, indent=1, default=str), "```", ""]
        if self.skipped:
            lines += ["## Skipped", "", table(self.skipped, [("what", lambda s: ", ".join(
                f"{k}={v}" for k, v in s.items() if k != "reason")), ("reason", lambda s: s.get("reason", ""))]), ""]
        return "\n".join(lines)
