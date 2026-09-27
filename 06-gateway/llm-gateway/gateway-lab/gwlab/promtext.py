"""Prometheus text format, both ways: render the gateway's and the fakes' `/metrics`, parse an engine's.

The one idea: a `/metrics` page is plain text — named samples with labels. Counters only go up (diff two
scrapes to get what happened in between), gauges are "now" values, histograms are cumulative buckets. The
gateway exposes its own (`gwlab_*`); the fake providers expose the vLLM names the ledger is reconciled against
(`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, ...). A standard-library stand-in for
`prometheus_client`, small enough to read.
"""
from __future__ import annotations

import math
import re
import threading

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+-?\d+)?\s*$")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')


def _fmt(v: float) -> str:
    if v == math.inf:
        return "+Inf"
    return repr(float(v)) if not float(v).is_integer() else f"{float(v):.1f}"


def _labels(d: dict) -> str:
    if not d:
        return ""
    esc = lambda s: str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")  # noqa: E731
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in d.items()) + "}"


class _Metric:
    def __init__(self, reg, kind, name, doc, labelnames, buckets=None):
        self.kind, self.name, self.doc, self.labelnames = kind, name, doc, tuple(labelnames)
        self.buckets = tuple(buckets or ()) + ((math.inf,) if buckets else ())
        self.values: dict[tuple, object] = {}
        self._lock = reg._lock

    def labels(self, **kw) -> "_Child":
        key = tuple(str(kw.get(n, "")) for n in self.labelnames)
        return _Child(self, key)

    # unlabelled shortcuts
    def inc(self, v: float = 1.0):
        self.labels().inc(v)

    def set(self, v: float):
        self.labels().set(v)

    def observe(self, v: float):
        self.labels().observe(v)

    def get(self, **kw) -> float:
        key = tuple(str(kw.get(n, "")) for n in self.labelnames)
        v = self.values.get(key, 0.0)
        return v[-1] if isinstance(v, list) else v

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} {self.kind}"]
        for key, v in sorted(self.values.items()):
            lab = dict(zip(self.labelnames, key))
            if self.kind == "histogram":
                counts, total, n = v[:-2], v[-2], v[-1]
                cum = 0
                for le, c in zip(self.buckets, counts):
                    cum += c
                    out.append(f"{self.name}_bucket{_labels({**lab, 'le': _fmt(le)})} {_fmt(cum)}")
                out.append(f"{self.name}_sum{_labels(lab)} {_fmt(total)}")
                out.append(f"{self.name}_count{_labels(lab)} {_fmt(n)}")
            else:
                suffix = "_total" if self.kind == "counter" and not self.name.endswith("_total") else ""
                out.append(f"{self.name}{suffix}{_labels(lab)} {_fmt(v)}")
        return out


class _Child:
    def __init__(self, m: _Metric, key: tuple):
        self.m, self.key = m, key

    def inc(self, v: float = 1.0):
        with self.m._lock:
            self.m.values[self.key] = self.m.values.get(self.key, 0.0) + v

    def set(self, v: float):
        with self.m._lock:
            self.m.values[self.key] = float(v)

    def observe(self, v: float):
        with self.m._lock:
            cur = self.m.values.get(self.key) or [0] * len(self.m.buckets) + [0.0, 0]
            for i, le in enumerate(self.m.buckets):
                if v <= le:
                    cur[i] += 1
                    break
            cur[-2] += v
            cur[-1] += 1
            self.m.values[self.key] = cur


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self.metrics: list[_Metric] = []

    def _add(self, *a, **kw) -> _Metric:
        m = _Metric(self, *a, **kw)
        self.metrics.append(m)
        return m

    def counter(self, name, doc, labelnames=()):
        return self._add("counter", name, doc, labelnames)

    def gauge(self, name, doc, labelnames=()):
        return self._add("gauge", name, doc, labelnames)

    def histogram(self, name, doc, buckets, labelnames=()):
        return self._add("histogram", name, doc, labelnames, buckets=buckets)

    def render(self) -> str:
        with self._lock:
            lines = [line for m in self.metrics for line in m.render()]
        return "\n".join(lines) + "\n"


def parse(text: str) -> dict[str, list[tuple[dict, float]]]:
    """Prometheus text -> {sample name: [(labels, value), ...]}. Comments are skipped."""
    out: dict[str, list[tuple[dict, float]]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"not a Prometheus sample line: {raw!r}")
        name, _, body, val, _ts = m.groups()
        labels = {k: v for k, v in _LABEL.findall(body or "")}
        value = math.inf if val in ("+Inf", "Inf") else float(val)
        out.setdefault(name, []).append((labels, value))
    return out


def total(samples: dict, name: str, **labels) -> float:
    """Sum of every sample called `name` whose labels include `labels` (0.0 when absent)."""
    return sum(v for lab, v in samples.get(name, []) if all(lab.get(k) == str(x) for k, x in labels.items()))
