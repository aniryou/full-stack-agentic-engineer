"""Prometheus text format: parse what engines expose, render what our own servers expose.

The one idea: a `/metrics` endpoint is plain text — named samples with labels. Gauges are
"now" values (queue depth, KV-cache usage), counters only go up (so you diff them to get a
rate), and histograms are *cumulative* buckets from which a quantile can only be estimated
by linear interpolation inside one bucket (exactly what PromQL's `histogram_quantile` does).

A router scrapes this text from every engine replica many times per second; an autoscaler
reads it through a metrics pipeline. Both are only as good as the numbers in here.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass

__all__ = ["Sample", "Families", "parse", "histogram_quantile", "Registry"]


@dataclass(frozen=True)
class Sample:
    name: str
    labels: tuple  # sorted ((key, value), ...) so samples are hashable
    value: float

    def label(self, key: str, default: str | None = None) -> str | None:
        return dict(self.labels).get(key, default)


_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+-?\d+)?\s*$")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')


def _unescape(v: str) -> str:
    return v.replace("\\\\", "\x00").replace('\\"', '"').replace("\\n", "\n").replace("\x00", "\\")


def _value(s: str) -> float:
    s = s.strip()
    if s in ("+Inf", "Inf"):
        return math.inf
    if s == "-Inf":
        return -math.inf
    return float(s)  # also handles NaN


def parse(text: str) -> list[Sample]:
    """Parse Prometheus text exposition (format 0.0.4) into samples. Comments are skipped."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"not a Prometheus sample line: {raw!r}")
        name, _, body, val, _ts = m.groups()
        labels = tuple(sorted((k, _unescape(v)) for k, v in _LABEL.findall(body or "")))
        out.append(Sample(name, labels, _value(val)))
    return out


class Families:
    """Index samples by metric name, with small helpers a scraper needs."""

    def __init__(self, samples):
        self.by_name: dict[str, list[Sample]] = {}
        for s in samples:
            self.by_name.setdefault(s.name, []).append(s)

    @classmethod
    def from_text(cls, text: str) -> "Families":
        return cls(parse(text))

    def get(self, name: str, **labels) -> list[Sample]:
        want = {k: str(v) for k, v in labels.items()}
        return [s for s in self.by_name.get(name, []) if all(dict(s.labels).get(k) == v for k, v in want.items())]

    def has(self, name: str) -> bool:
        return name in self.by_name

    def value(self, name: str, default=None, **labels):
        """First matching sample's value (or `default`)."""
        got = self.get(name, **labels)
        return got[0].value if got else default

    def sum(self, name: str, default=None, **labels):
        """Sum over all label sets (e.g. one gauge per engine index)."""
        got = self.get(name, **labels)
        return sum(s.value for s in got) if got else default

    def max(self, name: str, default=None, **labels):
        got = self.get(name, **labels)
        return max(s.value for s in got) if got else default

    def buckets(self, name: str, **labels) -> list[tuple[float, float]]:
        """Cumulative (le, count) pairs of histogram `name` (pass the base name, not `_bucket`),
        summed across any other labels."""
        acc: dict[float, float] = {}
        for s in self.get(name + "_bucket", **labels):
            le = _value(dict(s.labels)["le"])
            acc[le] = acc.get(le, 0.0) + s.value
        return sorted(acc.items())


def histogram_quantile(q: float, buckets) -> float:
    """PromQL `histogram_quantile` on cumulative (le, count) buckets.

    Finds the bucket holding the q-th observation and interpolates linearly inside it.
    The top bucket must be +Inf; if the rank lands there, the answer is the highest finite
    bound (the estimate cannot say more). Worked example: buckets (0.1: 50, 0.5: 90, +Inf: 100),
    q = 0.9 -> rank 90 lands exactly at the top of the 0.5 bucket -> 0.5.
    """
    if q < 0:
        return -math.inf
    if q > 1:
        return math.inf
    b = sorted((float(le), float(c)) for le, c in buckets)
    if not b or not math.isinf(b[-1][0]):
        return math.nan
    # coalesce duplicate bounds and force monotonic counts (as Prometheus does)
    merged: list[list[float]] = []
    for le, c in b:
        if merged and merged[-1][0] == le:
            merged[-1][1] += c
        else:
            merged.append([le, c])
    for i in range(1, len(merged)):
        merged[i][1] = max(merged[i][1], merged[i - 1][1])
    if len(merged) < 2:
        return math.nan
    total = merged[-1][1]
    if total == 0:
        return math.nan
    rank = q * total
    # first bucket (excluding +Inf) whose cumulative count reaches the rank, like sort.Search
    idx = len(merged) - 1
    for i in range(len(merged) - 1):
        if merged[i][1] >= rank:
            idx = i
            break
    if idx == len(merged) - 1:
        return merged[-2][0]
    if idx == 0 and merged[0][0] <= 0:
        return merged[0][0]
    end, count = merged[idx]
    start = 0.0
    if idx > 0:
        start = merged[idx - 1][0]
        count -= merged[idx - 1][1]
        rank -= merged[idx - 1][1]
    return start + (end - start) * (rank / count)


# ------------------------------------------------------------------ exposition
def _fmt(v: float) -> str:
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return f"{int(v)}.0"
    return repr(float(v))


def _labelstr(names, values, extra=()) -> str:
    pairs = list(zip(names, values)) + list(extra)
    if not pairs:
        return ""
    def esc(v):
        return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in pairs) + "}"


class _Metric:
    kind = ""

    def __init__(self, name, doc, labelnames=()):
        self.name, self.doc, self.labelnames = name, doc, tuple(labelnames)
        self._children: dict[tuple, object] = {}
        self._lock = threading.Lock()

    def labels(self, *values, **kv):
        key = tuple(str(kv[n]) for n in self.labelnames) if kv else tuple(str(v) for v in values)
        if len(key) != len(self.labelnames):
            raise ValueError(f"{self.name}: expected labels {self.labelnames}")
        with self._lock:
            if key not in self._children:
                self._children[key] = self._new_child()
            return self._children[key]

    def _default(self):
        return self.labels() if not self.labelnames else None


class _Value:
    def __init__(self):
        self.v = 0.0

    def inc(self, amount: float = 1.0):
        self.v += amount

    def dec(self, amount: float = 1.0):
        self.v -= amount

    def set(self, v: float):
        self.v = float(v)


class Counter(_Metric):
    """Exposed with the `_total` suffix, as prometheus_client (and hence vLLM) does."""
    kind = "counter"

    def _new_child(self):
        return _Value()

    def inc(self, amount=1.0):
        self._default().inc(amount)

    def render(self):
        exp = self.name if self.name.endswith("_total") else self.name + "_total"
        lines = [f"# HELP {exp} {self.doc}", f"# TYPE {exp} counter"]
        for key, child in sorted(self._children.items()):
            lines.append(f"{exp}{_labelstr(self.labelnames, key)} {_fmt(child.v)}")
        return lines


class Gauge(_Metric):
    kind = "gauge"

    def _new_child(self):
        return _Value()

    def set(self, v):
        self._default().set(v)

    def inc(self, amount=1.0):
        self._default().inc(amount)

    def dec(self, amount=1.0):
        self._default().dec(amount)

    def render(self):
        lines = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} gauge"]
        for key, child in sorted(self._children.items()):
            lines.append(f"{self.name}{_labelstr(self.labelnames, key)} {_fmt(child.v)}")
        return lines


class _Hist:
    def __init__(self, bounds):
        self.bounds = bounds
        self.counts = [0] * len(bounds)
        self.sum = 0.0
        self.count = 0

    def observe(self, v: float):
        self.sum += v
        self.count += 1
        for i, b in enumerate(self.bounds):
            if v <= b:
                self.counts[i] += 1


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name, doc, buckets, labelnames=()):
        super().__init__(name, doc, labelnames)
        self.bounds = sorted(float(b) for b in buckets) + [math.inf]

    def _new_child(self):
        return _Hist(self.bounds)

    def observe(self, v):
        self._default().observe(v)

    def render(self):
        lines = [f"# HELP {self.name} {self.doc}", f"# TYPE {self.name} histogram"]
        for key, h in sorted(self._children.items()):
            for b, c in zip(h.bounds, h.counts):
                lines.append(f"{self.name}_bucket{_labelstr(self.labelnames, key, [('le', _fmt(b))])} {_fmt(c)}")
            lines.append(f"{self.name}_sum{_labelstr(self.labelnames, key)} {_fmt(h.sum)}")
            lines.append(f"{self.name}_count{_labelstr(self.labelnames, key)} {_fmt(h.count)}")
        return lines


class Registry:
    """A minimal metrics registry: create metrics, then `render()` the exposition text."""

    def __init__(self):
        self.metrics: list[_Metric] = []

    def _add(self, m):
        self.metrics.append(m)
        return m

    def counter(self, name, doc, labelnames=()):
        return self._add(Counter(name, doc, labelnames))

    def gauge(self, name, doc, labelnames=()):
        return self._add(Gauge(name, doc, labelnames))

    def histogram(self, name, doc, buckets, labelnames=()):
        return self._add(Histogram(name, doc, buckets, labelnames))

    def render(self) -> str:
        lines = []
        for m in self.metrics:
            lines += m.render()
        return "\n".join(lines) + "\n"
