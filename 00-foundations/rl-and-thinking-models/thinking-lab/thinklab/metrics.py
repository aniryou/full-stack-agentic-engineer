"""metrics.py — vLLM's Prometheus metrics: the names, a tiny writer (for the fake server) and a parser.

One idea: nothing in vLLM's ``/metrics`` separates thinking from answering (v0.30.0 has no reasoning
metric) — thinking shows up as *more* of everything that decode drives: ``vllm:generation_tokens``,
longer ``vllm:request_generation_tokens``, more running requests (``vllm:num_requests_running``),
higher ``vllm:kv_cache_usage_perc`` and, once the pool is full, ``vllm:num_preemptions``. Per-request
reasoning counts come from the response instead: ``usage.completion_tokens_details.reasoning_tokens``.

Names are those of ``vllm/v1/metrics/loggers.py`` (v0.30.0) as the 04 lab's ``servelab.metrics``
lists them; histogram bucket edges are copied from ``vllm/v1/metrics/buckets.py`` via that lab.
Counters carry a ``_total`` suffix on the wire. Standard library only.
"""
from __future__ import annotations

import math
import re
import urllib.request
from dataclasses import dataclass

RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"
PREEMPTIONS = "vllm:num_preemptions"
PROMPT_TOKENS = "vllm:prompt_tokens"
GENERATION_TOKENS = "vllm:generation_tokens"
PREFIX_QUERIES = "vllm:prefix_cache_queries"
PREFIX_HITS = "vllm:prefix_cache_hits"
REQUEST_SUCCESS = "vllm:request_success"
TTFT = "vllm:time_to_first_token_seconds"
ITL = "vllm:inter_token_latency_seconds"
TPOT = "vllm:request_time_per_output_token_seconds"
E2E = "vllm:e2e_request_latency_seconds"
QUEUE = "vllm:request_queue_time_seconds"
REQUEST_PROMPT_TOKENS = "vllm:request_prompt_tokens"
REQUEST_GENERATION_TOKENS = "vllm:request_generation_tokens"
CACHE_CONFIG_INFO = "vllm:cache_config_info"

BUCKETS = {   # vllm/v1/metrics/buckets.py (v0.30.0), as copied in servelab.fakeserver
    "latency": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0],
    "ttft": [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0,
             7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0],
    "itl": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0,
            7.5, 10.0, 20.0, 40.0, 80.0],
}


def token_buckets(max_model_len: int) -> list:
    """vLLM's 1-2-5 series capped at max_model_len (``build_1_2_5_buckets``)."""
    out, e = [], 0
    while True:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if v > max_model_len:
                return out
            out.append(v)
        e += 1


# --- writer ------------------------------------------------------------------------------------
class Registry:
    """Just enough of the Prometheus client for one process: counters, gauges, histograms, one label set."""

    def __init__(self, labels: dict):
        self.labels = labels
        self.meta: dict = {}
        self.values: dict = {}

    def _key(self, name, extra):
        return name, tuple(sorted((extra or {}).items()))

    def counter(self, name, doc, value=1.0, extra=None):
        self.meta[name] = ("counter", doc)
        k = self._key(name, extra)
        self.values[k] = self.values.get(k, 0.0) + value

    def gauge(self, name, doc, value, extra=None):
        self.meta[name] = ("gauge", doc)
        self.values[self._key(name, extra)] = float(value)

    def observe(self, name, doc, value, buckets):
        self.meta[name] = ("histogram", doc)
        k = self._key(name, None)
        h = self.values.setdefault(k, {"buckets": list(buckets), "counts": [0] * len(buckets), "sum": 0.0, "count": 0})
        for i, b in enumerate(h["buckets"]):
            if value <= b:
                h["counts"][i] += 1
        h["sum"] += value
        h["count"] += 1

    def _lab(self, extra=()):
        items = list(self.labels.items()) + list(extra)
        return "{" + ",".join(f'{k}="{v}"' for k, v in items) + "}"

    def render(self) -> str:
        lines = []
        for name, (kind, doc) in sorted(self.meta.items()):
            wire = name + "_total" if kind == "counter" else name
            lines += [f"# HELP {wire} {doc}", f"# TYPE {wire} {kind}"]
            for (n, extra), v in sorted(self.values.items(), key=lambda kv: str(kv[0])):
                if n != name:
                    continue
                if kind == "histogram":
                    for b, c in zip(v["buckets"], v["counts"]):
                        lines.append(f"{name}_bucket{self._lab(list(extra) + [('le', repr(float(b)))])} {float(c)}")
                    lines.append(f"{name}_bucket{self._lab(list(extra) + [('le', '+Inf')])} {float(v['count'])}")
                    lines.append(f"{name}_sum{self._lab(extra)} {v['sum']}")
                    lines.append(f"{name}_count{self._lab(extra)} {float(v['count'])}")
                else:
                    lines.append(f"{wire}{self._lab(extra)} {v}")
        return "\n".join(lines) + "\n"


# --- parser ------------------------------------------------------------------------------------
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass
class Sample:
    name: str
    labels: dict
    value: float


def parse(text: str) -> list:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            out.append(Sample(m[1], dict(_LABEL.findall(m[2] or "")), float(m[3])))
    return out


def scrape(url: str, headers: dict | None = None, timeout: float = 10) -> list:
    req = urllib.request.Request(url.rstrip("/") + "/metrics", headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310
        return parse(r.read().decode())


def value(samples: list, name: str, **labels) -> float:
    """Sum of every sample called ``name`` (or ``name_total``) whose labels include ``labels``."""
    names = {name, name + "_total"}
    return sum(s.value for s in samples if s.name in names and labels.items() <= s.labels.items())


def histogram(samples: list, name: str) -> tuple:
    """([(le, cumulative count)], sum, count) summed over label sets."""
    b: dict = {}
    for s in samples:
        if s.name == name + "_bucket":
            le = math.inf if s.labels["le"] == "+Inf" else float(s.labels["le"])
            b[le] = b.get(le, 0.0) + s.value
    return sorted(b.items()), value(samples, name + "_sum"), value(samples, name + "_count")


def histogram_quantile(q: float, buckets: list) -> float:
    """PromQL's ``histogram_quantile``: linear interpolation inside the bucket holding rank q·N."""
    if not buckets or buckets[-1][1] == 0:
        return math.nan
    total = buckets[-1][1]
    rank = q * total
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / (c - prev_c) if c > prev_c else 0.0)
        prev_le, prev_c = le, c
    return prev_le


def delta(later: list, earlier: list) -> list:
    """Counters and histogram series of ``later`` minus ``earlier`` (gauges keep ``later``'s value)."""
    before = {(s.name, tuple(sorted(s.labels.items()))): s.value for s in earlier}
    out = []
    for s in later:
        k = (s.name, tuple(sorted(s.labels.items())))
        cumulative = s.name.endswith(("_total", "_bucket", "_sum", "_count"))
        out.append(Sample(s.name, s.labels, s.value - before.get(k, 0.0) if cumulative else s.value))
    return out


def summary(samples: list) -> dict:
    """The numbers a thinking workload moves, from one scrape (or a :func:`delta`)."""
    b, s, n = histogram(samples, REQUEST_GENERATION_TOKENS)
    itl_b, itl_s, itl_n = histogram(samples, ITL)
    return {"running": value(samples, RUNNING), "waiting": value(samples, WAITING),
            "kv_usage": value(samples, KV_USAGE), "preemptions": value(samples, PREEMPTIONS),
            "generation_tokens": value(samples, GENERATION_TOKENS),
            "requests": value(samples, REQUEST_SUCCESS),
            "mean_output_tokens": s / n if n else math.nan,
            "p90_output_tokens": histogram_quantile(0.9, b),
            "itl_mean_ms": 1e3 * itl_s / itl_n if itl_n else math.nan,
            "itl_p99_ms": 1e3 * histogram_quantile(0.99, itl_b)}
