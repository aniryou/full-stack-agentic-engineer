"""metrics.py — read an engine's own gauges: parse vLLM's Prometheus ``/metrics``.

One idea: the client sees latency; the engine's ``/metrics`` shows *why*. Queue depth
(``vllm:num_requests_waiting``), batch size (``vllm:num_requests_running``), KV pressure
(``vllm:kv_cache_usage_perc``), prefix-cache hit rate and preemptions explain a latency number
that a benchmark alone cannot. Two rules make the numbers honest:

* counters and histograms are **cumulative since start** — subtract two scrapes (:func:`delta`)
  to talk about a window, exactly as PromQL's ``rate()``/``increase()`` do;
* a latency percentile from a histogram is an **interpolation inside a bucket**
  (:func:`histogram_quantile`, the same algorithm as PromQL's ``histogram_quantile``), so it is
  only as precise as the bucket edges around it.

Metric names are from ``vllm/v1/metrics/loggers.py`` and ``v1/spec_decode/metrics.py`` (vLLM v0.30.0;
same on main, Sep 2026). The Prometheus
client exposes counters with a ``_total`` suffix; this parser accepts either spelling.
"""
from __future__ import annotations

import math
import re
import time
import urllib.request
from dataclasses import asdict, dataclass

# --- the vLLM metric names this lab reads (and the fake server emits) ------------------------
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_USAGE = "vllm:kv_cache_usage_perc"                 # 0-1
KV_USAGE_OLD = "vllm:gpu_cache_usage_perc"            # older releases
PREFIX_QUERIES = "vllm:prefix_cache_queries"          # counter, in tokens
PREFIX_HITS = "vllm:prefix_cache_hits"                # counter, in tokens
PREFIX_QUERIES_OLD = "vllm:gpu_prefix_cache_queries"  # older V1 releases (verify)
PREFIX_HITS_OLD = "vllm:gpu_prefix_cache_hits"
PREEMPTIONS = "vllm:num_preemptions"
PROMPT_TOKENS = "vllm:prompt_tokens"
GENERATION_TOKENS = "vllm:generation_tokens"
REQUEST_SUCCESS = "vllm:request_success"
TTFT = "vllm:time_to_first_token_seconds"
ITL = "vllm:inter_token_latency_seconds"
TPOT = "vllm:request_time_per_output_token_seconds"
E2E = "vllm:e2e_request_latency_seconds"
QUEUE = "vllm:request_queue_time_seconds"
PREFILL_TIME = "vllm:request_prefill_time_seconds"
DECODE_TIME = "vllm:request_decode_time_seconds"
INFERENCE_TIME = "vllm:request_inference_time_seconds"
ITERATION_TOKENS = "vllm:iteration_tokens_total"
REQUEST_PROMPT_TOKENS = "vllm:request_prompt_tokens"
REQUEST_GENERATION_TOKENS = "vllm:request_generation_tokens"
CACHE_CONFIG_INFO = "vllm:cache_config_info"
SPEC_DRAFTS = "vllm:spec_decode_num_drafts"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens"
SPEC_ACCEPTED_PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"

# The same questions in PromQL, for Cloud Monitoring / Grafana (a 5-minute window).
PROMQL = {
    "prefix_hit_rate": "sum(rate(vllm:prefix_cache_hits_total[5m])) / sum(rate(vllm:prefix_cache_queries_total[5m]))",
    "ttft_p99": "histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket[5m])))",
    "itl_p99": "histogram_quantile(0.99, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket[5m])))",
    "queue_depth": "sum(vllm:num_requests_waiting)",
    "kv_usage": "max(vllm:kv_cache_usage_perc)",
    "generation_tps": "sum(rate(vllm:generation_tokens_total[5m]))",
    "preemptions_per_s": "sum(rate(vllm:num_preemptions_total[5m]))",
    "spec_acceptance_rate": "sum(rate(vllm:spec_decode_num_accepted_tokens_total[5m])) / sum(rate(vllm:spec_decode_num_draft_tokens_total[5m]))",
}


@dataclass
class Sample:
    name: str
    labels: dict
    value: float


@dataclass
class Histogram:
    buckets: list          # [(upper_bound, cumulative_count)], ascending, ends with +Inf
    count: float
    sum: float

    def quantile(self, q: float) -> float:
        return histogram_quantile(q, self.buckets)

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else math.nan


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?\s*$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')
_SUFFIXES = ("_bucket", "_count", "_sum", "_total", "_created")


def _unescape(v: str) -> str:
    return v.replace("\\\\", "\x00").replace('\\"', '"').replace("\\n", "\n").replace("\x00", "\\")


def _float(v: str) -> float:
    special = {"+Inf": math.inf, "Inf": math.inf, "-Inf": -math.inf}
    return special[v] if v in special else float(v)


class Scrape:
    """One parsed ``/metrics`` page: samples plus the ``# TYPE`` of each family."""

    def __init__(self, samples: list, types: dict, timestamp: float | None = None):
        self.samples, self.types = samples, types
        self.timestamp = time.time() if timestamp is None else timestamp

    # -- lookups ---------------------------------------------------------------------------
    def family_type(self, sample_name: str) -> str:
        if sample_name in self.types:
            return self.types[sample_name]
        for s in _SUFFIXES:
            if sample_name.endswith(s) and sample_name[: -len(s)] in self.types:
                return self.types[sample_name[: -len(s)]]
        return "untyped"

    def find(self, name: str, **match) -> list:
        return [s for s in self.samples if s.name == name and all(s.labels.get(k) == str(v) for k, v in match.items())]

    def value(self, name: str, default: float = math.nan, **match) -> float:
        """Sum of the samples called ``name`` (or ``name_total``) whose labels match."""
        for n in (name, name + "_total"):
            found = self.find(n, **match)
            if found:
                return float(sum(s.value for s in found))
        return default

    def has(self, name: str) -> bool:
        return any(s.name in (name, name + "_total", name + "_count") for s in self.samples)

    def histogram(self, name: str, **match) -> Histogram:
        """The histogram ``name`` summed over all label sets that match (PromQL ``sum by (le)``)."""
        by_le: dict = {}
        for s in self.find(name + "_bucket", **match):
            le = _float(s.labels["le"])
            by_le[le] = by_le.get(le, 0.0) + s.value
        return Histogram(sorted(by_le.items()), self.value(name + "_count", 0.0, **match),
                         self.value(name + "_sum", 0.0, **match))

    def names(self) -> set:
        out = set()
        for s in self.samples:
            n = s.name
            for suf in _SUFFIXES:
                if n.endswith(suf):
                    n = n[: -len(suf)]
                    break
            out.add(n)
        return out


def parse(text: str, timestamp: float | None = None) -> Scrape:
    """Parse the Prometheus text exposition format (what ``curl :8000/metrics`` returns)."""
    samples, types = [], {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                types[parts[2]] = parts[3].strip()
                if parts[3].strip() == "counter" and parts[2].endswith("_total"):
                    types[parts[2][: -len("_total")]] = "counter"
            continue
        mt = _SAMPLE.match(line)
        if not mt:
            continue
        name, labels_s, value = mt.groups()
        if name.endswith("_created"):
            continue  # creation timestamps: not a measurement
        labels = {k: _unescape(v) for k, v in _LABEL.findall(labels_s or "")}
        samples.append(Sample(name, labels, _float(value)))
    return Scrape(samples, types, timestamp)


def scrape(url: str, timeout: float = 5.0, headers: dict | None = None) -> Scrape:
    """GET ``<url>/metrics`` (or ``url`` itself if it already ends in /metrics) and parse it."""
    if not url.rstrip("/").endswith("/metrics"):
        url = url.rstrip("/") + "/metrics"
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (lab tool, user-supplied URL)
        return parse(resp.read().decode("utf-8"), timestamp=time.time())


def delta(later: Scrape, earlier: Scrape) -> Scrape:
    """What happened *between* two scrapes: counters and histogram series are subtracted
    (a negative difference means the server restarted: keep the later value, as ``rate()``
    does); gauges keep their later value."""
    before = {(s.name, tuple(sorted(s.labels.items()))): s.value for s in earlier.samples}
    out = []
    for s in later.samples:
        if later.family_type(s.name) in ("gauge", "untyped") and not s.name.endswith(("_bucket", "_count", "_sum", "_total")):
            out.append(s)
            continue
        prev = before.get((s.name, tuple(sorted(s.labels.items()))), 0.0)
        out.append(Sample(s.name, s.labels, s.value - prev if s.value >= prev else s.value))
    return Scrape(out, later.types, later.timestamp)


def histogram_quantile(q: float, buckets) -> float:
    """PromQL ``histogram_quantile`` for classic histograms (Prometheus ``BucketQuantile``).

    ``buckets`` is ``[(upper_bound, cumulative_count), ...]`` and must end with ``+Inf``.
    rank = q × total; find the first bucket whose cumulative count >= rank; interpolate
    linearly inside it, from the previous bound (or 0 for the first bucket) to its upper bound.
    If the rank lands in the ``+Inf`` bucket, return the largest finite bound — the histogram
    cannot say more. No observations -> NaN."""
    if math.isnan(q):
        return math.nan
    if q < 0:
        return -math.inf
    if q > 1:
        return math.inf
    b = sorted((float(le), float(c)) for le, c in buckets)
    if not b or not math.isinf(b[-1][0]):
        return math.nan
    merged: list = []  # coalesce duplicate bounds
    for le, c in b:
        if merged and merged[-1][0] == le:
            merged[-1] = (le, merged[-1][1] + c)
        else:
            merged.append((le, c))
    fixed, prev = [], merged[0][1]  # force monotonic cumulative counts (tiny float noise, resets)
    for i, (le, c) in enumerate(merged):
        if i and (c < prev or math.isclose(c, prev, rel_tol=1e-12)):
            c = prev
        fixed.append((le, c))
        prev = c
    if len(fixed) < 2:
        return math.nan
    total = fixed[-1][1]
    if total == 0:
        return math.nan
    rank = q * total
    last = len(fixed) - 1  # the +Inf bucket is never searched, only fallen into
    idx = next((i for i in range(last) if fixed[i][1] >= rank), last)
    if idx == last:
        return fixed[-2][0]
    if idx == 0 and fixed[0][0] <= 0:
        return fixed[0][0]
    start, end, count = 0.0, fixed[idx][0], fixed[idx][1]
    if idx > 0:
        start = fixed[idx - 1][0]
        count -= fixed[idx - 1][1]
        rank -= fixed[idx - 1][1]
    if count == 0:  # only q == 0 with an empty first bucket: Go's 0/0 is NaN
        return math.nan
    return start + (end - start) * (rank / count)


# ---------------------------------------------------------------------------------------------
# The engine at a glance
# ---------------------------------------------------------------------------------------------
@dataclass
class EngineSnapshot:
    running: float
    waiting: float
    kv_cache_usage: float
    prefix_cache_queries: float
    prefix_cache_hits: float
    prefix_hit_rate: float
    preemptions: float
    prompt_tokens: float
    generation_tokens: float
    requests_finished: float
    ttft_mean: float           # exact: _sum / _count
    ttft_p50: float            # interpolated inside a bucket
    ttft_p99: float
    itl_mean: float
    itl_p50: float
    itl_p99: float
    e2e_mean: float
    e2e_p50: float
    queue_mean: float
    queue_p50: float
    queue_p99: float
    window_s: float | None = None
    generation_tps: float | None = None
    spec_acceptance_rate: float | None = None
    spec_mean_acceptance_length: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def table(self) -> str:
        def f(v, unit=""):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return "n/a"
            if unit == "ms":
                return f"{v * 1000:.1f} ms"
            if unit == "%":
                return f"{v:.1%}"
            return f"{v:,.0f}" if unit == "n" else f"{v:,.2f}"
        rows = [("requests running / waiting", f"{f(self.running, 'n')} / {f(self.waiting, 'n')}"),
                ("KV cache usage", f(self.kv_cache_usage, "%")),
                ("prefix cache hit rate", f"{f(self.prefix_hit_rate, '%')}  ({f(self.prefix_cache_hits, 'n')} of "
                                          f"{f(self.prefix_cache_queries, 'n')} tokens)"),
                ("preemptions", f(self.preemptions, "n")),
                ("requests finished", f(self.requests_finished, "n")),
                ("TTFT mean | p50 / p99", f"{f(self.ttft_mean, 'ms')} | {f(self.ttft_p50, 'ms')} / {f(self.ttft_p99, 'ms')}"),
                ("ITL  mean | p50 / p99", f"{f(self.itl_mean, 'ms')} | {f(self.itl_p50, 'ms')} / {f(self.itl_p99, 'ms')}"),
                ("queue mean | p50 / p99", f"{f(self.queue_mean, 'ms')} | {f(self.queue_p50, 'ms')} / {f(self.queue_p99, 'ms')}"),
                ("E2E  mean | p50", f"{f(self.e2e_mean, 'ms')} | {f(self.e2e_p50, 'ms')}")]
        if self.generation_tps is not None:
            rows.append(("generation tokens/s", f(self.generation_tps, "n")))
        if self.spec_acceptance_rate is not None:
            rows.append(("spec acceptance rate / mean length",
                         f"{f(self.spec_acceptance_rate, '%')} / {f(self.spec_mean_acceptance_length)}"))
        w = max(len(k) for k, _ in rows)
        head = f"window {self.window_s:.1f} s" if self.window_s else "since server start"
        return "\n".join([f"engine /metrics ({head}; means are exact, percentiles are bucket interpolations)"]
                         + [f"  {k:<{w}}  {v}" for k, v in rows])


def snapshot(later: Scrape, earlier: Scrape | None = None) -> EngineSnapshot:
    """Summarize the engine from one scrape (since start) or two (the window between them)."""
    s = delta(later, earlier) if earlier is not None else later

    def first(*names, default=math.nan):
        for n in names:
            v = s.value(n)
            if not math.isnan(v):
                return v
        return default

    q = first(PREFIX_QUERIES, PREFIX_QUERIES_OLD, default=0.0)
    h = first(PREFIX_HITS, PREFIX_HITS_OLD, default=0.0)
    ttft, itl, e2e, queue = (s.histogram(n) for n in (TTFT, ITL, E2E, QUEUE))
    window = (later.timestamp - earlier.timestamp) if earlier is not None else None
    gen = first(GENERATION_TOKENS, default=0.0)
    drafts, draft_toks, acc = first(SPEC_DRAFTS), first(SPEC_DRAFT_TOKENS), first(SPEC_ACCEPTED)
    spec_rate = acc / draft_toks if draft_toks and not math.isnan(draft_toks) and draft_toks > 0 else None
    spec_len = 1 + acc / drafts if drafts and not math.isnan(drafts) and drafts > 0 else None
    return EngineSnapshot(
        running=first(RUNNING), waiting=first(WAITING), kv_cache_usage=first(KV_USAGE, KV_USAGE_OLD),
        prefix_cache_queries=q, prefix_cache_hits=h, prefix_hit_rate=(h / q) if q else math.nan,
        preemptions=first(PREEMPTIONS, default=0.0), prompt_tokens=first(PROMPT_TOKENS, default=0.0),
        generation_tokens=gen, requests_finished=first(REQUEST_SUCCESS, default=0.0),
        ttft_mean=ttft.mean, ttft_p50=ttft.quantile(0.5), ttft_p99=ttft.quantile(0.99), itl_mean=itl.mean,
        itl_p50=itl.quantile(0.5), itl_p99=itl.quantile(0.99), e2e_mean=e2e.mean, e2e_p50=e2e.quantile(0.5),
        queue_mean=queue.mean, queue_p50=queue.quantile(0.5), queue_p99=queue.quantile(0.99), window_s=window,
        generation_tps=(gen / window) if window else None,
        spec_acceptance_rate=spec_rate, spec_mean_acceptance_length=spec_len,
    )
