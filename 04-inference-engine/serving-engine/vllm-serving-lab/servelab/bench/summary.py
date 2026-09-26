"""summary.py — turn per-request timings into the numbers a design review asks for.

One idea: every serving number is a ratio with a definition, and two benchmarks only compare if
the definitions match. These are ``vllm bench serve``'s (``vllm/benchmarks/serve.py``):

    request throughput = completed requests / duration
    output throughput  = sum(output tokens) / duration
    total throughput   = sum(input + output tokens) / duration
    TTFT, TPOT, E2E    = statistics over *requests*;  ITL = statistics over *gaps* (all requests pooled)
    goodput            = requests meeting every SLO (TTFT, TPOT, E2E) / duration

``duration`` is the wall time of the whole run, from the first send to the last byte, so a slow
tail lowers throughput. Percentiles use linear interpolation between order statistics (numpy's
default). Goodput is the number to optimize: throughput that users would accept.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


def percentile(values, p: float) -> float:
    """The p-th percentile (0-100) with linear interpolation: numpy.percentile's default method."""
    xs = sorted(values)
    if not xs:
        return math.nan
    h = (len(xs) - 1) * p / 100.0
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


@dataclass
class SLO:
    """Per-request latency targets in milliseconds; ``None`` = not constrained."""
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2el_ms: float | None = None

    def ok(self, r) -> bool:
        """vLLM's rule: good when every set target is >= the request's value; TPOT counts as 0 for
        single-token outputs."""
        if not r.ok:
            return False
        tpot = r.tpot if r.output_tokens > 1 else 0.0
        checks = [(self.ttft_ms, r.ttft), (self.tpot_ms, tpot), (self.e2el_ms, r.latency)]
        return all(target / 1000.0 >= value for target, value in checks if target is not None)

    def __str__(self) -> str:
        parts = [f"{k}<={v:g}ms" for k, v in (("TTFT", self.ttft_ms), ("TPOT", self.tpot_ms), ("E2E", self.e2el_ms))
                 if v is not None]
        return " & ".join(parts) or "none"


@dataclass
class Stat:
    n: int
    mean: float
    median: float
    std: float
    p: dict = field(default_factory=dict)     # {percentile: value}, all in milliseconds

    @classmethod
    def of(cls, seconds: list, percentiles=(50, 90, 99)) -> "Stat":
        ms = [s * 1000.0 for s in seconds if not math.isnan(s)]
        if not ms:
            return cls(0, math.nan, math.nan, math.nan, {q: math.nan for q in percentiles})
        mean = sum(ms) / len(ms)
        std = math.sqrt(sum((x - mean) ** 2 for x in ms) / len(ms))
        return cls(len(ms), mean, percentile(ms, 50), std, {q: percentile(ms, q) for q in percentiles})


@dataclass
class Summary:
    completed: int
    failed: int
    duration_s: float
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    goodput: float
    slo_attainment: float           # good / all requests sent (failed count as bad)
    ttft: Stat
    tpot: Stat
    itl: Stat
    e2el: Stat
    slo: str = "none"
    cached_fraction: float | None = None    # sum(cached_tokens) / sum(prompt_tokens), when reported
    label: str = ""

    def row(self, percentile_: int = 99) -> dict:
        """A flat dict for comparison tables."""
        return {"label": self.label, "req/s": self.request_throughput, "out tok/s": self.output_throughput,
                "goodput": self.goodput, "SLO met": self.slo_attainment,
                "TTFT p50": self.ttft.median, f"TTFT p{percentile_}": self.ttft.p.get(percentile_, math.nan),
                "TPOT p50": self.tpot.median, f"TPOT p{percentile_}": self.tpot.p.get(percentile_, math.nan),
                f"ITL p{percentile_}": self.itl.p.get(percentile_, math.nan), "E2E p50": self.e2el.median}

    def as_dict(self) -> dict:
        return asdict(self)

    def text(self) -> str:
        """A report block in the layout of ``vllm bench serve``."""
        lines = [f"{'=' * 14} Serving Benchmark Result {self.label} {'=' * 14}",
                 f"Successful requests:               {self.completed}",
                 f"Failed requests:                   {self.failed}",
                 f"Benchmark duration (s):            {self.duration_s:.2f}",
                 f"Total input tokens:                {self.total_input}",
                 f"Total generated tokens:            {self.total_output}",
                 f"Request throughput (req/s):        {self.request_throughput:.2f}",
                 f"Request goodput (req/s):           {self.goodput:.2f}   (SLO: {self.slo}; met by {self.slo_attainment:.0%})",
                 f"Output token throughput (tok/s):   {self.output_throughput:.2f}",
                 f"Total token throughput (tok/s):    {self.total_token_throughput:.2f}"]
        if self.cached_fraction is not None:
            lines.append(f"Prompt tokens served from cache:   {self.cached_fraction:.1%}")
        for name, st in (("Time to First Token", self.ttft), ("Time per Output Token (excl. 1st token)", self.tpot),
                         ("Inter-token Latency", self.itl), ("End-to-end Latency", self.e2el)):
            lines.append(f"{'-' * 10} {name} {'-' * 10}")
            lines.append(f"Mean / Median (ms):                {st.mean:.2f} / {st.median:.2f}")
            lines.append("  ".join(f"P{q} (ms): {v:.2f}" for q, v in st.p.items()))
        return "\n".join(lines)


def summarize(results: list, duration_s: float, slo: SLO | None = None, percentiles=(50, 90, 99),
              label: str = "") -> Summary:
    ok = [r for r in results if r.ok]
    total_in = sum(r.prompt_tokens for r in ok)
    total_out = sum(r.output_tokens for r in ok)
    good = sum(1 for r in ok if slo is None or slo.ok(r))
    itls = [g for r in ok for g in r.itl]
    cached = [r for r in ok if r.cached_tokens is not None]
    d = duration_s if duration_s > 0 else math.nan
    return Summary(
        completed=len(ok), failed=len(results) - len(ok), duration_s=duration_s, total_input=total_in,
        total_output=total_out, request_throughput=len(ok) / d, output_throughput=total_out / d,
        total_token_throughput=(total_in + total_out) / d, goodput=good / d,
        slo_attainment=good / len(results) if results else math.nan,
        ttft=Stat.of([r.ttft for r in ok], percentiles), tpot=Stat.of([r.tpot for r in ok], percentiles),
        itl=Stat.of(itls, percentiles), e2el=Stat.of([r.latency for r in ok], percentiles),
        slo=str(slo) if slo else "none",
        cached_fraction=(sum(r.cached_tokens for r in cached) / max(1, sum(r.prompt_tokens for r in cached)))
        if cached else None,
        label=label,
    )


def littles_law(arrival_rate: float, mean_latency_s: float) -> float:
    """Requests in the system = arrival rate × time each spends there (L = λW)."""
    return arrival_rate * mean_latency_s
