"""Metrics derived from traces (Primer §4.2, §5.3).

Everything here is arithmetic over ``Span`` objects and streamed chunks:

* latency percentiles by span kind (nearest-rank, no interpolation);
* dollars per trace from a ``PriceTable`` (input / cached input / output);
* time-to-first-token and tokens per second from a stream;
* tokens and cost per task, cost per *resolved* task;
* alert rules over those numbers, so a looping agent pages someone before
  the invoice does.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, NamedTuple, Sequence

from ..llm.types import StreamChunk, Usage
from .tracing import GEN_AI_REQUEST_MODEL, TOOL_OK, Span, Trace, Tracer, span_usage

Latency = Callable[[Span], float | None]


# ----------------------------------------------------------------- percentiles
def percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value below which ``p``% of the sample lies.

    No interpolation, so the result is always an observed value — the right choice
    for latency, where an interpolated "p95" may be a number nobody experienced.
    """
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0, 100], got {p}")
    data = sorted(values)
    if not data:
        raise ValueError("percentile of an empty sample")
    rank = max(1, math.ceil(p / 100 * len(data)))
    return data[rank - 1]


def describe(values: Iterable[float]) -> dict[str, float]:
    data = list(values)
    if not data:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {"count": len(data), "mean": sum(data) / len(data), "p50": percentile(data, 50),
            "p95": percentile(data, 95), "p99": percentile(data, 99), "max": max(data)}


def reported_latency_ms(span: Span) -> float | None:
    """The latency a span *reports* rather than the wall time it took.

    ``FakeLLM`` does not sleep by default; it reports the latency it would have
    had in ``model.latency_ms`` (tools likewise in ``tool.latency_ms``). In
    production the span duration is the truth and this helper is unnecessary.
    """
    for key in ("model.latency_ms", "tool.latency_ms"):
        if key in span.attrs and span.attrs[key] is not None:
            return float(span.attrs[key])
    return span.duration_ms


def _durations(spans: Iterable[Span], latency: Latency | None) -> list[float]:
    fn = latency or (lambda s: s.duration_ms)
    return [d for d in (fn(s) for s in spans if s.finished) if d is not None]


def summarize_latencies(spans: Iterable[Span], latency: Latency | None = None) -> dict[str, dict[str, float]]:
    """Per span kind: count, mean, p50, p95, p99, max (milliseconds)."""
    by_kind: dict[str, list[Span]] = {}
    for s in spans:
        by_kind.setdefault(s.kind, []).append(s)
    return {kind: describe(_durations(group, latency)) for kind, group in by_kind.items()}


# ------------------------------------------------------------------ streaming
class StreamStats(NamedTuple):
    ttft_ms: float
    tokens_per_sec: float
    output_tokens: int
    generation_ms: float


def now_ms() -> float:
    """Same clock as ``StreamChunk.t_ms`` — call it right before ``llm.stream(...)``."""
    return time.perf_counter() * 1000.0


def ttft_and_tps(chunks: Sequence[StreamChunk], start_ms: float) -> StreamStats:
    """Time-to-first-token and output tokens/second from streamed chunks.

    TTFT is the gap from the request (``start_ms``) to the first chunk carrying
    content (text or a tool call). Throughput is the output tokens reported in
    the final chunk's usage divided by the time from that first content chunk to
    the last chunk — i.e. the generation phase only, which is what users feel as
    "typing speed" after the initial wait.
    """
    content = [c for c in chunks if c.text or c.tool_call is not None]
    if not content:
        raise ValueError("stream produced no content chunks")
    first, last = content[0], chunks[-1]
    usage = next((c.usage for c in reversed(chunks) if c.usage is not None), None)
    output_tokens = usage.output_tokens if usage else 0
    generation_ms = max(0.0, last.t_ms - first.t_ms)
    tps = output_tokens / (generation_ms / 1000.0) if generation_ms > 0 else 0.0
    return StreamStats(ttft_ms=first.t_ms - start_ms, tokens_per_sec=tps,
                       output_tokens=output_tokens, generation_ms=generation_ms)


# --------------------------------------------------------------------- pricing
@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    cached_input: float
    output: float


class PriceTable:
    """Per-model prices plus aliases (``fake-flash`` → ``gemini-3-flash``).

    ``cost`` bills uncached input, cached input and output separately (Primer
    §5.3 A); thinking tokens are billed as output, as Gemini does.
    """

    def __init__(self, prices: dict[str, Price], aliases: dict[str, str] | None = None, note: str = ""):
        self.prices = dict(prices)
        self.aliases = dict(aliases or {})
        self.note = note

    def __contains__(self, model: object) -> bool:
        return model in self.prices or model in self.aliases

    def price_for(self, model: str) -> Price:
        key = self.aliases.get(model, model)
        if key not in self.prices:
            raise KeyError(f"no price for model {model!r}; known: {sorted(self.prices)} (+ aliases {sorted(self.aliases)})")
        return self.prices[key]

    def breakdown(self, usage: Usage, model: str) -> dict[str, float]:
        p = self.price_for(model)
        uncached = max(0, usage.input_tokens - usage.cached_tokens)
        return {
            "input": uncached * p.input / 1e6,
            "cached_input": usage.cached_tokens * p.cached_input / 1e6,
            "output": (usage.output_tokens + usage.thinking_tokens) * p.output / 1e6,
        }

    def cost(self, usage: Usage, model: str) -> float:
        return sum(self.breakdown(usage, model).values())


DEFAULT_PRICES = PriceTable(
    {
        "gemini-3.1-pro": Price(input=2.00, cached_input=0.20, output=12.00),
        "gemini-3-flash": Price(input=0.50, cached_input=0.05, output=3.00),
        "gemini-3.5-flash-lite": Price(input=0.30, cached_input=0.03, output=2.50),
    },
    aliases={"fake-flash": "gemini-3-flash", "fake-pro": "gemini-3.1-pro", "fake": "gemini-3-flash"},
    note="illustrative, verify against the official pricing page",
)


# ------------------------------------------------------------- trace summaries
@dataclass
class TraceSummary:
    """One conversation turn reduced to the numbers an operator watches."""

    trace_id: str
    name: str
    attrs: dict[str, Any]
    steps: int                      # model calls
    tool_calls: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    cost_usd: float
    wall_ms: float
    errors: int                     # spans that raised or tools that reported failure
    latency_by_kind: dict[str, dict[str, float]]
    longest_span: tuple[str, float] | None
    step_costs: list[tuple[str, float]] = field(default_factory=list)   # (label, usd) per model span

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def top_costs(self, n: int = 3) -> list[tuple[str, float]]:
        """The most expensive model calls — the first place to look when a task costs too much."""
        return sorted(self.step_costs, key=lambda item: item[1], reverse=True)[:n]

    @classmethod
    def from_trace(cls, trace: Trace, price_table: PriceTable = DEFAULT_PRICES, latency: Latency | None = None) -> "TraceSummary":
        usage, step_costs = Usage(), []
        for i, s in enumerate(trace.by_kind("model"), 1):
            u = span_usage(s)
            usage = usage + u
            step_costs.append((f"{s.name} #{i}", price_table.cost(u, s.attrs.get(GEN_AI_REQUEST_MODEL, "?"))))
        finished = [(s.name, s.duration_ms) for s in trace.spans if s.finished and s.parent_id is not None]
        errors = sum(1 for s in trace.spans if s.status == "error" or s.attrs.get(TOOL_OK) is False)
        return cls(
            trace_id=trace.trace_id, name=trace.name, attrs=dict(trace.root.attrs),
            steps=len(trace.by_kind("model")), tool_calls=len(trace.by_kind("tool")),
            input_tokens=usage.input_tokens, cached_tokens=usage.cached_tokens, output_tokens=usage.output_tokens,
            cost_usd=sum(c for _, c in step_costs), wall_ms=trace.wall_ms, errors=errors,
            latency_by_kind=summarize_latencies(trace.spans, latency),
            longest_span=max(finished, key=lambda item: item[1]) if finished else None,
            step_costs=step_costs,
        )

    @classmethod
    def from_tracer(cls, tracer: Tracer, price_table: PriceTable = DEFAULT_PRICES, latency: Latency | None = None) -> list["TraceSummary"]:
        return [cls.from_trace(t, price_table, latency) for t in tracer.traces()]

    def as_row(self) -> str:
        return (f"{self.name:32s} steps={self.steps} tools={self.tool_calls} "
                f"in={self.input_tokens} (cached {self.cached_tokens}) out={self.output_tokens} "
                f"${self.cost_usd:.6f} {self.wall_ms:.1f} ms")


# ---------------------------------------------------------------- task metrics
def tokens_per_task(summaries: Iterable[TraceSummary]) -> dict[str, float]:
    return describe(s.total_tokens for s in summaries)


def cost_per_resolved(summaries: Iterable[TraceSummary], resolved: Callable[[TraceSummary], bool]) -> float:
    """Total spend divided by the tasks actually resolved — the unit economics number (Primer §5.3).

    Cost per *conversation* flatters an agent that gives up cheaply; cost per
    *resolution* charges every failed attempt to the successes.
    """
    items = list(summaries)
    n_resolved = sum(1 for s in items if resolved(s))
    if n_resolved == 0:
        return math.inf
    return sum(s.cost_usd for s in items) / n_resolved


WRONG_TOOL_ERRORS = ("unknown_tool", "invalid_arguments", "duplicate_call")


def wrong_tool_rate(sessions: Iterable[Any]) -> float:
    """Share of tool calls the model got wrong: unknown tool, bad arguments, or a repeat.

    Computed from session event logs, because the loop rejects unknown and
    duplicate calls before a tool span is ever opened.
    """
    calls = wrong = 0
    for session in sessions:
        for ev in session.events:
            if ev.kind == "tool_call":
                calls += 1
            elif ev.kind == "tool_result" and ev.payload.get("error") in WRONG_TOOL_ERRORS:
                wrong += 1
    return wrong / calls if calls else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def agent_metrics(tracer: Tracer, sessions: Iterable[Any] = (), price_table: PriceTable = DEFAULT_PRICES,
                  latency: Latency | None = None) -> dict[str, float]:
    """The per-task numbers alert rules are written against (one task = one trace)."""
    summaries = TraceSummary.from_tracer(tracer, price_table, latency)
    latencies = summarize_latencies(tracer.spans, latency)
    return {
        "tasks": len(summaries),
        "cost_per_task": _mean([s.cost_usd for s in summaries]),
        "tokens_per_task": _mean([s.total_tokens for s in summaries]),
        "steps_per_task": _mean([s.steps for s in summaries]),
        "tool_calls_per_task": _mean([s.tool_calls for s in summaries]),
        "error_rate": _mean([1.0 if s.errors else 0.0 for s in summaries]),
        "p95_model_latency_ms": latencies.get("model", {}).get("p95", 0.0),
        "p95_tool_latency_ms": latencies.get("tool", {}).get("p95", 0.0),
        "wrong_tool_rate": wrong_tool_rate(sessions),
    }


# ---------------------------------------------------------------------- alerts
_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
}


@dataclass(frozen=True)
class AlertRule:
    """Fire when ``metric <comparator> threshold`` (e.g. ``cost_per_task > 0.01``)."""

    name: str
    metric: str
    threshold: float
    comparator: str = ">"
    severity: str = "page"          # "page" | "ticket"

    def fires(self, observed: float) -> bool:
        return _COMPARATORS[self.comparator](observed, self.threshold)


@dataclass
class Alert:
    rule: AlertRule
    observed: float

    def __str__(self) -> str:
        return f"[{self.rule.severity}] {self.rule.name}: {self.rule.metric}={self.observed:.4g} {self.rule.comparator} {self.rule.threshold:g}"


def evaluate_alerts(rules: Iterable[AlertRule], metrics: dict[str, float]) -> list[Alert]:
    """Return the alerts that fire. A rule naming a metric that is not reported is a config bug, so it raises."""
    fired: list[Alert] = []
    for r in rules:
        if r.metric not in metrics:
            raise KeyError(f"alert {r.name!r} references unknown metric {r.metric!r}; reported: {sorted(metrics)}")
        if r.fires(metrics[r.metric]):
            fired.append(Alert(rule=r, observed=metrics[r.metric]))
    return fired
