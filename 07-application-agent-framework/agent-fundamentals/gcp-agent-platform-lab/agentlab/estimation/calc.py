"""Back-of-the-envelope arithmetic for GenAI systems (notebook 12).

A design review wants three things from an estimate: the *shape* (what drives cost, what
drives latency), the *order of magnitude*, and the *levers*. Everything here is deliberately
simple arithmetic wrapped in names, so the numbers you quote are reproducible.

Prices are illustrative, dated September 2026, and must be verified before you rely on them (verify).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, NamedTuple

SECONDS_PER_DAY = 86_400
BATCH_DISCOUNT = 0.5           # batch/offline endpoints typically bill at half the online rate
PRICE_DISCLAIMER = "Illustrative USD per million tokens — verify against the current price list before quoting."


# ------------------------------------------------------------------- prices
@dataclass(frozen=True)
class Price:
    """USD per million tokens. ``long_context`` is the tier billed above ``long_context_threshold`` input tokens."""

    input: float
    cached_input: float
    output: float
    long_context: "Price | None" = None
    long_context_threshold: int = 200_000

    def tier_for(self, in_tokens: float) -> "Price":
        if self.long_context is not None and in_tokens > self.long_context_threshold:
            return self.long_context
        return self


PRICES: dict[str, Price] = {   # verify: illustrative list prices
    "gemini-3.1-pro": Price(2.00, 0.20, 12.00, long_context=Price(4.00, 0.40, 18.00)),
    "gemini-3.6-flash": Price(1.50, 0.15, 7.50),
    "gemini-3-flash": Price(0.50, 0.05, 3.00),
    "gemini-3.5-flash-lite": Price(0.30, 0.03, 2.50),
}


def token_cost(in_tokens: float, out_tokens: float, price: Price, cached_share: float = 0.0, batch: bool = False) -> float:
    """USD for one call: fresh input + cached input (at the cached rate) + output, halved for batch."""
    if not 0.0 <= cached_share <= 1.0:
        raise ValueError("cached_share must be between 0 and 1")
    tier = price.tier_for(in_tokens)
    cached = in_tokens * cached_share
    usd = ((in_tokens - cached) * tier.input + cached * tier.cached_input + out_tokens * tier.output) / 1e6
    return usd * (BATCH_DISCOUNT if batch else 1.0)


# ----------------------------------------------------------------- scenarios
def littles_law(arrival_rate: float, duration_s: float) -> float:
    """L = λW: items in flight = arrival rate × time each spends in the system."""
    return arrival_rate * duration_s


@dataclass
class Scenario:
    """A workload described by the few numbers that drive cost and capacity.

    ``units`` are whatever you bill by (conversations, documents, tickets); each unit makes
    ``calls_per_unit`` model calls of ``in_tokens``/``out_tokens``. ``model_mix`` splits calls
    across models by share; ``peak_factor`` turns the daily average into the rate you must
    provision for.
    """

    name: str
    units_per_day: float
    calls_per_unit: float
    in_tokens: float
    out_tokens: float
    peak_factor: float = 3.0
    model_mix: dict[str, float] = field(default_factory=lambda: {"gemini-3.1-pro": 1.0})
    cached_share: float = 0.0
    batch: bool = False
    seconds_per_call: float = 4.0
    prices: Mapping[str, Price] = field(default_factory=lambda: PRICES)

    def __post_init__(self) -> None:
        total_share = sum(self.model_mix.values())
        if abs(total_share - 1.0) > 1e-6:
            raise ValueError(f"model_mix shares must sum to 1, got {total_share}")
        missing = [m for m in self.model_mix if m not in self.prices]
        if missing:
            raise KeyError(f"no price for {missing}")

    # -- volume ---------------------------------------------------------------
    @property
    def calls_per_day(self) -> float:
        return self.units_per_day * self.calls_per_unit

    def cost_per_call(self) -> float:
        return sum(share * token_cost(self.in_tokens, self.out_tokens, self.prices[model], self.cached_share, self.batch)
                   for model, share in self.model_mix.items())

    # -- cost -----------------------------------------------------------------
    def daily_cost(self) -> float:
        return self.cost_per_call() * self.calls_per_day

    def cost_per_unit(self) -> float:
        return self.cost_per_call() * self.calls_per_unit

    def annual_cost(self) -> float:
        return self.daily_cost() * 365

    # -- capacity -------------------------------------------------------------
    def avg_calls_per_sec(self) -> float:
        return self.calls_per_day / SECONDS_PER_DAY

    def peak_calls_per_sec(self) -> float:
        return self.avg_calls_per_sec() * self.peak_factor

    def peak_input_tpm(self) -> float:
        """Input tokens per minute at peak — the number to compare with the model quota."""
        return self.peak_calls_per_sec() * self.in_tokens * 60

    def peak_output_tps(self) -> float:
        return self.peak_calls_per_sec() * self.out_tokens

    def concurrency(self) -> float:
        """Calls in flight at peak (Little's law) — sizes connection pools and worker counts."""
        return littles_law(self.peak_calls_per_sec(), self.seconds_per_call)

    # -- presentation ---------------------------------------------------------
    def report(self) -> str:
        mix = ", ".join(f"{m} {s:.0%}" for m, s in self.model_mix.items())
        rows = [
            ("units/day", f"{self.units_per_day:,.0f}", "calls/unit", f"{self.calls_per_unit:g}", "calls/day", f"{self.calls_per_day:,.0f}"),
            ("tokens/call", f"{self.in_tokens:,.0f} in / {self.out_tokens:,.0f} out", "cached share", f"{self.cached_share:.0%}", "batch", "yes" if self.batch else "no"),
            ("model mix", mix, "", "", "", ""),
            ("cost/day", f"${self.daily_cost():,.2f}", "cost/unit", f"${self.cost_per_unit():,.4f}", "cost/year", f"${self.annual_cost():,.0f}"),
            ("calls/s avg", f"{self.avg_calls_per_sec():,.2f}", f"peak (x{self.peak_factor:g})", f"{self.peak_calls_per_sec():,.2f}", "in flight", f"{self.concurrency():,.1f} @ {self.seconds_per_call:g}s/call"),
            ("peak input TPM", f"{self.peak_input_tpm():,.0f}", "peak output TPS", f"{self.peak_output_tps():,.0f}", "", ""),
        ]
        lines = [f"Scenario: {self.name}"]
        for label, value, label2, value2, label3, value3 in rows:
            line = f"  {label:<16}{value:<28}"
            if label2:
                line += f"{label2:<16}{value2:<16}"
            if label3:
                line += f"{label3:<12}{value3}"
            lines.append(line.rstrip())
        return "\n".join(lines)


# ------------------------------------------------------------------- latency
@dataclass(frozen=True)
class Segment:
    """One hop of a turn. Segments sharing a ``parallel_group`` (and adjacent) run concurrently."""

    name: str
    seconds: float
    parallel_group: str | None = None


@dataclass(frozen=True)
class Span:
    name: str
    start: float
    end: float


@dataclass(frozen=True)
class LatencyEstimate:
    total_s: float
    first_token_s: float
    p95_s: float
    spans: tuple[Span, ...]


def schedule(segments: list[Segment]) -> list[Span]:
    """Place segments on a timeline: sequential by default, side by side inside a parallel group."""
    spans: list[Span] = []
    cursor = 0.0            # when the next sequential segment may start
    group: str | None = None
    group_start = 0.0
    for seg in segments:
        if seg.parallel_group is not None and seg.parallel_group == group:
            start = group_start
        else:
            start = cursor
            group, group_start = seg.parallel_group, start
        end = start + seg.seconds
        cursor = max(cursor, end)
        spans.append(Span(seg.name, start, end))
    return spans


def latency_budget(segments: list[Segment], first_token_segment: str | None = None, first_token_offset_s: float = 0.0, p95_factor: float = 1.5) -> LatencyEstimate:
    """Total time, time to first visible token, and a p95 estimate.

    The first token appears when ``first_token_segment`` starts plus its own time to first
    token (``first_token_offset_s``); without streaming it is the total. ``p95_factor`` is a
    crude inflation for tail latency — replace it with measured percentiles as soon as you have them.
    """
    spans = schedule(segments)
    total = max((s.end for s in spans), default=0.0)
    first_token = total
    if first_token_segment is not None:
        span = next((s for s in spans if s.name == first_token_segment), None)
        if span is None:
            raise KeyError(f"no segment named {first_token_segment!r}")
        first_token = span.start + first_token_offset_s
    return LatencyEstimate(total_s=total, first_token_s=first_token, p95_s=total * p95_factor, spans=tuple(spans))


def waterfall_text(segments: list[Segment], width: int = 40, first_token_segment: str | None = None, first_token_offset_s: float = 0.0) -> str:
    """ASCII waterfall: one bar per segment on a shared time axis."""
    est = latency_budget(segments, first_token_segment, first_token_offset_s)
    scale = width / est.total_s if est.total_s > 0 else 0.0
    name_w = max(len(s.name) for s in est.spans) if est.spans else 4
    lines = []
    for span in est.spans:
        lead = int(round(span.start * scale))
        bar = max(1, int(round((span.end - span.start) * scale)))
        marker = "  ◀ first token" if span.name == first_token_segment else ""
        lines.append(f"{span.name:<{name_w}} |{' ' * lead}{'█' * bar}{' ' * max(0, width - lead - bar)}| {span.start:5.2f}–{span.end:5.2f}s{marker}")
    lines.append(f"{'':<{name_w}}  total {est.total_s:.2f}s · first token {est.first_token_s:.2f}s · p95≈{est.p95_s:.2f}s")
    return "\n".join(lines)


# ------------------------------------------------------- capacity & quality
class Throughput(NamedTuple):
    tokens_per_s: float
    tokens_per_min: float


def throughput_for_backlog(total_tokens: float, days: float) -> Throughput:
    """Sustained rate needed to clear a backlog in ``days`` — compare with the TPM quota."""
    per_s = total_tokens / (days * SECONDS_PER_DAY)
    return Throughput(per_s, per_s * 60)


def vector_store_bytes(chunks: int, dims: int, bytes_per_dim: int = 4, index_overhead: float = 1.5) -> float:
    """Raw vector bytes × index overhead (graph links, metadata, ids: roughly 1.5× for HNSW)."""
    return chunks * dims * bytes_per_dim * index_overhead


def human_bytes(n: float) -> str:
    """Decimal units, the way storage is priced (1 GB = 1e9 B)."""
    for unit, size in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n:.0f} B"


def ci_half_width(n: int, p: float = 0.5, z: float = 1.96) -> float:
    """Half-width of a normal-approximation confidence interval for a pass rate measured on ``n`` cases.

    p = 0.5 is the worst case; with n = 100 the interval is ±9.8 points, which is why a
    "2-point improvement" on a 100-case eval set is noise.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    return z * math.sqrt(p * (1 - p) / n)


def compounded_reliability(p: float, hops: int) -> float:
    """Success probability of ``hops`` independent steps that each succeed with probability ``p``.

    0.99 per step over 10 steps is 0.904: a "99% reliable" agent step is a 90% reliable agent.
    """
    return p ** hops
