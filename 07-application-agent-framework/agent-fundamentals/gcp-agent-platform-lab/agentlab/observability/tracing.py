"""Tracing for agent runs (notebook 09; the capstone's traces in notebook 14).

A *span* is one unit of work — an agent turn, a model call, a tool call, a
retrieval — with start/end times, a kind, and attributes. Spans nest: the tool
spans of a step hang off the agent span that requested them, and the agent
span is the root of the *trace* for that conversation turn. Nesting is tracked
with a ``contextvars.ContextVar`` so spans opened inside ``asyncio.gather``
tasks (parallel tool calls, parallel branches) still find the right parent.

Attribute names follow the OpenTelemetry GenAI semantic conventions as of
September 2026 (verify): ``gen_ai.operation.name``, ``gen_ai.request.model``,
``gen_ai.usage.input_tokens`` / ``output_tokens`` / ``cache_read.input_tokens``,
``gen_ai.response.finish_reasons`` (an array) and, on tool spans,
``gen_ai.tool.name`` / ``gen_ai.tool.call.id``. The conventions are still at
*Development* stability and now live in their own repository,
open-telemetry/semantic-conventions-genai, so names can change between
releases; a backend that implements the same version reads these attributes
without a mapping. Three things here are not the convention: the ``tool.ok`` /
``tool.latency_ms`` / ``tool.error`` attributes are this lab's own; the fake
model has no provider, so the ``gen_ai.provider.name`` a real instrumentation
must set is left to the adapter; and message content goes in the opt-in
``gen_ai.input.messages`` / ``gen_ai.output.messages`` (the old ``gen_ai.prompt``
is deprecated), which the lab never records by default.

Traces carry personal data: prompts, tool arguments, retrieved documents. The
``RedactingExporter`` scrubs values field by field *before* anything leaves the
process, which is the only place redaction can be enforced reliably.

Contract expected by the agent loop::

    with tracer.span(name, kind="model", **attrs) as span:
        ...
        span.set(**more_attrs)
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterator

from ..llm.types import Usage

# --------------------------------------------------------------- attribute names
# OpenTelemetry GenAI semantic conventions (the ones the loop and the metrics use).
# Checked against open-telemetry/semantic-conventions-genai, September 2026 (verify).
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_INPUT_TOKENS = "gen_ai.usage.input_tokens"               # includes the cached ones
GEN_AI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_CACHED_TOKENS = "gen_ai.usage.cache_read.input_tokens"  # the cached subset of input_tokens
GEN_AI_FINISH_REASONS = "gen_ai.response.finish_reasons"       # string[]: one reason per choice
GEN_AI_FINISH_REASON = GEN_AI_FINISH_REASONS                   # old name of the constant, kept as an alias
GEN_AI_INPUT_MESSAGES = "gen_ai.input.messages"                # opt-in: raw prompt content (never exported here)
GEN_AI_OUTPUT_MESSAGES = "gen_ai.output.messages"              # opt-in: raw completion content
# Tool attributes (the loop sets these on every tool span). The first two are the
# convention's; ok / latency / error are this lab's own, outside the gen_ai namespace.
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOOL_OK = "tool.ok"
TOOL_LATENCY_MS = "tool.latency_ms"
TOOL_ERROR = "tool.error"

# Attribute names older code in this lab wrote before it followed the convention.
# Read-only fallbacks, so spans written that way still price and render correctly.
_LEGACY_ATTRS = {GEN_AI_CACHED_TOKENS: "gen_ai.usage.cached_tokens",
                 GEN_AI_FINISH_REASONS: "gen_ai.response.finish_reason"}

SPAN_KINDS = ("agent", "model", "tool", "retrieval", "internal")

# ``gen_ai.operation.name`` values per span kind (all four are well-known values of
# the convention); "internal" spans carry none.
_OPERATION_BY_KIND = {"agent": "invoke_agent", "model": "chat", "tool": "execute_tool", "retrieval": "retrieval"}


def _attr(span: "Span", name: str, default: Any = None) -> Any:
    """``span.attrs[name]``, falling back to the pre-convention name this lab used."""
    if name in span.attrs:
        return span.attrs[name]
    return span.attrs.get(_LEGACY_ATTRS.get(name, name), default)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


# ------------------------------------------------------------------------- span
@dataclass
class Span:
    """One timed unit of work. ``start``/``end`` are ``time.perf_counter`` seconds."""

    name: str
    kind: str
    trace_id: str
    id: str = field(default_factory=_new_id)
    parent_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    start: float = 0.0
    end: float | None = None
    status: str = "ok"                 # "ok" | "error"
    exception: str | None = None       # "ExcType: message" when the block raised

    def set(self, **attrs: Any) -> "Span":
        self.attrs.update(attrs)
        return self

    @property
    def finished(self) -> bool:
        return self.end is not None

    @property
    def duration_ms(self) -> float | None:
        """Wall time in milliseconds, or ``None`` while the span is still open."""
        return None if self.end is None else (self.end - self.start) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "id": self.id, "parent_id": self.parent_id,
            "name": self.name, "kind": self.kind, "start": self.start, "end": self.end,
            "duration_ms": self.duration_ms, "status": self.status, "exception": self.exception,
            "attrs": dict(self.attrs),
        }


def span_usage(span: Span) -> Usage:
    """Token usage recorded on a model span (zeros for any other span)."""
    a = span.attrs
    return Usage(
        input_tokens=int(a.get(GEN_AI_INPUT_TOKENS, 0)),
        output_tokens=int(a.get(GEN_AI_OUTPUT_TOKENS, 0)),
        cached_tokens=int(_attr(span, GEN_AI_CACHED_TOKENS, 0)),
    )


@dataclass
class Trace:
    """All spans sharing one ``trace_id``, in start order; ``spans[0]`` is the root."""

    trace_id: str
    spans: list[Span]

    @property
    def root(self) -> Span:
        return self.spans[0]

    @property
    def name(self) -> str:
        return self.root.name

    @property
    def wall_ms(self) -> float:
        """Root duration when finished; otherwise the envelope of the finished spans."""
        if self.root.finished:
            return self.root.duration_ms or 0.0
        ends = [s.end for s in self.spans if s.end is not None]
        return (max(ends) - self.root.start) * 1000.0 if ends else 0.0

    def children(self, span: Span) -> list[Span]:
        return [s for s in self.spans if s.parent_id == span.id]

    def by_kind(self, kind: str) -> list[Span]:
        return [s for s in self.spans if s.kind == kind]


# ----------------------------------------------------------------------- tracer
class Tracer:
    """Collects spans in memory; nesting follows the current asyncio context.

    ``clock`` is injectable so tests can assert exact durations.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock
        self.spans: list[Span] = []
        self._current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            f"agentlab.tracer.{id(self)}", default=None
        )

    # -- opening spans -----------------------------------------------------
    def span(self, name: str, kind: str = "internal", **attrs: Any) -> ContextManager[Span]:
        """Open a child of the current span (or the root of an implicit trace)."""
        return self._open(name, kind, attrs, new_trace=False)

    def start_trace(self, name: str, **attrs: Any) -> ContextManager[Span]:
        """Group one unit of work (a conversation turn) under an explicit root span."""
        return self._open(name, "internal", attrs, new_trace=True)

    def current_span(self) -> Span | None:
        return self._current.get()

    @contextlib.contextmanager
    def _open(self, name: str, kind: str, attrs: dict[str, Any], new_trace: bool) -> Iterator[Span]:
        if kind not in SPAN_KINDS:
            raise ValueError(f"unknown span kind {kind!r}; expected one of {SPAN_KINDS}")
        parent = None if new_trace else self._current.get()
        span = Span(
            name=name, kind=kind,
            trace_id=parent.trace_id if parent else _new_id(),
            parent_id=parent.id if parent else None,
            attrs=dict(attrs), start=self.clock(),
        )
        if kind in _OPERATION_BY_KIND:
            span.attrs.setdefault(GEN_AI_OPERATION_NAME, _OPERATION_BY_KIND[kind])
        self.spans.append(span)
        token = self._current.set(span)
        try:
            yield span
        except BaseException as e:  # noqa: BLE001 - record, then re-raise: tracing never swallows
            span.status = "error"
            span.exception = f"{type(e).__name__}: {e}"
            raise
        finally:
            span.end = self.clock()
            self._restore(token, parent)

    def _restore(self, token: contextvars.Token, parent: Span | None) -> None:
        try:
            self._current.reset(token)
        except ValueError:
            # The span was closed in a different context than it was opened in (an async
            # generator finalised by the event loop). Restore the parent rather than crash.
            self._current.set(parent)

    # -- reading back --------------------------------------------------------
    def traces(self) -> list[Trace]:
        grouped: dict[str, list[Span]] = {}
        for s in self.spans:
            grouped.setdefault(s.trace_id, []).append(s)
        return [Trace(tid, spans) for tid, spans in grouped.items()]

    def trace(self, trace_id: str) -> Trace:
        for t in self.traces():
            if t.trace_id == trace_id or t.trace_id.startswith(trace_id):
                return t
        raise KeyError(f"no trace {trace_id!r}")

    def clear(self) -> None:
        self.spans.clear()

    # -- export ----------------------------------------------------------------
    def to_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.spans]

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dicts(), indent=indent, default=str)

    def render_tree(self, trace_id: str | None = None, price_table: Any = None) -> str:
        """An indented tree per trace with durations, tokens and cost."""
        if price_table is None:
            from .metrics import DEFAULT_PRICES  # metrics builds on this module; import lazily
            price_table = DEFAULT_PRICES
        traces = self.traces() if trace_id is None else [self.trace(trace_id)]
        lines: list[str] = []
        for t in traces:
            lines.append(_trace_header(t, price_table))
            _render_subtree(t, t.root, prefix="", last=True, price_table=price_table, out=lines)
        return "\n".join(lines)

    def print_tree(self, trace_id: str | None = None, price_table: Any = None) -> None:
        print(self.render_tree(trace_id, price_table))


# --------------------------------------------------------------- tree rendering
def _fmt_ms(ms: float | None) -> str:
    return "…" if ms is None else f"{ms:.1f} ms"


def _cost_of(span: Span, price_table: Any) -> float | None:
    model = span.attrs.get(GEN_AI_REQUEST_MODEL)
    if span.kind != "model" or model not in price_table:
        return None
    return price_table.cost(span_usage(span), model)


def _trace_header(t: Trace, price_table: Any) -> str:
    usage = sum((span_usage(s) for s in t.by_kind("model")), Usage())
    costs = [c for c in (_cost_of(s, price_table) for s in t.spans) if c is not None]
    return (f"trace {t.trace_id[:8]} · {t.name} · {len(t.spans)} spans · {_fmt_ms(t.wall_ms)} · "
            f"{usage.input_tokens + usage.output_tokens} tokens · ${sum(costs):.6f}")


def _span_detail(span: Span, price_table: Any) -> str:
    parts: list[str] = []
    if span.kind == "model":
        u = span_usage(span)
        parts.append(f"in {u.input_tokens} (cached {u.cached_tokens}) · out {u.output_tokens}")
        reasons = _attr(span, GEN_AI_FINISH_REASONS)
        if reasons is not None:
            parts.append(",".join(reasons) if isinstance(reasons, (list, tuple)) else str(reasons))
        cost = _cost_of(span, price_table)
        parts.append(f"${cost:.6f}" if cost is not None else "$? (model not priced)")
    elif span.kind == "tool" and TOOL_OK in span.attrs:
        parts.append("ok" if span.attrs[TOOL_OK] else f"error={span.attrs.get(TOOL_ERROR)}")
    if span.status == "error":
        parts.append(f"ERROR {span.exception}")
    return " · ".join(parts)


def _render_subtree(t: Trace, span: Span, prefix: str, last: bool, price_table: Any, out: list[str]) -> None:
    branch = "└─ " if last else "├─ "
    detail = _span_detail(span, price_table)
    out.append(f"{prefix}{branch}{span.name} [{span.kind}] {_fmt_ms(span.duration_ms)}" + (f" · {detail}" if detail else ""))
    children = t.children(span)
    child_prefix = prefix + ("   " if last else "│  ")
    for i, child in enumerate(children):
        _render_subtree(t, child, child_prefix, i == len(children) - 1, price_table, out)


# -------------------------------------------------------------------- redaction
@dataclass(frozen=True)
class RedactionRule:
    """Replace every match of ``pattern`` in a string value with ``replacement``."""

    name: str
    pattern: re.Pattern[str]
    replacement: str

    def apply(self, value: str) -> str:
        return self.pattern.sub(self.replacement, value)


def rule(name: str, pattern: str, replacement: str) -> RedactionRule:
    return RedactionRule(name, re.compile(pattern), replacement)


EMAIL_RULE = rule("email", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>")
CARD_RULE = rule("card", r"\b(?:\d{4}[ -]?){3}\d{4}\b", "<card>")
BEARER_RULE = rule("bearer", r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <token>")
DEFAULT_REDACTION_RULES: tuple[RedactionRule, ...] = (EMAIL_RULE, CARD_RULE, BEARER_RULE)


def redact(value: Any, rules: tuple[RedactionRule, ...] | list[RedactionRule] = DEFAULT_REDACTION_RULES) -> Any:
    """Return a copy of ``value`` with every string scrubbed; dicts, lists and tuples recurse."""
    if isinstance(value, str):
        for r in rules:
            value = r.apply(value)
        return value
    if isinstance(value, dict):
        return {k: redact(v, rules) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v, rules) for v in value)
    return value


class RedactingExporter:
    """Field-level redaction at the export boundary (notebook 09).

    ``drop_attrs`` removes whole attributes (raw prompts, retrieved passages)
    that should never be exported; ``rules`` scrub PII patterns from what
    remains. The tracer's own spans are left untouched, so redaction is applied
    exactly once, at the edge.
    """

    def __init__(self, rules: tuple[RedactionRule, ...] | list[RedactionRule] = DEFAULT_REDACTION_RULES,
                 drop_attrs: tuple[str, ...] = (), sink: Callable[[dict[str, Any]], None] | None = None):
        self.rules = tuple(rules)
        self.drop_attrs = set(drop_attrs)
        self.sink = sink
        self.exported: list[dict[str, Any]] = []

    def scrub(self, payload: Any) -> Any:
        """Redact any JSON-like structure (a span dict, a session dict, a log line)."""
        return redact(payload, self.rules)

    def export(self, tracer: Tracer) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in tracer.to_dicts():
            d["attrs"] = {k: v for k, v in d["attrs"].items() if k not in self.drop_attrs}
            out.append(self.scrub(d))
        self.exported.extend(out)
        if self.sink is not None:
            for d in out:
                self.sink(d)
        return out
