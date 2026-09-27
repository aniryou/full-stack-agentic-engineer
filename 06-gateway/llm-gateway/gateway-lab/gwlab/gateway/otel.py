"""Spans with the OpenTelemetry GenAI names, written as OTLP/JSON lines you can read back.

The one idea (PRIMER §5): the ledger is the bill; spans are the *explanation* — sampled, for debugging and
latency, with names that are still moving. Per request the gateway writes one SERVER span
(`POST /v1/chat/completions`) and one CLIENT span per upstream target it tried (`chat {model}`), because each
target is a different operation (retries against one target stay inside its span, semconv-genai #216).

Names are pinned to the set common to semantic-conventions **v1.41.0** (the last release that defines GenAI)
and semantic-conventions-genai main (e57c543, 2026-09-24), the same names the 07.2 lab's
`agentlab/observability/tracing.py` uses; where the two differ this module uses the released name and
`MAIN_NAMES` maps it (verify when you move the pin):

    gen_ai.usage.cache_creation.input_tokens (v1.41.0)  ->  gen_ai.usage.cache_write.input_tokens (main)
    gen_ai.client.token.usage + gen_ai.token.type        ->  gen_ai.client.inference.usage.* counters (main)

`gen_ai.provider.name` is required and has no value for a self-hosted OpenAI-compatible server; this lab uses
the dialect's well-known value (`openai`, `anthropic`) or the provider's `otel_name` (e.g. `vllm`) — a choice,
not a convention. OTLP/JSON rules (opentelemetry-proto): hex trace and span ids, the span kind as an integer
(1 internal, 2 server, 3 client), 64-bit integers and nanosecond timestamps as *strings*.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

# --- pinned attribute names ----------------------------------------------------------------------------
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_STREAM = "gen_ai.request.stream"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_ID = "gen_ai.response.id"
FINISH_REASONS = "gen_ai.response.finish_reasons"
TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"
INPUT_TOKENS = "gen_ai.usage.input_tokens"                       # includes cached tokens
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"                     # includes reasoning tokens
CACHE_READ = "gen_ai.usage.cache_read.input_tokens"
CACHE_WRITE = "gen_ai.usage.cache_creation.input_tokens"         # v1.41.0; main: ...cache_write.input_tokens
REASONING = "gen_ai.usage.reasoning.output_tokens"
CONVERSATION = "gen_ai.conversation.id"
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"
ERROR_TYPE = "error.type"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
TOKEN_USAGE_METRIC = "gen_ai.client.token.usage"                 # v1.41.0 histogram, attr gen_ai.token.type
OPERATION_DURATION_METRIC = "gen_ai.client.operation.duration"
MAIN_NAMES = {CACHE_WRITE: "gen_ai.usage.cache_write.input_tokens",
              TOKEN_USAGE_METRIC: "gen_ai.client.inference.usage.{input,output,...}_tokens"}
# The lab's own attributes (outside the gen_ai namespace)
GW_TENANT, GW_ALIAS, GW_TARGET, GW_ATTEMPT = "gwlab.tenant", "gwlab.alias", "gwlab.target", "gwlab.attempt"
GW_CACHE, GW_COST = "gwlab.cache", "gwlab.cost_usd"

KIND = {"internal": 1, "server": 2, "client": 3}
STATUS_UNSET, STATUS_OK, STATUS_ERROR = 0, 1, 2


@dataclass
class Span:
    name: str
    kind: int
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    start_ns: int = 0
    end_ns: int = 0
    attributes: dict = field(default_factory=dict)
    status: int = STATUS_UNSET

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9

    def set(self, **attrs) -> "Span":
        for k, v in attrs.items():
            if v is not None:
                self.attributes[k.replace("__", ".")] = v
        return self


def _value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_value(x) for x in v]}}
    return {"stringValue": str(v)}


def _unvalue(d: dict):
    if "boolValue" in d:
        return d["boolValue"]
    if "intValue" in d:
        return int(d["intValue"])
    if "doubleValue" in d:
        return float(d["doubleValue"])
    if "arrayValue" in d:
        return [_unvalue(x) for x in d["arrayValue"].get("values", [])]
    return d.get("stringValue")


def to_otlp(spans: list, service: str = "gwlab-gateway") -> dict:
    """Spans -> one OTLP/JSON `ExportTraceServiceRequest` object."""
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service}}]},
        "scopeSpans": [{"scope": {"name": "gwlab", "version": "0.1.0"}, "spans": [{
            "traceId": s.trace_id, "spanId": s.span_id, "parentSpanId": s.parent_span_id, "name": s.name,
            "kind": s.kind, "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns),
            "attributes": [{"key": k, "value": _value(v)} for k, v in s.attributes.items()],
            "status": {"code": s.status}} for s in spans]}]}]}


def from_otlp(obj: dict) -> list:
    out = []
    for rs in obj.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for s in ss.get("spans", []):
                out.append(Span(name=s["name"], kind=int(s["kind"]), trace_id=s["traceId"], span_id=s["spanId"],
                                parent_span_id=s.get("parentSpanId", ""), start_ns=int(s["startTimeUnixNano"]),
                                end_ns=int(s["endTimeUnixNano"]),
                                attributes={a["key"]: _unvalue(a["value"]) for a in s.get("attributes", [])},
                                status=int((s.get("status") or {}).get("code", 0))))
    return out


def read_spans(path: str) -> list:
    """Read back a JSON-lines file of OTLP/JSON objects."""
    spans = []
    with open(path) as f:
        for line in f:
            if line.strip():
                spans.extend(from_otlp(json.loads(line)))
    return spans


class Tracer:
    """Creates spans and hands finished ones to the exporters: a JSON-lines file (if `path`), an in-memory list,
    and OTLP/HTTP when the OpenTelemetry SDK is installed and `OTEL_EXPORTER_OTLP_ENDPOINT` is set."""

    def __init__(self, path: str = "", keep: int = 5000):
        self.path, self.keep = path, keep
        self.spans: list[Span] = []
        self._lock = threading.Lock()
        self._otlp = maybe_otlp_exporter()

    def start(self, name: str, kind: str, parent: Span | None = None, **attrs) -> Span:
        s = Span(name=name, kind=KIND[kind], trace_id=parent.trace_id if parent else secrets.token_hex(16),
                 span_id=secrets.token_hex(8), parent_span_id=parent.span_id if parent else "",
                 start_ns=time.time_ns())
        return s.set(**attrs)

    def end(self, span: Span, error: str | None = None) -> Span:
        span.end_ns = time.time_ns()
        span.status = STATUS_ERROR if error else STATUS_OK
        if error:
            span.attributes[ERROR_TYPE] = error
        with self._lock:
            self.spans.append(span)
            del self.spans[:-self.keep]
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(to_otlp([span]), separators=(",", ":")) + "\n")
        if self._otlp:
            self._otlp(span)
        return span

    def trace(self, trace_id: str) -> list:
        with self._lock:
            return [s for s in self.spans if s.trace_id == trace_id]


def maybe_otlp_exporter():
    """An OTLP/HTTP sender when the OpenTelemetry exporter package is installed and `OTEL_EXPORTER_OTLP_ENDPOINT`
    is set; None otherwise (the JSON-lines file is the T0 path). Spans are POSTed as OTLP/JSON (which OTLP/HTTP
    accepts with `Content-Type: application/json`) from a background thread, never from the event loop."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return None
    try:
        import opentelemetry.exporter.otlp.proto.http  # noqa: F401  (installed = the operator opted in)
    except ImportError:
        return None
    import queue
    import urllib.request

    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"].rstrip("/") + "/v1/traces"
    q: queue.Queue = queue.Queue(maxsize=10_000)

    def worker():
        while True:
            span = q.get()
            req = urllib.request.Request(endpoint, data=json.dumps(to_otlp([span])).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=5).read()
            except OSError:
                pass                          # telemetry must never fail a request

    threading.Thread(target=worker, name="gwlab-otlp", daemon=True).start()

    def send(span: Span):
        try:
            q.put_nowait(span)
        except queue.Full:
            pass
    return send
