"""GenAI telemetry at the gateway (PRIMER §5).

The one idea: one SERVER span per request and one CLIENT span per upstream target tried, with attributes
named by the OpenTelemetry GenAI semantic conventions -- pinned to one release, because the names are
still moving -- written as OTLP/JSON, one export request per line, and read back. Traces are sampled
evidence; the ledger (``metering``) is the bill.
"""
from __future__ import annotations

import json
import random

PINNED = "open-telemetry/semantic-conventions v1.41.0 (the last release that defines GenAI; verify)"
# The names shared by v1.41.0 and semantic-conventions-genai main e57c543 -- the 07.2 lab's tracing.py uses the same.
OPERATION_NAME, REQUEST_MODEL, REQUEST_STREAM = "gen_ai.operation.name", "gen_ai.request.model", "gen_ai.request.stream"
PROVIDER_NAME, RESPONSE_MODEL, RESPONSE_ID = "gen_ai.provider.name", "gen_ai.response.model", "gen_ai.response.id"
FINISH_REASONS, TIME_TO_FIRST_CHUNK = "gen_ai.response.finish_reasons", "gen_ai.response.time_to_first_chunk"
INPUT_TOKENS, OUTPUT_TOKENS = "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"        # input includes cached
CACHE_READ, REASONING = "gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.reasoning.output_tokens"
CONVERSATION_ID, SERVER_ADDRESS, ERROR_TYPE = "gen_ai.conversation.id", "server.address", "error.type"
TOOL_NAME, TOOL_CALL_ID = "gen_ai.tool.name", "gen_ai.tool.call.id"
# Where the two disagree we default to the released names and keep the map (verify on every upgrade).
CACHE_CREATION, TOKEN_USAGE, TOKEN_TYPE = "gen_ai.usage.cache_creation.input_tokens", "gen_ai.client.token.usage", "gen_ai.token.type"
RENAMED_ON_MAIN = {CACHE_CREATION: "gen_ai.usage.cache_write.input_tokens",
                   TOKEN_USAGE: "gen_ai.client.inference.usage.* counters + gen_ai.client.inference.operation.* histograms"}
# gen_ai.provider.name is required; there is no well-known value for a self-hosted server -- "vllm" is our choice.
PROVIDER_VALUE = {"openai": "openai", "anthropic": "anthropic", "gemini": "gcp.gemini", "vllm": "vllm"}
SPAN_KIND = {"INTERNAL": 1, "SERVER": 2, "CLIENT": 3}


class Tracer:
    def __init__(self, clock, *, service: str = "llm-gateway", seed: int = 0, epoch: float = 1_790_000_000.0):
        self.clock, self.service, self.rng, self.epoch = clock, service, random.Random(seed), epoch
        self.spans: list = []

    def start(self, name: str, kind: str, parent: dict | None = None, **attrs) -> dict:
        span = {"traceId": parent["traceId"] if parent else f"{self.rng.getrandbits(128):032x}",
                "spanId": f"{self.rng.getrandbits(64):016x}", "parentSpanId": parent["spanId"] if parent else "",
                "name": name, "kind": SPAN_KIND[kind], "start": self.clock.now(), "end": None, "attrs": dict(attrs),
                "status": {"code": 0}}
        self.spans.append(span)
        return span

    def end(self, span: dict, *, error: str | None = None, **attrs) -> None:
        span["attrs"].update({k: v for k, v in attrs.items() if v is not None})
        if error:
            span["attrs"][ERROR_TYPE] = error
        span["status"] = {"code": 2 if error else 1}   # STATUS_CODE_ERROR / OK
        span["end"] = self.clock.now()

    def _otlp(self, s: dict) -> dict:
        nanos = lambda t: str(int(round((self.epoch + t) * 1e9)))   # int64 as a decimal string, as OTLP/JSON requires
        span = {"traceId": s["traceId"], "spanId": s["spanId"], "name": s["name"], "kind": s["kind"],
                "startTimeUnixNano": nanos(s["start"]), "endTimeUnixNano": nanos(s["end"] if s["end"] is not None else s["start"]),
                "attributes": [{"key": k, "value": encode(v)} for k, v in s["attrs"].items()], "status": s["status"]}
        if s["parentSpanId"]:
            span["parentSpanId"] = s["parentSpanId"]
        return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": encode(self.service)}]},
                                   "scopeSpans": [{"scope": {"name": "gwcore", "version": "0.1.0"}, "spans": [span]}]}]}

    def export_jsonl(self, path) -> int:
        with open(path, "w") as f:
            for s in self.spans:
                f.write(json.dumps(self._otlp(s), separators=(",", ":")) + "\n")
        return len(self.spans)


def encode(v) -> dict:
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [encode(x) for x in v]}}
    return {"stringValue": str(v)}


def decode(v: dict):
    (kind, x), = v.items()
    return {"intValue": int, "arrayValue": lambda a: [decode(y) for y in a.get("values", [])]}.get(kind, lambda y: y)(x)


def read_jsonl(path) -> list:
    """OTLP/JSON lines back to flat spans: {name, kind, trace_id, span_id, parent, start_ns, end_ns, attrs}."""
    out = []
    for line in open(path):
        for rs in json.loads(line)["resourceSpans"]:
            for ss in rs["scopeSpans"]:
                for s in ss["spans"]:
                    out.append({"name": s["name"], "kind": s["kind"], "trace_id": s["traceId"], "span_id": s["spanId"],
                                "parent": s.get("parentSpanId", ""), "start_ns": int(s["startTimeUnixNano"]),
                                "end_ns": int(s["endTimeUnixNano"]), "status": s["status"]["code"],
                                "attrs": {a["key"]: decode(a["value"]) for a in s["attributes"]}})
    return out
