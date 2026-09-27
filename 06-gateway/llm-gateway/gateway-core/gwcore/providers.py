"""Provider adapters as data, a model catalogue, and a fake provider on a virtual clock (PRIMER §1, §2, §5).

The one idea: most of a provider adapter is a *table* -- the path, the auth header, which usage fields
add up to prompt and completion tokens, how finish reasons map -- and only a little is code (the
stream events). Normalise usage at the edge and every later stage (limits, ledger, traces) speaks
one dialect. The ``FakeProvider`` scripts what a real one does to a gateway: TTFT and ITL on a
virtual clock, 429 with ``Retry-After``, outages, and a failure in the middle of a 200 stream.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .api import DONE, chunk, error_body, estimate_tokens, prompt_text

PAYLOADS = json.loads((Path(__file__).parent / "data" / "payloads.json").read_text())

# Canonical usage field <- the provider fields that add up to it (field paths as each API spells them).
# ``cache_write_tokens`` are prompt tokens written to the provider's prompt cache: priced at their own rate (§5.3).
ADAPTERS = {
    "openai": {"path": "/v1/chat/completions", "auth": ("Authorization", "Bearer {key}"),
               "usage": {"prompt_tokens": ["prompt_tokens"], "completion_tokens": ["completion_tokens"],
                         "cached_tokens": ["prompt_tokens_details.cached_tokens"],
                         "cache_write_tokens": ["prompt_tokens_details.cache_write_tokens"],
                         "reasoning_tokens": ["completion_tokens_details.reasoning_tokens"]},
               "finish": {"stop": "stop", "length": "length", "tool_calls": "tool_calls", "content_filter": "content_filter"}},
    "anthropic": {"path": "/v1/messages", "auth": ("x-api-key", "{key}"), "headers": {"anthropic-version": "2023-06-01"},
                  "usage": {"prompt_tokens": ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"],
                            "completion_tokens": ["output_tokens"], "cached_tokens": ["cache_read_input_tokens"],
                            "cache_write_tokens": ["cache_creation_input_tokens"],
                            "reasoning_tokens": ["output_tokens_details.thinking_tokens"]},
                  # pause_turn (a server-tool turn the client must continue) is left unmapped on purpose: it is not a
                  # complete answer, so it passes through as itself and nothing downstream treats it as "stop"
                  "finish": {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
                             "refusal": "content_filter", "model_context_window_exceeded": "length"}},
    "gemini": {"path": "/v1beta/models/{model}:streamGenerateContent?alt=sse", "auth": ("x-goog-api-key", "{key}"),
               "usage": {"prompt_tokens": ["promptTokenCount", "toolUsePromptTokenCount"],
                         "completion_tokens": ["candidatesTokenCount", "thoughtsTokenCount"],
                         "cached_tokens": ["cachedContentTokenCount"], "cache_write_tokens": [],
                         "reasoning_tokens": ["thoughtsTokenCount"]},
               # MALFORMED_FUNCTION_CALL, OTHER and the rest are abnormal endings: unmapped, so they pass through
               "finish": {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "PROHIBITED_CONTENT": "content_filter",
                          "BLOCKLIST": "content_filter", "SPII": "content_filter", "RECITATION": "content_filter"}},
}
ADAPTERS["vllm"] = {**ADAPTERS["openai"], "note": "needs --enable-prompt-tokens-details for cached_tokens and "
                    "--reasoning-parser for reasoning_tokens; error code is the HTTP status as an int"}


def _get(d: dict, dotted: str) -> int:
    for part in dotted.split("."):
        d = (d or {}).get(part) if isinstance(d, dict) else None
    return int(d or 0)


def normalize_usage(dialect: str, raw: dict) -> dict:
    """Provider usage -> canonical ``prompt_tokens`` (cached included), ``completion_tokens`` (reasoning included)."""
    u = {k: sum(_get(raw, f) for f in fields) for k, fields in ADAPTERS[dialect]["usage"].items()}
    u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
    return u


def normalize_finish(dialect: str, reason: str | None) -> str | None:
    """A provider's stop reason -> the canonical one. An unknown or abnormal reason is *not* guessed to be "stop":
    it passes through as the provider spelled it, so the cache (which stores only ``stop``) and the ledger never
    mistake a paused, malformed or unexplained ending for a complete answer."""
    return None if reason is None else ADAPTERS[dialect]["finish"].get(reason, reason)


def normalize_stream(dialect: str, events: list) -> list:
    """A provider's stream events -> canonical chunks (OpenAI shape), ending with one usage chunk and DONE.

    Anthropic's usage is cumulative: ``message_start`` carries the input side (and ``output_tokens`` 1), and the final
    ``message_delta`` the whole count, ``output_tokens_details.thinking_tokens`` included when the API sends it. A
    stream cut before its end (no ``message_delta``/``message_stop``, or no Gemini ``finishReason``) yields no usage
    chunk and no DONE: ``message_start``'s partial usage is not the bill, so the gateway estimates from the deltas."""
    if dialect in ("openai", "vllm"):
        return list(events)
    out, usage, finish, calls, rid, model, complete = [], {}, None, {}, "gw", "", False
    def emit(**kw):
        out.append(chunk(rid, model, 0, **kw))
    for ev in events:
        if dialect == "anthropic":
            kind, d = ev["event"], ev["data"]
            if kind == "message_start":
                rid, model = d["message"]["id"], d["message"]["model"]
                usage.update(d["message"].get("usage", {}))
            elif kind == "content_block_start" and d["content_block"]["type"] == "tool_use":
                calls[d["index"]] = len(calls)
                emit(tool_calls=[{"index": calls[d["index"]], "id": d["content_block"]["id"], "type": "function",
                                  "function": {"name": d["content_block"]["name"], "arguments": ""}}])
            elif kind == "content_block_delta":
                delta = d["delta"]
                if delta["type"] == "text_delta":
                    emit(content=delta["text"])
                elif delta["type"] == "thinking_delta":
                    emit(reasoning=delta["thinking"])
                elif delta["type"] == "input_json_delta":
                    emit(tool_calls=[{"index": calls[d["index"]], "function": {"arguments": delta["partial_json"]}}])
            elif kind == "message_delta":
                usage.update(d.get("usage", {}))          # cumulative: the later value replaces the earlier
                finish = normalize_finish(dialect, d["delta"].get("stop_reason"))
            elif kind == "message_stop":
                complete = True
        else:                                             # gemini: one JSON object per SSE data frame
            model, rid = ev.get("modelVersion", model), ev.get("responseId", rid)
            usage.update(ev.get("usageMetadata", {}))
            for cand in ev.get("candidates", []):
                for part in cand.get("content", {}).get("parts", []):
                    if "functionCall" in part:            # Gemini sends whole calls, not argument fragments
                        i = len(calls)
                        calls[i] = i
                        emit(tool_calls=[{"index": i, "id": part["functionCall"].get("id") or f"call_{i}", "type": "function", "function": {
                            "name": part["functionCall"]["name"], "arguments": json.dumps(part["functionCall"].get("args", {}))}}])
                    elif part.get("thought"):
                        emit(reasoning=part.get("text", ""))
                    elif "text" in part:
                        emit(content=part["text"])
                if cand.get("finishReason"):
                    finish = "tool_calls" if calls and cand["finishReason"] == "STOP" else normalize_finish(dialect, cand["finishReason"])
                    complete = True
    if not complete:                                      # cut: no finish, no usage, no DONE -- report, never invent
        return out
    emit(finish_reason=finish)
    out.append(chunk(rid, model, 0, usage=normalize_usage(dialect, usage)))
    return out + [DONE]


# The model catalogue: USD per 1M tokens (input, output, cached input[, cache write]). Hosted prices dated 2026-09-26
# (verify): the Gemini rows equal scalelab.capacity.PRICES (checked 2026-09-05, unchanged), the others LiteLLM's price
# file @849f303. A cache write without its own price bills at the input rate.
@dataclass(frozen=True)
class Model:
    name: str
    provider: str
    dialect: str
    price: tuple | None            # (input, output, cached[, cache_write]) $/M; None = self-hosted, charged by GPU-seconds
    context: int
    capabilities: frozenset = frozenset({"tools"})
    region: str = "global"


CATALOGUE = {m.name: m for m in [
    Model("gemini-3.5-flash", "google", "gemini", (1.50, 9.00, 0.15), 1_048_576, frozenset({"tools", "reasoning"})),
    Model("gemini-3.5-flash-lite", "google", "gemini", (0.30, 2.50, 0.03), 1_048_576, frozenset({"tools", "reasoning"})),
    Model("gemini-3.1-flash-lite", "google", "gemini", (0.25, 1.50, 0.025), 1_048_576, frozenset({"tools", "reasoning"})),
    Model("gemini-3.8-flash", "google", "gemini", (0.75, 3.75, 0.075), 1_048_576, frozenset({"tools", "reasoning"})),
    Model("gemini-3.1-pro-preview", "google", "gemini", (2.00, 12.00, 0.20), 1_048_576, frozenset({"tools", "reasoning"})),
    Model("gpt-5.4-mini", "openai", "openai", (0.75, 4.50, 0.075), 272_000, frozenset({"tools", "reasoning"})),
    Model("claude-haiku-4-5", "anthropic", "anthropic", (1.00, 5.00, 0.10, 1.25), 200_000, frozenset({"tools", "reasoning"})),
    Model("lab/llm", "self-hosted", "vllm", None, 4_096, frozenset(), "us-central1"),   # Qwen2.5-0.5B-Instruct on vLLM (T1)
]}


class Clock:
    """Virtual seconds: ``sleep`` advances time instantly, so a scripted outage replays exactly."""

    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)


@dataclass
class Response:
    status: int | None              # None = the provider never answered (the gateway's timeout decides)
    headers: dict = field(default_factory=dict)
    body: dict | None = None
    events: object = None           # a streamed 200: a generator of chunks that advances the clock


# The host a provider's CLIENT spans name as ``server.address`` (OTel: the server's domain name). The fakes are
# in-process, so these are labels for the traces, not addresses anything connects to.
HOSTS = {"google": "generativelanguage.googleapis.com", "openai": "api.openai.com", "anthropic": "api.anthropic.com",
         "self": "lab-llm.inference-pool.svc.cluster.local"}


class FakeProvider:
    """An OpenAI-compatible provider on a virtual clock (T0; ``gateway-lab`` serves the same over HTTP).

    ``outages``: (start, end, status) windows, status 429/500/503 or None for "hangs". ``rpm``/``tpm``: its
    own limits over a 60 s window. ``fail_after``: a streamed 200 that breaks after that many chunks.
    ``max_completion_tokens`` bounds reasoning *and* visible tokens together, as OpenAI's spec defines it
    (reasoning first; ``finish_reason: length`` when the bound cut the answer).
    Token counts use ``api.estimate_tokens`` -- this fake has no tokenizer, and says so.
    """
    dialect = "openai"                           # the wire format it speaks, whatever model family it fronts

    def __init__(self, name: str, clock: Clock, *, ttft: float = 0.30, itl: float = 0.020, output_tokens: int = 40,
                 reasoning_tokens: int = 0, outages=(), rpm: int | None = None, tpm: int | None = None,
                 fail_after: int | None = None, context: int = 1_000_000, host: str | None = None):
        self.name, self.clock, self.ttft, self.itl = name, clock, ttft, itl
        self.output_tokens, self.reasoning_tokens, self.outages = output_tokens, reasoning_tokens, list(outages)
        self.rpm, self.tpm, self.fail_after, self.context = rpm, tpm, fail_after, context
        self.host = host or HOSTS.get(name, f"{name}.example.invalid")
        self.window: deque = deque()             # (t, tokens) admitted in the last 60 s
        self.calls = 0

    def _limited(self, tokens: int) -> float | None:
        now = self.clock.now()
        while self.window and self.window[0][0] <= now - 60:
            self.window.popleft()
        over_rpm = self.rpm is not None and len(self.window) + 1 > self.rpm
        over_tpm = self.tpm is not None and sum(n for _, n in self.window) + tokens > self.tpm
        return max(1, math.ceil(self.window[0][0] + 60 - now)) if (over_rpm or over_tpm) and self.window else None

    def chat(self, request: dict) -> Response:
        self.calls += 1
        now, prompt = self.clock.now(), estimate_tokens(prompt_text(request))
        for start, end, status in self.outages:
            if start <= now < end:
                if status is None:
                    return Response(None)
                return Response(status, {"retry-after": str(max(1, math.ceil(end - now)))} if status in (429, 503) else {},
                                error_body(status, f"{self.name}: scripted outage", "rate_limit_error" if status == 429 else "api_error",
                                           "slow_down" if status == 429 else None))
        if "stream_options" in request and not request.get("stream"):
            return Response(400, {}, error_body(400, "Stream options can only be defined when `stream=True`.", "BadRequestError"))
        budget = request.get("max_completion_tokens") or (self.reasoning_tokens + self.output_tokens)
        reasoning = min(self.reasoning_tokens, budget)             # the bound covers reasoning and visible tokens
        n = min(self.output_tokens, budget - reasoning)
        finish = "length" if reasoning + n < self.reasoning_tokens + self.output_tokens else "stop"
        if prompt + reasoning + n > self.context:
            return Response(400, {}, error_body(400, "prompt too long", "invalid_request_error", "context_length_exceeded"))
        wait = self._limited(prompt + n + reasoning)
        if wait is not None:
            return Response(429, {"retry-after": str(wait)}, error_body(429, "rate limited", "rate_limit_error", "slow_down"))
        self.window.append((now, prompt + n + reasoning))
        usage = {"prompt_tokens": prompt, "completion_tokens": n + reasoning, "total_tokens": prompt + n + reasoning,
                 "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": reasoning}}
        words = [f"{w} " for w in (f"{self.name} answers {request['model']}: " + "the quick brown fox " * n).split()][:n]
        if not request.get("stream"):
            self.clock.sleep(self.ttft + self.itl * max(0, reasoning + n - 1))
            return Response(200, {}, {"id": f"{self.name}-{self.calls}", "object": "chat.completion", "created": int(now),
                                      "model": request["model"], "usage": usage, "choices": [{"index": 0, "finish_reason": finish,
                                      "message": {"role": "assistant", "content": "".join(words)}}]})
        return Response(200, {"content-type": "text/event-stream"}, events=self._stream(request, words, usage, reasoning, finish))

    def _stream(self, request: dict, words: list, usage: dict, reasoning: int, finish: str):
        rid, model, sent = f"{self.name}-{self.calls}", request["model"], 0
        deltas = [{"reasoning": "hmm "}] * reasoning + [{"content": w} for w in words]
        for i, d in enumerate(deltas):
            self.clock.sleep(self.ttft if i == 0 else self.itl)
            if self.fail_after is not None and sent == self.fail_after:
                yield error_body(500, f"{self.name}: engine died mid-stream", "InternalServerError")
                yield DONE
                return
            last = i == len(deltas) - 1
            yield chunk(rid, model, int(self.clock.now()), finish_reason=finish if last else None, **d)
            sent += 1
        if (request.get("stream_options") or {}).get("include_usage"):
            yield chunk(rid, model, int(self.clock.now()), usage=usage)
        yield DONE
