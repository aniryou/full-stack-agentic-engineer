"""Provider adapters as data: one table per dialect, and the few functions that apply it.

The one idea: the gateway speaks chat completions to every client and translates to each provider's dialect
behind it. Most of the translation is a table — where the request goes, which header carries the key, where the
token counts live, which stop reason means what — so adding a provider is adding a row, not a code path.
What does *not* normalise cleanly is where the bugs live (PRIMER §1):

* usage — Anthropic's `input_tokens` **excludes** cache reads and writes, so prompt tokens are the sum of three
  fields; Gemini's thinking tokens sit **outside** `candidatesTokenCount`, so output is the sum of two;
* errors — OpenAI's `error.code` is a string (`slow_down`), vLLM's is the HTTP status as an int;
* streams — Anthropic sends named events (`message_start`, `content_block_delta`, `message_delta` with a
  cumulative `output_tokens`, `message_stop`, no `[DONE]`); tool arguments arrive as `input_json_delta`
  fragments per content block, which become `tool_calls[].function.arguments` fragments keyed by `index`;
* reasoning text — `reasoning` in vLLM 0.30.0, `thinking_delta` in Anthropic, absent from OpenAI's schema.

Field names are from openai/openai-openapi, anthropic-sdk-python 1.8.0 and python-genai 2.25.0 (read
2026-09-26, verify). Gemini is table data only here (no fake speaks it); the bundled payloads in
`gwlab/data/payloads/` are sample output in the documented format (illustrative).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from importlib import resources

from ..tokens import message_text, requested_output

ANTHROPIC_DEFAULT_MAX_TOKENS = 1024        # Anthropic requires max_tokens; a gateway must pick one


@dataclass(frozen=True)
class Usage:
    """Canonical usage: `prompt_tokens` INCLUDES cached tokens; `completion_tokens` INCLUDES reasoning tokens."""
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    source: str = "provider"                    # "provider" (authoritative) | "estimate" (admission, cut streams)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_openai(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
                "completion_tokens_details": {"reasoning_tokens": self.reasoning_tokens}}

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------------------------- the table
DIALECTS = {
    "openai": {                                   # OpenAI, vLLM, and every OpenAI-compatible server
        "path": "/v1/chat/completions",
        "auth": ("Authorization", "Bearer {key}"),
        "headers": {},
        "usage": {"prompt": ["prompt_tokens"], "completion": ["completion_tokens"],
                  "cached": "prompt_tokens_details.cached_tokens",
                  "cache_write": "prompt_tokens_details.cache_write_tokens",
                  "reasoning": "completion_tokens_details.reasoning_tokens"},
        "finish": {"stop": "stop", "length": "length", "tool_calls": "tool_calls", "content_filter": "content_filter",
                   "function_call": "tool_calls"},
        "stream_end": "data: [DONE]",
        "otel_provider": "openai",
    },
    "anthropic": {                                # Anthropic Messages (vLLM 0.30.0 also serves /v1/messages)
        "path": "/v1/messages",
        "auth": ("x-api-key", "{key}"),
        "headers": {"anthropic-version": "2023-06-01"},
        "usage": {"prompt": ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"],
                  "completion": ["output_tokens"], "cached": "cache_read_input_tokens",
                  "cache_write": "cache_creation_input_tokens", "reasoning": "output_tokens_details.thinking_tokens"},
        "finish": {"end_turn": "stop", "stop_sequence": "stop", "pause_turn": "stop", "max_tokens": "length",
                   "model_context_window_exceeded": "length", "tool_use": "tool_calls", "refusal": "content_filter"},
        "stream_end": "event: message_stop",
        "otel_provider": "anthropic",
    },
    "gemini": {                                   # Gemini API (table data only: no fake speaks it)
        "path": "/v1beta/models/{model}:streamGenerateContent?alt=sse",
        "auth": ("x-goog-api-key", "{key}"),
        "headers": {},
        "usage": {"prompt": ["promptTokenCount", "toolUsePromptTokenCount"],
                  "completion": ["candidatesTokenCount", "thoughtsTokenCount"],
                  "cached": "cachedContentTokenCount", "cache_write": None, "reasoning": "thoughtsTokenCount"},
        "finish": {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "RECITATION": "content_filter",
                   "BLOCKLIST": "content_filter", "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter",
                   "MALFORMED_FUNCTION_CALL": "stop", "OTHER": "stop"},
        "stream_end": "end of the HTTP body",
        "otel_provider": "gcp.gemini",
    },
}


def _get(d: dict | None, path: str | None) -> int:
    cur = d
    for part in (path or "").split("."):
        if not part or not isinstance(cur, dict):
            return 0
        cur = cur.get(part)
    return int(cur or 0) if isinstance(cur, (int, float)) else 0


def normalise_usage(dialect: str, raw: dict | None) -> Usage | None:
    """A provider's usage object -> canonical `Usage` (prompt includes cached; completion includes reasoning)."""
    if not raw:
        return None
    u = DIALECTS[dialect]["usage"]
    return Usage(prompt_tokens=sum(_get(raw, p) for p in u["prompt"]),
                 completion_tokens=sum(_get(raw, p) for p in u["completion"]),
                 cached_tokens=_get(raw, u["cached"]), cache_write_tokens=_get(raw, u["cache_write"]),
                 reasoning_tokens=_get(raw, u["reasoning"]))


def normalise_finish(dialect: str, reason: str | None) -> str | None:
    if reason is None:
        return None
    return DIALECTS[dialect]["finish"].get(reason, "stop")


def auth_headers(dialect: str, key: str) -> dict:
    name, fmt = DIALECTS[dialect]["auth"]
    return {name: fmt.format(key=key), **DIALECTS[dialect]["headers"], "Content-Type": "application/json"}


# ---------------------------------------------------------------------------------------------- errors
_STATUS_CODE = {400: "bad_request", 401: "invalid_api_key", 403: "permission_denied", 404: "model_not_found",
                408: "timeout", 413: "request_too_large", 429: "rate_limit_exceeded", 500: "internal_error",
                502: "bad_gateway", 503: "service_unavailable", 504: "gateway_timeout", 529: "overloaded"}
_STATUS_TYPE = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
                404: "not_found_error", 429: "rate_limit_error", 529: "overloaded_error"}
_ANTHROPIC_STATUS = {"invalid_request_error": 400, "authentication_error": 401, "permission_error": 403,
                     "not_found_error": 404, "request_too_large": 413, "rate_limit_error": 429, "api_error": 500,
                     "overloaded_error": 529}


@dataclass
class UpstreamError(Exception):
    """A provider error, normalised to OpenAI's shape: `code` is always a string."""
    status: int
    type: str
    code: str
    message: str
    retry_after: float | None = None

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": None, "code": self.code}}

    def __str__(self):
        return f"{self.status} {self.code}: {self.message}"


def normalise_error(dialect: str, status: int, body, retry_after: float | None = None) -> UpstreamError:
    """Any provider's error body -> `UpstreamError`. Recognises context-length and content-policy errors."""
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        err = {"message": str(body)[:300] if body else f"HTTP {status}"}
    msg = str(err.get("message") or "")
    etype = str(err.get("type") or _STATUS_TYPE.get(status, "api_error"))
    if dialect == "anthropic" and etype in _ANTHROPIC_STATUS and status in (0, 200):
        status = _ANTHROPIC_STATUS[etype]              # an `error` event inside a 200 stream
    code = err.get("code")
    code = code if isinstance(code, str) and code else _STATUS_CODE.get(status, f"http_{status}")
    low = msg.lower()
    if "maximum context length" in low or "prompt is too long" in low or "context_length_exceeded" in low:
        code = "context_length_exceeded"
    if code in ("content_policy_violation", "content_filter") or "content policy" in low:
        code = "content_filter"
    return UpstreamError(status=status, type=etype, code=code, message=msg, retry_after=retry_after)


# ---------------------------------------------------------------------------------------------- requests
def to_upstream(dialect: str, body: dict, upstream_model: str, *, stream: bool, want_usage: bool = True,
                cache_salt: str | None = None, output_cap: int | None = None) -> dict:
    """The body to send upstream for a client's chat-completions `body`.

    OpenAI dialect: the client's body with `model` rewritten; on a *streamed* request `stream_options.include_usage`
    is set (never on a non-streamed one: vLLM answers 400 "Stream options can only be defined when `stream=True`");
    `cache_salt` for engines that take it; the output cap applied as `max_completion_tokens`.
    Anthropic dialect: system prompt lifted out, tools and tool results re-shaped, `max_tokens` made explicit.
    """
    cap = requested_output(body)
    if output_cap is not None:
        cap = min(cap, output_cap) if cap else output_cap
    if dialect == "openai":
        up = {k: v for k, v in body.items() if k not in ("stream_options", "max_tokens", "max_completion_tokens")}
        up["model"] = upstream_model
        up["stream"] = bool(stream)
        if stream and want_usage:
            up["stream_options"] = {"include_usage": True}
        if cap:
            up["max_completion_tokens"] = cap
        if cache_salt:
            up["cache_salt"] = cache_salt
        return up
    if dialect == "anthropic":
        return _to_anthropic(body, upstream_model, stream=stream, cap=cap)
    raise ValueError(f"no HTTP adapter for dialect {dialect!r} in this lab (table data only)")


def _to_anthropic(body: dict, model: str, *, stream: bool, cap: int | None) -> dict:
    system, messages = [], []
    for m in body.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            system.append(message_text(m))
        elif role == "tool":
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                                                          "content": message_text(m)}]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            for tc in m["tool_calls"]:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id"), "name": fn.get("name"), "input": args})
            messages.append({"role": "assistant", "content": blocks})
        else:
            messages.append({"role": role, "content": message_text(m)})
    up = {"model": model, "messages": messages, "max_tokens": cap or ANTHROPIC_DEFAULT_MAX_TOKENS, "stream": bool(stream)}
    if system:
        up["system"] = "\n".join(system)
    for k in ("temperature", "top_p"):
        if k in body:
            up[k] = body[k]
    if body.get("stop"):
        up["stop_sequences"] = body["stop"] if isinstance(body["stop"], list) else [body["stop"]]
    if body.get("tools"):
        up["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                        "input_schema": t["function"].get("parameters") or {"type": "object"}}
                       for t in body["tools"] if t.get("type") == "function"]
        tc = body.get("tool_choice")
        if tc == "required":
            up["tool_choice"] = {"type": "any"}
        elif tc == "none":
            up["tool_choice"] = {"type": "none"}
        elif isinstance(tc, dict) and tc.get("function"):
            up["tool_choice"] = {"type": "tool", "name": tc["function"]["name"]}
    return up


# ---------------------------------------------------------------------------------------------- responses
def completion_from_anthropic(msg: dict, alias: str) -> tuple[dict, Usage | None]:
    """A non-streamed Anthropic message -> a `chat.completion` (served under the client's alias) and its usage."""
    text, calls = [], []
    for block in msg.get("content") or []:
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append({"id": block.get("id"), "type": "function",
                          "function": {"name": block.get("name"), "arguments": json.dumps(block.get("input") or {})}})
    usage = normalise_usage("anthropic", msg.get("usage"))
    message = {"role": "assistant", "content": "".join(text) or None}
    if calls:
        message["tool_calls"] = calls
    out = {"id": "chatcmpl-" + str(msg.get("id", uuid.uuid4().hex))[-16:], "object": "chat.completion",
           "created": int(time.time()), "model": alias,
           "choices": [{"index": 0, "message": message, "finish_reason": normalise_finish("anthropic", msg.get("stop_reason"))}]}
    if usage:
        out["usage"] = usage.to_openai()
    return out, usage


class AnthropicStreamTranslator:
    """Anthropic stream events -> `chat.completion.chunk` dicts, one event at a time.

    `feed(event_name, data)` returns the chunks to relay (possibly none). Input usage arrives in `message_start`,
    output usage (cumulative) in `message_delta`; `done` turns true at `message_stop`. An `error` event becomes
    a chunk with an `error` object, as vLLM would send it.
    """

    def __init__(self, alias: str):
        self.alias = alias
        self.id = "chatcmpl-" + uuid.uuid4().hex[:16]
        self.created = int(time.time())
        self.raw_usage: dict = {}
        self.finish: str | None = None
        self.done = False
        self._tool_index: dict[int, int] = {}

    def _chunk(self, delta: dict, finish: str | None = None) -> dict:
        return {"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.alias,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    @property
    def usage(self) -> Usage | None:
        return normalise_usage("anthropic", self.raw_usage) if self.raw_usage else None

    def feed(self, event: str | None, data: dict) -> list[dict]:
        kind = event or data.get("type")
        if kind == "message_start":
            self.raw_usage.update((data.get("message") or {}).get("usage") or {})
            return [self._chunk({"role": "assistant", "content": ""})]
        if kind == "content_block_start":
            block = data.get("content_block") or {}
            if block.get("type") == "tool_use":
                k = self._tool_index.setdefault(data["index"], len(self._tool_index))
                return [self._chunk({"tool_calls": [{"index": k, "id": block.get("id"), "type": "function",
                                                     "function": {"name": block.get("name"), "arguments": ""}}]})]
            return []
        if kind == "content_block_delta":
            d = data.get("delta") or {}
            if d.get("type") == "text_delta":
                return [self._chunk({"content": d.get("text", "")})]
            if d.get("type") == "input_json_delta":
                k = self._tool_index.setdefault(data["index"], len(self._tool_index))
                return [self._chunk({"tool_calls": [{"index": k, "function": {"arguments": d.get("partial_json", "")}}]})]
            if d.get("type") == "thinking_delta":
                return [self._chunk({"reasoning": d.get("thinking", "")})]
            return []
        if kind == "message_delta":
            self.raw_usage.update({k: v for k, v in (data.get("usage") or {}).items() if v is not None})
            self.finish = normalise_finish("anthropic", (data.get("delta") or {}).get("stop_reason"))
            return [self._chunk({}, self.finish)]
        if kind == "message_stop":
            self.done = True
            return []
        if kind == "error":
            err = normalise_error("anthropic", 200, data)
            return [{"error": err.body()["error"] | {"status": err.status}}]
        return []                                      # ping, content_block_stop


# ---------------------------------------------------------------------------------------------- bundled payloads
PAYLOAD_LABEL = "sample output in the documented format (illustrative)"


def payload(name: str) -> dict:
    """A bundled sample payload (`openai_usage`, `anthropic_message`, `gemini_response`, ...), labelled illustrative."""
    text = resources.files("gwlab").joinpath("data", "payloads", f"{name}.json").read_text()
    return json.loads(text)
