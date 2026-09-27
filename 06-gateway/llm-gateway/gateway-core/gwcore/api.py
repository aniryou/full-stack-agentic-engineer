"""The lingua franca: chat completions over server-sent events (PRIMER §1).

The one idea: a gateway speaks one API to every app -- OpenAI-style chat completions -- and a
streamed answer is not a response but a sequence of chunks you must *accumulate*: content by
concatenation, tool calls by ``index`` (only the first delta carries the id and name), and usage
from one extra chunk with ``choices: []`` that arrives before ``data: [DONE]`` only if the request
asked for it (``stream_options.include_usage``). Errors can arrive inside an HTTP 200 stream too.
"""
from __future__ import annotations

import json

DONE = "[DONE]"
ESTIMATE_LABEL = "estimate: len(text) // 4 (not a tokenizer)"


def estimate_tokens(text: str) -> int:
    """A labelled estimate (about four characters per token), used only where no ``usage`` exists."""
    return max(1, len(text) // 4) if text else 0


def chat_request(model: str, messages: list, *, stream: bool = False, include_usage: bool = False,
                 max_completion_tokens: int | None = None, **extra) -> dict:
    req = {"model": model, "messages": messages, **extra}
    if stream:
        req["stream"] = True
        if include_usage:
            req["stream_options"] = {"include_usage": True}
    if max_completion_tokens is not None:
        req["max_completion_tokens"] = max_completion_tokens
    return req


def prompt_text(request: dict) -> str:
    return "\n".join(str(m.get("content") or "") for m in request.get("messages", []))


def chunk(rid: str, model: str, created: int, *, content: str | None = None, tool_calls: list | None = None,
          reasoning: str | None = None, finish_reason: str | None = None, usage: dict | None = None) -> dict:
    """One ``chat.completion.chunk``. ``usage`` makes it the final usage chunk (``choices: []``)."""
    out = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model}
    if usage is not None:
        return {**out, "choices": [], "usage": usage}
    delta = {k: v for k, v in (("content", content), ("tool_calls", tool_calls), ("reasoning", reasoning)) if v is not None}
    return {**out, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}


def sse(event) -> str:
    """Encode one event as an SSE frame: ``data: <json>`` (or ``data: [DONE]``) and a blank line."""
    return f"data: {event if event == DONE else json.dumps(event, separators=(',', ':'))}\n\n"


def parse_sse(text: str) -> list:
    """Frames back to events (dicts, or the string ``DONE``). Comments (``: keep-alive``) are skipped."""
    events = []
    for frame in text.split("\n\n"):
        data = [line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")]
        if data:
            payload = "\n".join(data)
            events.append(DONE if payload == DONE else json.loads(payload))
    return events


def error_body(status: int, message: str, type_: str, code: str | None = None) -> dict:
    """OpenAI's error shape: ``code`` is a string or null (vLLM sends the HTTP status as an int)."""
    return {"error": {"message": message, "type": type_, "param": None, "code": code}}


def normalize_error(body: dict, status: int) -> dict:
    err = dict(body.get("error", body))
    if isinstance(err.get("code"), int):          # vLLM: code = HTTP status
        err["code"] = str(err["code"])
    err.setdefault("type", "api_error" if status >= 500 else "invalid_request_error")
    return {"error": {"message": err.get("message", ""), "type": err["type"], "param": err.get("param"), "code": err.get("code")}}


class StreamAccumulator:
    """Fold a stream of chunks into one message, usage and finish reason."""

    def __init__(self):
        self.content, self.reasoning, self.calls = [], [], {}
        self.usage = self.finish_reason = self.error = None
        self.chunks = 0

    def add(self, event) -> None:
        if event == DONE:
            return
        if "error" in event:                     # vLLM sends a mid-stream failure as a data chunk inside HTTP 200
            self.error = event["error"]
            return
        self.chunks += 1
        if event.get("usage"):
            self.usage = event["usage"]          # the last usage seen wins (continuous usage is cumulative)
        for choice in event.get("choices", []):
            d = choice.get("delta", {})
            self.content.append(d.get("content") or "")
            self.reasoning.append(d.get("reasoning") or d.get("reasoning_content") or "")
            for tc in d.get("tool_calls") or []:
                call = self.calls.setdefault(tc["index"], {"id": None, "name": None, "arguments": []})
                call["id"] = tc.get("id") or call["id"]
                fn = tc.get("function") or {}
                call["name"] = fn.get("name") or call["name"]
                call["arguments"].append(fn.get("arguments") or "")
            self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def tool_calls(self) -> list:
        out = []
        for i in sorted(self.calls):
            c = self.calls[i]
            raw = "".join(c["arguments"])
            try:
                args, valid = json.loads(raw or "{}"), True
            except json.JSONDecodeError:
                args, valid = raw, False         # a cut stream leaves half a JSON object: say so, never guess
            out.append({"index": i, "id": c["id"], "name": c["name"], "arguments": args, "valid_json": valid})
        return out

    def text(self) -> str:
        return "".join(self.content)

    def output_estimate(self) -> int:
        """Tokens generated so far, estimated from the deltas -- what a cut stream is billed on."""
        return estimate_tokens(self.text() + "".join(self.reasoning)) + sum(
            estimate_tokens("".join(c["arguments"])) for c in self.calls.values())
