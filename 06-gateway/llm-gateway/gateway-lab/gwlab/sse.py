"""Server-sent events, and what a gateway must do with a chat-completions stream.

The one idea: chat completions + SSE is the gateway's lingua franca, and a gateway (unlike the 05 router,
which forwards bytes untouched) must *parse* every chunk — to meter it, to translate other dialects into it,
to strip what the client did not ask for and to notice an error sent inside an HTTP 200. On the wire:

    data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n
    ...
    data: {"object":"chat.completion.chunk","choices":[],"usage":{...}}\n\n     <- only with include_usage
    data: [DONE]\n\n

`SSEParser` turns bytes that arrive split anywhere (mid-line, mid-UTF-8 character) into events;
`StreamAccumulator` rebuilds the full completion from chunks, including tool calls whose `arguments` arrive
as string fragments keyed by `index` (only the first fragment carries the call's `id` and `name`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .tokens import estimate_text

DONE = "[DONE]"


def encode(obj: dict, event: str | None = None) -> bytes:
    """One SSE event: optional `event:` line, one `data:` line of compact JSON, a blank line."""
    head = f"event: {event}\n".encode() if event else b""
    return head + b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def encode_done() -> bytes:
    return b"data: [DONE]\n\n"


@dataclass
class Event:
    data: str
    event: str | None = None

    def json(self):
        return json.loads(self.data)


class SSEParser:
    """Incremental SSE parser: `feed(bytes)` returns the events completed so far.

    Handles `\\r\\n` and `\\n`, multi-line `data:` (joined with `\\n`), `event:` names, and comment lines
    (`: keep-alive`, which vLLM sends with `--sse-keep-alive-interval`). Bytes are buffered until a full
    line exists, so a UTF-8 character split across two network reads is decoded intact.
    """

    def __init__(self):
        self._buf = b""
        self._data: list[str] = []
        self._event: str | None = None

    def feed(self, chunk: bytes) -> list[Event]:
        self._buf += chunk
        out = []
        while True:
            i = self._buf.find(b"\n")
            if i < 0:
                break
            line, self._buf = self._buf[:i].rstrip(b"\r").decode("utf-8"), self._buf[i + 1:]
            if line == "":
                if self._data:
                    out.append(Event("\n".join(self._data), self._event))
                self._data, self._event = [], None
            elif line.startswith(":"):
                continue
            else:
                name, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if name == "data":
                    self._data.append(value)
                elif name == "event":
                    self._event = value
        return out

    def flush(self) -> list[Event]:
        """Events left when the stream ends without a trailing blank line."""
        out = self.feed(b"\n\n") if self._buf.strip() or self._data else []
        return out


@dataclass
class StreamAccumulator:
    """Rebuild a chat completion from `chat.completion.chunk` objects.

    `usage` keeps the *last* non-null value (vLLM's `continuous_usage_stats` and `--enable-force-include-usage`
    put a running usage on every chunk: never sum them). `error` is set when a chunk carries an `error`
    object — vLLM sends a mid-stream failure that way, inside an HTTP 200.
    """
    id: str | None = None
    model: str | None = None
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    usage: dict | None = None
    error: dict | None = None
    chunks: int = 0
    content_chunks: int = 0
    tool_calls: dict = field(default_factory=dict)       # index -> {"id", "type", "function": {"name", "arguments"}}

    def add(self, chunk: dict) -> None:
        self.chunks += 1
        if chunk.get("error"):
            self.error = chunk["error"] if isinstance(chunk["error"], dict) else {"message": str(chunk["error"])}
            return
        self.id = self.id or chunk.get("id")
        self.model = self.model or chunk.get("model")
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.content += delta["content"]
                self.content_chunks += 1
            for key in ("reasoning", "reasoning_content"):      # vLLM 0.30.0 says `reasoning`
                if delta.get(key):
                    self.reasoning += delta[key]
            for tc in delta.get("tool_calls") or []:
                slot = self.tool_calls.setdefault(tc["index"], {"id": None, "type": "function",
                                                                "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    @property
    def calls(self) -> list[dict]:
        return [self.tool_calls[i] for i in sorted(self.tool_calls)]

    def output_estimate(self) -> int:
        """Output tokens estimated from what was streamed (content, reasoning, tool-call arguments)."""
        args = "".join(c["function"]["name"] + c["function"]["arguments"] for c in self.calls)
        return estimate_text(self.content) + estimate_text(self.reasoning) + estimate_text(args)

    def completion(self) -> dict:
        """The non-streamed `chat.completion` this stream is equivalent to."""
        msg = {"role": "assistant", "content": self.content or None}
        if self.reasoning:
            msg["reasoning"] = self.reasoning
        if self.tool_calls:
            msg["tool_calls"] = self.calls
        out = {"id": self.id, "object": "chat.completion", "model": self.model,
               "choices": [{"index": 0, "message": msg, "finish_reason": self.finish_reason}]}
        if self.usage:
            out["usage"] = self.usage
        return out


def parse_stream(raw: bytes) -> tuple[list[dict], bool]:
    """Parse a whole SSE body into chunk objects; returns (chunks, saw_done)."""
    p = SSEParser()
    events = p.feed(raw) + p.flush()
    chunks, done = [], False
    for ev in events:
        if ev.data.strip() == DONE:
            done = True
        else:
            chunks.append(json.loads(ev.data))
    return chunks, done
