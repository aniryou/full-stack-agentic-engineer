"""Core message and response types shared by every model adapter.

The shapes mirror what real SDKs return (a response that carries either text,
tool calls, or both, plus token usage), so that swapping ``FakeLLM`` for a real
model changes one import and nothing else.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable

# A message is a plain dict so it serialises trivially into traces and sessions.
#   {"role": "system"|"user"|"assistant"|"tool", "content": str,
#    "tool_calls": [...]?, "tool_call_id": str?, "name": str?}
Message = dict[str, Any]


def count_tokens(text: str) -> int:
    """Rough token estimate (≈ 4 characters per token for English).

    Real systems call the model's tokenizer; the estimate is enough to make
    budgets, caching and cost calculations behave realistically in the lab.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += count_tokens(str(m.get("content", "")))
        for tc in m.get("tool_calls", []) or []:
            total += count_tokens(str(tc))
    return total


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0     # subset of input_tokens served from a context cache
    thinking_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_tokens + other.cached_tokens,
            self.thinking_tokens + other.thinking_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:8])

    def signature(self) -> str:
        """Stable string used to detect the same call being repeated."""
        import json
        return f"{self.name}:{json.dumps(self.args, sort_keys=True, default=str)}"


@dataclass
class ModelResponse:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"          # "stop" | "tool_calls" | "length" | "safety"
    latency_ms: float = 0.0
    model: str = "fake"

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def as_message(self) -> Message:
        m: Message = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            m["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "args": tc.args} for tc in self.tool_calls
            ]
        return m


@dataclass
class StreamChunk:
    """One piece of a streamed response."""
    text: str = ""
    tool_call: ToolCall | None = None
    done: bool = False
    usage: Usage | None = None
    t_ms: float = field(default_factory=lambda: time.perf_counter() * 1000)


@runtime_checkable
class LLM(Protocol):
    """The only interface agents depend on."""

    model_name: str

    async def generate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **options: Any,
    ) -> AsyncIterator[StreamChunk]: ...
