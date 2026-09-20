"""Deterministic stand-ins for a language model.

Everything in this lab runs without an API key. ``FakeLLM`` behaves like a
function-calling model: it returns text or tool calls, reports token usage,
simulates context caching and latency, and records every call so exercises
can assert on what the agent sent.

Three ways to drive it:

* ``FakeLLM(responses=[...])`` – a scripted queue, one entry per call
  (strings become text answers, ``ToolCall`` lists become tool-call turns).
* ``FakeLLM(policy=fn)`` – ``fn(messages, tools) -> ModelResponse | str | list[ToolCall]``
  decides dynamically. ``KeywordPlanner`` is a ready-made policy.
* ``FakeLLM()`` – echoes the last user message (useful for plumbing tests).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Sequence

from .types import Message, ModelResponse, StreamChunk, ToolCall, Usage, count_tokens, messages_tokens

Policy = Callable[[list[Message], list[dict[str, Any]] | None], "ModelResponse | str | list[ToolCall]"]


class FakeLLMExhausted(RuntimeError):
    """Raised when a scripted FakeLLM runs out of responses."""


def text(t: str) -> ModelResponse:
    return ModelResponse(text=t, finish_reason="stop")


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, args=args)


def calls(*tool_calls: ToolCall) -> ModelResponse:
    return ModelResponse(text=None, tool_calls=list(tool_calls), finish_reason="tool_calls")


def _coerce(r: "ModelResponse | str | list[ToolCall] | ToolCall") -> ModelResponse:
    if isinstance(r, ModelResponse):
        return r
    if isinstance(r, str):
        return text(r)
    if isinstance(r, ToolCall):
        return calls(r)
    if isinstance(r, list):
        return calls(*r)
    raise TypeError(f"cannot coerce {type(r)!r} to ModelResponse")


class PrefixCache:
    """Simulates implicit context caching.

    A real model bills a request's leading tokens at the cached rate when they
    exactly match the prefix of a recent request. We approximate that with
    hashes of message prefixes.
    """

    def __init__(self, min_prefix_tokens: int = 32, ttl_s: float = 3600.0):
        self.min_prefix_tokens = min_prefix_tokens
        self.ttl_s = ttl_s
        self._seen: dict[str, float] = {}

    @staticmethod
    def _key(messages: Sequence[Message], upto: int) -> str:
        payload = json.dumps(messages[:upto], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    def lookup_and_record(self, messages: list[Message]) -> int:
        now = time.time()
        cached_tokens = 0
        # longest previously-seen prefix wins
        for upto in range(len(messages), 0, -1):
            k = self._key(messages, upto)
            ts = self._seen.get(k)
            if ts is not None and now - ts <= self.ttl_s:
                prefix_tokens = messages_tokens(messages[:upto])
                if prefix_tokens >= self.min_prefix_tokens:
                    cached_tokens = prefix_tokens
                break
        for upto in range(1, len(messages) + 1):
            self._seen[self._key(messages, upto)] = now
        return cached_tokens


@dataclass
class FakeLLM:
    responses: list[Any] | None = None
    policy: Policy | None = None
    model_name: str = "fake-flash"
    # latency model (only slept when simulate_time=True; always reported)
    ttft_ms: float = 400.0
    tokens_per_sec: float = 150.0
    simulate_time: bool = False
    time_scale: float = 0.01          # 1 simulated second = 10 ms real time
    cache: PrefixCache | None = field(default_factory=PrefixCache)
    calls: list[dict[str, Any]] = field(default_factory=list)
    _cursor: int = 0

    # ------------------------------------------------------------------ core
    def _next(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> ModelResponse:
        if self.responses is not None:
            if self._cursor >= len(self.responses):
                raise FakeLLMExhausted(
                    f"FakeLLM has no response left for call #{self._cursor + 1}; "
                    f"the agent made more model calls than the script expected."
                )
            r = _coerce(self.responses[self._cursor])
            self._cursor += 1
            return r
        if self.policy is not None:
            return _coerce(self.policy(messages, tools))
        # default: echo the last user message
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        return text(f"echo: {last_user['content'] if last_user else ''}")

    def _finish(self, messages: list[Message], resp: ModelResponse) -> ModelResponse:
        input_tokens = messages_tokens(messages)
        output_text = resp.text or ""
        output_tokens = count_tokens(output_text) + sum(count_tokens(json.dumps(tc.args, default=str)) + 4 for tc in resp.tool_calls)
        cached = self.cache.lookup_and_record(messages) if self.cache else 0
        resp.usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=min(cached, input_tokens))
        # latency: TTFT grows mildly with uncached input; generation with output size
        uncached = max(0, input_tokens - resp.usage.cached_tokens)
        gen_ms = output_tokens / self.tokens_per_sec * 1000.0
        resp.latency_ms = self.ttft_ms + uncached * 0.02 + gen_ms
        resp.model = self.model_name
        if not resp.tool_calls and resp.finish_reason == "tool_calls":
            resp.finish_reason = "stop"
        return resp

    async def generate(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **options: Any) -> ModelResponse:
        resp = self._finish(messages, self._next(messages, tools))
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t.get("name") for t in (tools or [])], "options": options, "response": resp})
        if self.simulate_time:
            await asyncio.sleep(resp.latency_ms / 1000.0 * self.time_scale)
        return resp

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **options: Any) -> AsyncIterator[StreamChunk]:
        resp = self._finish(messages, self._next(messages, tools))
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t.get("name") for t in (tools or [])], "options": options, "response": resp})
        if self.simulate_time:
            await asyncio.sleep(self.ttft_ms / 1000.0 * self.time_scale)
        for tc in resp.tool_calls:
            yield StreamChunk(tool_call=tc)
        if resp.text:
            words = resp.text.split(" ")
            per_word_s = (1.0 / self.tokens_per_sec) * 1.3  # ≈1.3 tokens per word
            for i, w in enumerate(words):
                yield StreamChunk(text=w + (" " if i < len(words) - 1 else ""))
                if self.simulate_time:
                    await asyncio.sleep(per_word_s * self.time_scale)
        yield StreamChunk(done=True, usage=resp.usage)

    # --------------------------------------------------------------- helpers
    @property
    def call_count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()
        self._cursor = 0


# ---------------------------------------------------------------- policies
@dataclass
class Rule:
    """When ``keyword`` appears in the latest user message, call ``tool``.

    ``args`` may be a dict or a function ``(user_text) -> dict`` so exercises can
    extract ids with a regex.
    """
    keyword: str
    tool: str
    args: dict[str, Any] | Callable[[str], dict[str, Any]] = field(default_factory=dict)

    def build_args(self, user_text: str) -> dict[str, Any]:
        return self.args(user_text) if callable(self.args) else dict(self.args)


class KeywordPlanner:
    """A toy planner: keywords → tool calls, then a templated final answer.

    Behaviour per model call:
    1. If the messages since the last user turn contain tool results, answer
       with a summary of those results (``answer_template`` receives ``results``).
    2. Otherwise match rules against the last user message; every matching rule
       becomes a tool call in the same turn (so the agent can run them in parallel).
    3. If nothing matches, reply with ``fallback``.
    """

    def __init__(self, rules: list[Rule], answer_template: str = "Here is what I found: {results}", fallback: str = "I can help with account questions. What do you need?", require_known_tools: bool = True):
        self.rules = rules
        self.answer_template = answer_template
        self.fallback = fallback
        self.require_known_tools = require_known_tools

    def __call__(self, messages: list[Message], tools: list[dict[str, Any]] | None) -> ModelResponse:
        # find last user turn and anything after it
        idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
        after = messages[idx + 1:] if idx >= 0 else []
        tool_results = [m for m in after if m.get("role") == "tool"]
        if tool_results:
            results = "; ".join(f"{m.get('name')}={m.get('content')}" for m in tool_results)
            return text(self.answer_template.format(results=results))
        user_text = str(messages[idx]["content"]) if idx >= 0 else ""
        known = {t.get("name") for t in (tools or [])}
        tcs = []
        for r in self.rules:
            if re.search(r.keyword, user_text, flags=re.IGNORECASE):
                if self.require_known_tools and tools is not None and r.tool not in known:
                    continue
                tcs.append(ToolCall(name=r.tool, args=r.build_args(user_text)))
        if tcs:
            return calls(*tcs)
        return text(self.fallback)


def scripted(*items: Any) -> FakeLLM:
    """Shorthand: ``scripted(call("a", x=1), "final answer")``."""
    return FakeLLM(responses=list(items))
