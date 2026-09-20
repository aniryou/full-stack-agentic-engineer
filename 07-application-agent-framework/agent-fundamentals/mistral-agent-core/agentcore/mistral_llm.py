"""Run the same agent loop against a real Mistral model.

The core loop (agent.py) never mentions a provider — it only needs something with
a ``.generate(messages, tools) -> Response`` method. ``FakeLLM`` is that for
offline practice; ``MistralLLM`` is that for Mistral's API. Swapping one for the
other is a single line:

    from agentcore import Agent, tool
    from agentcore.mistral_llm import MistralLLM
    agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[...])

This file is the *only* Mistral-specific code in the repo. It does two small jobs:
translate our tool schemas and messages into Mistral's shapes on the way out, and
translate Mistral's reply back into our ``Response`` on the way in. The four
conversion functions are pure and importable, so the tests exercise them without
a network call or an API key.

Needs ``pip install mistralai`` and ``MISTRAL_API_KEY`` in the environment. The
SDK surface moves — if a call fails, check https://docs.mistral.ai against the
notes here.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .fake_llm import Response, ToolCall


# -- pure converters (no SDK, no network — this is what the tests cover) --------
def to_mistral_tools(schemas: list[dict] | None) -> list[dict] | None:
    """Our ``{"name","description","parameters"}`` → Mistral's function-tool shape."""
    if not schemas:
        return None
    return [{"type": "function", "function": s} for s in schemas]


def to_mistral_messages(messages: list[dict]) -> list[dict]:
    """Our transcript → Mistral messages.

    Only assistant messages need reshaping: our tool calls carry ``args`` as a
    dict, Mistral wants ``function.arguments`` as a JSON string. System, user and
    tool messages already match Mistral's format and pass through unchanged.
    """
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            out.append({
                "role": "assistant",
                "content": m.get("content") or "",
                "tool_calls": [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc.get("args", {}))},
                } for tc in m["tool_calls"]],
            })
        else:
            out.append(m)
    return out


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` whether obj is an SDK object (attr) or a dict — so tests can
    pass plain dicts shaped like a Mistral response."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_response(mistral_response: Any) -> Response:
    """Mistral's reply → our ``Response`` (text, or a list of ToolCalls)."""
    choices = _get(mistral_response, "choices") or []
    if not choices:
        return Response(text="")
    msg = _get(choices[0], "message")
    text = _get(msg, "content")
    tool_calls = []
    for tc in (_get(msg, "tool_calls") or []):
        fn = _get(tc, "function")
        raw_args = _get(fn, "arguments") or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        tool_calls.append(ToolCall(name=_get(fn, "name"), args=args, id=_get(tc, "id") or ""))
    # Mistral returns content="" alongside tool calls; keep text None in that case
    return Response(text=text if not tool_calls else (text or None), tool_calls=tool_calls)


# -- the adapter ---------------------------------------------------------------
class MistralLLM:
    """A drop-in replacement for ``FakeLLM`` backed by Mistral's API.

    ``model`` is any Mistral model string, e.g. ``mistral-large-latest`` (flagship,
    best for agents/tool use), ``mistral-medium-latest`` (cheaper frontier),
    ``mistral-small-latest`` (open-weight, cheap), ``magistral-medium-latest``
    (reasoning), ``codestral-latest`` (code). See docs/MISTRAL.md.
    """

    def __init__(self, model: str = "mistral-large-latest", api_key: str | None = None,
                 tool_choice: str = "auto", temperature: float | None = None):
        try:
            from mistralai import Mistral            # v1.x SDK: `pip install mistralai`
        except ImportError as e:                      # pragma: no cover - depends on env
            raise ImportError(
                "MistralLLM needs the Mistral SDK. Install it with `pip install mistralai` "
                "and set MISTRAL_API_KEY. (The core lab runs fully offline with FakeLLM — "
                "you only need this to talk to a real model.)"
            ) from e
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("set MISTRAL_API_KEY (or pass api_key=...) to use MistralLLM")
        self.model = model
        self.model_name = model
        self.tool_choice = tool_choice
        self.temperature = temperature
        self._client = Mistral(api_key=key)
        self.calls = 0

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Response:
        self.calls += 1
        kwargs: dict[str, Any] = {"model": self.model, "messages": to_mistral_messages(messages)}
        mtools = to_mistral_tools(tools)
        if mtools:
            kwargs["tools"] = mtools
            kwargs["tool_choice"] = self.tool_choice
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return parse_response(self._client.chat.complete(**kwargs))
