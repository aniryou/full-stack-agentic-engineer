"""Run the same agent loop against a real Mistral model: the provider adapter.

The core loop (agent.py) never mentions a provider — it only needs something with
a ``.generate(messages, tools) -> Response`` method. ``FakeLLM`` is that for
offline practice; ``MistralLLM`` is that for Mistral's API. Swapping one for the
other is a single line:

    from agentcore import Agent, tool
    from agentcore.mistral_llm import MistralLLM
    agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[...])

This file is the *only* Mistral-specific code in the lab. It does two small jobs:
translate our tool schemas and messages into Mistral's shapes on the way out, and
translate Mistral's reply back into our ``Response`` on the way in. The conversion
functions are pure and importable, and ``MistralLLM`` accepts an injected client, so
the tests exercise all of it without the SDK, a network call or an API key.

A live call needs the optional extra (``pip install -e ".[mistral]"``, the
``mistralai`` client) and ``MISTRAL_API_KEY``. Without either, ``MistralLLM`` raises
``MistralUnavailable`` with a message that says which one is missing. The SDK surface
moves: this file supports ``mistralai`` 2.x (``from mistralai.client import Mistral``,
2.10 as of 2026-09-26 (verify)) and falls back to the 1.x import; if a call fails,
check https://docs.mistral.ai against the notes here.
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
class MistralUnavailable(RuntimeError):
    """The live path cannot run here: the ``mistralai`` client or ``MISTRAL_API_KEY`` is missing."""


def _sdk_client_class():
    """The SDK's client class: ``mistralai.client.Mistral`` (2.x), else ``mistralai.Mistral`` (1.x)."""
    try:
        from mistralai.client import Mistral       # mistralai 2.x
        return Mistral
    except ImportError:
        pass
    try:
        from mistralai import Mistral              # mistralai 1.x
        return Mistral
    except ImportError as e:
        raise MistralUnavailable(
            "MistralLLM needs the Mistral client: pip install -e \".[mistral]\" (or pip install mistralai), "
            "then set MISTRAL_API_KEY. The rest of the lab runs offline with FakeLLM."
        ) from e


class MistralLLM:
    """A drop-in replacement for ``FakeLLM`` backed by Mistral's API.

    ``model`` is any Mistral model string, e.g. ``mistral-large-latest`` (flagship,
    best for agents/tool use), ``mistral-small-latest`` (open-weight, cheap),
    ``magistral-medium-latest`` (reasoning), ``codestral-latest`` (code). See
    docs/MISTRAL.md. ``client`` injects anything with ``.chat.complete(**kwargs)``
    (a test double, or a preconfigured SDK client); when it is given, neither the
    SDK nor a key is needed.
    """

    def __init__(self, model: str = "mistral-large-latest", api_key: str | None = None,
                 tool_choice: str = "auto", temperature: float | None = None, client: Any = None):
        self.model = model
        self.model_name = model
        self.tool_choice = tool_choice
        self.temperature = temperature
        self.calls = 0
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise MistralUnavailable("set MISTRAL_API_KEY (or pass api_key=...) to call Mistral's API; "
                                     "the rest of the lab runs offline with FakeLLM")
        self._client = _sdk_client_class()(api_key=key)

    def request(self, messages: list[dict], tools: list[dict] | None = None) -> dict[str, Any]:
        """The keyword arguments ``generate`` sends to ``client.chat.complete`` (pure: no call is made)."""
        kwargs: dict[str, Any] = {"model": self.model, "messages": to_mistral_messages(messages)}
        mtools = to_mistral_tools(tools)
        if mtools:
            kwargs["tools"] = mtools
            kwargs["tool_choice"] = self.tool_choice
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Response:
        self.calls += 1
        return parse_response(self._client.chat.complete(**self.request(messages, tools)))
