"""Mistral as the model provider: an :class:`~lra.core.ports.LLM` adapter and a function-calling decider.

Optional: ``pip install -e ".[mistral]"`` (the ``mistralai`` SDK); the offline tests drive both classes
with a fake client, so no key is needed. Set ``MISTRAL_API_KEY`` for live calls.

* :class:`MistralLLM` implements the engine's ``LLM`` port (``generate``), so any workflow runs on a
  Mistral model: ``Engine(..., llm=MistralLLM())``. Usage is converted to cost for the run's budget.
* :class:`MistralDecider` is the decider for :mod:`lra.examples.tool_agent`: it turns the run's
  journal into a chat (the goal, then every earlier decision as an assistant tool call with its
  recorded result) and asks the model, via function calling, what to do next.

Two things worth knowing about the Mistral API here (as of 2026-09-26, verify):

* tool-call ids must be exactly 9 alphanumeric characters, so the journal's idempotency key
  (``run_ab12cd:3``) is hashed to a stable 9-character id;
* the SDK is a namespace package: ``from mistralai.client import Mistral``.

Model aliases: ``mistral-medium-latest`` (the default here), ``mistral-small-latest`` for cheap steps,
``mistral-large-latest`` (verify at docs.mistral.ai). The durable orchestration on Mistral Workflows
(Temporal underneath) is in :mod:`lra.adapters.mistral.workflow` (Python 3.12+).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from ...core.models import LLMResponse, LLMUsage

DEFAULT_MODEL = "mistral-medium-latest"
# ILLUSTRATIVE prices in USD per 1M tokens (input, output). Pull real numbers from Mistral's pricing page.
DEFAULT_PRICING_PER_1M: tuple[float, float] = (0.40, 2.00)

DEFAULT_INSTRUCTIONS = (
    "You are an operations agent working one step at a time. Use a tool when the goal needs one; "
    "when the goal is met, reply with a short plain-text summary and no tool call. "
    "Never repeat a tool call whose result you already have."
)


def _client(client: Any | None) -> Any:
    if client is not None:
        return client
    from mistralai.client import Mistral  # lazy: the tests pass a fake client

    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def _usage(resp: Any, pricing: tuple[float, float]) -> LLMUsage:
    u = getattr(resp, "usage", None)
    in_tok = int(getattr(u, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(u, "completion_tokens", 0) or 0)
    return LLMUsage(input_tokens=in_tok, output_tokens=out_tok,
                    cost_usd=in_tok / 1e6 * pricing[0] + out_tok / 1e6 * pricing[1])


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(c, "text", "") for c in content or [])


class MistralLLM:
    """The engine's ``LLM`` port on Mistral chat completions."""

    def __init__(self, *, model: str | None = None, client: Any | None = None,
                 pricing_per_1m: tuple[float, float] = DEFAULT_PRICING_PER_1M) -> None:
        self.model = model or os.environ.get("MISTRAL_MODEL", DEFAULT_MODEL)
        self.client = _client(client)
        self.pricing = pricing_per_1m

    def generate(self, prompt: str, *, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.2, max_output_tokens: int = 2048) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        kw: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                              "max_tokens": max_output_tokens}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        resp = self.client.chat.complete(**kw)
        return LLMResponse(text=_text(resp.choices[0].message.content).strip(), usage=_usage(resp, self.pricing),
                           model=self.model)


# ---------------------------------------------------------------- function calling over the journal
def tool_call_id(key: str) -> str:
    """Mistral requires 9 alphanumeric characters; derive them deterministically from the journal key."""
    return hashlib.sha1(key.encode()).hexdigest()[:9]


def to_messages(goal: str, journal: list[dict[str, Any]], instructions: str = DEFAULT_INSTRUCTIONS) -> list[dict[str, Any]]:
    """Journal -> chat messages. In the journal a tool decision is always followed by its intent, so each
    pair becomes an assistant tool call plus the matching tool result. This is the whole prompt on a
    retry: recorded facts, never the model's memory."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}, {"role": "user", "content": goal}]
    for i, step in enumerate(journal):
        if step["type"] != "decision" or "final" in step:
            continue
        intent = journal[i + 1]                       # the intent journaled right after this decision
        cid = tool_call_id(intent["key"])
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": step["tool"], "arguments": json.dumps(step["args"])}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "name": step["tool"],
                         "content": json.dumps(intent.get("result") if intent["done"] else {"pending": True})})
    return messages


def function_tools(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """``{"charge": {"description", "parameters"}}`` -> the tools list the API expects."""
    return [{"type": "function", "function": {"name": name, "description": spec["description"], "parameters": spec["parameters"]}}
            for name, spec in tools.items()]


class MistralDecider:
    """``decide(goal, journal)`` for :func:`lra.examples.tool_agent.make_tool_agent`, by function calling."""

    def __init__(self, tools: dict[str, dict[str, Any]], *, model: str | None = None, client: Any | None = None,
                 instructions: str = DEFAULT_INSTRUCTIONS) -> None:
        self.tools, self.instructions = tools, instructions
        self.model = model or os.environ.get("MISTRAL_MODEL", DEFAULT_MODEL)
        self.client = _client(client)
        self.calls = 0

    def decide(self, goal: str, journal: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        resp = self.client.chat.complete(model=self.model, messages=to_messages(goal, journal, self.instructions),
                                         tools=function_tools(self.tools), tool_choice="auto", parallel_tool_calls=False)
        msg = resp.choices[0].message
        if msg.tool_calls:
            call = msg.tool_calls[0]
            args = call.function.arguments
            return {"tool": call.function.name, "args": json.loads(args) if isinstance(args, str) else dict(args)}
        return {"final": _text(msg.content).strip()}


# The tool schema the notebook and tests use. ``key`` is injected by the loop, so the model never sees it.
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "charge": {
        "description": "Charge the customer's card. Requires human approval before it executes.",
        "parameters": {"type": "object", "properties": {"amount": {"type": "number", "description": "amount in SGD"}},
                       "required": ["amount"]},
    },
}

__all__ = ["DEFAULT_INSTRUCTIONS", "MistralDecider", "MistralLLM", "TOOL_SPECS", "function_tools", "to_messages",
           "tool_call_id"]
