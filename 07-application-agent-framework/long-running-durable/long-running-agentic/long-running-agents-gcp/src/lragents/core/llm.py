"""LLM boundary.

The whole point of the durable loop is that the LLM call is a *non-deterministic
side effect*: calling it twice with the same prompt can give different answers.
So we (a) isolate it behind a tiny interface, (b) record every decision in the
journal, and (c) never make a real side effect depend on re-asking the model.

``ScriptedLLM`` replays a list of decisions so every pattern can be exercised
offline, deterministically, in tests and notebooks. ``GeminiLLM`` is the real
thing on Vertex AI through the ``google-genai`` SDK.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})


@dataclass
class Decision:
    kind: Literal["tool_call", "final"]
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    text: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    raw: Any = None

    @staticmethod
    def call(tool_name: str, **args: Any) -> "Decision":
        return Decision(kind="tool_call", tool_name=tool_name, tool_args=args, tokens_in=300, tokens_out=40)

    @staticmethod
    def final(text: str) -> "Decision":
        return Decision(kind="final", text=text, tokens_in=300, tokens_out=80)


@dataclass
class PriceCard:
    """USD per 1M tokens. Set these from the current pricing page; defaults are 0."""

    input_per_m: float = 0.0
    output_per_m: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in / 1e6 * self.input_per_m + tokens_out / 1e6 * self.output_per_m


class LLM(Protocol):
    def decide(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Decision: ...


DecisionSource = Decision | Callable[[list[dict[str, Any]]], Decision]


class ScriptedLLM:
    """Replays decisions in order. Each entry is a Decision or a function of the
    message history (so a script can react to tool results)."""

    def __init__(self, script: list[DecisionSource]) -> None:
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def decide(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Decision:
        self.calls.append(messages)
        if not self._script:
            raise RuntimeError("ScriptedLLM ran out of script — the loop asked more than expected")
        nxt = self._script.pop(0)
        return nxt(messages) if callable(nxt) else nxt

    @property
    def remaining(self) -> int:
        return len(self._script)


class GeminiLLM:
    """Gemini on Vertex AI (Agent Platform) via the google-genai SDK.

    Auth is Application Default Credentials — no API keys in code. Set
    GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION (e.g. us-central1 or global).
    """

    def __init__(self, model: str | None = None, project: str | None = None, location: str | None = None, temperature: float = 0.2):
        from google import genai  # imported lazily so offline users never need it

        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.client = genai.Client(
            vertexai=True,
            project=project or os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        self.temperature = temperature

    def decide(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> Decision:
        from google.genai import types

        contents = [
            types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages
        ]
        decls = [
            types.FunctionDeclaration(name=t.name, description=t.description, parameters=t.parameters)
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            tools=[types.Tool(function_declarations=decls)] if decls else None,
        )
        resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        usage = resp.usage_metadata
        tin = getattr(usage, "prompt_token_count", 0) or 0
        tout = getattr(usage, "candidates_token_count", 0) or 0

        cand = resp.candidates[0] if resp.candidates else None
        parts = cand.content.parts if cand and cand.content else []
        for p in parts:
            if p.function_call:
                return Decision(
                    kind="tool_call",
                    tool_name=p.function_call.name,
                    tool_args=dict(p.function_call.args or {}),
                    tokens_in=tin,
                    tokens_out=tout,
                    raw=resp,
                )
        text = "".join(p.text for p in parts if p.text) or ""
        return Decision(kind="final", text=text, tokens_in=tin, tokens_out=tout, raw=resp)


def render_journal(journal_entries: list[dict[str, Any]]) -> str:
    """Compact, deterministic rendering of journal entries for the prompt."""
    return json.dumps(journal_entries, default=str)
