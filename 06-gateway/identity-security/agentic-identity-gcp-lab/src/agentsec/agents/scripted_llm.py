"""``ScriptedLlm`` — a deterministic stand-in for Gemini so the full ADK loop runs offline.

The point of the lab is the *control plane* around the model (identity, policy, screening,
audit), which is exactly the part a real model makes non-deterministic and un-testable. The
scripted model emits a pre-defined sequence of steps — either a function call or text — one
per model turn, so notebooks and tests can exercise the runner, plugin callbacks, confirmation
round-trips and MCP tool calls without an API key.

Swap it for ``"gemini-2.5-flash"`` (or any model string) in the ``gcp`` profile; nothing else
changes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterable
from typing import Any

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from pydantic import BaseModel, Field, PrivateAttr


class Step(BaseModel):
    text: str | None = None
    function_call: tuple[str, dict[str, Any]] | None = None

    @classmethod
    def call(cls, name: str, **args: Any) -> Step:
        return cls(function_call=(name, args))

    @classmethod
    def say(cls, text: str) -> Step:
        return cls(text=text)

    def __repr__(self) -> str:  # pragma: no cover - display only
        if self.function_call:
            name, args = self.function_call
            return f"Step.call({name!r}, {args})"
        return f"Step.say({self.text!r})"


class ScriptedLlm(BaseLlm):
    """Feed it a list of steps; each ``generate_content_async`` call consumes the next one."""

    model: str = "scripted"
    steps: list[Step] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"scripted.*"]

    def reset(self, steps: Iterable[Step] | None = None) -> None:
        if steps is not None:
            self.steps = list(steps)
        self._cursor = 0
        self.requests.clear()

    @property
    def remaining(self) -> int:
        return max(0, len(self.steps) - self._cursor)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if self._cursor >= len(self.steps):
            yield LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="(script exhausted)")])
            )
            return
        step = self.steps[self._cursor]
        self._cursor += 1
        if step.function_call:
            name, args = step.function_call
            part = types.Part(function_call=types.FunctionCall(name=name, args=args))
        else:
            part = types.Part(text=step.text or "")
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0, candidates_token_count=0, total_token_count=0
        )
        yield LlmResponse(content=types.Content(role="model", parts=[part]), usage_metadata=usage)

    # Convenience for notebooks: what did the model actually see last time?
    def last_tool_results(self) -> list[dict[str, Any]]:
        if not self.requests:
            return []
        out: list[dict[str, Any]] = []
        for content in self.requests[-1].contents or []:
            for p in content.parts or []:
                if p.function_response is not None:
                    out.append(
                        {"name": p.function_response.name, "response": p.function_response.response}
                    )
        return out
