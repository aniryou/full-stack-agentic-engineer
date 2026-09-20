"""Optional adapter for the real Gemini API via the ``google-genai`` SDK.

Nothing in the lab imports this module unless you ask for it. It exists so
that, once you have a key, you can swap ``FakeLLM`` for a real model in any
notebook with one line::

    from agentlab.llm.gemini import GeminiLLM
    llm = GeminiLLM(model="gemini-3-flash")   # needs GOOGLE_API_KEY (or Vertex env vars)

VERIFY BEFORE RELYING ON IT: the SDK surface moves. The calls below follow the
``google-genai`` 1.x shape (``genai.Client().aio.models.generate_content`` with
``types.GenerateContentConfig(tools=[types.Tool(function_declarations=[...])])``).
Check https://googleapis.github.io/python-genai/ and adjust field names if the
SDK has changed. Install with ``pip install -e ".[gemini]"``.
"""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

from .types import Message, ModelResponse, StreamChunk, ToolCall, Usage


def _to_genai_contents(messages: list[Message]) -> tuple[str | None, list[Any]]:
    """Convert lab messages to (system_instruction, contents).

    The SDK's own ``types`` module is imported lazily so the rest of the lab
    never needs it installed.
    """
    from google.genai import types  # type: ignore

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    contents: list[Any] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=str(m.get("content", "")))]))
        elif role == "assistant":
            parts = []
            if m.get("content"):
                parts.append(types.Part.from_text(text=str(m["content"])))
            for tc in m.get("tool_calls", []) or []:
                parts.append(types.Part.from_function_call(name=tc["name"], args=tc.get("args", {})))
            contents.append(types.Content(role="model", parts=parts))
        elif role == "tool":
            # function responses are sent back in a "user" turn in the Gemini API
            contents.append(types.Content(role="user", parts=[types.Part.from_function_response(name=m.get("name", "tool"), response={"result": m.get("content")})]))
    return ("\n".join(system_parts) or None), contents


class GeminiLLM:
    def __init__(self, model: str = "gemini-3-flash", **client_kwargs: Any):
        from google import genai  # type: ignore

        self.model_name = model
        self._client = genai.Client(**client_kwargs)

    def _config(self, tools: list[dict[str, Any]] | None, system: str | None, options: dict[str, Any]):
        from google.genai import types  # type: ignore

        cfg: dict[str, Any] = {}
        if system:
            cfg["system_instruction"] = system
        if tools:
            cfg["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t.get("description", ""), parameters=t.get("input_schema") or t.get("parameters"))
                for t in tools
            ])]
            # we execute tools ourselves; disable the SDK's automatic function calling
            cfg["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)
        if "temperature" in options:
            cfg["temperature"] = options["temperature"]
        if "response_schema" in options:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = options["response_schema"]
        return types.GenerateContentConfig(**cfg)

    async def generate(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **options: Any) -> ModelResponse:
        system, contents = _to_genai_contents(messages)
        t0 = time.perf_counter()
        resp = await self._client.aio.models.generate_content(model=self.model_name, contents=contents, config=self._config(tools, system, options))
        latency_ms = (time.perf_counter() - t0) * 1000
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for cand in (resp.candidates or [])[:1]:
            for part in (cand.content.parts or []):
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    tool_calls.append(ToolCall(name=fc.name, args=dict(fc.args or {})))
                elif getattr(part, "text", None):
                    text_parts.append(part.text)
        um = getattr(resp, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(um, "prompt_token_count", 0) or 0,
            output_tokens=getattr(um, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(um, "cached_content_token_count", 0) or 0,
            thinking_tokens=getattr(um, "thoughts_token_count", 0) or 0,
        )
        return ModelResponse(text="".join(text_parts) or None, tool_calls=tool_calls, usage=usage,
                             finish_reason="tool_calls" if tool_calls else "stop", latency_ms=latency_ms, model=self.model_name)

    async def stream(self, messages: list[Message], tools: list[dict[str, Any]] | None = None, **options: Any) -> AsyncIterator[StreamChunk]:
        system, contents = _to_genai_contents(messages)
        stream = await self._client.aio.models.generate_content_stream(model=self.model_name, contents=contents, config=self._config(tools, system, options))
        async for chunk in stream:
            for cand in (chunk.candidates or [])[:1]:
                for part in (cand.content.parts or []):
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        yield StreamChunk(tool_call=ToolCall(name=fc.name, args=dict(fc.args or {})))
                    elif getattr(part, "text", None):
                        yield StreamChunk(text=part.text)
        yield StreamChunk(done=True)
