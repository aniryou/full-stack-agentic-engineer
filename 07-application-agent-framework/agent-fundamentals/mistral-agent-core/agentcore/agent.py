"""The agent loop — the one idea everything else builds on.

An agent is a loop around a model call:

    1. show the model the conversation and the tools it may use
    2. it replies with either a final answer, or one or more tool calls
    3. if tool calls: run them, append the results, go back to 1
    4. stop on a final answer, or when the step budget runs out

Read ``Agent.run`` top to bottom — it is the whole thing, ~30 lines. Notebook 01
has you build this yourself before you use the packaged version here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .fake_llm import Response
from .tools import Tool, tool_message

# on_confirm(name, args) -> True to allow a confirm=True tool to run, False to decline.
ConfirmHook = Callable[[str, dict], bool]


@dataclass
class Result:
    text: str                      # the agent's final answer (or a stop note)
    messages: list[dict]           # the full transcript, for inspection
    steps: int                     # how many model calls it took
    done: bool = True              # False if it stopped on the step budget

    def transcript(self) -> str:
        """A readable dump of the conversation, for notebooks."""
        lines = []
        for m in self.messages:
            role = m["role"]
            if role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    lines.append(f"  model → call {tc['name']}({tc['args']})")
                if m.get("content"):
                    lines.append(f"  model: {m['content']}")
            elif role == "tool":
                lines.append(f"  tool  ← {m['name']}: {m['content']}")
            else:
                lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)


class Agent:
    def __init__(self, llm, tools: list[Tool] | None = None,
                 instruction: str = "You are a helpful assistant.", max_steps: int = 6):
        self.llm = llm
        self.instruction = instruction
        self.tools = {t.name: t for t in (tools or [])}
        self.max_steps = max_steps

    def run(self, user_message: str, history: list[dict] | None = None,
            on_confirm: ConfirmHook | None = None) -> Result:
        messages: list[dict] = list(history or [])
        if not any(m.get("role") == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": self.instruction})
        messages.append({"role": "user", "content": user_message})
        schemas = [t.schema for t in self.tools.values()]

        for step in range(1, self.max_steps + 1):
            resp: Response = self.llm.generate(messages, tools=schemas or None)
            messages.append(resp.as_message())

            if not resp.tool_calls:                                   # a final answer → done
                return Result(resp.text or "", messages, step, done=True)

            for tc in resp.tool_calls:                                # otherwise run each tool
                tool = self.tools.get(tc.name)
                if tool is None:
                    result = {"ok": False, "error": "unknown_tool",
                              "message": f"no tool named {tc.name!r}",
                              "hint": f"available tools: {', '.join(self.tools) or '(none)'}"}
                elif tool.confirm and not (on_confirm and on_confirm(tc.name, tc.args)):
                    result = {"ok": False, "error": "declined",
                              "message": "this action needs human approval and was not approved",
                              "hint": "Acknowledge and offer an alternative."}
                else:
                    result = tool.run(tc.args)
                messages.append(tool_message(tc, result))

        return Result("(stopped: reached the step budget without a final answer)",
                      messages, self.max_steps, done=False)
