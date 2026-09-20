"""A stand-in for a language model, so the whole lab runs offline.

A real tool-calling model returns one of two things on each turn: some text, or
a request to call one or more tools. ``FakeLLM`` returns exactly those two
shapes, from a script you write, so you can practise the *harness* around the
model without an API key or any randomness.

    llm = FakeLLM([call("get_time"), "It is 4pm."])   # 1st turn asks for a tool, 2nd answers

Nothing here is clever — that is the point. The interesting code is the agent
loop (agent.py); the model is deliberately dumb and predictable.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

_ids = itertools.count(1)


@dataclass
class ToolCall:
    """The model's request to run one tool with some arguments."""
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"call_{next(_ids)}"

    def signature(self) -> str:
        """A stable string for 'the same call again' checks (Notebook 03)."""
        items = ",".join(f"{k}={self.args[k]!r}" for k in sorted(self.args))
        return f"{self.name}({items})"


@dataclass
class Response:
    """What the model returns on one turn: text, or tool calls, not both here."""
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict[str, Any]:
        """Turn the response into an assistant message for the transcript."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in self.tool_calls]
        return msg


# -- little helpers so scripts read nicely -------------------------------------
def text(t: str) -> Response:
    return Response(text=t)


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, args=args)


def calls(*tool_calls: ToolCall) -> Response:
    """One turn that asks for several tools at once."""
    return Response(tool_calls=list(tool_calls))


def _coerce(item: Any) -> Response:
    if isinstance(item, Response):
        return item
    if isinstance(item, str):
        return text(item)
    if isinstance(item, ToolCall):
        return Response(tool_calls=[item])
    if isinstance(item, list):
        return Response(tool_calls=list(item))
    raise TypeError(f"cannot turn {item!r} into a Response")


# A policy is a function you write: (messages, tools) -> Response | str | ToolCall(s).
Policy = Callable[[list[dict], list[dict] | None], Any]


class FakeLLM:
    """Drive it three ways:

    * ``FakeLLM([...])``    – a script, one entry per turn (see the helpers above).
    * ``FakeLLM(policy=fn)`` – a function that decides each turn from the messages.
    * ``FakeLLM()``         – echoes the last user message (handy for plumbing).
    """

    def __init__(self, responses: list[Any] | None = None, policy: Policy | None = None):
        self.responses = list(responses) if responses is not None else None
        self.policy = policy
        self.seen: list[list[dict]] = []   # every message list we were asked to answer
        self._i = 0

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Response:
        self.seen.append([dict(m) for m in messages])
        if self.responses is not None:
            if self._i >= len(self.responses):
                raise RuntimeError(
                    f"FakeLLM script ran out on turn {self._i + 1}: the agent made more "
                    f"model calls than the script had answers for."
                )
            r = self.responses[self._i]
            self._i += 1
            return _coerce(r)
        if self.policy is not None:
            return _coerce(self.policy(messages, tools))
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
        return text(f"echo: {last_user['content'] if last_user else ''}")

    @property
    def call_count(self) -> int:
        return len(self.seen)
