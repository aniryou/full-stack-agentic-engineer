"""Context engineering: what the model sees each turn (Primer §2.5).

The context window is a budget. A ``ContextBuilder`` assembles it in a
cache-friendly order (stable prefix first), truncates oversized tool results,
and compacts old turns into a summary when the history grows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..llm.types import Message, count_tokens, messages_tokens
from .state import Session

Summarizer = Callable[[list[Message]], str]


def naive_summarizer(messages: list[Message]) -> str:
    """A stand-in for an LLM summary: keeps the user asks and tool names."""
    asks = [m["content"] for m in messages if m.get("role") == "user"]
    tools = [m.get("name") for m in messages if m.get("role") == "tool"]
    parts = []
    if asks:
        parts.append("Earlier the user asked: " + " | ".join(str(a)[:80] for a in asks))
    if tools:
        parts.append("Tools already used: " + ", ".join(sorted({str(t) for t in tools})))
    return " ".join(parts) or "(no earlier context)"


@dataclass
class ContextBuilder:
    """Builds the message list for one model call.

    Layout (stable → volatile), so the longest possible prefix is byte-identical
    across turns and across users and can be served from cache:

        [system instruction] [static reference material] [memory] [summary of old turns] [recent turns] [current turn]
    """
    instruction: str
    static_context: list[str] = field(default_factory=list)   # policies, schemas, reference docs (cacheable)
    max_recent_turns: int = 12                                  # user turns kept verbatim
    max_tool_result_chars: int = 1_500
    max_input_tokens: int | None = None                         # hard cap; oldest recent turns dropped first
    summarizer: Summarizer = naive_summarizer
    memory_provider: Optional[Callable[[Session], list[str]]] = None   # per-user facts, injected after the static prefix

    # -- pieces -----------------------------------------------------------
    def system_message(self, session: Session) -> Message:
        text = render_template(self.instruction, session.state)
        if self.static_context:
            text += "\n\n# Reference material\n" + "\n\n".join(self.static_context)
        return {"role": "system", "content": text}

    def memory_messages(self, session: Session) -> list[Message]:
        if not self.memory_provider:
            return []
        facts = self.memory_provider(session)
        if not facts:
            return []
        return [{"role": "system", "content": "# Known about this user\n" + "\n".join(f"- {f}" for f in facts)}]

    @staticmethod
    def _split_turns(messages: list[Message]) -> list[list[Message]]:
        turns: list[list[Message]] = []
        for m in messages:
            if m.get("role") == "user" or not turns:
                turns.append([m])
            else:
                turns[-1].append(m)
        return turns

    def _shape(self, m: Message) -> Message:
        if m.get("role") == "tool" and len(str(m.get("content", ""))) > self.max_tool_result_chars:
            c = str(m["content"])
            return {**m, "content": c[: self.max_tool_result_chars] + f"…[truncated {len(c) - self.max_tool_result_chars} chars; call the tool again with a narrower request if needed]"}
        return m

    # -- assembly ---------------------------------------------------------
    def build(self, session: Session) -> list[Message]:
        history = session.messages()
        turns = self._split_turns(history)
        old, recent = turns[:-self.max_recent_turns], turns[-self.max_recent_turns:]
        out: list[Message] = [self.system_message(session)]
        out += self.memory_messages(session)
        if old:
            flat_old = [m for t in old for m in t]
            out.append({"role": "system", "content": "# Summary of earlier conversation\n" + self.summarizer(flat_old)})
        recent_msgs = [self._shape(m) for t in recent for m in t]
        # enforce a hard token cap by dropping the oldest recent turns (never the current one)
        if self.max_input_tokens is not None:
            while len(recent) > 1 and messages_tokens(out + recent_msgs) > self.max_input_tokens:
                recent = recent[1:]
                recent_msgs = [self._shape(m) for t in recent for m in t]
        return out + recent_msgs

    def cacheable_prefix_tokens(self, session: Session) -> int:
        """Tokens in the part of the prompt that is identical across turns and users."""
        return count_tokens(self.system_message(session)["content"])


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][\w:.\-]*)\}")


def render_template(template: str, state: dict[str, Any]) -> str:
    """Substitute ``{key}`` placeholders from working state; unknown keys are left as-is.

    Unlike ``str.format`` this accepts scoped keys such as ``{user:tier}`` or ``{app:policy}``
    (a colon would otherwise be read as a format spec).
    """
    def sub(m: "re.Match[str]") -> str:
        key = m.group(1)
        return str(state[key]) if key in state else m.group(0)
    return _PLACEHOLDER.sub(sub, template)


def context_report(messages: list[Message]) -> dict[str, Any]:
    """Where the tokens go, by role — the first thing to look at when a prompt is expensive."""
    by_role: dict[str, int] = {}
    for m in messages:
        by_role[m.get("role", "?")] = by_role.get(m.get("role", "?"), 0) + count_tokens(str(m.get("content", "")))
    total = sum(by_role.values())
    return {"total_tokens": total, "by_role": by_role, "messages": len(messages)}
