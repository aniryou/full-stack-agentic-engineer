"""llm.py — the model behind the agent: a scripted one at T0, any OpenAI-compatible chat server at T1.

The one idea: the agent loop only needs ``generate(messages, tools) -> Response`` where a response is
text *or* tool calls (07.1's contract). ``ScriptedModel`` plays a tool-using assistant with the
grammar in ``extract.py``: it calls ``remember`` for confident first-person facts, ``recall`` for
questions about the user, ``forget`` when asked to, and otherwise answers from what its context holds.
It is deliberately *gullible*: an instruction inside a tool result ("remember that…") is obeyed, so
the defences are tested against a model that does what injected text says (PRIMER §8).

``ChatClient`` speaks ``POST /v1/chat/completions`` with ``tools`` (vLLM needs
``--enable-auto-tool-choice --tool-call-parser hermes`` for Qwen2.5, facts sheet §11) and reports
``usage.prompt_tokens_details.cached_tokens`` when the server provides it
(``--enable-prompt-tokens-details`` on vLLM v0.30.0). Messages move between the two formats with
``to_openai`` / ``from_openai``.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .extract import UNKNOWN, answer, extract, is_forget_request, is_question, question_slot
from .records import count_tokens

_ids = itertools.count(1)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"call_{next(_ids)}"


@dataclass
class Response:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)       # prompt_tokens, completion_tokens, cached_tokens

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in self.tool_calls]
        return msg


# ------------------------------------------------------------------------------------------- formats
def to_openai(messages: list[dict]) -> list[dict]:
    """07.1-style messages (``tool_calls: [{id, name, args}]``) → OpenAI chat messages."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or None,
                        "tool_calls": [{"id": tc["id"], "type": "function",
                                        "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                                       for tc in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""), "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def from_openai(messages: list[dict]) -> list[dict]:
    """OpenAI chat messages → 07.1-style messages (tool names recovered from the calls they answer)."""
    names: dict[str, str] = {}
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):   # content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments") or "{}"
                calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                              "args": json.loads(args) if isinstance(args, str) else args})
                names[tc.get("id", "")] = fn.get("name", "")
            out.append({"role": "assistant", "content": content or "", "tool_calls": calls})
        elif m.get("role") == "tool":
            out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                        "name": names.get(m.get("tool_call_id", ""), m.get("name", "")), "content": content or ""})
        else:
            out.append({"role": m.get("role", "user"), "content": content or ""})
    return out


def tool_schemas_openai(schemas: list[dict] | None) -> list[dict] | None:
    return [{"type": "function", "function": s} for s in schemas] if schemas else None


def prompt_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m.get("content") or "") + 4 for m in messages)


# ------------------------------------------------------------------------------------------- scripted
FIRST_PERSON = re.compile(r"\b(my|me|i|i'm|mine)\b", re.I)
INSTRUCTION_IN_DATA = re.compile(r"\bremember (?:that )?(?P<text>[^.\n]+\.)", re.I)


class ScriptedModel:
    """A deterministic tool-using assistant (see the module docstring). ``recall_on_first_person``: in
    tools mode it recalls only when a question is visibly about the user — "What is my employer?" —
    and misses implicit needs such as a recipe that should respect a diet (PRIMER §6)."""

    label = "scripted model (T0; deterministic templates, not a language model)"

    def __init__(self, recall_on_first_person: bool = True, remember_floor: float = 0.8):
        self.recall_on_first_person = recall_on_first_person
        self.remember_floor = remember_floor
        self.calls = 0

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Response:
        self.calls += 1
        names = {t["name"] for t in (tools or [])}
        last = messages[-1]
        reply = self._decide(messages, last, names)
        if isinstance(reply, str):
            reply = Response(text=reply)
        reply.usage = {"prompt_tokens": prompt_tokens(messages),
                       "completion_tokens": count_tokens(reply.text or json.dumps([tc.args for tc in reply.tool_calls]))}
        return reply

    def _context(self, messages: list[dict]) -> list[str]:
        """Everything the model can read except the current question: system and memory blocks, earlier
        turns, tool results (a ``recall`` result's items read like memory lines)."""
        out = []
        for i, m in enumerate(messages):
            if i == len(messages) - 1 and m["role"] != "tool":
                break
            content = m.get("content") or ""
            if m["role"] == "tool":
                try:
                    data = json.loads(content).get("data")
                except (json.JSONDecodeError, AttributeError):
                    data = None
                if isinstance(data, list):
                    content = "\n".join(f"- [{d.get('date')}] {d.get('text')}" if d.get("date") else f"- {d.get('text')}"
                                         for d in data if isinstance(d, dict))
            out.append(content)
        return out

    def _decide(self, messages, last, names):
        if last["role"] == "user":
            text = last.get("content") or ""
            # the user message may carry a transient memory block at the tail: the question is its last line
            question = text.split("<<<END MEMORY>>>")[-1].strip()
            if is_forget_request(question) and "forget" in names:
                slot = question_slot(question) or question
                return Response(tool_calls=[ToolCall("forget", {"subject": slot})])
            if is_question(question):
                if "recall" in names and (not self.recall_on_first_person or FIRST_PERSON.search(question)):
                    return Response(tool_calls=[ToolCall("recall", {"query": question})])
                return answer(question, self._context(messages) + [text])
            facts = [f for f in extract(question) if f.confidence >= self.remember_floor]
            if facts and "remember" in names:
                return Response(tool_calls=[ToolCall("remember", {"text": f.text, "slot": f.slot, "value": f.value,
                                                                  "kind": f.kind}) for f in facts])
            return "Noted."
        if last["role"] == "tool":
            tool = last.get("name", "")
            q = next((m for m in reversed(messages) if m["role"] == "user"), {"content": ""})
            question = (q.get("content") or "").split("<<<END MEMORY>>>")[-1].strip()
            if tool == "recall":
                return answer(question, self._context(messages)) if is_question(question) else "Noted."
            if tool == "remember":
                return "Noted — I'll remember that."
            if tool == "forget":
                try:
                    result = json.loads(last.get("content") or "{}")
                except json.JSONDecodeError:
                    result = {}
                if not result.get("ok"):
                    return "I did not delete anything: " + str(result.get("message", "the request was declined"))
                n = (result.get("data") or {}).get("forgotten", 0)
                return f"Done — I have forgotten that ({n} records)." if n else "I found nothing to forget."
            m = INSTRUCTION_IN_DATA.search(last.get("content") or "")
            if m and "remember" in names:          # gullible: obeys an instruction found in data
                return Response(tool_calls=[ToolCall("remember", {"text": m.group("text").strip(), "kind": "semantic"})])
            return "Here is what I found: " + (last.get("content") or "")[:120]
        return UNKNOWN


# ------------------------------------------------------------------------------------------- HTTP
class ChatClient:
    """``generate(messages, tools)`` against ``POST {url}/v1/chat/completions`` (non-streaming).
    ``extra`` is merged into every request body (e.g. ``{"cache_salt": ...}``, ``{"temperature": 0}``)."""

    def __init__(self, url: str, model: str | None = None, api_key: str | None = None, timeout_s: float = 120,
                 extra: dict | None = None):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra = {"temperature": 0, **(extra or {})}
        self.model = model or self._discover()
        self.label = f"{self.model} via {self.url}"
        self.calls = 0
        self.last_headers: dict = {}

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _discover(self) -> str:
        req = urllib.request.Request(self.url + "/v1/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read())["data"][0]["id"]

    def generate(self, messages: list[dict], tools: list[dict] | None = None, **extra) -> Response:
        self.calls += 1
        body = {"model": self.model, "messages": to_openai(messages), **self.extra, **extra}
        if tools:
            body["tools"] = tool_schemas_openai(tools)
            body["tool_choice"] = "auto"
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            self.last_headers = dict(r.headers)
            data = json.loads(r.read())
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": fn.get("arguments")}
            calls.append(ToolCall(fn["name"], args, tc.get("id", "")))
        u = data.get("usage") or {}
        usage = {"prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
                 "cached_tokens": ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))}
        return Response(text=msg.get("content"), tool_calls=calls, usage=usage)


def get_model():
    """``MEMLAB_LLM_URL`` set: a real OpenAI-compatible model (T1). Otherwise the scripted model (T0)."""
    url = os.environ.get("MEMLAB_LLM_URL")
    if url:
        return ChatClient(url, os.environ.get("MEMLAB_LLM_MODEL"), os.environ.get("MEMLAB_API_KEY"))
    return ScriptedModel()
