"""loop.py — the 07.1 agent loop, with the three things a code tool adds: tiers, budgets, keys.

One idea: the loop is the one from ``agent-core`` (07.1: model -> tool calls -> results -> model,
until a final answer or the step budget), and the tool contract is the same (``{"ok": True,
"data": ...}`` or ``{"ok": False, "error": kind, "message", "hint"}``). What a sandboxed tool adds
lives *outside the model*, in the loop:

* **Tiers, deny by default** (identity primer §4.2): each tool has a tier — ``read``, ``write``,
  ``destructive``, ``external`` — and the loop only runs tiers the deployment allows. An
  execute-code tool is destructive by definition (identity primer §6.2).
* **Turn budgets**: tool calls, ``run_code`` calls and sandbox CPU seconds per turn. A hijacked
  model that loops is stopped by arithmetic, not by its own judgement (ASI08).
* **Idempotency keys**: ``turn:step:call-index:args-hash`` (scaling primer §5.4), handed to the
  tool so a redelivered turn replays results instead of re-running code.

Every decision and every result becomes an ``AuditEvent``. The model here is scripted (no weights,
no network): ``ScriptedLLM`` re-implements agent-core's ``FakeLLM`` shapes.
"""
from __future__ import annotations

import itertools
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..audit import AuditEvent, AuditLog, args_digest

TIERS = ("read", "write", "destructive", "external")
_ids = itertools.count(1)


# ---- the model's side (agent-core's shapes) -----------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"call_{next(_ids)}"


@dataclass
class Response:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def as_message(self) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in self.tool_calls]
        return msg


def call(name: str, **args) -> ToolCall:
    return ToolCall(name, args)


def text(t: str) -> Response:
    return Response(text=t)


def _coerce(x) -> Response:
    if isinstance(x, Response):
        return x
    if isinstance(x, str):
        return text(x)
    if isinstance(x, ToolCall):
        return Response(tool_calls=[x])
    if isinstance(x, list):
        return Response(tool_calls=list(x))
    raise TypeError(f"cannot turn {x!r} into a Response")


class ScriptedLLM:
    """A list of turns, or a policy ``(messages, tools) -> Response | str | ToolCall(s)``."""

    def __init__(self, responses: list | None = None, policy: Callable | None = None):
        self.responses = list(responses) if responses is not None else None
        self.policy = policy
        self.calls = 0

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> Response:
        self.calls += 1
        if self.responses is not None:
            if self.calls > len(self.responses):
                raise RuntimeError(f"ScriptedLLM ran out of turns at call {self.calls}")
            return _coerce(self.responses[self.calls - 1])
        return _coerce(self.policy(messages, tools))


# ---- the harness's side --------------------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                                  # JSON-schema "properties"
    fn: Callable[[dict, "Context"], dict]             # returns the 07.1 result shape
    tier: str = "read"
    required: list[str] = field(default_factory=list)

    @property
    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.parameters, "required": self.required}}


@dataclass
class Context:
    """What a tool gets besides its arguments: who is calling, the key for this call, the log."""
    agent: str
    session_id: str
    invocation_id: str
    idempotency_key: str
    audit: AuditLog
    budget: "TurnBudget"


@dataclass
class TurnBudget:
    max_steps: int = 6
    max_tool_calls: int = 8
    max_run_code: int = 3
    max_sandbox_cpu_s: float = 10.0
    # spent so far this turn
    tool_calls: int = 0
    run_code: int = 0
    sandbox_cpu_s: float = 0.0

    def check(self, tool: Tool) -> str | None:
        if self.tool_calls >= self.max_tool_calls:
            return f"tool-call budget exhausted ({self.max_tool_calls} per turn)"
        if tool.name == "run_code" and self.run_code >= self.max_run_code:
            return f"run_code budget exhausted ({self.max_run_code} per turn)"
        if tool.name == "run_code" and self.sandbox_cpu_s >= self.max_sandbox_cpu_s:
            return f"sandbox CPU budget exhausted ({self.max_sandbox_cpu_s} s per turn)"
        return None


def idempotency_key(turn: int, step: int, index: int, name: str, args: dict) -> str:
    """Scaling primer §5.4: key every side effect by turn, step, call index and a hash of the arguments."""
    return f"turn{turn}:step{step}:call{index}:{name}:{args_digest(args)}"


def tool_message(tc: ToolCall, result: dict) -> dict:
    return {"role": "tool", "name": tc.name, "tool_call_id": tc.id, "content": json.dumps(result, default=str)}


@dataclass
class Result:
    text: str
    messages: list[dict]
    steps: int
    done: bool = True
    stopped_by: str | None = None

    def transcript(self, width: int = 160) -> str:
        lines = []
        for m in self.messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    lines.append(f"  model -> {tc['name']}({json.dumps(tc['args'])[:width]})")
            elif m["role"] == "tool":
                lines.append(f"  tool  <- {m['name']}: {m['content'][:width]}")
            elif m["role"] != "system":
                lines.append(f"{m['role']}: {m['content'][:width]}")
        return "\n".join(lines)


class Agent:
    def __init__(self, llm, tools: list[Tool], *, allowed_tiers: tuple[str, ...] = ("read", "external", "destructive"),
                 budget: TurnBudget | None = None, audit: AuditLog | None = None, agent_id: str = "sandbox-agent",
                 session_id: str | None = None, instruction: str = "You are a data assistant."):
        self.llm, self.tools = llm, {t.name: t for t in tools}
        self.allowed_tiers = allowed_tiers
        self.budget_template = budget or TurnBudget()
        self.audit = audit or AuditLog()
        self.agent_id = agent_id
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.instruction = instruction
        self.turn = 0

    def _decide(self, tool: Tool | None, tc: ToolCall, budget: TurnBudget) -> tuple[bool, dict | None, list[str]]:
        if tool is None:
            return False, {"ok": False, "error": "unknown_tool", "message": f"no tool named {tc.name!r}",
                           "hint": f"available tools: {', '.join(self.tools)}"}, ["unknown tool"]
        if tool.tier not in self.allowed_tiers:
            return False, {"ok": False, "error": "denied", "message": f"{tool.name} is {tool.tier}-tier; not allowed here"}, \
                [f"tier {tool.tier} not allowed (deny by default)"]
        missing = [p for p in tool.required if p not in tc.args]
        if missing:
            return False, {"ok": False, "error": "invalid_arguments", "message": f"missing {missing}"}, ["invalid arguments"]
        why = budget.check(tool)
        if why:
            return False, {"ok": False, "error": "budget_exceeded", "message": why,
                           "hint": "Answer with what you have."}, [why]
        return True, None, ["allowed"]

    def run(self, user_message: str) -> Result:
        self.turn += 1
        budget = TurnBudget(**{k: getattr(self.budget_template, k) for k in
                               ("max_steps", "max_tool_calls", "max_run_code", "max_sandbox_cpu_s")})
        messages = [{"role": "system", "content": self.instruction}, {"role": "user", "content": user_message}]
        schemas = [t.schema for t in self.tools.values()]
        for step in range(1, budget.max_steps + 1):
            resp = self.llm.generate(messages, schemas)
            messages.append(resp.as_message())
            if not resp.tool_calls:
                return Result(resp.text or "", messages, step)
            for i, tc in enumerate(resp.tool_calls):
                tool = self.tools.get(tc.name)
                inv = uuid.uuid4().hex[:12]
                ok, result, reasons = self._decide(tool, tc, budget)
                self.audit.emit(AuditEvent("tool.decision", self.agent_id, "own", tool=tc.name,
                                           decision="allow" if ok else "deny", reasons=reasons,
                                           args_hash=args_digest(tc.args), session_id=self.session_id,
                                           invocation_id=inv, policy_decision="allow" if ok else "deny"))
                if ok:
                    budget.tool_calls += 1
                    budget.run_code += tool.name == "run_code"
                    key = idempotency_key(self.turn, step, i, tc.name, tc.args)
                    ctx = Context(self.agent_id, self.session_id, inv, key, self.audit, budget)
                    try:
                        result = tool.fn(tc.args, ctx)
                    except Exception as e:                       # the boundary: never a traceback to the model
                        result = {"ok": False, "error": "tool_failure", "message": f"{type(e).__name__}: {e}"}
                    self.audit.emit(AuditEvent("tool.result", self.agent_id, "own", tool=tc.name,
                                               decision="ok" if result.get("ok") else "blocked",
                                               reasons=[result.get("error", "ok")], session_id=self.session_id,
                                               invocation_id=inv, idempotency_key=key,
                                               result_hash=args_digest(result)))
                messages.append(tool_message(tc, result))
        return Result("(stopped: step budget reached)", messages, budget.max_steps, done=False, stopped_by="max_steps")
