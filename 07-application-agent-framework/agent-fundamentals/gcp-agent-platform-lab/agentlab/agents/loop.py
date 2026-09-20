"""The agent loop (Primer §2.1) and delegation (§2.3).

``LlmAgent.run`` is the whole ReAct loop in one place: build context → call the
model → validate and execute tool calls (in parallel, under a semaphore, with
per-call timeouts) → append structured results → repeat until a final answer
or a budget stops it. Confirmation-required tools pause the loop for a human.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from ..llm.types import LLM, ModelResponse, ToolCall
from .budget import Budget, BudgetExceeded
from .context import ContextBuilder
from .state import Event, Session, SessionStatus
from .tools import Identity, Tool, ToolContext, ToolRegistry, ToolResult, ToolSpec

ConfirmHook = Callable[[ToolCall, ToolSpec], Awaitable[bool]]


class Paused(Exception):
    """The loop stopped to wait for a human decision. The session records why."""

    def __init__(self, tool_call: ToolCall, agent: str):
        super().__init__(f"{agent} needs approval for {tool_call.name}")
        self.tool_call = tool_call
        self.agent = agent


@dataclass
class InvocationContext:
    session: Session
    budget: Budget = field(default_factory=Budget)
    user: Identity | None = None
    depth: int = 0
    tracer: Any = None                          # agentlab.observability.tracing.Tracer, optional
    confirm: ConfirmHook | None = None          # if None, confirmation-required tools pause the run
    tool_concurrency: int = 8
    model_timeout_s: float = 30.0
    tool_timeout_s: float = 10.0
    _semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.tool_concurrency)
        return self._semaphore

    def child(self, session: Session) -> "InvocationContext":
        """A nested invocation shares the budget, identity and hooks."""
        return InvocationContext(session=session, budget=self.budget, user=self.user, depth=self.depth + 1,
                                 tracer=self.tracer, confirm=self.confirm, tool_concurrency=self.tool_concurrency,
                                 model_timeout_s=self.model_timeout_s, tool_timeout_s=self.tool_timeout_s,
                                 _semaphore=self.semaphore)


class BaseAgent(ABC):
    name: str = "agent"
    description: str = ""

    @abstractmethod
    def run(self, ctx: InvocationContext) -> AsyncIterator[Event]:
        """Yield events as the agent works. Must append the same events to ctx.session."""

    async def run_to_completion(self, ctx: InvocationContext) -> list[Event]:
        return [ev async for ev in self.run(ctx)]

    def find(self, name: str) -> "BaseAgent | None":
        """Locate an agent by name in this agent's tree (used to resume a paused sub-agent)."""
        if self.name == name:
            return self
        for child in getattr(self, "sub_agents", []) or []:
            found = child.find(name)
            if found is not None:
                return found
        return None


def _span(ctx: InvocationContext, name: str, **attrs: Any):
    """Open a tracing span if a tracer is attached; otherwise a no-op context."""
    if ctx.tracer is None:
        import contextlib
        return contextlib.nullcontext(None)
    return ctx.tracer.span(name, **attrs)


class LlmAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        llm: LLM,
        instruction: str,
        tools: list[Tool] | None = None,
        sub_agents: list[BaseAgent] | None = None,
        description: str = "",
        output_key: str | None = None,
        context: ContextBuilder | None = None,
        max_repeated_calls: int = 2,
        model_options: dict[str, Any] | None = None,
    ):
        self.name = name
        self.llm = llm
        self.instruction = instruction
        self.description = description or instruction[:120]
        self.output_key = output_key
        self.context = context or ContextBuilder(instruction=instruction)
        self.max_repeated_calls = max_repeated_calls
        self.model_options = model_options or {}
        self.registry = ToolRegistry(list(tools or []))
        self.sub_agents = list(sub_agents or [])
        for sa in self.sub_agents:
            self.registry.add(AgentTool(sa))

    # -- helpers -----------------------------------------------------------
    def _messages(self, session: Session):
        if self.context.instruction != self.instruction:
            self.context.instruction = self.instruction
        return self.context.build(session)

    def _idempotency_key(self, session: Session, step: int, tc: ToolCall) -> str:
        return f"{session.id}:{self.name}:{step}:{tc.signature()}"

    async def _execute(self, tc: ToolCall, tool: Tool, ctx: InvocationContext, step: int) -> ToolResult:
        tctx = ToolContext(session_id=ctx.session.id, tenant=ctx.session.tenant, user=ctx.user, agent_name=self.name,
                           idempotency_key=self._idempotency_key(ctx.session, step, tc), budget=ctx.budget, depth=ctx.depth,
                           extras={"invocation": ctx})
        async with ctx.semaphore:
            with _span(ctx, f"tool {tc.name}", kind="tool", **{"tool.name": tc.name, "agent": self.name}) as span:
                tctx.span = span
                try:
                    result = await asyncio.wait_for(tool.run(tc.args, tctx), timeout=ctx.tool_timeout_s)
                except asyncio.TimeoutError:
                    result = ToolResult.failure("timeout", f"{tc.name} exceeded {ctx.tool_timeout_s}s", retryable=True,
                                                hint="The system is slow; tell the user you will follow up if it fails again.")
                if span is not None:
                    span.set(**{"tool.ok": result.ok, "tool.latency_ms": round(result.latency_ms, 1),
                                "tool.error": result.error.type if result.error else None})
                return result

    # -- the loop ------------------------------------------------------------
    async def run(self, ctx: InvocationContext) -> AsyncIterator[Event]:  # type: ignore[override]
        session = ctx.session
        ctx.budget.check_depth(ctx.depth)
        seen: Counter[str] = Counter()
        schemas = self.registry.schemas()
        pause_for: ToolCall | None = None   # set when a confirmation-required call must wait for a human
        with _span(ctx, f"agent {self.name}", kind="agent", agent=self.name):
            step = 0
            while True:
                step += 1
                ctx.budget.consume_step()
                ctx.budget.check_time()
                messages = self._messages(session)
                with _span(ctx, "model.generate", kind="model", agent=self.name, **{"gen_ai.request.model": getattr(self.llm, "model_name", "?")}) as mspan:
                    t0 = time.perf_counter()
                    resp: ModelResponse = await asyncio.wait_for(
                        self.llm.generate(messages, tools=schemas or None, **self.model_options),
                        timeout=min(ctx.model_timeout_s, max(0.1, ctx.budget.remaining_seconds())),
                    )
                    wall_ms = (time.perf_counter() - t0) * 1000
                    if mspan is not None:
                        mspan.set(**{"gen_ai.usage.input_tokens": resp.usage.input_tokens,
                                     "gen_ai.usage.output_tokens": resp.usage.output_tokens,
                                     "gen_ai.usage.cached_tokens": resp.usage.cached_tokens,
                                     "gen_ai.response.finish_reason": resp.finish_reason,
                                     "model.latency_ms": round(resp.latency_ms, 1)})
                ctx.budget.consume_tokens(resp.usage.total)
                ev = session.append(Event(kind="model", agent=self.name, step=step, usage=resp.usage,
                                          latency_ms=resp.latency_ms or wall_ms, payload=resp.as_message()))
                yield ev

                if not resp.tool_calls:
                    text = resp.text or ""
                    if self.output_key:
                        session.set_state(self.output_key, text, agent=self.name)
                    fin = session.append(Event(kind="final", agent=self.name, step=step, payload={"text": text}))
                    yield fin
                    return

                # ---- tool calls -------------------------------------------
                to_run: list[tuple[ToolCall, Tool]] = []
                for tc in resp.tool_calls:
                    call_ev = session.append(Event(kind="tool_call", agent=self.name, step=step,
                                                   payload={"id": tc.id, "name": tc.name, "args": tc.args}))
                    yield call_ev
                    tool = self.registry.get(tc.name)
                    if tool is None:
                        res = ToolResult.failure("unknown_tool", f"no tool named {tc.name!r}", retryable=False,
                                                 hint=f"Available tools: {', '.join(self.registry.names())}")
                        yield self._record_result(session, step, tc, res)
                        continue
                    seen[tc.signature()] += 1
                    if seen[tc.signature()] > self.max_repeated_calls:
                        res = ToolResult.failure("duplicate_call", f"{tc.name} was already called with these arguments", retryable=False,
                                                 hint="You already have this result; use it or try a different approach.")
                        yield self._record_result(session, step, tc, res)
                        continue
                    if tool.spec.required_scope and (ctx.user is None or not ctx.user.has_scope(tool.spec.required_scope)):
                        # authorisation is decided before anyone is asked to confirm anything
                        res = ToolResult.failure("forbidden", f"{tc.name} requires scope {tool.spec.required_scope}", retryable=False,
                                                 hint="Tell the user this action is not permitted for their account and offer to raise a case.")
                        yield self._record_result(session, step, tc, res)
                        continue
                    if tool.spec.requires_confirmation:
                        if ctx.confirm is None:
                            # Pause for a human, but only after the rest of this batch has run:
                            # the other calls are independent and their results belong in the log.
                            if pause_for is None:
                                session.status = SessionStatus.AWAITING_APPROVAL
                                session.pending = {"agent": self.name, "step": step, "tool_call": {"id": tc.id, "name": tc.name, "args": tc.args}}
                                yield session.append(Event(kind="approval_required", agent=self.name, step=step, payload=dict(session.pending["tool_call"])))
                                pause_for = tc
                            else:
                                res = ToolResult.failure("deferred", "another action is already awaiting approval", retryable=False,
                                                         hint="Ask for one sensitive action at a time; request this again after the first is decided.")
                                yield self._record_result(session, step, tc, res)
                            continue
                        approved = await ctx.confirm(tc, tool.spec)
                        session.append(Event(kind="approval", agent=self.name, step=step, payload={"id": tc.id, "approved": approved}))
                        if not approved:
                            res = ToolResult.failure("declined", "the user declined this action", retryable=False,
                                                     hint="Acknowledge the decision and offer alternatives.")
                            yield self._record_result(session, step, tc, res)
                            continue
                    to_run.append((tc, tool))

                if to_run:
                    results = await asyncio.gather(*(self._execute(tc, tool, ctx, step) for tc, tool in to_run), return_exceptions=True)
                    for (tc, _tool), res in zip(to_run, results):
                        if isinstance(res, BaseException):
                            res = ToolResult.failure("tool_failure", f"{type(res).__name__}: {res}", retryable=False)
                        yield self._record_result(session, step, tc, res)
                if pause_for is not None:
                    break   # leave the span cleanly; the pause is control flow, not an error
        if pause_for is not None:
            raise Paused(pause_for, self.name)

    async def execute_pending(self, ctx: InvocationContext, pending: dict[str, Any]) -> Event:
        """Run a tool call that a human has just approved, and record its result.

        Called by the Runner on resume; the loop then continues from the log.
        """
        p = pending["tool_call"]
        tc = ToolCall(name=p["name"], args=p["args"], id=p["id"])
        step = int(pending.get("step") or 0)
        tool = self.registry.get(tc.name)
        if tool is None:
            res = ToolResult.failure("unknown_tool", f"no tool named {tc.name!r}")
        else:
            res = await self._execute(tc, tool, ctx, step)
        return self._record_result(ctx.session, step, tc, res)

    def _record_result(self, session: Session, step: int, tc: ToolCall, res: ToolResult) -> Event:
        return session.append(Event(kind="tool_result", agent=self.name, step=step, latency_ms=res.latency_ms,
                                    payload={"id": tc.id, "name": tc.name, "ok": res.ok,
                                             "error": res.error.type if res.error else None,
                                             "content": res.to_content(),
                                             "from_idempotency_cache": res.from_idempotency_cache}))


class AgentTool:
    """Expose an agent as a tool: hierarchical delegation (Primer §2.3).

    The sub-agent runs in its own child session so its transcript does not
    pollute the parent's context; only its final answer comes back. Two
    consequences worth knowing in a design discussion: the child shares the
    parent's budget (delegation cannot escape it), and a pause for human
    approval inside the child is reported to the parent as a structured
    ``needs_approval`` result rather than pausing the parent's turn — use a
    hand-off on the same session (see the capstone) when a specialist must
    own an approval.
    """

    def __init__(self, agent: BaseAgent, name: str | None = None, description: str | None = None):
        self.agent = agent
        self.spec = ToolSpec(
            name=name or agent.name,
            description=description or agent.description or f"Delegate to {agent.name}",
            input_schema={"type": "object", "properties": {"request": {"type": "string", "description": "What you need from this specialist"}}, "required": ["request"]},
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        inv: InvocationContext | None = ctx.extras.get("invocation")
        request = str((args or {}).get("request", ""))
        if not request:
            return ToolResult.failure("invalid_arguments", "request is required", hint="Say what you need from the specialist.")
        child_session = Session(id=f"{ctx.session_id}/{self.agent.name}", tenant=ctx.tenant, user=ctx.user.subject if ctx.user else "anonymous")
        child_session.append(Event(kind="user", payload={"content": request}))
        if inv is None:
            child_ctx = InvocationContext(session=child_session, user=ctx.user, depth=ctx.depth + 1)
        else:
            child_ctx = inv.child(child_session)
        t0 = time.perf_counter()
        try:
            async for _ in self.agent.run(child_ctx):
                pass
        except BudgetExceeded as e:
            return ToolResult.failure("budget_exceeded", str(e), retryable=False, hint="Answer with what you have.")
        except Paused as p:
            # A delegate cannot pause the parent's turn: its child session is not what the human resumes.
            # Designs that need approval inside a specialist use hand-off (same session) or a confirm hook.
            return ToolResult.failure("needs_approval", f"{self.agent.name} needs the user's confirmation for {p.tool_call.name}",
                                      retryable=False, hint="Ask the user to confirm, then call the confirmation-required tool directly.")
        answer = child_session.last_final_text() or ""
        if inv is not None:
            # carry the child's token usage so the parent's session.usage() counts delegated work
            inv.session.append(Event(kind="delegation", agent=self.agent.name, usage=child_session.usage(),
                                     payload={"request": request, "answer": answer, "child_session": child_session.id,
                                              "steps": sum(1 for e in child_session.events if e.kind == "model")}))
        return ToolResult.success({"answer": answer}, latency_ms=(time.perf_counter() - t0) * 1000)
