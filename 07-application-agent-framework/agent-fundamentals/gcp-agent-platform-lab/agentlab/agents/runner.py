"""The Runner: loads a session, runs the root agent, persists, and resumes after approvals.

Pause/resume is a state-machine transition, not a modal dialog: the session is
saved as ``awaiting_approval`` with the pending call; ``approve`` records the
decision and continues the same step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from .budget import Budget, BudgetExceeded
from .loop import BaseAgent, InvocationContext, Paused
from .state import Event, InMemorySessionStore, Session, SessionStatus, SessionStore, VersionConflict
from .tools import Identity


@dataclass
class RunResult:
    session: Session
    events: list[Event]
    paused: bool = False
    pending: dict[str, Any] | None = None
    error: str | None = None

    @property
    def text(self) -> str | None:
        return self.session.last_final_text()


@dataclass
class Runner:
    agent: BaseAgent
    store: SessionStore = field(default_factory=InMemorySessionStore)
    tracer: Any = None
    budget_factory: Any = Budget          # callable returning a fresh Budget per turn
    tool_timeout_s: float = 10.0
    model_timeout_s: float = 30.0

    def _ctx(self, session: Session, user: Identity | None) -> InvocationContext:
        return InvocationContext(session=session, budget=self.budget_factory(), user=user, tracer=self.tracer,
                                 confirm=None, tool_timeout_s=self.tool_timeout_s, model_timeout_s=self.model_timeout_s)

    async def _drive(self, session: Session, ctx: InvocationContext) -> RunResult:
        events: list[Event] = []
        try:
            async for ev in self.agent.run(ctx):
                events.append(ev)
            session.status = SessionStatus.ACTIVE
            session.pending = None
            session.clear_temp_state()
            self.store.put(session)
            return RunResult(session=session, events=events)
        except Paused as p:
            self.store.put(session)   # status/pending were set by the agent
            return RunResult(session=session, events=events, paused=True, pending=session.pending)
        except BudgetExceeded as e:
            session.append(Event(kind="error", payload={"error": str(e)}))
            session.status = SessionStatus.FAILED
            self.store.put(session)
            return RunResult(session=session, events=events, error=str(e))
        except VersionConflict:
            raise
        except Exception as e:  # noqa: BLE001
            session.append(Event(kind="error", payload={"error": f"{type(e).__name__}: {e}"}))
            session.status = SessionStatus.FAILED
            self.store.put(session)
            return RunResult(session=session, events=events, error=f"{type(e).__name__}: {e}")

    async def run(self, session_id: str, message: str, user: Identity | None = None, tenant: str = "default") -> RunResult:
        session = self.store.get_or_create(session_id, tenant=tenant, user=user.subject if user else "anonymous")
        if session.status == SessionStatus.AWAITING_APPROVAL:
            raise RuntimeError(f"session {session_id} is awaiting approval; call approve() first")
        session.append(Event(kind="user", payload={"content": message}))
        return await self._drive(session, self._ctx(session, user))

    async def approve(self, session_id: str, approved: bool, user: Identity | None = None, by: str = "human") -> RunResult:
        session = self.store.get(session_id)
        if session.status != SessionStatus.AWAITING_APPROVAL or not session.pending:
            raise RuntimeError(f"session {session_id} has nothing to approve")
        pending = session.pending
        session.append(Event(kind="approval", agent=pending.get("agent"), payload={"id": pending["tool_call"]["id"], "approved": approved, "by": by}))
        session.status = SessionStatus.ACTIVE
        session.pending = None
        ctx = self._ctx(session, user)
        if approved:
            # execute the approved call on the agent that asked, then let the loop continue from the log
            target = self.agent.find(pending.get("agent", "")) or self.agent
            if not hasattr(target, "execute_pending"):
                raise RuntimeError(f"agent {target.name} cannot resume a pending tool call")
            await target.execute_pending(ctx, pending)  # type: ignore[attr-defined]
        else:
            # record a declined result so the model can react on the next step
            from ..llm.types import ToolCall
            from .tools import ToolResult
            tc = ToolCall(name=pending["tool_call"]["name"], args=pending["tool_call"]["args"], id=pending["tool_call"]["id"])
            res = ToolResult.failure("declined", "the user declined this action", retryable=False, hint="Acknowledge and offer alternatives.")
            session.append(Event(kind="tool_result", agent=pending.get("agent"), step=pending.get("step"),
                                 payload={"id": tc.id, "name": tc.name, "ok": False, "error": "declined", "content": res.to_content()}))
        return await self._drive(session, ctx)

    async def stream(self, session_id: str, message: str, user: Identity | None = None, tenant: str = "default") -> AsyncIterator[Event]:
        """Like ``run`` but yields events as they happen (what an SSE endpoint would forward)."""
        session = self.store.get_or_create(session_id, tenant=tenant, user=user.subject if user else "anonymous")
        session.append(Event(kind="user", payload={"content": message}))
        ctx = self._ctx(session, user)
        try:
            async for ev in self.agent.run(ctx):
                yield ev
            session.status = SessionStatus.ACTIVE
            session.pending = None
        except Paused:
            pass
        except BudgetExceeded as e:
            yield session.append(Event(kind="error", payload={"error": str(e)}))
            session.status = SessionStatus.FAILED
        finally:
            self.store.put(session)
