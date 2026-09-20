"""Workflow agents: deterministic control flow around agents (Primer §2.2).

Use code where the logic is known; use the model where judgement is needed.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from .loop import BaseAgent, InvocationContext
from .state import Event, Session


class SequentialAgent(BaseAgent):
    """Runs sub-agents in order on the same session; each can read the others' ``output_key`` state."""

    def __init__(self, name: str, sub_agents: list[BaseAgent], description: str = ""):
        self.name, self.sub_agents, self.description = name, sub_agents, description

    async def run(self, ctx: InvocationContext) -> AsyncIterator[Event]:  # type: ignore[override]
        for agent in self.sub_agents:
            async for ev in agent.run(ctx):
                yield ev


class ParallelAgent(BaseAgent):
    """Runs sub-agents concurrently on branch copies of the session, then merges state.

    Branches never mutate the same session object; results come back through
    ``state`` (each sub-agent should set an ``output_key``) and a ``note`` event.
    """

    def __init__(self, name: str, sub_agents: list[BaseAgent], description: str = ""):
        self.name, self.sub_agents, self.description = name, sub_agents, description

    async def run(self, ctx: InvocationContext) -> AsyncIterator[Event]:  # type: ignore[override]
        branches = [ctx.session.branch(a.name) for a in self.sub_agents]

        async def run_branch(agent: BaseAgent, session: Session) -> list[Event]:
            child = ctx.child(session)
            child.depth = ctx.depth  # a branch is not a delegation level
            return [ev async for ev in agent.run(child)]

        results = await asyncio.gather(*(run_branch(a, s) for a, s in zip(self.sub_agents, branches)), return_exceptions=True)
        merged: dict[str, str] = {}
        for agent, session, res in zip(self.sub_agents, branches, results):
            if isinstance(res, BaseException):
                yield ctx.session.append(Event(kind="error", agent=agent.name, payload={"error": f"{type(res).__name__}: {res}"}))
                continue
            for ev in res:
                # keep the branch's events in the parent log for tracing, tagged by agent
                ctx.session.events.append(ev)
                yield ev
            for k, v in session.state.items():
                if k not in ctx.session.state or ctx.session.state[k] != v:
                    ctx.session.state[k] = v
                    merged[k] = v
        yield ctx.session.append(Event(kind="note", agent=self.name, payload={"merged_state_keys": list(merged)}))


class LoopAgent(BaseAgent):
    """Repeats its sub-agents until ``until(session)`` is true or ``max_iterations`` is hit.

    The exit criterion lives in code, not in the prompt.
    """

    def __init__(self, name: str, sub_agents: list[BaseAgent], max_iterations: int = 3,
                 until: Callable[[Session], bool] | None = None, description: str = ""):
        self.name, self.sub_agents, self.max_iterations, self.until, self.description = name, sub_agents, max_iterations, until, description

    async def run(self, ctx: InvocationContext) -> AsyncIterator[Event]:  # type: ignore[override]
        for i in range(1, self.max_iterations + 1):
            ctx.session.set_state("temp:loop_iteration", i, agent=self.name)
            for agent in self.sub_agents:
                async for ev in agent.run(ctx):
                    yield ev
            if self.until is not None and self.until(ctx.session):
                yield ctx.session.append(Event(kind="note", agent=self.name, payload={"loop_exit": "condition met", "iterations": i}))
                return
        yield ctx.session.append(Event(kind="note", agent=self.name, payload={"loop_exit": "max_iterations", "iterations": self.max_iterations}))
