"""ADK 2 ``Workflow`` graph for a long-running task — runnable offline.

This is the same shape as Google's "long-running agents" codelab (a ticket
queue that takes 40 minutes) expressed with ADK 2 primitives:

    START → agree_budget → plan → queue_up → check_front ─┬─ "ready"    → buy
                 (HITL)     (judgement)  (rule)   (wait)   └─ "sold_out" → abandon

* ``agree_budget`` and ``check_front`` return ``RequestInput`` — an *interrupt*.
  The invocation parks in the session store; the process can exit. Anything can
  answer it later: a person (approval) or a clock (Cloud Scheduler). Same primitive.
* ``rerun_on_resume`` matters only for the node that *interrupted*: ``True`` re-runs
  it with the resume input (``agree_budget`` reads the answer, ``check_front``
  re-checks the world); ``False`` (the default) marks it complete and uses the
  resume input as its output. Nodes that finished before the interrupt, such as
  ``queue_up``, are never replayed by a resume of the same invocation, whatever
  their flag; a *new* invocation replays them (the classic mistake, tested below),
  and that is how a second queue ticket happens.
* ``ResumabilityConfig(is_resumable=True)`` on the ``App`` lets a run resume from
  its last event after a crash. ADK's own docstring: resume is at-least-once,
  so tools must be idempotent, and ``temp:`` state does not survive.
* The purchase carries an idempotency key derived from stable ids, so a
  replayed ``buy`` is harmless.

``plan`` is a plain function by default so the graph runs with no model at all;
set ``use_model=True`` to swap in an ``LlmAgent`` node (needs Vertex AI credentials).

Optional: ``pip install -e ".[adk]"`` (ADK 2 and the Google Cloud clients, about 220 MB; measured
2026-09-26, verify). Run it: ``python examples/adk_ticket_queue/nightly_workflow.py``; notebook 04
walks it and ``check_adk_workflow`` grades the notebook's exercise.
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.request_input import RequestInput
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow, node
from google.genai import types

ASK_BUDGET = "budget"
WAKE = "wake"


class Venue:
    """Fake external system with a queue that advances on demand and an idempotent purchase."""

    def __init__(self) -> None:
        self.tickets: dict[str, int] = {}
        self.orders: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.available: dict[str, int] = {}

    def join_queue(self, event_id: str, idempotency_key: str) -> dict[str, Any]:
        self.calls.append("join_queue")
        ticket = f"q_{idempotency_key}"
        self.tickets.setdefault(ticket, 14203)
        return {"ticket": ticket, "position": self.tickets[ticket], "event_id": event_id}

    def advance(self, ticket: str, by: int) -> None:
        self.tickets[ticket] = max(0, self.tickets[ticket] - by)

    def position(self, ticket: str) -> int:
        return self.tickets[ticket]

    def seats_left(self, event_id: str) -> int:
        return self.available.get(event_id, 100)

    def sell_out(self, event_id: str) -> None:
        self.available[event_id] = 0

    def purchase(self, event_id: str, seats: int, idempotency_key: str) -> dict[str, Any]:
        self.calls.append("purchase")
        if idempotency_key in self.orders:
            return {**self.orders[idempotency_key], "replayed": True}
        order = {"order_id": f"ord_{len(self.orders) + 1}", "event_id": event_id, "seats": seats}
        self.orders[idempotency_key] = order
        self.available[event_id] = self.seats_left(event_id) - seats
        return {**order, "replayed": False}


def build_workflow(venue: Venue, use_model: bool = False, model: str = "gemini-3.8-flash") -> Workflow:
    @node(rerun_on_resume=True)                              # must re-run: it is waiting on a person
    def agree_budget(ctx):
        answers = ctx.resume_inputs or {}
        said = (answers.get(ASK_BUDGET) or {}).get("budget")
        if not said:
            return RequestInput(interrupt_id=ASK_BUDGET, message="What are you willing to spend per seat?")
        ctx.state["budget_per_seat"] = float(said)
        return {"budget_per_seat": float(said)}

    if use_model:
        plan_node = LlmAgent(
            name="plan", model=model, include_contents="none",
            instruction="Pick the event id from state['events'] that best fits the user's prefs. Reply with only the id.",
            output_key="event_id",
        )
    else:
        @node
        def plan(ctx):                                       # judgement, stubbed deterministically
            events = ctx.state.get("events", [{"id": "ams-sat", "weekday": "Saturday"}])
            weekend = [e for e in events if e.get("weekday") in ("Saturday", "Sunday")]
            ctx.state["event_id"] = (weekend or events)[0]["id"]
            return {"event_id": ctx.state["event_id"]}
        plan_node = plan

    @node                                                    # a rule and a side effect: done once, before any interrupt
    def queue_up(ctx):
        key = f"{ctx.session.id}:{ctx.state['event_id']}"
        t = venue.join_queue(ctx.state["event_id"], idempotency_key=key)
        ctx.state["ticket"] = t["ticket"]
        return t

    @node(rerun_on_resume=True)                              # must re-run: it re-checks the world
    def check_front(ctx):
        pos = venue.position(ctx.state["ticket"])
        if pos > 0:                                          # not our turn: park again (costs nothing while parked)
            return RequestInput(interrupt_id=WAKE, message=f"Still at #{pos}. Wake me and I'll check.")
        left = venue.seats_left(ctx.state["event_id"])       # staleness guard: re-verify before acting
        ctx.route = "ready" if left >= 2 else "sold_out"
        return {"position": pos, "seats_left": left}

    @node
    def buy(ctx):
        key = f"{ctx.session.id}:{ctx.state['event_id']}:purchase"
        order = venue.purchase(ctx.state["event_id"], seats=2, idempotency_key=key)
        ctx.state["order"] = order
        return order

    @node
    def abandon(ctx):
        ctx.state["order"] = None
        return {"abandoned": True, "reason": "sold out while queueing"}

    return Workflow(
        name="nightly",
        edges=[(START, agree_budget, plan_node, queue_up, check_front, {"ready": buy, "sold_out": abandon})],
    )


def build_app(venue: Venue, use_model: bool = False) -> App:
    return App(name="nightly_app", root_agent=build_workflow(venue, use_model),
               resumability_config=ResumabilityConfig(is_resumable=True))


# ---------------------------------------------------------------- drivers
async def run_until_interrupt(runner: Runner, user_id: str, session_id: str, text: str | None = None,
                              invocation_id: str | None = None, answers: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Start (text) or resume (invocation_id + answers) a run; return what it stopped on."""
    if answers:
        parts = [types.Part(function_response=types.FunctionResponse(id=k, name="adk_request_input", response=v))
                 for k, v in answers.items()]
        msg = types.Content(role="user", parts=parts)
    else:
        msg = types.Content(role="user", parts=[types.Part(text=text or "go")])
    interrupts: list[str] = []
    inv = invocation_id
    async for ev in runner.run_async(user_id=user_id, session_id=session_id, invocation_id=invocation_id, new_message=msg):
        inv = ev.invocation_id
        for p in (ev.content.parts if ev.content else []) or []:
            if p.function_call and p.function_call.name == "adk_request_input":
                interrupts.append(p.function_call.id)
    return {"invocation_id": inv, "interrupts": interrupts}


async def demo(venue: Venue | None = None) -> dict[str, Any]:
    """Offline end-to-end: ask budget → answer → park in queue → wake twice → buy."""
    venue = venue or Venue()
    svc = InMemorySessionService()
    runner = Runner(app=build_app(venue), session_service=svc)
    sess = await svc.create_session(app_name="nightly_app", user_id="u1",
                                    state={"events": [{"id": "ams-tue", "weekday": "Tuesday"}, {"id": "ams-sat", "weekday": "Saturday"}]})
    r1 = await run_until_interrupt(runner, "u1", sess.id, text="Get us two tickets")
    assert r1["interrupts"] == [ASK_BUDGET]
    r2 = await run_until_interrupt(runner, "u1", sess.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 250}})
    assert r2["interrupts"] == [WAKE]
    ticket = (await svc.get_session(app_name="nightly_app", user_id="u1", session_id=sess.id)).state["ticket"]
    venue.advance(ticket, 10_000)                               # the world moves while nothing of ours runs
    r3 = await run_until_interrupt(runner, "u1", sess.id, invocation_id=r2["invocation_id"], answers={WAKE: {"ok": True}})
    assert r3["interrupts"] == [WAKE]                           # still not at the front → parks again
    venue.advance(ticket, 10_000)
    r4 = await run_until_interrupt(runner, "u1", sess.id, invocation_id=r3["invocation_id"], answers={WAKE: {"ok": True}})
    final = await svc.get_session(app_name="nightly_app", user_id="u1", session_id=sess.id)
    return {"state": final.state, "venue_calls": venue.calls, "orders": venue.orders, "last": r4}


def _run(coro: Any) -> Any:
    """Run a coroutine to completion, also from inside Jupyter (whose event loop is already running)."""
    import threading

    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def check_adk_workflow(build_workflow_fn: Any) -> str:
    """Grade a ``build_workflow(venue)`` written in notebook 04: interrupts, one ticket across wake-ups, one order."""

    async def go() -> str:
        venue = Venue()
        wf = build_workflow_fn(venue)
        assert getattr(wf, "edges", None), "the Workflow has no edges: wire START -> agree_budget -> ... in the edges list"
        app = App(name="p", root_agent=wf, resumability_config=ResumabilityConfig(is_resumable=True))
        svc = InMemorySessionService()
        runner = Runner(app=app, session_service=svc)
        sess = await svc.create_session(app_name="p", user_id="u", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
        r1 = await run_until_interrupt(runner, "u", sess.id, text="go")
        assert r1["interrupts"] == [ASK_BUDGET], f"first stop must be the budget gate, got {r1['interrupts']}"
        r2 = await run_until_interrupt(runner, "u", sess.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
        assert r2["interrupts"] == [WAKE], f"after the budget the run must park on the queue, got {r2['interrupts']}"
        st = (await svc.get_session(app_name="p", user_id="u", session_id=sess.id)).state
        venue.advance(st["ticket"], 5000)
        r3 = await run_until_interrupt(runner, "u", sess.id, invocation_id=r2["invocation_id"], answers={WAKE: {"ok": True}})
        assert r3["interrupts"] == [WAKE], "still queued: the run must park again"
        assert venue.calls.count("join_queue") == 1, "queue_up ran again: resume the same invocation, never start a new one"
        venue.advance(st["ticket"], 99_999)
        await run_until_interrupt(runner, "u", sess.id, invocation_id=r3["invocation_id"], answers={WAKE: {"ok": True}})
        assert len(venue.orders) == 1, "exactly one purchase expected"
        assert venue.calls == ["join_queue", "purchase"], venue.calls
        return "adk workflow ok"

    return _run(go())


if __name__ == "__main__":
    out = asyncio.run(demo())
    print(out["venue_calls"], out["orders"])
