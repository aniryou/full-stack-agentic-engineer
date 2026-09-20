"""ADK 2 Workflow: interrupts, resume, rerun_on_resume semantics, staleness routing. Offline (no model)."""

import asyncio
import warnings

import pytest

warnings.filterwarnings("ignore")

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from lragents.adk.nightly_workflow import ASK_BUDGET, WAKE, Venue, build_app, demo, run_until_interrupt


def test_end_to_end_demo_takes_one_ticket_and_one_order():
    out = asyncio.run(demo())
    assert out["venue_calls"] == ["join_queue", "purchase"]      # 4 wake-ups, side effects once each
    assert len(out["orders"]) == 1 and out["state"]["order"]["replayed"] is False
    assert out["last"]["interrupts"] == []


def test_sold_out_while_waiting_routes_to_abandon():
    async def go():
        venue = Venue()
        svc = InMemorySessionService()
        runner = Runner(app=build_app(venue), session_service=svc)
        sess = await svc.create_session(app_name="nightly_app", user_id="u1", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
        r1 = await run_until_interrupt(runner, "u1", sess.id, text="go")
        r2 = await run_until_interrupt(runner, "u1", sess.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
        assert r2["interrupts"] == [WAKE]
        ticket = (await svc.get_session(app_name="nightly_app", user_id="u1", session_id=sess.id)).state["ticket"]
        venue.advance(ticket, 99_999)
        venue.sell_out("ams-sat")                                  # the world changed while we were parked
        await run_until_interrupt(runner, "u1", sess.id, invocation_id=r2["invocation_id"], answers={WAKE: {"ok": True}})
        final = await svc.get_session(app_name="nightly_app", user_id="u1", session_id=sess.id)
        return venue, final
    venue, final = asyncio.run(go())
    assert venue.orders == {} and final.state["order"] is None
    assert "purchase" not in venue.calls


def test_new_invocation_instead_of_resume_takes_a_second_ticket():
    """The classic mistake: typing in the chat box instead of answering the interrupt
    starts a NEW invocation → the graph replays from START → second queue ticket."""
    async def go():
        venue = Venue()
        svc = InMemorySessionService()
        runner = Runner(app=build_app(venue), session_service=svc)
        sess = await svc.create_session(app_name="nightly_app", user_id="u1", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
        r1 = await run_until_interrupt(runner, "u1", sess.id, text="go")
        await run_until_interrupt(runner, "u1", sess.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
        # WRONG: a fresh invocation (no invocation_id) — re-asks the budget and would re-run queue_up after answering
        r3 = await run_until_interrupt(runner, "u1", sess.id, text="where am I?")
        await run_until_interrupt(runner, "u1", sess.id, invocation_id=r3["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
        return venue
    venue = asyncio.run(go())
    assert venue.calls.count("join_queue") == 2
