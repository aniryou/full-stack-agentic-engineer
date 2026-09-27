"""ADK 2 Workflow, ticket-queue scenario (examples/adk_ticket_queue): interrupts, resume, rerun_on_resume
semantics, staleness routing, and the notebook 04 grader. Offline (no model); skipped without the adk extra."""

import asyncio
import sys
import warnings
from pathlib import Path

import pytest

pytest.importorskip("google.adk.workflow", reason='pip install -e ".[adk]" installs ADK 2')
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "adk_ticket_queue"))

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402

from nightly_workflow import (  # noqa: E402
    ASK_BUDGET, WAKE, Venue, build_app, build_workflow, check_adk_workflow, demo, run_until_interrupt,
)


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


def test_the_notebook_grader_passes_the_reference_graph_and_catches_a_node_that_does_not_recheck():
    assert check_adk_workflow(build_workflow) == "adk workflow ok"

    from google.adk.events.request_input import RequestInput
    from google.adk.workflow import Workflow, node

    def no_recheck(venue):                           # check_front with rerun_on_resume=False: the wake-up is its answer
        ref = build_workflow(venue)
        (chain,) = ref.edges
        nodes = list(chain)

        @node(rerun_on_resume=False)
        def check_front(ctx):
            pos = venue.position(ctx.state["ticket"])
            if pos > 0:
                return RequestInput(interrupt_id=WAKE, message=f"Still at #{pos}")
            ctx.route = "ready" if venue.seats_left(ctx.state["event_id"]) >= 2 else "sold_out"
            return {"position": pos}

        nodes[4] = check_front
        nodes[5] = dict(nodes[5])
        return Workflow(name="wrong", edges=[tuple(nodes)])

    with pytest.raises(AssertionError):
        check_adk_workflow(no_recheck)


def test_rerun_on_resume_only_matters_for_the_node_that_interrupted():
    """queue_up finished before the interrupt; resuming the same invocation never replays it, whatever its flag."""
    from google.adk.workflow import Workflow, node

    def rerun_flagged_queue(venue):
        ref = build_workflow(venue)
        (chain,) = ref.edges
        nodes = list(chain)

        @node(rerun_on_resume=True)
        def queue_up(ctx):
            t = venue.join_queue(ctx.state["event_id"], idempotency_key=f"{ctx.session.id}:{ctx.state['event_id']}")
            ctx.state["ticket"] = t["ticket"]
            return t

        nodes[3] = queue_up
        return Workflow(name="flagged", edges=[tuple(nodes)])

    assert check_adk_workflow(rerun_flagged_queue) == "adk workflow ok"
