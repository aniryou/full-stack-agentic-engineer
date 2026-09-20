"""Solutions — Practice 04 (ADK 2 Workflow)."""

from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, Workflow, node

from lragents.adk.nightly_workflow import ASK_BUDGET, WAKE


def build_workflow(venue):
    @node(rerun_on_resume=True)
    def agree_budget(ctx):
        said = ((ctx.resume_inputs or {}).get(ASK_BUDGET) or {}).get("budget")
        if not said:
            return RequestInput(interrupt_id=ASK_BUDGET, message="What are you willing to spend per seat?")
        ctx.state["budget_per_seat"] = float(said)
        return {"budget_per_seat": float(said)}

    @node
    def plan(ctx):
        ctx.state["event_id"] = ctx.state["events"][0]["id"]
        return {"event_id": ctx.state["event_id"]}

    @node(rerun_on_resume=False)                              # side effect: must not repeat on resume
    def queue_up(ctx):
        t = venue.join_queue(ctx.state["event_id"], idempotency_key=f"{ctx.session.id}:{ctx.state['event_id']}")
        ctx.state["ticket"] = t["ticket"]
        return t

    @node(rerun_on_resume=True)                               # re-check the world every wake-up
    def check_front(ctx):
        pos = venue.position(ctx.state["ticket"])
        if pos > 0:
            return RequestInput(interrupt_id=WAKE, message=f"Still at #{pos}")
        left = venue.seats_left(ctx.state["event_id"])
        ctx.route = "ready" if left >= 2 else "sold_out"
        return {"position": pos, "seats_left": left}

    @node
    def buy(ctx):
        order = venue.purchase(ctx.state["event_id"], 2, idempotency_key=f"{ctx.session.id}:{ctx.state['event_id']}:purchase")
        ctx.state["order"] = order
        return order

    @node
    def abandon(ctx):
        ctx.state["order"] = None
        return {"abandoned": True}

    return Workflow(name="practice", edges=[(START, agree_budget, plan, queue_up, check_front, {"ready": buy, "sold_out": abandon})])
