"""Research-and-publish pipeline: the reference long-running workflow.

Timeline of one run (wall-clock could be minutes or days):

    plan ──FanOut──▶ research_subtopic × N (child runs, parallel, retried)
      ▲                                       │
      └──── reaper respawns missing children ◀┘
    synthesize ─▶ reflect_critique ⇄ reflect_revise (bounded loop)
    ─▶ request_review  ……… WAITING (days) ……… ─▶ publish (effect + compensation)
    ─▶ notify (effect) ─▶ Done

Every pattern in docs/primer.md §2 appears here once, in the place a real
system would use it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from ..core.models import Budget
from ..core.workflow import Done, Next, StepContext, StepFailed, Workflow
from ..patterns.hitl import approval_decision, request_approval
from ..patterns.orchestrator_worker import child_outcomes, plan_and_fan_out
from ..patterns.reflection import install_reflection_steps
from ..patterns.saga import compensating

# ---------------------------------------------------------------------------
# Child workflow: one focused research task (cheap model, small budget)
# ---------------------------------------------------------------------------
research_subtopic = Workflow(
    "research_subtopic",
    version="1",
    default_budget=Budget(max_steps=5, max_tokens=50_000, max_cost_usd=0.25),
)


@research_subtopic.step(start=True, max_attempts=3, backoff_base_s=1.0)
def research(ctx: StepContext) -> Any:
    """Summarise a subtopic. Real version: web/RAG tool calls, all idempotent."""
    resp = ctx.llm(
        f"Research this subtopic and write 3 bullet findings with evidence.\n"
        f"TITLE: {ctx.input.get('title')}\nINSTRUCTIONS: {ctx.input.get('instructions')}"
    )
    return Done({"title": ctx.input.get("title"), "findings": resp.text})


# ---------------------------------------------------------------------------
# Parent workflow
# ---------------------------------------------------------------------------
research_pipeline = Workflow(
    "research_pipeline",
    version="1",
    default_budget=Budget(max_steps=40, max_tokens=400_000, max_cost_usd=3.0),
)


@research_pipeline.step(start=True)
def plan(ctx: StepContext) -> Any:
    goal = ctx.input["goal"]
    ctx.state["goal"] = goal
    return plan_and_fan_out(
        ctx,
        goal=goal,
        child_workflow="research_subtopic",
        then="synthesize",
        max_children=ctx.input.get("max_children", 4),
        child_budget=Budget(max_steps=5, max_tokens=50_000, max_cost_usd=0.25),
    )


@research_pipeline.step()
def synthesize(ctx: StepContext) -> Any:
    results, failures = child_outcomes(ctx)
    if len(results) < max(1, len(ctx.state.get("plan", [])) // 2):
        raise StepFailed(f"too many research failures: {failures}")  # business rule, no retry
    findings = "\n\n".join(f"## {r['title']}\n{r['findings']}" for r in results.values())
    resp = ctx.llm(f"Write a concise report that answers the GOAL using the FINDINGS.\nGOAL: {ctx.state['goal']}\n\nFINDINGS:\n{findings}")
    ctx.state["draft"] = resp.text
    ctx.state["partial_failures"] = failures
    return Next("reflect_critique")


install_reflection_steps(research_pipeline, prefix="reflect", draft_key="draft", goal_key="goal", exit_step="request_review", max_iterations=2, threshold=8)


@research_pipeline.step()
def request_review(ctx: StepContext) -> Any:
    return request_approval(
        ctx,
        then="publish",
        gate="editor",
        summary=ctx.state["draft"][:200],
        timeout=timedelta(days=3),
        on_timeout="fail",
    )


def _unpublish(ctx: StepContext, rec: dict[str, Any]) -> None:
    ctx.state["unpublished"] = rec["post_id"]  # real: DELETE /posts/{id}


@research_pipeline.step(compensate=compensating("publish", _unpublish))
def publish(ctx: StepContext) -> Any:
    decision = approval_decision(ctx, gate="editor")  # raises StepFailed on reject/timeout
    rec = ctx.effect("publish", lambda: {"post_id": f"post_{ctx.fingerprint(ctx.state['draft'])}"})
    ctx.state["post_id"] = rec["post_id"]
    ctx.state["approved_by"] = decision.get("by")
    return Next("notify")


@research_pipeline.step()
def notify(ctx: StepContext) -> Any:
    # Cannot be undone, so it runs last (after everything compensable).
    if ctx.input.get("fail_notify"):
        raise RuntimeError("SMTP down")  # exercised by the saga notebook
    ctx.effect("notify", lambda: {"sent_to": ctx.input.get("notify", "editors@example.com")})
    return Done({"post_id": ctx.state["post_id"], "iterations": ctx.state["reflect_loop"]["iteration"], "exit": ctx.state["reflect_loop"]["exit_reason"]})


WORKFLOWS = [research_pipeline, research_subtopic]
