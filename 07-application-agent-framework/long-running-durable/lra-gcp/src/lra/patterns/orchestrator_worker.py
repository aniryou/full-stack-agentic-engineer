"""Orchestrator–worker (hierarchical delegation).

A planner step asks the model for a list of independent sub-tasks, validates
the plan, then returns :class:`FanOut`. The engine spawns one *child run*
per sub-task (each durable, retried and budgeted on its own) and suspends
the parent until all children reach a terminal state. The parent resumes at
``then`` with ``ctx.state["children"] = {"results": {...}, "failures": {...}}``.

Design points:

* Children are runs, not threads: a 40-way fan-out costs nothing while it
  waits, survives worker restarts, and shows up individually in the run
  store for debugging.
* ``child_key`` is derived from the sub-task content, so re-planning after
  a crash reuses existing children instead of duplicating work.
* The planner caps the fan-out (``max_children``) — the model does not get to
  decide how much money to spend.
* Partial failure is data, not an exception: the aggregator decides whether
  3/5 successful children are enough.
"""

from __future__ import annotations

from typing import Any

from ..core.models import Budget, ChildSpec, digest
from ..core.workflow import FanOut, StepContext, StepFailed

PLAN_PROMPT = """Break the GOAL into at most {max_children} independent sub-tasks that can run in parallel.
Respond ONLY with JSON: {{"subtasks": [{{"title": "...", "instructions": "..."}}]}}

GOAL: {goal}
"""


def plan_and_fan_out(
    ctx: StepContext,
    *,
    goal: str,
    child_workflow: str,
    then: str,
    max_children: int = 5,
    child_budget: Budget | None = None,
    extra_input: dict[str, Any] | None = None,
) -> FanOut:
    resp = ctx.llm(PLAN_PROMPT.format(goal=goal, max_children=max_children), json_mode=True, temperature=0.0)
    try:
        plan = resp.json()
        subtasks = plan["subtasks"]
        assert isinstance(subtasks, list) and subtasks
    except Exception as e:  # noqa: BLE001
        raise StepFailed(f"planner returned an unusable plan: {e!r}: {resp.text[:200]}") from e

    subtasks = subtasks[:max_children]  # hard cap regardless of what the model said
    children: list[ChildSpec] = []
    seen: set[str] = set()
    for st in subtasks:
        key = f"t{digest({'title': st.get('title'), 'instructions': st.get('instructions')})[:8]}"
        if key in seen:
            continue
        seen.add(key)
        children.append(
            ChildSpec(
                workflow=child_workflow,
                child_key=key,
                input={"title": st.get("title", ""), "instructions": st.get("instructions", ""), **(extra_input or {})},
                budget=child_budget,
            )
        )
    ctx.state["plan"] = [{"child_key": c.child_key, **c.input} for c in children]
    ctx.emit("plan.created", {"n_children": len(children)})
    return FanOut(children=children, then=then)


def child_outcomes(ctx: StepContext) -> tuple[dict[str, Any], dict[str, str]]:
    """Convenience accessor for the aggregator step."""
    children = ctx.state.get("children", {})
    return children.get("results", {}), children.get("failures", {})
