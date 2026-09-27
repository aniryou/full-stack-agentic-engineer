"""The same long-running research pipeline, expressed as an ADK 2 ``Workflow``.

This is the *managed-runtime* path: instead of our Cloud Tasks/Firestore
engine you get ADK's graph orchestrator (nodes, edges, routing, parallel
workers, retries, interrupts) plus Agent Engine's session persistence and
scale-to-zero runtime.

Node-by-node mapping to ``lra`` (src/lra/examples/research_pipeline.py):

=================  ==========================================  ===========================
lra concept        ADK 2 equivalent                            Notes
=================  ==========================================  ===========================
Workflow/Step      ``Workflow`` + ``@node`` FunctionNode        state via ``ctx.state``
Next(step)         ``ctx.route = "..."`` + routing-map edge     model may set the route
FanOut/children    ``@node(parallel_worker=True)`` over a list  in-process asyncio, not runs
Wait/resume        yield Event(long_running_tool_ids={id});     resume with FunctionResponse
                   ``ResumabilityConfig(is_resumable=True)``    for that id
retries/backoff    ``RetryConfig``                              in-process; count not persisted
Budget             (bring your own) — plugins/callbacks         no first-class run budget
Checkpoint         session events + state (persisted by         node boundaries when resumable
                   Database/VertexAiSessionService)
=================  ==========================================  ===========================

Run locally (no model needed — every node here is a plain function):

    python examples/adk_agent_engine/run_local.py

Swap ``synthesize`` for an ``LlmAgent`` node (``node(LlmAgent(...))``) to use
Gemini; the graph does not change.
"""

from __future__ import annotations

import json
import os
from typing import Any

from google.adk.agents import Context
from google.adk.apps import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events import Event
from google.adk.workflow import START, RetryConfig, Workflow, node
from google.genai import types

REVIEW_CALL_ID = "review-gate"  # the interrupt id the approval webhook must answer
MAX_ITERATIONS = 2
THRESHOLD = 8


def _text_of(node_input: Any) -> str:
    if hasattr(node_input, "parts") and node_input.parts:
        return node_input.parts[0].text or ""
    return str(node_input)


# --------------------------------------------------------------------- nodes
@node(name="plan")
def plan(ctx: Context, node_input: Any) -> list[str]:
    """Split the goal into sub-tasks. Returning a list feeds the parallel worker one item each."""
    goal = _text_of(node_input)
    ctx.state["goal"] = goal
    subtopics = [f"{goal}: background", f"{goal}: current approaches", f"{goal}: open problems"]
    ctx.state["subtopics"] = subtopics
    return subtopics


@node(
    name="research",
    parallel_worker=True,          # one worker per list item, concurrently
    max_parallel_workers=3,        # protect model quota
    retry_config=RetryConfig(max_attempts=3, initial_delay=0.5, max_delay=5.0, backoff_factor=2.0),
)
def research(ctx: Context, node_input: str) -> dict[str, str]:
    """Per-subtopic research. Real version: tool calls (search/RAG), idempotent."""
    return {"topic": node_input, "findings": f"- three findings about {node_input}"}


@node(name="synthesize")
def synthesize(ctx: Context, node_input: list[dict[str, str]]) -> str:
    findings = "\n".join(f"## {r['topic']}\n{r['findings']}" for r in node_input)
    ctx.state["draft"] = f"# Report on {ctx.state['goal']}\n\n{findings}"
    ctx.state["iteration"] = 0
    return ctx.state["draft"]


@node(name="critique")
def critique(ctx: Context, node_input: Any) -> dict[str, Any]:
    """Evaluator: sets ``ctx.route`` so the graph loops or exits. Bounded by MAX_ITERATIONS."""
    iteration = ctx.state.get("iteration", 0)
    score = 6 if iteration == 0 else 9  # a real evaluator calls the model with json_mode
    verdict = {"iteration": iteration, "score": score}
    if score >= THRESHOLD or iteration >= MAX_ITERATIONS:
        ctx.state["exit_reason"] = "threshold" if score >= THRESHOLD else "max_iterations"
        ctx.route = "approved"
    else:
        ctx.route = "revise"
    return verdict


@node(name="revise")
def revise(ctx: Context, node_input: Any) -> str:
    ctx.state["draft"] = ctx.state["draft"] + "\n\n(revised: added evidence and tightened claims)"
    ctx.state["iteration"] = ctx.state.get("iteration", 0) + 1
    return ctx.state["draft"]


@node(name="review_gate")
async def review_gate(ctx: Context, node_input: Any):
    """Human-in-the-loop: interrupt the invocation until a reviewer answers.

    Yielding an event with ``long_running_tool_ids`` pauses the workflow; the
    session is persisted and the container may scale to zero. The approval
    webhook resumes it by sending a ``FunctionResponse`` with the same id.
    """
    call = types.Part(
        function_call=types.FunctionCall(id=REVIEW_CALL_ID, name="request_review", args={"draft": ctx.state["draft"][:500]})
    )
    yield Event(author="review_gate", content=types.Content(role="model", parts=[call]), long_running_tool_ids={REVIEW_CALL_ID})


@node(name="publish")
def publish(ctx: Context, node_input: Any) -> dict[str, Any]:
    """Runs after resume; ``node_input`` is the reviewer's FunctionResponse ``response`` dict."""
    decision: dict[str, Any] = dict(node_input) if isinstance(node_input, dict) else {}
    if decision.get("decision") != "approve":
        ctx.state["published"] = False
        ctx.state["rejection"] = decision
        return {"published": False, "decision": decision}
    ctx.state["published"] = True
    ctx.state["post_id"] = f"post_{abs(hash(ctx.state['draft'])) % 10**8}"
    return {"published": True, "post_id": ctx.state["post_id"], "approved_by": decision.get("by")}


# --------------------------------------------------------------------- graph
research_workflow = Workflow(
    name="research_workflow",
    description="Plan -> parallel research -> synthesize -> bounded critique/revise loop -> human review -> publish",
    edges=[
        (START, plan, research, synthesize, critique, {"revise": revise, "approved": review_gate}),
        (revise, critique),
        (review_gate, publish),
    ],
)

# Resumability is what turns the interrupt into a durable multi-day wait.
app = App(
    name=os.environ.get("ADK_APP_NAME", "lra_research"),
    root_agent=research_workflow,
    resumability_config=ResumabilityConfig(is_resumable=True),
)


def review_response(decision: str, by: str, comment: str = "") -> types.Content:
    """Build the message that resumes a paused run (used by the webhook and tests)."""
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=REVIEW_CALL_ID, name="request_review", response={"decision": decision, "by": by, "comment": comment}
                )
            )
        ],
    )


def state_summary(state: dict[str, Any]) -> str:
    keys = ("goal", "iteration", "exit_reason", "published", "post_id")
    return json.dumps({k: state.get(k) for k in keys if k in state})
