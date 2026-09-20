"""Reflection (evaluator–optimizer) loop as durable steps.

Naive form: ``while score < threshold: draft = revise(critique(draft))``.
That loop is the classic budget sink. Two changes make it production-safe:

1. Each iteration is its own *step*, so the run checkpoints after every
   critique and every revision. A crash mid-loop resumes at the right
   iteration instead of restarting from scratch.
2. The exit conditions are explicit and stored in state: max iterations,
   a quality threshold, *and* the run budget (enforced by ``ctx.llm``).
   Loops end with a reason you can read in the run history.

``install_reflection_steps`` registers ``<prefix>_critique`` and
``<prefix>_revise`` on a workflow. Enter the loop with ``Next(f"{prefix}_critique")``
once ``ctx.state[draft_key]`` exists; it exits to ``exit_step``.
"""

from __future__ import annotations

from typing import Any

from ..core.workflow import Next, StepContext, Workflow

CRITIQUE_PROMPT = """You are a rigorous reviewer.
Score the DRAFT from 0-10 against the GOAL and list concrete, actionable fixes.
Respond ONLY with JSON: {{"score": <int>, "fixes": ["..."], "verdict": "<one sentence>"}}

GOAL: {goal}

DRAFT:
{draft}
"""

REVISE_PROMPT = """Improve the DRAFT by applying the FIXES. Keep what already works.
Return only the improved draft.

GOAL: {goal}

FIXES:
{fixes}

DRAFT:
{draft}
"""


def install_reflection_steps(
    wf: Workflow,
    *,
    prefix: str = "reflect",
    draft_key: str = "draft",
    goal_key: str = "goal",
    exit_step: str,
    max_iterations: int = 3,
    threshold: int = 8,
) -> None:
    critique_name, revise_name = f"{prefix}_critique", f"{prefix}_revise"

    @wf.step(critique_name)
    def critique(ctx: StepContext) -> Any:
        loop = ctx.state.setdefault(f"{prefix}_loop", {"iteration": 0, "history": []})
        resp = ctx.llm(
            CRITIQUE_PROMPT.format(goal=ctx.state.get(goal_key, ""), draft=ctx.state[draft_key]),
            json_mode=True,
            temperature=0.0,
        )
        try:
            verdict = resp.json()
            score = int(verdict.get("score", 0))
            fixes = list(verdict.get("fixes", []))
        except (ValueError, AttributeError):
            # A malformed critique should not sink the run: treat as "not good enough, no fixes".
            score, fixes, verdict = 0, ["critique was not valid JSON; revise for clarity"], {"raw": resp.text}
        loop["history"].append({"iteration": loop["iteration"], "score": score, "fixes": fixes})
        loop["last_fixes"] = fixes
        ctx.emit("reflection.critique", {"iteration": loop["iteration"], "score": score})

        if score >= threshold:
            loop["exit_reason"] = f"threshold met (score {score} >= {threshold})"
            return Next(exit_step)
        if loop["iteration"] >= max_iterations:
            loop["exit_reason"] = f"max iterations reached ({max_iterations}); best score {max(h['score'] for h in loop['history'])}"
            return Next(exit_step)
        return Next(revise_name)

    @wf.step(revise_name)
    def revise(ctx: StepContext) -> Any:
        loop = ctx.state[f"{prefix}_loop"]
        resp = ctx.llm(
            REVISE_PROMPT.format(
                goal=ctx.state.get(goal_key, ""),
                fixes="\n".join(f"- {f}" for f in loop.get("last_fixes", [])),
                draft=ctx.state[draft_key],
            )
        )
        ctx.state[draft_key] = resp.text
        loop["iteration"] += 1
        return Next(critique_name)
