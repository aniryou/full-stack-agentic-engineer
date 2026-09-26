"""A model-chosen tool loop on the engine: the model decides, the run journals, then acts.

The other examples are fixed pipelines in which the model writes text. Here the model picks the
*next action* (a tool call or a final answer), which is the case where a retry is most dangerous:
ask the model again after a crash and it may choose a different call, so a second, different
side effect happens. The fix is structural and takes two effect records per turn:

    decide  ctx.effect("decide:N", model.decide)   the decision is recorded before anything acts on it;
                                                   a retried step reads the record, never re-asks the model
    act     ctx.effect("act:N", tool(key=...))    the call runs at most once; its idempotency key
                                                   ``run_id:N`` goes downstream too (defence in depth)

A tool listed in ``gated`` parks the run on a human approval between the two (``Wait``); a rejection
is fed back to the model as the tool's result, so the model can choose again. The journal lives in
``ctx.state["journal"]`` as decision/intent pairs, which is also the prompt: recorded facts, never
the model's memory. ``max_steps`` in the run's :class:`Budget` bounds the loop in code.

A *decider* is anything with ``decide(goal, journal) -> {"tool": name, "args": {...}} | {"final": text}``:
:class:`ScriptedDecider` here (T0), :class:`lra.adapters.mistral.MistralDecider` (function calling),
or a thin wrapper over any :class:`~lra.core.ports.LLM`.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable, Iterable

from ..core.models import Budget
from ..core.workflow import Done, Next, StepContext, Wait, Workflow
from ..patterns.hitl import approval_key


class ScriptedDecider:
    """Replays a list of decisions, one per call. ``calls`` counts how often the model was asked."""

    def __init__(self, script: Iterable[dict[str, Any]]) -> None:
        self.script = list(script)
        self.calls = 0

    def decide(self, goal: str, journal: list[dict[str, Any]]) -> dict[str, Any]:
        self.calls += 1
        if not self.script:
            raise RuntimeError("ScriptedDecider ran out of decisions")
        return dict(self.script.pop(0))


class PaymentAPI:
    """A downstream API that de-duplicates on the ``Idempotency-Key`` it is given (Stripe-style)."""

    def __init__(self) -> None:
        self.charges: dict[str, float] = {}
        self.calls = 0

    def __call__(self, amount: float, key: str) -> dict[str, Any]:
        self.calls += 1
        self.charges.setdefault(key, float(amount))
        return {"charge_id": key, "amount": self.charges[key]}


def make_tool_agent(
    decider: Any,
    tools: dict[str, Callable[..., dict[str, Any]]],
    *,
    gated: Iterable[str] = ("charge",),
    approval_timeout: timedelta = timedelta(days=1),
    name: str = "tool_agent",
    max_steps: int = 20,
) -> Workflow:
    """Build the loop. Each tool is called as ``tool(**args, key=idempotency_key)``."""
    gated = set(gated)
    wf = Workflow(name, version="1", default_budget=Budget(max_steps=max_steps))

    @wf.step(start=True)
    def decide(ctx: StepContext):
        journal = ctx.state.setdefault("journal", [])
        n = len(journal)
        d = ctx.effect(f"decide:{n}", lambda: decider.decide(ctx.input["goal"], journal))
        journal.append({"type": "decision", **d})
        if "final" in d:
            return Done({"result": d["final"]})
        key = f"{ctx.run_id}:{n + 1}"                       # stable across retries, unique across runs
        journal.append({"type": "intent", "tool": d["tool"], "args": d["args"], "key": key, "done": False})
        if d["tool"] in gated:
            return Wait(key=approval_key(ctx, f"tool-{n + 1}"), then="act", kind="approval", timeout=approval_timeout)
        return Next("act")

    @wf.step()
    def act(ctx: StepContext):
        journal = ctx.state["journal"]
        intent = journal[-1]
        n = len(journal) - 1
        if intent["tool"] in gated:
            evt = ctx.state.get("events", {}).get(approval_key(ctx, f"tool-{n}"), {})
            payload = evt.get("payload", {})                               # (a timeout fails the run: reaper)
            if payload.get("decision") != "approve":                        # tell the model why, let it choose again
                intent.update(done=True, result={"rejected": payload.get("comment", "rejected")})
                return Next("decide")
        tool = tools[intent["tool"]]
        result = ctx.effect(f"act:{n}", lambda: tool(**intent["args"], key=intent["key"]))
        intent.update(done=True, result=result)
        return Next("decide")

    return wf
