"""Practice 6 — the approval gate on Mistral Workflows. Edit the TODO, then re-run the grader cell.

Workflow code must live in an importable module (the worker's sandbox re-imports it by name),
which is why this exercise is a file rather than a notebook cell."""

import asyncio
from datetime import timedelta

import mistralai.workflows as workflows
from mistralai.workflows import workflow

import mistral_workflow as mw


@workflows.workflow.define(name="invoice-agent-practice", execution_timeout=timedelta(days=7))
class MyInvoiceAgent:
    def __init__(self):
        self.decision = None

    @workflows.workflow.signal(name="decide")
    async def on_decide(self, approved: bool, approver: str = "", reason: str = "") -> None:
        self.decision = {"approved": approved, "approver": approver, "reason": reason}

    @workflows.workflow.entrypoint
    async def run(self, goal: str, max_steps: int = 10, approval_timeout_s: int = 86400,
                  gated_tools: list[str] = ["charge"]) -> dict:
        journal = []
        for _ in range(max_steps):
            d = await mw.decide(goal, journal)
            journal.append({"type": "decision", **d})
            if "final" in d:
                return {"status": "DONE", "result": d["final"], "journal": journal}
            key = f"{workflows.get_execution_id()}:{len(journal)}"
            intent = {"type": "intent", "tool": d["tool"], "args": d["args"], "key": key, "done": False}
            journal.append(intent)
            if d["tool"] in gated_tools:
                # TODO: reset self.decision; await workflow.wait_condition(lambda: self.decision is not None,
                #       timeout=timedelta(seconds=approval_timeout_s)) inside try/except asyncio.TimeoutError
                #       → return {"status": "FAILED", "result": "approval expired", "journal": journal};
                #       if rejected: intent.update(done=True, result={"rejected": self.decision["reason"]}); continue
                raise workflows.WorkflowError("TODO: the approval gate")   # WorkflowError fails the execution; a plain exception only fails the *task*, which the platform retries forever
            result = await mw.TOOLS[d["tool"]](**d["args"], key=key)
            intent.update(done=True, result=result)
        return {"status": "FAILED", "result": f"budget: {max_steps} steps", "journal": journal}
