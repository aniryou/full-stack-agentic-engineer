"""mistral_workflow.py — the same agent, written on Mistral Workflows (Temporal underneath).

Read durable.py first. Then notice what disappears here: no Store, no Queue, no lease,
no park/resume bookkeeping. The platform provides each of the five rules:

  (1) durable state   – the execution's EVENT HISTORY is the store. A worker that dies is
                        replaced; the next worker replays history and continues.
  (2) intent → act    – every ACTIVITY call is recorded before it runs and its result is
                        recorded once it succeeds. On replay a finished activity is never
                        re-run; a failed one is RETRIED — so activities that act on the world
                        must be safe to repeat (our charge carries an idempotency key).
                        The model call is an activity too: a replay never re-asks the model.
  (3) lease           – a task goes to exactly one worker at a time; a worker that stops
                        heartbeating is presumed dead and the task is rescheduled.
  (4) budget          – a plain loop bound in the (deterministic) workflow code, plus
                        execution_timeout as the wall-clock cap.
  (5) park, don't wait – workflow.wait_condition() suspends at zero cost until a SIGNAL
                        arrives or the timeout fires. The approval is one line.

Run it locally (a Temporal dev server is downloaded on first use):   python demo.py workflow
Run it on Mistral's hosted orchestrator: export MISTRAL_API_KEY and
    python -c "import asyncio, mistralai.workflows as w, mistral_workflow as m; asyncio.run(w.run_worker([m.InvoiceAgent]))"
then start an execution from the Studio console / API and send the `decide` signal.
"""

import asyncio
from datetime import timedelta

import mistralai.workflows as workflows
from mistralai.workflows import workflow

from durable import FakeModel, PaymentAPI

# --- what runs in your worker process; swapped by demo/tests ----------------
MODEL = FakeModel([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
PAYMENTS = PaymentAPI()
CRASH_ONCE = {"after_charge": False}          # fault injection for the demo/tests


# --- activities: the only place side effects happen -------------------------
@workflows.activity(start_to_close_timeout=timedelta(minutes=2), retry_policy_max_attempts=5)
async def decide(goal: str, journal: list) -> dict:
    """One model call. Recorded in history → a replay reuses the recorded decision. (2)"""
    return MODEL.decide(goal, journal)


@workflows.activity(start_to_close_timeout=timedelta(seconds=30), retry_policy_max_attempts=5,
                    retry_policy_backoff_coefficient=2.0)
async def charge(amount: float, key: str) -> dict:
    """Idempotent by key. If the worker dies after the API call, the retry repeats the SAME
    call with the SAME key and the payment API de-duplicates. (2)"""
    result = PAYMENTS(amount, key)
    if CRASH_ONCE["after_charge"]:            # simulate: money moved, then the process died
        CRASH_ONCE["after_charge"] = False
        raise RuntimeError("worker died after the side effect")
    return result


TOOLS = {"charge": charge}


# --- the workflow: deterministic orchestration ------------------------------
@workflows.workflow.define(name="invoice-agent", execution_timeout=timedelta(days=7))
class InvoiceAgent:
    def __init__(self):
        self.decision = None                  # (5) set by the signal handler below

    @workflows.workflow.signal(name="decide")
    async def on_decide(self, approved: bool, approver: str = "", reason: str = "") -> None:
        self.decision = {"approved": approved, "approver": approver, "reason": reason}

    @workflows.workflow.entrypoint
    async def run(self, goal: str, max_steps: int = 10, approval_timeout_s: int = 86400,
                  gated_tools: list[str] = ["charge"]) -> dict:
        journal = []
        for _ in range(max_steps):                                            # (4) budget in code
            d = await decide(goal, journal)                                   # (2) recorded once
            journal.append({"type": "decision", **d})
            if "final" in d:
                return {"status": "DONE", "result": d["final"], "journal": journal}
            key = f"{workflows.get_execution_id()}:{len(journal)}"           # (2) idempotency key
            intent = {"type": "intent", "tool": d["tool"], "args": d["args"], "key": key, "done": False}
            journal.append(intent)
            if d["tool"] in gated_tools:                                      # (5) park on a human
                self.decision = None
                try:
                    await workflow.wait_condition(lambda: self.decision is not None,
                                                  timeout=timedelta(seconds=approval_timeout_s))
                except asyncio.TimeoutError:
                    return {"status": "FAILED", "result": "approval expired", "journal": journal}
                if not self.decision["approved"]:                             # tell the model why
                    intent.update(done=True, result={"rejected": self.decision["reason"]})
                    continue
            result = await TOOLS[d["tool"]](**d["args"], key=key)             # (1)(2)(3) activity
            intent.update(done=True, result=result)
        return {"status": "FAILED", "result": f"budget: {max_steps} steps", "journal": journal}
