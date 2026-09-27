"""The Mistral provider path, narrated.

    python -m lra.adapters.mistral.demo workflow  # the tool loop on Mistral Workflows, on a local Temporal dev server
    python -m lra.adapters.mistral.demo live      # the lra tool loop with a real Mistral model deciding (MISTRAL_API_KEY)

``workflow`` needs Python 3.12-3.14 and the ``mistral`` extra (mistralai-workflows); the Temporal dev server is
downloaded on first use. ``live`` runs on 3.11 too and needs only ``mistralai`` and a key.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta


def workflow_demo() -> None:
    """Approve -> one charge; the worker dies after charging -> the retry converges; a 2-second approval timeout."""
    import asyncio
    import importlib.util

    if sys.version_info < (3, 12) or importlib.util.find_spec("mistralai.workflows") is None:
        sys.exit("`demo workflow` needs Python >= 3.12 and mistralai-workflows "
                 '(pip install -e ".[mistral]" on 3.12+); `demo live` runs on 3.11.')

    from lra.adapters.mistral import workflow as mw
    from lra.adapters.mistral.local_temporal import local_worker, start
    from lra.examples.tool_agent import PaymentAPI, ScriptedDecider

    async def go() -> None:
        mw.MODEL = ScriptedDecider([{"tool": "charge", "args": {"amount": 42}}, {"final": "charged 42"}] * 3)
        mw.PAYMENTS = PaymentAPI()
        async with local_worker([mw.InvoiceAgent], [mw.decide, mw.charge]) as client:
            print("\n1) APPROVAL GATE: wait_condition parks the execution; a signal resumes it")
            h = await start(client, "pay invoice 42", "demo-approve")
            await asyncio.sleep(0.5)
            print("  status while parked:", (await h.describe()).status.name)
            await h.signal("decide", {"approved": True, "approver": "cfo"})
            r = (await h.result())["result"]
            print("  result:", r["status"], r["result"], "| charges:", mw.PAYMENTS.charges)
            print("\n2) WORKER DIES after the charge: Temporal retries the activity with the SAME key")
            mw.CRASH_ONCE["after_charge"] = True
            h = await start(client, "pay invoice 42", "demo-crash")
            await asyncio.sleep(0.5)
            await h.signal("decide", {"approved": True})
            r = (await h.result())["result"]
            print("  result:", r["status"], "| charges:", {k: v for k, v in mw.PAYMENTS.charges.items() if "crash" in k},
                  "| model calls total:", mw.MODEL.calls)
            print("\n3) NOBODY ANSWERS: the gate has a deadline")
            h = await start(client, "pay invoice 42", "demo-expire", approval_timeout_s=2)
            r = (await h.result())["result"]
            print("  result:", r["status"], r["result"])
        print("\nSame invariants, provided by the platform: history = store, activity = intent + result, "
              "task = lease, loop bound = budget, wait_condition = park.")
        os._exit(0)                                   # the SDK's runtime threads make interpreter shutdown noisy

    asyncio.run(go())


def live() -> None:
    """The lra tool loop unchanged; only the model is real. export MISTRAL_API_KEY=... (and MISTRAL_MODEL to override)."""
    from lra import Engine, Event
    from lra.adapters.memory import FakeClock, InMemoryEventBus, InMemoryStateStore, InMemoryTaskQueue, LocalRunner
    from lra.adapters.mistral import TOOL_SPECS, MistralDecider
    from lra.examples.tool_agent import PaymentAPI, make_tool_agent

    pay, clock = PaymentAPI(), FakeClock()
    store, queue = InMemoryStateStore(), InMemoryTaskQueue(clock)
    engine = Engine(store=store, queue=queue, bus=InMemoryEventBus(), llm=None, clock=clock,
                    workflows=[make_tool_agent(MistralDecider(TOOL_SPECS), {"charge": pay})], lease_ttl=timedelta(seconds=60))
    runner = LocalRunner(engine, queue, clock)
    run = engine.start("tool_agent", {"goal": "Pay supplier invoice INV-1042: 42 SGD."})
    runner.run_until_idle()
    r = store.get(run.run_id)
    print("model proposed:", r.state["journal"][0], "| parked:", r.status.value, r.wait.key if r.wait else None)
    if r.wait:
        engine.resume(Event(run_id=run.run_id, key=r.wait.key, payload={"decision": "approve", "by": "cfo"}))
        runner.run_until_idle()
    r = store.get(run.run_id)
    print("result:", r.status.value, r.result, "| charges:", pay.charges)


if __name__ == "__main__":
    {"workflow": workflow_demo, "live": live}.get((sys.argv[1:] or [""])[0], lambda: sys.exit(__doc__))()
