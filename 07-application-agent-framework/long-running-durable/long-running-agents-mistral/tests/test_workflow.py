"""The Mistral Workflow, run on a local Temporal dev server (downloaded on first use; ~10s per test).
Run with `python -m pytest tests/test_workflow.py -q` or `python tests/test_workflow.py`."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from durable import FakeModel, PaymentAPI
import mistral_workflow as mw
from local_temporal import local_worker, start


def fresh(script):
    mw.MODEL, mw.PAYMENTS, mw.CRASH_ONCE["after_charge"] = FakeModel(script), PaymentAPI(), False


SCRIPT = [{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}]


def run(coro):
    return asyncio.run(coro)


def test_approval_signal_then_one_charge():
    async def go():
        fresh(SCRIPT)
        async with local_worker([mw.InvoiceAgent], [mw.decide, mw.charge]) as client:
            h = await start(client, "pay invoice 42", "wf-approve")
            await asyncio.sleep(0.5)                           # parked on wait_condition, costing nothing
            assert (await h.describe()).status.name == "RUNNING"
            await h.signal("decide", {"approved": True, "approver": "cfo"})
            r = (await h.result())["result"]
        assert r["status"] == "DONE" and r["result"] == "Charged 42."
        assert list(mw.PAYMENTS.charges) == ["wf-approve:1"]  # key = execution id + journal position
        assert mw.MODEL.calls == 2
    run(go())


def test_worker_dies_after_the_charge_retry_converges_to_one_charge():
    async def go():
        fresh(SCRIPT)
        mw.CRASH_ONCE["after_charge"] = True                   # the activity raises after acting
        async with local_worker([mw.InvoiceAgent], [mw.decide, mw.charge]) as client:
            h = await start(client, "pay invoice 42", "wf-crash")
            await asyncio.sleep(0.5)
            await h.signal("decide", {"approved": True})
            r = (await h.result())["result"]
        assert r["status"] == "DONE"
        assert list(mw.PAYMENTS.charges.values()) == [42.0]    # retried with the same key → one charge
        assert mw.MODEL.calls == 2                              # the retry did not re-ask the model
    run(go())


def test_rejection_is_fed_back_to_the_model():
    async def go():
        fresh([{"tool": "charge", "args": {"amount": 42}}, {"final": "Understood, not paying."}])
        async with local_worker([mw.InvoiceAgent], [mw.decide, mw.charge]) as client:
            h = await start(client, "pay invoice 42", "wf-reject")
            await asyncio.sleep(0.5)
            await h.signal("decide", {"approved": False, "reason": "vendor not onboarded"})
            r = (await h.result())["result"]
        assert r["status"] == "DONE" and mw.PAYMENTS.charges == {}
        assert r["journal"][1]["result"] == {"rejected": "vendor not onboarded"}
    run(go())


def test_approval_timeout_and_budget():
    async def go():
        fresh(SCRIPT)
        async with local_worker([mw.InvoiceAgent], [mw.decide, mw.charge]) as client:
            h = await start(client, "pay invoice 42", "wf-expire", approval_timeout_s=1)
            r = (await h.result())["result"]
            assert r["status"] == "FAILED" and r["result"] == "approval expired" and mw.PAYMENTS.charges == {}
            fresh([{"tool": "charge", "args": {"amount": 1}}] * 5)   # a model that never says "done"
            h2 = await start(client, "loop", "wf-budget", max_steps=3, gated_tools=[])   # no gate
            r2 = (await h2.result())["result"]
        assert r2["status"] == "FAILED" and r2["result"].startswith("budget")
        assert len(mw.PAYMENTS.charges) == 3                   # three distinct keys, then the budget stopped it
    run(go())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    os._exit(0)
