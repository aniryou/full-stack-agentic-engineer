"""Run with `python -m pytest tests -q` or plain `python tests/test_core.py` (no pytest needed)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from durable import Agent, Crash, FakeClock, FakeModel, LeaseHeld, PaymentAPI, Queue, Store, Wait


def make(script, tmp_path="/tmp/lra_test_runs.json", tools=None, clock=None):
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    pay = PaymentAPI()
    tools = tools or {"charge": pay}
    clock = clock or FakeClock()
    agent = Agent(Store(tmp_path), Queue(), FakeModel(script), tools, clock=clock)
    return agent, pay, clock


SCRIPT = [{"tool": "charge", "args": {"amount": 42}}, {"final": "charged 42"}]


def test_happy_path_one_step_per_wakeup():
    agent, pay, _ = make(SCRIPT)
    run = agent.start("pay 42")
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "DONE" and run.result == "charged 42"
    assert [s["type"] for s in run.journal] == ["decision", "intent", "decision"]
    assert list(pay.charges.values()) == [42]
    assert agent.queue.delivered == 2                      # two wake-ups: decide+charge, then final


def test_state_survives_a_process_restart():
    agent, pay, _ = make(SCRIPT)
    run = agent.start("pay 42")
    agent.queue.deliver_one(agent.handle)
    # "restart": a brand-new Store reads the file a dead process left behind
    fresh = Store(agent.store.path)
    assert fresh.get(run.id).journal[1]["done"] is True


def test_crash_after_side_effect_charges_once():
    agent, pay, clock = make(SCRIPT)
    run = agent.start("pay 42")
    agent.crash_at.add("after_side_effect")
    try:
        agent.queue.deliver_one(agent.handle)
        assert False, "expected a crash"
    except Crash:
        pass
    assert list(pay.charges.values()) == [42]              # the charge happened...
    assert agent.store.get(run.id).pending_intent() is not None   # ...but the journal says "not sure"
    other = Agent(agent.store, agent.queue, agent.model, agent.tools, worker="worker-2", clock=clock)
    try:
        agent.queue.deliver_one(other.handle)              # retry lands on another instance, too early
        assert False
    except LeaseHeld:
        pass
    clock.advance(61)                                      # the dead worker's lease expires
    agent.queue.drain(other.handle)                        # retry: re-executes the SAME intent, same key
    assert agent.store.get(run.id).status == "DONE"
    assert list(pay.charges.values()) == [42]              # still one charge
    assert agent.model.calls == 2                          # the model was never re-asked for step 1


def test_crash_after_intent_before_side_effect_executes_once():
    agent, pay, clock = make(SCRIPT)
    run = agent.start("pay 42")
    agent.crash_at.add("after_intent")
    try:
        agent.queue.deliver_one(agent.handle)
    except Crash:
        pass
    assert pay.charges == {}
    clock.advance(61)
    agent.queue.drain(agent.handle)
    assert list(pay.charges.values()) == [42] and agent.store.get(run.id).status == "DONE"


def test_duplicate_delivery_is_ignored():
    agent, pay, _ = make(SCRIPT)
    run = agent.start("pay 42")
    agent.queue.duplicate_next()
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "DONE" and len(run.journal) == 3 and list(pay.charges.values()) == [42]
    assert agent.model.calls == 2


def test_budget_stops_a_runaway_loop():
    agent, pay, _ = make([{"tool": "charge", "args": {"amount": 1}}] * 50)
    run = agent.start("loop", max_steps=3)
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "FAILED" and "budget" in run.result and len(pay.charges) == 3


def test_human_gate_parks_then_executes_exactly_what_was_approved():
    pay = PaymentAPI()
    pay.needs_approval = True
    agent, _, _ = make(SCRIPT, tools={"charge": pay})
    run = agent.start("pay 42")
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "WAITING" and run.waiting_on["why"] == "approval" and pay.charges == {}
    assert not agent.queue.items                           # nothing scheduled while parked
    token = run.waiting_on["token"]
    agent.resume(run.id, "wrong-token", {"approved": True})
    assert agent.store.get(run.id).status == "WAITING"
    agent.resume(run.id, token, {"approved": True})
    agent.resume(run.id, token, {"approved": True})        # double click → no-op
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "DONE" and list(pay.charges.values()) == [42]
    assert agent.model.calls == 2                          # approval never re-asks the model


def test_rejection_is_recorded_and_the_loop_continues():
    pay = PaymentAPI()
    pay.needs_approval = True
    agent, _, _ = make([{"tool": "charge", "args": {"amount": 42}}, {"final": "ok, not paying"}], tools={"charge": pay})
    run = agent.start("pay 42")
    agent.queue.drain(agent.handle)
    agent.resume(run.id, agent.store.get(run.id).waiting_on["token"], {"approved": False, "reason": "too much"})
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "DONE" and pay.charges == {} and run.journal[1]["result"] == {"rejected": "too much"}


def test_slow_tool_parks_until_the_world_calls_back():
    started = []

    def export(region, key):
        started.append(key)
        return Wait(token="job-" + key)                    # the job continues elsewhere

    agent, _, _ = make([{"tool": "export", "args": {"region": "apac"}}, {"final": "report ready"}], tools={"export": export})
    run = agent.start("export apac")
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "WAITING" and run.waiting_on["why"] == "event"
    agent.resume(run.id, "job-" + started[0], {"rows": 1200})   # the webhook
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    assert run.status == "DONE" and run.journal[1]["result"] == {"rows": 1200} and len(started) == 1


def test_two_workers_cannot_advance_the_same_run():
    agent, _, clock = make(SCRIPT)
    run = agent.start("pay 42")
    agent.store.acquire_lease(run.id, "worker-2", 60, clock())
    try:
        agent.handle(agent.queue.items[0])
        assert False
    except LeaseHeld:
        pass
    clock.advance(61)
    agent.queue.drain(agent.handle)
    assert agent.store.get(run.id).status == "DONE"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
