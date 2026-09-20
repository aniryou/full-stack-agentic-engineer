"""demo.py — see the core idea run.

  python demo.py           # four scenarios, printed: happy path, crash+retry, human gate, slow tool
  python demo.py kill      # start a run, charge the card, then DIE before the checkpoint
  python demo.py resume    # a fresh process finds the run on disk and finishes it — one charge

The kill/resume pair is the whole point: durability means a *different process* can finish the work.
"""

import os
import sys

from durable import Agent, Crash, FakeClock, FakeModel, LeaseHeld, PaymentAPI, Queue, Store, Wait

RUNS = "runs.json"


def show(run):
    print(f"  {run.id}  status={run.status:<8} waiting_on={run.waiting_on}  result={run.result}")
    for i, s in enumerate(run.journal):
        if s["type"] == "decision":
            print(f"    [{i}] decision  {s.get('tool') or 'final'} {s.get('args', s.get('final', ''))}")
        else:
            print(f"    [{i}] intent    {s['tool']} key={s['key']} done={s['done']} result={s.get('result')}")


def fresh(script, tools=None, clock=None):
    if os.path.exists(RUNS):
        os.remove(RUNS)
    pay = PaymentAPI()
    return Agent(Store(RUNS), Queue(), FakeModel(script), tools or {"charge": pay}, clock=clock or FakeClock()), pay


def scenario_happy():
    print("\n1) HAPPY PATH — one step per wake-up, the store is the only memory")
    agent, pay = fresh([{"tool": "charge", "args": {"amount": 42}}, {"final": "charged 42"}])
    run = agent.start("pay invoice 42")
    while agent.queue.deliver_one(agent.handle):
        print(f"  wake-up #{agent.queue.delivered}: journal={len(agent.store.get(run.id).journal)} queued={len(agent.queue.items)}")
    show(agent.store.get(run.id))
    print("  charges:", pay.charges)


def scenario_crash():
    print("\n2) CRASH after the card is charged, before the checkpoint — then a retry from another worker")
    clock = FakeClock()
    agent, pay = fresh([{"tool": "charge", "args": {"amount": 42}}, {"final": "charged 42"}], clock=clock)
    run = agent.start("pay invoice 42")
    agent.crash_at.add("after_side_effect")
    try:
        agent.queue.deliver_one(agent.handle)
    except Crash as e:
        print("  💥", e)
    show(agent.store.get(run.id))
    print("  charges:", pay.charges, "← the money moved; the journal only says 'intent, not done'")
    worker2 = Agent(agent.store, agent.queue, agent.model, agent.tools, worker="worker-2", clock=clock)
    try:
        agent.queue.deliver_one(worker2.handle)
    except LeaseHeld as e:
        print("  retry refused:", e)
    clock.advance(61)
    print("  ...lease expired; retry re-executes the SAME intent with the SAME key")
    agent.queue.drain(worker2.handle)
    show(agent.store.get(run.id))
    print("  charges:", pay.charges, "← still one; model calls:", agent.model.calls)


def scenario_human():
    print("\n3) HUMAN GATE — park with a token, execute exactly what was approved")
    pay = PaymentAPI()
    pay.needs_approval = True
    agent, _ = fresh([{"tool": "charge", "args": {"amount": 4200}}, {"final": "paid"}], tools={"charge": pay})
    run = agent.start("pay invoice 4200")
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    print(f"  parked: status={run.status} why={run.waiting_on['why']} queue={agent.queue.items}  (nothing runs, nothing costs)")
    agent.resume(run.id, run.waiting_on["token"], {"approved": True})
    agent.queue.drain(agent.handle)
    show(agent.store.get(run.id))
    print("  charges:", pay.charges, "| model calls:", agent.model.calls, "← approval never re-asks the model")


def scenario_slow_tool():
    print("\n4) SLOW TOOL — the job runs elsewhere; the run parks until the webhook calls resume()")
    def export(region, key):
        return Wait(token="job-" + key)
    agent, _ = fresh([{"tool": "export", "args": {"region": "apac"}}, {"final": "report ready"}], tools={"export": export})
    run = agent.start("export apac")
    agent.queue.drain(agent.handle)
    run = agent.store.get(run.id)
    print(f"  parked: status={run.status} token={run.waiting_on['token']}")
    agent.resume(run.id, run.waiting_on["token"], {"rows": 1200})       # what the webhook does
    agent.queue.drain(agent.handle)
    show(agent.store.get(run.id))


# ------------------------------------------------- a real crash, across two processes
def kill():
    agent, pay = fresh([{"tool": "charge", "args": {"amount": 42}}, {"final": "charged 42"}])
    run = agent.start("pay invoice 42")
    agent.crash_at.add("after_side_effect")
    print("started", run.id, "— charging, then dying without a checkpoint")
    try:
        agent.queue.deliver_one(agent.handle)
    except Crash:
        os._exit(137)                                     # SIGKILL-style: no cleanup, no lease release


def resume():
    store = Store(RUNS)
    pay = PaymentAPI()                                    # in real life this is the same external system:
    stuck = [Run for Run in (store.get(r) for r in store.runs) if Run.status == "RUNNING"]
    for run in stuck:                                     # a reaper/Cloud Tasks retry re-creates the wake-up
        print("found", run.id, "with pending intent:", run.pending_intent() is not None)
        agent = Agent(store, Queue(), FakeModel([{"final": "charged 42"}]), {"charge": pay}, worker="worker-2", clock=lambda: 10**12)
        agent._wake(run)
        agent.queue.drain(agent.handle)
        show(store.get(run.id))
        print("charges this process made with the journaled key:", pay.charges, "→ the payment API de-duplicates on the key")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "kill":
        kill()
    elif cmd == "resume":
        resume()
    else:
        scenario_happy()
        scenario_crash()
        scenario_human()
        scenario_slow_tool()
