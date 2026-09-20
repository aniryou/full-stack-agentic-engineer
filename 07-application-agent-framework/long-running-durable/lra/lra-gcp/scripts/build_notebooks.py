"""Generate the worked/practice notebook pairs.

Each code cell is written once with ``{{name}}`` holes and a dict of answers.
The *worked* notebook substitutes the answers; the *practice* notebook replaces
each hole with ``____`` and lists the hole names in a TODO comment. Solutions
to the practice notebooks are, by construction, the worked notebooks.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


class NB:
    def __init__(self, title: str, intro: str) -> None:
        self.cells: list[tuple[str, str, dict[str, str]]] = [("md", f"# {title}\n\n{intro}", {})]

    def md(self, text: str) -> "NB":
        self.cells.append(("md", text.strip("\n"), {}))
        return self

    def code(self, template: str, **answers: str) -> "NB":
        self.cells.append(("code", template.strip("\n"), answers))
        return self

    def build(self, practice: bool) -> nbf.NotebookNode:
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        for kind, body, answers in self.cells:
            if kind == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(body))
                continue
            src = body
            for name, value in answers.items():
                src = src.replace("{{" + name + "}}", "____" if practice else value)
            if practice and answers:
                src = "# TODO: fill in the blanks: " + ", ".join(answers) + "\n" + src
            assert "{{" not in src, f"unfilled hole in cell: {src[:80]}"
            nb.cells.append(nbf.v4.new_code_cell(src))
        return nb


SETUP = '''
from datetime import timedelta
import json
from lra import Engine, Event, Budget, Workflow, Next, Done, Wait, StepFailed, RunStatus, StepTask, SimulatedCrash
from lra.adapters.memory import FakeClock, FakeLLM, InMemoryStateStore, InMemoryTaskQueue, InMemoryEventBus, LocalRunner
from lra.examples import ALL_WORKFLOWS
from lra.examples.scripted import research_routes

def harness(routes=None, fail_times=0, chaos=None, lease_ttl_s=60):
    """One engine over in-memory adapters, driven the way Cloud Tasks would drive it."""
    clock, store, bus = FakeClock(), InMemoryStateStore(), InMemoryEventBus()
    queue = InMemoryTaskQueue(clock)
    llm = FakeLLM(routes=routes or research_routes(), fail_times=fail_times)
    engine = Engine(store=store, queue=queue, bus=bus, llm=llm, clock=clock, workflows=ALL_WORKFLOWS,
                    worker_id="worker-A", lease_ttl=timedelta(seconds=lease_ttl_s), chaos=chaos)
    return engine, LocalRunner(engine, queue, clock), clock, store, queue, bus

def show(run):
    print(f"{run.run_id} {run.status.value:12s} step={run.current_step} attempts={run.attempts} "
          f"wait={run.wait.key if run.wait else None} steps_used={run.budget.steps_used}")
'''

# =============================================================================
nb0 = NB(
    "00 · The core idea — a durable agent loop in 60 lines",
    "Before the engine, the *idea*. Using only the standard library we build a run store, a task queue and a worker, "
    "then break it with a crash and a duplicate delivery. The three invariants from the primer (§2) are the whole game:\n\n"
    "1. **Checkpoint before enqueue.**\n2. **`(run, step, attempt)` identifies work; anything else is stale.**\n"
    "3. **Record side effects before the checkpoint, keyed by intent.**",
)
nb0.md("## The store, the queue and the run document")
nb0.code(
    '''
from collections import deque
import copy

RUNS = {}           # run_id -> run document (Firestore stands in for this)
EFFECTS = {}        # "run:intent" -> result (idempotent side effects)
QUEUE = deque()     # (run_id, step, attempt)   (Cloud Tasks)
SEEN_TASKS = set()  # Cloud Tasks rejects a second task with the same name

def enqueue(run_id, step, attempt):
    key = (run_id, step, attempt)
    if key in SEEN_TASKS:
        return False           # dedup by name
    SEEN_TASKS.add(key)
    QUEUE.append(key)
    return True

def start(run_id, first_step):
    RUNS[run_id] = {"step": first_step, "attempts": {first_step: 1}, "state": {}, "status": "RUNNING", "version": 0}
    enqueue(run_id, first_step, 1)
''',
)
nb0.md("## Steps with an idempotent side effect\n\n`effect()` runs `fn` at most once per `(run, intent)`. Note the order: the effect is recorded *before* the caller checkpoints.")
nb0.code(
    '''
def effect(run_id, intent, fn):
    key = f"{run_id}:{intent}"
    if key in EFFECTS:
        return EFFECTS[key]            # already happened (retry after a crash)
    result = fn()
    EFFECTS[key] = result
    return result

CHARGES = []
def reserve(run_id, state):
    state["reservation"] = "res-1"
    return "charge"                    # next step
def charge(run_id, state):
    rec = effect(run_id, {{intent}}, lambda: CHARGES.append("charged") or {"payment_id": "pay-1"})
    state["payment"] = rec["payment_id"]
    return "ship"
def ship(run_id, state):
    state["shipment"] = "shp-1"
    return None                        # done

STEPS = {"reserve": reserve, "charge": charge, "ship": ship}
''',
    intent='"charge"',
)
nb0.md("## The worker\n\nThis is the loop the whole repo elaborates. Fill in the guard and the ordering.")
nb0.code(
    '''
CRASH_AFTER_COMMIT = {"on": False}

def worker(run_id, step, attempt):
    run = RUNS[run_id]
    # (2) stale guard: is this exactly the work the run expects right now?
    if run["status"] != "RUNNING" or run["step"] != step or run["attempts"][step] != {{expected_attempt}}:
        return "stale"
    state = copy.deepcopy(run["state"])            # work on a copy; commit atomically below
    nxt = STEPS[step](run_id, state)
    # --- (1) checkpoint FIRST ---
    run["state"] = state
    run["version"] += 1
    if nxt is None:
        run["status"], run["step"] = "SUCCEEDED", None
        return "done"
    run["step"] = nxt
    run["attempts"][nxt] = run["attempts"].get(nxt, 0) + 1
    if CRASH_AFTER_COMMIT["on"]:
        CRASH_AFTER_COMMIT["on"] = False
        raise RuntimeError("worker died after the checkpoint, before enqueue")
    # --- then enqueue ---
    {{enqueue_call}}
    return "ok"

def drain():
    log = []
    while QUEUE:
        task = QUEUE.popleft()
        try:
            log.append((task[1], worker(*task)))
        except RuntimeError as e:
            log.append((task[1], f"CRASH: {e}"))
    return log
''',
    expected_attempt="attempt",
    enqueue_call='enqueue(run_id, nxt, run["attempts"][nxt])',
)
nb0.md("## Happy path")
nb0.code(
    '''
start("order-1", "reserve")
print(drain())
print(RUNS["order-1"]["status"], RUNS["order-1"]["state"], "charges:", CHARGES)
assert RUNS["order-1"]["status"] == "SUCCEEDED" and CHARGES == ["charged"]
'''
)
nb0.md("## Crash after the checkpoint, before the enqueue\n\nThe run is consistent (`step=ship`, attempt 1) but no task exists. Something must re-drive it: that is the **reaper**. Because the checkpoint already moved on, the reaper's re-enqueue is the *only* task that can execute.")
nb0.code(
    '''
CHARGES.clear()
start("order-2", "reserve")
CRASH_AFTER_COMMIT["on"] = True
print(drain())
run = RUNS["order-2"]
print("after crash:", run["status"], run["step"], run["attempts"], "queue:", list(QUEUE))

def reaper():
    """Re-enqueue the *current* attempt of every RUNNING run. Dedup makes this safe to call often."""
    n = 0
    for run_id, r in RUNS.items():
        if r["status"] == "RUNNING":
            n += enqueue(run_id, r["step"], r["attempts"][r["step"]])
    return n

print("reaper re-enqueued:", reaper())
print(drain())
assert RUNS["order-2"]["status"] == "SUCCEEDED" and CHARGES == ["charged"], "exactly one charge"
'''
)
nb0.md("## Duplicate delivery\n\nCloud Tasks (and every real queue) is at-least-once. Deliver the `charge` task twice and watch the guard reject the second.")
nb0.code(
    '''
CHARGES.clear()
start("order-3", "reserve")
task = QUEUE.popleft(); print(task, worker(*task))            # reserve
task = QUEUE.popleft(); print(task, worker(*task))            # charge (attempt 1)
print("duplicate:", worker(*task))                            # same (run, step, attempt) again
assert worker(*task) == {{dup_outcome}} and CHARGES == ["charged"]
print(drain())
'''
    ,
    dup_outcome='"stale"',
)
nb0.md(
    "## What the real engine adds\n\n"
    "* **Leases** so two replicas can't run the same step concurrently (`Engine.execute_task`).\n"
    "* **Retries** with backoff by bumping the attempt (so the old task becomes stale).\n"
    "* **`Wait`** for human input, **`FanOut`** into child runs, **compensation** for sagas, **budgets**.\n"
    "* **Optimistic concurrency** on `version` instead of the single-threaded dict here.\n\n"
    "Continue with `01_durable_execution`."
)

# =============================================================================
nb1 = NB(
    "01 · Durable execution with the `lra` engine",
    "Same invariants, real engine: leases, optimistic concurrency, explicit retries, a reaper. Everything runs on in-memory "
    "adapters that mimic Firestore/Cloud Tasks semantics, with a controllable clock and chaos hooks.",
)
nb1.code(SETUP)
nb1.md("## A three-step workflow\n\nSteps return `Next`, `Done`, `Wait` or `FanOut`. They mutate `ctx.state`, call the model through `ctx.llm` (budgeted), and do side effects through `ctx.effect` (idempotent).")
nb1.code(
    '''
wf = Workflow("triage", version="1", default_budget=Budget(max_steps=10, max_cost_usd=0.5))

@wf.step(start=True)
def classify(ctx):
    resp = ctx.llm("Classify this ticket as billing/technical/other: " + ctx.input["ticket"])
    ctx.state["category"] = resp.text.strip()
    return {{next_after_classify}}

@wf.step(max_attempts={{attempts}}, backoff_base_s=1.0)
def lookup(ctx):
    resp = ctx.llm("Find the relevant policy for: " + ctx.state["category"])   # may 503 -> retried
    ctx.state["policy"] = resp.text
    return Next("respond")

TICKETS_UPDATED = []
@wf.step()
def respond(ctx):
    rec = ctx.effect({{effect_key}}, lambda: TICKETS_UPDATED.append(ctx.run_id) or {"comment_id": "c-1"})
    return Done({"category": ctx.state["category"], "comment_id": rec["comment_id"]})

routes = {"Classify": "billing", "policy": "Refunds within 14 days."}
engine, runner, clock, store, queue, bus = harness(routes=routes, fail_times=1)   # first model call fails once
engine.registry.register(wf)
''',
    next_after_classify='Next("lookup")',
    attempts="3",
    effect_key='"update-ticket"',
)
nb1.md("## Run it one task at a time\n\n`LocalRunner.step()` delivers one due task, exactly like one Cloud Tasks push.")
nb1.code(
    '''
run = engine.start("triage", {"ticket": "I was charged twice"})
show(store.get(run.run_id))
while runner.step(auto_advance=True):        # auto_advance: jump the clock to a delayed retry
    show(store.get(run.run_id))
r = store.get(run.run_id)
print("trace:", runner.trace)
print("history:", [(h.step, h.attempt, h.status) for h in r.history])
assert r.status == RunStatus.SUCCEEDED and r.attempt_of("classify") == {{classify_attempts}} and TICKETS_UPDATED == [run.run_id]
''',
    classify_attempts="2",
)
nb1.md("The first `classify` attempt hit the simulated 503; the engine bumped the attempt (making the old task stale), enqueued a delayed retry, and the runner fast-forwarded the clock to it. Retries live in run history, not hidden in the queue.")
nb1.md("## Chaos: crash after the checkpoint, before the enqueue\n\nThe chaos hook raises `SimulatedCrash` at a named point. A real crash never releases its lease, so the reaper must wait for it to expire.")
nb1.code(
    '''
crashed = []
def chaos(point, run):
    if point == {{chaos_point}} and run.current_step == "respond" and not crashed:
        crashed.append(run.run_id); raise SimulatedCrash()

engine, runner, clock, store, queue, bus = harness(routes=routes, chaos=chaos, lease_ttl_s=60)
engine.registry.register(wf)
run = engine.start("triage", {"ticket": "app crashes on login"})
try:
    runner.run_until_idle()
except SimulatedCrash:
    print("crashed while", store.get(run.run_id).current_step, "was being enqueued")
r = store.get(run.run_id); show(r); print("lease held by:", r.lease.owner, "until", r.lease.expires_at.time())

runner.run_until_idle()                       # Cloud Tasks redelivers the crashed task -> stale
print("redelivery outcome:", runner.trace[-1][1], "| queue:", len(queue))
print("reap now:", engine.reap()["leases_recovered"])
clock.advance(seconds={{advance_s}})
print("reap after lease expiry:", engine.reap()["leases_recovered"])
runner.run_until_idle()
assert store.get(run.run_id).status == RunStatus.SUCCEEDED
''',
    chaos_point='"after_commit_before_enqueue"',
    advance_s="61",
)
nb1.md("## Crash after the side effect, before the checkpoint\n\nThe most expensive window: the effect happened, the checkpoint didn't. On redelivery the engine re-runs the step, finds the effect record, and skips it.")
nb1.code(
    '''
TICKETS_UPDATED.clear(); crashed.clear()
def chaos2(point, run):
    if point == {{chaos_point}} and run.current_step == "respond" and not crashed:
        crashed.append(run.run_id); raise SimulatedCrash()
engine, runner, clock, store, queue, bus = harness(routes=routes, chaos=chaos2)
engine.registry.register(wf)
run = engine.start("triage", {"ticket": "refund please"})
try:
    runner.run_until_idle()
except SimulatedCrash:
    pass
print("effects applied before crash:", TICKETS_UPDATED)
clock.advance(seconds=61); engine.reap(); runner.run_until_idle()
r = store.get(run.run_id)
print(r.status.value, "attempts of respond:", r.attempt_of("respond"), "effects:", TICKETS_UPDATED)
assert r.status == RunStatus.SUCCEEDED and len(TICKETS_UPDATED) == {{n_effects}}
''',
    chaos_point='"after_step_before_commit"',
    n_effects="1",
)
nb1.md("## Leases: a second worker\n\nTwo replicas receive the same task 50 ms apart. The loser gets `lease-held`, which the HTTP layer maps to **503** so Cloud Tasks retries later.")
nb1.code(
    '''
engine, runner, clock, store, queue, bus = harness(routes=routes)
engine.registry.register(wf)
other = Engine(store=store, queue=queue, bus=bus, llm=engine.llm, clock=clock, workflows=engine.registry, worker_id="worker-B")
run = engine.start("triage", {"ticket": "x"})
task = queue.pop_due()
store.acquire_lease(run.run_id, "worker-A", timedelta(seconds=60), clock.now())      # A is mid-step
print("B tries:", other.execute_task(task))
clock.advance(seconds=61)                                                             # A died
print("B after expiry:", other.execute_task(task))
assert store.get(run.run_id).current_step == {{after_b}}
''',
    after_b='"lookup"',
)
nb1.md("## Takeaways\n\n* Retries are **explicit attempts**; the old task becomes stale by construction.\n* Two crash windows, two recovery mechanisms: reaper (lost enqueue) and effect records (lost checkpoint).\n* The worker is stateless — every replica can execute any step; the lease is the only coordination.")

# =============================================================================
nb2 = NB(
    "02 · Human-in-the-loop: suspend for days, resume from anywhere",
    "A `Wait` outcome parks the run: no task, no lease, no compute. An external event with the matching key resumes it. "
    "Timeouts are absolute timestamps enforced by the reaper.",
)
nb2.code(SETUP)
nb2.md("## An approval gate\n\n`patterns.hitl.request_approval` records what is being approved and returns `Wait`. `approval_decision` reads the event after resume and fails the run on reject/timeout.")
nb2.code(
    '''
from lra.patterns.hitl import request_approval, approval_decision

wf = Workflow("expense", version="1")

@wf.step(start=True)
def draft(ctx):
    ctx.state["amount"] = ctx.input["amount"]
    return Next("gate")

@wf.step()
def gate(ctx):
    if ctx.state["amount"] < 100:
        return Next("pay")                       # small amounts skip the human
    return request_approval(ctx, then={{then_step}}, gate="manager",
                            summary=f"expense of ${ctx.state['amount']}",
                            timeout=timedelta(days={{days}}), on_timeout="fail")

PAID = []
@wf.step()
def pay(ctx):
    if ctx.state["amount"] >= 100:
        decision = approval_decision(ctx, gate="manager")   # raises StepFailed on reject / timeout
        ctx.state["approved_by"] = decision["by"]
    ctx.effect("pay", lambda: PAID.append(ctx.run_id) or {"tx": "t-1"})
    return Done({"paid": ctx.state["amount"], "by": ctx.state.get("approved_by")})

engine, runner, clock, store, queue, bus = harness()
engine.registry.register(wf)
''',
    then_step='"pay"',
    days="3",
)
nb2.md("## Suspend")
nb2.code(
    '''
run = engine.start("expense", {"amount": 900})
runner.run_until_idle()
r = store.get(run.run_id); show(r)
print("wait:", r.wait.model_dump(mode="json"))
print("queue:", len(queue), "| lease:", r.lease, "| approval requested event:", bus.of_type("approval.requested")[0]["summary"])
assert r.status == RunStatus.WAITING and r.wait.key == {{expected_key}}
''',
    expected_key='f"manager:{run.run_id}"',
)
nb2.md("## Resume — idempotent and key-scoped\n\nThe API endpoint `POST /runs/{id}/events` does exactly this. Duplicates and wrong-gate events return `None`, never an error.")
nb2.code(
    '''
wrong = Event(run_id=run.run_id, key="cfo:" + run.run_id, payload={"decision": "approve"})
print("wrong gate:", engine.resume(wrong))
evt = Event(run_id=run.run_id, key=r.wait.key, payload={"decision": {{decision}}, "by": "manager@example.com"})
print("first delivery:", engine.resume(evt).status.value)
print("duplicate:", engine.resume(evt))
runner.run_until_idle()
r = store.get(run.run_id); show(r); print(r.result)
assert r.status == RunStatus.SUCCEEDED and PAID == [run.run_id]
''',
    decision='"approve"',
)
nb2.md("## Rejection is a business failure: no retry, no compensation needed here")
nb2.code(
    '''
run2 = engine.start("expense", {"amount": 5000}); runner.run_until_idle()
engine.resume(Event(run_id=run2.run_id, key=f"manager:{run2.run_id}", payload={"decision": "reject", "by": "cfo", "comment": "no"}))
runner.run_until_idle()
r2 = store.get(run2.run_id); show(r2); print(r2.error)
assert r2.status == RunStatus.FAILED and r2.attempt_of("pay") == {{pay_attempts}}
''',
    pay_attempts="1",
)
nb2.md("## Timeout via the reaper\n\nNothing runs for three days. Then Cloud Scheduler calls `/internal/reap`.")
nb2.code(
    '''
run3 = engine.start("expense", {"amount": 250}); runner.run_until_idle()
print("before:", engine.reap()["waits_timed_out"])
clock.advance(days=3, seconds=1)
print("after 3 days:", engine.reap()["waits_timed_out"])
r3 = store.get(run3.run_id); print(r3.status.value, "|", r3.error)
assert r3.status == RunStatus.{{status}}
''',
    status="FAILED",
)
nb2.md("## Auto-approve on timeout (`on_timeout=\"resume\"`)\n\nOnly for effects you would be comfortable auto-approving. The resumed step sees `{\"timed_out\": True}` in the event payload.")
nb2.code(
    '''
soft = Workflow("soft_gate")
@soft.step(start=True)
def ask(ctx):
    return Wait(key=f"ok:{ctx.run_id}", then="finish", timeout=timedelta(hours=4), on_timeout={{policy}})
@soft.step()
def finish(ctx):
    payload = ctx.state["events"][f"ok:{ctx.run_id}"]["payload"]
    return Done({"auto_approved": bool(payload.get("timed_out"))})
engine.registry.register(soft)
run4 = engine.start("soft_gate"); runner.run_until_idle()
clock.advance(hours=4, seconds=1); engine.reap(); runner.run_until_idle()
print(store.get(run4.run_id).result)
assert store.get(run4.run_id).result == {"auto_approved": True}
''',
    policy='"resume"',
)
nb2.md("## Cancel while waiting\n\nNo task will ever run for a waiting run, so cancel finishes it immediately (and would compensate if anything compensable had completed).")
nb2.code(
    '''
run5 = engine.start("expense", {"amount": 400}); runner.run_until_idle()
engine.cancel(run5.run_id)
print(store.get(run5.run_id).status.value)
print("late approval:", engine.resume(Event(run_id=run5.run_id, key=f"manager:{run5.run_id}", payload={"decision": "approve"})))
'''
)
nb2.md("## The same gate in Cloud Workflows and ADK\n\n* **Cloud Workflows**: `events.create_callback_endpoint` → send the URL to the reviewer → `events.await_callback(timeout=259200)`. See `workflows/research_approval.yaml`.\n* **ADK 2**: a node yields an event with `long_running_tool_ids`; the webhook resumes the invocation with a `FunctionResponse`. See `examples/adk_agent_engine/`.")

# =============================================================================
nb3 = NB(
    "03 · Fan-out/fan-in, saga compensation, bounded reflection, budgets",
    "The research pipeline in `lra.examples` composes every pattern. Here we drive it, break it, and watch it recover.",
)
nb3.code(SETUP)
nb3.md("## Orchestrator–worker: a plan becomes child runs")
nb3.code(
    '''
engine, runner, clock, store, queue, bus = harness()
run = engine.start("research_pipeline", {"goal": "Why do long-running agents need explicit state?"})
runner.step()                                   # plan -> FanOut
parent = store.get(run.run_id); show(parent)
print("fan_in:", parent.fan_in.model_dump())
children = [c for c in store.list_runs() if c.parent_run_id == run.run_id]
print("children:", [(c.run_id, c.status.value, c.budget.max_cost_usd) for c in children])
assert parent.wait.kind == {{wait_kind}} and len(children) == 3
runner.run_until_idle()
parent = store.get(run.run_id); show(parent)
print("child results:", list(parent.state["children"]["results"]))
print("reflection:", parent.state["reflect_loop"]["exit_reason"])
''',
    wait_kind='"children"',
)
nb3.md("Children are **runs**: they have their own ids (`parent--childkey`, so respawning after a crash is idempotent), budgets, retries and history. The parent slept while they ran.")
nb3.md("## Partial failure is data\n\nOne subtopic's source is permanently down. The child fails after its 3 attempts; the aggregator decides that 2/3 is enough.")
nb3.code(
    '''
routes = research_routes()
def flaky(prompt):
    if "Subtopic 2" in prompt:
        raise RuntimeError("source unavailable")
    return "- finding"
routes[r"Research this subtopic"] = flaky
engine, runner, clock, store, queue, bus = harness(routes=routes)
run = engine.start("research_pipeline", {"goal": "partial"}); runner.run_until_idle()
parent = store.get(run.run_id); show(parent)
print("failures:", parent.state["partial_failures"])
failed = [c for c in store.list_runs() if c.parent_run_id == run.run_id and c.status == RunStatus.FAILED]
assert parent.status == RunStatus.WAITING and failed[0].attempt_of("research") == {{child_attempts}}
''',
    child_attempts="3",
)
nb3.md("## Saga: undo in reverse order\n\n`procurement`: reserve stock → charge → book shipment. Shipment fails after the first two succeeded; the engine runs their compensations in reverse, using the stored effect records.")
nb3.code(
    '''
from lra.examples.procurement_saga import ExternalSystems
ExternalSystems.reset()
engine, runner, clock, store, queue, bus = harness()
run = engine.start("procurement", {"sku": "GPU", "qty": 1, "amount": 25000, "flaky_at": "charge_payment", "fail_at": "book_shipment"})
runner.run_until_idle()
r = store.get(run.run_id); show(r); print(r.error)
print("external calls:", [c[0] for c in ExternalSystems.calls])
print("history:", [(h.step, h.kind, h.status) for h in r.history])
assert r.status == RunStatus.COMPENSATED
assert [c[0] for c in ExternalSystems.calls] == {{expected_calls}}
''',
    expected_calls='["reserve_stock", "charge", "refund", "release_stock"]',
)
nb3.md("## Write your own compensable step\n\n`compensating(effect_key, undo)` builds an idempotent compensation that reads the effect record.")
nb3.code(
    '''
from lra.patterns.saga import compensating
CALLS = []
wf = Workflow("ticketing")

@wf.step(start=True, compensate=compensating({{effect_key}}, lambda ctx, rec: CALLS.append(("close", rec["ticket_id"]))))
def open_ticket(ctx):
    rec = ctx.effect({{effect_key}}, lambda: CALLS.append(("open", "T-1")) or {"ticket_id": "T-1"})
    ctx.state["ticket"] = rec["ticket_id"]
    return Next("assign")

@wf.step(max_attempts=1)
def assign(ctx):
    raise StepFailed("no engineer available in region")     # business failure -> compensate

engine.registry.register(wf)
run = engine.start("ticketing"); runner.run_until_idle()
r = store.get(run.run_id); show(r); print(CALLS)
assert r.status == RunStatus.COMPENSATED and CALLS == [("open", "T-1"), ("close", "T-1")]
''',
    effect_key='"open"',
)
nb3.md("## Reflection loop exits, and the budget failing closed\n\nMake the critic never satisfied: the loop must exit on `max iterations`. Then give the run a tiny step budget: it must fail *before* another model call.")
nb3.code(
    '''
engine, runner, clock, store, queue, bus = harness(routes=research_routes(first_score=3, second_score=3))
run = engine.start("research_pipeline", {"goal": "stubborn"}); runner.run_until_idle()
loop = store.get(run.run_id).state["reflect_loop"]
print(loop["exit_reason"], "| scores:", [h["score"] for h in loop["history"]])
assert loop["iteration"] == {{iterations}}

engine, runner, clock, store, queue, bus = harness(routes=research_routes(first_score=1, second_score=1))
run = engine.start("research_pipeline", {"goal": "runaway"}, budget=Budget(max_steps={{max_steps}}, max_cost_usd=10))
runner.run_until_idle()
r = store.get(run.run_id); show(r); print(r.error)
calls_before = len(engine.llm.calls); runner.run_until_idle()
assert r.status == RunStatus.FAILED and "step budget" in r.error and len(engine.llm.calls) == calls_before
''',
    iterations="2",
    max_steps="6",
)
nb3.md("## Deadlines are absolute\n\nA run that waited past its deadline must not publish when the approval finally arrives.")
nb3.code(
    '''
engine, runner, clock, store, queue, bus = harness()
run = engine.start("research_pipeline", {"goal": "late"}, budget=Budget(deadline=clock.now() + timedelta(hours=1)))
runner.run_until_idle()
clock.advance(hours={{hours}})
engine.resume(Event(run_id=run.run_id, key=f"editor:{run.run_id}", payload={"decision": "approve", "by": "ed"}))
runner.run_until_idle()
r = store.get(run.run_id); show(r); print(r.error)
assert r.status == RunStatus.FAILED and "publish" not in r.completed_steps
''',
    hours="2",
)
nb3.md("## Takeaways\n\n* Fan-out = child runs; fan-in = an atomic counter on the parent; partial failure is data.\n* Sagas need effect records, not recomputation, and compensations must be idempotent.\n* Every loop has three exits, and budgets/deadlines gate the *next* model call, not the current one.")


def main() -> None:
    for name, nb in (("00_core_idea", nb0), ("01_durable_execution", nb1), ("02_human_in_the_loop", nb2), ("03_fanout_saga_reflection", nb3)):
        (OUT / "worked").mkdir(parents=True, exist_ok=True)
        (OUT / "practice").mkdir(parents=True, exist_ok=True)
        nbf.write(nb.build(practice=False), OUT / "worked" / f"{name}.ipynb")
        nbf.write(nb.build(practice=True), OUT / "practice" / f"{name}.ipynb")
        print("wrote", name)


if __name__ == "__main__":
    main()
