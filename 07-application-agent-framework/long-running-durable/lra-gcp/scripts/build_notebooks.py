"""Generate the exercise/solution notebook pairs: ``notebooks/NN_name.ipynb`` (blanks) and
``solutions/NN_name.ipynb`` (worked answers, same file name).

Each code cell is written once with ``{{name}}`` holes and a dict of answers.
The solution substitutes the answers; the exercise replaces each hole with
``____``, lists the hole names in a TODO comment and stops at the cell with
``NotImplementedError`` until the learner fills the blanks and deletes that
line (the repo's ``run_notebooks.py --expect-fail`` convention).

Rebuilding unchanged sources is a no-op (see ``write_nb``); ``make notebooks``
then executes the solutions and checks that every exercise stops where it should.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"          # exercises (blanks)
SOLUTIONS = ROOT / "solutions"    # worked answers, same file names
REPO = next(p for p in ROOT.parents if (p / "tools" / "inject_colab_bootstrap.py").is_file())
_spec = importlib.util.spec_from_file_location("inject_colab_bootstrap", REPO / "tools" / "inject_colab_bootstrap.py")
_inject = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_inject)


def _content(d: dict):
    """What a rebuild decides: cell types, sources and tags -- not ids, outputs or execution records."""
    cells = [(c["cell_type"], "".join(c["source"]) if isinstance(c["source"], list) else c["source"],
              {k: v for k, v in c.get("metadata", {}).items() if k != "execution"}) for c in d["cells"]]
    meta = {k: v for k, v in d["metadata"].items() if k != "language_info"}
    return cells, meta, d.get("nbformat"), d.get("nbformat_minor")


def write_nb(nb: nbf.NotebookNode, path: Path) -> bool:
    """Write ``nb`` with the repo's Colab setup cell first, exactly as tools/inject_colab_bootstrap.py
    writes it (so running the injector afterwards is a no-op), and stable cell ids.

    A notebook whose content is unchanged is left alone, so rebuilding is a no-op and the outputs
    that executing the worked notebooks recorded survive; a changed notebook is rewritten without outputs.
    """
    for i, cell in enumerate(nb.cells):
        cell.id = hashlib.sha1(f"{path.stem}/{i}".encode()).hexdigest()[:12]
    d = json.loads(nbf.writes(nb))
    d["cells"].insert(0, _inject.make_cell(path.resolve().parent.relative_to(REPO).as_posix()))
    if path.exists() and _content(json.loads(path.read_text(encoding="utf-8"))) == _content(d):
        return False
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


class NB:
    def __init__(self, title: str, intro: str, requires: tuple[str, ...] = ()) -> None:
        self.cells: list[tuple[str, str, dict[str, str]]] = [("md", f"# {title}\n\n{intro}", {})]
        self.requires = requires       # optional modules; scripts/run_notebooks.py skips the notebook without them

    def md(self, text: str) -> "NB":
        self.cells.append(("md", text.strip("\n"), {}))
        return self

    def code(self, template: str, **answers: str) -> "NB":
        self.cells.append(("code", template.strip("\n"), answers))
        return self

    def build(self, practice: bool) -> nbf.NotebookNode:
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if self.requires:
            nb.metadata["lra"] = {"requires": list(self.requires)}
        for kind, body, answers in self.cells:
            if kind == "md":
                nb.cells.append(nbf.v4.new_markdown_cell(body))
                continue
            src = body
            for name, value in answers.items():
                src = src.replace("{{" + name + "}}", "____" if practice else value)
            if practice and answers:
                names = ", ".join(answers)
                src = (f"# TODO: fill in the blanks: {names}\n"
                       "# YOUR CODE HERE: replace each ____ in this cell, then delete the next line\n"
                       f"raise NotImplementedError(\"fill in the blanks: {names}\")\n" + src)
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
    "This notebook shows the *idea* before the engine. We use only the standard library to build a run store, a task queue "
    "and a worker. Then we break the loop with a crash and a duplicate delivery. Everything depends on the three invariants of "
    "the engine (`docs/primer.md` §2):\n\n"
    "1. **Write the checkpoint before you enqueue the task.**\n2. **`(run, step, attempt)` identifies the work. Any other task is stale.**\n"
    "3. **Record side effects before the checkpoint, with the intent as the key.**",
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
nb0.md("## Steps with an idempotent side effect\n\n`effect()` runs `fn` at most one time for each `(run, intent)`. Look at the order: `effect()` records the effect *before* the caller writes the checkpoint.")
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
nb0.md("## The worker\n\nThe whole repository builds on this loop. Fill in the guard and the order of the operations.")
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
nb0.md("## Crash after the checkpoint, before the enqueue\n\nThe run is consistent (`step=ship`, attempt 1), but no task exists. Something must re-drive the run, and that is the job of the **reaper**. The checkpoint already moved on, so the task that the reaper enqueues again is the *only* task that can execute.")
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
nb0.md("## Duplicate delivery\n\nCloud Tasks (and every real queue) gives at-least-once delivery. Deliver the `charge` task two times. Then look at how the guard rejects the second delivery.")
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
    "* **Leases**, so that two replicas cannot run the same step at the same time (`Engine.execute_task`).\n"
    "* **Retries** with backoff, through an increase of the attempt, so that the old task becomes stale.\n"
    "* **`Wait`** for input from a person, **`FanOut`** into child runs, **compensation** for sagas, and **budgets**.\n"
    "* **Optimistic concurrency** on `version`, in place of the single-threaded dict of this notebook.\n\n"
    "Continue with `01_durable_execution`."
)

# =============================================================================
nb1 = NB(
    "01 · Durable execution with the `lra` engine",
    "This notebook has the same invariants and a real engine: leases, optimistic concurrency, explicit retries and a reaper. "
    "Everything runs on in-memory adapters that copy the semantics of Firestore and Cloud Tasks. Everything also runs with "
    "a clock that you control and with chaos hooks.",
)
nb1.code(SETUP)
nb1.md("## A three-step workflow\n\nSteps return `Next`, `Done`, `Wait` or `FanOut`. A step changes `ctx.state`. It calls the model through `ctx.llm`. Each call through `ctx.llm` counts against the budget. The step does side effects through `ctx.effect`, and each side effect through `ctx.effect` is idempotent.")
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
nb1.md("## Run it one task at a time\n\n`LocalRunner.step()` delivers one task that is due, the same as one push from Cloud Tasks.")
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
nb1.md("The first `classify` attempt got the simulated 503. The engine increased the attempt. The new attempt made the old task stale. Then the engine enqueued a delayed retry, and the runner moved the clock forward to that retry. The retries are in the run history. The queue does not hide them.")
nb1.md("## Chaos: crash after the checkpoint, before the enqueue\n\nThe chaos hook raises `SimulatedCrash` at a named point. A real crash never releases its lease. Thus the reaper must wait until the lease expires.")
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
nb1.md("## Crash after the side effect, before the checkpoint\n\nThis is the crash window with the highest cost. The effect occurred, but the checkpoint did not. When the queue delivers the task again, the engine runs the step again. It finds the effect record and does not do the effect again.")
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
nb1.md("## Leases: a second worker\n\nTwo replicas receive the same task with 50 ms between them. The replica that loses gets `lease-held`. The HTTP layer returns `lease-held` as **503**, so Cloud Tasks retries later.")
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
nb1.md("## Takeaways\n\n* Retries are **explicit attempts**. The design makes the old task stale.\n* There are two crash windows and two recovery mechanisms: the reaper (for a lost enqueue) and the effect records (for a lost checkpoint).\n* The worker is stateless. Every replica can execute any step. The lease is the only coordination.")

# =============================================================================
nb2 = NB(
    "02 · Human-in-the-loop: suspend for days, resume from anywhere",
    "A `Wait` outcome makes the run wait. The run then has no task, no lease and no compute. An external event with the key "
    "of the wait resumes the run. Timeouts are absolute timestamps, and the reaper enforces them.",
)
nb2.code(SETUP)
nb2.md("## An approval gate\n\n`patterns.hitl.request_approval` records what the approval is for, and returns `Wait`. After the resume, `approval_decision` reads the event. If the event is a rejection or a timeout, `approval_decision` fails the run.")
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
nb2.md("## Resume — idempotent and key-scoped\n\nThe API endpoint `POST /runs/{id}/events` does this same thing. A duplicate event and an event for an incorrect gate return `None`, never an error.")
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
nb2.md("## Auto-approve on timeout (`on_timeout=\"resume\"`)\n\nUse `on_timeout=\"resume\"` only for effects where an automatic approval is acceptable to you. The resumed step sees `{\"timed_out\": True}` in the event payload.")
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
nb2.md("## Cancel while waiting\n\nNo task will ever run for a run that waits. Thus a cancel ends the run immediately. If a step with a compensation completed before the cancel, the cancel also compensates that step.")
nb2.code(
    '''
run5 = engine.start("expense", {"amount": 400}); runner.run_until_idle()
engine.cancel(run5.run_id)
print(store.get(run5.run_id).status.value)
print("late approval:", engine.resume(Event(run_id=run5.run_id, key=f"manager:{run5.run_id}", payload={"decision": "approve"})))
'''
)
nb2.md("## The same gate in Cloud Workflows and ADK\n\n* **Cloud Workflows**: call `events.create_callback_endpoint`, then send the URL to the reviewer, then call `events.await_callback(timeout=259200)`. See `workflows/research_approval.yaml`.\n* **ADK 2**: a node yields an event with `long_running_tool_ids`. The webhook resumes the invocation with a `FunctionResponse`. See `examples/adk_agent_engine/`.")

# =============================================================================
nb3 = NB(
    "03 · Fan-out/fan-in, saga compensation, bounded reflection, budgets",
    "The research pipeline in `lra.examples` uses all of the patterns together. In this notebook, we run the pipeline, break it "
    "and look at how it recovers.",
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
nb3.md("The children are **runs**. Each child has its own id (`parent--childkey`), budget, retries and history. Because of this id, a second spawn of the children after a crash is idempotent. The parent waited while the children ran.")
nb3.md("## Partial failure is data\n\nThe source of one subtopic is permanently down. The child fails after its 3 attempts. The aggregator decides that 2/3 is sufficient.")
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
nb3.md(
    "## The fan-in counter, by hand\n\n"
    "Children finish in any order, and their completion notices have at-least-once delivery. The parent keeps a "
    "counter. The write that makes the counter reach `expected` wakes the aggregator.\n\n"
    "Write the rule that the store applies atomically (`InMemoryStateStore.record_child_result` under a lock, "
    "Firestore in a transaction). Record each child **one time**. When the counter gets to `expected`, report "
    "*all done* for every notice, and for a late duplicate too. The wake-up of that duplicate is a named, de-duplicated "
    "task, so another try causes no harm. If the first wake-up did not arrive and the store answers *not done*, it is "
    "possible that the parent stays stuck."
)
nb3.code(
    '''
def record(fan, child_key, result):
    """fan = {"expected": n, "completed": 0, "results": {}}. Returns True iff every child is in."""
    if {{not_seen}}:
        fan["results"][child_key] = result
        fan["completed"] += 1
    return {{all_done}}

fan = {"expected": 3, "completed": 0, "results": {}}
notices = [("a", 1), ("b", 2), ("a", 1), ("c", 3), ("b", 2)]          # "a" twice, and a late duplicate of "b"
seen = [record(fan, k, v) for k, v in notices]
print(seen, fan)
assert seen == [False, False, False, True, True] and fan["completed"] == 3 and fan["results"] == {"a": 1, "b": 2, "c": 3}
''',
    not_seen='child_key not in fan["results"]',
    all_done='fan["completed"] >= fan["expected"]',
)
nb3.md("## Saga: undo in reverse order\n\n`procurement` has three steps: reserve stock, charge, and book the shipment. The shipment fails after the first two steps succeeded. The engine then runs their compensations in reverse order, and it uses the stored effect records.")
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
nb3.md("## Write your own compensable step\n\n`compensating(effect_key, undo)` makes an idempotent compensation that reads the effect record.")
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
nb3.md("## Reflection loop exits, and the budget failing closed\n\nMake the critic never satisfied. The loop must then exit on `max iterations`. After that, give the run a small step budget. The run must fail *before* another model call.")
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
nb3.md("## Deadlines are absolute\n\nA run that waited beyond its deadline must not publish when the approval arrives at last.")
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
nb3.md("## Takeaways\n\n* A fan-out is a set of child runs. A fan-in is an atomic counter on the parent. Partial failure is data.\n* Sagas need the effect records, not a new calculation. Compensations must be idempotent.\n* Every loop has three exits. Budgets and deadlines are a gate for the *next* model call, not for the current one.")


# =============================================================================
nb4 = NB(
    "04 · The same patterns with ADK 2 `Workflow` (optional: the `adk` extra)",
    "Google's ADK 2 gives you these primitives as part of the framework:\n\n"
    "* **interrupts** (`RequestInput`)\n* **resumability** (`ResumabilityConfig`)\n"
    "* **re-run or complete on resume** (`rerun_on_resume`, for the node that interrupted)\n* **routing** (`ctx.route`)\n"
    "* **session stores** that you can replace (from SQLite to Cloud SQL)\n\n"
    "This notebook runs the ticket-queue graph in `examples/adk_ticket_queue/nightly_workflow.py` *without a model*, so "
    "that you can see the mechanics. `use_model=True` puts Gemini in the `plan` node.\n\n"
    "This notebook needs `pip install -e \".[adk]\"` (ADK 2 and the Google Cloud clients, about 220 MB, measured 2026-09-26, verify). "
    "If you do not install the `adk` extra, `scripts/run_notebooks.py` skips this notebook. Every other notebook runs on the default install.\n\n"
    "```mermaid\nflowchart LR\n  S((START)) --> B[agree_budget<br/>RequestInput 'budget'<br/>rerun_on_resume=True]\n"
    "  B --> P[plan<br/>judgement]\n  P --> Q[queue_up<br/>side effect, runs once]\n"
    "  Q --> C[check_front<br/>RequestInput 'wake'<br/>rerun_on_resume=True]\n  C -- ready --> Y[buy<br/>idempotency key]\n"
    "  C -- sold_out --> A[abandon]\n```\n(The cells use top-level `await`, because Jupyter already runs an event loop.)",
    requires=("google.adk",),
)
nb4.code(
    '''
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path("..").resolve() / "examples" / "adk_ticket_queue"))   # from notebooks/ or solutions/

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from nightly_workflow import ASK_BUDGET, WAKE, Venue, build_app, check_adk_workflow, run_until_interrupt

venue = Venue()
svc = InMemorySessionService()          # production: DatabaseSessionService("postgresql+asyncpg://...") or VertexAiSessionService
runner = Runner(app=build_app(venue), session_service=svc)
session = await svc.create_session(app_name="nightly_app", user_id="u1",
                                   state={"events": [{"id": "ams-tue", "weekday": "Tuesday"}, {"id": "ams-sat", "weekday": "Saturday"}]})
async def state(s=svc, sid=None):
    return (await s.get_session(app_name="nightly_app", user_id="u1", session_id=sid or session.id)).state

r1 = await run_until_interrupt(runner, "u1", session.id, text="Get us two tickets")
print("stopped on:", r1["interrupts"], "| venue calls:", venue.calls)
assert r1["interrupts"] == [ASK_BUDGET]
'''
)
nb4.md("The graph asked a question and **stopped**. The invocation is in the session store, and the Python process can stop at this point.\n\nTo answer the question, resume the *same invocation* with a `FunctionResponse` that has the interrupt id as its id:")
nb4.code(
    '''
r2 = await run_until_interrupt(runner, "u1", session.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 250}})
print("stopped on:", r2["interrupts"], "| venue calls:", venue.calls, "| state:", {k: v for k, v in (await state()).items() if k != "events"})
assert r2["interrupts"] == [WAKE] and venue.calls == ["join_queue"]
'''
)
nb4.md("Four nodes ran from one answer. `plan` selected Saturday (the weekend rule), `queue_up` took **one** ticket, and `check_front` found us at #14,203 and waited on `wake`.\n\n## Wake-ups (what Cloud Scheduler → Pub/Sub → `/wake` does)")
nb4.code(
    '''
ticket = (await state())["ticket"]
venue.advance(ticket, 10_000)                        # the queue moves while nothing of ours runs
r3 = await run_until_interrupt(runner, "u1", session.id, invocation_id=r2["invocation_id"], answers={WAKE: {"ok": True}})
print("wake 1 -> stopped on:", r3["interrupts"], "| join_queue calls:", venue.calls.count("join_queue"), "| position:", venue.position(ticket))
venue.advance(ticket, 10_000)
r4 = await run_until_interrupt(runner, "u1", session.id, invocation_id=r3["invocation_id"], answers={WAKE: {"ok": True}})
print("wake 2 -> stopped on:", r4["interrupts"], "| venue calls:", venue.calls)
print("order:", (await state())["order"])
assert venue.calls == ["join_queue", "purchase"] and r4["interrupts"] == []
'''
)
nb4.md("`queue_up` never ran again during the wake-ups. A resume continues from the node that interrupted, and ADK does not replay the nodes that already finished. `check_front` is the node that interrupted. It ran again on every wake-up because it has `rerun_on_resume=True`. With `False`, ADK marks it complete, uses the resume input (`{\"ok\": true}`) as its output and does not look at the queue. `buy` executed one time, with an idempotency key.\n\n## Staleness guard: the world changed while we waited")
nb4.code(
    '''
venue2 = Venue(); svc2 = InMemorySessionService(); runner2 = Runner(app=build_app(venue2), session_service=svc2)
s2 = await svc2.create_session(app_name="nightly_app", user_id="u1", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
a = await run_until_interrupt(runner2, "u1", s2.id, text="go")
b = await run_until_interrupt(runner2, "u1", s2.id, invocation_id=a["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
t2 = (await state(svc2, s2.id))["ticket"]
venue2.advance(t2, 99_999); venue2.sell_out("ams-sat")
await run_until_interrupt(runner2, "u1", s2.id, invocation_id=b["invocation_id"], answers={WAKE: {"ok": True}})
print("calls:", venue2.calls, "| orders:", venue2.orders, "| routed to abandon:", (await state(svc2, s2.id))["order"] is None)
assert venue2.orders == {} and "purchase" not in venue2.calls
'''
)
nb4.md("## The classic mistake: a new invocation instead of a resume\n\nA message in the chat box (no `invocation_id`) starts a **new** invocation. The graph replays from START. When you answer the budget question again, the graph takes a second queue ticket.")
nb4.code(
    '''
venue3 = Venue(); svc3 = InMemorySessionService(); runner3 = Runner(app=build_app(venue3), session_service=svc3)
s3 = await svc3.create_session(app_name="nightly_app", user_id="u1", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
a = await run_until_interrupt(runner3, "u1", s3.id, text="go")
await run_until_interrupt(runner3, "u1", s3.id, invocation_id=a["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
c = await run_until_interrupt(runner3, "u1", s3.id, text="where am I in line?")       # WRONG: a new invocation
await run_until_interrupt(runner3, "u1", s3.id, invocation_id=c["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
print("join_queue calls:", venue3.calls.count("join_queue"), "<- two tickets")
assert venue3.calls.count("join_queue") == 2
'''
)
nb4.md(
    "## Your turn: build the graph\n\n"
    "Complete the nodes. `check_adk_workflow` runs your graph through a budget answer and three wake-ups:\n\n"
    "* `agree_budget` interrupts with `interrupt_id=ASK_BUDGET` until the budget arrives, and **runs again on resume**.\n"
    "* `queue_up` is a side effect: one ticket for all of the wake-ups. It finishes before any interrupt, so a resume of the same invocation never replays it.\n"
    "* `check_front` interrupts while we are still in the queue. It must examine the world again on every wake-up (`rerun_on_resume`?). At the front, it sets the route to `\"ready\"` if at least two seats are still available, and to `\"sold_out\"` if not.\n"
    "* Connect `START → agree_budget → plan → queue_up → check_front → {\"ready\": buy, \"sold_out\": abandon}`."
)
nb4.code(
    '''
from google.adk.events.request_input import RequestInput
from google.adk.workflow import START, Workflow, node

def build_workflow(venue):
    @node(rerun_on_resume=True)
    def agree_budget(ctx):
        said = ((ctx.resume_inputs or {}).get(ASK_BUDGET) or {}).get("budget")
        if not said:
            return {{ask_budget}}
        ctx.state["budget_per_seat"] = float(said)
        return {"budget_per_seat": float(said)}

    @node
    def plan(ctx):
        ctx.state["event_id"] = ctx.state["events"][0]["id"]
        return {"event_id": ctx.state["event_id"]}

    @node                                            # finished before any interrupt: a resume never replays it
    def queue_up(ctx):
        t = venue.join_queue(ctx.state["event_id"], idempotency_key=f"{ctx.session.id}:{ctx.state['event_id']}")
        ctx.state["ticket"] = t["ticket"]
        return t

    @node(rerun_on_resume={{check_rerun}})           # re-check the world, or take the wake-up payload as the answer?
    def check_front(ctx):
        pos = venue.position(ctx.state["ticket"])
        if pos > 0:
            return RequestInput(interrupt_id=WAKE, message=f"Still at #{pos}")
        left = venue.seats_left(ctx.state["event_id"])     # staleness guard: re-verify before acting
        ctx.route = {{route}}
        return {"position": pos, "seats_left": left}

    @node
    def buy(ctx):
        order = venue.purchase(ctx.state["event_id"], 2, idempotency_key=f"{ctx.session.id}:{ctx.state['event_id']}:purchase")
        ctx.state["order"] = order
        return order

    @node
    def abandon(ctx):
        ctx.state["order"] = None
        return {"abandoned": True}

    return Workflow(name="practice", edges=[{{edges}}])

print(check_adk_workflow(build_workflow))
''',
    ask_budget='RequestInput(interrupt_id=ASK_BUDGET, message="What are you willing to spend per seat?")',
    check_rerun="True",
    route='"ready" if left >= 2 else "sold_out"',
    edges='(START, agree_budget, plan, queue_up, check_front, {"ready": buy, "sold_out": abandon})',
)
nb4.md(
    "## Deploying this shape on Google Cloud\n\n"
    "| Local | Cloud | Change |\n|---|---|---|\n"
    "| `InMemorySessionService` | Cloud SQL (Postgres) through `DatabaseSessionService(\"postgresql+asyncpg://...?host=/cloudsql/...\")`, or Agent Runtime Sessions | a connection string |\n"
    "| `adk web` | one Cloud Run service (`examples/adk_ticket_queue/main.py`) | a Dockerfile and an entry point |\n"
    "| you call `run_until_interrupt` | Cloud Scheduler, then Pub/Sub, then a push to `POST /wake` | `trigger_sources=[\"pubsub\"]` and the custom `/wake` |\n"
    "| `Venue` | the real API with an `Idempotency-Key` header | none |\n\n"
    "The built-in Pub/Sub trigger route of ADK **creates a new session for each message**. A long-running run needs a wake "
    "endpoint that resumes a session that *already exists*, and `main.py` adds this endpoint. The design questions for this "
    "notebook are in the drills of the topic primer (`../../PRIMER.md` §11, part D).\n\n"
    "## Takeaways\n\n* `RequestInput` is one primitive for two types of wait: a wait for a person and a wait for the world.\n"
    "* `rerun_on_resume` decides what a resumed node does. With `True`, the node runs again and examines the world again. With `False`, the node uses the resume input as its answer. Side effects stay safe because ADK does not replay the finished nodes and the side effects have idempotency keys.\n"
    "* Resume the invocation. Do not start a new one.\n* A prompt cannot wake itself. Something with a clock must call the endpoint."
)

# =============================================================================
nb5 = NB(
    "05 · A model-chosen tool loop, and Mistral as the provider",
    "In notebooks 01–03, the model writes text and the workflow sets the order of the steps. In this notebook, the model "
    "selects the *next action*: a tool call or a final answer. Thus the model call becomes a non-deterministic side effect "
    "(primer §1.3). If you ask again after a crash, it is possible that the model selects a different call. The solution is to "
    "**write the decision to the journal before you act on it**. Then act at most one time, under a stable idempotency key.\n\n"
    "In the second half of the notebook, Mistral becomes the model provider (offline, with a fake client). The second half also shows the "
    "same loop on Mistral Workflows. There, the platform gives the journal, the retries and the wait.",
)
nb5.code(SETUP)
nb5.code(
    '''
from lra.examples.tool_agent import PaymentAPI, ScriptedDecider, make_tool_agent

def tool_harness(wf, chaos=None):
    clock, store, bus = FakeClock(), InMemoryStateStore(), InMemoryEventBus()
    queue = InMemoryTaskQueue(clock)
    engine = Engine(store=store, queue=queue, bus=bus, llm=FakeLLM(), clock=clock, workflows=[wf],
                    worker_id="worker-A", lease_ttl=timedelta(seconds=60), chaos=chaos)
    return engine, LocalRunner(engine, queue, clock), clock, store

def show_journal(run):
    for e in run.state.get("journal", []):
        print("   ", e)
'''
)
nb5.md("## Your turn: the two effect records of a turn\n\n`decide` records the decision of the model with `ctx.effect` *before* anything acts on it. `act` runs the tool at most one time. It gives the tool an idempotency key (`run_id:N`, where N is the position of the intent in the journal). The key is stable across retries and unique across runs. Fill in the two keys.")
nb5.code(
    '''
def build_loop(decider, tools):
    wf = Workflow("my_loop", default_budget=Budget(max_steps=10))

    @wf.step(start=True)
    def decide(ctx):
        journal = ctx.state.setdefault("journal", [])
        n = len(journal)
        d = ctx.effect({{decide_key}}, lambda: decider.decide(ctx.input["goal"], journal))
        journal.append({"type": "decision", **d})
        if "final" in d:
            return Done(d["final"])
        journal.append({"type": "intent", "tool": d["tool"], "args": d["args"], "key": {{idem_key}}, "done": False})
        return Next("act")

    @wf.step()
    def act(ctx):
        journal = ctx.state["journal"]
        intent, n = journal[-1], len(journal) - 1
        intent["result"] = ctx.effect(f"act:{n}", lambda: tools[intent["tool"]](**intent["args"], key=intent["key"]))
        intent["done"] = True
        return Next("decide")

    return wf

# A model that would choose differently if asked again, and a crash right after the first decision.
crashed = []
def chaos(point, run):
    if point == "after_step_before_commit" and run.current_step == "decide" and not crashed:
        crashed.append(run.run_id); raise SimulatedCrash()

pay = PaymentAPI()
decider = ScriptedDecider([{"tool": "charge", "args": {"amount": 42}}, {"tool": "charge", "args": {"amount": 99}}, {"final": "done"}])
engine, runner, clock, store = tool_harness(build_loop(decider, {"charge": pay}), chaos=chaos)
run = engine.start("my_loop", {"goal": "pay invoice 42"})
try:
    runner.run_until_idle()
except SimulatedCrash:
    print("crashed after the model decided, before the checkpoint")
clock.advance(seconds=61); engine.reap(); runner.run_until_idle()
r = store.get(run.run_id); show(r); show_journal(r)
print("charges:", pay.charges, "| model calls:", decider.calls)
assert pay.charges == {f"{run.run_id}:1": 42.0, f"{run.run_id}:3": 99.0} and decider.calls == 3
''',
    decide_key='f"decide:{n}"',
    idem_key='f"{ctx.run_id}:{n + 1}"',
)
nb5.md("If there is no decision record, the retry asks the model again and acts on a different answer. Here, that answer is `99` in place of the `42` that the model selected first. Then the world and the run do not agree about the decision of the agent. With the decision record, the `decide` step asked the model three times in all. The answers were `42` (recorded one time and used again by the retry), then `99`, then `final`.\n\n## Crash after the charge\n\nThe library version, `lra.examples.tool_agent.make_tool_agent`, is the same loop with an approval gate added. Cause a crash immediately after the charge. The retry executes *the same intent under the same key* again, finds the effect record and never asks the model again.")
nb5.code(
    '''
crashed = []
def chaos(point, run):
    if point == "after_step_before_commit" and run.current_step == "act" and not crashed:
        crashed.append(run.run_id); raise SimulatedCrash()

pay, decider = PaymentAPI(), ScriptedDecider([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
engine, runner, clock, store = tool_harness(make_tool_agent(decider, {"charge": pay}, gated=()), chaos=chaos)
run = engine.start("tool_agent", {"goal": "pay invoice 42"})
try:
    runner.run_until_idle()
except SimulatedCrash:
    print("crashed after the charge, before the checkpoint | charges so far:", pay.charges)
clock.advance(seconds=61); engine.reap(); runner.run_until_idle()
r = store.get(run.run_id); show(r)
print("charges:", pay.charges, "| payment API calls:", pay.calls, "| model calls:", decider.calls)
assert pay.calls == 1 and decider.calls == 2 and r.status == RunStatus.SUCCEEDED
'''
)
nb5.md("## When the model gets it wrong\n\nTwo types of mistake are for the model to correct, not for the engine. The first is a tool name that the model made up. The second is a tool that answers with a declared, non-retryable error (`ToolError`: incorrect arguments, no such invoice). The loop writes either one to the journal as the result of the call, and asks the model again. The run does not fail. Any other exception (a timeout, a 503) is an infrastructure problem, and the engine retries the step under the same key.")
nb5.code(
    '''
from lra.examples.tool_agent import ToolError

def lookup(invoice, key):
    if invoice != "INV-1042":
        raise ToolError(f"no invoice {invoice}")
    return {"invoice": invoice, "amount": 42}

pay = PaymentAPI()
decider = ScriptedDecider([{"tool": "refund", "args": {"amount": 42}},           # a tool that does not exist
                           {"tool": "lookup", "args": {"invoice": "INV-1024"}},  # a typo in the argument
                           {"tool": "lookup", "args": {"invoice": "INV-1042"}},
                           {"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
engine, runner, clock, store = tool_harness(make_tool_agent(decider, {"charge": pay, "lookup": lookup}, gated=()))
run = engine.start("tool_agent", {"goal": "pay invoice INV-1042"})
runner.run_until_idle()
r = store.get(run.run_id); show(r); show_journal(r)
assert r.status == RunStatus.SUCCEEDED and decider.calls == 5
assert r.state["journal"][1]["result"] == {"error": "unknown tool 'refund'"}
assert r.state["journal"][3]["result"] == {"error": "no invoice INV-1024"}
'''
)
nb5.md("## Approve what you execute\n\nA gated tool makes the run wait, with the *exact* proposed call in the journal. An approval executes that call. A rejection becomes the result of the tool, so the model can select again. A double click is a no-op.")
nb5.code(
    '''
pay = PaymentAPI()
decider = ScriptedDecider([{"tool": "charge", "args": {"amount": 4200}}, {"final": "Not paid: vendor not onboarded."}])
engine, runner, clock, store = tool_harness(make_tool_agent(decider, {"charge": pay}))
run = engine.start("tool_agent", {"goal": "pay invoice 4200"})
runner.run_until_idle()
r = store.get(run.run_id); show(r); show_journal(r)
reject = Event(run_id=run.run_id, key=r.wait.key, payload={"decision": "reject", "by": "cfo", "comment": "vendor not onboarded"})
print("first:", engine.resume(reject) is not None, "| second:", engine.resume(reject))
runner.run_until_idle()
r = store.get(run.run_id); show(r); show_journal(r)
assert pay.charges == {} and r.result == {"result": "Not paid: vendor not onboarded."}
'''
)
nb5.md("## Swap the model: Mistral decides, by function calling\n\n`lra.adapters.mistral.MistralDecider` changes the journal into a chat. The chat has the goal first. Then it has every earlier decision as an assistant tool call with its recorded result. The adapter then asks for the next call with `tool_choice=\"auto\"`.\n\nThe fake client below uses the response shape of the SDK, so this cell runs offline. With `pip install -e \".[mistral]\"` and `MISTRAL_API_KEY`, `MistralDecider(TOOL_SPECS)` calls the real API (`python -m lra.adapters.mistral.demo live`). A tool-call id for Mistral must have exactly nine alphanumeric characters, so the adapter hashes the journal key.")
nb5.code(
    '''
from types import SimpleNamespace as NS
from lra.adapters.mistral import DEFAULT_INSTRUCTIONS, TOOL_SPECS, MistralDecider, MistralLLM, to_messages, tool_call_id

class FakeMistral:
    """Scripted chat completions in the SDK's response shape; keeps every request."""
    def __init__(self, script):
        self.script, self.requests, self.chat = list(script), [], self
    def complete(self, **kw):
        self.requests.append(kw)
        item = self.script.pop(0)
        if "tool" in item:
            msg = NS(content="", tool_calls=[NS(id="abc123def", type="function",
                                                  function=NS(name=item["tool"], arguments=json.dumps(item["args"])))])
        else:
            msg = NS(content=item["final"], tool_calls=None)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=900, completion_tokens=40))

fake = FakeMistral([{"tool": "charge", "args": {"amount": 42}}, {"final": "Charged 42."}])
pay = PaymentAPI()
engine, runner, clock, store = tool_harness(make_tool_agent(MistralDecider(TOOL_SPECS, model="mistral-small-latest", client=fake), {"charge": pay}))
run = engine.start("tool_agent", {"goal": "Pay supplier invoice INV-1042: 42 SGD."})
runner.run_until_idle()
r = store.get(run.run_id)
engine.resume(Event(run_id=run.run_id, key=r.wait.key, payload={"decision": "approve", "by": "cfo"}))
runner.run_until_idle()
r = store.get(run.run_id); show(r)
print("second prompt:", [m["role"] for m in fake.requests[1]["messages"]])
print("the tool result the model saw:", fake.requests[1]["messages"][-1]["content"])
assert r.status == RunStatus.SUCCEEDED and pay.charges == {f"{run.run_id}:1": 42.0}
'''
)
nb5.md("## Your turn: the journal as the prompt\n\nBuild `to_messages` again. Each decision becomes an assistant tool call. The intent that comes immediately after the decision in the journal becomes the tool message for that call. The content of that message is the recorded result of the intent, or `{\"pending\": true}` if the intent has not run. Fill in the tool-call id and the content of the tool message.")
nb5.code(
    '''
def my_to_messages(goal, journal):
    messages = [{"role": "system", "content": DEFAULT_INSTRUCTIONS}, {"role": "user", "content": goal}]
    for i, step in enumerate(journal):
        if step["type"] != "decision" or "final" in step:
            continue
        intent = journal[i + 1]
        cid = {{cid}}
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": step["tool"], "arguments": json.dumps(step["args"])}}]})
        messages.append({"role": "tool", "tool_call_id": cid, "name": step["tool"], "content": json.dumps({{content}})})
    return messages

journals = [
    [],
    r.state["journal"],
    [{"type": "decision", "tool": "charge", "args": {"amount": 1}},
     {"type": "intent", "tool": "charge", "args": {"amount": 1}, "key": "r:1", "done": False}],
    [{"type": "decision", "tool": "charge", "args": {"amount": 5}},
     {"type": "intent", "tool": "charge", "args": {"amount": 5}, "key": "r:1", "done": True, "result": {"rejected": "no"}},
     {"type": "decision", "final": "stopped"}],
]
for j in journals:
    assert my_to_messages("goal", j) == to_messages("goal", j), j
print("my_to_messages matches the adapter on", len(journals), "journals; ids look like", tool_call_id("r:1"))
''',
    cid='tool_call_id(intent["key"])',
    content='intent.get("result") if intent["done"] else {"pending": True}',
)
nb5.md("## Mistral as the engine's model\n\n`MistralLLM` implements the same `LLM` port as `GeminiLLM` and `FakeLLM`. Thus every workflow in `lra.examples` runs on it with no change. The usage becomes cost for the budget of the run (the prices in the adapter are illustrative).")
nb5.code(
    '''
fake = FakeMistral([{"final": "billing"}])
wf = Workflow("classify")
@wf.step(start=True)
def classify(ctx):
    return Done(ctx.llm("Classify this ticket as billing/technical/other: I was charged twice").text)
clock, store = FakeClock(), InMemoryStateStore(); queue = InMemoryTaskQueue(clock)
engine = Engine(store=store, queue=queue, bus=InMemoryEventBus(), llm=MistralLLM(model="mistral-small-latest", client=fake),
                clock=clock, workflows=[wf])
run = engine.start("classify", {}); LocalRunner(engine, queue, clock).run_until_idle()
r = store.get(run.run_id)
print(r.status.value, r.result, "| tokens:", r.budget.tokens_used, "| cost (illustrative prices): $%.6f" % r.budget.cost_usd)
assert r.result == "billing" and r.budget.tokens_used == 940
'''
)
nb5.md(
    "## The same loop on Mistral Workflows\n\n"
    "`lra/adapters/mistral/workflow.py` is this loop on Mistral Workflows, with Temporal below it. The infrastructure code "
    "goes away: there is no store, no queue, no lease and no reaper. In their place:\n\n"
    "* The **event history** of the execution is the store.\n"
    "* The platform records every **activity** (the model call and the charge) before the activity runs. The platform retries the activity "
    "on failure, and never runs the activity again after the activity is complete.\n"
    "* `workflow.wait_condition(...)` and a **signal** are the approval gate, with a timeout.\n"
    "* A loop bound in deterministic workflow code and `execution_timeout` are the budget.\n\n"
    "Note one detail that is easy to miss. An *unexpected* exception in workflow code fails only the workflow **task**. The platform "
    "retries that task until you repair the code and deploy it again. To fail an execution on purpose, raise `WorkflowError`.\n\n"
    "The Mistral Workflows loop needs Python 3.12–3.14 and the `mistral` extra (`mistralai-workflows` 3.15 declares `Requires-Python >=3.12,<3.15`, "
    "checked 2026-09-26, verify). The loop also needs a local Temporal dev server, which the SDK downloads on first use. "
    "`python -m lra.adapters.mistral.demo workflow` runs the approval, crash-after-charge and timeout scenarios. "
    "`tests/test_mistral_workflow.py` has a test for each of these scenarios. If you installed the extra, the cell below shows the gate."
)
nb5.code(
    '''
import importlib.util, inspect, sys
if sys.version_info >= (3, 12) and importlib.util.find_spec("mistralai") and importlib.util.find_spec("mistralai.workflows"):
    from lra.adapters.mistral import workflow as mw
    print(inspect.getsource(mw.InvoiceAgent.run))
else:
    print("Mistral Workflows is not installed here (Python 3.12+ and the mistral extra); read lra/adapters/mistral/workflow.py.")
'''
)
nb5.md("## Takeaways\n\n* A model-chosen action is two side effects. Record the *decision*, then the *act*, each under its own key.\n* On every retry, the journal is the prompt. It contains recorded facts, never the memory of the model.\n* The provider is one adapter (`MistralDecider`, `MistralLLM`). The invariants do not change. A durable-execution platform (Mistral Workflows, Temporal, Cloud Workflows) gives you the invariants, and you still pass idempotency keys downstream.")


NOTEBOOKS = (("00_core_idea", nb0), ("01_durable_execution", nb1), ("02_human_in_the_loop", nb2),
             ("03_fanout_saga_reflection", nb3), ("04_adk_workflow", nb4), ("05_tool_loop_and_mistral", nb5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOLUTIONS.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS:
        for folder, practice in ((SOLUTIONS, False), (OUT, True)):
            wrote = write_nb(nb.build(practice=practice), folder / f"{name}.ipynb")
            print("wrote" if wrote else "unchanged", f"{folder.name}/{name}")


if __name__ == "__main__":
    main()
