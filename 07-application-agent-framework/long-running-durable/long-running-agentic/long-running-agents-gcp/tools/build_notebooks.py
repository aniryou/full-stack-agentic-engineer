"""Generate the worked + practice notebooks. Run: python tools/build_notebooks.py"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = '''
import sys, os, json, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))          # repo root when run from notebooks/
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "notebooks")]

def show_journal(run):
    print(f"run {run.run_id}  status={run.status.value}  version={run.version}  steps={run.usage.steps}  tokens={run.usage.tokens}  cost=${run.usage.cost_usd:.4f}")
    for s in run.journal:
        out = json.dumps(s.output, default=str)[:70] if s.output is not None else (s.error or "")
        print(f"  [{s.index}] {s.kind.value:<6} {s.status.value:<7} {s.name:<18} key={s.idempotency_key or '-':<20} {out}")
'''


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def write(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}}
    (OUT / name).write_text(nbf.writes(nb))
    print("wrote", name)


# =====================================================================
# 01 — Durable loop
# =====================================================================
w1 = [
    md("""
    # 01 · The Durable Agent Loop — worked example

    **What you'll see:** an agent loop that survives a process crash *after* a side effect and still charges the card exactly once.

    The whole pattern in one line: **wake → do one step → checkpoint → sleep**. Nothing waits in memory; a *dispatcher* (Cloud Tasks in prod) delivers the next wake-up.

    ```mermaid
    flowchart LR
      T[Cloud Tasks<br/>named task] --> H[POST /internal/tasks/step]
      H --> L[acquire lease]
      L --> P{pending STARTED<br/>step?}
      P -- yes --> X[execute idempotently]
      P -- no --> M[LLM decides]
      M -- final --> S[SUCCEEDED]
      M -- tool --> I[journal intent<br/>checkpoint #1]
      I --> X
      X --> C[checkpoint #2]
      C --> N[enqueue next<br/>named task]
    ```

    Everything below runs offline: `ScriptedLLM` stands in for Gemini, `InMemoryRunStore` for Firestore, `InMemoryDispatcher` for Cloud Tasks. The GCP implementations are in `src/lragents/core/firestore_store.py` and `transport.py`.
    """),
    code(SETUP),
    code("""
    from lragents.core import *
    from lragents.patterns import DurableAgentLoop

    gateway = PaymentGateway()                       # a downstream API that honours Idempotency-Key

    def lookup_price(args, ctx):
        return {"sku": args["sku"], "price": 42.0}

    def charge_card(args, ctx):                      # ctx.idempotency_key is stable across retries
        return gateway.charge(float(args["amount"]), idempotency_key=ctx.idempotency_key)

    tools = ToolRegistry([
        Tool("lookup_price", "Look up a SKU price", lookup_price),
        Tool("charge_card",  "Charge the card",     charge_card),
    ])

    SCRIPT = [Decision.call("lookup_price", sku="ABC"),
              Decision.call("charge_card", amount=42.0),
              Decision.final("Charged 42.00 for ABC")]

    def build(script=SCRIPT, faults=None, clock=None, gateway=gateway):
        clock = clock or FakeClock()
        dispatcher = InMemoryDispatcher(clock=clock)
        loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM(list(script)), tools=tools,
                                dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(),
                                price=PriceCard(input_per_m=0.5, output_per_m=3.0), clock=clock, faults=faults)
        return loop, dispatcher, clock
    """),
    md("## 1. Happy path\nEach `deliver_one` is one Cloud Tasks delivery → one HTTP request → one step."),
    code("""
    loop, dispatcher, clock = build()
    run = loop.start("Buy SKU ABC")
    while dispatcher.deliver_one(loop.handle):
        r = loop.store.get(run.run_id)
        print(f"after wake-up: status={r.status.value:<10} journal={len(r.journal)} pending_tasks={dispatcher.pending()}")
    show_journal(loop.store.get(run.run_id))
    print("charges:", [(c.key, c.amount) for c in gateway.charges])
    """),
    md("""
    Read the journal: every LLM decision is recorded *before* the tool runs, every tool step carries an idempotency key `run:index`, and `usage` accumulates tokens/cost so the budget guard has something to check.

    ## 2. Crash after the side effect
    The dangerous window: the card **has been charged** but checkpoint #2 has not been written. We kill the process there.
    """),
    code("""
    gateway2 = PaymentGateway()
    faults = FaultInjector()
    loop, dispatcher, clock = build(faults=faults, gateway=gateway2)
    tools.get("charge_card").fn = lambda a, c: gateway2.charge(float(a["amount"]), idempotency_key=c.idempotency_key)

    run = loop.start("Buy SKU ABC")
    dispatcher.deliver_one(loop.handle)                 # step 0: lookup
    faults.crash_once_at("after_side_effect")           # next tool call dies after acting, before saving
    env = dispatcher.queue.pop(0)
    try:
        loop.handle(env)
    except SimulatedCrash as e:
        print("💥", e)

    mid = loop.store.get(run.run_id)
    print("charges so far:", len(gateway2.charges))
    print("pending step  :", mid.pending_step().name, mid.pending_step().status.value, mid.pending_step().idempotency_key)
    print("lease         :", mid.lease)
    """),
    md("""
    The store says: *intent recorded, not finished*. The lease is still held by the dead worker — a real crash never releases it. Cloud Tasks retries the delivery; the retry must **wait for the lease to expire** (TTL), then take the recovery path.
    """),
    code("""
    retry_worker = DurableAgentLoop(store=loop.store, llm=loop.llm, tools=tools, dispatcher=dispatcher,
                                    idempotency=loop.idem, worker_id="worker-2", clock=clock, price=loop.price)
    try:
        retry_worker.handle(env)
    except LeaseHeld as e:
        print("retry refused:", e)

    clock.advance(61)                                    # lease TTL is 60s
    retry_worker.handle(env)                             # recovery: re-executes the SAME intent under the SAME key
    dispatcher.drain(retry_worker.handle)
    final = loop.store.get(run.run_id)
    show_journal(final)
    print("charges:", len(gateway2.charges), "| recoveries:", final.state.get("recoveries"), "| replayed:", final.journal[3].output["replayed"])
    """),
    md("""
    One charge. The retry found the memoised result (`replayed=True`), finished the journal entry, and the loop went on to the final answer.

    **Why record intent first?** Because the model is non-deterministic. If the retry had *re-asked the LLM*, it might have said `charge_card(amount=42.5)` — a different key, a second charge. Journaling the decision turns a probabilistic step into a replayable one.

    ## 3. Duplicate delivery (at-least-once)
    """),
    code("""
    gateway3 = PaymentGateway()
    loop, dispatcher, clock = build(gateway=gateway3)
    tools.get("charge_card").fn = lambda a, c: gateway3.charge(float(a["amount"]), idempotency_key=c.idempotency_key)
    run = loop.start("Buy SKU ABC")
    dispatcher.deliver_one(loop.handle)
    dispatcher.duplicate_next()                          # Cloud Tasks hands us the same task twice
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    print("status:", final.status.value, "| journal:", len(final.journal), "| charges:", len(gateway3.charges))
    print("task names seen (de-dup keys):", sorted(dispatcher.seen_names))
    """),
    md("""
    The second delivery finds the step already done (`next_index > expected_index`), re-enqueues the *named* next task (collapsed by de-dup) and returns 200.

    ## 4. Budget = the real stop condition
    """),
    code("""
    endless = [Decision.call("lookup_price", sku="X")] * 100
    loop, dispatcher, clock = build(script=endless)
    run = loop.start("loop forever", budget=Budget(max_steps=6, max_cost_usd=0.01))
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    print(final.status.value, "|", final.error, "| tokens:", final.usage.tokens, f"| cost: ${final.usage.cost_usd:.5f}")
    """),
    md("""
    ## 5. The counter-example
    Same crash, but the downstream API has **no idempotency key** and we bypass the memo store.
    """),
    code("""
    naive = NaivePaymentGateway()
    class NoMemo(InMemoryIdempotencyStore):
        def __contains__(self, k): return False
    faults = FaultInjector(); clock = FakeClock(); dispatcher = InMemoryDispatcher(clock=clock)
    loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM(SCRIPT[1:]),
                            tools=ToolRegistry([Tool("charge_card", "charge", lambda a, c: naive.charge(float(a["amount"])))]),
                            dispatcher=dispatcher, idempotency=NoMemo(), clock=clock, faults=faults)
    run = loop.start("charge")
    try:
        loop.handle(dispatcher.queue.pop(0))
    except SimulatedCrash:
        pass
    clock.advance(61); loop.step(run.run_id, expected_index=2)
    print("charges with a naive gateway after one crash:", naive.charges)
    """),
    md("""
    ## 6. Async tools: park, don't poll in-process
    A tool that kicks off a 2-minute job returns a ticket. The run parks (`WAITING_EVENT`) and schedules a **delayed** Cloud Task to poll with exponential back-off — or an external webhook resumes it. No process waits.
    """),
    code("""
    clock = FakeClock()
    jobs = SlowJobService(clock, duration_s=120)
    atools = ToolRegistry([Tool("run_report", "start a long report", lambda a, c: jobs.submit(a, idempotency_key=c.idempotency_key),
                                is_async=True, poll=jobs.status)])
    dispatcher = InMemoryDispatcher(clock=clock)
    loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM([Decision.call("run_report", region="apac"),
                            lambda msgs: Decision.final(msgs[-1]["content"])]), tools=atools, dispatcher=dispatcher,
                            idempotency=InMemoryIdempotencyStore(), clock=clock, poll_base_s=10, poll_max_s=60)
    run = loop.start("report")
    dispatcher.drain(loop.handle)
    while loop.store.get(run.run_id).status == RunStatus.WAITING_EVENT:
        nxt = dispatcher.queue[0]
        print(f"t={clock()-1_700_000_000:>4.0f}s  parked; next poll in {nxt.not_before-clock():.0f}s  (task {nxt.name})")
        clock.t = nxt.not_before; dispatcher.drain(loop.handle)
    show_journal(loop.store.get(run.run_id))
    """),
    md("""
    ## 7. Mapping to GCP
    | In this notebook | On GCP | File |
    |---|---|---|
    | `InMemoryDispatcher` | Cloud Tasks queue, HTTP target with OIDC token, task **name** = `run-step-N` | `core/transport.py` |
    | `InMemoryRunStore` | Firestore document per run, transactions for `version` + `lease` | `core/firestore_store.py` |
    | `InMemoryIdempotencyStore` | Firestore `idempotency_keys` collection with TTL | `core/firestore_store.py` |
    | `ScriptedLLM` | Gemini on Vertex AI via `google-genai` | `core/llm.py` |
    | `loop.handle(env)` | `POST /internal/tasks/step` on Cloud Run | `service/app.py` |
    | `FakeClock.advance` | lease TTL / `schedule_time` on tasks | — |
    """),
    code("""
    from lragents.core.transport import Envelope
    for e in [Envelope("run_ab12", 3, "step"), Envelope("run_ab12", 3, "step"), Envelope("run_ab12", 1, "poll", payload={"attempt": 2})]:
        print(e.name)
    import inspect; from lragents.service import app as service
    print(inspect.getsource(service.create_app).split('@app.post("/internal/tasks/{kind}"')[1][:700])
    """),
    md("""
    ## Takeaways
    1. **One step per wake-up**; the store is the only memory.
    2. **Intent before side effect** — the LLM is a non-deterministic side effect; journal its decision, then act.
    3. **Idempotency key = `run:step`**, passed downstream *and* memoised locally.
    4. **Leases expire**; a retry that arrives too early is refused, not raced.
    5. **Budgets are code**, not prompt text.
    6. **Waiting is free**: async tools park the run; polls are delayed tasks with back-off.
    """),
]

p1 = [
    md("""
    # 01 · Durable loop — practice (fill in the blanks)

    Implement the two pieces that make the loop durable. Run the check cell after each; it raises with a hint on failure.
    Reference solutions: `notebooks/solutions/ex1_durable_loop.py` (peek only after trying).
    """),
    code(SETUP),
    code("""
    from lragents.core import *
    from lragents.practice_checks import check_run_idempotent, check_mini_loop
    """),
    md("""
    ## Exercise 1 — `run_idempotent`
    Execute `fn()` **only if** no result is stored under `key`; return `(result, replayed)`.
    Think about the order of *execute* vs *record* and what happens if the process dies between them.
    """),
    code("""
    def run_idempotent(store, key, fn):
        # TODO: if key already in store → return (store.get(key), True)
        # TODO: otherwise run fn(), store.put(key, result), return (result, False)
        raise NotImplementedError
    """),
    code("print(check_run_idempotent(run_idempotent))"),
    md("""
    ## Exercise 2 — a 40-line durable loop
    Complete `MiniDurableLoop.step`. Contract:
    * If the run is terminal, return it.
    * If there is **no pending step**: ask the LLM; journal an `LLM` step (`DONE`); on `final` → set `result`, `SUCCEEDED`, save, return. On a tool call → append a `TOOL` step with status **`STARTED`** and `idempotency_key=f"{run_id}:{index}"`, then **save** (checkpoint #1).
    * Execute the pending tool via `run_idempotent` (use your Exercise 1 function or `lragents.core.run_idempotent`), then call `self.faults.maybe_crash("after_side_effect")`, then `finish()` the record and save (checkpoint #2).
    """),
    code("""
    from lragents.core import Run, RunStatus, StepKind, StepRecord, StepStatus, ToolContext, new_run_id

    class MiniDurableLoop:
        def __init__(self, store, llm, tools, idem, faults):
            self.store, self.llm, self.tools, self.idem, self.faults = store, llm, tools, idem, faults

        def start(self, goal):
            return self.store.create(Run(run_id=new_run_id("mini"), goal=goal, status=RunStatus.RUNNING))

        def step(self, run_id):
            run = self.store.get(run_id)
            if run.status.terminal:
                return run
            pending = run.pending_step()
            if pending is None:
                d = self.llm.decide("", [{"role": "user", "content": run.goal}], self.tools.specs())
                rec = StepRecord(index=run.next_index(), kind=StepKind.LLM, status=StepStatus.DONE, name="decide")
                rec.finish(output={"kind": d.kind, "tool": d.tool_name, "args": d.tool_args, "text": d.text})
                run.append(rec)
                if d.kind == "final":
                    # TODO: set run.result / run.status and return self.store.save(run)
                    raise NotImplementedError("TODO: final answer")
                # TODO: append a STARTED TOOL StepRecord (name=d.tool_name, input=d.tool_args, idempotency_key=...)
                # TODO: run = self.store.save(run)   # checkpoint #1 — why here and not after execution?
                # TODO: pending = run.journal[-1]
                raise NotImplementedError("TODO: journal the intent, then checkpoint")
            tool = self.tools.get(pending.name)
            ctx = ToolContext(run.run_id, pending.index, pending.idempotency_key, run.state)
            # TODO: out, replayed = run_idempotent(self.idem, pending.idempotency_key, lambda: tool.fn(pending.input, ctx))
            # TODO: self.faults.maybe_crash("after_side_effect")
            # TODO: pending.finish(output={"result": out, "replayed": replayed}); return self.store.save(run)
            raise NotImplementedError
    """),
    code("print(check_mini_loop(MiniDurableLoop))"),
    md("""
    ## Exercise 3 — reason about it (write answers in the cell below)
    1. Your loop saves twice per tool step. Which crash windows does each save close? Which window is still open, and what closes it?
    2. Cloud Tasks delivered the same task twice, 50 ms apart, to two Cloud Run instances. Walk through what each instance does with the code you wrote. What primitive is missing from `MiniDurableLoop` that `DurableAgentLoop` has?
    3. A tool call takes 45 minutes (a BigQuery export). What changes? (Hint: `is_async`, `WAITING_EVENT`, `schedule_time`.)
    """),
    code("""
    answers = '''
    1.
    2.
    3.
    '''
    """),
    md("### Reveal the reference implementation"),
    code("""
    import inspect
    from solutions.ex1_durable_loop import MiniDurableLoop as Ref
    print(inspect.getsource(Ref.step))
    """),
]

# =====================================================================
# 02 — Fan-out / fan-in + reflection
# =====================================================================
w2 = [
    md("""
    # 02 · Orchestrator / workers (fan-out, fan-in) and the reflection loop — worked

    Two ways to compose LLM steps beyond a single loop: **in parallel** (planner → N workers → aggregator) and **iteratively** (generate → critique → revise).

    ```mermaid
    flowchart LR
      P[planner LLM] -->|N subtasks| T[(Pub/Sub topic /<br/>N Cloud Tasks)]
      T --> W1[worker] & W2[worker] & W3[worker]
      W1 & W2 & W3 -->|transaction:<br/>results[id]=r, completed+=1| F[(Firestore run doc)]
      F -->|completed == expected| A[aggregate task<br/>named → dedup]
      A --> S[synthesis LLM]
    ```
    The fan-in is the hard part: workers are at-least-once, the counter must be transactional, and the aggregate trigger must be idempotent.
    """),
    code(SETUP),
    code("""
    from lragents.core import *
    from lragents.patterns import FanOutFanIn, ReflectionLoop

    calls = []
    def worker(task):                              # imagine: a Cloud Run worker calling Gemini + a search API
        calls.append(task["title"])
        return {"summary": f"findings for {task['title']}"}

    subtasks = [{"title": f"vendor-{i}", "instructions": "assess pricing and SLA"} for i in range(6)]
    planner    = ScriptedLLM([Decision.final(json.dumps(subtasks))])
    aggregator = ScriptedLLM([lambda msgs: Decision.final("SYNTHESIS over " + str(len(json.loads(msgs[0]["content"])["results"])) + " results")])
    dispatcher = InMemoryDispatcher()
    fan = FanOutFanIn(store=InMemoryRunStore(), planner=planner, aggregator=aggregator, dispatcher=dispatcher,
                      worker=worker, idempotency=InMemoryIdempotencyStore())
    run = fan.start("Compare six vendors")
    print("fanned out:", dispatcher.pending(), "subtask tasks; expected =", fan.store.get(run.run_id).state["fan"]["expected"])
    """),
    code("""
    dispatcher.duplicate_next()                   # Pub/Sub redelivers the first subtask
    dispatcher.drain(fan.handle)
    final = fan.store.get(run.run_id)
    print("status:", final.status.value, "| result:", final.result)
    print("worker calls:", calls)
    print("aggregate deliveries:", sum(1 for e in dispatcher.delivered if e.kind == "aggregate"))
    print("fan state:", {k: v for k, v in final.state["fan"].items() if k != "subtasks"})
    """),
    md("""
    Six workers ran once each despite a duplicate delivery; aggregation ran once.

    ## Crash after commit, before the aggregate is enqueued
    The last worker commits `completed == expected`, then dies before enqueuing `aggregate`. Retry → the transaction sees the subtask already recorded (no double count) **but still reports `all_done`**, so it enqueues the named aggregate task again — de-duplicated, harmless.
    """),
    code("""
    faults = FaultInjector(); calls.clear()
    dispatcher = InMemoryDispatcher()
    fan = FanOutFanIn(store=InMemoryRunStore(), planner=ScriptedLLM([Decision.final(json.dumps(subtasks[:2]))]),
                      aggregator=ScriptedLLM([Decision.final("SYNTH")]), dispatcher=dispatcher, worker=worker,
                      idempotency=InMemoryIdempotencyStore(), faults=faults)
    run = fan.start("two vendors")
    dispatcher.deliver_one(fan.handle)
    faults.crash_once_at("after_commit")
    dispatcher.drain(fan.handle)
    print(fan.store.get(run.run_id).status.value, "| workers ran:", calls, "| dead-letter:", dispatcher.dead_letter)
    """),
    md("""
    ## Where it breaks at scale — and the two fixes
    Every worker updates the *same* Firestore document. Fine for ~50 subtasks; at hundreds, transaction contention and the ~1 write/s per-document guidance bite.

    1. **One doc per subtask + count query** (`runs/{id}/subtasks/{sid}`), orchestrator polls with a delayed task.
    2. **Let Cloud Workflows do the join**: a `parallel` `for` loop calling Cloud Run workers; Workflows waits for all branches. Read `infra/workflows/fan_out_fan_in.yaml`.
    """),
    code("""
    print(open(os.path.join(ROOT, "infra", "workflows", "fan_out_fan_in.yaml")).read())
    """),
    md("""
    ## Reflection loop (evaluator–optimizer)
    Iterative composition. The stop condition is **code** (`threshold`, `max_iters`), every iteration is a checkpoint, and generator/critic are separate LLM configurations.
    """),
    code("""
    gen    = ScriptedLLM([Decision.final("v1: an abstract"), Decision.final("v2: a better abstract"), Decision.final("v3")])
    critic = ScriptedLLM([Decision.final(json.dumps({"score": 5, "feedback": "state the contribution first"})),
                          Decision.final(json.dumps({"score": 9, "feedback": "good"}))])
    store, dispatcher = InMemoryRunStore(), InMemoryDispatcher()
    refl = ReflectionLoop(store=store, generator=gen, critic=critic, dispatcher=dispatcher, threshold=8, max_iters=3,
                          price=PriceCard(0.5, 3.0))
    run = refl.start("Write an abstract for a paper on durable agents")
    while dispatcher.deliver_one(refl.handle):
        r = store.get(run.run_id)
        print(f"iter={r.state['iter']} scores={r.state['scores']} status={r.status.value} checkpoints={store.save_count}")
    print(store.get(run.run_id).result)
    """),
    md("""
    ## Takeaways
    * Fan-out is easy; **fan-in is a distributed counter** → transaction + idempotent trigger.
    * Side effects live *outside* the transaction (transactions re-run).
    * Prefer a managed join (Cloud Workflows `parallel`) when the branch count is known up front.
    * Iterative loops need deterministic exits and per-iteration checkpoints.
    """),
]

p2 = [
    md("""
    # 02 · Fan-out / fan-in — practice
    Reference: `notebooks/solutions/ex2_fan_out_fan_in.py`.
    """),
    code(SETUP),
    code("""
    from lragents.core import *
    from lragents.core.transport import Envelope
    from lragents.patterns import FanOutFanIn
    from lragents.practice_checks import check_fan_in_mutate, check_fan_out_fan_in
    """),
    md("""
    ## Exercise 1 — the fan-in transaction function
    `make_mutate(subtask_id, result)` returns `mutate(run) -> bool`. It must:
    * record `result` under `run.state["fan"]["results"][subtask_id]` and increment `completed` — **only if not already recorded** (duplicate delivery);
    * return `True` iff `completed == expected` — including for a *late duplicate* that arrives after completion (why?).
    """),
    code("""
    def make_mutate(subtask_id, result):
        def mutate(run):
            fan = run.state["fan"]
            # TODO
            raise NotImplementedError
        return mutate
    """),
    code("print(check_fan_in_mutate(make_mutate))"),
    md("""
    ## Exercise 2 — the worker handler
    Subclass `FanOutFanIn` and implement `handle_subtask`. Order matters:
    1. run the worker **idempotently** (`run_idempotent(self.idem, key, ...)`) — outside any transaction;
    2. `run, all_done = transact(self.store, run_id, make_mutate(subtask_id, result))`;
    3. if `all_done`: `self.dispatcher.enqueue(Envelope(run_id, 1, "aggregate"))`.
    """),
    code("""
    class MyFan(FanOutFanIn):
        def handle_subtask(self, run_id, subtask_id, task):
            key = f"{run_id}:sub:{subtask_id}"
            # TODO
            raise NotImplementedError
    """),
    code("print(check_fan_out_fan_in(MyFan))"),
    md("""
    ## Exercise 3 — design questions
    1. You have 500 subtasks. What breaks first in this design on Firestore, and what are two fixes?
    2. A worker takes 20 minutes. Which GCP service should run it and why (Cloud Run service vs job vs Workflows)?
    3. The aggregator LLM call fails with a 429. What retries it, and what guarantees only one synthesis is written?
    """),
    code("answers = '''\n1.\n2.\n3.\n'''"),
    code("""
    import inspect
    from solutions import ex2_fan_out_fan_in as ref
    print(inspect.getsource(ref.make_mutate)); print(inspect.getsource(ref.MyFan))
    """),
]

# =====================================================================
# 03 — HITL, saga, scheduled
# =====================================================================
w3 = [
    md("""
    # 03 · Waiting for humans, undoing work, waking up on time — worked

    Three patterns that share one idea: **the run parks in the store and costs nothing until something wakes it.**

    | Pattern | Who wakes it | GCP trigger |
    |---|---|---|
    | Human-in-the-loop gate | a person clicking approve/reject | HTTP endpoint (IAP), or Cloud Workflows callback |
    | Saga | the next Cloud Task | Cloud Tasks |
    | Scheduled agent | a clock | Cloud Scheduler → Pub/Sub → Cloud Run |
    """),
    code(SETUP),
    code("""
    from lragents.core import *
    from lragents.patterns import DurableAgentLoop, SagaRunner, SagaStep, ScheduledAgent, approve, expire_stale_approvals, approval_link

    clock = FakeClock()
    gateway = PaymentGateway()
    tools = ToolRegistry([Tool("charge_card", "Charge the card (needs approval)",
                               lambda a, c: gateway.charge(float(a["amount"]), idempotency_key=c.idempotency_key),
                               requires_approval=True)])
    inbox = []                                                     # stands in for Slack / email
    dispatcher = InMemoryDispatcher(clock=clock)
    loop = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM([Decision.call("charge_card", amount=1200.0), Decision.final("Paid.")]),
                            tools=tools, dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(), clock=clock,
                            notify=lambda run: inbox.append(approval_link("https://ops.example.com", run)))
    run = loop.start("Pay invoice 1200")
    dispatcher.drain(loop.handle)
    parked = loop.store.get(run.run_id)
    print("status:", parked.status.value, "| pending tasks:", dispatcher.pending(), "| LLM calls left:", loop.llm.remaining)
    print("proposed call:", parked.waiting_on["tool"], parked.waiting_on["args"])
    print("notification :", inbox[-1])
    """),
    md("""
    Nothing is scheduled, nothing is running, the model is not consulted again. The **exact** proposed call is stored with a one-time token.

    ## Approve → execute exactly that
    """),
    code("""
    approve(loop, run.run_id, parked.waiting_on["token"], approved=True, approver="cfo@example.com", comment="ok")
    dispatcher.drain(loop.handle)
    final = loop.store.get(run.run_id)
    show_journal(final)
    print("charges:", [(c.amount) for c in gateway.charges], "| LLM calls left after approval:", 0 if loop.llm.remaining == 0 else loop.llm.remaining)
    approve(loop, run.run_id, parked.waiting_on["token"], True, "cfo@example.com")     # double click → no-op
    print("charges after a second approve:", len(gateway.charges))
    """),
    md("""
    The journal reads LLM → HUMAN → TOOL → LLM: the approved call was journaled as a STARTED intent and executed by the loop's normal recovery path. No re-planning between approval and execution.

    ## Reject → the model hears why
    """),
    code("""
    loop2 = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM([Decision.call("charge_card", amount=1200.0), Decision.final("Understood, not paying.")]),
                             tools=tools, dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(), clock=clock)
    run2 = loop2.start("Pay invoice 1200"); dispatcher.drain(loop2.handle)
    approve(loop2, run2.run_id, loop2.store.get(run2.run_id).waiting_on["token"], approved=False, approver="cfo", comment="vendor not onboarded")
    dispatcher.drain(loop2.handle)
    print(loop2.store.get(run2.run_id).result, "| last message to the model:", loop2.llm.calls[-1][-1]["content"])
    """),
    md("## Every gate has a deadline\nCloud Scheduler hits `/internal/scheduler/tick`; stale approvals become an explicit outcome."),
    code("""
    loop3 = DurableAgentLoop(store=InMemoryRunStore(clock=clock), llm=ScriptedLLM([Decision.call("charge_card", amount=5.0)]),
                             tools=tools, dispatcher=dispatcher, idempotency=InMemoryIdempotencyStore(), clock=clock)
    run3 = loop3.start("Pay 5"); dispatcher.drain(loop3.handle)
    print("expired now:", [r.run_id for r in expire_stale_approvals(loop3, ttl_s=72*3600)])
    clock.advance(73*3600)
    print("expired after 73h:", [(r.run_id, r.error) for r in expire_stale_approvals(loop3, ttl_s=72*3600, escalate=lambda r: print('  escalating', r.run_id))])
    """),
    md("""
    The Cloud Workflows flavour of the same gate (create a callback endpoint, wait up to N days, resume) is in `infra/workflows/hitl_approval.yaml`:
    """),
    code("""print(open(os.path.join(ROOT, "infra", "workflows", "hitl_approval.yaml")).read())"""),
    md("""
    ## Saga — undoing what already happened
    Book flight → hotel → charge card. The card fails. Compensations run **in reverse**, one per wake-up, each journaled with an idempotency key. We also kill the process mid-compensation.
    """),
    code("""
    log = []
    def act(name, fail=False):
        def _a(ctx, key):
            if fail: raise ToolError(f"{name}: upstream error")
            log.append(f"+{name}"); return {"ref": f"{name}-{key[-4:]}"}
        return _a
    def comp(name):
        def _c(ctx, prior, key): log.append(f"-{name} (was {prior['ref']})")
        return _c
    steps = [SagaStep("flight", act("flight"), comp("flight")), SagaStep("hotel", act("hotel"), comp("hotel")), SagaStep("card", act("card", fail=True), comp("card"))]
    faults = FaultInjector(); d = InMemoryDispatcher()
    saga = SagaRunner(store=InMemoryRunStore(), dispatcher=d, idempotency=InMemoryIdempotencyStore(), steps=steps, faults=faults)
    run = saga.start({"trip": "SIN→AMS"})
    for _ in range(3): d.deliver_one(saga.handle)
    print("after 3 wake-ups:", saga.store.get(run.run_id).state["saga"]["phase"], log)
    faults.crash_once_at("after_side_effect")          # die right after compensating the hotel
    d.drain(saga.handle)
    final = saga.store.get(run.run_id)
    print(final.status.value, "|", final.error)
    print("log:", log)
    show_journal(final)
    """),
    md("""
    `-hotel` appears once even though the process died right after it: the retry found the STARTED intent and the memoised compensation.

    ## Scheduled agent — leases and due times
    """),
    code("""
    clock = FakeClock(); store = InMemoryRunStore(clock=clock); seen = []
    a = ScheduledAgent(store=store, work=lambda st: seen.append(("A", clock())) or "checked presale", interval_s=300, clock=clock, lease_ttl_s=120)
    b = ScheduledAgent(store=store, work=lambda st: seen.append(("B", clock())) or "checked presale", interval_s=300, clock=clock, lease_ttl_s=120, worker_id="worker-2")
    a.ensure("presale-monitor")
    print("tick 1:", a.tick("presale-monitor"))
    print("tick 1 again (Scheduler retried):", a.tick("presale-monitor").reason)
    clock.advance(301)
    store.acquire_lease("presale-monitor", "zombie-worker", 120)        # a crashed instance still holds the lease
    print("tick 2 while zombie holds lease:", b.tick("presale-monitor").reason)
    clock.advance(121)
    print("tick 2 after TTL:", b.tick("presale-monitor"))
    print("work executed:", seen)
    """),
    md("""
    ## Takeaways
    * A parked run has **no process**; approvals, callbacks and clocks all resume it through the same store.
    * *Approve what you execute, execute what was approved* — journal the approved call, don't re-plan.
    * Sagas: compensations are first-class side effects with their own keys; a failed compensation is an **escalation**, never silence.
    * Cron is not a coordination primitive; **lease + due-time** is.
    """),
]

p3 = [
    md("# 03 · HITL + saga — practice\nReference: `notebooks/solutions/ex3_hitl_saga.py`."),
    code(SETUP),
    code("""
    import hmac
    from lragents.core import *
    from lragents.practice_checks import check_approve, check_saga_next
    """),
    md("""
    ## Exercise 1 — `approve`
    Implement `approve(loop, run_id, token, approved, approver, comment="")`:
    1. `acquire_lease`; if status is not `WAITING_HUMAN`, return the run (idempotent);
    2. compare the token with `hmac.compare_digest` — raise on mismatch;
    3. append a `HUMAN` step (`DONE`) with the decision; clear `waiting_on`; set `RUNNING`;
    4. if approved, append a `TOOL` step with status **`STARTED`**, the *stored* tool name/args and key `f"{run_id}:{index}"` — do **not** call the LLM;
    5. `loop.store.save(run)`, `loop._enqueue_next(run)`, release the lease in `finally`.
    """),
    code("""
    def approve(loop, run_id, token, approved, approver, comment=""):
        run = loop.store.acquire_lease(run_id, loop.worker_id, loop.lease_ttl_s)
        try:
            # TODO
            raise NotImplementedError
        finally:
            loop.store.release_lease(run_id, loop.worker_id)
    """),
    code("print(check_approve(approve))"),
    md("""
    ## Exercise 2 — saga state machine
    `saga_next(order, saga)` returns which step runs next and in what phase, given
    `saga = {"phase": "forward"|"compensating", "cursor": int, "comp_cursor": int|None}`.
    Return `("__done__", "succeeded")` when all forward steps completed, `("__done__", "failed")` when compensation walked past the first step.
    """),
    code("""
    def saga_next(order, saga):
        # TODO
        raise NotImplementedError
    """),
    code("print(check_saga_next(saga_next))"),
    md("""
    ## Exercise 3 — design questions
    1. Why must the approved tool call be journaled *before* the loop is woken, rather than passed in the wake-up payload?
    2. The approval endpoint is public behind IAP. List three checks the handler must do before mutating the run.
    3. A compensation ("refund") fails 5 times. What state should the saga end in, who is notified, and what must NOT happen?
    """),
    code("answers = '''\n1.\n2.\n3.\n'''"),
    code("""
    import inspect
    from solutions import ex3_hitl_saga as ref
    print(inspect.getsource(ref.approve)); print(inspect.getsource(ref.saga_next))
    """),
]

# =====================================================================
# 04 — ADK 2 Workflow
# =====================================================================
w4 = [
    md("""
    # 04 · The same patterns with ADK 2 `Workflow` — worked

    Google's ADK 2 gives you the primitives natively: **interrupts** (`RequestInput`), **resumability** (`ResumabilityConfig`), **replay-safe nodes** (`rerun_on_resume`), **routing** (`ctx.route`) and pluggable **session stores** (SQLite → Cloud SQL). This notebook runs the graph *without a model* so you can see the mechanics; swap `use_model=True` to put Gemini in the `plan` node.

    ```mermaid
    flowchart LR
      S((START)) --> B[agree_budget<br/>RequestInput 'budget'<br/>rerun_on_resume=True]
      B --> P[plan<br/>judgement]
      P --> Q[queue_up<br/>side effect<br/>rerun_on_resume=False]
      Q --> C[check_front<br/>RequestInput 'wake'<br/>rerun_on_resume=True]
      C -- ready --> Y[buy<br/>idempotency key]
      C -- sold_out --> A[abandon]
    ```
    (Cells use top-level `await` — Jupyter already runs an event loop.)
    """),
    code(SETUP),
    code("""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from lragents.adk.nightly_workflow import ASK_BUDGET, WAKE, Venue, build_app, run_until_interrupt

    venue = Venue()
    svc = InMemorySessionService()                       # prod: DatabaseSessionService("postgresql+asyncpg://...") or VertexAiSessionService
    runner = Runner(app=build_app(venue), session_service=svc)
    session = await svc.create_session(app_name="nightly_app", user_id="anil",
                                       state={"events": [{"id": "ams-tue", "weekday": "Tuesday"}, {"id": "ams-sat", "weekday": "Saturday"}]})
    async def state(s=svc, sid=None):
        return (await s.get_session(app_name="nightly_app", user_id="anil", session_id=sid or session.id)).state

    r1 = await run_until_interrupt(runner, "anil", session.id, text="Get us two tickets")
    print("stopped on:", r1["interrupts"], "| invocation:", r1["invocation_id"][:12], "| venue calls:", venue.calls)
    """),
    md("The graph asked a question and **stopped**. The invocation lives in the session store; the Python process could exit here.\n\nAnswer it by resuming the *same invocation* with a `FunctionResponse` whose id is the interrupt id:"),
    code("""
    r2 = await run_until_interrupt(runner, "anil", session.id, invocation_id=r1["invocation_id"], answers={ASK_BUDGET: {"budget": 250}})
    print("stopped on:", r2["interrupts"], "| venue calls:", venue.calls, "| state:", {k: v for k, v in (await state()).items() if k != "events"})
    """),
    md("""
    Four nodes ran off one answer: `plan` chose Saturday (weekend rule), `queue_up` took **one** ticket, `check_front` found us at #14,203 and parked on `wake`.

    ## Wake-ups (what Cloud Scheduler → Pub/Sub → `/wake` does)
    """),
    code("""
    ticket = (await state())["ticket"]
    venue.advance(ticket, 10_000)                        # the queue moves while nothing of ours runs
    r3 = await run_until_interrupt(runner, "anil", session.id, invocation_id=r2["invocation_id"], answers={WAKE: {"ok": True}})
    print("wake 1 → stopped on:", r3["interrupts"], "| join_queue calls:", venue.calls.count("join_queue"), "| position:", venue.position(ticket))
    venue.advance(ticket, 10_000)
    r4 = await run_until_interrupt(runner, "anil", session.id, invocation_id=r3["invocation_id"], answers={WAKE: {"ok": True}})
    print("wake 2 → stopped on:", r4["interrupts"], "| venue calls:", venue.calls)
    print("order:", (await state())["order"])
    """),
    md("""
    `queue_up` never re-ran across four wake-ups (`rerun_on_resume=False`); `check_front` re-ran every time (`True`); `buy` executed once with an idempotency key.

    ## Staleness guard: the world changed while we waited
    """),
    code("""
    venue2 = Venue(); svc2 = InMemorySessionService(); runner2 = Runner(app=build_app(venue2), session_service=svc2)
    s2 = await svc2.create_session(app_name="nightly_app", user_id="anil", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
    a = await run_until_interrupt(runner2, "anil", s2.id, text="go")
    b = await run_until_interrupt(runner2, "anil", s2.id, invocation_id=a["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
    t2 = (await state(svc2, s2.id))["ticket"]
    venue2.advance(t2, 99_999); venue2.sell_out("ams-sat")
    await run_until_interrupt(runner2, "anil", s2.id, invocation_id=b["invocation_id"], answers={WAKE: {"ok": True}})
    print("calls:", venue2.calls, "| orders:", venue2.orders, "| routed to abandon:", (await state(svc2, s2.id))["order"] is None)
    """),
    md("""
    ## The classic mistake: a new invocation instead of a resume
    Typing in the chat box (no `invocation_id`) starts a **new** invocation → the graph replays from START → a second queue ticket.
    """),
    code("""
    venue3 = Venue(); svc3 = InMemorySessionService(); runner3 = Runner(app=build_app(venue3), session_service=svc3)
    s3 = await svc3.create_session(app_name="nightly_app", user_id="anil", state={"events": [{"id": "ams-sat", "weekday": "Saturday"}]})
    a = await run_until_interrupt(runner3, "anil", s3.id, text="go")
    await run_until_interrupt(runner3, "anil", s3.id, invocation_id=a["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
    c = await run_until_interrupt(runner3, "anil", s3.id, text="where am I in line?")       # WRONG: new invocation
    await run_until_interrupt(runner3, "anil", s3.id, invocation_id=c["invocation_id"], answers={ASK_BUDGET: {"budget": 200}})
    print("join_queue calls:", venue3.calls.count("join_queue"), "← two tickets")
    """),
    md("""
    ## What the session store holds
    Every event (interrupts included) is a row; this is what survives a restart when the store is Cloud SQL.
    """),
    code("""
    sess = await svc.get_session(app_name="nightly_app", user_id="anil", session_id=session.id)
    print("events:", len(sess.events), "| invocations:", len({e.invocation_id for e in sess.events}))
    for e in sess.events[:12]:
        fc = [p.function_call.name for p in (e.content.parts if e.content else []) or [] if p.function_call]
        print(f"  {e.invocation_id[:10]}  author={e.author:<8} fc={fc} state_delta={list((e.actions.state_delta or {}).keys()) if e.actions else []}")
    """),
    md("""
    ## Deploying this shape on GCP
    | Local | Cloud | Change |
    |---|---|---|
    | `InMemorySessionService` | Cloud SQL (Postgres) via `DatabaseSessionService("postgresql+asyncpg://...?host=/cloudsql/...")` or Agent Runtime Sessions | a connection string |
    | `adk web` | one Cloud Run service (`src/lragents/adk/main.py`) | a Dockerfile + entrypoint |
    | you calling `run_until_interrupt` | Cloud Scheduler → Pub/Sub → push → `POST /wake` | `trigger_sources=["pubsub"]` + custom `/wake` |
    | `Venue` | the real API with an `Idempotency-Key` header | — |

    Note the codelab's warning that ADK's built-in Pub/Sub trigger route **creates a new session per message** — for long-running runs you need a wake endpoint that resumes an *existing* session, which is what `main.py` adds.

    ## Takeaways
    * `RequestInput` is one primitive for both "waiting for a person" and "waiting for the world".
    * `rerun_on_resume` is the ADK spelling of *idempotent vs re-check* — decide it per node, deliberately.
    * Resume the invocation; don't start a new one.
    * Prompts can't wake themselves; something with a clock has to call the endpoint.
    """),
]

p4 = [
    md("# 04 · ADK 2 Workflow — practice\nReference: `notebooks/solutions/ex4_adk_workflow.py`."),
    code(SETUP),
    code("""
    from google.adk.events.request_input import RequestInput
    from google.adk.workflow import START, Workflow, node
    from lragents.adk.nightly_workflow import ASK_BUDGET, WAKE, Venue
    from lragents.practice_checks import check_adk_workflow
    """),
    md("""
    ## Exercise — build the graph
    Complete the nodes. Requirements the checker enforces:
    * `agree_budget` interrupts with `interrupt_id=ASK_BUDGET` until `ctx.resume_inputs[ASK_BUDGET]["budget"]` is present; it must **re-run on resume**.
    * `queue_up` calls `venue.join_queue(event_id, idempotency_key=...)` and must **never re-run on resume** (one ticket across all wake-ups).
    * `check_front` interrupts with `interrupt_id=WAKE` while `venue.position(ticket) > 0`; when at the front it sets `ctx.route` to `"ready"` if `venue.seats_left(event_id) >= 2` else `"sold_out"`; it must re-run on resume.
    * `buy` calls `venue.purchase(event_id, 2, idempotency_key=...)` and stores the order in `ctx.state["order"]`.
    * Wire the edges: `START → agree_budget → plan → queue_up → check_front → {"ready": buy, "sold_out": abandon}`.
    """),
    code("""
    def build_workflow(venue):
        @node(rerun_on_resume=True)
        def agree_budget(ctx):
            said = ((ctx.resume_inputs or {}).get(ASK_BUDGET) or {}).get("budget")
            # TODO: if not said → return RequestInput(interrupt_id=ASK_BUDGET, message="...")
            # TODO: else store float(said) in ctx.state["budget_per_seat"] and return it
            raise NotImplementedError

        @node
        def plan(ctx):
            ctx.state["event_id"] = ctx.state["events"][0]["id"]
            return {"event_id": ctx.state["event_id"]}

        @node(rerun_on_resume=None)   # TODO: True or False? think: is this a side effect or a re-check?
        def queue_up(ctx):
            # TODO: t = venue.join_queue(ctx.state["event_id"], idempotency_key=f"{ctx.session.id}:{ctx.state['event_id']}")
            # TODO: ctx.state["ticket"] = t["ticket"]; return t
            raise NotImplementedError

        @node(rerun_on_resume=None)   # TODO: True or False?
        def check_front(ctx):
            pos = venue.position(ctx.state["ticket"])
            # TODO: interrupt while pos > 0 (interrupt_id=WAKE)
            # TODO: else set ctx.route based on venue.seats_left(...) and return a dict
            raise NotImplementedError

        @node
        def buy(ctx):
            # TODO
            raise NotImplementedError

        @node
        def abandon(ctx):
            ctx.state["order"] = None
            return {"abandoned": True}

        return Workflow(name="practice", edges=[  # TODO: the chain with the routing map
        ])
    """),
    code("print(check_adk_workflow(build_workflow))"),
    md("""
    ## Design questions
    1. Which of these nodes could be an `LlmAgent` instead of a function, and which must never be? Why?
    2. Where does the ticket live so that it survives a Cloud Run restart? What would break if it were in a `temp:` state key?
    3. Sketch the Cloud Scheduler → Pub/Sub → `/wake` payload. What identifies the paused run?
    """),
    code("answers = '''\n1.\n2.\n3.\n'''"),
    code("""
    import inspect
    from solutions import ex4_adk_workflow as ref
    print(inspect.getsource(ref.build_workflow))
    """),
]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    write("01_durable_loop_worked.ipynb", w1)
    write("01_durable_loop_practice.ipynb", p1)
    write("02_fan_out_fan_in_worked.ipynb", w2)
    write("02_fan_out_fan_in_practice.ipynb", p2)
    write("03_hitl_saga_scheduled_worked.ipynb", w3)
    write("03_hitl_saga_practice.ipynb", p3)
    write("04_adk_workflow_worked.ipynb", w4)
    write("04_adk_workflow_practice.ipynb", p4)
