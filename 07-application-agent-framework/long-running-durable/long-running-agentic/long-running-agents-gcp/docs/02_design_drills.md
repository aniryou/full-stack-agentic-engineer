# Design drills — long-running agents

Two halves, matching the loop's format: **System Design** (broad ask → clarifying questions → design → trade-offs → estimation) and **Code Evaluation** (read code, find the bug, name the optimal fix). Cover the answers before reading.

---

## Part A · System design prompts

### A1. "A bank wants an agent that processes supplier invoices end to end: read the PDF, match to a PO, get approval above a threshold, schedule payment. Design it."
**Clarify:** volume/day; approval SLA (hours? days?); what "schedule payment" touches (a core-banking API with idempotency keys?); regulatory audit needs; human override paths.
**Shape:** P1 durable loop per invoice (Cloud Run + Cloud Tasks + Firestore); P3 gate on `schedule_payment` above the threshold with a 3-business-day TTL and escalation; Document AI for extraction as a synchronous tool (< 60 s) or P7 if batch; journal = audit trail; Gemini Flash for matching, Pro only for exceptions.
**Trade-offs to voice:** Workflows callback vs your own approval endpoint (declarative durability vs dynamic control flow); Firestore vs Cloud SQL for the journal (ops simplicity vs SQL reporting — you can also export the journal to BigQuery); re-verify PO status *immediately* before payment (staleness guard).
**Estimate:** 50k invoices/day → ~1M wake-ups/day (~12/s) → one Tasks queue; tokens ≈ 50k × 8 steps × 5k ≈ 2B/day → route by model tier and quote the resulting order-of-magnitude monthly cost.
**Failure walk:** payment API times out after charging → intent journaled with key → retry re-sends the same key → bank de-dups → one payment.

### A2. "Design a research agent that answers a question by reading 200 web pages in parallel."
**Shape:** P2. Planner splits into ≤ 50 subtasks (cap!), Pub/Sub fan-out, idempotent workers write `runs/{id}/subtasks/{sid}` (not a single counter doc at this N), orchestrator polls a count query via a delayed task every 30 s, or Cloud Workflows `parallel` with `concurrency_limit`. Aggregator with a token budget; store fetched pages in GCS, pass URIs.
**Trade-offs:** Pub/Sub (fan-out, no per-message scheduling) vs Tasks (per-task scheduling/rate limiting, no fan-out semantics); why not one giant prompt (context limits, cost, no partial progress).
**Gotcha to name:** the 200th worker's crash after commit — late duplicates must also see `all_done`; the aggregate task is named so the enqueue is idempotent.

### A3. "An agent must wait for a customer to upload a document — could take two weeks — then continue."
**Shape:** park `WAITING_EVENT`; Eventarc on the GCS bucket (object finalized) → Pub/Sub → `/internal/callbacks` resumes the run; no polling. Backstop: a Cloud Task at +14 days to expire/escalate. In Workflows: `await_callback` with a 14-day timeout (execution limit is a year).
**Trade-offs:** callback vs poll (cost, latency, coupling); where the reminder lives (Tasks `schedule_time` max 30 days — chain tasks beyond that).

### A4. "Nightly, an agent reconciles 3 systems and fixes discrepancies. Sometimes a fix has to be undone."
**Shape:** Cloud Scheduler → Pub/Sub → tick (P5 lease + due-time) → for each discrepancy, a P4 saga with compensations; Cloud Run job if the scan itself takes hours; every fix idempotent by discrepancy id.
**Voice:** compensation failure = escalation ticket, never silent; the LLM proposes the fix plan, code owns the saga state machine.

### A5. "Migrate this LangGraph agent to Google's stack with minimal rewrite."
**Shape:** ADK 2 `Workflow` on Cloud Run with Cloud SQL sessions (architecture C), or Agent Runtime for managed sessions/memory; interrupts → `RequestInput`; checkpointer → session service URI; scheduler → Cloud Scheduler → Pub/Sub → `/wake`.
**Gotchas:** ADK resume is at-least-once (tools must be idempotent); `temp:` state lost on resume; the built-in Pub/Sub trigger route makes a new session — add a resume endpoint; agent nodes cannot be first after START without an input.

### A6. "What would you measure to know the agent fleet is healthy?"
steps/run, tokens/run, cost/run (by model), time-in-WAITING_*, approvals older than TTL/2, recoveries/run, lease conflicts, dead-letter depth, poll attempts per async tool, Tasks queue depth vs dispatch rate, p95 step latency; plus offline replay evals from journals on every prompt/model change.

---

## Part B · Code evaluation — find the bug

### B1. Retry-safe? (Python)
```python
def step(run_id):
    run = store.get(run_id)
    decision = llm.decide(prompt(run))
    if decision.kind == "tool":
        result = tools[decision.name](**decision.args)      # e.g. charge a card
        run.journal.append({"tool": decision.name, "args": decision.args, "result": result})
        store.save(run)
    enqueue_next(run_id)
```
**Bugs:** (1) no write-ahead intent — a crash after the tool call re-asks the model on retry and may charge again with different args; (2) no idempotency key passed downstream; (3) `store.save` has no version check — two concurrent deliveries both succeed; (4) `enqueue_next` after save is unnamed — duplicates fan out. **Fix:** journal `STARTED` + key → save → execute under key → save with `version` → named task.

### B2. Fan-in counter (Python, Firestore)
```python
def on_subtask_done(run_id, sid, result):
    doc = db.collection("runs").document(run_id)
    snap = doc.get()
    fan = snap.get("fan")
    fan["results"][sid] = result
    fan["completed"] += 1
    doc.update({"fan": fan})
    if fan["completed"] == fan["expected"]:
        publish("aggregate", run_id)
```
**Bugs:** read-modify-write without a transaction → lost updates under concurrency; no duplicate guard (`sid` already present); `publish` is not idempotent, and if the process dies between `update` and `publish` nobody aggregates. **Fix:** transaction with the `sid in results` guard; return `all_done` from the transaction regardless of duplicate; enqueue a *named* task.

### B3. Approval handler
```python
@app.post("/runs/{run_id}/approve")
def approve(run_id, approved: bool):
    run = store.get(run_id)
    if approved:
        decision = llm.decide(prompt(run) + "\nThe human approved. Proceed.")
        execute(decision)
    run.status = "RUNNING"; store.save(run)
```
**Bugs:** no token / caller check; the model is re-asked after approval so what runs may differ from what was approved; not idempotent (double click executes twice); status set to RUNNING even on reject with no feedback to the model; no lease. **Fix:** verify token (constant-time), journal `HUMAN` + the stored call as a `STARTED` intent, let the loop execute it, no-op if not `WAITING_HUMAN`.

### B4. ADK node
```python
@node
def check_front(ctx):
    while venue.position(ctx.state["ticket"]) > 0:
        time.sleep(5)
    return {"ready": True}
```
**Bugs:** blocks the process (Cloud Run request timeout, no resume, pays for idle CPU); should return `RequestInput` and be `rerun_on_resume=True`; a synchronous check must not be marked long-running, but a wait must yield.

### B5. Which tool is long-running?
```python
tools=[LongRunningFunctionTool(func=check_queue), join_queue]
```
**Bug:** backwards. `join_queue` returns a handle to work still in progress (long-running); `check_queue` returns a point-in-time answer — marking it long-running pauses the run on every status check, including the one that says "you're at the front, buy now".

### B6. Scheduler tick
```python
@app.post("/tick")
def tick():
    run = store.get("presale-monitor")
    if time.time() - run.state["last_tick"] > 300:
        do_work(run)
        run.state["last_tick"] = time.time(); store.save(run)
```
**Bugs:** two instances receiving the same tick both pass the time check (no lease); Scheduler double-fire → double work; a crash inside `do_work` holds nothing so a retry re-runs it (fine only if `do_work` is idempotent — is it?). **Fix:** `acquire_lease` (TTL) + `next_due` + intent record.

### B7. Cloud Tasks handler status codes
```python
@app.post("/internal/tasks/step")
def handle(env):
    try:
        loop.step(env["run_id"])
    except Exception:
        return {"ok": False}          # HTTP 200
```
**Bug:** swallowing errors with a 200 tells Cloud Tasks the work is done → the run silently stalls. **Fix:** let exceptions surface as 5xx (retry with back-off), return 429 on `LeaseHeld`, and only 200 after a durable commit; add a dead-letter/alert on max attempts.

### B8. Budget in the prompt
```python
instruction = "Stop after at most 10 tool calls and never spend more than $2."
```
**Bug:** prompts are not enforcement; the model has no counter and no cost meter. **Fix:** `check_budget(run)` before every model call with `usage` accumulated from `usage_metadata`; the run FAILS with a reason.

---

## Part C · Rapid-fire (one sentence each)
* Why two saves per tool step? — intent before act, result after; the model is non-deterministic.
* Lease vs lock? — a lease expires; a dead worker can't wedge the run.
* Exactly-once? — only pull-subscription delivery; never actions → idempotent handlers.
* Workflows vs hand-rolled loop? — static graph and declarative waits vs model-chosen next step.
* Why cap subtasks? — cost, fan-in contention, aggregator context.
* `rerun_on_resume` True/False? — re-check the world vs never repeat a side effect.
* Where does a 6 KB seat map go? — an artifact (GCS), return a filename.
* Why not `time.sleep` in a node? — it holds a process; return an interrupt and let a clock wake you.
