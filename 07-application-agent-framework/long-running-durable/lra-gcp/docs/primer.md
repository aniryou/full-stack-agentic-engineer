# Long-Running Agentic Workflows on Google Cloud — a primer

*Design patterns for agents that run for hours, days or weeks, and how each one maps to a GCP service. Every pattern here is implemented and tested in this repo; file pointers are given inline.*

---

## 0. TL;DR

A long-running agent is not a chatbot with a bigger context window. It is a **durable state machine** whose transitions are chosen (partly) by a model. Everything hard about it follows from four facts:

1. **Workers die.** Cloud Run scales to zero, deploys roll, containers OOM. The run must survive its executor.
2. **Delivery is at-least-once.** Queues redeliver; webhooks are sent twice. Every unit of work needs an idempotency key.
3. **Models loop.** A "keep improving until good" loop is a credit-card sink unless bounded by steps, tokens, cost *and* time.
4. **Most of the run is idle.** Waiting three days for a human must cost nothing and must not depend on any process staying alive.

The pattern that follows: **checkpoint state in a document store, drive steps through a queue, make every side effect idempotent, suspend by writing `WAITING` and walking away.** On GCP that is Firestore + Cloud Tasks + Cloud Run, with Pub/Sub for fan-out notifications, Cloud Scheduler for repair, Cloud Workflows when the control flow is fixed, and Agent Engine (ADK) when you want the runtime managed.

---

## 1. What "long-running" changes

| Short-lived agent (seconds–minutes) | Long-running agent (hours–weeks) |
|---|---|
| State = the conversation | State = an explicit document; the conversation is a *log* |
| One process, one request | Many processes over time; no process owns the run |
| Failure = return an error | Failure = decide what to undo, then who to tell |
| Retry = call again | Retry = re-run *this step* from *this checkpoint* without repeating its effects |
| Cost bounded by one context window | Cost unbounded unless you bound it |
| Waiting = blocking | Waiting = not existing until an event arrives |

Three concrete ways the naive design fails over days (the Google ADK team lists the same three in their long-running-agents guide):

- **Context pollution** — hundreds of turns, stale tool outputs; the model loses track of which step it is on.
- **Token cost explosion** — replaying the whole history on every step.
- **Hallucinated progress after idle time** — resuming from a giant history, the model "remembers" approvals that never happened.

The fix is architectural, not a bigger model: the agent reads *where it is* from state, not from chat.

---

## 2. The core idea in ~40 lines

Strip everything away and a durable agent loop is three things: a **run document**, a **queue of step tasks**, and a **step function that can be re-run safely**.

```python
# Pseudocode of the whole engine (src/lra/core/engine.py is the real one, ~600 lines)
def worker(task):                                  # called by the queue, at-least-once
    run = store.get(task.run_id)
    if run.current_step != task.step or run.attempts[task.step] != task.attempt:
        return "stale"                             # duplicate/late delivery: ack and ignore
    run = store.acquire_lease(run.run_id, me, ttl=60s)   # one live worker per run
    state = deepcopy(run.state)                    # step works on a copy
    try:
        outcome = STEPS[run.workflow][task.step](state)   # Next(step) | Done | Wait(key) | FanOut(children)
    except Retryable:
        run.attempts[task.step] += 1               # bump attempt => old task becomes stale
        store.save(run, expected_version=run.version)
        queue.enqueue(task.run_id, task.step, run.attempts[task.step], delay=backoff(...))
        return "retry"
    run.state = state                              # --- one atomic checkpoint ---
    run.current_step = outcome.next_step           # (None if Done / a wait key if Wait)
    run.attempts[run.current_step] = 1
    store.save(run, expected_version=run.version)  # compare-and-set on version
    if outcome is Next: queue.enqueue(run.run_id, run.current_step, 1)   # AFTER the checkpoint
    if outcome is Done: notify_parent(run)
    store.release_lease(run.run_id, me)
```

Three invariants make that safe, and they are worth being able to recite:

- **I1 — Checkpoint before enqueue.** A crash between the two leaves a consistent run with an expired lease; a reaper re-enqueues it. (The reverse order risks running step N+1 against state that was never saved.)
- **I2 — `(run, step, attempt)` is the identity of work.** The queue dedups on it; the worker rejects anything that doesn't match the run document. Duplicate delivery becomes a no-op.
- **I3 — Effects are recorded before the checkpoint, keyed by intent.** `ctx.effect("charge", fn)` writes `{run:charge -> payment_id}` *then* the checkpoint. A crash in between is detected on retry (`effect exists → skip`), so retries never double-charge.

Everything else in this primer is a consequence of, or a decoration on, those three lines.

---

## 3. Pattern catalogue

Each pattern: the problem, the shape, the GCP mapping, the trade-offs, and where it lives in this repo.

### 3.1 Durable state machine (explicit state, checkpoint per step)

**Problem.** The run must be resumable from any point by any worker, days later.

**Shape.** One document per run: `status`, `current_step`, `state` (agent-owned, small), `attempts`, `history`, `budget`, `wait`, `lease`, `version`. Steps mutate a copy of `state`; the engine commits it atomically with the transition. The model's *prompt* is built from `state` (e.g. `Current step: {current_step}`), never from replayed chat.

**GCP.** Firestore (document = run, transactions for compare-and-set). Spanner if you need cross-document transactions or > ~1 write/s/document sustained. Large artefacts (drafts, tool outputs) go to Cloud Storage; only URIs live in `state`.

**Trade-offs.** Firestore's 1 MiB document cap is a feature: it forces you to keep state small and summarised. Optimistic concurrency (`version`) instead of locks means conflicting writers retry, which is rare when leases are respected.

**Repo.** `Run` in `core/models.py`; `FirestoreStateStore.save` (transactional CAS) in `adapters/gcp/firestore_store.py`.

### 3.2 Queue-driven step execution (at-least-once + idempotency keys)

**Problem.** Steps take seconds to minutes and must run somewhere that can scale to zero and back.

**Shape.** Each transition enqueues one task `{run_id, step, attempt}`. Workers are stateless HTTP endpoints. Delivery is at-least-once; correctness comes from the stale-task guard (I2) and idempotent effects (I3).

**GCP.** **Cloud Tasks** for dispatch: *named tasks* give at-most-once *enqueue* (creating an existing name fails), `schedule_time` gives delays/timers up to 30 days, per-queue rate limits protect model quota, OIDC tokens authenticate the worker call. **Pub/Sub** is the wrong tool for dispatch (no dedup by name, no scheduling) but the right one for *notifications* — progress events, audit, analytics, alerts.

**Trade-offs.** Cloud Tasks' dispatch deadline caps a single step at 30 min; anything longer is a Cloud Run Job or Batch job that the step *starts* and then `Wait`s on. Task names are reserved for a while after completion — encode the attempt in the name so retries never collide.

**Repo.** `CloudTasksQueue` in `adapters/gcp/cloud_tasks_queue.py`; worker `/tasks/step` in `services/worker/main.py`; queue config in `infra/terraform/main.tf`.

### 3.3 Leases and the reaper (crash recovery)

**Problem.** Two workers must not execute the same run at once, and a worker that dies mid-step must not strand the run.

**Shape.** Before executing, a worker atomically claims `lease = {owner, expires_at}`. A live lease held by someone else → return 503 (the queue retries later). The lease is held through commit *and* enqueue (I1) and released after. A periodic **reaper** re-enqueues the current attempt of any run whose lease expired, re-drives runs that lost their enqueue outside a lease, expires waits, and respawns missing children.

**GCP.** Lease in the Firestore document (one transaction); reaper = Cloud Scheduler → worker `/internal/reap` every 1–5 min.

**Trade-offs.** Lease TTL vs step length: TTL must exceed the longest step or the reaper double-drives (safe, thanks to I2, but wasteful). The reaper's Firestore scan is fine to ~10⁴ active runs; beyond that, index `lease.expires_at` or move leases to a separate collection.

**Repo.** `Engine.execute_task`, `Engine.reap` in `core/engine.py`; tests in `tests/test_durability.py` (chaos hooks simulate crashes at each window).

### 3.4 Retries, backoff, dead letters — and the two kinds of failure

**Problem.** Model APIs 429/503, tools time out; but some failures are *business* failures that must not be retried.

**Shape.** Distinguish `Retryable` (infra/model errors → retry with exponential backoff + jitter, bounded attempts) from `StepFailed` (business rule → fail immediately, maybe compensate). Retries are *explicit* in the engine (new attempt, delayed task), so they appear in run history and per-step policies are possible. Poison messages on notification subscriptions go to a dead-letter topic.

**GCP.** Cloud Tasks `schedule_time` for engine-managed backoff; Cloud Tasks `RetryConfig` only for the 503 "lease held" case; Pub/Sub dead-letter topics for consumers; the model adapter absorbs short 429/5xx blips itself so a step-level retry is not triggered by a 2-second outage.

**Trade-offs.** Engine-managed retries cost one Firestore write per attempt but give observability and control; queue-managed retries are free but opaque and can't distinguish failure classes.

**Repo.** `_run_step` error branches in `core/engine.py`; `GeminiLLM.generate` retry loop; `StepFailed` in `core/workflow.py`.

### 3.5 Human-in-the-loop: suspend and resume

**Problem.** The run must wait days for an approval with zero compute and zero chance of "forgetting" it was waiting.

**Shape.** A step returns `Wait(key, then, timeout, on_timeout)`. The engine writes `status=WAITING`, `wait={key, then, timeout_at}` and enqueues *nothing*. A resume is an external POST carrying `{run_id, key, payload}`; it is idempotent per `event_id`, key-scoped (a stale link cannot resume the wrong gate), and transitions the run to `RUNNING` at `then`. Timeouts are **absolute timestamps** (they survive restarts), enforced by the reaper.

**GCP.** Three equivalent mechanisms, choose by layer:
- Engine: `POST /runs/{id}/events` (this repo).
- Cloud Workflows: `events.create_callback_endpoint` + `events.await_callback` (waits up to a year; the callback URL *is* the capability).
- ADK 2 / Agent Engine: a node yields an event with `long_running_tool_ids`; the app resumes the invocation with a `FunctionResponse` for that id (`ResumabilityConfig(is_resumable=True)`).

**Trade-offs.** Timeouts need a policy: fail (safe default for anything with side effects) vs resume-with-marker (auto-approve low-risk cases). The approval link must be unguessable and single-use; treat it like a token.

**Repo.** `patterns/hitl.py`; `Engine.resume`; `workflows/research_approval.yaml`; `examples/adk_agent_engine/workflow_agent.py::review_gate`.

### 3.6 Orchestrator–worker fan-out / fan-in

**Problem.** A planner decomposes work into N independent sub-tasks; each is slow, may fail, and should be retried and budgeted on its own.

**Shape.** The planner step returns `FanOut(children, then)`. The parent checkpoints `fan_in={expected, completed, then}` **before** spawning, then starts one *child run* per sub-task with a deterministic id (`parent--childkey`), so a crash mid-spawn is repaired by respawning only the missing ones. Each child completion atomically increments the parent's counter; the completion that reaches `expected` transitions the parent. Partial failure is **data** (`state.children.failures`), and the aggregator decides whether 3/5 is enough. The planner caps N; the model does not decide the budget.

**GCP.** Children are just runs (Firestore docs + Cloud Tasks), so a 40-way fan-out spreads across Cloud Run replicas and costs nothing while waiting. Cloud Workflows `parallel` + `concurrency_limit` is the declarative equivalent for fixed fan-outs. ADK 2 `parallel_worker` fans out with asyncio *inside one invocation* — simpler, but not distributed and not individually durable.

**Trade-offs.** Fan-in via a counter in the parent document means N writes to one document; for N > ~50 batch the completions through Pub/Sub or shard the counter.

**Repo.** `patterns/orchestrator_worker.py`; `Engine._spawn_children`, `_notify_parent`, `_maybe_complete_fan_in`; `StateStore.record_child_result`.

### 3.7 Saga: acting across systems with compensation

**Problem.** The agent reserves stock, charges a card, books a shipment. There is no transaction spanning those systems; step 3 can fail after 1–2 succeeded.

**Shape.** Side-effecting steps declare a `compensate` function. On a non-retryable failure (or cancel, or budget breach), the engine runs compensations of *completed* steps in reverse order, each retried and idempotent, using the **effect record** (the payment id) rather than recomputation. If a compensation itself fails after retries, the run parks `FAILED` with a loud alert — that is a human's problem now, and hiding it would be worse.

**GCP.** No special service; this is engine logic. The alert goes to a dedicated Pub/Sub topic (`agent-alerts`) wired to on-call.

**Trade-offs.** Some effects can't be undone (an e-mail). Sequence them last, or make the compensation a corrective action. Every effect needs an idempotency key derived from *intent* (`run:charge`), not from arguments.

**Repo.** `patterns/saga.py`; `examples/procurement_saga.py`; `Engine._run_compensation`.

### 3.8 Reflection (evaluator–optimizer) as durable steps

**Problem.** "Draft → critique → revise until score ≥ 8" is the highest-value and highest-risk loop in agent design.

**Shape.** Make each iteration a step (checkpoint after every critique and revision) with **three** exits stored in state: threshold met, max iterations, budget exhausted. The critique is JSON-mode with a schema; a malformed critique counts as "not good enough" rather than crashing the loop. The exit reason is recorded so you can audit *why* the loop stopped.

**GCP.** Gemini JSON mode (`response_mime_type=application/json`) for the evaluator; nothing else special.

**Trade-offs.** Per-iteration checkpoints cost writes but make a 6-iteration loop resumable and observable. Use a cheaper model for critique than for generation if quality allows.

**Repo.** `patterns/reflection.py` (`install_reflection_steps`).

### 3.9 Budgets, deadlines, cancellation (fail closed)

**Problem.** A run that keeps going is worse than a run that stops.

**Shape.** `Budget{max_steps, max_tokens, max_cost_usd, deadline}` on the run, charged after each step and checked *before* each model call. `deadline` is absolute. Breach → the run fails (and compensates) rather than making one more call. Cancellation is a flag checked at each step boundary (cooperative); a `WAITING` run cancels immediately since no step will ever run. Budget and cancel gate *forward* progress only — a compensating run must be allowed to finish undoing.

**GCP.** Token counts from `usage_metadata`; cost from a pricing table you own; Cloud Tasks queue rate limits as the global ceiling.

**Repo.** `Budget` in `core/models.py`; `StepContext.llm`; `Engine.cancel`; tests in `tests/test_fanout_saga_budget.py`.

### 3.10 Memory and context management

**Problem.** Over weeks, what the model sees must stay small and *true*.

**Shape.** Separate four things: (1) **run state** — the state machine, always in the prompt; (2) **working memory** — this step's inputs, rebuilt from state and artefacts; (3) **episodic log** — history/events for humans and debugging, *not* replayed to the model; (4) **long-term memory** — cross-run facts about the user/domain, retrieved on demand. Summarise or drop tool outputs once they are folded into state.

**GCP.** State in Firestore; artefacts in GCS; long-term memory in Agent Engine Memory Bank or Vertex AI Vector Search; ADK sessions if you are on the managed path (they persist `ToolContext.state` every tool call — the same checkpoint idea).

**Repo.** `Run.state` vs `Run.history` split; `plan_and_fan_out` stores the plan, not the prompt.

### 3.11 Versioning: deploying while runs are in flight

**Problem.** A run started on v1 of a workflow is half-way through when v2 (which removed a step) deploys.

**Shape.** Runs record `workflow_version`; the registry holds every deployed version; new runs use the latest, in-flight runs keep theirs. If a run's version is no longer deployed, it fails loudly ("no longer deployed; redeploy or migrate") instead of guessing. Migration is an explicit operation.

**GCP.** Nothing special; ship old versions in the image until their runs drain (query Firestore by `workflow_version`).

**Repo.** `WorkflowRegistry`; tests in `tests/test_fanout_saga_budget.py`.

### 3.12 Observability: per-run, not per-request

**Problem.** A run spans dozens of requests over days; request-level dashboards see nothing wrong while a run is quietly stuck.

**Shape.** Structured logs carrying `run_id/step/attempt`; one trace per run (a span per step, attributes for tokens/cost); an event stream (`run.started`, `step.retry`, `run.waiting`, `run.fan_in_complete`, `run.compensated`, …) from which you derive the metrics that matter: **runs waiting past SLA, retries per step, cost per run, compensations per day, reaper recoveries** (a rising reaper count means workers are dying).

**GCP.** Cloud Logging JSON (`severity`, trace field), Cloud Trace via OpenTelemetry, Pub/Sub → BigQuery subscription for the event stream, alerting on `agent-alerts`.

**Repo.** `observability.py`; `Engine._publish`; Terraform BigQuery sink.

### 3.13 Scheduled and event-triggered runs

**Problem.** "Every Monday, review last week's tickets" / "when a file lands, process it".

**Shape.** A trigger starts a run with a deterministic `run_id` (e.g. `weekly-review-2026-W37`) so a double-fire is idempotent.

**GCP.** Cloud Scheduler → `POST /runs`; Eventarc (GCS/Audit events) → `POST /runs`; Pub/Sub push → `POST /runs`.

**Repo.** `POST /runs` with `Idempotency-Key` in `services/api/main.py`.

---

## 4. Reference architecture on GCP

```
                 ┌──────────────┐  start/resume/cancel   ┌───────────────────┐
  UI / Slack ───▶│ Cloud Run    │───────────────────────▶│ Firestore         │
  Scheduler      │ lra-api      │◀─ status/history ──────│ agent_runs        │
  Eventarc       └──────┬───────┘                        │ agent_effects     │
  Workflows             │ enqueue (run, step, attempt)   └────────▲──────────┘
                        ▼                                         │ CAS checkpoints, leases
                 ┌──────────────┐   OIDC push (2xx ack / 503)  ┌──┴────────────────┐
                 │ Cloud Tasks  │─────────────────────────────▶│ Cloud Run         │──▶ Vertex AI Gemini
                 │ agent-steps  │◀── enqueue next / retry ─────│ lra-worker (0..N) │──▶ tools / APIs
                 └──────────────┘                              └──┬────────┬───────┘
                        ▲                                         │        │ progress / alerts
                 Cloud Scheduler ── /internal/reap ───────────────┘        ▼
                                                                    Pub/Sub agent-events ─▶ BigQuery / UI / DLQ
```

**Data flow for one step.** Cloud Tasks POSTs `{run_id, step, attempt}` to the worker → stale guard → lease → execute against Firestore state (model + tools) → transactional checkpoint → enqueue next → publish events → release lease → 200.

**Data model (Firestore).**
- `agent_runs/{run_id}` — the `Run` document (§3.1). Indexes: `status` (reaper), `(workflow, workflow_version)` (drain before removing a version), `parent_run_id` (debugging fan-outs).
- `agent_effects/{run_id:effect_key}` — `{value}`; written with `create()` so the second writer loses.

**API contracts (`services/api`).**

| Endpoint | Semantics |
|---|---|
| `POST /runs` `{workflow, input, budget?}` + `Idempotency-Key` | 201 with the run; same key → same run |
| `GET /runs/{id}` | status, current step, wait, budget, history |
| `POST /runs/{id}/events` `{key, payload, event_id?}` | resume; `applied:false` for duplicates/wrong key (not an error) |
| `POST /runs/{id}/cancel` | cooperative; immediate if WAITING; compensates if applicable |

**Service choice table.**

| Concern | Choice | Why not the alternative |
|---|---|---|
| Run state | Firestore | Spanner only if you need cross-doc transactions or very hot documents; Memorystore is not durable enough |
| Step dispatch | Cloud Tasks | Pub/Sub lacks name dedup and scheduling; Workflows can't run dynamic graphs |
| Notifications | Pub/Sub | Cloud Tasks is point-to-point |
| Compute | Cloud Run services | GKE if you already run it; Functions for tiny tools |
| Long steps (> 30 min) | Cloud Run Jobs / Batch, started by a step that `Wait`s | Cloud Tasks dispatch deadline |
| Fixed control flow | Cloud Workflows | Less flexible, but zero worker code and managed callbacks |
| Managed agent runtime | Agent Engine + ADK | You give up some control for sessions, Memory Bank, tracing, scale-to-zero |
| Repair cron | Cloud Scheduler | Anything else needs a process alive |

---

## 5. Three ways to run it

| | Code engine (this repo) | Cloud Workflows | ADK 2 `Workflow` on Agent Engine |
|---|---|---|---|
| Control flow | dynamic; the model may choose `Next(step)` | fixed YAML DAG with branches/loops | graph with routes set by nodes (model or code) |
| Durability unit | every step is a task; state in Firestore | every step logged by the service; execution can wait ≤ 1 year | node outputs persisted in the session; replayed on resume |
| Fan-out | child runs across replicas | `parallel` branches, `concurrency_limit` | `parallel_worker` (asyncio, one invocation) |
| Retries | per-step policy in run history | declarative `retry` blocks | `RetryConfig`, in-process; count not persisted across resume |
| HITL | `Wait` + `POST /events` | `await_callback` | interrupt + `FunctionResponse` |
| Budgets | first-class | none (add in code) | via plugins/callbacks |
| Ops burden | you own queues/reaper/dashboards | nearly none | managed runtime |
| Best for | agents that **act** across systems, sagas, heavy fan-out, strict cost control | ETL-like agent pipelines with a few model calls | conversational + workflow agents, fastest path to production with Gemini tooling |

They compose. A Workflows step can `POST /runs` and `await_callback` on the run's completion event; an ADK tool can do the same. Start with the managed layers; drop to the engine where you need sagas, budgets or distributed fan-out.

---

## 6. Scale, cost and limits (worked estimate)

Assume **10,000 runs/day**, average **8 steps**, 2 model calls per step, 3 k tokens/call, one 3-day human wait per run.

- **Firestore writes**: ~3 per step (lease, checkpoint, release) → 240 k/day + effects ≈ 300 k writes/day → ~3.5 writes/s average; trivial for Firestore. Storage: 10 k docs × ~50 KB = 0.5 GB/day before TTL/archival — add a TTL policy or archive terminal runs to GCS/BigQuery after 30 days.
- **Cloud Tasks**: 80 k tasks/day ≈ 1/s average; bursts of 50/s are fine (queue `max_dispatches_per_second`). Well under quotas.
- **Cloud Run worker**: if a step averages 20 s at concurrency 8, sustained load is ~2.5 instances; peaks scale out; nights scale to zero. The **idle 3-day waits cost nothing** — the point of the design.
- **Model**: 160 k calls × 3 k tokens ≈ 480 M tokens/day. This dominates cost by orders of magnitude. Budgets (§3.9), caching, and a cheap model for critiques/routing are where money is saved; infrastructure is a rounding error.
- **Hot spots**: a parent with a 200-way fan-in writes 200 times to one document — shard or batch. The reaper's full status scan gets slow past ~10⁴ concurrently active runs — index by lease expiry.

Limits to design around: Firestore 1 MiB/doc and ~1 write/s/doc sustained; Cloud Tasks 30 min dispatch deadline and 30-day max schedule; Cloud Run 60 min request timeout (services) — for longer, Jobs; Pub/Sub at-least-once with ack deadline ≤ 10 min. Always confirm current quotas on the product pages before committing numbers in a design review.

---

## 7. Security

- **Identity per hop**: the API and worker run as distinct service accounts; only the *tasks* SA may invoke the worker (Cloud Run IAM + OIDC audience check in code); Workflows executions use their own SA.
- **Approval links are capabilities**: `run_id + gate key` must be unguessable and ideally single-use; deliver over an authenticated channel.
- **Least privilege for tools**: the worker's SA gets exactly the APIs its tools call; anything money-moving goes through an effect with an idempotency key *and* a human gate.
- **Ingress**: worker internal-only; API behind IAP/API Gateway.
- **Data**: state documents may contain PII — apply Firestore CMEK/retention as your policy requires; never put secrets in `state`.

---

## 8. Testing long-running behaviour without waiting

- **Controllable clock + in-memory adapters**: fast-forward through a 3-day wait in a test (`FakeClock.advance(days=3)`), then run the reaper.
- **Chaos hooks at each crash window** (`before_step`, `after_step_before_commit`, `after_commit_before_enqueue`): assert that recovery is exactly one re-drive with no repeated effects. See `tests/test_durability.py`.
- **Scripted model** (`FakeLLM` routes) for deterministic control-flow tests; keep a small set of *recorded* real-model runs as golden files.
- **Managed path**: ADK evalsets pre-seed session state (`current_step = ...`) to test resumption after simulated idle time.

---

## 9. Trade-offs cheat sheet

| Decision | Chose | Because | Revisit when |
|---|---|---|---|
| Checkpoint before enqueue | yes | consistent state wins over occasional reaper latency | never |
| Engine-managed retries vs queue-managed | engine | per-step policy, visible history, failure classes | you need zero-write retries at extreme scale |
| Lease in the run doc vs separate lock service | in doc | one transaction, no extra system | > 10⁴ active runs or sub-second contention |
| Children as runs vs in-process parallelism | runs | durability + individual budgets | small fan-outs where latency matters more than durability |
| Fail closed on budget | yes | an over-budget agent is a liability | never; tune limits instead |
| Compensation failure → park + alert | yes | hiding it is worse | you have an automated escalation path |
| Firestore vs Spanner | Firestore | doc-per-run fits; cost scales to zero | cross-run transactions, very hot docs |

**What I'd revisit as it grows**: an outbox table for enqueue (removes the reaper's role in the enqueue window entirely), sharded fan-in counters, per-tenant queues for isolation, and moving the run archive to BigQuery for cost analytics.

---

## 10. References (verify against current docs)

- Google Developers Blog, *Build long-running AI agents that pause, resume, and never lose context with ADK* (May 2026) — durable state schemas, event-driven dormancy, `state_delta` resumption.
- Cloud Tasks docs (named tasks, schedule time, OIDC targets, dispatch deadline).
- Cloud Workflows docs (`events.create_callback_endpoint`, `parallel`, `retry`).
- Firestore docs (transactions, document limits, TTL policies).
- ADK docs (adk.dev) — `Workflow`, `node`, `RetryConfig`, `ResumabilityConfig`; Agent Engine sessions and Memory Bank.
