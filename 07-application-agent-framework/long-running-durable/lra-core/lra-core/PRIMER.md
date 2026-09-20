# Long-running agentic workflows — the core concept

## The problem in one paragraph

A long-running agent runs for hours, days or weeks: it drafts something, waits for a human, acts on
external systems, retries when a model call fails. Over that span **workers die** (scale-to-zero,
deploys, OOM), **queues redeliver** (at-least-once), and **most of the time nothing should be
running at all**. So the run cannot live in a process or in a conversation. It lives in a
document, and any worker can pick it up.

## The core idea (this is `core.py`)

```
   Store (Firestore)          Queue (Cloud Tasks)             Worker (Cloud Run)
   one document per run  <--  task = (run, step, attempt) -->  guard -> lease -> step -> checkpoint -> enqueue next
```

One step, end to end:

1. A task `(run_id, step, attempt)` is delivered.
2. **Guard (I2):** if the run document does not currently expect exactly that `(step, attempt)`, ack and ignore. Duplicate and late deliveries cost nothing.
3. **Lease:** claim the run for `lease_ttl` seconds. Another live worker holding it → back off (HTTP 503; the queue retries).
4. **Run the step** against a *copy* of the state. The step returns `next`, `done` or `wait`.
5. **Checkpoint (I1):** write state + transition in one compare-and-set on `version`, **then** enqueue the next task, then release the lease.
6. On an exception: bump the attempt (the old task is now stale by construction) and enqueue the same step with a backoff delay. After `max_attempts`, fail.

Two things make side effects safe:

- **Effects before the checkpoint, keyed by intent (I3):** `ctx.effect("charge", fn)` records the result under `run:charge` *before* the checkpoint. If the worker dies in between, the retry finds the record and skips the call. No double charge — ever.
- **A wait is just a status:** `("wait", key, then)` writes `WAITING` and enqueues nothing. A webhook with the matching key flips it back to `RUNNING` at `then`. Days of waiting cost zero compute and zero attention.

One repair job covers what the loop cannot: the **reaper** (a cron) re-enqueues runs whose lease
expired (worker died mid-step, or between checkpoint and enqueue) and fails waits past their
absolute timeout.

## The same thing on Google Cloud (this is `gcp/main.py`)

| Concept | Service | Why this one |
|---|---|---|
| Store | **Firestore** | document = run; transactions give compare-and-set on `version`; `create()` gives at-most-once effect records |
| Queue | **Cloud Tasks** | *named tasks* reject duplicates; `schedule_time` gives backoff and timers; OIDC tokens authenticate the worker call |
| Worker | **Cloud Run** | stateless, scales to zero between steps; `/tasks` executes one step per request |
| Reaper | **Cloud Scheduler** | `POST /reap` every 2 minutes |
| Model | **Gemini on Vertex AI** | swap `workflow.model()` for `google-genai` |

Pub/Sub is *not* used for dispatch (no name dedup, no scheduling); use it for notifications.
If the control flow is fixed and simple, Cloud Workflows can replace the engine (its
`await_callback` is the same wait/resume). If you want the runtime managed, ADK 2 `Workflow` on
Agent Engine gives you nodes, retries, interrupts and persisted sessions.

## The patterns, in one line each (the step-up implements all of them)

- **Human-in-the-loop** — `wait` + webhook resume + absolute timeout. *(here)*
- **Retries with backoff** — attempt bump, delayed task, bounded. *(here)*
- **Idempotent effects** — `effect_once`, intent-keyed. *(here)*
- **Fan-out / fan-in** — a planner spawns child *runs* with deterministic ids; parent waits on an atomic counter.
- **Saga** — side-effecting steps declare compensations; on failure, undo completed steps in reverse using the effect records.
- **Reflection loop** — critique/revise as separate steps with three exits: threshold, max iterations, budget.
- **Budgets** — steps/tokens/cost/absolute deadline checked *before* each model call; fail closed.
- **Versioning** — runs record their workflow version; new code must not silently re-interpret in-flight runs.
- **Observability** — log `run_id/step/attempt` on every line; alert on runs waiting past SLA and on reaper recoveries.

## What to be able to say in a design conversation

- "Checkpoint before enqueue; the reaper covers the gap." (I1)
- "`(run, step, attempt)` is the identity of work; anything else is stale." (I2)
- "Effects are recorded before the checkpoint, keyed by intent, so retries never repeat them." (I3)
- "A waiting run is a document with `status=WAITING` and no task — sleeping is free."
- "Retries are explicit attempts, so they're visible in the run's history."
- "Firestore for the document, Cloud Tasks for named/scheduled dispatch, Cloud Run for stateless workers, Scheduler for repair."
