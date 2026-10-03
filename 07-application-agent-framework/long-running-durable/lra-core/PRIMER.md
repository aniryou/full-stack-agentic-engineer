# Long-running agentic workflows — the core concept

## The problem in one paragraph

A long-running agent runs for hours, days or weeks. It writes a draft, waits for a person, changes external
systems, and retries when a model call fails. During that time, **workers stop** (scale-to-zero, deployments, OOM),
**queues deliver tasks again** (at-least-once), and **for most of the time, nothing needs to run at all**. Thus the
run cannot live in a process or in a conversation. It lives in a document, and any worker can take it.

## The core idea (this is `core.py`)

```
   Store (Firestore)          Queue (Cloud Tasks)             Worker (Cloud Run)
   one document per run  <--  task = (run, step, attempt) -->  guard -> lease -> step -> checkpoint -> enqueue next
```

This is one step, from end to end:

1. The queue delivers a task `(run_id, step, attempt)`.
2. **Guard (I2):** If the run document does not currently expect exactly that `(step, attempt)`, ack the task and do nothing more. Duplicate and late deliveries cost nothing.
3. **Lease:** Take the lease on the run for `lease_ttl` seconds. If another live worker holds the lease, back off: return HTTP 503. The queue then retries the task.
4. **Run the step** on a *copy* of the state. The step returns `next`, `done` or `wait`.
5. **Checkpoint (I1):** Write the state and the transition in one compare-and-set on `version`. **Then** enqueue the next task. Then release the lease.
6. On an exception, increase the attempt. Because of the design, this change always makes the old task stale. Enqueue the same step with a backoff delay. After `max_attempts`, fail the run.

Two things make side effects safe:

- **Effects before the checkpoint, keyed by intent (I3):** `ctx.effect("charge", fn)` records the result under `run:charge` *before* the checkpoint. If the worker crashes between the two, the retry finds the record and does not make the call again. A double charge never occurs.
- **A wait is just a status:** `("wait", key, then)` writes `WAITING` and enqueues nothing. A webhook with the same key changes the status back to `RUNNING` at `then`. A wait of days costs zero compute and zero attention.

One repair job does what the loop cannot do. This job is the **reaper** (a cron). It re-enqueues the current step of
two kinds of stuck `RUNNING` run. It also fails the waits that are past their absolute timeout. The two kinds of
stuck run are these:

- **expired lease**: The worker crashes during the step, or after a `next` checkpoint and before its enqueue. That
  checkpoint keeps the lease set until the enqueue is complete. Thus the lease expires.
- **orphan**: The run has no lease, and no process wrote to the document for `2 × lease_ttl`. `start`, `resume` or a
  retry wrote `RUNNING` without a lease and crashed before its enqueue. Thus there is no lease that can expire.

In both cases, it is safe to enqueue the step again. If the old task still exists, the queue rejects the new task as
a duplicate, because of the task name `(run, step, attempt)`.

## The same thing on Google Cloud (this is `gcp/main.py`)

| Concept | Service | Why this one |
|---|---|---|
| Store | **Firestore** | One document is one run. Transactions give compare-and-set on `version`. `create()` gives at-most-once effect records. |
| Queue | **Cloud Tasks** | *Named tasks* reject duplicates. `schedule_time` gives backoff and timers. OIDC tokens authenticate the call to the worker. |
| Worker | **Cloud Run** | It is stateless, and it scales to zero between steps. `/tasks` runs one step for each request. |
| Reaper | **Cloud Scheduler** | Cloud Scheduler sends `POST /reap` every 2 minutes. |
| Model | **Gemini on Vertex AI** | Replace `workflow.model()` with `google-genai`. |

The design does *not* use Pub/Sub for dispatch, because Pub/Sub cannot reject a duplicate by its name and cannot
schedule a message. Use Pub/Sub for notifications. If the control flow is simple and does not change, Cloud Workflows
can replace the engine. Its `await_callback` is the same wait and resume. If you want a managed runtime, ADK 2
`Workflow` on Agent Engine gives you nodes, retries, interrupts and persisted sessions.

## The patterns, in one line each (the step-up implements all of them)

- **Human-in-the-loop**: `wait`, a webhook resume and an absolute timeout. *(here)*
- **Retries with backoff**: an attempt increase, a task with a delay, and a limit on the attempts. *(here)*
- **Idempotent effects**: `effect_once`, with the intent as the key. *(here)*
- **Fan-out / fan-in**: A planner spawns child *runs* with deterministic ids. The parent waits on an atomic counter.
- **Saga**: Each step with a side effect declares its compensation. On a failure, use the effect records to compensate the completed steps in reverse sequence.
- **Reflection loop**: Critique and revision are separate steps. The loop has three exits: a threshold, a maximum number of iterations and a budget.
- **Budgets**: Do a check of the steps, the tokens, the cost and the absolute deadline *before* each model call. Fail closed.
- **Versioning**: Each run records its workflow version. New code must not silently give a different meaning to a run that is in flight.
- **Observability**: Log `run_id/step/attempt` on each line. Send an alert when a run waits past its SLA, and when the reaper recovers a run.

## What to be able to say in a design conversation

- "Write the checkpoint before the enqueue. The reaper closes the gap: expired leases and orphans with no lease." (I1)
- "`(run, step, attempt)` is the identity of work. Anything else is stale." (I2)
- "The engine records effects before the checkpoint, with the intent as the key. Thus retries never do them again." (I3)
- "A run that waits is a document with `status=WAITING` and no task. A wait costs nothing."
- "Each retry is an explicit attempt. Thus the history of the run shows each retry."
- "Use Firestore for the document, Cloud Tasks for named and scheduled dispatch, Cloud Run for stateless workers and Scheduler for repair."
