# Code-evaluation drills

Six snippets, each with one real defect that shows up only in long-running operation. Cover the answer, find the bug, then check. Every defect corresponds to an invariant in `docs/primer.md` §2 and is tested in `tests/`.

---

### Drill 1 — order of operations

```python
def commit(run, outcome):
    queue.enqueue(StepTask(run.run_id, outcome.step, run.attempt_of(outcome.step) + 1))
    run.current_step = outcome.step
    run.bump_attempt(outcome.step)
    store.save(run, expected_version=run.version)
```

<details><summary>Answer</summary>

Enqueue happens **before** the checkpoint. If the process dies between the two lines, a worker receives a task for a step the run document doesn't yet expect — it is either rejected as stale forever (the checkpoint never lands) or, worse, executes against un-saved state if the guard is weak. Invariant I1: checkpoint, *then* enqueue; let the reaper cover the reverse window.
</details>

### Drill 2 — idempotency key

```python
def charge(ctx):
    rec = ctx.effect(f"charge:{ctx.input['amount']}", lambda: payments.charge(ctx.input["amount"]))
```

<details><summary>Answer</summary>

The effect key is derived from the *arguments*, not the *intent*. Two different steps (or a re-planned run) charging the same amount share a key and the second one silently reuses the first result; conversely a legitimate retry after the amount was corrected creates a new charge. Keys must be `run:intent` (`"charge"`), and the record must carry the external id used later by the compensation.
</details>

### Drill 3 — lease released too early

```python
run.state = ctx.state
run.lease = None                      # "we're done with it"
run = store.save(run, expected_version=run.version)
queue.enqueue(next_task)
```

<details><summary>Answer</summary>

Clearing the lease *inside* the checkpoint means a crash between `save` and `enqueue` leaves a `RUNNING` run with no lease and no task. The reaper's expired-lease scan can't see it. Hold the lease through commit **and** enqueue; release it afterwards (this exact bug was found by `test_crash_after_commit_before_enqueue_is_repaired_by_reaper`).
</details>

### Drill 4 — relative timeout

```python
def request_review(ctx):
    ctx.state["review_deadline_s"] = 3 * 24 * 3600
    return Wait(key=f"editor:{ctx.run_id}", then="publish")

# reaper
if now - run.updated_at > timedelta(seconds=run.state["review_deadline_s"]): fail(run)
```

<details><summary>Answer</summary>

The deadline is anchored to `updated_at`, which changes on *every* write (a duplicate webhook, a cancel flag, a reaper touch). The wait silently extends. Store an absolute `timeout_at` at suspension time and compare against that.
</details>

### Drill 5 — the loop with two exits

```python
def critique(ctx):
    score = int(ctx.llm(CRITIQUE, json_mode=True).json()["score"])
    if score >= 8:
        return Next("publish")
    return Next("revise")
```

<details><summary>Answer</summary>

No iteration cap, no budget, and a malformed critique raises (`KeyError`/`ValueError`) which is treated as *retryable* — the step is retried three times, each costing a model call, and then the run fails for the wrong reason. Bound the loop by iteration and budget, record the exit reason in state, and treat a malformed critique as "score 0" rather than an exception.
</details>

### Drill 6 — cancellation vs compensation

```python
def execute_task(task):
    run = store.get(task.run_id)
    if run.cancel_requested:
        return fail_or_compensate(run, "cancelled")
    ...
```

<details><summary>Answer</summary>

`fail_or_compensate` enqueues a *compensation* task for the same run; when that task arrives, the guard sees `cancel_requested` again and calls `fail_or_compensate` again → a fresh compensation attempt is enqueued every delivery. The run never finishes compensating. Cancellation and budget must gate **forward** steps only; compensation must always be allowed to complete (`test_cancel_while_running_is_cooperative`).
</details>

---

### Design-round prompts (5-minute answers)

1. A worker replica is OOM-killed after calling the payments API but before checkpointing. Walk through exactly what happens on the next delivery. (I3: effect record found → skipped → checkpoint proceeds.)
2. Two Cloud Tasks deliveries for the same step arrive 50 ms apart on two replicas. What prevents double execution, and what does the loser return? (Lease → 503 → queue retries → stale.)
3. The product team wants "auto-approve after 48 h if no one responds." What changes, and what would you refuse to auto-approve? (`on_timeout="resume"` + marker; refuse for money-moving effects.)
4. Fan-out of 500 children per run at 1 k runs/day. Where does this design break first? (Fan-in writes to one document; shard/batch.)
5. Explain why the primer says "checkpoint before enqueue" and what the reaper's job is in that ordering.
