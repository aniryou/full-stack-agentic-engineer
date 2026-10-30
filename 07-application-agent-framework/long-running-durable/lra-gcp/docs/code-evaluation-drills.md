# Code-evaluation drills

There are six snippets. Each snippet has one real defect that occurs only in long-running operation. Cover the answer. Find the bug. Then compare your result with the answer. Each defect has a related invariant in `docs/primer.md` §2, and a test in `tests/` examines the defect.

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

The enqueue occurs **before** the checkpoint. If the process crashes between the two lines, a worker receives a task for a step that the run document does not yet expect. Then one of two things occurs. The first thing is that the guard rejects the task as stale forever, because the engine never writes the checkpoint. The second thing is worse: if the guard is weak, the task runs against state that the engine did not save.

Invariant I1: write the checkpoint, *then* enqueue. Let the reaper cover the opposite window.
</details>

### Drill 2 — idempotency key

```python
def charge(ctx):
    rec = ctx.effect(f"charge:{ctx.input['amount']}", lambda: payments.charge(ctx.input["amount"]))
```

<details><summary>Answer</summary>

The code makes the effect key from the *arguments*, not from the *intent*. If two different steps (or a run with a new plan) charge the same amount, they share a key. Then the second charge uses the first result again, and nothing gives a warning. But the opposite error also occurs: a valid retry after a correction of the amount creates a new charge. A key must be `run:intent` (`"charge"`). The record must also carry the external id that the compensation uses later.
</details>

### Drill 3 — lease released too early

```python
run.state = ctx.state
run.lease = None                      # "we're done with it"
run = store.save(run, expected_version=run.version)
queue.enqueue(next_task)
```

<details><summary>Answer</summary>

The code clears the lease *inside* the checkpoint. Thus a crash between `save` and `enqueue` leaves a `RUNNING` run with no lease and no task. The reaper does a scan for expired leases, and this scan cannot see the run. Hold the lease through the commit **and** the enqueue. Release the lease after the enqueue. The test `test_crash_after_commit_before_enqueue_is_repaired_by_reaper` found this exact bug.
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

The code measures the deadline from `updated_at`. `updated_at` changes on *every* write (a duplicate webhook, a cancel flag, a touch by the reaper). Thus the wait becomes longer, and nothing gives a warning. When the run starts to wait, store an absolute `timeout_at`. Compare the current time with `timeout_at`.
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

The loop has no iteration limit and no budget. Also, a critique with an incorrect format raises an exception (`KeyError`/`ValueError`), and the engine treats that exception as *retryable*. The engine retries the step three times, and each retry costs a model call. Then the run fails for the incorrect reason.

Limit the loop by the iteration count and by the budget. Record the exit reason in the state. Treat a critique with an incorrect format as "score 0", not as an exception.
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

`fail_or_compensate` enqueues a *compensation* task for the same run. When that task arrives, the guard sees `cancel_requested` again and calls `fail_or_compensate` again. Thus `fail_or_compensate` enqueues a new compensation attempt on each delivery. The run never completes its compensation. Cancellation and the budget must be a gate for **forward** steps only. The engine must always let the compensation complete (`test_cancel_while_running_is_cooperative`).
</details>

---

### Design-review prompts (5-minute answers)

1. A worker replica calls the payments API. Then, before the replica writes the checkpoint, the platform stops the replica because the replica has no more memory (OOM). Tell exactly what occurs on the next delivery, step by step. (I3: the worker finds the effect record and does not do the call again. Then the worker writes the checkpoint.)
2. Two Cloud Tasks deliveries for the same step arrive 50 ms apart on two replicas. What prevents two executions of the step, and what does the replica that loses return? (The lease, then 503, then the queue retries, then the task is stale.)
3. The product team wants "auto-approve after 48 h if no one responds." What changes? What do you refuse to auto-approve? (`on_timeout="resume"` and a marker. Refuse the auto-approval for effects that move money.)
4. Each run has a fan-out of 500 children, and there are 1 k runs/day. Where does this design break first? (The fan-in writes go to one document. Use sharding or batching.)
5. Explain why the primer says "checkpoint before enqueue". Also explain the job of the reaper in that order.
