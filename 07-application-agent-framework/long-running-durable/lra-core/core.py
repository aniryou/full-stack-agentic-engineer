"""core.py — a durable agent loop in ~150 lines. Read top to bottom.

A long-running agent is a state machine whose transitions a model may choose.
Three primitives make it survive crashes, duplicate deliveries and multi-day waits:

    Store   one document per run (Firestore in production)
    Queue   tasks named (run_id, step, attempt) with delays (Cloud Tasks in production)
    Engine  the worker loop: guard -> lease -> run step -> checkpoint -> enqueue next

Three invariants — everything else follows from them:
    I1  checkpoint BEFORE enqueueing the next task
    I2  a task is executed only if (step, attempt) matches the run document
    I3  side effects are recorded BEFORE the checkpoint, keyed by intent
"""

import copy
import time


class Conflict(Exception):
    """Someone else wrote the run since we read it. Reload and retry."""


class Crash(BaseException):
    """Simulates the process dying. BaseException so `except Exception` does NOT catch it —
    exactly like a real crash: no handler runs, the lease is never released."""


# ---------------------------------------------------------------- Store
class Store:
    """In-memory twin of Firestore: documents + compare-and-set on `version`."""

    def __init__(self):
        self.runs, self.effects = {}, {}

    def get(self, run_id):
        return copy.deepcopy(self.runs.get(run_id))

    def save(self, run):
        current = self.runs.get(run["id"])
        if current and current["version"] != run["version"]:
            raise Conflict(run["id"])
        run["version"] += 1
        self.runs[run["id"]] = copy.deepcopy(run)
        return run

    def effect_once(self, key, fn):
        """I3: run `fn` at most once per key, across retries and duplicate deliveries."""
        if key not in self.effects:
            self.effects[key] = fn()
        return self.effects[key]

    def active(self):
        return [copy.deepcopy(r) for r in self.runs.values() if r["status"] in ("RUNNING", "WAITING")]


# ---------------------------------------------------------------- Queue
class Queue:
    """In-memory twin of Cloud Tasks: named tasks (dedup) with a `due` time."""

    def __init__(self):
        self.tasks, self.names = [], set()

    def push(self, run_id, step, attempt, now, delay=0):
        name = f"{run_id}/{step}/{attempt}"
        if name in self.names:
            return False  # Cloud Tasks rejects a duplicate name -> at-most-once enqueue
        self.names.add(name)
        self.tasks.append((now + delay, run_id, step, attempt))
        return True

    def pop(self, now):
        due = [t for t in self.tasks if t[0] <= now]
        if not due:
            return None
        task = min(due)
        self.tasks.remove(task)
        return task[1:]


# ---------------------------------------------------------------- Steps
class Ctx:
    """What a step sees: a copy of the state (committed atomically if the step returns),
    the run input, and `effect()` for side effects."""

    def __init__(self, run, store):
        self.state, self.input, self.run_id, self._store = run["state"], run["input"], run["id"], store

    def effect(self, key, fn):
        return self._store.effect_once(f"{self.run_id}:{key}", fn)


# A step returns one of:
#   ("next", "step_name")               continue
#   ("done", result)                    finish
#   ("wait", "event_key", "step_name")  suspend until an event with that key arrives (days are fine),
#                                       then continue at step_name


# ---------------------------------------------------------------- Engine
class Engine:
    def __init__(self, store, queue, steps, clock=time.time, lease_ttl=60, max_attempts=3):
        self.store, self.queue, self.steps = store, queue, steps
        self.clock, self.lease_ttl, self.max_attempts = clock, lease_ttl, max_attempts
        self.crash_before_enqueue = False   # test hook: die between checkpoint and enqueue (once)

    def start(self, run_id, first_step, input):
        if self.store.get(run_id):           # idempotent start
            return self.store.get(run_id)
        run = {"id": run_id, "status": "RUNNING", "step": first_step, "attempts": {first_step: 1},
               "state": {}, "input": input, "history": [], "wait": None, "lease": None,
               "result": None, "error": None, "version": 0}
        self.store.save(run)
        self.queue.push(run_id, first_step, 1, self.clock())
        return run

    def execute(self, run_id, step, attempt):
        """Worker entry point (one Cloud Tasks delivery). Returns a short outcome code."""
        run = self.store.get(run_id)
        now = self.clock()
        # I2: is this exactly the work the run expects right now?
        if not run or run["status"] != "RUNNING" or run["step"] != step or run["attempts"].get(step) != attempt:
            return "stale"
        # one live worker per run
        if run["lease"] and run["lease"] > now:
            return "busy"                     # HTTP 503 -> Cloud Tasks retries later
        run["lease"] = now + self.lease_ttl
        run = self.store.save(run)

        ctx = Ctx(copy.deepcopy(run), self.store)
        try:
            outcome = self.steps[step](ctx)
        except Exception as e:                # infra/model error: retry with backoff
            run["history"].append((step, attempt, "error", repr(e)))
            if attempt >= self.max_attempts:
                return self._finish(run, "FAILED", error=f"{step} failed after {attempt} attempts: {e!r}")
            run["attempts"][step] = attempt + 1           # old task becomes stale by construction
            run["lease"] = None
            self.store.save(run)
            self.queue.push(run_id, step, attempt + 1, now, delay=2 ** attempt)
            return "retry"

        # --- I1: one atomic checkpoint, then enqueue ---
        kind, value = outcome[0], outcome[1]
        run["state"] = ctx.state
        run["history"].append((step, attempt, "ok", kind))
        if kind == "done":
            return self._finish(run, "SUCCEEDED", result=value)
        if kind == "wait":
            run["status"], run["lease"] = "WAITING", None
            run["wait"] = {"key": value, "then": outcome[2], "timeout": now + 3 * 86400}
            self.store.save(run)              # no task is enqueued: sleeping costs nothing
            return "waiting"
        run["step"] = value
        run["attempts"][value] = run["attempts"].get(value, 0) + 1
        self.store.save(run)                  # checkpoint first (the lease stays set until we are done) ...
        if self.crash_before_enqueue:
            self.crash_before_enqueue = False
            raise Crash("died after the checkpoint, before the enqueue")
        self.queue.push(run_id, value, run["attempts"][value], self.clock())   # ... then enqueue
        run["lease"] = None
        self.store.save(run)                  # ... then release the lease
        return "ok"

    def resume(self, run_id, key, payload):
        """Wake a WAITING run. Idempotent: a second delivery of the same event is ignored."""
        run = self.store.get(run_id)
        if not run or run["status"] != "WAITING" or run["wait"]["key"] != key:
            return None
        then = run["wait"]["then"]
        run["state"].setdefault("events", {})[key] = payload
        run["status"], run["wait"], run["step"] = "RUNNING", None, then
        run["attempts"][then] = run["attempts"].get(then, 0) + 1
        self.store.save(run)
        self.queue.push(run_id, then, run["attempts"][then], self.clock())
        return run

    def reap(self):
        """Repair job (Cloud Scheduler -> /reap): expired leases and timed-out waits."""
        now, repaired = self.clock(), []
        for run in self.store.active():
            if run["status"] == "RUNNING" and run["lease"] and run["lease"] <= now:
                run["lease"] = None
                self.store.save(run)
                self.queue.push(run["id"], run["step"], run["attempts"][run["step"]], now)  # dedup makes this safe
                repaired.append(("lease", run["id"]))
            elif run["status"] == "WAITING" and run["wait"]["timeout"] <= now:
                self._finish(run, "FAILED", error=f"timed out waiting for {run['wait']['key']}")
                repaired.append(("timeout", run["id"]))
        return repaired

    def _finish(self, run, status, result=None, error=None):
        run.update(status=status, step=None, lease=None, wait=None, result=result, error=error)
        self.store.save(run)
        return status.lower()


def drain(engine, queue, clock):
    """Local stand-in for Cloud Tasks pushing to the worker until nothing is due."""
    log = []
    while True:
        task = queue.pop(clock())
        if task is None:
            return log
        try:
            log.append((task[1], engine.execute(*task)))
        except Crash:
            # The worker gave no response, so Cloud Tasks redelivers the same task later.
            queue.names.discard("/".join(map(str, task)))
            queue.push(*task, clock(), delay=30)
            log.append((task[1], "CRASH"))
