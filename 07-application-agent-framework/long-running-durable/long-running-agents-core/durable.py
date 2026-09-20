"""durable.py — the core of a long-running agent in one file, stdlib only.

THE IDEA
  A long-running agent is not a long-running process.
      wake up → do ONE step → checkpoint → go back to sleep
  The store is the only memory. A queue delivers wake-ups (at-least-once).
  Waiting (for a human, a slow job, the world) means PARKING the run — nothing
  runs, nothing costs — until something calls resume().

FIVE RULES that make it safe, each marked in the code with ## (n)
  (1) durable state   – every step ends with save(); after a crash, only the store is real
  (2) intent → act    – journal what you're about to do (with an idempotency key), save,
                        THEN do it. A retry re-does the same call with the same key;
                        it never asks the model again (the model is non-deterministic).
  (3) lease           – one worker advances a run at a time; the lease EXPIRES, so a
                        dead worker cannot wedge the run forever.
  (4) budget          – a hard step limit in code; prompts cannot enforce limits.
  (5) park, don't wait – a wait is a status + a token, not a sleeping process.

GCP mapping: Store → Firestore doc per run · Queue → Cloud Tasks (named tasks, OIDC)
· Agent.handle → a Cloud Run POST handler · resume() → an HTTP endpoint hit by a person,
a webhook, or Cloud Scheduler · Model → Gemini · PaymentAPI → any API with Idempotency-Key.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field


class Crash(RuntimeError):
    """Simulated process death (SIGKILL). Nothing after it runs, no lease is released."""


class LeaseHeld(RuntimeError):
    pass


class Wait:
    """A tool returns this when the work continues elsewhere (a slow job, a webhook)."""

    def __init__(self, token):
        self.token = token


# ---------------------------------------------------------------- the run
@dataclass
class Run:
    id: str
    goal: str
    status: str = "RUNNING"                       # RUNNING | WAITING | DONE | FAILED
    journal: list = field(default_factory=list)   # append-only: decision / intent(+result)
    max_steps: int = 10                           ## (4)
    lease: dict | None = None                     ## (3) {"owner", "until"}
    waiting_on: dict | None = None                ## (5) {"token", "why"}
    result: str | None = None

    def pending_intent(self):
        """Find a step we started but never finished, if there is one.

        This is the first thing we check when the agent wakes up. A step that isn't marked "done"
        means we wrote down "about to do X" but never recorded that X worked. Maybe it did, maybe a
        crash stopped us first. Either way we redo that same step (safely) rather than ask the model
        for a new one. Returns the unfinished step, or None if nothing is half-done.
        """
        return next((s for s in self.journal if s["type"] == "intent" and not s["done"]), None)

    def decisions(self):
        """How many times we've asked the model so far. The step budget (rule 4) caps this."""
        return sum(1 for s in self.journal if s["type"] == "decision")


class Store:
    """JSON on disk — durable enough to prove the point (kill the process, restart, resume).
    Real: one Firestore document per run; acquire_lease is a transaction (compare-and-set)."""

    def __init__(self, path):
        self.path = path
        # Read the whole store into memory at startup (empty if the file isn't there yet). The file
        # on disk is what's real: kill the process and restart, and we pick up exactly where we left off.
        self.runs = json.load(open(path)) if os.path.exists(path) else {}
        self.saves = 0                            # how many times we've written; the tests check this

    def get(self, run_id):
        # Build a fresh Run from what's stored. Callers change their own copy, so the store isn't
        # touched until they hand it back to save().
        return Run(**self.runs[run_id])

    def save(self, run):                                                    ## (1)
        self.runs[run.id] = asdict(run)
        # Write to a temp file first, then rename it over the real one in a single step. If we crash
        # partway through the write, the old file is still intact — we never leave a half-written store.
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.runs, f, indent=1)
        os.replace(tmp, self.path)                                          # atomic rename
        self.saves += 1
        return run

    def acquire_lease(self, run_id, owner, ttl, now):                       ## (3)
        run = self.get(run_id)
        # Only say no if someone ELSE holds the lease and it hasn't run out yet. An expired lease is
        # free to take (its worker likely died), and taking our own again is fine.
        if run.lease and run.lease["until"] > now and run.lease["owner"] != owner:
            raise LeaseHeld(f"{run_id} is held by {run.lease['owner']} until {run.lease['until']:.0f}")
        run.lease = {"owner": owner, "until": now + ttl}
        return self.save(run)                     # saving the lease is a checkpoint in itself

    def release_lease(self, run_id):
        # Read the run again before clearing the lease. While we held it, the step changed the run
        # and saved it, so the copy on disk is newer than the one step() still has in hand.
        run = self.get(run_id)
        run.lease = None
        self.save(run)


class Queue:
    """Stands in for Cloud Tasks: at-least-once delivery, NAMED tasks are de-duplicated,
    and a delivery whose handler raises is retried (the message stays in the queue)."""

    def __init__(self):
        self.items, self.seen, self.delivered = [], set(), 0

    def enqueue(self, name, msg):
        if name in self.seen:                     # a second enqueue of the same unit of work collapses
            return
        self.seen.add(name)
        self.items.append(msg)

    def deliver_one(self, handler):
        if not self.items:
            return False
        handler(self.items[0])                    # raises → message stays → retried later
        self.items.pop(0)
        self.delivered += 1
        return True

    def drain(self, handler):
        while self.deliver_one(handler):
            pass

    def duplicate_next(self):
        """Simulate at-least-once: the next message will be delivered twice."""
        self.items.insert(0, dict(self.items[0]))


# ---------------------------------------------------------------- the agent
class Agent:
    def __init__(self, store, queue, model, tools, worker="worker-1", clock=time.time, lease_ttl=60):
        self.store, self.queue, self.model, self.tools = store, queue, model, tools
        self.worker, self.clock, self.lease_ttl = worker, clock, lease_ttl
        self.crash_at = set()                     # fault injection: {"after_side_effect", "after_intent"}

    # --- lifecycle -----------------------------------------------------------
    def start(self, goal, max_steps=10):
        """Create a run, queue its first wake-up, and return right away. No thinking happens here —
        that waits until the queue delivers the wake-up and someone calls handle()."""
        run = self.store.save(Run(id="run_" + uuid.uuid4().hex[:6], goal=goal, max_steps=max_steps))
        self._wake(run)
        return run

    def handle(self, msg):
        """What the Cloud Run handler does with one wake-up. Return only after a durable commit."""
        return self.step(msg["run_id"], expect=tuple(msg["expect"]))

    def _wake(self, run):
        """Queue the next wake-up for this run.

        Two small tricks keep this safe even though the queue may deliver the same message twice.
        The message NAME describes the exact job to do, so queuing it twice keeps only one copy.
        The `expect` payload records what the run looked like just now, so if a duplicate shows up
        late, it can tell the run has moved on and quietly skip itself.
        """
        todo = run.pending_intent()
        # A snapshot of where the run is right now: how long the journal is, and whether a step is
        # half-finished. When this wake-up is delivered, the step compares against it — if things
        # have changed since, this wake-up is out of date and does nothing.
        expect = (len(run.journal), todo is not None)
        # The name is the job's identity. "exec:<key>" means "finish this half-done step";
        # "decide:<n>" means "ask the model what to do next". Same name means same job, and the
        # queue keeps only one — so an accidental double-queue can't cause double work.
        name = f"{run.id}:exec:{todo['key']}" if todo else f"{run.id}:decide:{len(run.journal)}"
        self.queue.enqueue(name, {"run_id": run.id, "expect": expect})

    # --- one step ---------------------------------------------------------------
    def step(self, run_id, expect=None):
        """Move a run forward by one step, holding a lease so no other worker joins in.

        How we leave matters as much as what we do:
          - a Crash keeps the lease held — a dead worker must NOT look like it finished cleanly;
          - any other error hands the lease back, so a healthy worker can pick the run up and retry;
          - finishing normally hands the lease back too.
        Whatever happens, the only thing that survives is what we saved to the store.
        """
        run = self.store.acquire_lease(run_id, self.worker, self.lease_ttl, self.clock())   ## (3)
        try:
            run = self._step(run, expect)
        except Crash:
            raise                                 # a dead worker never releases its lease
        except Exception:
            self.store.release_lease(run_id)
            raise
        self.store.release_lease(run_id)
        return run

    def _step(self, run, expect):
        """One step: work out what to do, then do it. Doing those in this order is what keeps retries safe."""
        if run.status != "RUNNING":
            return run                            # done, failed, or parked and waiting → nothing to do here
        intent = run.pending_intent()
        # Nothing half-finished, but this wake-up expected the run to look different. So the work it
        # was sent to do has already happened (the queue handed us a duplicate). Skip it — don't
        # kick off a brand-new step nobody asked for.
        if intent is None and expect is not None and expect != (len(run.journal), False):
            return run                            # duplicate or stale wake-up: this was already handled
        if intent is None:
            intent = self._decide(run)            # ask the model and write down the plan (checkpoint 1)
            if intent is None:
                return run                        # the model gave a final answer, or we hit the budget
        return self._execute(run, intent)         # carry out the plan (checkpoint 2), then queue the next step

    def _decide(self, run):
        """Ask the model what to do, and write the answer down BEFORE acting on it.
        Returns the step to carry out, or None if the run is now finished."""
        if run.decisions() >= run.max_steps:                                            ## (4)
            self._finish(run, "FAILED", f"budget: {run.max_steps} steps")
            return None                           # a hard limit in code; no prompt can argue its way past it
        d = self.model.decide(run.journal)        # the one unpredictable call — everything after can be replayed
        run.journal.append({"type": "decision", **d})
        if "final" in d:                          # the model gave an answer instead of picking a tool
            self._finish(run, "DONE", d["final"])
            return None
        # Write down the tool call we're about to make. The key is built from the journal position,
        # so if we ever replay this step we build the same key again — and the tool treats both
        # tries as the same call rather than doing the work twice.
        intent = {"type": "intent", "tool": d["tool"], "args": d["args"],
                  "key": f"{run.id}:{len(run.journal)}", "done": False, "approved": False}
        run.journal.append(intent)
        self.store.save(run)                      ## (2) checkpoint 1: save the plan BEFORE doing it
        self._crash_maybe("after_intent")         # crash here? recovery finds the unfinished step and redoes it
        return intent

    def _execute(self, run, intent):
        """Carry out the step we wrote down. Running it twice is safe — the key keeps the tool call to one."""
        tool = self.tools[intent["tool"]]
        # Some tools need a person's OK first. Park and wait; resume() sets `approved` and comes
        # back through here to actually run it.
        if getattr(tool, "needs_approval", False) and not intent["approved"]:
            return self._park(run, intent["key"], "approval")                             ## (5)
        # The real side effect. We hand the tool our key so that if this step runs again after a
        # crash, the tool recognizes the key and doesn't repeat the work.
        result = tool(**intent["args"], key=intent["key"])
        self._crash_maybe("after_side_effect")    # the risky moment: the call went through, but we die before saving
        if isinstance(result, Wait):              # the real work happens elsewhere — park until someone calls back
            return self._park(run, result.token, "event")                                 ## (5)
        intent.update(done=True, result=result)
        self.store.save(run)                      ## (2) checkpoint 2: save the result — the step is now finished
        self._wake(run)                           # line up the next step (we only ever do one at a time)
        return run

    def _park(self, run, token, why):
        """Put the run to sleep: note what it's waiting for, then stop. Nothing gets queued, so the
        run costs nothing until someone calls resume() with the matching token."""
        run.status, run.waiting_on = "WAITING", {"token": token, "why": why}
        return self.store.save(run)               # no wake-up queued: nothing runs until resume()

    def _finish(self, run, status, result):
        """The end of the road: DONE with an answer, or FAILED with a reason. Nothing runs after this."""
        run.status, run.result = status, result
        return self.store.save(run)

    def _crash_maybe(self, point):
        """A test hook for pretending the process dies. If this spot is armed, we raise right here:
        nothing below runs and the lease stays held, exactly the mess a real crash would leave."""
        if point in self.crash_at:
            self.crash_at.discard(point)          # fires only once, so the retry afterward can succeed
            raise Crash(f"process died {point}")

    # --- resuming a parked run: a person, a webhook, or a clock calls this -------------
    def resume(self, run_id, token, payload):
        """Wake a parked run back up. This is called from outside the loop — someone clicking
        approve, a webhook firing, a scheduled nudge. The token has to match, which is what makes it
        safe to call late or more than once: a wrong or stale token just does nothing."""
        run = self.store.get(run_id)
        if run.status != "WAITING" or run.waiting_on["token"] != token:
            return run                            # wrong token, or already resumed → do nothing
        intent = run.pending_intent()
        if run.waiting_on["why"] == "approval":   # we were waiting on a person to approve a risky tool
            if payload.get("approved"):
                intent["approved"] = True         # next time through, run exactly the call we wrote down
            else:
                # Turned down: mark the step done without running the tool, and let the run carry on.
                intent.update(done=True, result={"rejected": payload.get("reason", "")})
        else:                                     # a slow job finished; whatever it returned becomes the result
            intent.update(done=True, result=payload)
        run.status, run.waiting_on = "RUNNING", None
        self.store.save(run)                      # save the wake-up before queuing, just like every other step
        self._wake(run)
        return run


# ---------------------------------------------------------------- test doubles
class FakeModel:
    """Scripted 'LLM'. Real: Gemini via google-genai, given the journal as context."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def decide(self, journal):
        self.calls += 1
        return self.script.pop(0)


class PaymentAPI:
    """A downstream system that honours idempotency keys (Stripe-style)."""

    def __init__(self):
        self.charges = {}

    def __call__(self, amount, key):
        self.charges.setdefault(key, amount)      # same key twice = one charge
        return {"charge_id": key, "amount": amount}


class FakeClock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s
