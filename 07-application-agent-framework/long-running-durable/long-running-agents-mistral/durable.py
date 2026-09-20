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

Mistral mapping: Model → mistral_model.MistralModel (function calling via the mistralai SDK)
· the whole Store/Queue/lease/park machinery → Mistral Workflows (Temporal): see
mistral_workflow.py for the same agent written on the platform · PaymentAPI → any API
that honours an Idempotency-Key.
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
        """An intent that was journaled but not marked done = work that may or may not have happened."""
        return next((s for s in self.journal if s["type"] == "intent" and not s["done"]), None)

    def decisions(self):
        return sum(1 for s in self.journal if s["type"] == "decision")


class Store:
    """JSON on disk — durable enough to prove the point (kill the process, restart, resume).
    Real: one Firestore document per run; acquire_lease is a transaction (compare-and-set)."""

    def __init__(self, path):
        self.path = path
        self.runs = json.load(open(path)) if os.path.exists(path) else {}
        self.saves = 0

    def get(self, run_id):
        return Run(**self.runs[run_id])

    def save(self, run):                                                    ## (1)
        self.runs[run.id] = asdict(run)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.runs, f, indent=1)
        os.replace(tmp, self.path)                                          # atomic rename
        self.saves += 1
        return run

    def acquire_lease(self, run_id, owner, ttl, now):                       ## (3)
        run = self.get(run_id)
        if run.lease and run.lease["until"] > now and run.lease["owner"] != owner:
            raise LeaseHeld(f"{run_id} is held by {run.lease['owner']} until {run.lease['until']:.0f}")
        run.lease = {"owner": owner, "until": now + ttl}
        return self.save(run)

    def release_lease(self, run_id):
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
        run = self.store.save(Run(id="run_" + uuid.uuid4().hex[:6], goal=goal, max_steps=max_steps))
        self._wake(run)
        return run

    def handle(self, msg):
        """What the Cloud Run handler does with one wake-up. Return only after a durable commit."""
        return self.step(msg["run_id"], expect=tuple(msg["expect"]))

    def _wake(self, run):
        """Enqueue the next wake-up. Its NAME is the unit of work it will do, so a duplicate
        enqueue collapses; its payload says what it expects to find, so a stale delivery is ignored."""
        todo = run.pending_intent()
        expect = (len(run.journal), todo is not None)
        name = f"{run.id}:exec:{todo['key']}" if todo else f"{run.id}:decide:{len(run.journal)}"
        self.queue.enqueue(name, {"run_id": run.id, "expect": expect})

    # --- one step ---------------------------------------------------------------
    def step(self, run_id, expect=None):
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
        if run.status != "RUNNING":
            return run
        intent = run.pending_intent()
        if intent is None and expect is not None and expect != (len(run.journal), False):
            return run                            # duplicate/stale delivery: the work was already done
        if intent is None:
            intent = self._decide(run)            # journal a decision (+ intent), checkpoint 1
            if intent is None:
                return run                        # final answer or budget → run is finished
        return self._execute(run, intent)         # park / act idempotently / checkpoint 2 / wake

    def _decide(self, run):
        """Ask the model once; journal the answer BEFORE acting on it. Returns the new intent, or None."""
        if run.decisions() >= run.max_steps:                                            ## (4)
            self._finish(run, "FAILED", f"budget: {run.max_steps} steps")
            return None
        d = self.model.decide(run.goal, run.journal)
        run.journal.append({"type": "decision", **d})
        if "final" in d:
            self._finish(run, "DONE", d["final"])
            return None
        intent = {"type": "intent", "tool": d["tool"], "args": d["args"],
                  "key": f"{run.id}:{len(run.journal)}", "done": False, "approved": False}
        run.journal.append(intent)
        self.store.save(run)                      ## (2) checkpoint 1: intent BEFORE the side effect
        self._crash_maybe("after_intent")
        return intent

    def _execute(self, run, intent):
        """Carry out a journaled intent. Safe to call twice: the key makes the tool idempotent."""
        tool = self.tools[intent["tool"]]
        if getattr(tool, "needs_approval", False) and not intent["approved"]:
            return self._park(run, intent["key"], "approval")                             ## (5)
        result = tool(**intent["args"], key=intent["key"])
        self._crash_maybe("after_side_effect")
        if isinstance(result, Wait):
            return self._park(run, result.token, "event")                                 ## (5)
        intent.update(done=True, result=result)
        self.store.save(run)                      ## (2) checkpoint 2: result
        self._wake(run)
        return run

    def _park(self, run, token, why):
        run.status, run.waiting_on = "WAITING", {"token": token, "why": why}
        return self.store.save(run)               # no wake-up: nothing runs until resume()

    def _finish(self, run, status, result):
        run.status, run.result = status, result
        return self.store.save(run)

    def _crash_maybe(self, point):
        if point in self.crash_at:
            self.crash_at.discard(point)
            raise Crash(f"process died {point}")

    # --- resuming a parked run: a person, a webhook, or a clock calls this -------------
    def resume(self, run_id, token, payload):
        run = self.store.get(run_id)
        if run.status != "WAITING" or run.waiting_on["token"] != token:
            return run                            # stale, duplicate, or wrong token → no-op
        intent = run.pending_intent()
        if run.waiting_on["why"] == "approval":
            if payload.get("approved"):
                intent["approved"] = True         # the loop will execute EXACTLY the journaled call
            else:
                intent.update(done=True, result={"rejected": payload.get("reason", "")})
        else:                                     # the slow job finished; its result becomes the tool result
            intent.update(done=True, result=payload)
        run.status, run.waiting_on = "RUNNING", None
        self.store.save(run)
        self._wake(run)
        return run


# ---------------------------------------------------------------- test doubles
class FakeModel:
    """Scripted 'LLM'. Real: MistralModel in mistral_model.py (function calling over the journal)."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def decide(self, goal, journal):
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
