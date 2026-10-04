# %% [markdown]
# # 03 · State, sessions and checkpoints
#
# An agent that forgets between turns is a chatbot. An agent that cannot survive a restart is a demo.
# This notebook is about the two data structures under a production agent:
# - The **session**: an append-only event log, with a small typed state dict next to it.
# - The **durable task record**: it lets a multi-step job crash and resume, and do nothing two times.
#
# The mechanics are small. The design questions about them are not small.
#
# **Concept map:** see [docs/PRIMER_MAP.md](../docs/PRIMER_MAP.md). For more depth in this repo, see the [durable-workflows primer](../../../long-running-durable/PRIMER.md) §2–§3 (the run as a durable state machine, the five invariants).
#
# In this notebook, you will:
# 1. Read the event log of a session and the view for the model that the runtime *derives* from it. See what the model never sees.
# 2. Use compare-and-set stores and a file-backed store that survives a restart. Use the pause and resume of the Runner for irreversible tools.
# 3. Build a workflow that writes checkpoints, crashes in the middle and resumes exactly-once. Also build bounded retries, a saga and per-session locks.

# %%
import asyncio
import json
import pathlib
import tempfile
import time

from agentlab.llm import scripted, call
from agentlab.agents import (Event, InMemorySessionStore, InvocationContext, JsonFileSessionStore, LlmAgent, Runner, Session,
                           SessionStatus, SideEffect, TaskRecord, TaskStatus, TaskStore, VersionConflict, tool)

# %% [markdown]
# ## 1. The event log is the source of truth
#
# A `Session` holds these items:
# - `events`: everything that occurred, in order.
# - `state`: a small dict of current values.
# - `status`: the status of the session.
# - `pending`: an optional payload.
# - `version`: the version for optimistic concurrency.
#
# The agent loop **appends** events. It edits nothing in place. Let one turn run. Then look at what it wrote.

# %%
@tool
def get_balance(account_id: str) -> dict:
    """Balance of an account."""
    return {"account_id": account_id, "balance": 1234.5, "currency": "SGD"}


agent = LlmAgent("bank", scripted(call("get_balance", account_id="acc-1"), "Your balance is SGD 1,234.50."),
                 "You are a bank assistant.", tools=[get_balance], output_key="last_answer")

session = Session(id="s-1", tenant="t1", user="u1")
session.set_state("user:tier", "gold")                      # writes state AND logs a state event
session.append(Event(kind="user", payload={"content": "What's my balance?"}))
await agent.run_to_completion(InvocationContext(session=session))
for e in session.events:
    print(f"{e.kind:12s} agent={str(e.agent):5s} step={str(e.step):5s} {json.dumps(e.payload)[:78]}")
print("state:", session.state, "| status:", session.status.value)

# %% [markdown]
# ### What the model sees is derived, not stored
#
# `session.messages()` goes through the log. It changes only `user`, `model` and `tool_result` events into messages.
# `state`, `approval`, `delegation`, `note` and `error` events never reach the model, unless a context builder decides to inject
# them. This is intentional. The log is for operators and audits. The view of the model is a projection that you control.

# %%
messages = session.messages()
print(f"{len(session.events)} events → {len(messages)} messages")
for m in messages:
    extra = f"  tool_calls={[tc['name'] for tc in m['tool_calls']]}" if m.get("tool_calls") else ""
    print(f"  {m['role']:10s} {str(m.get('content'))[:60]!r}{extra}")
print("event kinds the model never sees:", sorted({e.kind for e in session.events} - {"user", "model", "tool_result"}))
assert not any("gold" in str(m.get("content")) for m in messages)     # the tier lives in state, not in the transcript

# %% [markdown]
# ### Scoped state keys
#
# By convention, a state key has a scope prefix:
# - `app:`: shared configuration.
# - `user:`: facts that live longer than this session.
# - `temp:`: scratch values for the current turn. The Runner calls `clear_temp_state()` after every turn.
# - no prefix: the current values of this conversation, such as an `output_key`.
#
# The scope is a *promise about lifetime*. The persistence layer must keep this promise.
# Write `user:` facts to a profile store, never into the transcript. Also note this practical point.
# A scoped key cannot be a `{placeholder}` in an instruction, because the colon is format-spec syntax.
# Thus copy it to a plain key, or inject it through a memory provider (Notebook 04).

# %%
session.set_state("app:region", "sg")
session.set_state("temp:loop_iteration", 3)
session.set_state("order_id", "ORD-10442")
print("before:", sorted(session.state))
session.clear_temp_state()
print("after clear_temp_state():", sorted(session.state))

# %% [markdown]
# ## 2. Stores: compare-and-set, and surviving a restart
#
# A store gives out **copies** of a session. It refuses a `put` whose `version` is stale.
# Two workers that both load version 1 cannot both write. The first `put` increases the store to version 2.
# The second `put` raises `VersionConflict`.
#
# A lost update becomes an explicit error that you can handle: read again and apply again.
# It does not become silent corruption.

# %%
store = InMemorySessionStore()
store.create(Session(id="shared"))
copy_a, copy_b = store.get("shared"), store.get("shared")
print("both copies start at version", copy_a.version, "and", copy_b.version)
copy_a.set_state("note", "written by A")
store.put(copy_a)
copy_b.set_state("note", "written by B")
try:
    store.put(copy_b)
except VersionConflict as e:
    print("B lost:", e)
print("stored:", store.get("shared").state, "| version", store.get("shared").version)

# %% [markdown]
# `JsonFileSessionStore` has the same semantics. It writes one JSON file for each session.
# A "restart" is a new store over the same directory. It reads the files back: status, pending approval, state and every event.

# %%
root = pathlib.Path(tempfile.mkdtemp(prefix="agentlab-sessions-"))
durable = JsonFileSessionStore(root)
s = durable.create(Session(id="durable-1", user="u1"))
s.append(Event(kind="user", payload={"content": "remember that my language is en"}))
s.set_state("user:language", "en")
durable.put(s)
print("files on disk:", [p.name for p in root.iterdir()])
del durable                                        # the process dies
after_restart = JsonFileSessionStore(root)         # a fresh process boots over the same directory
back = after_restart.get("durable-1")
print("recovered:", back.state, "| events:", [e.kind for e in back.events], "| version:", back.version)

# %% [markdown]
# ## 3. Pause and resume for irreversible actions
#
# An irreversible tool (`SideEffect.IRREVERSIBLE`) must get a confirmation. Without a `confirm` hook, the loop **pauses**.
# It saves the session with `status=awaiting_approval` and a `pending` payload that names the exact call. Then the run returns.
#
# Approval is a state-machine transition, not a modal dialog. It can arrive minutes later, from another process, after a restart.
# `approve(True)` executes the pending call **once** and lets the loop continue from the log.
# `approve(False)` records a `declined` tool result, so that the model can respond gracefully.

# %%
refunds_issued: list[tuple[str, float]] = []


@tool(side_effect=SideEffect.IRREVERSIBLE)
def issue_refund(order_id: str, amount: float) -> dict:
    """Refund an order. Irreversible."""
    refunds_issued.append((order_id, amount))
    return {"refunded": amount, "order_id": order_id}


def refund_agent(final_text: str) -> LlmAgent:
    return LlmAgent("refunds", scripted(call("issue_refund", order_id="ORD-9", amount=20.0), final_text), "Handle refunds.", tools=[issue_refund])


runner = Runner(refund_agent("Done — SGD 20 refunded on ORD-9."), store=InMemorySessionStore())
paused = await runner.run("s-approve", "Refund ORD-9, twenty dollars")
print("paused:", paused.paused, "| status:", paused.session.status.value)
print("pending:", json.dumps(paused.pending))
print("refunds executed so far:", refunds_issued)
try:
    await runner.run("s-approve", "hello?")
except RuntimeError as e:
    print("a new turn is refused while paused:", e)
resumed = await runner.approve("s-approve", True, by="supervisor@bank")
print("after approve(True):", resumed.text, "| executed:", refunds_issued, "| status:", resumed.session.status.value)
print("log tail:", [e.kind for e in resumed.session.events][-5:])
print("of those, the model sees:", [m["role"] for m in resumed.session.messages()][-3:], "— approval events stay out")

# %%
runner2 = Runner(refund_agent("Understood — no refund issued. Would you like a voucher instead?"), store=InMemorySessionStore())
await runner2.run("s-decline", "Refund ORD-9")
declined = await runner2.approve("s-decline", False)
last_tool = [e for e in declined.session.events if e.kind == "tool_result"][-1]
print("declined result fed to the model:", last_tool.payload["content"])
print("model's reply:", declined.text, "| refunds executed in total:", len(refunds_issued))

# %% [markdown]
# ## 4. Long-running work: a durable state machine
#
# A multi-step job that talks to external systems must survive when its worker stops at any line.
# The pattern is a `TaskRecord` in a `TaskStore`. The record has `completed_steps` (the steps already applied)
# and a `checkpoint` (the data that the next step needs). The worker saves the record **after every step**.
#
# On resume, the worker skips the completed steps. The next cell has a four-step claims workflow.
# The worker crashes immediately after step 2, and a new worker takes the record.

# %%
STEPS = ["validate", "fetch_policy", "reserve_payout", "notify"]
side_effects: dict[str, int] = {step: 0 for step in STEPS}


def apply_step(step: str, task: TaskRecord) -> None:
    """The real work; each call is a side effect we want exactly once."""
    side_effects[step] += 1
    task.checkpoint[step] = f"{step} done for {task.checkpoint['claim_id']}"


def run_claim(task_id: str, store: TaskStore, crash_after: str | None = None) -> TaskRecord:
    task = store.get(task_id)                                   # a worker always starts from the record
    for step in STEPS:
        if step in task.completed_steps:
            print(f"  skip {step} (already applied)")
            continue
        apply_step(step, task)
        task.completed_steps.append(step)
        task.stage = step
        store.put(task)                                         # checkpoint AFTER the step is applied
        print(f"  did  {step} → checkpoint v{task.version}")
        if step == crash_after:
            raise RuntimeError(f"worker died right after {step}")
    task.mark(TaskStatus.COMPLETED, "claim processed")
    store.put(task)
    return task


tasks = TaskStore()
claim = tasks.create(TaskRecord(kind="claim", checkpoint={"claim_id": "CLM-7"}))
print("worker 1:")
try:
    run_claim(claim.id, tasks, crash_after="fetch_policy")
except RuntimeError as e:
    print("  💥", e)
print("record after the crash:", tasks.get(claim.id).completed_steps, "| status:", tasks.get(claim.id).status.value)
print("worker 2:")
done = run_claim(claim.id, tasks)
print("status:", done.status.value, "| side effects per step:", side_effects)

# %% [markdown]
# ### The gap between the call and the checkpoint
#
# A checkpoint after each step does **not** make a step exactly-once.
# The worker can stop after the payout call returns but before it saves the record. Then the retry repeats the call.
# The solution is not a smaller gap. The solution is an **idempotency key** on the external call, made from the task and the step.
# Then the second attempt is a no-op at the receiver.

# %%
class PayoutService:
    """A downstream that honours idempotency keys, like any serious payments API."""

    def __init__(self):
        self.reservations: dict[str, dict] = {}
        self.calls = 0

    def reserve(self, claim_id: str, amount: float, idempotency_key: str) -> dict:
        self.calls += 1
        if idempotency_key not in self.reservations:
            self.reservations[idempotency_key] = {"claim_id": claim_id, "amount": amount, "id": f"res-{len(self.reservations) + 1}"}
        return self.reservations[idempotency_key]


def reserve_with_gap(store: TaskStore, task_id: str, payouts: PayoutService, stable_key: bool, crash: dict) -> None:
    task = store.get(task_id)
    if "reserve_payout" in task.completed_steps:
        return
    key = f"{task.id}:reserve_payout" if stable_key else f"attempt-{payouts.calls + 1}"    # a fresh key per attempt is no key at all
    reservation = payouts.reserve(task.checkpoint["claim_id"], 1500.0, idempotency_key=key)
    if crash.pop("now", False):
        raise RuntimeError("worker died between the payout call and the checkpoint")
    task.checkpoint["reservation"] = reservation["id"]
    task.completed_steps.append("reserve_payout")
    store.put(task)


for stable_key in (False, True):
    payouts, gap_store = PayoutService(), TaskStore()
    t = gap_store.create(TaskRecord(kind="claim", checkpoint={"claim_id": "CLM-8"}))
    crash = {"now": True}
    for attempt in (1, 2):
        try:
            reserve_with_gap(gap_store, t.id, payouts, stable_key, crash)
        except RuntimeError:
            pass
    print(f"idempotency key derived from task+step: {str(stable_key):5s} → payout API called {payouts.calls}×, "
          f"distinct reservations: {len(payouts.reservations)}")

# %% [markdown]
# ## 5. Two workers, one session
#
# Two turns on the same session race to `put`. Examples are a retry, a duplicate webhook and a double-click.
# The loser must not overwrite. It gets `VersionConflict`, reads the new copy and applies its change again on top.
# Here, the code does the retry by hand, one time. Exercise 6.2 asks you for the general, bounded version.

# %%
race_store = InMemorySessionStore()
race_store.create(Session(id="race", state={"replies": []}))


async def worker(name: str, delay: float) -> str:
    mine = race_store.get("race")
    await asyncio.sleep(delay)                                   # the model call between read and write
    mine.set_state("replies", mine.state["replies"] + [name])
    try:
        race_store.put(mine)
        return f"{name}: wrote version {mine.version}"
    except VersionConflict as e:
        fresh = race_store.get("race")                           # re-read, reapply, retry once
        fresh.set_state("replies", fresh.state["replies"] + [name])
        race_store.put(fresh)
        return f"{name}: conflict (had v{e.expected}, store at v{e.actual}) → re-read and wrote version {fresh.version}"


print(await asyncio.gather(worker("A", 0.01), worker("B", 0.02)))
print("final state:", race_store.get("race").state["replies"], "— nothing lost")

# %% [markdown]
# ## 6. Exercises
#
# ### Exercise 6.1 — a generic resume
#
# Write `resume(task_id, steps, store)`. `steps` is a list of `(name, fn)`, and each `fn(task)` does the side effect of one step.
# The function must do these things:
#
# * Load the record from the store.
# * Skip each step that is already in `completed_steps`.
# * After each applied step, append the name, set `task.stage`, and do `store.put(task)` **before** you go on.
# * When all steps are complete, mark the task `COMPLETED` and save it.
# * Let exceptions propagate. A crash is a crash, and the record is the thing that survives.

# %% exercise
def resume(task_id: str, steps: list, store: TaskStore) -> TaskRecord:
    ### BEGIN SOLUTION
    task = store.get(task_id)
    for name, fn in steps:
        if name in task.completed_steps:
            continue
        fn(task)
        task.completed_steps.append(name)
        task.stage = name
        store.put(task)
    task.mark(TaskStatus.COMPLETED, "all steps applied")
    store.put(task)
    return task
    ### END SOLUTION

# %% check
class WorkerCrashed(Exception):
    pass


effects = {"validate": 0, "fetch_policy": 0, "reserve_payout": 0, "notify": 0}
crash_before = {"reserve_payout": True}


def make_step(name):
    def fn(task):
        if crash_before.pop(name, False):
            raise WorkerCrashed(f"worker died before {name}")
        effects[name] += 1
        task.checkpoint[name] = "ok"
    return fn


steps = [(n, make_step(n)) for n in effects]
ts = TaskStore()
rec = ts.create(TaskRecord(kind="claim", checkpoint={"claim_id": "CLM-9"}))
try:
    resume(rec.id, steps, ts)
    raise AssertionError("the crash should have propagated")
except WorkerCrashed:
    pass
after_crash = ts.get(rec.id)
assert after_crash.completed_steps == ["validate", "fetch_policy"], f"checkpoint after each step: {after_crash.completed_steps}"
assert after_crash.status == TaskStatus.WORKING and after_crash.stage == "fetch_policy", (after_crash.status, after_crash.stage)
final = resume(rec.id, steps, ts)
assert final.status == TaskStatus.COMPLETED and final.completed_steps == list(effects), final.completed_steps
assert effects == {"validate": 1, "fetch_policy": 1, "reserve_payout": 1, "notify": 1}, effects
assert ts.get(rec.id).status == TaskStatus.COMPLETED, "the completed record must be saved"
print("✅ crashed before step 3, resumed from the record; every side effect happened exactly once:", effects)

# %% [markdown]
# ### Exercise 6.2 — bounded retry on conflict
#
# Write `async update_with_retry(store, session_id, mutate, attempts=5)`.
# Load the session, call `await mutate(session)`, and `put` it. On `VersionConflict`, read the session again and try again.
# Try at most `attempts` times in total, then raise `RuntimeError`. Return the saved session.
#
# The check runs ten concurrent increments. Their `mutate` awaits between the read and the write.
# An await between the read and the write is the shape of "read state, call the model, write state". The check expects all ten increments to land.

# %% exercise
async def update_with_retry(store, session_id: str, mutate, attempts: int = 5) -> Session:
    ### BEGIN SOLUTION
    for _ in range(attempts):
        session = store.get(session_id)
        await mutate(session)
        try:
            return store.put(session)
        except VersionConflict:
            continue                       # someone else won; start again from their version
    raise RuntimeError(f"could not update {session_id} after {attempts} attempts")
    ### END SOLUTION

# %% check
class CountingStore(InMemorySessionStore):
    conflicts = 0

    def put(self, session):
        try:
            return super().put(session)
        except VersionConflict:
            self.conflicts += 1
            raise


async def increment(session: Session) -> None:
    await asyncio.sleep(0)                                # the "model call" between read and write
    session.set_state("counter", session.state.get("counter", 0) + 1)


counting = CountingStore()
counting.create(Session(id="ctr"))
await asyncio.gather(*(update_with_retry(counting, "ctr", increment, attempts=25) for _ in range(10)))
final_counter = counting.get("ctr").state["counter"]
assert final_counter == 10, f"lost updates: counter={final_counter}"
assert counting.conflicts >= 9, f"expected real contention, saw {counting.conflicts} conflicts"


class AlwaysConflicts(InMemorySessionStore):
    attempts = 0

    def put(self, session):
        self.attempts += 1
        raise VersionConflict(session.id, session.version, session.version + 1)


stubborn = AlwaysConflicts()
stubborn.create(Session(id="x"))
gave_up = False
try:
    await update_with_retry(stubborn, "x", increment, attempts=3)
except RuntimeError:
    gave_up = True
assert gave_up, "must raise RuntimeError after the attempt limit"
assert stubborn.attempts == 3, f"expected exactly 3 attempts, saw {stubborn.attempts}"
print(f"✅ 10 concurrent increments → counter=10 with {counting.conflicts} conflicts retried; gives up after the attempt limit")

# %% [markdown]
# ### Exercise 6.3 — compensation when there is no undo button
#
# A database cannot roll back some steps. Examples are a payout reserved with a partner, and an email put in a queue.
# The saga pattern records a **compensating action** for every step that succeeded.
# When a later step fails, it runs these actions in **reverse order**. Complete `Saga`:
#
# * `add(name, action, compensate)` registers a step (`action` and `compensate` are `async` no-argument callables).
# * `await run()` executes the actions in order. It appends each name to `self.completed` after the action succeeds, and returns `self.completed`.
# * If an action raises, run the compensations of the completed steps, newest first. Append their names to `self.compensated`.
#   Then raise the original exception again.

# %% exercise
class Saga:
    def __init__(self):
        self.steps: list[tuple] = []
        self.completed: list[str] = []
        self.compensated: list[str] = []

    def add(self, name: str, action, compensate) -> "Saga":
        self.steps.append((name, action, compensate))
        return self

    ### BEGIN SOLUTION
    async def run(self) -> list[str]:
        for name, action, _compensate in self.steps:
            try:
                await action()
            except Exception:
                await self._rollback()
                raise
            self.completed.append(name)
        return self.completed

    async def _rollback(self) -> None:
        done = {name: compensate for name, _action, compensate in self.steps if name in self.completed}
        for name in reversed(self.completed):
            await done[name]()
            self.compensated.append(name)
    ### END SOLUTION

# %% check
trail: list[str] = []


def make_saga_step(name: str, fail: bool = False):
    async def action():
        if fail:
            trail.append(f"do:{name}:FAIL")
            raise IOError(f"{name} failed")
        trail.append(f"do:{name}")

    async def compensate():
        trail.append(f"undo:{name}")
    return action, compensate


saga = Saga()
for name, fail in (("reserve_payout", False), ("queue_email", False), ("post_ledger", True), ("notify", False)):
    saga.add(name, *make_saga_step(name, fail))
reraised = False
try:
    await saga.run()
except IOError:
    reraised = True
assert reraised, "the failing step's exception must be re-raised after compensating"
assert saga.completed == ["reserve_payout", "queue_email"], saga.completed
assert saga.compensated == ["queue_email", "reserve_payout"], f"compensate newest-first: {saga.compensated}"
assert trail == ["do:reserve_payout", "do:queue_email", "do:post_ledger:FAIL", "undo:queue_email", "undo:reserve_payout"], trail
failure_trail = list(trail)
happy = Saga().add("a", *make_saga_step("a")).add("b", *make_saga_step("b"))
assert await happy.run() == ["a", "b"] and happy.compensated == [], (happy.completed, happy.compensated)
print("✅ saga trail on failure:", " → ".join(failure_trail))

# %% [markdown]
# ### Exercise 6.4 — serialise turns per session, not globally
#
# Optimistic concurrency is the safety net. The usual solution is different: never let two turns of the **same** session run at the same time.
# Turns on **different** sessions can overlap freely.
#
# Write `SessionLock` with `for_session(session_id) -> asyncio.Lock`.
# It returns the same lock object for the same id. It creates the lock on first use.
# Use it like this: `async with locks.for_session(sid): ...`.

# %% exercise
class SessionLock:
    ### BEGIN SOLUTION
    def __init__(self):
        self._locks: dict[str, asyncio.Lock] = {}

    def for_session(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = asyncio.Lock()
        return lock
    ### END SOLUTION

# %% check
locks = SessionLock()
assert locks.for_session("s1") is locks.for_session("s1"), "same session → same lock object"
assert locks.for_session("s1") is not locks.for_session("s2"), "different sessions → different locks"


async def turn(session_id: str) -> None:
    async with locks.for_session(session_id):
        await asyncio.sleep(0.05)                     # a model call


t0 = time.perf_counter()
await asyncio.gather(turn("s1"), turn("s1"))
same = time.perf_counter() - t0
t0 = time.perf_counter()
await asyncio.gather(turn("s1"), turn("s2"))
different = time.perf_counter() - t0
assert same >= 0.09, f"two turns on one session must serialise (took {same:.3f}s)"
assert different < 0.09, f"turns on different sessions must overlap (took {different:.3f}s)"
print(f"✅ same session: {same * 1000:.0f} ms (serialised) | different sessions: {different * 1000:.0f} ms (parallel)")

# %% [markdown]
# ### Exercise 6.5 — what does the model see each turn?
#
# Derive the view of the model from the log again, by hand. Write `derive_messages(events)`:
#
# * `user` becomes `{"role": "user", "content": ...}`.
# * `model` becomes `{"role": "assistant", "content": ...}`. Add `"tool_calls"` **only** when the payload has them.
# * `tool_result` becomes `{"role": "tool", "name": ..., "tool_call_id": <payload id>, "content": ...}`.
# * Skip every other kind (`state`, `approval_required`, `approval`, `delegation`, `note`, `error`, `final`).
#
# The check builds a six-turn session that has approvals, state changes and notes in it.
# Then it compares your output with `session.messages()`.

# %% exercise
def derive_messages(events: list) -> list[dict]:
    ### BEGIN SOLUTION
    out: list[dict] = []
    for ev in events:
        if ev.kind == "user":
            out.append({"role": "user", "content": ev.payload.get("content", "")})
        elif ev.kind == "model":
            m = {"role": "assistant", "content": ev.payload.get("content", "") or ""}
            if ev.payload.get("tool_calls"):
                m["tool_calls"] = ev.payload["tool_calls"]
            out.append(m)
        elif ev.kind == "tool_result":
            out.append({"role": "tool", "name": ev.payload.get("name"), "tool_call_id": ev.payload.get("id"),
                        "content": ev.payload.get("content", "")})
    return out
    ### END SOLUTION

# %% check
six = Session(id="six")
six.set_state("user:tier", "gold")
for i in range(1, 7):
    six.append(Event(kind="user", payload={"content": f"turn {i}: what about ORD-{i}?"}))
    if i % 2 == 0:
        six.append(Event(kind="model", payload={"content": "", "tool_calls": [{"id": f"c{i}", "name": "get_order", "args": {"order_id": f"ORD-{i}"}}]}))
        six.append(Event(kind="approval_required", payload={"id": f"c{i}", "name": "get_order"}))
        six.append(Event(kind="approval", payload={"id": f"c{i}", "approved": True, "by": "SECRET-approver@bank"}))
        six.append(Event(kind="tool_result", payload={"id": f"c{i}", "name": "get_order", "ok": True, "content": '{"ok":true,"data":{"status":"shipped"}}'}))
    six.set_state("temp:scratch", i)
    six.append(Event(kind="note", payload={"internal": "compaction summary v1"}))
    six.append(Event(kind="model", payload={"content": f"answer {i}"}))
    six.append(Event(kind="final", payload={"text": f"answer {i}"}))
mine = derive_messages(six.events)
assert mine == six.messages(), "your derivation differs from session.messages()"
assert len(mine) == 6 * 2 + 3 * 2, len(mine)
flat = json.dumps(mine)
for leaked in ("gold", "SECRET", "compaction", "scratch"):
    assert leaked not in flat, f"{leaked!r} leaked into the model's context"
assert [m["role"] for m in mine][:5] == ["user", "assistant", "user", "assistant", "tool"], [m["role"] for m in mine][:5]
print(f"✅ {len(six.events)} events → {len(mine)} messages; state, approvals and notes stay out of the model's context")

# %% [markdown]
# ### Exercise 6.6 — the three questions
#
# In `long_running_design_questions`, put the three questions to ask about *any* long-running agent design:
# - Where does it write checkpoints?
# - What is the idempotency key on each external write?
# - What occurs if the worker stops between the payment call and the record update?
#
# Write them as questions that you can ask a candidate, or that a person can ask you.

# %% exercise
### BEGIN SOLUTION
long_running_design_questions = [
    "Where does the workflow checkpoint, and is the record saved after every step that has an external effect?",
    "What is the idempotency key on each external write, and does the downstream system actually honour it?",
    "What happens if the worker dies between the payment call returning and the record update — who retries, and how do we know the first call landed?",
]
### END SOLUTION

# %% check
qs = long_running_design_questions
assert isinstance(qs, list) and len(qs) == 3, "exactly three questions"
assert all(isinstance(q, str) and q.strip().endswith("?") and len(q.split()) >= 8 for q in qs), "each must be a real question"
_all = " ".join(qs).lower()
assert "checkpoint" in _all, "one question is about checkpointing"
assert "idempoten" in _all, "one question is about idempotency keys"
assert any(w in _all for w in ("dies", "die", "crash", "between")), "one question is about the worker dying mid-step"
print("✅ questions:", *qs, sep="\n   ")

# %% [markdown]
# ## The one-minute version
#
# When the design has a "workflow" box in it, say what is under it:
#
# *"Each session is an append-only event log with a small state dict. The runtime derives the context of the model from the log. Thus approvals, state and notes never leak into prompts.
# Writes go through compare-and-set on a version. A turn holds a per-session lock, thus retries cannot interleave.*
#
# *"Anything irreversible pauses the loop with `awaiting_approval` and a pending payload.
# Approval is a state transition that can arrive from another process after a restart.
# A long-running job is a task record. The worker saves its `completed_steps` after every step.*
#
# *"Every external write has an idempotency key made from the task and the step. The key is necessary
# because one day the worker will stop between the payment call and the checkpoint. A saga of compensations covers the steps that
# nobody can undo."*
#
# Then ask the important question: *what happens if the worker dies right here?* Then point at the line.
