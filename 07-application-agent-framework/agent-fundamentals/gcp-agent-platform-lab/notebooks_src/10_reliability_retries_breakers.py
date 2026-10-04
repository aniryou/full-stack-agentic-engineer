# %% [markdown]
# # 10 · Reliability: retries, breakers, bulkheads, deadlines, fallbacks
#
# Each dependency of an agent fails sometimes: the model, each tool, the session store. The difference
# between a demo and a product is what occurs *next*. This notebook shows the standard mechanisms in a
# concrete form, with fake clocks and injected sleeps. Thus nothing waits, and you can see each state
# transition.
#
# **Concept map:** see [docs/PRIMER_MAP.md](../docs/PRIMER_MAP.md). For more depth in this repo, see the [scaling primer](../../../../06-gateway/scaling-admission-cost/agentic-scaling-lab/docs/01-scaling-primer.md) §5.2 (retries, jitter, breakers, fallbacks, hedging).
#
# In this notebook, you do these steps:
# 1. Watch a simple retry refund a customer two times. Then repair it with an idempotency key.
# 2. Move a circuit breaker through the states closed, open, half-open and closed again, with a fake clock.
# 3. Put deadlines, bulkheads and a fallback chain together. When you do this, a dead dependency gives a degraded answer, not a dead agent.

# %%
import asyncio
import random

from agentlab.agents import Event, InvocationContext, LlmAgent, Session, ToolContext, ToolTransientError, tool
from agentlab.agents.state import TaskStore
from agentlab.llm import FakeLLM, call, scripted
from agentlab.reliability import (Bulkhead, BulkheadFull, CircuitBreaker, CircuitOpen, Deadline,
                                DeadlineExceeded, FallbackChain, GracefulTool, IdempotentCall, PermanentError,
                                RetryableError, RetryPolicy, backoff_schedule, classify_http, is_retryable,
                                lab_fallback_chain, per_hop_budget, retry)


class FakeClock:
    """Time you control. Every breaker and deadline below reads this instead of the wall clock."""
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


async def instant_sleep(seconds):
    """Stands in for asyncio.sleep: the schedule is computed, nothing waits."""
    return None

# %% [markdown]
# ## 1. The flaky payments stub — and why a retry needs a key
#
# The dangerous failure is not "the call failed". It is "the call *succeeded* and the response was lost".
# From the side of the client, both look like a timeout. `LossyTransport` makes exactly that failure: it
# applies the operation, and then the reply disappears.

# %%
class PaymentsStub:
    def __init__(self):
        self.ledger = []                                  # every refund that actually happened
    async def refund(self, order_id: str, amount: float) -> dict:
        self.ledger.append((order_id, amount))
        return {"refunded": amount, "order_id": order_id}


class LossyTransport:
    """Applies the operation, then loses the first `lose_first` responses."""
    def __init__(self, lose_first=1):
        self.lose_first, self.lost = lose_first, 0
    async def send(self, fn):
        result = await fn()                               # the write lands here...
        if self.lost < self.lose_first:
            self.lost += 1
            raise asyncio.TimeoutError("response lost")   # ...and the caller never learns it
        return result


payments, network = PaymentsStub(), LossyTransport(lose_first=1)
await retry(lambda: network.send(lambda: payments.refund("O-1", 20.0)), RetryPolicy(jitter_s=0), sleep=instant_sleep)
print("naive retry → ledger:", payments.ledger)

# %% [markdown]
# There are two refunds for one request. The retry did exactly what the code told it to do. The *contract*
# was not there. `IdempotentCall` is the server-side half of that contract. The same key returns the recorded
# outcome, and it does not apply the write again.

# %%
payments, network = PaymentsStub(), LossyTransport(lose_first=1)
once = IdempotentCall()                                   # lives with the payments service, keyed by the caller
key = "refund:O-1:session-42:step-3"                      # the agent loop derives keys like this (see the agent-loop notebook)
result = await retry(lambda: network.send(lambda: once(key, lambda: payments.refund("O-1", 20.0))),
                     RetryPolicy(jitter_s=0), sleep=instant_sleep)
print("keyed retry  → ledger:", payments.ledger, "| result:", result, "| applied:", once.applied, "replayed:", once.replayed)

# %% [markdown]
# ## 2. The backoff schedule
#
# Retries must spread out in time. They must not collect at one time. Each wait:
# * doubles, so a dependency with problems gets time to recover,
# * has a cap, so nobody waits forever,
# * has jitter, so a thousand clients do not retry in the same millisecond.
#
# `backoff_schedule` is a pure function. This is why you can test it and print it.

# %%
def show_schedule(policy, rng_seed=0):
    delays = backoff_schedule(policy, random.Random(rng_seed))
    t = 0.0
    for i, d in enumerate(delays, start=1):
        t += d
        print(f"  attempt {i} fails → wait {d:5.2f}s {'▇' * int(d * 10):<40} (t={t:5.2f}s)")
    print(f"  attempt {len(delays) + 1} is the last; total wait {t:.2f}s")

print("default policy:")
show_schedule(RetryPolicy())
print("\nsix attempts, cap 4 s, no jitter:")
show_schedule(RetryPolicy(max_attempts=6, cap_s=4.0, jitter_s=0.0))

# %% [markdown]
# ### Exercise 2.1 — implement the schedule
#
# Write `my_backoff_schedule(policy, rng)`. It returns the delay before each retry (thus `max_attempts - 1` values):
#
# $$
# \min(\mathrm{cap\_s}, \mathrm{base\_s} \cdot 2^i) + \text{rng.uniform}(0, \mathrm{jitter\_s})
# $$
#
# for the attempt index $i = 0, 1, \ldots$.
# Get the jitter from `rng.uniform` one time for each delay, in order. If you do this, the result agrees with the library for the same seed.

# %% exercise
def my_backoff_schedule(policy: RetryPolicy, rng: random.Random) -> list[float]:
    ### BEGIN SOLUTION
    delays = []
    for i in range(policy.max_attempts - 1):
        exponential = min(policy.cap_s, policy.base_s * (2 ** i))
        delays.append(exponential + rng.uniform(0.0, policy.jitter_s))
    return delays
    ### END SOLUTION

# %% check
for pol in (RetryPolicy(), RetryPolicy(max_attempts=7, base_s=0.25, cap_s=2.0, jitter_s=0.1), RetryPolicy(max_attempts=1)):
    mine, ref = my_backoff_schedule(pol, random.Random(7)), backoff_schedule(pol, random.Random(7))
    assert [round(x, 9) for x in mine] == [round(x, 9) for x in ref], f"{pol}: {mine} != {ref}"
assert my_backoff_schedule(RetryPolicy(max_attempts=7, jitter_s=0.0), random.Random(0)) == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
print("✅ schedule matches: capped, exponential, jittered")

# %% [markdown]
# ## 3. Classify before you retry
#
# A retry is only useful when a later attempt can see a different answer. `is_retryable` decides by type
# (`RetryableError`, timeouts, `ToolTransientError`) or by HTTP status. It treats all unknown errors as bugs.

# %%
for exc in (RetryableError("503", status=503), asyncio.TimeoutError(), ToolTransientError("rate limited"),
            PermanentError("404", status=404), KeyError("bug in my code")):
    print(f"  {type(exc).__name__:20s} retryable={is_retryable(exc)}")

# %% [markdown]
# ### Exercise 3.1 — implement `classify_http`
#
# Return `True` for the statuses that are worth a retry: **408, 429, 502, 503, 504**. Each other 4xx is the
# fault of the caller, and it is permanent. Treat 500 as permanent too. It is usually deterministic, and a
# retry of deterministic failures at scale is how a short problem becomes an outage.

# %% exercise
def my_classify_http(status: int) -> bool:
    ### BEGIN SOLUTION
    return status in {408, 429, 502, 503, 504}
    ### END SOLUTION

# %% check
for status in (200, 400, 401, 403, 404, 408, 409, 422, 429, 500, 501, 502, 503, 504):
    assert my_classify_http(status) == classify_http(status), f"status {status}"
print("✅ classification agrees with the library")

# %% [markdown]
# ## 4. The circuit breaker
#
# A retry against a dependency that is *down* makes the problem worse. Each client adds load, and each
# request waits for a timeout. A breaker counts consecutive failures. Then it **opens**: it fails fast and
# makes no call. After a recovery period, it lets **one probe** through, to decide if it closes again.

# %%
clock = FakeClock()
breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_s=30.0, half_open_max_calls=1, clock=clock, name="legacy-core")
legacy_up = False

async def legacy_core():
    if not legacy_up:
        raise RetryableError("503 legacy core unavailable")
    return "balance=1234.50"

async def attempt(label):
    try:
        out = await breaker.call(legacy_core)
        print(f"  t={clock.now:5.1f}s {label:28s} → ok: {out:18s} state={breaker.state.value}")
    except CircuitOpen as e:
        print(f"  t={clock.now:5.1f}s {label:28s} → REJECTED ({e.retry_after_s:.0f}s left)  state={breaker.state.value}")
    except RetryableError as e:
        print(f"  t={clock.now:5.1f}s {label:28s} → failed: {e}   state={breaker.state.value}")

for i in range(3):
    await attempt(f"call {i + 1}")
await attempt("call 4 (fast fail)")
clock.advance(30)
await attempt("call 5 (probe, still down)")
clock.advance(30)
legacy_up = True
await attempt("call 6 (probe, recovered)")
await attempt("call 7")
print("metrics:", breaker.metrics)

# %% [markdown]
# Note what the breaker gave you. Call 4 cost nothing, and it returned immediately with a *retry-after*.
# Also, the dependency got exactly one request in each recovery period while it was not healthy.
#
# ### Exercise 4.1 — implement the half-open transition
#
# The next cell contains `MiniBreaker`. The code that keeps its state is already there. Implement `allow()`:
#
# * **closed**: permit the call.
# * **open**: if `recovery_timeout_s` has not passed since `opened_at`, refuse the call. If it has passed, move to **half_open**.
# * **half_open**: permit exactly one probe at a time (`probe_in_flight`). Refuse a call while a probe runs.

# %% exercise
class MiniBreaker:
    def __init__(self, failure_threshold: int, recovery_timeout_s: float, clock):
        self.failure_threshold, self.recovery_timeout_s, self.clock = failure_threshold, recovery_timeout_s, clock
        self.state, self.failures, self.opened_at, self.probe_in_flight = "closed", 0, None, False

    def allow(self) -> bool:
        """May a call proceed right now? Applies the time-based open → half_open transition."""
        ### BEGIN SOLUTION
        if self.state == "open":
            if self.clock() - self.opened_at < self.recovery_timeout_s:
                return False
            self.state, self.probe_in_flight = "half_open", False
        if self.state == "half_open":
            if self.probe_in_flight:
                return False
            self.probe_in_flight = True
        return True
        ### END SOLUTION

    def record(self, ok: bool) -> None:
        self.probe_in_flight = False
        if ok:
            self.failures, self.state = 0, "closed"
            return
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state, self.opened_at, self.failures = "open", self.clock(), 0

# %% check
c = FakeClock()
mb = MiniBreaker(failure_threshold=2, recovery_timeout_s=10.0, clock=c)
assert mb.allow() and mb.state == "closed"
mb.record(False); mb.record(False)
assert mb.state == "open" and not mb.allow(), "open circuit must refuse"
c.advance(9.9); assert not mb.allow(), "too early"
c.advance(0.1); assert mb.allow() and mb.state == "half_open", "timeout elapsed → half-open, one probe allowed"
assert not mb.allow(), "a second call during the probe must be refused"
mb.record(False); assert mb.state == "open", "failed probe re-opens"
c.advance(10.0); assert mb.allow(); mb.record(True)
assert mb.state == "closed" and mb.allow(), "successful probe closes"
print("✅ half-open transition behaves")

# %% [markdown]
# ## 5. A bulkhead in front of a slow legacy system
#
# A breaker reacts to *failures*. A bulkhead reacts to *load*. It puts a cap on concurrent calls (and on a
# short queue). Thus one slow system cannot block every worker. The bulkhead tells the other callers
# "busy" immediately. They do not go into a queue that only grows.

# %%
legacy = Bulkhead(max_concurrent=2, max_queue=2, name="legacy-core")

async def legacy_lookup(i):
    await asyncio.sleep(0.02)                              # the legacy system is slow
    return f"row-{i}"

outcomes = await asyncio.gather(*(legacy.run(legacy_lookup, i) for i in range(8)), return_exceptions=True)
served = [o for o in outcomes if isinstance(o, str)]
shed = [o for o in outcomes if isinstance(o, BulkheadFull)]
print(f"served {len(served)} (2 at a time + 2 queued), shed {len(shed)} immediately; peak in flight = {legacy.peak_active}")

# %% [markdown]
# ## 6. Deadlines compose into a turn budget
#
# A timeout is a duration. A **deadline** is a point in time. If you promised the user an answer within 8 s,
# each hop gets a *child* deadline. A child deadline can never end after the parent. Also, you keep time for
# the answer itself. If you do not, the turn uses all its time on tools and times out silently.

# %%
clock = FakeClock()
turn = Deadline(8.0, clock=clock)
hop_s = per_hop_budget(turn.seconds, hops=3, reserve_s=2.0)   # 2 s per hop, 2 s kept for the final answer
print(f"turn budget {turn.seconds}s → {hop_s}s per hop with 2.0s reserved\n")

def simulated_hop(name, needs_s, deadline):
    """Advance the fake clock as if the hop ran; cut it off when its deadline passes."""
    if needs_s <= deadline.remaining():
        clock.advance(needs_s)
        return f"{name}: done in {needs_s}s"
    clock.advance(deadline.remaining())
    raise DeadlineExceeded(f"{name}: needed {needs_s}s, had {deadline.seconds:.1f}s")

for name, needs in (("plan", 1.5), ("legacy lookup", 3.5), ("policy lookup", 1.0)):
    hop = turn.sub(hop_s)                                  # min(per-hop budget, what is left of the turn)
    started = clock.now
    try:
        outcome = simulated_hop(name, needs, hop)
    except DeadlineExceeded as e:
        outcome = f"CUT — {e}"
    print(f"  {started:4.1f}s → {clock.now:4.1f}s  {outcome}")
print(f"  at {clock.now:4.1f}s the answer still has {turn.remaining():.1f}s (the reserve was never touched)")

# %% [markdown]
# ## 7. The fallback chain: degrade, don't disappear
#
# When nothing can serve the request *now*, the honest answer is "I've queued this and will follow up". That
# answer has a durable task behind it, not a promise. The chain tries each step in order. When the primary
# did not serve the result, the chain puts the label `degraded` on it. Then the UI can say so, and a
# dashboard can count it.

# %%
class DownLLM(FakeLLM):
    async def generate(self, messages, tools=None, **o):
        raise RetryableError("503 model overloaded")

tasks = TaskStore()
chain = lab_fallback_chain(primary=DownLLM(), smaller=DownLLM(), cache={"What is the card fee?": "SGD 5 per year."}, tasks=tasks)
for question in ("What is the card fee?", "Close my savings account"):
    r = await chain.run(question)
    print(f"{question:28s} → served by {r.step:17s} degraded={r.degraded}  value={r.value if isinstance(r.value, str) else r.value['message']}")
task_id = r.value["task_id"]
print("\ndurable task:", tasks.get(task_id).kind, tasks.get(task_id).status.value, tasks.get(task_id).checkpoint)
print("degraded share so far:", f"{chain.degraded_share():.0%}", "| failures per step:", dict(chain.failed))

# %% [markdown]
# ### Exercise 7.1 — the fallback rule
#
# Some errors must not fall through. A smaller model cannot repair a *malformed request*. Also, a silent
# answer to a broken request from a lower-cost model hides a bug. Implement `should_fall_through(exc)`:
#
# * `CircuitOpen` and `BulkheadFull`: fall through (the step is unavailable, not incorrect).
# * anything for which `is_retryable` is true (transient, timeout): fall through.
# * everything else (for example `PermanentError`, or a `ValueError` from bad input): **do not** fall through.

# %% exercise
def should_fall_through(exc: BaseException) -> bool:
    ### BEGIN SOLUTION
    if isinstance(exc, (CircuitOpen, BulkheadFull)):
        return True
    return is_retryable(exc)
    ### END SOLUTION

# %% check
async def failing(exc):
    async def step(request):
        raise exc
    return step

async def small(request):
    return "small model answer"

for exc, expect_fallthrough in ((RetryableError("503"), True), (asyncio.TimeoutError(), True), (CircuitOpen("primary", 12.0), True),
                                (PermanentError("invalid request", status=400), False), (ValueError("bad input"), False)):
    ch = FallbackChain([("primary", await failing(exc)), ("smaller", small)], fallback_on=should_fall_through)
    try:
        res = await ch.run("q")
        assert expect_fallthrough, f"{type(exc).__name__} should have been raised, not served by {res.step}"
        assert res.step == "smaller" and res.degraded
    except type(exc):
        assert not expect_fallthrough, f"{type(exc).__name__} should have fallen through"
print("✅ transient and unavailable fall through; wrong requests surface")

# %% [markdown]
# ## 8. A GracefulTool inside an agent: the model is told the tool is down
#
# The last part puts the mechanisms of sections 1 to 7 around an agentlab tool. `GracefulTool` retries transient
# failures and sends each attempt through a breaker. When the circuit opens, it returns a **structured**
# `unavailable` result. The hint in that result tells the model what to do. Thus there is no stack trace, no
# hang, and no model that continues to call a dead tool.

# %%
legacy_calls = 0

@tool
def legacy_balance(account_id: str) -> dict:
    """Balance from the legacy core (currently down)."""
    global legacy_calls
    legacy_calls += 1
    raise ToolTransientError("503 legacy core unavailable")

clock = FakeClock()
graceful = GracefulTool(legacy_balance, CircuitBreaker(failure_threshold=2, recovery_timeout_s=60.0, clock=clock),
                        RetryPolicy(max_attempts=4, jitter_s=0.0), sleep=instant_sleep)
res = await graceful.run({"account_id": "acc-1"}, ToolContext())
print("what the model receives:", res.to_content())
print("attempts against the dependency:", legacy_calls, "| breaker:", graceful.breaker.state.value)

# %% [markdown]
# There were two attempts, not four. The breaker opened after the second failure, and `CircuitOpen` is not
# retryable. Thus the retry loop stopped. The breaker answers each later call in this minute, and it does not
# touch the dependency.
#
# ### Exercise 8.1 — wire it into an `LlmAgent`
#
# Implement `build_resilient_agent(llm, legacy_tool, clock)`. It returns an `LlmAgent` with the name
# `"assistant"`. Its only tool is `legacy_tool` in a `GracefulTool` with these settings:
# * a breaker with `failure_threshold=2, recovery_timeout_s=60` on `clock`,
# * `RetryPolicy(max_attempts=3, jitter_s=0.0)`,
# * `sleep=instant_sleep`.
#
# The check makes sure that the *model saw the "unavailable" hint*. That is the whole point of the pattern.

# %% exercise
def build_resilient_agent(llm, legacy_tool, clock) -> LlmAgent:
    ### BEGIN SOLUTION
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=60.0, clock=clock, name=legacy_tool.spec.name)
    protected = GracefulTool(legacy_tool, breaker, RetryPolicy(max_attempts=3, jitter_s=0.0), sleep=instant_sleep)
    return LlmAgent("assistant", llm, "You are a bank assistant. If a tool is unavailable, say so and offer to follow up.",
                    tools=[protected])
    ### END SOLUTION

# %% check
legacy_calls = 0
llm = scripted(call("legacy_balance", account_id="acc-1"), "The balance system is unavailable right now; I'll follow up when it recovers.")
agent = build_resilient_agent(llm, legacy_balance, FakeClock())
assert isinstance(agent, LlmAgent) and agent.name == "assistant"
tool_obj = agent.registry.get("legacy_balance")
assert isinstance(tool_obj, GracefulTool), "the tool must be wrapped in a GracefulTool"
s = Session(id="resilient")
s.append(Event(kind="user", payload={"content": "what is my balance?"}))
await agent.run_to_completion(InvocationContext(session=s))
tool_msg = next(m for m in llm.calls[1]["messages"] if m["role"] == "tool")
assert '"error":"unavailable"' in tool_msg["content"], tool_msg["content"]
assert "Do not call it again this turn" in tool_msg["content"], "the hint must reach the model"
assert legacy_calls == 2, f"breaker should stop retries after 2 attempts, saw {legacy_calls}"
assert s.last_final_text().startswith("The balance system is unavailable")
print("✅ the model was told the tool is down and answered honestly")

# %% [markdown]
# ## The one-minute version
#
# When someone asks *"what happens when a tool is down?"*, go through the layers in order. Name the number
# that each layer owns:
#
# * **classify**: retry 429/5xx/timeouts, never 4xx. Unknown errors are bugs.
# * **retry with capped exponential backoff + jitter**. For writes, retry only with an idempotency key. The
#   timeout-after-success case is the one that costs money.
# * **circuit breaker** for each dependency: fail fast, send one probe in each recovery period, and alert on
#   each open.
# * **bulkhead** for each slow dependency: bounded concurrency *and* a bounded queue. Refuse the rest.
# * **deadlines** that go down the call chain, with time kept for the answer.
# * **fallback chain** that puts a label on degraded answers and ends in a durable task, never in silence.
# * and, at the tool boundary, a structured `unavailable` result with a hint. Then the *model* can tell the
#   user the truth.
#
# The key sentence: *"Reliability is a property of the harness, not the model — the model only sees structured results."*
