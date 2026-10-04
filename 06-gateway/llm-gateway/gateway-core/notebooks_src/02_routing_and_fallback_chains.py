# %% [markdown]
# # 02 · Routing and fallback chains
#
# **Tier:** T0. It uses only the CPU, no network and a few seconds. A virtual clock simulates every latency. The same
# outages over HTTP, and a real vLLM that we stop in the middle of a run (T1), are `gateway-lab` notebook
# `02_outages_fallbacks_and_breakers`.
#
# ## The one-minute version
# Clients ask for an **alias**. The gateway resolves the alias to an ordered **chain** of (provider, model, region)
# targets. Then it filters the chain by what the request needs (tools, reasoning effort, context length, residency). It puts
# the chain in order by a policy, and it walks the chain.
#
# A request **falls through** only on a failure that another target can solve: 429, 5xx, a timeout, a context that is
# too long. It never falls through on a bad request, bad credentials or a policy refusal, and **never after the first
# byte** reached the client. A **breaker per target** changes a dead provider into an instant skip, not into one
# timeout for each request. What its targets share puts a cap on the availability of a chain. A slow failure (a
# timeout) costs time for every request behind it.
#
# By the end, you can do these things:
#
# - classify failures,
# - calculate the availability, the expected latency and the expected cost of a chain,
# - predict how many requests a breaker lets reach a dead target when the requests overlap,
# - apply the first-byte rule,
# - select a first-byte deadline from a latency budget.
#
# Primer: §2 *Model routing and fallback chains* (`../PRIMER.md`). Retries and full jitter are in §5.2 of the scaling
# primer. The full state machine of the breaker is in notebook 10 of the 07.2 lab.

# %%
import random

from gwcore import api, keys, routing
from gwcore.gateway import Gateway
from gwcore.metering import price_call
from gwcore.providers import CATALOGUE, Clock, FakeProvider
from gwcore.routing import Router, Target

chain = [Target("google", "gemini-3.5-flash"), Target("openai", "gpt-5.4-mini"), Target("anthropic", "claude-haiku-4-5"),
         Target("self", "lab/llm", "us-central1")]
router = Router({"chat": chain, "chat@gold": [Target("openai", "gpt-5.4-mini"), Target("google", "gemini-3.5-flash")]})
show = lambda ts: [f"{t.provider}/{t.model}" for t in ts]
print("plain chat, 800-token prompt :", show(router.candidates("chat", {}, now=0, prompt_tokens=800)))
print("with tools                   :", show(router.candidates("chat", {"tools": [{}]}, now=0, prompt_tokens=800)))
print("reasoning_effort=high        :", show(router.candidates("chat", {"reasoning_effort": "high"}, now=0)))
print("a 280,000-token prompt       :", show(router.candidates("chat", {}, now=0, prompt_tokens=280_000)))
print("residency us-central1 only   :", show(router.candidates("chat", {}, now=0, regions={"us-central1"})))
print("tier gold                    :", show(router.candidates("chat", {}, now=0, tier="gold")))
print("policy cheapest (800 in/500) :", show(router.candidates("chat", {"max_completion_tokens": 500}, now=0,
                                                               prompt_tokens=800, policy="cheapest")))

# %% [markdown]
# Read the filters:
#
# - The self-hosted `lab/llm` has no tool calling configured and a 4,096-token window. Thus tools and long prompts
#   drop it.
# - Only thinking-capable targets take high effort (§7 "routing by effort" of the 00.5 primer).
# - At 280,000 tokens (plus the default 1,024-token output reservation), only the million-token model stays.
#   claude-haiku-4-5 holds 200,000 and gpt-5.4-mini holds 272,000 (context windows dated, verify).
#
# `cheapest` puts the self-hosted pool first, because its marginal price per token is zero while it has headroom. Its
# real cost is the GPU bill. Notebook 04 shows the chargeback of that bill.
#
# ## Worked example 1 — an outage, with and without a breaker
# The primary hangs for five minutes (it gives no answer at all). The timeout of the gateway is 10 s. One request
# arrives every 12 s for eight minutes.
#
# The in-process gateway handles one request at a time on the virtual clock. Thus the gap is longer than a timeout on
# purpose: no request waits behind another. Also, the measurement of each time to first token starts at the arrival of
# its request (the loop checks it). Exercise 2.4 has requests that overlap, which is the production case.

# %%
def outage_run(threshold, n=40, gap=12.0):
    clock = Clock()
    provs = {"google": FakeProvider("google", clock, outages=[(0, 300, None)]), "openai": FakeProvider("openai", clock)}
    ks = keys.KeyStore(seed=1)
    k = ks.issue("acme")
    gw = Gateway(keys=ks, router=Router({"chat": chain[:2]}, threshold=threshold, cooldown=30), providers=provs,
                 clock=clock, timeout=10.0)
    ttft, timed_out = [], 0
    for i in range(n):
        assert clock.t <= i * gap, "a request would have queued behind the previous one"
        clock.t = i * gap                                    # the request arrives
        r = gw.handle(k, api.chat_request("chat", [{"role": "user", "content": "hi"}], stream=True))
        ttft.append(r.row.ttft)                              # measured from the start of handle() = the arrival
        timed_out += r.attempts[0][1] == "timeout"
    in_outage = sum(1 for i in range(n) if i * gap < 300)
    return in_outage, timed_out, max(ttft), sum(ttft) / n


for label, th in (("no breaker", 10 ** 9), ("breaker: 3 consecutive, 30 s cooldown", 3)):
    arrived, paid, worst, mean = outage_run(th)
    print(f"{label:40s} {paid:2d} of the {arrived} requests that arrived during the outage paid the 10 s timeout | "
          f"worst TTFT {worst:5.2f} s | mean TTFT {mean:4.2f} s (simulated)")

# %% [markdown]
# Without a breaker, every one of the 25 requests that arrive during the outage pays the 10 s timeout before it falls
# through. With a breaker, three requests pay it, and then the breaker opens. After that, only the one probe per
# cooldown pays it. In total, 8 of 25 pay it. The other requests skip the primary at once.
#
# The worst case is the same 10.3 s. The breaker decreases how *often* requests pay a timeout, not what one timeout
# costs. That is the job of the first-byte deadline (Exercise 2.6). Also, here the task of the breaker was easy: each
# failure completed before the next request arrived. In production, requests overlap, and the breaker learns only as
# fast as failures complete (Exercise 2.4).
#
# ## Worked example 2 — availability, latency and cost of a chain

# %%
print(f"99.5 % + 99.0 %, independent          : {routing.chain_availability([0.995, 0.99]):.5%}")
print(f"   the same with 0.1 % common-mode   : {routing.chain_availability([0.995, 0.99], 0.001):.5%}")
print(f"   a third independent 99 % target   : {routing.chain_availability([0.995, 0.99, 0.99], 0.001):.5%}  <- capped")
cost = {m: price_call(m, 5000, 350, 2700) for m in ("gemini-3.5-flash", "gpt-5.4-mini", "claude-haiku-4-5")}
print("the §5.3 call priced per target:", {m: f"${c:.6f}" for m, c in cost.items()})
for t_fail in (0.15, 10.0):
    steps = [dict(p_ok=0.95, t_ok=0.6, t_fail=t_fail, cost_ok=cost["gemini-3.5-flash"]),
             dict(p_ok=0.99, t_ok=0.5, t_fail=0.15, cost_ok=cost["gpt-5.4-mini"]),
             dict(p_ok=0.99, t_ok=0.7, t_fail=0.15, cost_ok=cost["claude-haiku-4-5"])]
    r = routing.chain_cost(steps)
    print(f"primary fails 5 % in {t_fail:5.2f} s: mean TTFT {r['latency']:.3f} s, mean ${r['cost']:.6f}, P(all fail) {r['p_fail']:.1e}")

# %% [markdown]
# ## Exercise 2.1 — what falls through
# Write `should_fall_through(status, code)`. `status` is an HTTP status or `None` (no answer before the timeout).
# `code` is the `code` of the error body. Return True only when another target has a real chance to succeed. Two
# statuses are easy to forget: 408 (the request timed out at the provider) and the 529 `overloaded_error` of
# Anthropic.

# %% exercise
def should_fall_through(status, code=None):
    ### BEGIN SOLUTION
    if status == 400:
        return code == "context_length_exceeded"
    return status is None or status in (408, 429, 500, 502, 503, 504, 529)
    ### END SOLUTION

# %% check
cases = [(429, None), (500, None), (502, None), (503, None), (504, None), (None, None), (400, "context_length_exceeded"),
         (408, None), (529, None), (400, None), (400, "content_policy"), (401, None), (403, None), (404, None)]
assert [should_fall_through(*c) for c in cases] == [routing.falls_through(*c) for c in cases]
print("✅ falls through:", [c for c in cases if should_fall_through(*c)])
print("   stops:        ", [c for c in cases if not should_fall_through(*c)])

# %% [markdown]
# ## Exercise 2.2 — chain availability
# Write `chain_avail(avails, common_mode)`: the probability that at least one target answers. Each target fails
# independently with probability $1 - a_i$. Separately, a common-mode failure with probability `common_mode` makes all
# of them fail.

# %% exercise
def chain_avail(avails, common_mode=0.0):
    ### BEGIN SOLUTION
    all_fail = 1.0
    for a in avails:
        all_fail *= 1 - a
    return (1 - common_mode) * (1 - all_fail)
    ### END SOLUTION

# %% check
rng = random.Random(2)
for _ in range(200):
    av, cm = [rng.uniform(0.9, 0.9999) for _ in range(rng.randint(1, 4))], rng.uniform(0, 0.01)
    assert abs(chain_avail(av, cm) - routing.chain_availability(av, cm)) < 1e-12
assert round(chain_avail([0.995, 0.99], 0.001), 8) == 0.99895005
print(f"✅ {chain_avail([0.995, 0.99], 0.001):.3%}: the common-mode term, not the fallbacks, sets the ceiling")

# %% [markdown]
# ## Exercise 2.3 — expected latency of walking a chain
# Each step has `p_ok`, `t_ok` (the time to first token on success) and `t_fail` (the time to detect a failure). Write
# `expected_latency(steps)`: the mean time to the first token *or* to the final failure. (Hint: every outcome after a
# failure pays the time of that failure.)

# %% exercise
def expected_latency(steps):
    ### BEGIN SOLUTION
    reach, t = 1.0, 0.0
    for s in steps:
        t += reach * (s["p_ok"] * s["t_ok"] + (1 - s["p_ok"]) * s["t_fail"])
        reach *= 1 - s["p_ok"]
    return t
    ### END SOLUTION

# %% check
for _ in range(200):
    st = [dict(p_ok=rng.uniform(0.5, 1), t_ok=rng.uniform(0.1, 2), t_fail=rng.uniform(0.05, 30), cost_ok=0.0)
          for _ in range(rng.randint(1, 4))]
    assert abs(expected_latency(st) - routing.chain_cost(st)["latency"]) < 1e-12
print("✅ expected latency matches routing.chain_cost on 200 random chains")

# %% [markdown]
# ## Exercise 2.4 — a breaker learns only as fast as failures complete
# The breaker itself (closed, open, one probe after the cooldown) belongs to the 07.2 lab. Notebook 10 of
# gcp-agent-platform-lab builds it, half-open included. Here it is `routing.Breaker`, as given. The gateway adds
# *concurrency*. Requests arrive every $1/\text{rate}$ seconds. Each attempt at the dead primary takes `t_fail` seconds
# to fail (the full timeout, a first-byte deadline, or a fast 503).
#
# The breaker opens when `threshold` failures have **completed**. But at that time, every request that arrived in the
# meantime already waits on the dead target.
#
# Write `reach_before_open(rate, t_fail, threshold)`. It returns how many requests reach the dead target before its
# breaker opens. The value is a whole number: round up. Then set `at_50_rps` to a tuple of that number at 50 requests a
# second with threshold 3. The tuple holds the values for a 10 s timeout, a 1 s first-byte deadline and a 0.15 s 503,
# in that order.

# %% exercise
import math


def reach_before_open(rate, t_fail, threshold):
    ### BEGIN SOLUTION
    # request k arrives at k / rate; the threshold-th failure completes at (threshold - 1) / rate + t_fail
    return math.ceil(threshold - 1 + rate * t_fail)
    ### END SOLUTION

### BEGIN SOLUTION
at_50_rps = tuple(reach_before_open(50, tf, 3) for tf in (10.0, 1.0, 0.15))
### END SOLUTION

# %% check
def overlapping(rate, t_fail, threshold):
    """Arrivals every 1/rate s against a dead target; failures are recorded when they complete (routing.Breaker)."""
    b, in_flight, reached, k = routing.Breaker(threshold, cooldown=1e9), [], 0, 0
    while True:
        t = k / rate
        for done in sorted(x for x in in_flight if x <= t):
            b.record(False, done)
        in_flight = [x for x in in_flight if x > t]
        if b.state(t) == "open":
            return reached
        if b.allow(t):
            reached += 1
            in_flight.append(t + t_fail)
        k += 1


for rate, tf, th in [(50, 10.0, 3), (50, 1.0, 3), (50, 0.15, 3), (10, 2.5, 5), (200, 0.4, 3), (1, 10.0, 3), (7, 0.3, 2)]:
    assert abs(reach_before_open(rate, tf, th) - overlapping(rate, tf, th)) <= 1, (rate, tf, th, overlapping(rate, tf, th))
assert at_50_rps == (502, 52, 10), at_50_rps
print(f"✅ at 50 requests a second, {at_50_rps[0]} requests wait on a 10 s timeout before the breaker opens; a 1 s "
      f"first-byte deadline cuts that to {at_50_rps[1]}, a fast 503 to {at_50_rps[2]}. The breaker needs fast failures.")

# %% [markdown]
# ## Exercise 2.5 — fall back only before the first byte
# Write `relay(events)`: walk the stream events of a provider. If an event with an `"error"` key arrives before the
# gateway relayed any content chunk, return `"fall through"`.
#
# If not, return `(relayed, outcome)`.
# The first value is how many chunks with choices the gateway relayed to the client. The second value is `"error"` if
# the stream ended in an error, and `"ok"` if not. (Ignore the usage chunk.)

# %% exercise
def relay(events):
    ### BEGIN SOLUTION
    relayed = 0
    for ev in events:
        if ev == api.DONE:
            break
        if "error" in ev:
            return "fall through" if relayed == 0 else (relayed, "error")
        if ev.get("choices"):
            relayed += 1
    return relayed, "ok"
    ### END SOLUTION

# %% check
stream = lambda fail: list(FakeProvider("p", Clock(), output_tokens=12, fail_after=fail).chat(
    api.chat_request("m", [{"role": "user", "content": "hi"}], stream=True)).events)
assert relay(stream(0)) == "fall through" and relay(stream(5)) == (5, "error") and relay(stream(None)) == (12, "ok")
for fail, expect in ((0, "error before first byte"), (5, "cut after first byte")):
    clock = Clock()
    ks = keys.KeyStore(seed=3)
    k = ks.issue("acme")
    gw = Gateway(keys=ks, router=Router({"chat": chain[:2]}), clock=clock,
                 providers={"google": FakeProvider("google", clock, fail_after=fail), "openai": FakeProvider("openai", clock)})
    r = gw.handle(k, api.chat_request("chat", [{"role": "user", "content": "hi"}], stream=True))
    assert r.attempts[0][1] == expect, r.attempts
print("✅ before the first byte: fall through; after it: surface the error and meter what was relayed")

# %% [markdown]
# ## Exercise 2.6 — pick a first-byte deadline from a latency budget
# The primary fails 5 % of requests because it hangs. The gateway stops the wait after `t_fail` seconds and falls
# through (the steps are as in worked example 2). The budget of the route is a **mean** time to first token of 0.70 s.
# Set `deadline` to the largest `t_fail`, to 0.01 s, that meets the budget. Calculate it. Do not search blindly.
#
# Then show why a whole-response timeout cannot do the job. A healthy answer on this route is 350 tokens at 20 ms each,
# after a 0.6 s first token. Thus set `full_response_s` to the shortest whole-response timeout that never stops a
# healthy stream. Set `mean_with_it` to the mean time to first token of the route if the gateway detects failures only
# that late.

# %% exercise
def steps_for(t_fail):
    return [dict(p_ok=0.95, t_ok=0.6, t_fail=t_fail, cost_ok=0.0), dict(p_ok=0.99, t_ok=0.5, t_fail=0.15, cost_ok=0.0),
            dict(p_ok=0.99, t_ok=0.7, t_fail=0.15, cost_ok=0.0)]

### BEGIN SOLUTION
base = routing.chain_cost(steps_for(0.0))["latency"]      # latency is linear in t_fail with slope 0.05
deadline = int((0.70 - base) / 0.05 * 100) / 100
full_response_s = 0.6 + 349 * 0.02
mean_with_it = routing.chain_cost(steps_for(full_response_s))["latency"]
### END SOLUTION

# %% check
assert routing.chain_cost(steps_for(deadline))["latency"] <= 0.70 < routing.chain_cost(steps_for(deadline + 0.01))["latency"]
assert abs(full_response_s - 7.58) < 1e-9 and abs(mean_with_it - routing.chain_cost(steps_for(7.58))["latency"]) < 1e-12
assert mean_with_it > 0.70
print(f"✅ a {deadline:.2f} s first-byte deadline keeps the mean at {routing.chain_cost(steps_for(deadline))['latency']:.3f} s; "
      f"a whole-response timeout can be no shorter than {full_response_s:.2f} s, which puts the mean at {mean_with_it:.3f} s "
      "-- over budget, so time out on the first byte and on the whole response separately")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Clients ask for aliases. Each alias is an ordered chain of provider, model and region.
# The gateway filters the chain by what the request needs: tools, reasoning effort, context, residency. Then a policy puts
# the chain in order. The policies are: as written, cheapest capable, lowest moving-average TTFT, a canary share, or a
# separate chain for each tier.
#
# "We fall through on 429, 5xx, timeouts and context-too-long. We never fall through on a bad request, credentials or a
# policy refusal, and never after the first byte. After the first byte, we send an error chunk to the client.
#
# "Each target has a breaker. Three consecutive failures open it for 30 s, and one probe decides. Thus an outage costs
# a few timeouts, not one for each request. Also, we apply the timeout to the first byte, not to the whole response,
# because the breaker learns only as fast as failures complete.
#
# "What the targets share puts a cap on the availability of the chain. The targets 99.5 % and 99 % give 99.995 %
# independently, but 99.895 % with a 0.1 % common-mode failure. Thus the fallbacks go in different failure domains,
# with a quota that matches the traffic that they will take."
#
# **Drill questions**
# 1. *Why not fall back on a 400?* The next target gets the same bad request. A fall-through multiplies the cost and
#    the latency, and it hides a bug in the client. The exception is `context_length_exceeded`, because a target with a
#    longer context can serve it.
# 2. *We added a third provider and availability did not move. Why?* A common-mode failure (the gateway, a region, one
#    provider behind two aliases) already put a cap on the chain. Independent targets only decrease the part that was
#    already small.
# 3. *The breaker is configured, yet the first ten seconds of an outage were terrible. Why?* The requests overlap. The
#    breaker opens only after failures *complete*. Thus every request in flight during the first timeout pays it:
#    about 500 at 50 requests a second (Exercise 2.4). A first-byte deadline and fast failure signals (connect errors,
#    503s) make that window shorter.
