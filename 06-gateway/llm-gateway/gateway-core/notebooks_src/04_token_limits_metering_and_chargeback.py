# %% [markdown]
# # 04 · Token limits, metering and chargeback
#
# **Tier:** T0. It uses only the CPU and no network, and it runs in a few seconds. The notebook simulates the tokens-per-minute meter
# of the provider and every stream. The same limits over HTTP, and a ledger reconciled against the `usage` and the
# `/metrics` of a real vLLM (T1), are `gateway-lab` notebook `04_streaming_limits_metering_and_chargeback`.
#
# ## The one-minute version
# When the gateway admits a request, the cost of the request is unknown. Most of the cost is output that the model has
# not generated yet. Also, the output length is heavy-tailed. Thus a limit that charges **per request** admits the
# outputs, whatever their length. On the day that a thinking model goes into service, that limit lets through two times
# the tokens per minute of the provider. Charge **tokens**:
#
# - **reserve** the prompt + an output bound at admission,
# - **debit** the tokens as the stream delivers them,
# - **reconcile** with `usage` at the end.
#
# Then **meter**:
#
# - The `usage` of the provider is the bill (thinking tokens bill as output).
# - A ledger row for each request puts a price on it.
# - Estimates cover only cut streams.
# - The gateway reconciles the ledger against the counters of the provider.
# - The chargeback of a shared GPU pool is by GPU time, not by tokens.
#
# By the end, you can do these things:
#
# - predict over-admission from two means,
# - make the limiter that reserves tokens,
# - find the size of a reservation from a simulation,
# - put a price on a thinking call from raw usage,
# - do the chargeback of a pool,
# - find and explain ledger drift.
#
# Primer: §4 *Streaming-aware rate limits* and §5 *Metering, tracing and chargeback* (`../PRIMER.md`). The token bucket
# itself belongs to 06.3 (scaling primer §5.1). The thinking workload is in §7 of the 00.5 primer.

# %%
import random

from gwcore import metering, ratelimit
from gwcore.providers import normalize_usage

b = ratelimit.TokenBucket(rate=10, capacity=60)
print("take 50 at t=0 -> wait", b.acquire(50, 0), "| left", b.tokens)
print("take 30 at t=1 -> wait", b.acquire(30, 1), "s (10 left + 10 refilled = 20; 10 missing at 10/s)")
print("take 100 at t=2 -> wait", b.acquire(100, 2), "s, then", b.tokens, "tokens: larger than the burst goes into debt")
old, new = ratelimit.lognormal_mean(300, 1.0), ratelimit.lognormal_mean(1500, 1.0)
print(f"\noutput means: median 300 -> {old:.1f}; median 1,500 (the 00.5 primer's thinking workload) -> {new:,.0f}")
print(f"a per-request charge sized for the old mix over-admits x{ratelimit.overadmission_ratio(1500, old, new):.2f}")

# %% [markdown]
# ## Worked example 1 — the experiment
# `compare_buckets()` offers 12 requests a second for ten minutes to a key with a 1,000,000 tokens-per-minute limit.

# %%
res = ratelimit.compare_buckets()
print(f"{'limiter':34s} {'admitted':>8s} {'peak/limit':>10s} {'s over':>7s} {'when (s)':>10s} {'served/limit':>12s} {'overrun':>9s}")
for name, r in res.items():
    when = f"{r['first_over']}-{r['last_over']}" if r["first_over"] else "-"
    print(f"{name:34s} {r['admitted']:8,d} {r['peak_ratio']:10.2f} {r['seconds_over_limit']:7d} {when:>10s} "
          f"{r['utilisation']:12.3f} {r['overrun_tokens']:9,.0f}")
print("(arrivals stop at t = 600 s; each run goes on until its admitted streams drain -- simulated)")

# %% [markdown]
# The per-request bucket admits exactly as many requests after the rollout as before, because it cannot see tokens. It
# pushes the provider to 1.97× its limit at the peak. After the bucket binds ($t = 45$ s), the window of the provider
# stays over its limit continuously until $t = 648$ s, while the admitted thinking streams drain. On the outputs of
# yesterday, the same bucket was over for 116 scattered seconds between $t = 59$ and 600. Its size came from the mean,
# and it still meets the tail.
#
# If you reserve the cap, the result is exact, but it wastes three quarters of the budget. If you reserve an estimate and
# reconcile, the limiter serves 70 % with no second over the limit *in this run*. All of it models a provider that
# counts tokens when it processes them. A hosted API that charges the requested `max_tokens` at admission needs a
# reservation that matches what the gateway sends upstream (PRIMER §4.2, verify).
#
# ## Worked example 2 — usage is the bill
# The same call from the adapter samples: 5,000 prompt tokens (2,700 cached), 350 visible tokens and 1,200 reasoning
# tokens.

# %%
g = {"promptTokenCount": 5000, "cachedContentTokenCount": 2700, "candidatesTokenCount": 350, "thoughtsTokenCount": 1200}
u = normalize_usage("gemini", g)
right = metering.price_call("gemini-3.5-flash", u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"])
wrong = metering.price_call("gemini-3.5-flash", 5000, g["candidatesTokenCount"], 2700)
print(f"billed on usage: ${right:.6f} | billed on candidates only: ${wrong:.6f} (x{right / wrong:.2f} under-billed)")
for m in ("gemini-3.5-flash", "claude-haiku-4-5", "gpt-5.4-mini", "gemini-3.5-flash-lite"):
    print(f"{m:22s} ${metering.price_call(m, 5000, 350, 2700):.7f} per call | "
          f"${metering.cost_per_million(m, 5000, 350, 2700):.4f} per 1M tokens (blended)")
print(f"self-hosted H100 Spot, 6,846.5 tok/s: ${metering.self_hosted_per_million(3.7, 6846.515877965477):.3f} per 1M output tokens")

# %% [markdown]
# ## Exercise 4.1 — predict the over-admission
# The prompts of a route are 800 tokens. Its bucket charges a constant estimate, and the outputs of today give its size:
# lognormal, median 200, $\sigma = 1.2$. The outputs of a new model are lognormal with median 2,000 and $\sigma = 1.2$.
# When the bucket binds, it will let through more tokens than its size assumed. Set `ratio` to that multiple. Use the
# lognormal mean $\text{median} \cdot \exp(\sigma^2/2)$.

# %% exercise
import math

### BEGIN SOLUTION
m_old, m_new = 200 * math.exp(1.2 ** 2 / 2), 2000 * math.exp(1.2 ** 2 / 2)
ratio = (800 + m_new) / (800 + m_old)
### END SOLUTION

# %% check
assert abs(ratio - ratelimit.overadmission_ratio(800, ratelimit.lognormal_mean(200, 1.2), ratelimit.lognormal_mean(2000, 1.2))) < 1e-9
print(f"✅ x{ratio:.2f}: the output mean grew 10x, but the prompt dilutes it -- short-prompt routes suffer most")

# %% [markdown]
# ## Exercise 4.2 — the reserving limiter
# Write `MyLimiter(limit, window=60)` with the three moves:
#
# * `admit(rid, reserve, now)`: return True and hold `reserve` if `used(now) + outstanding + reserve <= limit`.
# * `debit(rid, n, now)`: the provider processed `n` tokens at `now`. Count them in the window, and remove them from the
#   reservation of the request (never below zero).
# * `finish(rid)`: release all that the request still holds.
#
# `used(now)` is the tokens debited in the last `window` seconds (an event at time $t$ leaves the window at
# $t + \text{window}$).

# %% exercise
from collections import deque


class MyLimiter:
    def __init__(self, limit, window=60.0):
        ### BEGIN SOLUTION
        self.limit, self.window, self.events, self.held = limit, window, deque(), {}
        ### END SOLUTION

    def used(self, now):
        ### BEGIN SOLUTION
        while self.events and self.events[0][0] <= now - self.window:
            self.events.popleft()
        return sum(n for _, n in self.events)
        ### END SOLUTION

    @property
    def outstanding(self):
        ### BEGIN SOLUTION
        return sum(self.held.values())
        ### END SOLUTION

    def admit(self, rid, reserve, now):
        ### BEGIN SOLUTION
        if self.used(now) + self.outstanding + reserve > self.limit:
            return False
        self.held[rid] = reserve
        return True
        ### END SOLUTION

    def debit(self, rid, n, now):
        ### BEGIN SOLUTION
        self.events.append((now, n))
        self.held[rid] = max(0, self.held.get(rid, 0) - n)
        ### END SOLUTION

    def finish(self, rid):
        ### BEGIN SOLUTION
        self.held.pop(rid, None)
        ### END SOLUTION

# %% check
rng = random.Random(4)
mine, ref, live, t = MyLimiter(40_000), ratelimit.ReserveLimiter(40_000), {}, 0.0
for rid in range(1500):
    t += rng.expovariate(3.0)
    for r in list(live):
        n = min(live[r], rng.randint(0, 300))
        live[r] -= n
        mine.debit(r, n, t)
        ref.debit(r, n, t)
        if not live[r]:
            mine.finish(r)
            ref.finish(r)
            del live[r]
    reserve = rng.choice([1_500, 4_000, 9_000])
    a, b = mine.admit(rid, reserve, t), ref.admit(rid, reserve, t)
    assert a == b, rid
    if a:
        live[rid] = rng.randint(1, reserve)
    assert mine.used(t) + mine.outstanding <= 40_000
print("✅ same admit decisions as ratelimit.ReserveLimiter on 1,500 requests, and used + reserved never passed the limit")

# %% [markdown]
# ## Exercise 4.3 — size the reservation
# If you reserve too much, every stream holds unused budget for all of its life. If you reserve too small a quantity, tokens
# overrun their reservations. Use the thinking workload of worked example 1
# (`ratelimit.workload(600, 12, 1500, 1500, 1.0, seed=0)`) and a 1,000,000 TPM limit. With
# `ratelimit.simulate(..., "reserve", ...)`, try output reservations of 1,024, 2,048, 4,096, 8,192 and 16,384. Of the
# reservations with **zero** seconds over the limit, set `best_reserve` to the one with the highest utilisation.

# %% exercise
work = ratelimit.workload(600, 12, 1500, 1500, 1.0, seed=0)
### BEGIN SOLUTION
runs = {ro: ratelimit.simulate(work, "reserve", limit=1_000_000, reserve_out=ro) for ro in (1024, 2048, 4096, 8192, 16384)}
best_reserve = max((r["utilisation"], ro) for ro, r in runs.items() if r["seconds_over_limit"] == 0)[1]
### END SOLUTION

# %% check
runs_ref = {ro: ratelimit.simulate(work, "reserve", limit=1_000_000, reserve_out=ro) for ro in (1024, 2048, 4096, 8192, 16384)}
for ro, r in runs_ref.items():
    print(f"reserve {ro:6,d}: served {r['utilisation']:.3f} of the limit, peak {r['peak_ratio']:.2f}, "
          f"{r['seconds_over_limit']} s over, overrun {r['overrun_tokens']:,.0f} tokens (simulated)")
assert best_reserve == max((r["utilisation"], ro) for ro, r in runs_ref.items() if r["seconds_over_limit"] == 0)[1]
print(f"✅ reserve {best_reserve:,}: smaller reservations serve more but overrun into seconds over the limit; "
      "the cap never overruns and strands three quarters of the budget")

# %% [markdown]
# ## Exercise 4.4 — bill a thinking call from raw usage
# A Gemini response reports `usageMetadata = {promptTokenCount: 12000, cachedContentTokenCount: 8000,
# candidatesTokenCount: 600, thoughtsTokenCount: 4400}` for **gemini-3.5-flash**. Set `bill` in dollars. Do it by hand
# first (the prices are in `gwcore.providers.CATALOGUE`). Then compare with `normalize_usage` + `metering.price_call`.

# %% exercise
### BEGIN SOLUTION
inp, out, cached = 1.50, 9.00, 0.15
bill = ((12000 - 8000) * inp + 8000 * cached + (600 + 4400) * out) / 1e6
### END SOLUTION

# %% check
u = normalize_usage("gemini", {"promptTokenCount": 12000, "cachedContentTokenCount": 8000, "candidatesTokenCount": 600,
                               "thoughtsTokenCount": 4400})
assert abs(bill - metering.price_call("gemini-3.5-flash", u["prompt_tokens"], u["completion_tokens"], u["cached_tokens"])) < 1e-12
print(f"✅ ${bill:.4f}: the 4,400 thinking tokens are {4400 * 9 / 1e6 / bill:.0%} of the bill")

# %% [markdown]
# ## Exercise 4.5 — charge back a shared pool
# A self-hosted pool cost $10,000 this month. Write `by_gpu_seconds(usage, pool_cost, prefill_tps, decode_tps)`. Each
# tenant pays in proportion to its GPU time. Calculate that time for prompt tokens at `prefill_tps` and for completion
# tokens at `decode_tps`.

# %% exercise
def by_gpu_seconds(usage, pool_cost, prefill_tps=68_000.0, decode_tps=6_846.5):
    ### BEGIN SOLUTION
    sec = {t: u["prompt_tokens"] / prefill_tps + u["completion_tokens"] / decode_tps for t, u in usage.items()}
    return {t: pool_cost * s / sum(sec.values()) for t, s in sec.items()}
    ### END SOLUTION

# %% check
usage = {"rag": {"prompt_tokens": 90_000_000, "completion_tokens": 1_000_000},
         "thinking": {"prompt_tokens": 5_000_000, "completion_tokens": 10_000_000},
         "chat": {"prompt_tokens": 20_000_000, "completion_tokens": 3_000_000}}
mine = by_gpu_seconds(usage, 10_000)
ref = metering.chargeback(usage, 10_000, by="gpu_seconds")
assert all(abs(mine[t] - ref[t]) < 1e-6 for t in usage)
tok = metering.chargeback(usage, 10_000, by="tokens")
for t in usage:
    print(f"{t:9s} by tokens ${tok[t]:9,.2f}   by GPU-seconds ${mine[t]:9,.2f}")
print("✅ token-proportional chargeback makes the prompt-heavy tenant pay for the output-heavy one's decode time")

# %% [markdown]
# ## Exercise 4.6 — find the drift, and predict it
# The ledger in the next cell recorded a day of traffic, and the usage export of the provider does not agree with it.
# Use `Ledger.reconcile` (1 % tolerance). Set `drifted` to the set of `(model, field)` pairs that are out of tolerance.
#
# Then explain the number before anyone opens a ticket. Of the 400 requests of the day, 40 had a cut in the middle of
# the stream (there were no retries). The gateway billed each cut stream on the 90 tokens that it relayed, and it marked
# the rows estimated. The logs of the provider show that it generated 150 tokens on each cut stream before it detected
# the disconnect. For every other request, the provider generated 300 tokens and the gateway billed 300. Set
# `predicted_drift` to the completion-token difference ($\text{provider} - \text{ledger}$) that you expect from that
# cause alone.

# %%
rng = random.Random(9)
ledger = metering.Ledger()
for i in range(400):
    cut = i < 40
    ledger.add(metering.row_from_usage(f"r{i}", "acme", "k", "gpt-5.4-mini",
                                       {"prompt_tokens": 1200, "completion_tokens": 90 if cut else 300}, estimated=cut))
provider = {"gpt-5.4-mini": {"prompt_tokens": 480_000, "completion_tokens": 360 * 300 + 40 * 150}}
print(ledger.totals()["acme"])

# %% exercise
### BEGIN SOLUTION
report = ledger.reconcile(provider)
drifted = {k for k, v in report.items() if not v["ok"]}
predicted_drift = 40 * (150 - 90)
### END SOLUTION

# %% check
assert drifted == {("gpt-5.4-mini", "completion_tokens")}
d = ledger.reconcile(provider)[("gpt-5.4-mini", "completion_tokens")]
assert predicted_drift == d["diff"], (predicted_drift, d["diff"])
print(f"✅ ledger {d['ledger']:,} vs provider {d['provider']:,} completion tokens ({d['diff']:+,}) = 40 cut streams x "
      "(150 generated - 90 relayed): the estimated rows explain all of it, so correct those rows, not the price table")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Limits are in tokens, because the cost of a request is unknown at admission and
# heavy-tailed. We reserve the prompt plus an output bound, debit tokens as the stream delivers them, and reconcile with
# usage at the end. Used plus reserved never passes the limit, so the provider never sees more. On the day that thinking
# went into service, a per-request bucket let through 1.97× the tokens per minute of the provider. And the bucket
# cannot even see the change.
#
# "The reservation is a setting, and we find its size by simulation. The cap is exact and strands budget. An estimate
# serves more, with small overruns. Limits nest: key, tenant, org, provider key. One Lua script does a check of all of them
# and commits them atomically.
#
# "Every request writes a ledger row with a price from usage. We bill thinking as output. If you bill Gemini on
# candidates alone, you under-bill a thinking call 2.5×. We bill cut streams on the relayed deltas and mark them
# estimated. The gateway reconciles the ledger daily against the export of the provider.
#
# "The chargeback of the self-hosted pool is by GPU-seconds, not by tokens. If not, the RAG tenant pays for the decode
# of the thinking tenant."
#
# **Drill questions**
# 1. *After a thinking-model rollout the provider returns 429s, but our request limit never tripped. Why?* It charged
#    per request. The outputs grew by an order of magnitude, with a heavy tail. Thus the same request rate carried ~2×
#    the tokens. Reserve, debit and reconcile in tokens. Apply TPM next to RPM for each tenant. Put a budget on thinking.
# 2. *Why not simply reserve `max_tokens` for every request?* It is exact, but it holds budget for the life of every
#    stream. At a 16K cap, the key served 26.5 % of its limit. Reserve an estimate, debit as the stream goes, and keep a
#    margin. Reserve the cap only where a 429 from the provider is unacceptable.
# 3. *The ledger and the invoice disagree by 3 % on output tokens. Where do you look?* Look at four causes. First, cut
#    streams billed on estimates. Second, retries that the provider billed and that the ledger recorded once. Third, the
#    pricing of cached tokens. Fourth, estimator drift. Reconcile for each model and field, daily, and alert on it.
