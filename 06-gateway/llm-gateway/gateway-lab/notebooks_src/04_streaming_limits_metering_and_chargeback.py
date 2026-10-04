# %% [markdown]
# # 04 · Streaming limits, metering and chargeback: count tokens, not requests
#
# **Tier:** T0. Tenants send requests over localhost HTTP to fake providers, and each fake provider has its own
# tokens-per-minute limit. The rate-limit "minute" is **compressed to 2 seconds**, so each run takes seconds. All
# timings and token counts are **simulated**. T1: the lab reconciles the ledger with the `usage` and the `/metrics`
# of a real vLLM.
#
# ## The one-minute version
#
# The token bucket (§5.1) and the admission control (§5.3) of the scaling primer decide if a request can start. For
# LLM calls, the cost of a request is unknown when the request starts. The model sets the output length during the
# stream. The output length is heavy-tailed (00.5 PRIMER §7: the outputs of a thinking model are an order of
# magnitude longer).
#
# A bucket that charges each request a constant guess at the start **over-admits** by
# $(\text{prompt} + \text{actual output}) \:/$ $(\text{prompt} + \text{guess})$. Then the provider refuses the
# excess with its own TPM limit, and the tenant cannot explain these 429s (PRIMER §4). The solution is the metering
# of the tokens where they flow:
#
#     admit      reserve prompt estimate + min(requested cap or a default, a hard cap) from the tenant's TPM
#     stream     debit anything beyond the reservation as chunks arrive
#     reconcile  refund (reservation − actual) from the provider's `usage` at the end
#
# The gateway keeps RPM beside TPM, for each tenant, under a gateway-wide share. Thus a noisy neighbour empties only
# its own bucket.
#
# Then the same `usage` sets the price of the request in the **ledger**. The `usage` is the authoritative count. An
# estimate covers only the admission and the streams that stop before the usage chunk (PRIMER §5). The ledger
# answers chargeback, also for a shared self-hosted GPU (01 PRIMER §8.1).

# %%
import statistics, time
from gwlab import bench, client, env, promtext, report, t1
from gwlab.fakes import FakeSpec
from gwlab.gateway import metering
from gwlab.gateway.adapters import normalise_usage, payload
from gwlab.gateway.config import Limits, Tenant
from gwlab.gateway.ratelimit import Limiter
from gwlab.stack import LocalStack

tiers = env.describe()
MINUTE = 2.0                                     # seconds per rate-limit "minute" (compressed, simulated)
HEAVY = dict(ttft_s=0.02, prefill_s_per_token=0.0, itl_s=0.002, output_median=60, output_sigma=0.9, output_max=600)


def make_stack(mode, provider_tpm=3000, tenant_tpm=1500, global_tpm=0, seed=1):
    fakes = {"acme": FakeSpec(name="acme", tpm=provider_tpm, minute_s=MINUTE, seed=seed, **HEAVY),
             "bolt": FakeSpec.anthropic("bolt", seed=seed, **HEAVY)}
    ov = {"limits": {"mode": mode, "minute_s": MINUTE, "default_output_estimate": 200, "hard_output_cap": 600,
                     "per_request_output_guess": 20, "global_tpm": global_tpm},
          "aliases": {"solo": {"targets": ["acme/fast"]}},
          "tenants": {t: {"aliases": ["chat", "solo"], "tpm": tenant_tpm, "rpm": 100000} for t in ("team-a", "team-b")}}
    return LocalStack(fakes=fakes, overrides=ov).start()


def outcome(stack, since=0):
    """Who answered or refused: the provider (ok), the gateway's own limit (429 at the door), the provider's limit
    (429 after a round trip), or nobody (503: acme's breaker opened after the provider's 429s)."""
    ds = stack.decisions()[since:]
    return {"ok": sum(1 for d in ds if d.get("status") == 200),
            "gateway_429": sum(1 for d in ds if d.get("error") == "rate_limit_exceeded" and not d["attempts"]),
            "provider_429": sum(1 for d in ds if any(a.get("status") == 429 for a in d["attempts"])),
            "breaker_503": sum(1 for d in ds if d.get("status") == 503)}

print(f"a rate-limit minute here lasts {MINUTE} s; acme allows 3,000 tokens per minute; outputs are lognormal "
      "(median 60 tokens, a tail to 600) -- simulated")

# %% [markdown]
# ## Worked example: a per-request bucket vs reserve -> stream -> reconcile, against the same provider
#
# One tenant sends 30 requests a second for 6 seconds through the alias `solo`. This alias goes to acme only, so you
# can see the refusals of the provider. The provider counts tokens over a sliding window of one "minute" (2 s). The
# bucket of the gateway holds a burst (its capacity) and refills continuously. Thus, over any window, the bucket
# admits up to $\text{capacity} + \text{rate} \times \text{window}$.
#
# A bucket of 1,500 tokens per minute for the tenant keeps that inside the limit of the provider:
# $1{,}500 + 750/\text{s} \times 2\,\text{s} = 3{,}000$. A bucket set to the full 3,000 of the provider can admit
# up to 6,000 in the first window. This occurs because the burst is part of the budget.
#
# The `ReserveLimiter` of the core counts a sliding window instead. It admits a request only if
# $\text{used} + \text{reserved} + \text{reserve} \le \text{limit}$ (PRIMER §4.2). By construction, this window
# aligns with the window of a provider. This lab keeps the token bucket of the scaling lab (06.3), and thus it needs
# the size rule of the previous paragraph. The per-request bucket charges $\text{prompt} + 20$ for each request. The
# bucket in `reserve` mode charges $\text{prompt} + 200$ at the start, and at the end it settles to the real count.

# %%
runs = {}
for mode in ("per_request", "reserve"):
    s = make_stack(mode)
    k = s.issue_key("team-a")
    res = bench.run(s.url, [bench.TenantScript("team-a", k, rate=30, n=180, model="solo")], seed=2, label=mode)
    runs[mode] = (s, res, outcome(s))
    print(f"{mode:12s} {runs[mode][2]}  | served {sum(x.completion_tokens or 0 for x in res.samples)} output tokens "
          f"in {res.duration_s:.1f} s")

# %% [markdown]
# The demand is the same, and the provider is the same. The per-request bucket admits almost everything. Then the
# provider refuses the load that its TPM cannot carry. These 429s arrive after a round trip, and three of them in a
# row open the breaker of acme. After that, the gateway refuses *every* request with a 503 until a probe gets
# through. The refused requests include requests well inside the budget of the tenant.
#
# If the alias has a fallback chain, these requests go to a higher-cost provider at full traffic instead. The
# bucket in `reserve` mode refuses at the door of the gateway. It sends a `Retry-After` header and `x-ratelimit-*`
# headers, which tell which limit applies and when. The provider gets a load that it can carry.

# %% [markdown]
# ## Exercise 4.1 — the reservation
#
# Write `reservation(prompt_estimate, requested_output, default_estimate, hard_cap)`. It returns the tokens that
# `reserve` mode takes from the TPM of the tenant at admission. This value is the prompt estimate plus the output
# cap that the caller asked for. If the caller asked for no cap, use the default estimate. The output part is never
# more than the hard cap. The check compares your function with the limiter of the gateway over a grid.

# %% exercise
def reservation(prompt_estimate: int, requested_output, default_estimate: int, hard_cap: int) -> int:
    ### BEGIN SOLUTION
    return prompt_estimate + min(requested_output or default_estimate, hard_cap)
    ### END SOLUTION

# %% check
for d_est, cap in ((200, 600), (512, 2048), (1024, 1024)):
    lim = Limiter(Limits(mode="reserve", default_output_estimate=d_est, hard_output_cap=cap),
                  {"t": Tenant("t", ["chat"], rpm=10**6, tpm=10**9)})
    for p in (0, 12, 900):
        for req in (None, 1, 150, 600, 5000):
            assert lim.admit("t", p, req).reservation.charged == reservation(p, req, d_est, cap), (p, req, d_est, cap)
print("✅ a caller that sets max_completion_tokens gets exactly that reserved; one that does not pays the default, "
      "and nobody reserves past the hard cap")

# %% [markdown]
# ## Exercise 4.2 — how far a per-request bucket over-admits
#
# Write `over_admission(prompt, mean_output, charged_output)`. It returns the ratio of the tokens that the requests
# really use to the tokens that the bucket charged. The bucket charges each request
# $\text{prompt} + \mathrm{charged\_output}$, but each request costs $\text{prompt} + \mathrm{mean\_output}$.
#
# The check uses the per-request run of the first worked example. It takes the mean prompt that the gateway
# estimated, the mean output from the ledger and the 20-token guess. Then it compares your ratio with the measured
# ratio of all tokens served to all tokens charged.

# %% exercise
def over_admission(prompt: float, mean_output: float, charged_output: float) -> float:
    ### BEGIN SOLUTION
    return (prompt + mean_output) / (prompt + charged_output)
    ### END SOLUTION

# %% check
s_pr = runs["per_request"][0]
served = [r for r in s_pr.ledger() if r["status"] == 200]
decs = {d["request_id"]: d for d in s_pr.decisions()}
charged = [decs[r["request_id"]]["reserved_tokens"] for r in served]
prompt_est = statistics.mean(c - 20 for c in charged)
mean_out = statistics.mean(r["completion_tokens"] for r in served)
measured = sum(r["prompt_tokens"] + r["completion_tokens"] for r in served) / sum(charged)
pred = over_admission(prompt_est, mean_out, 20)
print(f"[SIMULATED] prompt ~{prompt_est:.0f} (estimated), mean output {mean_out:.0f}, guess 20: predicted {pred:.2f}x, measured {measured:.2f}x")
assert abs(pred / measured - 1) < 0.15
print("✅ the bucket admits", f"{pred:.1f}x", "the tokens it thinks it admits -- and a thinking model's 10x outputs make it 10x worse")

# %% [markdown]
# ## Worked example: the noisy neighbour
#
# Two tenants share a provider that permits 6,000 tokens per minute. The global bucket of the gateway has a size that
# matches it: 3,000 per minute. Thus the burst plus the refill over one window equals 6,000.
#
# `team-a` sends a flood at several times its share. `team-b` sends a small, steady flow. With only the global
# bucket, team-a empties it, and the gateway refuses the requests of team-b too. With a bucket for each tenant
# (1,500 each) under the same global bucket, the gateway refuses only team-a.

# %%
for label, tenant_tpm in (("one shared bucket", 10**9), ("a bucket per tenant", 1500)):
    s = make_stack("reserve", provider_tpm=6000, tenant_tpm=tenant_tpm, global_tpm=3000, seed=3)
    ka, kb = s.issue_key("team-a"), s.issue_key("team-b")
    r = bench.run(s.url, [bench.TenantScript("team-a", ka, rate=60, n=240, model="solo"),
                          bench.TenantScript("team-b", kb, rate=6, n=24, model="solo")], seed=4)
    sm = r.summary()
    print(f"{label:20s} team-a ok {sm['team-a']['ok']:3d}/{sm['team-a']['sent']}   team-b ok {sm['team-b']['ok']:3d}/{sm['team-b']['sent']}")
    s.stop()

# %% [markdown]
# ## Worked example: a stream cut short is still billed
#
# A client that walks away during the stream never receives the usage chunk (the spec of OpenAI says so). The
# gateway stops the upstream and writes the row from what it relayed. It counts one token for each content chunk,
# and it marks the row as an estimate. A bill of 0 for this row makes "disconnect just before the end" a free lunch.

# %%
s_r = runs["reserve"][0]
kc = s_r.issue_key("team-b")
walk = bench.run(s_r.url, [bench.TenantScript("team-b", kc, rate=5, n=10, model="solo", disconnect_after=5)], seed=6)
time.sleep(0.3)
rows_b = s_r.ledger(tenant="team-b")
print(report.ledger_table(rows_b, last=5))
print("rows billed from an estimate:", sum(r["usage_source"] == "estimate" for r in rows_b), "of", len(rows_b))

# %% [markdown]
# ## Exercise 4.3 — price a request from its usage
#
# Write `price(row, p)` for a ledger row (a dict with `prompt_tokens`, `cached_tokens`, `completion_tokens`) and a
# price row `p` (`p.input`, `p.cached`, `p.output`, dollars per 1M tokens). Use these rates:
#
# - the uncached prompt tokens at the input rate,
# - the cached prompt tokens at the cached rate,
# - the completion tokens, reasoning tokens included, at the output rate.
#
# The check calculates the price of every row of the earlier runs with the model that served the row. It compares
# each price with the ledger. Then it prices the §3.4 call of the scaling primer (5,000 in, 2,700 cached, 350 out on
# `gemini-3.5-flash`: $0.007005). After that, it prices the §1.5 call of the primer. This is the same call plus
# 1,200 thinking tokens, reported in the shape of Anthropic (normalise it first).

# %% exercise
def price(row: dict, p) -> float:
    ### BEGIN SOLUTION
    uncached = row["prompt_tokens"] - row["cached_tokens"]
    return (uncached * p.input + row["cached_tokens"] * p.cached + row["completion_tokens"] * p.output) / 1e6
    ### END SOLUTION

# %% check
n = 0
for s, _, _ in runs.values():
    for r in s.ledger():
        if r["target"]:
            assert abs(price(r, s.cfg.models[r["target"]].price) - r["cost_usd"]) < 1e-12, r
            n += 1
call = {"prompt_tokens": 5000, "cached_tokens": 2700, "completion_tokens": 350}
assert abs(price(call, metering.PRICES["gemini-3.5-flash"]) - 0.007005) < 1e-12
an = normalise_usage("anthropic", payload("anthropic_message")["usage"])
thinking = price({"prompt_tokens": an.prompt_tokens, "cached_tokens": an.cached_tokens, "completion_tokens": an.completion_tokens},
                 metering.PRICES["claude-haiku-4-5"])
assert (an.prompt_tokens, an.completion_tokens, an.reasoning_tokens) == (5000, 1550, 1200)
assert abs(thinking - (0.00432 + 1200 * 5.00 / 1e6)) < 1e-12
print(f"✅ {n} ledger rows re-priced exactly; the §3.4 call is $0.007005 on gemini-3.5-flash; the §1.5 call on "
      f"claude-haiku-4-5 is ${thinking:.5f}, of which $0.00600 is its 1,200 thinking tokens (prices dated 2026-09-26, verify)")

# %% [markdown]
# ## Worked example: $ per 1M tokens, hosted and self-hosted
#
# The cell gives the same call shape, blended over its 5,350 tokens, for each hosted row (verify, 2026-09-26). It
# also gives self-hosted rows from the hourly price of a GPU and a throughput. These are the Llama-3.1-8B numbers of
# 01 PRIMER §8.1: an H100 Spot at \$3.7/h and 6,846.5 tokens/s, and an L4 at \$0.70/h and 294.39 tokens/s. The cell
# shows each self-hosted row at 100 % and 60 % utilisation.

# %%
for name, p in metering.PRICES.items():
    print(f"{name:22s} ${metering.blended_per_million(p, 5000, 2700, 350):.4f} per 1M (this call shape)")
for name, gpu_h, tps in (("H100 Spot, self-hosted", 3.7, 6846.5), ("L4, self-hosted", 0.70, 294.39)):
    print(f"{name:22s} ${metering.cost_per_million_tokens(gpu_h, tps):.4f} per 1M output tokens at 100 %, "
          f"${metering.cost_per_million_tokens(gpu_h, tps, 0.6):.4f} at 60 %")

# %% [markdown]
# ## Exercise 4.4 — chargeback for a shared GPU
#
# A self-hosted pool costs the same per hour, whoever uses it. The question is how to divide the bill. By tokens,
# every token has the same weight. By GPU-seconds, a prompt token (prefill: batched, compute-bound) has a much
# smaller weight than an output token (decode: one step per token).
#
# Write `chargeback(rows, total_usd, by, prefill_s, decode_s)`. It returns `{tenant: dollars}`. The argument `by` is
# `"tokens"` or `"gpu_seconds"` ($\text{weight} = \text{prompt} \times \mathrm{prefill\_s} \:+$
# $\text{completion} \times \mathrm{decode\_s}$). The check sends a RAG tenant (long prompts, short answers) and a
# chat tenant (short prompts, long answers) through the gateway. Then it compares your result with
# `metering.chargeback` of the lab.

# %% exercise
def chargeback(rows: list, total_usd: float, by: str, prefill_s: float = 0.0, decode_s: float = 1.0) -> dict:
    ### BEGIN SOLUTION
    w = {}
    for r in rows:
        x = (r["prompt_tokens"] + r["completion_tokens"] if by == "tokens"
             else r["prompt_tokens"] * prefill_s + r["completion_tokens"] * decode_s)
        w[r["tenant"]] = w.get(r["tenant"], 0.0) + x
    tot = sum(w.values())
    return {t: total_usd * x / tot for t, x in w.items()}
    ### END SOLUTION

# %% check
s = make_stack("reserve", provider_tpm=0, tenant_tpm=10**7)
kr, kc2 = s.issue_key("team-a"), s.issue_key("team-b")
doc = "The contract says the supplier must deliver the goods within thirty days of the order. " * 25
bench.run(s.url, [bench.TenantScript("team-a", kr, rate=20, n=20, prompt=lambda i: doc + f" Question {i}: what is the deadline?",
                                     max_tokens=15),
                  bench.TenantScript("team-b", kc2, rate=20, n=20, prompt=lambda i: f"Write a story, part {i}.", max_tokens=400)],
          seed=7)
rows = [r for r in s.ledger() if r["status"] == 200]
for by, kw in (("tokens", {}), ("gpu_seconds", {"prefill_s": 0.0002, "decode_s": 0.02})):
    mine = chargeback(rows, 100.0, by, **kw)
    ref = metering.chargeback(rows, 100.0, by, kw.get("prefill_s", 0.0), kw.get("decode_s", 1.0))
    assert all(abs(mine[t] - ref[t]) < 1e-9 for t in ref)
print(report.totals_table(rows))
print(report.chargeback_table(rows, 100.0, 0.0002, 0.02))
print("✅ the RAG tenant (team-a) sends most of the tokens but little of the GPU time: pick the split before the invoice does")
s.stop()

# %% [markdown]
# ## Worked example: reconcile the ledger with the provider
#
# The provider keeps its own counters (the fakes use the names of vLLM). Compare them with the ledger. On the
# per-request run, every served row carries the `usage` of the provider, and the two agree to the token.
#
# On the reserve run, after the walk-away tenant, the gaps come only from the rows billed from an estimate. The
# prompt count of these rows is the chars/4 estimate of the gateway, because no usage chunk arrived. Their
# completion count is the number of relayed chunks. But the provider counted what it tokenized and generated before
# it saw the disconnect. That gap is the reconciliation item that you examine each month. It is not a rounding
# error.

# %%
for label, st in (("per_request run", runs["per_request"][0]), ("reserve run + walk-aways", s_r)):
    m = promtext.parse(st.provider_metrics("acme"))
    counters = {"acme": {"prompt_tokens": promtext.total(m, "vllm:prompt_tokens_total"),
                         "generation_tokens": promtext.total(m, "vllm:generation_tokens_total")}}
    print(label)
    print(report.reconcile_table(metering.reconcile(st.ledger(), counters)))

# %% [markdown]
# ## T1: reconcile with a real vLLM
#
# This cell runs when you set `GWLAB_VLLM_URL` and vLLM answers `/health`. It does these steps
# (`gwlab.t1.reconcile`):
#
# 1. Scrape the `/metrics` of vLLM.
# 2. Send twenty requests through a gateway whose first target is vLLM.
# 3. Scrape again.
# 4. Compare the vLLM rows of the ledger with the differences of the counters.
#
# If `/metrics` does not answer any more, the cell skips. It does not fail.

# %%
for st, _, _ in runs.values():
    st.stop()
if tiers["vllm_url"]:
    out = t1.reconcile(tiers["vllm_url"], n=20, fallback=FakeSpec(name="acme"))
    if out is None:
        print("T1 skipped: vLLM's /metrics did not answer (is it running? deploy/any-gpu/serve.sh)")
    else:
        tag = "MEASURED" if out["served_by_vllm"] == out["sent"] else (
            f"MEASURED for the {out['served_by_vllm']} of {out['sent']} requests vLLM served; the rest fell back (SIMULATED)")
        print(tag + "\n" + report.reconcile_table(out["reconcile"]))
else:
    print("T1 skipped: deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We limit tokens, not only requests, because the model sets the cost of a request during its
# stream. At admission, we reserve the prompt estimate plus the output cap of the caller, or a default, from the
# tokens-per-minute bucket of the tenant. This bucket is next to the requests-per-minute bucket of the tenant,
# under a gateway-wide share of the quota of the provider. We debit anything that the stream sends past the
# reservation. At the end, we reconcile to the `usage` of the provider, and a hard cap sets the limit for any one response.
#
# "Refusals occur at our door with a Retry-After, not at the door of the provider after a round trip. Every request
# writes a ledger row. We price the row from `usage`, and thinking tokens bill as output. When the stream stopped
# before its end, we price the row from the relayed chunks and mark it as an estimate. We reconcile the ledger
# against the counters of the provider. We divide shared GPUs by GPU-seconds, not by raw tokens."
#
# **Drill 1.** *After the rollout of a thinking model, the provider started to return 429s, but our request limit
# never refused a request. Why?* The bucket charged per request, but the outputs grew ten-fold with a heavy tail.
# Thus the same request rate carried many times the tokens.
#
# Reserve, stream and reconcile. Put TPM beside RPM for each tenant. Give the thinking a budget, and do not cut it
# with `max_tokens` (CURRICULUM cross-layer drill 20).
#
# **Drill 2.** *Two gateway replicas each permit the tenant 3,000 tokens per minute. What does the tenant get?* The
# tenant gets up to 6,000.
#
# Buckets shared between replicas live in Redis behind a Lua script (one round trip per check). Or the gateway
# divides them per replica and balances them again. The scaling primer §5.1 says the same about request buckets.
#
# **Drill 3.** *Why not reserve the caller's `max_completion_tokens` for every request?* The cap is the upper bound
# of the reservation. If you reserve the cap for every stream, you hold budget for the full life of the stream. At a
# 16K cap, the experiment of the core served 26.5 % of the limit (PRIMER §4.3).
#
# The default reservation is an estimate. You find its size by simulation on your own output distribution
# (gateway-core notebook 04, exercise 4.3), and here it is `default_output_estimate`. The gateway debits what the
# stream sends past it, and it reconciles at the end. Honest callers who set a tight cap get that cap reserved as
# they gave it, and they get more throughput.
#
# Reserve the full cap only where a provider 429 is unacceptable, or where the provider itself charges `max_tokens`
# at admission (PRIMER §4.2). Then reserve what you send upstream, or send what you
# reserved. This gateway forwards the reserved bound as `max_completion_tokens`.
