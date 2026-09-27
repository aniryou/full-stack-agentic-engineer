# %% [markdown]
# # 04 · Streaming limits, metering and chargeback: count tokens, not requests
#
# **Tier:** T0 — tenants driven over localhost HTTP against fake providers with their own tokens-per-minute
# limits; the rate-limit "minute" is **compressed to 2 seconds** so each run takes seconds, and all timings and
# token counts are **simulated**. T1: the ledger reconciled with a real vLLM's `usage` and `/metrics`.
#
# ## The one-minute version
#
# The scaling primer's token bucket (§5.1) and admission control (§5.3) decide whether a request may start. For
# LLM calls the cost of a request is unknown when it starts — the output length is decided while it streams, and
# it is heavy-tailed (00.5 PRIMER §7: a thinking model's outputs are an order of magnitude longer). A bucket that
# charges each request a fixed guess up front **over-admits** by `(prompt + actual output) / (prompt + guess)`,
# and the provider's own TPM limit then does the refusing, with 429s the tenant cannot explain (PRIMER §4). The
# fix is to meter tokens where they flow:
#
#     admit      reserve prompt estimate + min(requested cap or a default, a hard cap) from the tenant's TPM
#     stream     debit anything beyond the reservation as chunks arrive
#     reconcile  refund (reservation − actual) from the provider's `usage` at the end
#
# with RPM beside TPM, per tenant (so a noisy neighbour exhausts only its own bucket) under a gateway-wide share.
# The same `usage` then prices the request in the **ledger** — `usage` is authoritative; an estimate covers only
# admission and streams cut before the usage chunk (PRIMER §5) — and the ledger answers chargeback, including
# for a shared self-hosted GPU (01 PRIMER §8.1).

# %%
import statistics, time
from gwlab import bench, client, env, promtext, report
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
# One tenant sends 30 requests a second for 6 seconds through the alias `solo` (acme only, so the provider's
# refusals are visible). The provider counts tokens over a sliding window of one "minute" (2 s); the gateway's
# bucket holds a burst (its capacity) and refills continuously, so over any window it admits up to
# `capacity + rate × window`. Sizing the tenant's bucket at 1,500 tokens per minute keeps that inside the
# provider's limit: `1,500 + 750/s × 2 s = 3,000`. (A bucket sized at the provider's full 3,000 would admit up to
# 6,000 in the first window: the burst is part of the budget.) The core's `ReserveLimiter` counts a sliding window
# instead — admit only if `used + reserved + reserve ≤ limit` (PRIMER §4.2) — which lines up with a provider's
# window by construction; this lab keeps the scaling lab's token bucket (06.3) and needs the sizing rule above.
# The per-request bucket charges `prompt + 20` per request; the reserving bucket charges `prompt + 200` up front
# and settles to the real count at the end.

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
# Same demand, same provider. The per-request bucket admits almost everything and the provider refuses what its
# TPM cannot carry: those 429s arrive after a round trip, three in a row open acme's breaker, and then *every*
# request is refused with a 503 until a probe gets through — including requests well inside the tenant's budget.
# With a fallback chain they would instead spill onto a pricier provider at full traffic. The reserving bucket
# refuses at the gateway's door, with a `Retry-After` and `x-ratelimit-*` headers that say which limit and when,
# and the provider sees a load it can carry.

# %% [markdown]
# ## Exercise 4.1 — the reservation
#
# Write `reservation(prompt_estimate, requested_output, default_estimate, hard_cap)`: the tokens `reserve` mode
# takes from the tenant's TPM at admission — the prompt estimate plus the output cap the caller asked for (or the
# default estimate when it asked for none), never more than the hard cap for the output part. The check compares
# with the gateway's limiter over a grid.

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
# Write `over_admission(prompt, mean_output, charged_output)`: the ratio of tokens really consumed to tokens the
# bucket charged, when each request is charged `prompt + charged_output` but costs `prompt + mean_output`. The
# check takes the per-request run above: the mean prompt the gateway estimated, the mean output from the ledger,
# the 20-token guess — and compares with the measured ratio of all tokens served to all tokens charged.

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
# Two tenants share a provider allowing 6,000 tokens per minute; the gateway's global bucket is sized to it
# (3,000 per minute: burst plus refill over one window is 6,000). `team-a` floods at several times its share;
# `team-b` sends a steady trickle. With the global bucket only, team-a drains it and team-b's requests are
# refused too. With a bucket per tenant (1,500 each) under the same global one, team-a is refused alone.

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
# A client that walks away mid-stream never receives the usage chunk (OpenAI's spec says so). The gateway
# stops the upstream and writes the row from what it relayed — one token per content chunk, marked as an
# estimate. Billing it as 0 would make "disconnect just before the end" a free lunch.

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
# price row `p` (`p.input`, `p.cached`, `p.output`, dollars per 1M tokens): uncached prompt tokens at the input
# rate, cached ones at the cached rate, completion tokens — reasoning included — at the output rate. The check
# prices every row of the runs above with the model that served it and compares with the ledger, then the
# scaling primer's §3.4 call (5,000 in, 2,700 cached, 350 out on `gemini-3.5-flash`: $0.007005) and the same call
# reported in Anthropic's shape (normalise it first).

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
assert abs(price({"prompt_tokens": an.prompt_tokens, "cached_tokens": an.cached_tokens, "completion_tokens": an.completion_tokens},
                 metering.PRICES["claude-haiku-4-5"]) - 0.00432) < 1e-12
print(f"✅ {n} ledger rows re-priced exactly; the §3.4 call is $0.007005 on gemini-3.5-flash, $0.00432 on claude-haiku-4-5 "
      "(prices dated 2026-09-26, verify)")

# %% [markdown]
# ## Worked example: $ per 1M tokens, hosted and self-hosted
#
# The same call shape blended over its 5,350 tokens, per hosted row (verify, 2026-09-26), and self-hosted rows
# from a GPU's hourly price and a throughput — 01 PRIMER §8.1's Llama-3.1-8B numbers (H100 Spot at $3.7/h and
# 6,846.5 tokens/s; an L4 at $0.70/h and 294.39 tokens/s), at 100 % and 60 % utilisation.

# %%
for name, p in metering.PRICES.items():
    print(f"{name:22s} ${metering.blended_per_million(p, 5000, 2700, 350):.4f} per 1M (this call shape)")
for name, gpu_h, tps in (("H100 Spot, self-hosted", 3.7, 6846.5), ("L4, self-hosted", 0.70, 294.39)):
    print(f"{name:22s} ${metering.cost_per_million_tokens(gpu_h, tps):.4f} per 1M output tokens at 100 %, "
          f"${metering.cost_per_million_tokens(gpu_h, tps, 0.6):.4f} at 60 %")

# %% [markdown]
# ## Exercise 4.4 — chargeback for a shared GPU
#
# A self-hosted pool costs the same per hour whoever uses it; the question is how to split the bill. By tokens,
# every token weighs the same. By GPU-seconds, a prompt token (prefill: batched, compute-bound) weighs far less
# than an output token (decode: one step per token). Write `chargeback(rows, total_usd, by, prefill_s, decode_s)`
# returning `{tenant: dollars}`; `by` is `"tokens"` or `"gpu_seconds"` (weight = prompt × prefill_s +
# completion × decode_s). The check runs a RAG tenant (long prompts, short answers) and a chat tenant (short
# prompts, long answers) through the gateway and compares with the lab's `metering.chargeback`.

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
# The provider keeps its own counters (the fakes expose vLLM's names). Compare them with the ledger. On the
# per-request run every served row carries the provider's `usage`, and the two agree to the token. On the reserve
# run, after the walk-away tenant, the gaps come only from the rows billed from an estimate: their prompt count is
# the gateway's chars/4 estimate (no usage chunk ever arrived) and their completion count is the chunks relayed,
# while the provider counted what it tokenized and generated before it noticed the disconnect. That is the
# reconciliation item you chase each month, not a rounding error.

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
# With `GWLAB_VLLM_URL` set: scrape vLLM's `/metrics`, send twenty requests through a gateway whose first target is
# vLLM, scrape again, and compare the ledger with the counters' differences.

# %%
for st, _, _ in runs.values():
    st.stop()
if tiers["vllm_url"]:
    url = tiers["vllm_url"].rstrip("/")
    before = promtext.parse(client.get_text(url + "/metrics"))
    t1 = LocalStack(config="vllm", upstreams={"local": url}, fakes={"acme": FakeSpec(name="acme")}).start()
    k1 = t1.issue_key("team-a")
    bench.run(t1.url, [bench.TenantScript("team-a", k1, rate=4, n=20, max_tokens=64)], seed=1)
    after = promtext.parse(client.get_text(url + "/metrics"))
    delta = {k: promtext.total(after, k) - promtext.total(before, k) for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")}
    rec = metering.reconcile(t1.ledger(), {"local": {"prompt_tokens": delta["vllm:prompt_tokens_total"],
                                                     "generation_tokens": delta["vllm:generation_tokens_total"]}})
    print("MEASURED\n" + report.reconcile_table(rec))
    t1.stop()
else:
    print("T1 skipped: deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We limit tokens, not only requests, because a request's cost is decided while it streams.
# At admission we reserve the prompt estimate plus the caller's output cap — or a default — from the tenant's
# tokens-per-minute bucket, next to its requests-per-minute bucket, under a gateway-wide share of the provider's
# quota; we debit anything that streams past the reservation and reconcile to the provider's `usage` at the
# end, and a hard cap bounds any one response. Refusals happen at our door with a Retry-After, not at the
# provider's after a round trip. Every request writes a ledger row priced from `usage` — thinking tokens bill as
# output — or from the relayed chunks when the stream was cut, flagged as an estimate; we reconcile the ledger
# against the provider's counters, and we split shared GPUs by GPU-seconds, not raw tokens."
#
# **Drill 1.** *After a thinking-model rollout the provider started returning 429s but our request limit never
# tripped. Why?* — The bucket charged per request while outputs grew ten-fold with a heavy tail, so the same
# request rate carried many times the tokens. Reserve, stream, reconcile; TPM beside RPM per tenant; budget
# thinking rather than truncating it with `max_tokens` (CURRICULUM cross-layer drill 18).
#
# **Drill 2.** *Two gateway replicas each allow the tenant 3,000 tokens per minute. What does the tenant get?* —
# Up to 6,000. Buckets shared between replicas live in Redis behind a Lua script (one round trip per check) or
# are split per replica and re-balanced; the scaling primer §5.1 says the same about request buckets.
#
# **Drill 3.** *Why reserve the caller's `max_completion_tokens` instead of an average?* — It is the only bound
# known at admission; reserving the average admits too much when the tail arrives. Reconciliation returns the
# unused part within the same request, so honest callers who set a tight cap get more throughput.
