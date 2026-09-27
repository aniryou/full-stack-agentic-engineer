# %% [markdown]
# # 02 · Outages, fallbacks and breakers: what falls through, and what a chain buys
#
# **Tier:** T0 — fake providers with scripted faults (503, 429, a stall before the first byte, an error or a
# reset mid-stream) behind the real gateway over localhost HTTP; timings are **simulated**. T1: with vLLM as the
# first target (`GWLAB_VLLM_URL`), stop it mid-run and watch the chain move to the fallback.
#
# ## The one-minute version
#
# A client names an alias; the gateway tries an ordered chain of targets (PRIMER §2). Three rules keep a chain
# from making outages worse:
#
# 1. **Only some failures fall through**: 429, 5xx (and Anthropic's 529), timeouts, connection errors and
#    context-length errors. A 400, a provider auth failure (our credential, not the caller's) or a content-policy
#    refusal fails the request: the next provider would refuse it too, or the fault is ours to fix.
# 2. **Fall back only before the first byte.** The gateway holds the first chunk until it knows the upstream is
#    producing; once a token has reached the client, a failure is surfaced *in the stream* — splicing another
#    model's continuation onto a half-answer is wrong.
# 3. **A breaker per target** turns repeated failures into instant skips for a cooling period. This lab follows
#    07.2's `CircuitBreaker` (gcp-agent-platform-lab notebook 10, the canonical home): open after 3 *consecutive*
#    failures, half-open after the recovery timeout, one probe decides. Retries with jitter are the scaling
#    primer's §5.2; here the retry is the next target.
#
# Chain arithmetic: with independent failures `A = 1 − Π(1 − aᵢ)`; a common-mode event (the gateway's own
# region, a shared upstream) multiplies it by `(1 − c)` — a fallback in the same failure domain buys little.
# The fallback also costs: latency (the failed attempt, plus a possibly slower target) and money (a pricier
# model at full traffic until its own quota runs out).

# %%
import statistics, time
from gwlab import bench, client, env
from gwlab.fakes import FakeSpec
from gwlab.gateway import routing
from gwlab.stack import LocalStack

tiers = env.describe()
FAST = dict(ttft_s=0.02, prefill_s_per_token=0.0, itl_s=0.003, output_tokens=16)
fakes = {"acme": FakeSpec(name="acme", **FAST), "bolt": FakeSpec.anthropic("bolt", **{**FAST, "ttft_s": 0.12})}
overrides = {"providers": {"acme": {"first_byte_timeout_s": 0.4}, "bolt": {"first_byte_timeout_s": 2.0}},
             "breaker": {"failure_threshold": 3, "recovery_timeout_s": 1.0}}
stack = LocalStack(fakes=fakes, overrides=overrides).start()
key = stack.issue_key("team-a")


def reset_breakers(s=None):
    """Close every breaker (on the gateway's loop thread), so the next example starts clean."""
    s = s or stack
    s.call(lambda: [b.record(True) for b in s.gateway.router.breakers.values()])


print("chain for alias chat:", stack.cfg.aliases["chat"].targets, "| bolt is slower on purpose: TTFT 120 ms vs 20 ms (simulated)")

# %% [markdown]
# ## Worked example: one fault of each kind
#
# Each row switches on one fault at `acme` for the next request and shows what the gateway did. Before the first
# byte (503, 429, a stall past the 0.4 s first-byte timeout) the request falls through to `bolt`. After it
# (an error chunk mid-stream, a dropped connection) the client gets the partial answer and an `error` event.
# The breakers are reset between rows so that each row shows one fault on its own (three in a row would open
# acme's breaker — the subject of the second half).

# %%
def once(mode, **kw):
    reset_breakers()
    stack.fault("acme", mode, count=1, **kw)
    r = stack.chat(key, f"fault {mode}", stream=True)
    d = stack.last_decision()
    atts = " -> ".join(f"{a['target']}:{a['outcome']}" + (f"({a.get('code')})" if a.get("code") else "") for a in d["attempts"])
    return r, f"{mode:10s} status {r.status}  served by {r.header('x-gwlab-target')}  text chunks {sum(1 for c in r.chunks if c.get('choices') and c['choices'][0].get('delta', {}).get('content'))}  error {(r.error or {}).get('code')}  | {atts}"

for mode, kw in (("503", {}), ("429", {}), ("timeout", {"stall_s": 2}), ("midstream", {"after_chunks": 5}), ("reset", {"after_chunks": 5})):
    print(once(mode, **kw)[1])

# %% [markdown]
# ## Exercise 2.1 — what falls through
#
# Write `classify(status, code=None, timeout=False)` returning `"fallthrough"` or `"fail"` by the rules above
# (`timeout=True` covers timeouts and connection errors; `code` is the normalised error code, a string such as
# `"context_length_exceeded"` or `"content_filter"`). The check compares with the gateway's `routing.classify`
# on a table, then provokes two live cases: a 503 (must fall through) and a provider rejecting the gateway's
# own key with a 401 (must not — the client gets a 502, because the credential problem is the operator's).

# %% exercise
def classify(status, code=None, timeout=False) -> str:
    ### BEGIN SOLUTION
    if timeout or code == "context_length_exceeded":
        return "fallthrough"
    if code == "content_filter":
        return "fail"
    return "fallthrough" if status in (408, 429, 500, 502, 503, 504, 529) else "fail"
    ### END SOLUTION

# %% check
cases = [(429, None, False), (500, None, False), (502, None, False), (503, None, False), (529, None, False), (None, None, True),
         (400, "context_length_exceeded", False), (400, None, False), (401, None, False), (403, None, False),
         (404, "model_not_found", False), (400, "content_filter", False), (500, "content_filter", False), (422, None, False)]
for s, c, t in cases:
    assert classify(s, c, t) == routing.classify(s, c, timeout=t), (s, c, t)
reset_breakers()
stack.fault("acme", "503", count=1)
stack.chat(key, "live 503", stream=True)
assert stack.last_decision()["attempts"][0]["outcome"] == classify(503)
client.post_json(stack.url + "/admin/providers/acme/key", {"key": "revoked-key"}, token=stack.admin_token)
r = stack.chat(key, "live 401")
assert r.status == 502 and stack.last_decision()["attempts"][0]["status"] == 401 and classify(401) == "fail"
client.post_json(stack.url + "/admin/providers/acme/key", {"key": "acme-key-1"}, token=stack.admin_token)
print("✅ 429/5xx/timeouts/context-length fall through; 400/401/403/content-policy fail the request (a 401 upstream is a 502 here)")

# %% [markdown]
# ## Exercise 2.2 — chain availability, predicted and measured
#
# Write `chain_availability(availabilities, common_mode=0.0)`: the probability that some target in the chain
# answers when target *i* is up with probability `aᵢ` independently, and a common-mode event that takes every
# target down happens with probability `common_mode`. Then the check measures it: `acme` fails 25 % of requests
# and `bolt` 20 % (seeded, before the first byte), breakers are set so high they never open (so every request
# walks the whole chain), and 240 requests go through. Predicted `1 − 0.25 × 0.20 = 0.95`; one-target
# availability would be 0.75.

# %% exercise
def chain_availability(availabilities, common_mode: float = 0.0) -> float:
    ### BEGIN SOLUTION
    p = 1.0
    for a in availabilities:
        p *= 1.0 - a
    return (1.0 - common_mode) * (1.0 - p)
    ### END SOLUTION

# %% check
assert abs(chain_availability([0.99, 0.99]) - 0.9999) < 1e-12 and abs(chain_availability([0.99, 0.99], 0.001) - 0.999 * 0.9999) < 1e-12
for avs, c in (([0.9], 0), ([0.9, 0.8, 0.7], 0.01), ([0.999, 0.99], 0.0005)):
    assert abs(chain_availability(avs, c) - routing.chain_availability(avs, c)) < 1e-12
lossy = {"acme": FakeSpec(name="acme", fail_rate=0.25, seed=3, **FAST),
         "bolt": FakeSpec.anthropic("bolt", fail_rate=0.20, seed=4, **{**FAST, "ttft_s": 0.12})}
chain = LocalStack(fakes=lossy, overrides={**overrides, "breaker": {"failure_threshold": 10**6, "recovery_timeout_s": 1}}).start()
k2 = chain.issue_key("team-a")
run = bench.run(chain.url, [bench.TenantScript("team-a", k2, rate=120, n=240)], seed=5)
measured = sum(s.status == 200 for s in run.samples) / len(run.samples)
predicted = chain_availability([0.75, 0.80])
print(f"[SIMULATED] measured {measured:.3f} vs predicted {predicted:.3f} (acme alone: 0.75)")
assert abs(measured - predicted) < 0.05
print("✅ two lossy targets in independent failure domains: 0.95; put both behind one common failure and (1 − c) caps it")

# %% [markdown]
# ## Exercise 2.3 — what a fallback costs in latency
#
# On the same run, a request either got `acme` at once (TTFT ≈ acme's), or paid a failed attempt and then got
# `bolt`. Write `expected_ttft(p_fail, fail_s, ttft_primary, ttft_fallback, p_fail_fallback)`: the mean TTFT of
# the requests that succeeded — `(1 − p)·t₁ + p·(1 − p₂)·(f + t₂)`, divided by the probability of success.
# The check fills in the components *measured* on the run (each target's TTFT from the requests it served, the
# failed attempt's duration from the gateway's decision log) and compares with the measured mean.

# %% exercise
def expected_ttft(p_fail, fail_s, ttft_primary, ttft_fallback, p_fail_fallback=0.0) -> float:
    ### BEGIN SOLUTION
    ok_primary = 1 - p_fail
    ok_fallback = p_fail * (1 - p_fail_fallback)
    return (ok_primary * ttft_primary + ok_fallback * (fail_s + ttft_fallback)) / (ok_primary + ok_fallback)
    ### END SOLUTION

# %% check
ok = [s for s in run.samples if s.status == 200 and s.ttft_s]
first = [s.ttft_s for s in ok if s.target == "acme/fast"]
second = [s.ttft_s for s in ok if s.target == "bolt/haiku"]
fails = [a["ms"] / 1e3 for d in chain.decisions() for a in d.get("attempts", []) if a["target"] == "acme/fast" and a["outcome"] == "fallthrough"]
direct_bolt = statistics.mean(second) - statistics.mean(fails)       # bolt's own TTFT, without the failed attempt
pred = expected_ttft(0.25, statistics.mean(fails), statistics.mean(first), direct_bolt, 0.20)
meas = statistics.mean(s.ttft_s for s in ok)
e = routing.expected_chain([(0.25, statistics.mean(fails), statistics.mean(first), 0), (0.20, 0, direct_bolt, 0)])
assert abs(pred - e["ttft_s"]) < 1e-9
print(f"[SIMULATED] TTFT: acme {statistics.mean(first) * 1e3:.0f} ms, via fallback {statistics.mean(second) * 1e3:.0f} ms; "
      f"mean measured {meas * 1e3:.0f} ms vs predicted {pred * 1e3:.0f} ms")
assert abs(pred / meas - 1) < 0.2
chain.stop()
print("✅ the chain's mean TTFT is a mixture: most requests at the primary's speed, p of them at failure + fallback")

# %% [markdown]
# ## Worked example: a breaker through an outage
#
# `acme` goes down for 4 s while one tenant sends 10 requests a second for about 8 s. For the first three
# requests the chain pays a failed attempt (fast here: a 503); then the breaker opens and requests skip `acme`
# without touching it; every recovery timeout (1 s) one probe is let through, fails, and re-opens it — until the
# outage is over, a probe succeeds, and traffic returns to `acme`. With a *stall* instead of a 503, every one of those attempts would cost the 0.4 s first-byte
# timeout: that is the breaker's real value.

# %%
reset_breakers()
n_before = len(stack.decisions())
stack.fault("acme", "503", for_s=4.0)
t0 = time.monotonic()
outage = bench.run(stack.url, [bench.TenantScript("team-a", key, rate=10, n=80)], seed=9)
decs = stack.decisions()[n_before:]
line = "".join({"acme/fast": "a", "bolt/haiku": "b"}.get(d.get("served_by"), "x") for d in decs)
reached = sum(1 for d in decs for a in d["attempts"] if a["target"] == "acme/fast" and a["outcome"] != "breaker_open")
dead = sum(1 for d in decs for a in d["attempts"] if a["target"] == "acme/fast" and a["outcome"] == "fallthrough")
print("served by, in order of completion (a = acme, b = bolt):", line)
print(f"requests that reached acme while it was down: {dead}; skipped by the open breaker: "
      f"{sum(1 for d in decs for a in d['attempts'] if a['outcome'] == 'breaker_open')}; breaker opens: {stack.gateway.router.breakers['acme/fast'].opens}")

# %% [markdown]
# ## Exercise 2.4 — how many requests hit a dead target
#
# Predict it: requests arrive at `rate` per second; the target is dead for `outage_s`; the breaker opens after
# `threshold` consecutive failures and lets one probe through every `recovery_s` while open. Write
# `requests_reaching_dead_target(outage_s, rate, threshold, recovery_s)`: the `threshold` failures that open it
# (they take `threshold / rate` seconds), plus one failed probe per full `recovery_s` in the rest of the outage.
# The check compares with the run above (a small tolerance: arrivals are Poisson and the fault window is
# not aligned with them).

# %% exercise
def requests_reaching_dead_target(outage_s, rate, threshold, recovery_s) -> int:
    ### BEGIN SOLUTION
    t_open = threshold / rate
    if outage_s <= t_open:
        return int(round(outage_s * rate))
    return threshold + int((outage_s - t_open) // recovery_s)
    ### END SOLUTION

# %% check
assert requests_reaching_dead_target(10, 10, 3, 2) == 3 + 4 and requests_reaching_dead_target(0.2, 10, 3, 1) == 2
pred = requests_reaching_dead_target(4.0, 10, 3, 1.0)
print(f"[SIMULATED] predicted {pred}, measured {dead} of ~40 requests during the outage")
assert abs(dead - pred) <= 2
print("✅ without the breaker all ~40 would have paid a failed attempt; with it,", dead)

# %% [markdown]
# ## T1: stop vLLM mid-run
#
# With `GWLAB_VLLM_URL` set, a second gateway puts the real vLLM first in `chat` (config `vllm`) and a fake
# second. The next cell sends a steady stream for 40 s: stop vLLM in a terminal while it runs
# (`pkill -f "vllm serve"`, or `docker compose -f deploy/any-gpu/docker-compose.yaml stop vllm`) and watch the
# served-by line switch from `v` to `a` after a few connection errors, with the breaker then skipping vLLM.

# %%
if tiers["vllm_url"]:
    t1 = LocalStack(config="vllm", upstreams={"local": tiers["vllm_url"]}, fakes={"acme": FakeSpec(name="acme", **FAST)},
                    overrides={"breaker": {"failure_threshold": 3, "recovery_timeout_s": 5}}).start()
    k1 = t1.issue_key("team-a")
    print("sending 2 requests/s for 40 s -- stop vLLM now in another terminal")
    r1 = bench.run(t1.url, [bench.TenantScript("team-a", k1, rate=2, n=80)], seed=1)
    print("".join({"local/llm": "v", "acme/fast": "a"}.get(s.target, "x") for s in sorted(r1.samples, key=lambda s: s.t_send)))
    print(r1.table())
    t1.stop()
else:
    print("T1 skipped: deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000 and re-run")

# %%
stack.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Each alias is an ordered chain of targets in independent failure domains — another
# provider, another region, or our own pool. A request falls through on 429, 5xx, timeouts and context-length
# errors, never on a 400, an auth failure or a content-policy refusal. The gateway holds the first chunk until
# the upstream is producing, so it can fall back before the first byte; after it, the error goes into the stream
# and the client decides. Every target has a breaker — three consecutive failures open it, one probe per
# recovery window — so an outage costs a handful of failed attempts, not one per request. Two independent 99 %
# targets give 99.99 %; a common-mode failure caps that, so the fallback lives in another region or provider,
# and its capacity and price are planned before the day we need it."
#
# **Drill 1.** *A provider outage lasted five minutes; our incident lasted forty and the bill doubled. Why?* —
# Retries without a budget outlived the outage; streams that failed midway were re-run from the start and paid
# twice; the chain fell through to a pricier model at full traffic until its quota ran out. Retry budgets with
# jitter, a breaker per target, fallback only before the first byte, fallback capacity sized and priced ahead,
# and a cost alert per tenant (CURRICULUM cross-layer drill 16).
#
# **Drill 2.** *Why not retry a stream on another model after it failed at token 300?* — The client has already
# shown 300 tokens from model A; B's continuation of A's text is not A's answer, and B re-generates (and bills)
# from the start. Surface the error; let the client retry the whole request if it wants.
#
# **Drill 3.** *Our fallback is the same model in another zone of the same region. What does the chain buy?* —
# Protection from zonal failures only; a regional outage, a provider-wide incident or a bad deploy is
# common-mode, so `(1 − c)` bounds availability no matter how many zonal replicas are in the chain.
