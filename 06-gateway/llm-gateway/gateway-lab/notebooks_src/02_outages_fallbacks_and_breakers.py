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
import os, statistics, time
from gwlab import bench, client, env, t1
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
# ## Exercise 2.1 — what the client sees
#
# The core classified statuses (gateway-core notebook 02, exercise 2.1). Over HTTP the question a client owner asks is
# different: *what do I get back?* For each fault below, switched on at `acme` (the first target of `chat`) for one
# streamed request, write `client_sees(fault)` returning `(status, served_by, error_in_stream)`: the HTTP status the
# client receives, the `x-gwlab-target` that served it (`None` if nothing did) and whether the stream carries an
# `error` event. The faults: `"503"`; `"429"`; `"timeout"` (acme stalls past its 0.4 s first-byte timeout);
# `"context"` (a prompt the gateway's chars/4 estimate says fits acme's 8,192-token window, which acme's own tokenizer
# counts as 9,000 tokens); `"midstream"` (an error chunk after 5 content chunks); `"reset"` (the connection drops
# after 5 chunks); `"bad_key"` (acme rejects the *gateway's* provider key with a 401). Predict before you run
# anything: the check provokes each fault, one at a time (breakers reset between them, since three in a row would
# open acme's — the subject of the second half), and prints what the gateway did.

# %% exercise
def client_sees(fault: str) -> tuple:
    ### BEGIN SOLUTION
    if fault in ("503", "429", "timeout", "context"):
        return 200, "bolt/haiku", False        # before the first byte: fall through
    if fault in ("midstream", "reset"):
        return 200, "acme/fast", True          # after it: the error goes into the stream, never another model's text
    if fault == "bad_key":
        return 502, None, False                # our credential, not the caller's request: no fall through, a 502
    raise ValueError(fault)
    ### END SOLUTION

# %% check
def provoke(fault):
    reset_breakers()
    prompt = "a " * 9000 if fault == "context" else f"fault {fault}"
    if fault == "bad_key":
        client.post_json(stack.url + "/admin/providers/acme/key", {"key": "revoked-key"}, token=stack.admin_token)
    elif fault != "context":
        stack.fault("acme", fault, count=1, **({"stall_s": 2} if fault == "timeout" else
                                               {"after_chunks": 5} if fault in ("midstream", "reset") else {}))
    try:
        r = stack.chat(key, prompt, stream=True)
    finally:
        client.post_json(stack.url + "/admin/providers/acme/key", {"key": "acme-key-1"}, token=stack.admin_token)
    return (r.status, r.header("x-gwlab-target"), r.error is not None and r.status == 200), stack.last_decision()

def path(dec):
    return " -> ".join(f"{a['target']}:{a['outcome']}" + (f"({a['code']})" if a.get("code") else "") for a in dec["attempts"])

for fault in ("503", "429", "timeout", "context", "midstream", "reset", "bad_key"):
    seen, dec = provoke(fault)
    assert client_sees(fault) == seen, (fault, seen, dec["attempts"])
    print(f"{fault:9s} -> {seen} | {path(dec)}")
print("✅ before the first byte a fault is invisible (another target answers); after it, an error event in a 200; "
      "the gateway's own bad key is a 502, never a fall-through")

# %% [markdown]
# ## Exercise 2.2 — availability, read from the decision log
#
# The chain formula is the core's (`routing.chain_availability`, given). What the lab adds is where its inputs come
# from in production: the gateway's own decisions. `acme` fails 25 % of requests and `bolt` 20 % (seeded, before the
# first byte); breakers are set so high they never open, so every request walks the whole chain; 240 requests go
# through. Write `failure_rate(decisions, target)`: of the attempts at `target` (an attempt skipped by an open breaker
# is not one), the share that failed (`outcome` `"fallthrough"` or `"fail"`). Then set `predicted` from the two
# measured rates with `routing.chain_availability`, and `measured` to the share of requests that got a 200.

# %%
lossy = {"acme": FakeSpec(name="acme", fail_rate=0.25, seed=3, **FAST),
         "bolt": FakeSpec.anthropic("bolt", fail_rate=0.20, seed=4, **{**FAST, "ttft_s": 0.12})}
chain = LocalStack(fakes=lossy, overrides={**overrides, "breaker": {"failure_threshold": 10**6, "recovery_timeout_s": 1}}).start()
k2 = chain.issue_key("team-a")
run = bench.run(chain.url, [bench.TenantScript("team-a", k2, rate=120, n=240)], seed=5)
decisions = chain.decisions()
print(len(decisions), "decisions; one:", decisions[0]["attempts"])

# %% exercise
def failure_rate(decisions, target):
    ### BEGIN SOLUTION
    tried = [a for d in decisions for a in d.get("attempts", []) if a["target"] == target and a["outcome"] != "breaker_open"]
    return sum(a["outcome"] in ("fallthrough", "fail") for a in tried) / len(tried)
    ### END SOLUTION

### BEGIN SOLUTION
predicted = routing.chain_availability([1 - failure_rate(decisions, "acme/fast"), 1 - failure_rate(decisions, "bolt/haiku")])
measured = sum(d.get("status") == 200 for d in decisions) / len(decisions)
### END SOLUTION

# %% check
fa, fb = failure_rate(decisions, "acme/fast"), failure_rate(decisions, "bolt/haiku")
assert abs(fa - 0.25) < 0.08 and abs(fb - 0.20) < 0.10, (fa, fb)
assert abs(measured - sum(x.status == 200 for x in run.samples) / len(run.samples)) < 1e-9
assert abs(predicted - (1 - fa * fb)) < 1e-9 and abs(measured - predicted) < 0.05
print(f"[SIMULATED] acme failed {fa:.1%} of its attempts, bolt {fb:.1%}: predicted {predicted:.3f}, measured {measured:.3f} "
      f"(acme alone: {1 - fa:.2f})")
print("✅ two lossy targets in independent failure domains; put both behind one common failure and (1 − c) caps it")

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
# second. The next cell sends two requests a second for 20 s and — because Colab and Kaggle give you no second
# terminal — stops vLLM itself 8 s in (`gwlab.t1.stop_vllm`: `pkill -f "vllm serve"`, else
# `docker compose ... stop vllm`), behind an explicit opt-in (`GWLAB_T1_STOP=1`) so re-running the notebook never
# kills your engine by surprise. The served-by line switches from `v` to `a` after a few connection errors, and the
# breaker then skips vLLM. It prints how to restart vLLM: notebooks 03 and 04 need it again (if you forget, their
# T1 cells find no vLLM on `/health` and skip).

# %%
if tiers["vllm_url"] and os.environ.get("GWLAB_T1_STOP") == "1":
    res = t1.stop_midrun(tiers["vllm_url"], t1.stop_vllm, fallback=FakeSpec(name="acme", **FAST), rate=2, n=40,
                         stop_after_s=8.0)
    print("served by, in send order (v = vLLM: MEASURED; a = the fake fallback: SIMULATED):", res["line"])
    print(f"requests that reached the stopped vLLM: {res['reached_dead']}; breaker opens: {res['breaker_opens']}")
    print(res["run"].table())
    assert res["line"].startswith("v"), "vLLM served nothing before the stop"
    assert res["line"].endswith("a") and res["breaker_opens"] >= 1, "vLLM was not stopped, or nothing fell back"
    print(t1.RESTART_HINT)
elif tiers["vllm_url"]:
    print("T1 ready. This cell stops vLLM itself (pkill -f 'vllm serve', or docker compose stop vllm) 8 s into a "
          "20 s run: set GWLAB_T1_STOP=1 and re-run it. Afterwards: " + t1.RESTART_HINT)
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
