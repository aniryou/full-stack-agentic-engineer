# %% [markdown]
# # 02 · Outages, fallbacks and breakers: what falls through, and what a chain buys
#
# **Tier:** T0. Fake providers with scripted faults run behind the real gateway over localhost HTTP. The faults are a
# 503, a 429, a stall before the first byte, and an error or a reset in the middle of the stream. The timings are
# **simulated**. T1: with vLLM as the first target (`GWLAB_VLLM_URL`), stop vLLM during the run and watch the chain move
# to the fallback.
#
# ## The one-minute version
#
# A client gives an alias. The gateway tries an ordered chain of targets (PRIMER §2). Three rules make sure that a chain
# does not make outages worse:
#
# 1. **Only some failures fall through**: 429, 5xx (and Anthropic's 529), timeouts, connection errors and
#    context-length errors. A 400, a provider auth failure or a content-policy refusal fails the request. A provider
#    auth failure is a failure of our credential, not of the credential of the caller. In these cases, the next
#    provider also refuses the request, or the fault is ours and we must repair it.
# 2. **Fall back only before the first byte.** The gateway holds the first chunk until it knows that the upstream
#    produces tokens. After a token reaches the client, the gateway reports a failure *in the stream*. To join the
#    continuation of another model to half an answer is incorrect.
# 3. **A breaker per target** changes repeated failures into instant skips for a cool-down period. This lab uses the
#    same rules as 07.2's `CircuitBreaker` (gcp-agent-platform-lab notebook 10, the canonical home). The breaker opens
#    after 3 *consecutive* failures. It goes half-open after the recovery timeout, and then one probe decides. Retries
#    with jitter are in §5.2 of the scaling primer. Here, the retry is the next target.
#
# Chain arithmetic: with independent failures, $A = 1 - \prod_i (1 - a_i)$. A common-mode event multiplies it by
# $(1 - c)$. Examples of a common-mode event are an outage of the region of the gateway itself and an outage of a
# shared upstream. Thus a fallback in the same failure domain adds only a small gain.
#
# The fallback also has costs. One cost is latency: the failed attempt, plus a target that is possibly slower. The other
# cost is money: a model with a higher price at full traffic, until that model reaches the end of its own quota.

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
# The core classified statuses (gateway-core notebook 02, exercise 2.1). Over HTTP, the owner of a client asks a
# different question: *what do I get back?* The check switches on each fault below at `acme` (the first target of
# `chat`) for one streamed request. Write `client_sees(fault)`. It returns `(status, served_by, error_in_stream)`:
#
# - the HTTP status that the client receives,
# - the `x-gwlab-target` that served the request (`None` if no target did),
# - a flag that shows if the stream has an `error` event.
#
# The faults:
#
# - `"503"`.
# - `"429"`.
# - `"timeout"`: acme stalls for longer than its 0.4 s first-byte timeout.
# - `"context"`: a prompt that fits the 8,192-token window of acme by the chars/4 estimate of the gateway. The
#   tokenizer of acme itself counts this prompt as 9,000 tokens.
# - `"midstream"`: an error chunk after 5 content chunks.
# - `"reset"`: the connection stops after 5 chunks.
# - `"bad_key"`: acme rejects the *gateway's* provider key with a 401.
#
# Predict the results before you run anything. The check causes each fault, one at a time, and prints what the gateway
# did. The breakers reset between the faults, because three faults in sequence are sufficient to open the breaker of acme. That breaker
# is the subject of the second half.

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
# The chain formula comes from the core (`routing.chain_availability`, given). The lab adds the source of its inputs in
# production: the decisions of the gateway itself.
#
# In this run, `acme` fails 25 % of the requests and `bolt` fails 20 %. The failures use a seed and occur before the
# first byte. The breaker settings are so high that the breakers never open. Thus every request goes through the whole
# chain. 240 requests go through.
#
# Write `failure_rate(decisions, target)`. Return the share of the attempts at `target` that failed (`outcome`
# `"fallthrough"` or `"fail"`). An attempt that an open breaker skipped is not an attempt. Then set `predicted` from the
# two measured rates with `routing.chain_availability`. Set `measured` to the share of requests that got a 200.

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
# In the same run, each request that succeeded had one of two paths. It got `acme` immediately (TTFT ≈ acme's), or it
# paid for a failed attempt and then got `bolt`.
# Write `expected_ttft(p_fail, fail_s, ttft_primary, ttft_fallback, p_fail_fallback)`. Return the mean TTFT of
# the requests that succeeded: $(1 - p)\,t_1 + p\,(1 - p_2)\,(f + t_2)$, divided by the probability of success.
#
# The check uses the components *measured* in the run. The TTFT of each target comes from the requests that this target
# served. The duration of the failed attempt comes from the decision log of the gateway. Then the check compares the
# result with the measured mean.

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
# `acme` has an outage for 4 s while one tenant sends 10 requests a second for approximately 8 s. For the first three
# requests, the chain pays for a failed attempt. Here this attempt is fast: a 503. Then the breaker opens, and the
# requests skip `acme` and do not touch it.
#
# At each recovery timeout (1 s), the breaker lets one probe through. The probe fails and opens the breaker again. This
# continues until the outage ends. Then a probe succeeds, and the traffic goes back to `acme`. With a *stall* instead of
# a 503, each of those attempts costs the 0.4 s first-byte timeout. That is the real value of the breaker.

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
# Predict the count. These are the conditions:
#
# - Requests arrive at `rate` per second.
# - The target is dead for `outage_s`.
# - The breaker opens after `threshold` consecutive failures. While it is open, it lets one probe through every
#   `recovery_s`.
#
# Write `requests_reaching_dead_target(outage_s, rate, threshold, recovery_s)`. Return the `threshold` failures that
# open the breaker, plus one failed probe for each full `recovery_s` in the rest of the outage. The `threshold`
# failures take $\text{threshold} / \text{rate}$ seconds. The check compares the result with the run in the previous
# worked example. It uses a small tolerance, because arrivals are Poisson and the fault window does not align with them.

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
# When you set `GWLAB_VLLM_URL`, a second gateway puts the real vLLM first in `chat` (config `vllm`) and a fake second.
# The next cell sends two requests a second for 20 s. After 8 s, the cell itself stops vLLM, because Colab and Kaggle
# give you no second terminal. It uses `gwlab.t1.stop_vllm`: `pkill -f "vllm serve"`, else `docker compose ... stop
# vllm`. This step needs an explicit opt-in (`GWLAB_T1_STOP=1`). Thus, if you run the notebook again, it never stops
# your engine by surprise.
#
# The served-by line changes from `v` to `a` after a few connection errors. Then the breaker skips vLLM. The cell
# prints how to start vLLM again, because notebooks 03 and 04 need it again. If you forget, their T1 cells find no vLLM
# on `/health` and skip.

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
# **Two minutes:** "Each alias is an ordered chain of targets in independent failure domains: another provider,
# another region, or our own pool. A request falls through on 429, 5xx, timeouts and context-length errors. It never
# falls through on a 400, an auth failure or a content-policy refusal. The gateway holds the first chunk until the
# upstream produces tokens. Thus it can fall back before the first byte. After the first byte, the error goes into the
# stream, and the client decides.
#
# "Every target has a breaker. Three consecutive failures open it, and it lets one probe through in each recovery
# window. Thus an outage costs a small number of failed attempts, not one for each request. Two independent 99 %
# targets give 99.99 %. A common-mode failure puts a limit on that. Thus the fallback is in another region or provider,
# and we plan its capacity and price before the day that we need it."
#
# **Drill 1.** *A provider outage lasted five minutes; our incident lasted forty and the bill doubled. Why?* Retries
# without a budget continued after the outage ended. Streams that failed in the middle ran again from the start and
# paid two times. The chain fell through to a model with a higher price at full traffic, until that model reached the
# end of its quota.
#
# Use retry budgets with jitter and a breaker per target. Fall back only before the first byte. Set
# the size and the price of the fallback capacity in advance, and put a cost alert on each tenant (CURRICULUM
# cross-layer drill 18).
#
# **Drill 2.** *Why not retry a stream on another model after it failed at token 300?* The client already showed 300
# tokens from model A. The continuation of A's text by B is not the answer of A. Also, B generates again from the start,
# and bills again from the start.
#
# Report the error. Let the client retry the whole request if it wants.
#
# **Drill 3.** *Our fallback is the same model in another zone of the same region. What does the chain buy?* The chain
# gives protection from zonal failures only. A regional outage, a provider-wide incident or an incorrect deploy is
# common-mode. Thus $(1 - c)$ is the limit on availability, for any number of zonal replicas in the chain.
