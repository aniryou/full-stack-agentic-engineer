# %% [markdown]
# # 04 · Serving thinking models: long outputs, KV pressure, ITL, budgets and the prefix cache
#
# **Tier:** T0 (default) — the fake vLLM and the engine emulator; every latency, KV and preemption
# number is **simulated** from a roofline step model of Qwen3-0.6B on a T4. T1 — set `THINKLAB_URL`
# to a real `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --enable-prompt-tokens-details`, and
# the live cells measure ITL, KV usage, preemptions and cached tokens on the real engine.
#
# ## The one-minute version
#
# * Thinking makes serving **output-heavy**: ten times the output tokens per request, drawn from a
#   heavy-tailed distribution (p99 several times p50).
# * A request holds KV for its prompt plus everything generated so far, so its KV × time grows like
#   `P·L + L²/2`, faster than L. Fewer requests fit the KV pool. Each decode step reads more KV
#   bytes, so **ITL** rises with the batch's *context*, not just its size. Requests live ten times
#   longer, so by Little's law a given rate needs ten times the concurrency. TTFT barely moves. ITL
#   and KV capacity become the binding constraints (PRIMER §7 "What thinking does to serving"; the
#   arithmetic is the capacity primer's, `00-foundations/gpu-capacity-planning/PRIMER.md`).
# * Thinking is **dropped from history** by the chat template. Turn N generated
#   `<think>…</think>answer`, but turn N+1's prompt holds only `answer`, so the prefix-cache hit
#   ends at turn N's assistant header (04 serving-engine PRIMER §5 "Prefix caching").
# * The knobs: a thinking budget (caps the tail), `max_model_len` (must admit prompt + budget +
#   answer, and sets the worst case the KV pool must hold), and routing by effort at the gateway
#   (think only where it pays; 06 `scaling-admission-cost`).

# %%
import math, statistics
from thinklab import engine, env, metrics as M, templates
from thinklab.report import histogram, table
from thinklab.thinking.client import ThinkingClient
from thinklab.thinking.evalset import make_evalset
from thinklab.thinking import recorded
from thinklab.workload import (capacity_primer_view, derive_shape, fit_lognormal, from_completions, kv_token_steps,
                               run_with_gauges, simulate_modes)

print(env.describe())
PROF = engine.profile("t4-qwen3-0.6b")
print(PROF.describe())
TIME_SCALE = 0.02
target = env.connect(time_scale=TIME_SCALE)
client = ThinkingClient(target.url, headers=target.headers)
LABEL = target.label
print(target)

# %% [markdown]
# ## Worked example: the output-length distribution, measured from the server
#
# Forty problems with thinking off and on. Lengths come from `usage` (`completion_tokens`,
# `completion_tokens_details.reasoning_tokens`), the same fields a real vLLM fills.

# %%
problems = make_evalset(40, seed=1)
comps = {m: client.chat_many([p.messages() for p in problems], thinking=m == "on", max_tokens=7000)
         for m in ("off", "on")}
rows = []
for m, cs in comps.items():
    st = from_completions(cs)
    rows += [st["total"].row(f"thinking {m}: output"), st["reasoning"].row(f"thinking {m}: reasoning")]
print(table(rows, title=f"[{LABEL}] output tokens per request (40 problems)"))
med, sigma = fit_lognormal([c.reasoning_tokens for c in comps["on"] if c.reasoning_tokens])
print(f"log-normal fit of reasoning length: median {med:.0f} tokens, sigma {sigma:.2f} -> p99/p50 ~ {math.exp(2.326 * sigma):.1f}")
print(histogram([c.completion_tokens for c in comps["on"]], bins=8, log=True, label="thinking on: output tokens (log bins)"))

# %% [markdown]
# ## Exercise 4.1 — KV × time from the measured lengths: the tail pays
#
# A request with prompt P that emits L tokens holds P + t tokens of KV at decode step t, so over its
# life it costs `kv_token_steps(P, L) = P·L + L(L + 1)/2` token-steps (derived in rl-core notebook 05,
# exercise 5.1; imported here). The formula is convex in L, so a heavy tail costs more than its mean
# suggests. For a list of measured completions `cs` (each has `prompt_tokens` and
# `completion_tokens`), return `(mean_kv, kv_at_means, top10_share)`: the mean over requests of
# `kv_token_steps`, `kv_token_steps` at the mean prompt and mean completion length, and the share of
# the total held by the 10% of requests with the largest KV × time (at least one request). Before
# you run the check, guess the top-10% share with thinking on.

# %% exercise
def kv_time(cs: list) -> tuple:
    ### BEGIN SOLUTION
    ok = [c for c in cs if c.ok]
    kv = sorted((kv_token_steps(c.prompt_tokens, c.completion_tokens) for c in ok), reverse=True)
    at_means = kv_token_steps(statistics.fmean(c.prompt_tokens for c in ok), statistics.fmean(c.completion_tokens for c in ok))
    return statistics.fmean(kv), at_means, sum(kv[: max(1, len(kv) // 10)]) / sum(kv)
    ### END SOLUTION

# %% check
res = {m: kv_time(cs) for m, cs in comps.items()}
for m, (mean_kv, at_means, top) in res.items():
    assert mean_kv >= at_means and 0.1 <= top <= 1.0
    print(f"thinking {m:3}: mean KV x time {mean_kv:>12,.0f} token-steps = {mean_kv / at_means:.2f}x the value at the "
          f"mean length; the top 10% of requests hold {top:.0%}")
assert res["on"][2] > res["off"][2] and res["on"][0] / res["on"][1] > res["off"][0] / res["off"][1]
print(f"✅ [{LABEL}] with thinking on, the mean output grows {statistics.fmean(c.completion_tokens for c in comps['on']) / statistics.fmean(c.completion_tokens for c in comps['off']):.0f}x "
      f"but KV x time grows {res['on'][0] / res['off'][0]:.0f}x: size the KV pool and preemption headroom from the "
      "distribution, not its mean")

# %% [markdown]
# ## Worked example: the capacity primer's arithmetic, with long outputs
#
# The capacity primer's worked example serves Mistral Small 3 (24B) on H100s in fp8: 8.33 requests
# per second, 1,500 tokens in and 300 out, 40 ms per output token. The same formulas with
# 3,000-token outputs (thinking plus answer) follow. `capacity_primer_view` re-implements
# `capacity.py`, and `tests/test_reuse.py` checks it against that file.

# %%
rps = 10_000 * 0.10 * 0.5 / 60
views = [("no thinking (300 out)", capacity_primer_view(rps, 1500, 300)), ("thinking (3,000 out)", capacity_primer_view(rps, 1500, 3000))]
print(table([{"": n, "request s": round(v["duration_s"], 2), "concurrency": round(v["concurrency"], 1),
              "KV/session GB": round(v["kv_per_session_gb"], 4), "sessions/GPU": round(v["sessions_per_gpu"], 1),
              "GPUs for KV": round(v["gpus_for_memory"], 2), "decode tok/s needed": round(v["decode_tok_s_needed"])}
             for n, v in views], title="Capacity primer formulas (arithmetic, H100 80 GB, fp8, TPOT 40 ms)"))
print(f"memory: {views[1][1]['gpus_for_memory'] / views[0][1]['gpus_for_memory']:.0f}x the GPUs for 10x the output")

# %% [markdown]
# Ten times the output needs about 18 times the GPUs for KV. Duration grows 10×, so concurrency
# grows 10×, and each session's average context nearly doubles (1,650 → 3,000 tokens). These
# formulas price every token at the SLO's 40 ms, which overstates the GPU count (6 here) and makes
# a looser SLO look dearer; PRIMER §7 "What thinking does to serving" closes Little's law on the
# step the fleet actually runs at (`rlcore.workload.plan_steady`: 3 GPUs) and still finds 18× the
# GPUs for KV (2.75 vs 0.15). `derive_shape` below works the same way: ITL at the batch. One
# caveat carries over from the primer: its `decode_aggregate` has no memory cap. At 1,000
# concurrent sessions × 0.246 GB the KV alone is 246 GB, over three H100s' worth, so the batch that
# sets TPOT must be capped by the KV pool, as the next exercise does.
#
# ## Exercise 4.2 — the batch the KV pool allows, and the ITL it produces
#
# In steady state each running request sits, on average, at context `P + L/2`. Return
# `(batch, itl_ms)`: `batch` is how many such requests fit `profile.kv_capacity_tokens` (integer
# division, at least 1, at most `max_num_seqs`), and `itl_ms` is the decode step at that batch,
# `profile.decode_step_s(batch, batch × (P + L/2))` in milliseconds.

# %% exercise
def batch_and_itl(profile, P: float, L_mean: float, max_num_seqs: int = 256) -> tuple:
    ### BEGIN SOLUTION
    ctx = P + L_mean / 2
    batch = max(1, min(int(profile.kv_capacity_tokens // ctx), max_num_seqs))
    return batch, 1e3 * profile.decode_step_s(batch, batch * ctx)
    ### END SOLUTION

# %% check
L_on = statistics.fmean(c.completion_tokens for c in comps["on"])
L_off = statistics.fmean(c.completion_tokens for c in comps["off"])
for L in (L_off, L_on, 3000):
    s = derive_shape(PROF, 60, [L], max_num_seqs=256)
    b, itl = batch_and_itl(PROF, 60, L)
    assert b == s.batch_by_memory and abs(itl - s.itl_ms) < 1e-9
b_on, itl_on = batch_and_itl(PROF, 60, L_on)
b_off, itl_off = batch_and_itl(PROF, 60, L_off)
print(f"✅ [simulated roofline] thinking off: batch {b_off}, ITL {itl_off:.1f} ms | thinking on: batch {b_on}, ITL {itl_on:.1f} ms")
print(table([derive_shape(PROF, 60, [c.completion_tokens for c in comps[m]], f"thinking {m}").row() for m in ("off", "on")],
            title="[SIMULATED] steady-state shape on one T4"))

# %% [markdown]
# ## Worked example: the same arrivals, three modes, in virtual time
#
# The engine emulator (FCFS continuous batching, KV blocks, recompute preemption, the roofline step
# model) runs 300 requests arriving at 2/s and at 4/s, for thinking off, thinking on, and a
# 512-token budget. All numbers are **simulated**; with the real engine they are what `vllm bench
# serve` and `/metrics` would show.

# %%
qs = [p.prompt for p in make_evalset(300, seed=2)]
modes = {"thinking off": {"thinking": False}, "thinking on": {}, "budget 512": {"budget": 512}}
for rate in (2.0, 4.0):
    print(table(simulate_modes(PROF, qs, modes, rate=rate),
                ["mode", "accuracy", "out mean", "answer starts p50 s", "ITL p50 ms", "ITL p99 ms", "E2E p99 s",
                 "KV peak", "max running", "preemptions", "tokens/correct"],
                title=f"[SIMULATED] 300 requests at {rate}/s, Qwen3-0.6B on a T4"), "\n")

# %% [markdown]
# Thinking multiplies ITL several times over, and the answer starts seconds rather than
# milliseconds after the request. Unlimited thinking is already at saturation at 2/s: it asks for
# about 2 × 1,184 ≈ 2,400 output tokens a second, close to the thinking shape's steady-state
# throughput in the table above, so the KV pool peaks near 0.9 and the E2E p99 is minutes. At 4/s the pool is full (peak
# 1.0): new requests wait for blocks, running ones are preempted and recomputed, and the tail
# grows further. The 512-token budget keeps the pool at 5–11% and ITL close to the no-thinking
# value, and among the thinking modes it costs less than half the tokens per correct answer (784
# vs 1,692). Thinking off is cheapest per correct answer of all (364) but tops out at 0.28
# accuracy. So the modes trade accuracy for cost, which Exercise 4.5 turns into a routing choice;
# the budget's capacity argument (a pool that never fills) comes on top of its cost argument.
#
# ## Worked example: a live run, read from the engine's `/metrics`
#
# An open-loop burst against the server: 24 thinking requests with a 512-token budget, at 20 per
# second of *simulated* time. Against the fake server, `/metrics` reports simulated seconds on the
# engine's own clock. The client-side wall-clock numbers are compressed 50× by the time scale, so we
# read the engine side.
# Nothing in vLLM v0.30.0's metrics separates reasoning from answer tokens; thinking shows up as
# more of everything decode drives.

# %%
reqs = [(p.messages(), {"thinking": True, "budget": 512, "max_tokens": 2000}, "think") for p in make_evalset(24, seed=4)]
before = M.scrape(target.url, headers=target.headers)
run, gauges = run_with_gauges(target.url, reqs, rate=20 / (TIME_SCALE if target.simulated else 1.0),
                              interval_s=0.01, headers=target.headers)
after = M.scrape(target.url, headers=target.headers)
d = M.summary(M.delta(after, before))
d.update(kv_usage_peak=max((g[1] for g in gauges), default=math.nan), running_peak=max((g[2] for g in gauges), default=math.nan))
for k in ("running", "waiting", "kv_usage"):
    d.pop(k)                                   # end-of-run gauges (idle); the peaks above are what mattered
print(table([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}],
            title=f"[{LABEL}] engine-side view of the run (/metrics; seconds are simulated on the fake server)"))
print(f"{len(gauges)} gauge scrapes during the run; {run.summary()['completed']} requests completed, "
      f"reasoning share of output tokens {run.summary()['reasoning_share']:.2f}")

# %% [markdown]
# ## Exercise 4.3 — pick `max_model_len` for a thinking budget
#
# vLLM rejects a request whose prompt plus `max_tokens` exceeds `max_model_len`, and a request
# without `max_tokens` may generate up to `max_model_len − prompt`. Given the p99 prompt, the thinking
# budget, the tokens of the forced end phrase and the p99 answer, return the smallest
# `max_model_len`, rounded up to a multiple of 1,024, that lets every request finish. Then return
# how many requests *at that worst case* the KV pool of `PROF` holds at once.

# %% exercise
def plan_max_model_len(prompt_p99: int, budget: int, end_phrase: int, answer_p99: int) -> int:
    ### BEGIN SOLUTION
    need = prompt_p99 + budget + end_phrase + answer_p99
    return math.ceil(need / 1024) * 1024
    ### END SOLUTION

def worst_case_concurrency(profile, max_model_len: int) -> int:
    ### BEGIN SOLUTION
    return profile.kv_capacity_tokens // max_model_len
    ### END SOLUTION

# %% check
assert plan_max_model_len(200, 512, 20, 300) == 2048 and plan_max_model_len(200, 4096, 20, 300) == 5120
assert plan_max_model_len(0, 1024, 0, 0) == 1024
assert worst_case_concurrency(PROF, 8192) == 13 and worst_case_concurrency(PROF, 2048) == 54
print(f"✅ budget 512 -> max_model_len 2048 -> {worst_case_concurrency(PROF, 2048)} worst-case requests in KV; "
      f"unbudgeted 8K -> {worst_case_concurrency(PROF, 8192)}")

# %% [markdown]
# ## Worked example: why the prefix cache stops at the assistant header
#
# A two-turn conversation rendered with Qwen3's template logic (`thinklab.templates.render_qwen3`).
# Turn 1 generated its thinking into the KV cache. Turn 2's prompt drops it, because assistant turns
# before the last user message lose their reasoning. The cache can only serve the common prefix,
# in whole 16-token blocks.

# %%
system = {"role": "system", "content": "You are a careful assistant for a bank's operations team. " * 6}
history = [system, {"role": "user", "content": "What is 47 * 23 - 318?"}]
turn1 = {"reasoning": "47 times 23 is 1081 . 1081 minus 318 is 763 . Let me double check 47 times 23 . " * 12,
         "content": "The answer is \\boxed{763}."}
rows = []
for keep in (False, True):
    r = templates.turn_reuse(history, turn1, "Now divide it by 7.", keep_all_reasoning=keep)
    rows.append({"history rendering": "reasoning kept (interleaved)" if keep else "Qwen3 template (reasoning dropped)",
                 **{k: v for k, v in r.items()}})
print(table(rows, title="turn 2 vs what turn 1 left in the KV cache (toy tokens)"))

# the same on the server: two turns, cached_tokens from usage
msgs = history[:]
c1 = client.chat(msgs, max_tokens=3000)
msgs += [{"role": "assistant", "content": c1.content}, {"role": "user", "content": "Now divide it by 7."}]
c2 = client.chat(msgs, max_tokens=3000)
print(f"[{LABEL}] turn 2: prompt {c2.prompt_tokens} tokens, cached {c2.cached_tokens} "
      f"(turn 1 generated {c1.reasoning_tokens} reasoning tokens that turn 2 cannot reuse)")

# %% [markdown]
# ## Exercise 4.4 — tokens a block-hash prefix cache can serve
#
# `prev` is the previous request's full context (its prompt and the tokens it generated), and `nxt`
# is the new prompt. Return the number of prompt tokens served from cache: the length of their
# common prefix, capped at `len(nxt) − 1` (the engine always computes at least the last prompt
# token), rounded *down* to whole blocks.

# %% exercise
def my_cached_tokens(prev: list, nxt: list, block: int = 16) -> int:
    ### BEGIN SOLUTION
    n = 0
    for a, b in zip(prev, nxt):
        if a != b:
            break
        n += 1
    n = min(n, len(nxt) - 1)
    return n // block * block
    ### END SOLUTION

# %% check
assert my_cached_tokens(list(range(100)), list(range(40)) + [-1] * 10) == 32
assert my_cached_tokens(list(range(100)), list(range(33))) == 32 and my_cached_tokens(list(range(100)), list(range(32))) == 16
assert my_cached_tokens([1, 2], [3, 4]) == 0
a, b = templates.tokens(templates.render_qwen3(history) + "x " * 50), templates.tokens(templates.render_qwen3(history) + "y " * 9)
assert my_cached_tokens(a, b) == templates.cached_tokens(a, b)
print("✅ common prefix, minus the last prompt token, rounded down to 16-token blocks")

# %% [markdown]
# ## Exercise 4.5 — route by effort at the gateway
#
# A gateway can decide per request whether the model thinks (06 `scaling-admission-cost` routes by
# task and effort level). Using the records of notebook 03 (simulated unless you collected your
# own), and assuming the gateway knows each problem's difficulty (1–4), write
# `policy_stats(think_levels)`. It returns `(accuracy, output tokens per question)` over the whole
# mix when the model thinks only on problems whose difficulty is in `think_levels`, and answers
# directly otherwise. The check evaluates all 16 policies and picks the cheapest one that keeps 90%
# of always-think accuracy.

# %% exercise
RECS = recorded.load()
OFF = {r["id"]: r for r in RECS if r["mode"] == "off"}
ON = {r["id"]: r for r in RECS if r["mode"] == "on"}

def policy_stats(think_levels: set) -> tuple:
    ### BEGIN SOLUTION
    acc, tok = [], []
    for pid, off in OFF.items():
        r = ON[pid] if off["difficulty"] in think_levels else off
        acc += r["correct"]
        tok += [a + b for a, b in zip(r["reasoning_tokens"], r["answer_tokens"])]
    return statistics.fmean(acc), statistics.fmean(tok)
    ### END SOLUTION

# %% check
from itertools import combinations
levels = (1, 2, 3, 4)
policies = {tuple(c): policy_stats(set(c)) for n in range(5) for c in combinations(levels, n)}
never, always = policies[()], policies[levels]
assert never[1] < always[1] and never[0] < always[0]
target_acc = 0.9 * always[0]
ok = {k: v for k, v in policies.items() if v[0] >= target_acc}
choice = min(ok, key=lambda k: ok[k][1])
frontier = sorted(policies.items(), key=lambda kv: kv[1][1])
rows = [{"think on difficulty": "+".join(map(str, k)) or "never", "accuracy": round(a, 3), "tokens/question": round(t),
         "tokens/correct": round(t / a) if a else "inf", "chosen": "*" if k == choice else ""}
        for k, (a, t) in frontier if all(not (a2 >= a and t2 < t) for a2, t2 in policies.values())]
print(table(rows, title="[SIMULATED records] routing policies on the accuracy/token frontier (* = chosen)"))
assert ok[choice][0] >= target_acc and ok[choice][1] <= always[1]
print(f"✅ think on {'+'.join(map(str, choice)) or 'nothing'}: {ok[choice][0]:.3f} accuracy for {ok[choice][1]:.0f} tokens/question "
      f"(always think: {always[0]:.3f} for {always[1]:.0f}; never: {never[0]:.3f} for {never[1]:.0f})")

# %% [markdown]
# Where thinking pays per token is an empirical question. In these simulated records the hardest
# problems cost the most thinking for the least accuracy per token: dropping thinking there alone
# halves the tokens per question for about nine points of accuracy. Which policy wins depends on
# the accuracy you must keep. A real model on your traffic may differ, so the router's table is
# built from measurements (this notebook's T1 path), not from intuition.

# %%
target.stop()

# %% [markdown]
# ## On a real GPU (T1)
#
# ```bash
# vllm serve Qwen/Qwen3-0.6B --dtype half --max-model-len 8192 --reasoning-parser qwen3 \
#   --enable-prompt-tokens-details --gpu-memory-utilization 0.85 &
# export THINKLAB_URL=http://127.0.0.1:8000
# ```
#
# Re-run the notebook: the length table, the live run and the two-turn `cached_tokens` become
# measurements. Then change one knob at a time and watch `vllm:kv_cache_usage_perc`,
# `vllm:num_preemptions` and `vllm:inter_token_latency_seconds`: `--max-model-len 2048` with a
# 512-token budget against 8192 unbudgeted; `--max-num-seqs 32`; and the rate. Speculative decoding
# (04 serving-engine PRIMER §7 "Speculative decoding") is worth testing on long outputs: n-gram
# drafting finds repetition in reasoning traces, and the long decode is memory-bound, which is where
# speculation pays. Measure it with the 04 lab's notebook 05.

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Turning on thinking multiplies output tokens by roughly ten, with a heavy tail.
# Serving is now decode-bound. KV × time per request grows faster than the output, so the batch the
# KV pool holds shrinks and every step reads more KV, and ITL rises. Requests live ten times
# longer, so the same request rate needs ten times the concurrency. With the capacity primer's
# numbers that is about 18 times the GPUs for KV. TTFT hardly changes, but the *answer* starts
# seconds later. We cap the tail with a thinking budget and size `max_model_len` from it, which also
# sets the worst case the KV pool must hold. We route by effort at the gateway, thinking only where
# it pays, and track cost per correct answer. Multi-turn chats don't reuse the thinking KV, because
# the template drops it. The prefix cache stops at the last assistant header, and the answer is
# re-prefilled."
#
# **Drill 1.** *We enabled thinking and TTFT is unchanged, but users say it got slow. Which metric?*
# The time to the first *content* token and ITL. TTFT is the first reasoning token. Report time to
# first content, and ITL p99, which rises with the batch's context.
#
# **Drill 2.** *KV usage hits 100% and `vllm:num_preemptions` climbs only on the thinking route.
# Fixes?* A thinking budget (shorter tail), a lower `max_model_len` to match it, fewer concurrent
# sequences per replica (`--max-num-seqs`), or more replicas. The long tail, not the mean, fills the
# pool.
#
# **Drill 3.** *Why is our multi-turn prefix hit rate lower with the thinking model?* The template
# renders earlier assistant turns without their reasoning. Turn N's generated tokens are not a
# prefix of turn N+1's prompt, so the cache match ends at turn N's assistant header. Inside a single
# tool-calling loop the thinking is kept, so the prefix keeps matching there.
