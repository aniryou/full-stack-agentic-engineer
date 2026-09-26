# %% [markdown]
# # 05 · Thinking models and the serving workload
#
# **Tier:** T0 — CPU only, standard library + numpy, no network, well under a minute. Every latency and GPU
# count is a **model** (the capacity primer's formulas plus a roofline step), not a measurement. The same
# questions against a server — a T0 fake server that emits `reasoning` deltas, or real vLLM with Qwen3 and
# `--reasoning-parser` on a T4 — are `thinking-lab` notebooks `02_a_thinking_model_on_one_gpu` and
# `04_serving_thinking_models`.
#
# ## The one-minute version
# A thinking model is trained to spend output tokens before it answers: RL with verifiable rewards lengthens
# thinking, rejection-sampling SFT keeps it, and distillation copies it into small models without RL. For serving,
# that makes the workload **output-heavy and decode-bound**, with a **heavy-tailed** length distribution (the p99
# trace is ten times the median). A request's KV grows while it thinks, so its KV-token-steps are P·L + L²/2 —
# ten times the output is about eighteen times the memory-time. Little's law multiplies concurrency by the output
# length; **KV memory** sets the GPU count, and a tight **ITL** SLO caps the batch before HBM does. `max_tokens`
# counts thinking, so a cap truncates answers; a **thinking budget** closes the think block and answers instead.
# Templates drop earlier turns' thinking, so it is never a cached prefix. And every thinking token is billed as
# output: the metric is **cost per correct answer**, and effort is something to route. Primer: `../PRIMER.md`
# §5 and §7.

# %%
import math

import numpy as np

from rlcore import Policy, ThinkTask, pg, workload as w

H100, SMALL = w.GPUS["H100"], w.MISTRAL_SMALL

# %% [markdown]
# ## Worked example 1 — how a model comes to think: rejection sampling, then distillation
# DeepSeek-R1's recipe alternates RL with **rejection-sampling SFT**: sample many answers, keep the correct ones,
# fine-tune on them. On the ThinkTask, correct answers are longer on average (thinking helps), so imitation of
# the survivors lengthens thinking even with no RL at all.

# %%
think = ThinkTask(e0=0.8, q=0.15, max_think=16)
base = Policy.for_task(think)                         # answers at once half the time: mean 1 thinking token
print("base   ", {k: round(v, 3) for k, v in think.expected(base.stop_probs(think)).items() if k != "reward"})
rng = np.random.default_rng(0)
pol = base.copy()
for rnd in range(3):
    samples = pol.sample(think, rng, 512)
    kept = [s for s in samples if s.info["correct"]]
    for _ in range(20):
        pg.sft_step(pol, kept, lr=1.0)
    ex = think.expected(pol.stop_probs(think))
    print(f"round {rnd}: kept {len(kept)}/512 correct samples (mean length {np.mean([s.info['length'] for s in kept]):.2f} "
          f"vs {np.mean([s.info['length'] for s in samples]):.2f} overall) → accuracy {ex['accuracy']:.3f}, thinking {ex['length']:.2f}")

# %% [markdown]
# Now **distillation**: an RL-trained teacher (notebook 01's run) generates traces, and a fresh student is trained
# on them with plain SFT — no reward, no RL.

# %%
teacher = base.copy()
pg.train_reinforce(teacher, think, np.random.default_rng(0), steps=300, batch=16, lr=2.0)
student = base.copy()
traces = teacher.sample(think, np.random.default_rng(1), 1000)
for _ in range(60):
    pg.sft_step(student, traces, lr=2.0)
for name, p in (("teacher (RL)", teacher), ("student (SFT on traces)", student)):
    ex = think.expected(p.stop_probs(think))
    print(f"{name:24} accuracy {ex['accuracy']:.3f}, mean thinking {ex['length']:.1f} tokens")

# %% [markdown]
# The student inherits the teacher's thinking behaviour from its outputs alone. That is why the R1 distills
# (Qwen- and Llama-based, 1.5B–70B) were SFT-only on ~800k samples, and why distilling a strong model beat running
# RL directly on the small base in R1's comparison (AIME 2024: 72.6 vs 47.0 for 32B; primer §5, verify).
#
# ## Worked example 2 — a heavy-tailed length distribution
# Thinking lengths vary by question difficulty — a mixture that looks lognormal. Median 1,500 tokens, σ = 1:

# %%
med, sig = 1500, 1.0
print(f"mean {w.lognormal_mean(med, sig):,.0f}; " + ", ".join(
    f"p{int(p * 100)} {w.lognormal_quantile(med, sig, p):,.0f}" for p in (0.5, 0.9, 0.99)))
kv = w.kv_per_token_kb(w.QWEN3_0_6B, "fp16")
print(f"Qwen3-0.6B: {kv:.0f} KiB of KV per token ({kv * 1024:,.0f} B); an 8K-token trace holds "
      f"{8192 * kv * 1024 / 1e9:.2f} GB, against {w.weight_gb(w.QWEN3_0_6B, 'fp16'):.2f} GB of weights")

# %% [markdown]
# The mean is 1.6× the median and the p99 ten times it. Size `max_model_len` and preemption headroom for the tail,
# not the mean — and remember that on a small model one long trace's KV is the size of the model.
#
# ## Worked example 3 — the capacity primer's bank, with and without thinking
# `00-foundations/gpu-capacity-planning` sizes an internal assistant: 8.33 requests/s, 1,500 tokens in, 300 out,
# 40 ms TPOT, Mistral Small 3 (24B) in FP8 on H100s. `w.plan` reproduces its numbers, then adds what its
# `decode_aggregate` leaves out: the batch per GPU is capped by HBM and by the ITL SLO. Like the primer, `w.plan`
# prices every output token at the SLO's TPOT. `w.plan_steady` instead lets a request live as long as the step the
# fleet actually runs at: N = rps·(TTFT + out·step(b))/b GPUs hold a steady batch b per GPU (`w.gpus_for_batch`).

# %%
rps = 10_000 * 0.10 * 0.5 / 60
for label, out, tpot in (("no thinking", 300, 40), ("2,700 thinking + 300 answer", 3000, 40),
                         ("same, 20 ms ITL SLO", 3000, 20)):
    p = w.plan(SMALL, H100, rps, 1500, out, tpot_ms=tpot)
    print(f"{label:28} duration {p['duration_s']:6.2f} s  live {p['concurrency']:7.1f}  KV/session "
          f"{w.kv_per_session_gb(SMALL, p['avg_ctx'], 'fp8'):.3f} GB  sessions/GPU {p['sessions_per_gpu']:5.1f}  "
          f"ITL batch {p['itl_batch']:4d}")
    print(f"{'':28} GPUs by memory {p['gpus']['memory']:.2f}, ITL slots {p['gpus']['itl_slots']:.2f}, decode "
          f"{p['gpus']['decode']:.2f}, prefill {p['gpus']['prefill']:.2f} → {p['gpus_needed']} GPU(s), bound by {p['binding']}")
    q = w.plan_steady(SMALL, H100, rps, 1500, out, tpot_ms=tpot)
    print(f"{'':28} at the step the fleet runs at: GPUs by memory {q['gpus']['memory']:.2f}, ITL slots "
          f"{q['gpus']['itl_slots']:.2f} → {q['gpus_needed']} GPU(s), bound by {q['binding']}; settles at "
          f"{q['batch']:.1f}/GPU, {q['step_s'] * 1e3:.1f} ms a step")
print(f"decode_aggregate alone would batch all 1,000 on one GPU: "
      f"{1000 * w.kv_per_session_gb(SMALL, 3000, 'fp8'):.0f} GB of KV on an 80 GB card")

# %% [markdown]
# Ten times the output tokens needs eighteen times the GPUs for memory (4.77 vs 0.26 at the SLO's TPOT, 2.57 vs
# 0.14 at the step the fleet runs at): concurrency grows with the output (Little's law) and each live session holds
# 1.8× the KV (3,000 average context vs 1,650). The decode *throughput* need is not the binding constraint.
#
# Read the last two rows of `w.plan` with care: a 20 ms SLO halves the lifetime the convention assumes, so it needs
# *fewer* GPUs (3) than the 40 ms SLO (5). That is an artefact. At 3 GPUs the fleet settles at 139 per GPU and
# 16.7 ms a step, under both SLOs, so `w.plan_steady` needs 3 for both, and the tight SLO's need is the larger. With
# a 20 ms ITL SLO the step time caps each GPU at 187 sessions, below the 209 that fit in HBM: ITL binds first.
#
# ## Worked example 4 — KV working set: memory × time
# A request holds P + t tokens of KV at decode step t, for L steps: Σ = P·L + L(L+1)/2.

# %%
base_kv = w.kv_token_steps(1500, 300)
for out in (300, 1000, 1500, 3000, 8000):
    print(f"output {out:5,d}: {w.kv_token_steps(1500, out):>12,d} KV-token-steps = {w.kv_token_steps(1500, out) / base_kv:5.1f}× the 300-token answer")

# %% [markdown]
# ## Worked example 5 — `max_tokens` is a cap, a thinking budget is a budget
# Each question needs L_req ~ lognormal(1,500, σ = 1) thinking tokens; the answer is 300 tokens. With
# `max_tokens`, a request that is still thinking when the cap hits returns no answer (`finish_reason="length"`,
# empty content). With budget forcing — vLLM's `thinking_token_budget`, or Qwen's two-call recipe — the think
# block is closed at the budget and the model answers with what it has (right 30% of the time here, e0 = 0.7).

# %%
print(f"{'limit':>7} {'max_tokens: accuracy':>21} {'truncated':>10} {'budget: accuracy':>17} {'mean output':>12}")
for cap in (2048, 4096, 8192, 16384):
    cut = w.budget_outcome(med, sig, answer_tokens=300, e0=0.7, max_tokens=cap)
    forced = w.budget_outcome(med, sig, answer_tokens=300, e0=0.7, budget=cap - 300)
    print(f"{cap:7,d} {cut['accuracy']:21.3f} {cut['truncated']:10.3f} {forced['accuracy']:17.3f} {forced['tokens']:12,.0f}")

# %% [markdown]
# Same tokens, different outcome: at 4K, one request in six gets no answer at all under `max_tokens`. Set
# `max_tokens` (and `max_model_len`) generously and enforce cost with a thinking budget. Note what does **not** work
# for this on Qwen3: `reasoning_effort` other than "none" only switches thinking on (primer §5, verify).
#
# ## Worked example 6 — prefix caching when the template drops old thinking
# Qwen3's and gpt-oss's templates render earlier assistant turns *without* their thinking. Turn N's KV holds
# prompt + thinking + answer; turn N+1's prompt contains only the answer, so the cache hit ends where turn N's
# prompt ended (rounded down to a 16-token block) and the answer is prefilled again. Inside one tool-calling loop
# the thinking is kept and the whole previous sequence is a hit.

# %%
turns = [(100, 800, 200)] * 4                          # (user, thinking, answer) tokens per turn; 1,000-token system prompt
for keep in (False, True):
    rows = w.turn_prefills(1000, turns, keep_thinking=keep)
    print(("thinking kept   " if keep else "thinking dropped"), " ".join(f"[{p:,} prompt, {p - h:,} prefilled]" for p, h in rows))

# %% [markdown]
# Dropping thinking keeps prompts short (it never re-enters the context), but its KV — 800 tokens per turn here —
# is computed, held while decoding and then never reused. Prefix-cache hit rates on multi-turn thinking traffic
# are therefore lower than on the same conversation without thinking, and cache-aware routing (layer 05) still
# pays off only on the stable prefix.
#
# ## Worked example 7 — cost per *correct* answer, and routing by effort
# Per-call API cost at the 06 scaling lab's example prices ($1.50 / $9.00 / $0.15 per 1M input / output / cached
# input, verify): 5,000 input tokens (2,700 cached) and 350 output tokens, or 3,500 with thinking.

# %%
no_think, thinking = w.api_cost(5000, 350, 2700), w.api_cost(5000, 3500, 2700)
print(f"per call: ${no_think:.6f} without thinking, ${thinking:.6f} with ({thinking / no_think:.2f}×)")
# accuracy by request class — inputs for the arithmetic (illustrative), not measurements
acc = {"easy": {"off": 0.95, "on": 0.97}, "hard": {"off": 0.30, "on": 0.85}}
share = {"easy": 0.7, "hard": 0.3}
for policy in ({"easy": "off", "hard": "off"}, {"easy": "on", "hard": "on"}, {"easy": "off", "hard": "on"}):
    cost = sum(share[c] * (thinking if policy[c] == "on" else no_think) for c in share)
    accuracy = sum(share[c] * acc[c][policy[c]] for c in share)
    print(f"easy {policy['easy']:3} / hard {policy['hard']:3}: accuracy {accuracy:.3f}, ${cost:.6f} per call, "
          f"${w.cost_per_correct(cost, accuracy):.6f} per correct answer")

# %% [markdown]
# Thinking everywhere buys accuracy at 5× the cost; thinking only where it pays keeps nearly all of the accuracy
# for a fraction of it. That is routing by effort at the gateway (layer 06) — and it needs a classifier, or a
# cheap first pass, that knows which requests are hard.
#
# ## Exercise 5.1 — memory × time
# Write `kv_steps(P, L)`, the KV-token-steps of a request with a P-token prompt and L output tokens, and compute
# `ratio`: a 3,000-token output against a 300-token one on a 1,500-token prompt.

# %% exercise
def kv_steps(P, L):
    ### BEGIN SOLUTION
    return P * L + L * (L + 1) // 2
    ### END SOLUTION


### BEGIN SOLUTION
ratio = kv_steps(1500, 3000) / kv_steps(1500, 300)
### END SOLUTION

# %% check
assert kv_steps(1500, 300) == w.kv_token_steps(1500, 300) and round(ratio, 1) == 18.2
print(f"✅ 10× the output, {ratio:.1f}× the KV-token-steps: the L²/2 term takes over once L passes P")

# %% [markdown]
# ## Exercise 5.2 — size the thinking deployment by hand
# The bank with thinking: 3,000 output tokens, 1,500 in, FP8, 80 GB H100 with 10% held back, 24 GB of weights,
# 80 KiB of KV per token (FP8). Compute `live` (Little's law, with duration = TTFT + 3,000 × 40 ms),
# `per_gpu` (sessions per GPU at the average context 1,500 + 3,000/2) and `gpus` = live / per_gpu — the capacity
# primer's convention. Then drop the convention: with `per_gpu` sessions in every GPU's batch a decode step takes
# `step = (24 + per_gpu × KV per session) / 3,350` seconds (H100: 3.35 TB/s; memory-bound), a request lives
# TTFT + 3,000 × step, and the fleet that holds that batch steadily is `steady_gpus` = rps × lifetime / per_gpu.

# %% exercise
ttft = w.ttft_s(24, 1500, H100)
### BEGIN SOLUTION
live = rps * (ttft + 3000 * 0.040)
per_gpu = (80 * 0.9 - 24) / (80 * 3000 / 1024 ** 2)
gpus = live / per_gpu
step = (24 + per_gpu * 80 * 3000 / 1024 ** 2) / 3350
steady_gpus = rps * (ttft + 3000 * step) / per_gpu
### END SOLUTION

# %% check
p = w.plan(SMALL, H100, rps, 1500, 3000)
assert abs(live - p["concurrency"]) < 1e-6 and abs(per_gpu - p["sessions_per_gpu"]) < 1e-6
assert round(gpus, 2) == 4.77
assert abs(steady_gpus - w.plan_steady(SMALL, H100, rps, 1500, 3000)["gpus"]["memory"]) < 1e-6
print(f"✅ {live:,.0f} live sessions / {per_gpu:.1f} per GPU = {gpus:.2f} GPUs — vs 0.26 without thinking. With a full "
      f"batch a step takes {step * 1e3:.1f} ms, not 40, so {steady_gpus:.2f} GPUs hold it steadily (vs 0.14): still 18×")

# %% [markdown]
# ## Exercise 5.3 — choose `max_model_len`
# Prompts are 1,500 tokens, answers 300, thinking lognormal(1,500, σ = 1). Pick the smallest `max_model_len` that
# is a multiple of 1,024 and truncates at most 1% of requests. (vLLM's `max_model_len` bounds prompt + output.)

# %% exercise
### BEGIN SOLUTION
need = 1500 + w.lognormal_quantile(1500, 1.0, 0.99) + 300
max_model_len = int(math.ceil(need / 1024) * 1024)
### END SOLUTION

# %% check
assert max_model_len == 17408
assert w.budget_outcome(1500, 1.0, answer_tokens=300, max_tokens=max_model_len - 1500)["truncated"] <= 0.01
assert w.budget_outcome(1500, 1.0, answer_tokens=300, max_tokens=max_model_len - 1024 - 1500)["truncated"] > 0.01
print(f"✅ max_model_len {max_model_len:,}: over five times the median request (3,300 tokens) — size preemption headroom "
      "for the tail, not the mean")

# %% [markdown]
# ## Exercise 5.4 — predict the cache hit on turn 2
# System prompt 1,000 tokens; turn 1: user 100 (+ a 4-token assistant header), thinking 800, answer 200. The
# template drops turn 1's thinking. Turn 2: user 100 (+ header). Predict `prompt2` (turn 2's prompt length),
# `hit2` (tokens served from the cache, full 16-token blocks only) and `prefill2`.

# %% exercise
### BEGIN SOLUTION
prompt1 = 1000 + 100 + 4
prompt2 = prompt1 + 200 + 100 + 4
hit2 = min(prompt1, prompt2 - 1) // 16 * 16
prefill2 = prompt2 - hit2
### END SOLUTION

# %% check
assert (prompt2, hit2) == w.turn_prefills(1000, [(100, 800, 200), (100, 800, 200)])[1]
print(f"✅ turn 2: {prompt2:,} tokens, {hit2:,} cached, {prefill2} prefilled — the 200-token answer again, plus the new message")

# %% [markdown]
# ## Exercise 5.5 — route by effort
# Using `acc`, `share`, `no_think` and `thinking` from worked example 7, set `best_policy` to the routing
# (`{"easy": ..., "hard": ...}` with values "on"/"off") with the lowest cost per correct answer, subject to overall
# accuracy of at least 0.90.

# %% exercise
### BEGIN SOLUTION
options = [{"easy": e, "hard": h} for e in ("off", "on") for h in ("off", "on")]
def score(pol):
    cost = sum(share[c] * (thinking if pol[c] == "on" else no_think) for c in share)
    accuracy = sum(share[c] * acc[c][pol[c]] for c in share)
    return accuracy, cost / accuracy
best_policy = min((o for o in options if score(o)[0] >= 0.90), key=lambda o: score(o)[1])
### END SOLUTION

# %% check
assert best_policy == {"easy": "off", "hard": "on"}
print("✅ think where it pays: route hard requests to thinking and keep easy ones fast and cheap")

# %% [markdown]
# ## Exercise 5.6 — one round of rejection-sampling SFT
# From `base`, sample 512 completions with `rng5`, keep the correct ones, and take 20 SFT steps (lr 1.0) on them
# into `rs_policy`. Its mean thinking length must rise.

# %% exercise
rng5 = np.random.default_rng(7)
rs_policy = base.copy()
### BEGIN SOLUTION
kept5 = [s for s in rs_policy.sample(think, rng5, 512) if s.info["correct"]]
for _ in range(20):
    pg.sft_step(rs_policy, kept5, lr=1.0)
### END SOLUTION

# %% check
before, after = think.expected(base.stop_probs(think)), think.expected(rs_policy.stop_probs(think))
assert after["length"] > before["length"] + 0.3 and after["accuracy"] > before["accuracy"]
print(f"✅ keeping only correct samples: thinking {before['length']:.2f} → {after['length']:.2f} tokens, accuracy "
      f"{before['accuracy']:.3f} → {after['accuracy']:.3f} — selection alone rewards longer thinking")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Thinking models spend output tokens before answering — RL with verifiable rewards
# taught them to, and distillation copies the behaviour into small models. For serving, that turns a
# prefill-heavy chat workload into a decode-heavy one with a heavy tail: our p99 thinking length is ten times the
# median. Because KV grows while a request thinks, memory-time per request goes as P·L + L²/2, so 10× the
# output is ~18× the GPUs for memory at the same arrival rate — the capacity primer's formulas with the batch
# capped by HBM and by the ITL SLO, which binds first when it is tight. We size at the step the fleet actually
# runs at, not at the SLO's TPOT, or the looser SLO looks dearer. We size max_model_len for the tail and
# enforce cost with a thinking budget rather than max_tokens, which counts thinking and truncates answers. The
# templates drop old thinking, so multi-turn prefix-cache hits stop at each turn's prompt. Every thinking token
# is billed as output, so we track cost per correct answer and route effort per request class at the gateway."
#
# **Drill questions**
# 1. *We enabled thinking and p99 ITL doubled at the same QPS. Why?* — Concurrency rose with output length
#    (Little's law) and each session's KV grew, so decode batches are larger and read more KV per step; if HBM
#    fills, preemptions add recomputation. Capacity for ITL, not tokens/s, is what changed.
# 2. *Users report empty answers from the thinking model. Where do you look?* — `finish_reason="length"` with
#    empty `content`: `max_tokens` ran out inside the think block. Raise the cap and use a thinking budget.
# 3. *Our multi-turn prefix-cache hit rate fell when we switched to a thinking model. Bug?* — Expected: the
#    template drops earlier turns' thinking, so each turn's KV diverges from the next prompt right after that
#    turn's prompt; the answer is re-prefilled and the thinking KV is never reused.
