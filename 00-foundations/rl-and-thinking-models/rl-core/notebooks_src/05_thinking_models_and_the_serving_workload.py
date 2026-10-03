# %% [markdown]
# # 05 · Thinking models and the serving workload
#
# **Tier:** T0. It needs only a CPU, the standard library and numpy. It needs no network. It runs in well under a
# minute. Every latency and GPU count is a **model**, not a measurement. The model is the capacity primer's
# formulas plus a roofline step.
#
# Two `thinking-lab` notebooks ask the same questions against a server:
# `02_a_thinking_model_on_one_gpu` and `04_serving_thinking_models`. The server is a T0 fake server that emits
# `reasoning` deltas, or real vLLM with Qwen3 and `--reasoning-parser` on a T4.
#
# ## The one-minute version
# Training teaches a thinking model to spend output tokens before it answers. RL with verifiable rewards makes the
# thinking longer. Rejection-sampling SFT keeps it. Distillation copies it into small models without RL.
#
# - Thus the serving workload is **output-heavy and decode-bound**. Its length distribution is **heavy-tailed**:
#   the p99 trace is ten times the median. The KV of a request grows while the request thinks. Thus its
#   KV-token-steps are $P \cdot L + L^2/2$. Ten times the output is approximately eighteen times the memory-time.
# - Little's law multiplies the concurrency by the output length. The **KV memory** sets the GPU count. A tight
#   **ITL** SLO limits the batch before HBM does.
# - `max_tokens` counts the thinking. Thus a cap truncates answers. A **thinking budget** closes the think block,
#   and then the model answers instead.
# - The templates remove the thinking of earlier turns. Thus the thinking is never a cached prefix.
# - Also, the bill counts every thinking token as output. Thus the metric is **cost per correct answer**, and effort
#   is a thing that you route.
#
# Primer: `../PRIMER.md` §5 and §7.

# %%
import math

import numpy as np

from rlcore import Policy, ThinkTask, pg, workload as w

H100, SMALL = w.GPUS["H100"], w.MISTRAL_SMALL

# %% [markdown]
# ## Worked example 1 — how a model comes to think: rejection sampling, then distillation
# The recipe of DeepSeek-R1 alternates RL with **rejection-sampling SFT**. This method samples many answers, keeps
# the correct answers and fine-tunes on them. On the ThinkTask, correct answers are longer on average, because
# thinking helps. Thus an imitation of the correct answers that the method keeps makes the thinking longer, even
# with no RL at all.

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
# Now **distillation**. An RL-trained teacher (the run of notebook 01) samples traces. Then plain SFT trains a new
# student on these traces, with no reward and no RL.

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
# The student gets the thinking behaviour of the teacher from the outputs of the teacher only. This is why the R1
# distills (Qwen- and Llama-based, 1.5B–70B) used only SFT on ~800k samples. This is also why, in the R1
# comparison, the distillation of a strong model gave better results than RL directly on the small base. The scores
# on AIME 2024 are 72.6 against 47.0 for 32B (primer §5, verify).
#
# ## Worked example 2 — a heavy-tailed length distribution
# Thinking lengths change with the difficulty of the question. The result is a mixture that looks lognormal. The
# median is 1,500 tokens, and $\sigma$ = 1:

# %%
med, sig = 1500, 1.0
print(f"mean {w.lognormal_mean(med, sig):,.0f}; " + ", ".join(
    f"p{int(p * 100)} {w.lognormal_quantile(med, sig, p):,.0f}" for p in (0.5, 0.9, 0.99)))
kv = w.kv_per_token_kb(w.QWEN3_0_6B, "fp16")
print(f"Qwen3-0.6B: {kv:.0f} kB of KV per token ({kv * 1000:,.0f} B); an 8K-token trace holds "
      f"{8192 * kv * 1000 / 1e9:.2f} GB, against {w.weight_gb(w.QWEN3_0_6B, 'fp16'):.2f} GB of weights")

# %% [markdown]
# The mean is 1.6× the median, and the p99 is ten times the median. Set `max_model_len` and the preemption headroom
# for the tail, not for the mean. Also, remember this: on a small model, the KV of one long trace is the size of
# the model.
#
# ## Worked example 3 — the capacity primer's bank, with and without thinking
# `00-foundations/gpu-capacity-planning` calculates the size of an internal assistant. The workload is 8.33
# requests/s, 1,500 tokens in, 300 out and 40 ms TPOT, with Mistral Small 3 (24B) in FP8 on H100s. `w.plan` gives
# the same numbers as the capacity primer. Then it adds the thing that the `decode_aggregate` of the capacity primer
# does not include: HBM and the ITL SLO limit the batch per GPU.
#
# Like the primer, `w.plan` gives every output token the TPOT of the SLO. But `w.plan_steady`
# lets a request live as long as the step at which the fleet actually runs. With this model,
# $N = \text{rps} \cdot (\text{TTFT} + \text{out} \cdot \operatorname{step}(b))/b$ GPUs hold a steady batch $b$ per
# GPU (`w.gpus_for_batch`).

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
# Ten times the output tokens needs eighteen times the GPUs for memory. The numbers are 5.12 against 0.28 at the
# TPOT of the SLO, and 2.75 against 0.15 at the step at which the fleet runs. There are two causes. The concurrency
# increases with the output (Little's law). Also, each live session holds 1.8× the KV (3,000 average context
# against 1,650). The decode *throughput* that is necessary is not the binding constraint.
#
# Read the last two rows of `w.plan` carefully. A 20 ms SLO halves the lifetime that the convention assumes. Thus
# it needs *fewer* GPUs (3) than the 40 ms SLO (6). That is an artefact.
#
# At 3 GPUs, the fleet becomes stable at 154 per GPU and 18.5 ms a step, under the two SLOs. Thus `w.plan_steady`
# needs 3 for the two SLOs, and the need of the tight SLO is the larger. With a 20 ms ITL SLO, the step time limits
# each GPU to 174 sessions. This is less than the 195 sessions that fit in HBM. Thus ITL binds first.
#
# ## Worked example 4 — KV working set: memory × time
# A request holds ${P + t}$ tokens of KV at decode step $t$, for $L$ steps:
#
# $$
# \Sigma = P \cdot L + \frac{L(L+1)}{2}.
# $$

# %%
base_kv = w.kv_token_steps(1500, 300)
for out in (300, 1000, 1500, 3000, 8000):
    print(f"output {out:5,d}: {w.kv_token_steps(1500, out):>12,d} KV-token-steps = {w.kv_token_steps(1500, out) / base_kv:5.1f}× the 300-token answer")

# %% [markdown]
# ## Worked example 5 — `max_tokens` is a cap, a thinking budget is a budget
# Each question needs $L_{\text{req}} \sim \operatorname{lognormal}(1{,}500, \sigma = 1)$ thinking tokens. The
# answer is 300 tokens. With `max_tokens`, a request can get to the cap before its thinking stops. Then the request
# returns no answer (`finish_reason="length"`, empty content).
#
# Budget forcing uses vLLM's `thinking_token_budget` or Qwen's two-call recipe. With budget forcing, the think block closes at the budget. Then
# the model answers with what it has. Here this answer is correct 30% of the time ($e_0$ = 0.7).

# %%
print(f"{'limit':>7} {'max_tokens: accuracy':>21} {'truncated':>10} {'budget: accuracy':>17} {'mean output':>12}")
for cap in (2048, 4096, 8192, 16384):
    cut = w.budget_outcome(med, sig, answer_tokens=300, e0=0.7, max_tokens=cap)
    forced = w.budget_outcome(med, sig, answer_tokens=300, e0=0.7, budget=cap - 300)
    print(f"{cap:7,d} {cut['accuracy']:21.3f} {cut['truncated']:10.3f} {forced['accuracy']:17.3f} {forced['tokens']:12,.0f}")

# %% [markdown]
# The tokens are the same, but the outcome is different. At 4K, one request in six gets no answer at all under
# `max_tokens`. Set `max_tokens` (and `max_model_len`) to a large value. Use a thinking budget to limit the cost.
# Note the thing that does **not** work for this on Qwen3. A `reasoning_effort` other than "none" only switches
# thinking on (primer §5, verify).
#
# ## Worked example 6 — prefix caching when the template drops old thinking
# The templates of Qwen3 and gpt-oss render earlier assistant turns *without* their thinking. The KV of turn $N$
# holds prompt + thinking + answer. The prompt of turn ${N+1}$ contains only the answer. Thus the cache hit ends
# where the prompt of turn $N$ ended, rounded down to a 16-token block. The engine prefills the answer again. In one
# tool-calling loop, the template keeps the thinking, and all of the previous sequence is a hit.

# %%
turns = [(100, 800, 200)] * 4                          # (user, thinking, answer) tokens per turn; 1,000-token system prompt
for keep in (False, True):
    rows = w.turn_prefills(1000, turns, keep_thinking=keep)
    print(("thinking kept   " if keep else "thinking dropped"), " ".join(f"[{p:,} prompt, {p - h:,} prefilled]" for p, h in rows))

# %% [markdown]
# When the template removes the thinking, the prompts stay short, because the thinking never goes into the context
# again. But the engine calculates the KV of the thinking (here 800 tokens per turn) and holds it during the decode.
# Then the engine never uses this KV again. Thus the prefix-cache hit rates on multi-turn thinking traffic are lower
# than on the same conversation without thinking. Cache-aware routing (layer 05) still gives a benefit only on the
# stable prefix.
#
# ## Worked example 7 — cost per *correct* answer, and routing by effort
# This is the API cost per call at the 06 scaling lab's example prices (\$1.50 / \$9.00 / \$0.15 per 1M input /
# output / cached input, verify). A call has 5,000 input tokens (2,700 cached) and 350 output tokens, or 3,500 with
# thinking.

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
# Thinking on all requests gives more accuracy at 5× the cost. Thinking only where it gives a benefit keeps almost
# all of the accuracy for a fraction of the cost. That is routing by effort at the gateway (layer 06). It needs a
# classifier, or a low-cost first pass, that knows which requests are hard.
#
# ## Exercise 5.1 — memory × time
# Write `kv_steps(P, L)`. It gives the KV-token-steps of a request with a $P$-token prompt and $L$ output tokens.
# Then calculate `ratio`: a 3,000-token output against a 300-token output on a 1,500-token prompt.

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
# This is the bank with thinking. It has 3,000 output tokens, 1,500 in, FP8 and an 80 GB H100 with 10% kept in
# reserve. It also has 24 GB of weights and 81.92 kB of KV per token. In FP8, this is 2 × 40 layers × 8 KV heads ×
# 128 × 1 byte. As in `capacity.py`, kB = 1,000 bytes and GB = 10⁹. Calculate these values with the convention of
# the capacity primer:
#
# - `live`: Little's law, with duration = TTFT + 3,000 × 40 ms.
# - `per_gpu`: the sessions per GPU at the average context 1,500 + 3,000/2.
# - `gpus` = live / per_gpu.
#
# Then do not use the convention. With `per_gpu` sessions in the batch of each GPU, a decode step takes
# `step = (24 + per_gpu × KV per session) / 3,350` seconds (H100: 3.35 TB/s, memory-bound). A request lives
# TTFT + 3,000 × step. The fleet that holds that batch steadily is `steady_gpus` = rps × lifetime / per_gpu.

# %% exercise
ttft = w.ttft_s(24, 1500, H100)
### BEGIN SOLUTION
live = rps * (ttft + 3000 * 0.040)
per_gpu = (80 * 0.9 - 24) / (81.92 * 3000 / 1e6)
gpus = live / per_gpu
step = (24 + per_gpu * 81.92 * 3000 / 1e6) / 3350
steady_gpus = rps * (ttft + 3000 * step) / per_gpu
### END SOLUTION

# %% check
p = w.plan(SMALL, H100, rps, 1500, 3000)
assert abs(live - p["concurrency"]) < 1e-6 and abs(per_gpu - p["sessions_per_gpu"]) < 1e-6
assert round(gpus, 2) == 5.12
assert abs(steady_gpus - w.plan_steady(SMALL, H100, rps, 1500, 3000)["gpus"]["memory"]) < 1e-6
print(f"✅ {live:,.0f} live sessions / {per_gpu:.1f} per GPU = {gpus:.2f} GPUs — vs 0.28 without thinking. With a full "
      f"batch a step takes {step * 1e3:.1f} ms, not 40, so {steady_gpus:.2f} GPUs hold it steadily (vs 0.15): still 18×")

# %% [markdown]
# ## Exercise 5.3 — choose `max_model_len`
# Prompts are 1,500 tokens, answers are 300 tokens, and thinking is $\operatorname{lognormal}(1{,}500, \sigma = 1)$.
# Select the smallest `max_model_len` that is a multiple of 1,024 and truncates at most 1% of requests. (vLLM's
# `max_model_len` sets the limit for prompt + output.)

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
# The system prompt is 1,000 tokens. Turn 1 has user 100 (+ a 4-token assistant header), thinking 800 and answer
# 200. The template removes the thinking of turn 1. Turn 2 has user 100 (+ header). Predict these values:
#
# - `prompt2`: the prompt length of turn 2.
# - `hit2`: the tokens that come from the cache, in full 16-token blocks only.
# - `prefill2`.

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
# Use `acc`, `share`, `no_think` and `thinking` from worked example 7. Set `best_policy` to the routing with the
# lowest cost per correct answer. The routing is a dictionary `{"easy": ..., "hard": ...}` with the values "on" or
# "off". The overall accuracy must be at least 0.90.

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
# From `base`, sample 512 completions with `rng5`. Keep the correct completions. Then train `rs_policy` on them
# with 20 SFT steps (lr 1.0). The mean thinking length of `rs_policy` must increase.

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
# **The two-minute version.** "Thinking models spend output tokens before they answer. RL with verifiable rewards
# taught them this behaviour, and distillation copies the behaviour into small models. For serving, this changes a
# prefill-heavy chat workload into a decode-heavy workload with a heavy tail. Our p99 thinking length is ten times
# the median.
#
# "The KV grows while a request thinks. Thus the memory-time per request goes as $P \cdot L + L^2/2$. Thus, at the
# same arrival rate, 10× the output is ~18× the GPUs for memory. This calculation uses the capacity primer's
# formulas, with HBM and the ITL SLO as the limits of the batch. When the ITL SLO is tight, it binds first.
#
# "We calculate the size at the step at which the fleet actually runs, not at the TPOT of the SLO. If we do not,
# the looser SLO seems to have a higher cost. We set max_model_len for the tail. We control the cost with a
# thinking budget, not with max_tokens. The max_tokens limit counts thinking and truncates answers.
#
# "The templates remove old thinking. Thus multi-turn
# prefix-cache hits stop at the prompt of each turn. The bill counts every thinking token as output. Thus we monitor
# cost per correct answer, and we route effort per request class at the gateway."
#
# **Drill questions**
# 1. *We enabled thinking and p99 ITL doubled at the same QPS. Why?* The concurrency increased with the output
#    length (Little's law), and the KV of each session grew. Thus decode batches are larger and read more KV per
#    step. If HBM becomes full, preemptions add recomputation. The thing that changed is the capacity for ITL, not
#    tokens/s.
# 2. *Users report empty answers from the thinking model. Where do you look?* Look for `finish_reason="length"`
#    with empty `content`. The request used all of `max_tokens` inside the think block. Increase the cap and use a
#    thinking budget.
# 3. *Our multi-turn prefix-cache hit rate fell when we switched to a thinking model. Bug?* This is the expected result. The
#    template removes the thinking of earlier turns. Thus the KV of each turn becomes different from the next
#    prompt immediately after the prompt of that turn. The engine prefills the answer again and never uses the
#    thinking KV again.
