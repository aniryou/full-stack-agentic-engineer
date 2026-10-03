# %% [markdown]
# # 05 · RL rollouts with an inference engine: the bookkeeping of one GRPO step
#
# **Tier:** T1: vLLM generates $G$ = 8 rollouts for each of 8 prompts with `Qwen/Qwen2.5-0.5B-Instruct` (the
# quick-start model of TRL). A verifier scores them, and transformers takes one GRPO step on a T4. The cell at the
# end runs only with a GPU, `vllm` and `transformers`. `deploy/any-gpu/rl_step.sh` wraps it.
#
# T0 (default): the same bookkeeping on the rollouts of the tiny transformer. With torch, a bfloat16 "engine" copy
# generates them now, and a float32 "trainer" copy scores them again. Without torch, the notebook uses a recorded
# set (illustrative).
#
# ## The one-minute version
#
# * In RL for LLMs, the **rollout is an inference workload**. An engine inside the training loop generates $G$
#   completions per prompt. This phase takes the largest part of the step time. The docs of verl report about 70% of
#   the total time for DAPO training of a 32B model. All that the serving layers taught applies: batching, KV
#   capacity and long tails (PRIMER §8 "The RL training stack in brief").
# * The trainer does this bookkeeping in each step:
#   1. It groups the rollouts by prompt.
#   2. It scores them with the verifier.
#   3. It calculates the group-normalised advantages.
#   4. It drops the groups with no signal (the dynamic sampling of DAPO).
#   5. It recomputes the log-probs with the weights of the trainer.
#   6. It corrects for the **train–inference mismatch**. The kernels and the precision of the engine never match
#      those of the trainer exactly.
#   7. It calculates the loss.
#   8. It does the update.
#   9. It **pushes the weights back** to the engine.
# * Two systems costs come from this. First, a synchronous rollout batch waits for its longest completion. Thus
#   heavy-tailed thinking lengths leave GPUs idle. This is why one-step-off RL and fully async RL exist. Second, the
#   weight sync sends every parameter in every step, unless the trainer sends only deltas.

# %%
import json, math, statistics
from importlib import resources
from thinklab import engine, env
from thinklab.report import table
from thinklab.rollout import (Rollout, dynamic_sampling, frac_zero_std, group_advantages, group_by_prompt, is_ratios,
                              rollout_phase, soft_overlong, trl_grpo_config, weight_sync_bytes)

print(env.describe())
if env.has_torch():
    from thinklab.tinyrl import train as T
    model, task = T.warm_start(T.TinyRLConfig(), log=lambda *a: None)       # SFT warm-up only, ~30 s on a CPU
    RAW = T.make_rollouts(model, task, prompts=8, generations=8, seed=0)
    LABEL = "MEASURED now: bf16 engine copy vs fp32 trainer copy of the tiny transformer"
else:
    data = json.loads(resources.files("thinklab.data").joinpath("tinyrl_recorded_rollouts.json").read_text())
    RAW, LABEL = data["rollouts"], data["source"]
ROLLOUTS = [Rollout(r["prompt_id"], r["length"], r["reward"], r["sampler_logps"], r["trainer_logps"]) for r in RAW]
print(LABEL, "|", len(ROLLOUTS), "rollouts")
groups = group_by_prompt(ROLLOUTS)
print(table([{"prompt": k, "rewards": "".join(str(int(r.reward)) for r in g), "mean": round(statistics.fmean(r.reward for r in g), 3),
              "lengths": " ".join(str(r.tokens) for r in g)} for k, g in groups.items()], title="8 prompts x G = 8 rollouts"))

# %% [markdown]
# ## Exercise 5.1 — from rollouts to a training batch
#
# Group the rollouts by prompt. Then return `(advantages, kept)`:
#
# * `advantages`: a map from the prompt id to the list of group-normalised advantages, `(r − mean) / (std + 1e-4)`
#   with Bessel's std, in rollout order.
# * `kept`: the prompt ids that the dynamic sampling of DAPO keeps. These are the groups whose rewards are *not* all
#   equal. The other groups give zero gradient. In a full trainer, new prompts replace them.

# %% exercise
def grpo_batch(rollouts: list) -> tuple:
    ### BEGIN SOLUTION
    groups = {}
    for r in rollouts:
        groups.setdefault(r.prompt_id, []).append(r.reward)
    adv, kept = {}, []
    for pid, rs in groups.items():
        m = sum(rs) / len(rs)
        s = math.sqrt(sum((x - m) ** 2 for x in rs) / (len(rs) - 1)) if len(rs) > 1 else 0.0
        adv[pid] = [(x - m) / (s + 1e-4) for x in rs]
        if s > 0:
            kept.append(pid)
    return adv, kept
    ### END SOLUTION

# %% check
adv, kept = grpo_batch(ROLLOUTS)
for pid, g in groups.items():
    assert all(abs(a - b) < 1e-9 for a, b in zip(adv[pid], group_advantages([r.reward for r in g])))
assert sorted(kept) == sorted(dynamic_sampling(groups))
toy = [Rollout("a", 3, 1.0), Rollout("a", 3, 0.0), Rollout("b", 3, 1.0), Rollout("b", 3, 1.0)]
assert grpo_batch(toy)[1] == ["a"] and grpo_batch(toy)[0]["b"] == [0.0, 0.0]
print(f"✅ {len(kept)} of {len(groups)} groups carry signal (frac_reward_zero_std = {frac_zero_std(groups):.2f})")

# %% [markdown]
# ## Exercise 5.2 — the train–inference mismatch, and TRL's default correction
#
# The engine reports the log-probability of each token that it sampled (`sampler_logps`). The trainer recomputes
# them with its own weights and kernels (`trainer_logps`). You expect them to be equal, but they are not.
#
# Implement the default correction of TRL, `vllm_importance_sampling_mode="sequence_mask"`. It calculates one ratio per
# sequence, `exp(Σ (trainer − sampler))`. If the ratio is in `[c_min, c_max]`, it keeps the ratio. If not, it sets
# the ratio to 0, and thus masks the sequence out of the loss.
#
# Return the per-token weights (the same value for every token of the sequence). The default `c_max` of TRL is 3.0,
# with no lower bound.

# %% exercise
def seq_mask_weights(sampler: list, trainer: list, c_max: float = 3.0, c_min: float | None = None) -> list:
    ### BEGIN SOLUTION
    ratio = math.exp(sum(t - s for s, t in zip(sampler, trainer)))
    lo = -math.inf if c_min is None else c_min
    w = ratio if lo <= ratio <= c_max else 0.0
    return [w] * len(sampler)
    ### END SOLUTION

# %% check
assert seq_mask_weights([-1.0, -1.0], [-1.0, -1.0]) == [1.0, 1.0]
assert seq_mask_weights([-2.0], [0.0]) == [0.0]                          # e^2 = 7.4 > 3: masked
assert seq_mask_weights([-1.0], [-1.5], c_min=0.7) == [0.0]              # e^-0.5 = 0.61 < 0.7: masked
for r in ROLLOUTS:
    assert all(abs(a - b) < 1e-9 for a, b in zip(seq_mask_weights(r.sampler_logps, r.trainer_logps),
                                                   is_ratios(r.sampler_logps, r.trainer_logps, "sequence_mask")))
diffs = [abs(t - s) for r in ROLLOUTS for s, t in zip(r.sampler_logps, r.trainer_logps)]
w = [seq_mask_weights(r.sampler_logps, r.trainer_logps)[0] for r in ROLLOUTS]
print(f"✅ [{LABEL.split(':')[0]}] mean |log p_trainer - log p_sampler| = {statistics.fmean(diffs):.4f} per token "
      f"(max {max(diffs):.4f}); sequence weights in [{min(w):.3f}, {max(w):.3f}]")

# %% [markdown]
# On the tiny model, the only difference between the two copies is bfloat16 against float32. The mismatch is
# already visible. With a real engine, the mismatch also has these causes:
#
# * different attention kernels,
# * batch-size-dependent reductions,
# * sampling tricks such as top-k or FP8 KV,
# * weights that are one step stale.
#
# TRL logs it as `sampling/sampling_logp_difference/mean`.
#
# ## Worked example: the rollout phase is a serving problem with a straggler
#
# A synchronous GRPO step generates the full batch, and then trains. The batch gets smaller as completions finish.
# The phase ends only when the *longest* completion finishes. With thinking-length tails, this leaves most sequence
# slots idle at the end. The next cell takes 512 rollouts (64 prompts × $G$ = 8), with lengths from the simulated
# thinking model. It decodes them on the Qwen3-0.6B/T4 step model of the engine emulator, at the size of the batch
# that still runs (**simulated**).

# %%
from thinklab.workload import sample_lengths
from thinklab.thinking.evalset import make_evalset
PROF = engine.profile("t4-qwen3-0.6b")
qs = [p.prompt for p in make_evalset(64, seed=9) for _ in range(8)]
SAMPLES = sample_lengths(qs, seed=0)                              # (reasoning, answer, correct) per rollout
lens = [min(r + a, 8000) for r, a, _ in SAMPLES]
avg_ctx = 60 + statistics.fmean(lens) / 2
step = lambda b: PROF.decode_step_s(b, b * avg_ctx)              # noqa: E731 — the roofline step at batch b
ph = rollout_phase(lens, step, overlap_training_s=20.0)
print(table([{"rollouts": len(lens), "mean tokens": round(ph["mean"]), "longest": ph["longest"],
              "phase s": round(ph["phase_s"], 1), "idle slot share": round(ph["idle_share"], 2),
              "sync step s (+20 s train)": round(ph["sync_step_s"], 1),
              "one-step-off step s": round(ph["overlapped_step_s"], 1)}], title="[SIMULATED] one rollout phase"))

# %% [markdown]
# ## Exercise 5.3 — pick a generation cap with DAPO's overlong shaping
#
# A lower `max_completion_length` cuts the straggler tail of the worked example. But DAPO has a soft overlong
# penalty (rl-core notebook 03, exercise 3.5, and `soft_overlong` here). It charges the completions in the last
# `cache` tokens before the cap. It gives −1 to the completions after the cap. This includes completions that get to a
# correct answer when they have sufficient room. The cap is a reward-shaping decision, not only a systems decision.
#
# Use the proportions of DAPO: cache = cap / 5, as 4,096 of 20,480. For each cap in `CAPS`, `overlong_view` returns
# two shares:
#
# * `truncated`: the share of these 512 simulated rollouts that are longer than the cap.
# * `hit_correct`: take the rollouts that are correct in the uncapped outcome of the simulated model. `hit_correct`
#   is the share of these rollouts that get a penalty below 0.
#
# Then set `cap` to the smallest cap in `CAPS` that gives a penalty to at most 5% of these correct rollouts.

# %% exercise
CAPS = (1024, 2048, 4096, 6144, 8192)
RAW_LENS, CORRECT = [r + a for r, a, _ in SAMPLES], [c for _, _, c in SAMPLES]

def overlong_view(lengths: list, correct: list, cap: int) -> dict:
    ### BEGIN SOLUTION
    pen = [soft_overlong(n, cap, cap // 5) for n in lengths]
    return {"cap": cap, "truncated": statistics.fmean(n > cap for n in lengths),
            "hit_correct": statistics.fmean(x < 0 for x, c in zip(pen, correct) if c)}
    ### END SOLUTION

cap = None
### BEGIN SOLUTION
cap = min(v["cap"] for v in (overlong_view(RAW_LENS, CORRECT, c) for c in CAPS) if v["hit_correct"] <= 0.05)
### END SOLUTION

# %% check
v = overlong_view([100, 90, 50, 120], [True, True, True, False], 100)       # cache 20: 100 → −1, 90 → −0.5, 50 → 0
assert v == {"cap": 100, "truncated": 0.25, "hit_correct": 2 / 3}
views = [overlong_view(RAW_LENS, CORRECT, c) for c in CAPS]
print(table([{k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()} for v in views],
            title="[SIMULATED] DAPO's soft overlong penalty at cache = cap / 5"))
assert cap == min(v["cap"] for v in views if v["hit_correct"] <= 0.05) == 6144
tight = next(v for v in views if v["cap"] == 2048)
print(f"✅ cap {cap:,}: at 2,048 the penalty would push against {tight['hit_correct']:.0%} of the reasoning that was "
      "on its way to a right answer, and teach the policy to stop early where it should not")

# %% [markdown]
# ## Exercise 5.4 — the idle share of a synchronous rollout batch
#
# Assume a constant step time, so only the *shape* of the length distribution is important. All sequences start at
# the same time, and each sequence holds its slot until it finishes. The batch reserves `n × max(lengths)`
# slot-steps and uses `Σ lengths` of them. Return the idle share, `1 − Σ lengths / (n × max)`. Then answer this
# question: for the lengths of the worked example, what share of the slots of the batch is idle?

# %% exercise
def idle_share(lengths: list) -> float:
    ### BEGIN SOLUTION
    return 1 - sum(lengths) / (len(lengths) * max(lengths))
    ### END SOLUTION

# %% check
assert idle_share([10, 10, 10]) == 0.0 and abs(idle_share([100, 200, 1000]) - (1 - 1300 / 3000)) < 1e-12
assert abs(idle_share(lens) - rollout_phase(lens, lambda b: 1.0)["idle_share"]) < 1e-9
print(f"✅ {idle_share(lens):.0%} of the slot-steps are idle while the tail finishes; "
      f"a thinking budget (or async rollouts) attacks exactly this")

# %% [markdown]
# ## Exercise 5.5 — what goes back to the engine every step
#
# After each optimizer step, the engine needs the new weights. Return these values:
#
# * the bytes of a full sync,
# * the bytes of a delta sync when only `changed` of the bf16 weight bytes change. The measurement of verl shows
#   that more than 99% do not change from one step to the next.
# * the seconds that each sync takes at `link_gbs` gigabytes per second.
#
# Use the 494 M parameters of Qwen2.5-0.5B-Instruct.

# %% exercise
def sync_cost(params: float, changed: float, link_gbs: float, bytes_per_param: float = 2.0) -> dict:
    ### BEGIN SOLUTION
    full = params * bytes_per_param
    delta = full * changed
    return {"full_bytes": full, "delta_bytes": delta, "full_s": full / (link_gbs * 1e9), "delta_s": delta / (link_gbs * 1e9)}
    ### END SOLUTION

# %% check
c = sync_cost(494e6, 0.02, 16.0)                              # ~PCIe Gen4 x16 order of magnitude (verify for your box)
assert abs(c["full_bytes"] - 988e6) < 1 and abs(c["delta_bytes"] - 19.76e6) < 1
assert abs(c["full_s"] - weight_sync_bytes(494e6) / 16e9) < 1e-12 and c["delta_s"] < c["full_s"] / 49
big = sync_cost(32e9, 0.02, 16.0)
print(f"✅ 0.5B: {c['full_bytes'] / 1e9:.2f} GB per step ({c['full_s']:.3f} s) vs {c['delta_bytes'] / 1e6:.0f} MB delta; "
      f"a 32B policy: {big['full_bytes'] / 1e9:.0f} GB = {big['full_s']:.1f} s per step at 16 GB/s")

# %% [markdown]
# ## On a real GPU (T1): vLLM as the rollout generator for one GRPO step
#
# ```bash
# pip install -q "vllm==0.30.0" "transformers>=4.56.2"      # optional: "trl==1.14.0" for GRPOTrainer
# python -m thinklab rl-step --model Qwen/Qwen2.5-0.5B-Instruct --prompts 8 -g 8
# ```
#
# `thinklab.rollout.one_grpo_step` does these steps:
#
# 1. It loads the model in vLLM with `gpu_memory_utilization=0.3`. Thus the trainer copy fits next to it on a 15 GB
#    T4 (verify).
# 2. It generates 8 × 8 rollouts with `logprobs=0`, that is, the log-prob of the sampled token at every position.
# 3. It scores them with the eval-set verifier.
# 4. It calculates the advantages.
# 5. It recomputes the log-probs with a float32 transformers copy.
# 6. It applies a sequence-level importance weight, masked above 3 (`is_ratios(..., mode="sequence_mask")`, the
#    function from Exercise 5.2 and the default of TRL).
# 7. It takes one SGD step.
#
# It reports rollout seconds against training seconds, the reward, the zero-std share and the sampler/trainer
# log-prob gap. It does not push the weights back into vLLM. The colocate mode of TRL does that in every step. The
# configuration in the next cell is where you start with TRL on a T4.

# %%
print(json.dumps(trl_grpo_config("T4"), indent=1))
if env.gpu_name() and env.has_vllm():
    from thinklab.rollout import one_grpo_step
    one_grpo_step("Qwen/Qwen2.5-0.5B-Instruct", n_prompts=8, n=8)
else:
    print("T0: no GPU with vLLM here; the rollout bookkeeping above is the part that transfers. "
          "On a T4: python -m thinklab rl-step (deploy/any-gpu/rl_step.sh)")

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "Our RL loop is an inference service with a trainer attached. In each step, vLLM generates $G$ =
# 8 completions per prompt and returns their log-probs. A verifier scores them. The trainer normalises the rewards
# in each group. It drops the groups where every sample agrees, because they carry no gradient.
#
# "The trainer recomputes the log-probs. The gap to the log-probs of the engine comes from kernels, precision and
# stale weights. A sequence-level importance weight, masked above 3, corrects this gap.
#
# "Rollouts are most of the step time. A synchronous batch waits for its longest completion. Thus, with
# thinking-length tails, most slots are idle at the end. Budgets, overlong penalties and one-step-off rollouts
# attack that.
#
# "The new weights go back to the engine in every step. A full copy is the size of the model. Delta sync sends ~50×
# fewer bytes when 2% of the weights change (Exercise 5.5). But verl measured 1.3–21× less wall time. This is
# because the work to calculate and encode the diff and the constant overheads stay (PRIMER §8 "The RL training
# stack in brief", verify). The serving skills transfer directly: batch shape, KV capacity and tail latency."
#
# **Drill 1.** *Half of our groups have all-correct rewards. Is that a problem?* Those prompts are too easy for the
# current policy, and they give no gradient. Filter them out (dynamic sampling), increase the difficulty, or accept
# a smaller effective batch. TRL logs the share as `frac_reward_zero_std`.
#
# **Drill 2.** *Why not trust the log-probs of vLLM as the old policy?* They come from different kernels, batch
# shapes and sometimes stale weights. Thus they are different from the log-probs of the trainer. The trainer
# recomputes them, and the ratio corrects the update (`vllm_importance_sampling_correction` of TRL, on by default).
#
# **Drill 3.** *Rollouts are 70% of step time, and more GPUs do not help. Why?* The longest completions set the
# limit of the phase, not the throughput. Overlap generation with training (one-step-off or fully async RL, which
# accept some staleness). Or put a cap on the lengths (budgets, overlong shaping), or pack new prompts into the
# slots that become free.
