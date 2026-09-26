# %% [markdown]
# # 01 · GRPO on a tiny transformer: a model discovers that thinking pays
#
# **Tier:** T0 with torch on a laptop CPU (the training run takes about a minute; this notebook was
# checked on a shared 4-core container at ~40 s for the run and ~1.5 min in all). T1 (any GPU) runs the
# same code faster, which this model does not need. Without torch the notebook still runs: the
# training cells show a recorded run, labelled illustrative, and the torch exercises say they were skipped.
#
# ## The one-minute version
#
# * **RL for a language model is "sample, score, reweight".** Sample completions from the model,
#   score each with a reward, and raise the probability of the tokens in completions that scored
#   above the baseline — the policy gradient over token sequences (PRIMER §2 "Policy gradients over
#   token sequences").
# * **GRPO** makes the baseline the *group*: sample G completions per prompt, and a completion's
#   advantage is its reward minus the group mean, divided by the group's standard deviation. No
#   value model is trained (PRIMER §4 "RL with verifiable rewards and GRPO").
# * **Verifiable rewards** need no reward model: a program checks the final answer. Here the task
#   is the last digit of a sum of six base-5 digits. A 2-layer transformer cannot add six numbers in
#   the one token it emits for the answer, but it can if it first writes the running sums in a
#   `<think>` scratchpad. Each thinking token adds one more step of serial computation.
# * **What you will watch:** a short supervised warm-up shows the model both behaviours, answering
#   straight away and thinking first, half the time each. Then GRPO, rewarded only for correct
#   answers, drives the thinking rate toward 100%, and the reward and the completion length rise
#   *together*. This is R1-style rather than R1-Zero: a cold-start SFT shows the behaviour, and RL
#   makes the model *choose* it (PRIMER §5 "Thinking models"). R1-Zero had no SFT and its traces grew
#   gradually; `sft_mix="uniform"` (end of the notebook) is the closer analogue here.
# * **What you will build:** the GRPO loss in torch, which you then train with, and a reading of the
#   KL curve that explains its spikes.

# %%
import math, random
from thinklab import env
from thinklab.report import plot, table
import statistics
from dataclasses import replace
from thinklab.rollout import k3
from thinklab.tinyrl.task import EOS, END_THINK, THINK, DigitSum, render, reward
from thinklab.tinyrl.curves import load_recorded, show

print(env.describe())
HAVE_TORCH = env.has_torch()
print("torch available:", HAVE_TORCH, "-> training cells run for real" if HAVE_TORCH else "-> recorded run (illustrative)")

# %% [markdown]
# ## Worked example: the task, three completions and the verifier
#
# Six digits in base 5, then `=`. A completion opens `<think>`, optionally writes running sums
# (mod 5), closes `</think>`, gives the answer and `<eos>`. The verifier checks only the format and
# the final answer, never the scratchpad. That makes it an *outcome* reward, like R1's rule-based
# accuracy reward.

# %%
task = DigitSum(k=6, base=5)
p = task.sample(random.Random(7))
print("prompt            ", render(p.prompt), "   answer:", p.answer)
for j in (0, 3, 6):
    c = task.demo(p, j)
    print(f"scratchpad {j} steps ", render(c).ljust(46), "reward", reward(p, c))
wrong = [THINK, END_THINK, (p.answer + 1) % 5, EOS]
broken = [THINK, 3, 1, (p.answer), EOS]                    # never closed the thinking block
print("wrong answer      ", render(wrong).ljust(46), "reward", reward(p, wrong))
print("broken format     ", render(broken).ljust(46), "reward", reward(p, broken))

# %% [markdown]
# ## Exercise 1.1 — write the verifier
#
# Return 1.0 when `completion` is `<think>`, zero or more digit tokens (0–9), `</think>`, the
# answer digit, `<eos>`, in that order and nothing else before `<eos>` (tokens after `<eos>` are
# padding and ignored). The answer must equal `problem.answer`. Return 0.0 otherwise. Token ids:
# digits are 0–9, `THINK`, `END_THINK` and `EOS` are imported above.

# %% exercise
def my_reward(problem, completion: list) -> float:
    ### BEGIN SOLUTION
    toks = list(completion)
    if EOS in toks:
        toks = toks[: toks.index(EOS) + 1]
    if len(toks) < 4 or toks[0] != THINK or toks[-1] != EOS or END_THINK not in toks:
        return 0.0
    end = toks.index(END_THINK)
    body, tail = toks[1:end], toks[end + 1:]
    if any(t > 9 for t in body) or len(tail) != 2 or tail[0] > 9:
        return 0.0
    return 1.0 if tail[0] == problem.answer else 0.0
    ### END SOLUTION

# %% check
rng = random.Random(0)
for _ in range(300):
    q = task.sample(rng)
    cand = task.demo(q, rng.randrange(7))
    if rng.random() < 0.5:                                  # corrupt a random position
        i = rng.randrange(len(cand))
        cand = cand[:i] + [rng.randrange(15)] + cand[i + 1:]
    assert my_reward(q, cand + [14, 14]) == reward(q, cand + [14, 14]), (render(cand), q)
assert my_reward(p, broken) == 0.0 and my_reward(p, task.demo(p, 2)) == 1.0
print("✅ your verifier agrees with thinklab's on 300 random (partly corrupted) completions")

# %% [markdown]
# ## Worked example: the warm-up, and what a scratchpad is worth to this model
#
# Supervised fine-tuning on demonstrations, half with no scratchpad and half with the full one:
# the "cold start" of the R1 recipe. Then we sample one completion per problem at temperature 1
# and split accuracy by the scratchpad length the model *chose*. With torch this is measured now,
# on this machine. Without torch it is the recorded run.

# %%
if HAVE_TORCH:
    import torch
    from thinklab.tinyrl import train as T
    cfg = T.TinyRLConfig()
    RUN = T.run(cfg, log=lambda *a: None)            # SFT warm-up + GRPO; ~1 min on a laptop CPU
else:
    RUN = load_recorded()
b = RUN["before"]
rows = [{"scratchpad": j, "samples": b["scratch_hist"].get(str(j), b["scratch_hist"].get(j, 0)),
         "accuracy": b["acc_by_scratch"].get(str(j), b["acc_by_scratch"].get(j))} for j in (0, 6)]
LABEL = "MEASURED on this machine" if RUN["source"] == "measured" else RUN["source"]
print(table(rows, title=f"[{LABEL}] after SFT, one sample per problem (of {RUN['config']['eval_prompts']})"))
print(f"chance = 1/base = {1 / task.base:.2f}")

# %% [markdown]
# Without the scratchpad the model is at chance: it cannot compute a six-term sum in one step.
# With it, it is nearly always right. The model already *can* think. It just does not yet
# *choose* to. Nothing in SFT told it which behaviour is better; the verifier will.
#
# The group-relative advantage, (r − mean) / (std + 1e-4) with Bessel's std, is derived and
# implemented in rl-core notebook 03 (exercise 3.1); here `train.group_advantages` computes it in
# torch. When one completion in four is right it gets +1.4997 and each wrong one −0.4999: the rare
# success is what the update is about. A group whose rewards are all equal gets zero advantages.
#
# ## Exercise 1.2 — predict the reward before RL starts, and where it can end
#
# Suppose that after SFT the model thinks with probability `share`, is right with probability
# `acc_think` when it does and `acc_direct` when it does not. Write the expected reward. Then check
# it against the run. The accuracies come from the table above. The share comes from GRPO's own
# step-0 batch: every completion is 4 tokens without a scratchpad or 10 with one, so the batch's
# mean length gives its thinking share, `(length − 4) / 6`. With `share = 1` you get the ceiling RL
# can reach by changing the *choice* alone.

# %% exercise
def expected_reward(share: float, acc_think: float, acc_direct: float) -> float:
    ### BEGIN SOLUTION
    return share * acc_think + (1 - share) * acc_direct
    ### END SOLUTION

# %% check
acc = {int(k): v for k, v in b["acc_by_scratch"].items()}
share0 = (RUN["rl"][0]["length"] - 4) / 6                  # thinking share of the step-0 batch
pred0 = expected_reward(share0, acc[6], acc[0])
ceiling = expected_reward(1.0, acc[6], acc[0])
assert expected_reward(0.5, 1.0, 0.2) == 0.6 and expected_reward(1, 0.9, 0.1) == 0.9
assert abs(pred0 - RUN["rl"][0]["reward"]) < 0.15, (pred0, RUN["rl"][0]["reward"])   # 128 samples: some noise
print(f"✅ step-0 batch thought {share0:.0%} of the time: predicted reward {pred0:.3f} vs logged {RUN['rl'][0]['reward']:.3f}; "
      f"ceiling from choosing to think: {ceiling:.3f}")

# %% [markdown]
# ## Worked example: GRPO, and the curves this run produced
#
# `RUN` already holds the whole training run: 40 steps of 16 prompts × G = 8 samples, `beta = 0`
# (TRL's default: no reference-model penalty in the loss; the KL to the SFT model is still
# *logged*), `loss_type = "dapo"`, `num_iterations = 1`.

# %%
print(show(RUN))
if plot(RUN["rl"], "step", ["reward", "length", "frac_zero_std", "kl"], title=LABEL):
    pass

# %% [markdown]
# Did *this* run learn? The next cell compares the evaluation before and after RL and the first
# and last RL steps, and says so either way. Nothing below assumes the answer.

# %%
rl, before, after = RUN["rl"], RUN["before"], RUN["after"]
K = RUN["config"]["k"]
think = lambda ev: (ev["scratch_hist"].get(str(K), ev["scratch_hist"].get(K, 0))) / RUN["config"]["eval_prompts"]
last5 = rl[-5:]
summary = {"accuracy": (before["accuracy"], after["accuracy"]),
           "full-scratchpad share": (think(before), think(after)),
           "RL reward (step 0 -> mean of last 5)": (rl[0]["reward"], statistics.fmean(r["reward"] for r in last5)),
           "RL completion length (step 0 -> mean of last 5)": (rl[0]["length"], statistics.fmean(r["length"] for r in last5))}
for name, (x0, x1) in summary.items():
    print(f"{name:48} {x0:7.3f} -> {x1:7.3f}")
LEARNED = (after["accuracy"] > before["accuracy"] + 0.15 and think(after) > think(before) + 0.2
           and summary["RL reward (step 0 -> mean of last 5)"][1] > rl[0]["reward"] + 0.15)
kls = [r["kl"] for r in rl]
top = max(rl, key=lambda r: r["kl"])
print(f"kl: {kls[0]:.3f} at step 0; mean of the last 10 steps {statistics.fmean(kls[-10:]):.3f}; "
      f"largest {top['kl']:.3f} at step {top['step']}")
if LEARNED:
    print(f"[{LABEL}] this run learned to think: accuracy and the scratchpad share rose together.")
else:
    print(f"[{LABEL}] WARNING: this run did NOT converge well on this machine. The curves above are what it "
          "produced; compare them with the recorded run (THINKLAB_NO_TORCH=1) or try another seed.")

# %% [markdown]
# How to read the curves (the numbers are the ones printed above, for whichever run you have):
#
# * **reward** should climb from the SFT mix toward the ceiling of Exercise 1.2, and **length**,
#   the mean completion in tokens, from about 7 (half 4, half 10) toward 10. Longer completions
#   were never rewarded directly; length grows because thinking is *instrumentally* useful, the toy
#   version of R1's response-length growth.
# * **frac_zero_std**, the share of groups whose 8 rewards are all equal, *rises* as the policy
#   gets good. Those groups have zero advantage and teach nothing. That is why DAPO resamples them
#   ("dynamic sampling") and why TRL logs this number.
# * **kl** is TRL's logged k3 estimate over one batch, so it is noisy. In the recorded run it rises
#   over the first ten steps and never settles: it spikes to 0.1–0.4 in some batches, and its
#   largest value is at the last step. Nothing pulls the policy back (`beta = 0`), and Exercise 1.5
#   finds where the spikes come from.
#
# `clip_frac` is 0 at every step. With `num_iterations = 1` the rollouts are used for exactly one
# gradient step, so π_θ = π_old when the loss is computed, the ratio is exactly 1 and the clip never
# binds. Exercise 1.4 changes that.
#
# ## Exercise 1.3 — the GRPO loss, in torch
#
# Write the loss `train.py` minimises. Inputs are `(B, T)` tensors of per-token log-probabilities
# under the current policy (`logp`), the policy that sampled the batch (`old_logp`) and the reference
# (`ref_logp`), a `(B,)` tensor of advantages `adv` and a `(B, T)` float `mask` of live tokens.
# Per token: `ρ = exp(logp − old_logp)`, loss `−min(ρ·A, clip(ρ, 1 − cfg.epsilon, 1 + cfg.epsilon_high)·A)`,
# plus `cfg.beta · k3` when `cfg.beta > 0`, with `k3 = exp(d) − d − 1`, `d = ref_logp − logp`. Then
# aggregate the live tokens as `cfg.loss_type` says: `"grpo"` averages each completion, then the
# completions; `"dr_grpo"` divides the sum by `B · max_len`; `"dapo"` divides it by the number of live
# tokens. The pieces, and the length bias of each aggregation, are derived in rl-core notebook 03
# (exercises 3.2–3.4); here they have to work together, on tensors, with autograd.

# %% exercise
def my_grpo_loss(logp, old_logp, ref_logp, adv, mask, cfg, max_len: int):
    ### BEGIN SOLUTION
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    per_token = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.epsilon, 1 + cfg.epsilon_high) * a)
    if cfg.beta:
        d = ref_logp - logp
        per_token = per_token + cfg.beta * (torch.exp(d) - d - 1)
    if cfg.loss_type == "grpo":
        return ((per_token * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()
    if cfg.loss_type == "dr_grpo":
        return (per_token * mask).sum() / (mask.shape[0] * max_len)
    if cfg.loss_type == "dapo":
        return (per_token * mask).sum() / mask.sum().clamp(min=1)
    raise ValueError(cfg.loss_type)
    ### END SOLUTION

# %% check
if HAVE_TORCH:
    g = torch.Generator().manual_seed(0)
    B, L = 6, 5
    base = -torch.rand(B, L, generator=g) * 3
    old, ref_ = base + 0.4 * torch.randn(B, L, generator=g), base + 0.4 * torch.randn(B, L, generator=g)
    adv = torch.randn(B, generator=g)
    mask = (torch.arange(L)[None, :] < torch.tensor([5, 3, 1, 4, 5, 2])[:, None]).float()
    for lt in ("grpo", "dr_grpo", "dapo"):
        for beta in (0.0, 0.04):
            c = T.TinyRLConfig(loss_type=lt, beta=beta, epsilon=0.2, epsilon_high=0.28)
            x1, x2 = base.clone().requires_grad_(True), base.clone().requires_grad_(True)
            mine = my_grpo_loss(x1, old, ref_, adv, mask, c, L)
            theirs, st = T.grpo_loss(x2, old, ref_, adv, mask, c, L)
            mine.backward(); theirs.backward()
            assert abs(mine.item() - theirs.item()) < 1e-6, (lt, beta, mine.item(), theirs.item())
            assert torch.allclose(x1.grad, x2.grad, atol=1e-7), (lt, beta)
    assert st["clip_frac"] > 0                                   # the test batch really exercises the clip
    print(f"✅ loss and gradient match train.grpo_loss for grpo / dr_grpo / dapo, beta 0 and 0.04, "
          f"with {st['clip_frac']:.0%} of live tokens past the clip")
else:
    print("torch is missing: the torch loss cannot be checked here (the pure-Python per-token loss is "
          "thinklab.rollout.clipped_token_loss; rl-core notebook 03 builds it in numpy)")

# %% [markdown]
# ## Exercise 1.4 — train with your loss, and make the clip bind
#
# The check below restarts GRPO from the same SFT model twice for 10 steps, both times minimising
# *your* loss: once with `num_iterations = 1`, once with `num_iterations = 2` (each rollout batch
# is used for two optimizer steps, as TRL does with μ > 1). Before you run it, predict:
#
# * `clip_mu1`: the `clip_frac` the μ = 1 run logs at every step (a number);
# * `clip_binds_mu2`: will the μ = 2 run log `clip_frac > 0` at some step (`True` or `False`)?
#
# `clip_frac` is measured on the last pass over the batch. The check runs both, prints them side by
# side and confirms that your loss actually trains the model.

# %% exercise
clip_mu1 = None
clip_binds_mu2 = None
### BEGIN SOLUTION
clip_mu1 = 0.0          # one step per batch: π_θ = π_old when the loss is taken, ρ = 1 exactly
clip_binds_mu2 = True   # the second pass sees weights one step away from the ones that sampled
### END SOLUTION

# %% check
assert clip_mu1 == 0.0 and clip_binds_mu2 is True
if HAVE_TORCH:                                                   # the runs: ~5 s on a laptop CPU
    SFT_MODEL, TASK = T.warm_start(cfg, log=lambda *a: None)    # RUN's SFT model (reused, not retrained)
    EXP = {mu: T.grpo_from(SFT_MODEL, TASK, replace(cfg, rl_steps=10, num_iterations=mu), loss_fn=my_grpo_loss,
                           log=lambda *a: None) for mu in (1, 2)}
    print(table([{"step": a["step"], "reward μ=1": a["reward"], "reward μ=2": b["reward"],
                  "clip_frac μ=1": a["clip_frac"], "clip_frac μ=2": b["clip_frac"]}
                 for a, b in zip(EXP[1]["rl"], EXP[2]["rl"])], title="MEASURED: 10 GRPO steps with your loss"))
else:
    EXP = None
    print("torch is missing: no runs (the recorded run above has clip_frac 0 at every step, with μ = 1)")
if EXP:
    assert all(r["clip_frac"] == clip_mu1 for r in EXP[1]["rl"])
    assert any(r["clip_frac"] > 0 for r in EXP[2]["rl"]) == clip_binds_mu2
    gain = statistics.fmean(r["reward"] for r in EXP[1]["rl"][-3:]) - EXP[1]["rl"][0]["reward"]
    assert gain > 0.1, f"your loss did not train: reward rose only {gain:+.3f} in 10 steps"
    peak = max(EXP[2]["rl"], key=lambda r: r["clip_frac"])
    print(f"✅ your loss trains (reward +{gain:.2f} in 10 steps); μ = 2 clips up to {peak['clip_frac']:.1%} of "
          f"tokens, most at step {peak['step']}, when Adam's first update moves the rarest tokens' odds furthest")
else:
    print("✅ predictions match the theory (no torch here to measure them)")

# %% [markdown]
# ## Exercise 1.5 — where the KL spikes come from
#
# After RL the policy almost never takes the no-scratchpad path: the probability it gives `</think>`
# straight after `<think>` has fallen to a fraction of a percent, while the frozen SFT reference
# still gives it about 0.5. When a batch happens to sample that token anyway, its k3 term is huge.
# Using `k3` from `thinklab.rollout` (defined in rl-core notebook 03, exercise 3.2), compute
# `one_token`, the k3 of a token with π_θ = 0.005 and π_ref = 0.5, and `batch_kl`, what that one
# token adds to the batch-mean kl the trainer logs when the batch has 128 completions of about 10
# live tokens each. Then check the claim on the run: steps where *every* completion wrote the full
# scratchpad (mean length exactly 10) should show only the scratchpad's small drift.

# %% exercise
one_token = batch_kl = None
### BEGIN SOLUTION
one_token = k3(math.log(0.005), math.log(0.5))        # e^d − d − 1 with d = ln 100
batch_kl = one_token / (128 * 10)
### END SOLUTION

# %% check
assert abs(one_token - (100 - math.log(100) - 1)) < 1e-9 and abs(batch_kl - one_token / 1280) < 1e-12
late = [r for r in rl if r["step"] >= 10]
all_think = [r["kl"] for r in late if r["length"] >= K + 4 - 1e-9]
mixed = [r["kl"] for r in late if r["length"] < K + 4 - 1e-9]
print(f"one such token: k3 = {one_token:.1f} nats, +{batch_kl:.3f} on the logged batch mean")
if all_think and mixed:
    print(f"steps >= 10: every completion thought in {len(all_think)} (mean kl {statistics.fmean(all_think):.3f}); "
          f"some did not in {len(mixed)} (mean kl {statistics.fmean(mixed):.3f}, max {max(mixed):.3f})")
    assert statistics.fmean(mixed) > statistics.fmean(all_think)
else:
    print("this run has no all-thinking batch after step 10 to compare with")
print("✅ the spikes are k3's variance where π_θ << π_ref: unbiased on average, but one rare token can "
      "outweigh a thousand others. A policy that has abandoned a behaviour the reference liked shows them")

# %% [markdown]
# The *true* KL for this choice is bounded: as π_θ(`</think>`) → 0 it tends to log(1 / 0.5) =
# 0.69 nats per completion. The estimator's variance is not bounded: it grows like π_ref² / π_θ.
# With `beta > 0` the same term is in the *loss*, so those rare tokens also get large gradients.
# Read a logged KL as a running mean over steps, not step by step.
#
# ## On a real GPU (T1)
#
# `TinyRLConfig(device="auto")` puts the model on CUDA when one is visible, and the whole run moves
# with it (`python -m thinklab tinyrl` from a terminal). At this size a GPU mostly buys you the
# freedom to scale the toy: more digits (`k=10`), a larger base, more layers, `num_iterations=2` to
# make the clip bind, `beta=0.04` to watch the KL penalty hold the policy near the SFT model, or
# `sft_mix="uniform"` to start from every scratchpad length and watch the length distribution shift
# as a whole. The step that needs a GPU is notebook 05's: an engine generating rollouts for a real
# 0.5B model.

# %%
print(table([{"knob": "k / base", "try": "10 / 10", "what changes": "a harder task; SFT needs more steps"},
             {"knob": "num_iterations", "try": "2", "what changes": "clip_frac > 0: the ratio moves off 1"},
             {"knob": "beta", "try": "0.04", "what changes": "KL stays small; slower move to always-think"},
             {"knob": "sft_mix", "try": "'uniform'", "what changes": "partial scratchpads; length rises gradually"},
             {"knob": "loss_type", "try": "'grpo'", "what changes": "per-sequence mean: long completions weigh less per token"}],
            title="Experiments for a second run"))

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "RL post-training samples completions, scores them with a verifier and
# reweights the tokens of the better ones. GRPO's baseline is the group: G samples of the same
# prompt, advantage = reward minus the group mean over the group std, so no value model has to
# be trained or served. We verified the mechanism on a toy: a 100K-parameter transformer that
# can only add six digits if it writes running sums first. After a warm-up that showed both
# behaviours, rewarding correct answers alone took accuracy from 0.63 to 0.94 in the recorded run,
# and the full-scratchpad share from 54% to 96%. Completion length rose with it, because thinking
# was what made answers right; the logged KL to the SFT model is spiky, not a sign of drift, because
# k3 is noisy where the policy has abandoned a path the reference liked. The same dynamics, at scale, are why
# RL-trained reasoning models produce long outputs. That is a serving problem as much as a training
# one (notebook 04)."
#
# **Drill 1.** *Why does GRPO not need a critic, and what does that cost?* The group mean is the
# baseline. That saves a policy-sized value model in memory and compute, but needs G > 1 samples
# per prompt: G times the generation per step, which is why rollouts dominate RL step time.
#
# **Drill 2.** *Reward went up and so did response length. Is that reward hacking?* Not by
# itself. Check whether length *causes* correctness (accuracy by length, as in the table above) and
# whether the verifier can be satisfied without doing the work. Also check the loss normalisation.
# With per-sequence averaging (`"grpo"`), long wrong answers are under-penalised, which inflates
# length on its own.
#
# **Drill 3.** *`clip_frac` is always 0. Is the clip broken?* No. With one gradient step per rollout
# batch (`num_iterations=1`, TRL's default) the importance ratio is exactly 1. The clip only acts
# when a batch is reused, or generated by weights that differ from the trainer's.
