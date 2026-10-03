# %% [markdown]
# # 01 · GRPO on a tiny transformer: a model discovers that thinking pays
#
# **Tier:** T0 with torch on a laptop CPU. The training run takes about a minute. We examined this notebook
# on a shared 4-core container: ~40 s for the run, under a minute in all. T1 (any GPU) runs the same code
# faster, but this model does not need that speed. Without torch, the notebook still runs. In that
# case, the training cells show a recorded run, labelled illustrative, and the torch checks tell you
# that the notebook skipped them.
#
# ## The one-minute version
#
# * **RL for a language model is "sample, score, reweight".** Sample completions from the model, and
#   give each completion a score with a reward. Then increase the probability of the tokens in the
#   completions that scored above the baseline. This is the policy gradient over token sequences
#   (PRIMER §2 "Policy gradients over token sequences").
# * **GRPO** makes the *group* the baseline. Sample $G$ completions per prompt. The advantage of a
#   completion is its reward minus the group mean, divided by the standard deviation of the group.
#   GRPO trains no value model (PRIMER §4 "RL with verifiable rewards and GRPO").
# * **Verifiable rewards** need no reward model: a program examines the final answer. Here the task
#   is the last digit of a sum of six base-5 digits. A 2-layer transformer cannot add six numbers in
#   the one token that it emits for the answer. But it can add them if it first writes the partial
#   sums in a `<think>` scratchpad. Each thinking token adds one more step of serial computation.
# * **What you will watch:** a short supervised warm-up shows the model both behaviours, half the
#   time each: it answers straight away, or it thinks first. Then GRPO, with a reward only for correct
#   answers, drives the thinking rate toward 100%, and the reward and the completion length rise
#   *together*. This is R1-style, not R1-Zero: a cold-start SFT shows the behaviour, and RL makes the
#   model *select* it (PRIMER §5 "Thinking models"). R1-Zero had no SFT, and its traces grew
#   gradually. Here, `sft_mix="uniform"` (end of the notebook) is the nearer analogue.
# * **What you will build:** the GRPO loss in torch, and an analysis of the KL curve that explains
#   its spikes. You then use your loss for the training.

# %%
import math, random, statistics
from dataclasses import replace
from thinklab import env
from thinklab.report import plot, table
from thinklab.rollout import k3
from thinklab.tinyrl.task import EOS, END_THINK, THINK, DigitSum, render, reward
from thinklab.tinyrl.curves import load_recorded, show

print(env.describe())
HAVE_TORCH = env.has_torch()
print("torch available:", HAVE_TORCH, "-> training cells run for real" if HAVE_TORCH else "-> recorded run (illustrative)")

# %% [markdown]
# ## Worked example: the task, three completions and the verifier
#
# The prompt is six digits in base 5, then `=`. A completion opens `<think>`. It can write partial
# sums (mod 5). Then it closes `</think>` and gives the answer and `<eos>`. The verifier examines only
# the format and the final answer, never the scratchpad. Thus it is an *outcome* reward, like the
# rule-based accuracy reward of R1.

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
# Return 1.0 when `completion` has these tokens in this order, and nothing else before `<eos>`:
#
# 1. `<think>`
# 2. zero or more digit tokens (0–9)
# 3. `</think>`
# 4. the answer digit
# 5. `<eos>`
#
# Ignore the tokens after `<eos>`: they are padding. The answer must be equal to `problem.answer`.
# In all other cases, return 0.0. The token ids: the digits are 0–9. The import cell near the top of
# the notebook imports `THINK`, `END_THINK` and `EOS`.

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
# First comes supervised fine-tuning on demonstrations: half of them have no scratchpad, and half
# have the full one. This is the "cold start" of the R1 recipe. Then we sample one completion per
# problem at temperature 1. We show the accuracy for each scratchpad length that the model
# *selected*. With torch, the notebook measures this now, on this machine. Without torch, the table
# shows the recorded run.

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
# Without the scratchpad, the model is at chance: it cannot calculate a six-term sum in one step.
# With the scratchpad, it is almost always correct. The model already *can* think. But it does not
# yet *select* that behaviour. Nothing in SFT told it which behaviour is better. The verifier will
# tell it.
#
# The rl-core notebook 03 (exercise 3.1) derives and implements the group-relative advantage,
# $(r - \operatorname{mean})/(\operatorname{std} + 10^{-4})$ with Bessel's std. Here,
# `train.group_advantages` calculates it in torch. When one completion in four is correct, that
# completion gets +1.4997 and each incorrect one gets −0.4999. Thus the update is about the rare
# success. A group with rewards that are all equal gets zero advantages.
#
# ## Exercise 1.2 — predict the reward before RL starts, and where it can end
#
# After SFT, the model thinks with the probability `share`. When it thinks, it is correct with the
# probability `acc_think`. When it does not think, it is correct with the probability `acc_direct`.
# Write the expected reward. Then compare it with the run.
#
# The accuracies come from the table of the warm-up example. The share comes from the step-0 batch
# of GRPO. Each completion is 4 tokens without a scratchpad or 10 tokens with one. Thus the mean
# length of the batch gives its thinking share, `(length − 4) / 6`. With `share = 1`, you get the
# ceiling that RL can reach when it changes only the *choice*.

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
# `RUN` already holds the full training run, with these settings:
#
# * 40 steps of 16 prompts × $G$ = 8 samples.
# * `beta = 0`. This is the default of TRL: the loss has no reference-model penalty. The trainer
#   still *logs* the KL to the SFT model.
# * `loss_type = "dapo"`.
# * `num_iterations = 1`.

# %%
print(show(RUN))
if plot(RUN["rl"], "step", ["reward", "length", "frac_zero_std", "kl"], title=LABEL):
    pass

# %% [markdown]
# Did *this* run learn? The next cell compares the evaluation before RL with the evaluation after
# RL. It also compares the first RL step with the last RL steps. Then it tells you the result, yes
# or no. No cell after this one assumes the answer.

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
# Read the curves as follows. The numbers are the ones that the two previous code cells printed, for
# the run that you have:
#
# * **reward**: expect it to increase from the SFT mix toward the ceiling of Exercise 1.2. Expect
#   **length**, the mean completion in tokens, to increase from about 7 (half 4, half 10) toward 10.
#   The training never gave a reward to longer completions directly. Length grows because thinking
#   is *instrumentally* useful. This is the toy version of the response-length growth of R1.
# * **frac_zero_std** is the share of groups whose 8 rewards are all equal. It *increases* as the
#   policy becomes better. Those groups have zero advantage and teach nothing. That is why DAPO
#   resamples them ("dynamic sampling"), and that is why TRL logs this number.
# * **kl** is the $k_3$ estimate that TRL logs over one batch. Thus it is noisy. In the recorded run,
#   it increases over the first ten steps and never becomes stable. Some batches show a spike to
#   0.1–0.4, and the largest value of kl is at the last step. Nothing pulls the policy back (`beta = 0`).
#   Exercise 1.5 finds the cause of the spikes.
#
# `clip_frac` is 0 at every step. With `num_iterations = 1`, the trainer uses the rollouts for
# exactly one gradient step. Thus $\pi_\theta = \pi_{\text{old}}$ when the trainer calculates the
# loss. The ratio is exactly 1, and the clip never binds. Exercise 1.4 changes that.
#
# ## Exercise 1.3 — the GRPO loss, in torch
#
# Write the loss that `train.py` minimises. The inputs are:
#
# * `(B, T)` tensors of per-token log-probabilities under three policies: the current policy
#   (`logp`), the policy that sampled the batch (`old_logp`) and the reference (`ref_logp`).
# * a `(B,)` tensor of advantages, `adv`.
# * a `(B, T)` float `mask` of live tokens.
#
# For each token, `ρ = exp(logp − old_logp)`. The loss of the token is
# `−min(ρ·A, clip(ρ, 1 − cfg.epsilon, 1 + cfg.epsilon_high)·A)`. When `cfg.beta > 0`, add
# `cfg.beta · k3`, with `k3 = exp(d) − d − 1` and `d = ref_logp − logp`. Then aggregate the live
# tokens as `cfg.loss_type` specifies:
#
# * `"grpo"` calculates the mean of each completion, then the mean of the completions.
# * `"dr_grpo"` divides the sum by `B · max_len`.
# * `"dapo"` divides the sum by the number of live tokens.
#
# The rl-core notebook 03 (exercises 3.2–3.4) derives the pieces and the length bias of each
# aggregation. Here, the pieces must work together, on tensors, with autograd.

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
# The check cell after the exercise starts GRPO again two times from the same SFT model, for 10
# steps each time. Both runs minimise *your* loss. The first run uses `num_iterations = 1`. The
# second run uses `num_iterations = 2`: it uses each rollout batch for two optimizer steps, as TRL
# does with $\mu > 1$. Before you run the check, predict these values:
#
# * `clip_mu1`: the `clip_frac` that the $\mu$ = 1 run logs at every step (a number).
# * `clip_binds_mu2`: will the $\mu$ = 2 run log `clip_frac > 0` at some step (`True` or `False`)?
#
# The trainer measures `clip_frac` on the last pass over the batch. The check does the two runs and
# prints them side by side. It also makes sure that your loss actually trains the model.

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
# After RL, the policy almost never takes the no-scratchpad path. The probability that it gives
# `</think>` directly after `<think>` has decreased to a fraction of a percent. But the frozen SFT
# reference still gives that token about 0.5. When a batch samples that token by chance, its $k_3$
# term is large.
#
# Use `k3` from `thinklab.rollout` (defined in rl-core notebook 03, exercise 3.2) to calculate two
# values:
#
# * `one_token`: the $k_3$ of a token with $\pi_\theta$ = 0.005 and $\pi_{\text{ref}}$ = 0.5.
# * `batch_kl`: what that one token adds to the batch-mean kl that the trainer logs. The batch has
#   128 completions of about 10 live tokens each.
#
# Then examine the claim on the run. In the steps where *every* completion wrote the full
# scratchpad (mean length exactly 10), expect only the small drift of the scratchpad.

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
# The *true* KL for this choice has a bound. As $\pi_\theta$(`</think>`) $\to 0$, it goes to
# $\log(1/0.5)$ = 0.69 nats per completion. The variance of the estimator has no bound: it grows
# like $\pi_{\text{ref}}^2/\pi_\theta$. With `beta > 0`, the same term is in the *loss*, thus those
# rare tokens also get large gradients. Read a logged KL as a moving mean over steps, not step by
# step.
#
# ## On a real GPU (T1)
#
# `TinyRLConfig(device="auto")` puts the model on CUDA when a GPU is visible, and the full run moves
# with it. From a terminal, the command is `python -m thinklab tinyrl`. At this size, a GPU mostly
# gives you the freedom to scale the toy:
#
# * more digits (`k=10`)
# * a larger base
# * more layers
# * `num_iterations=2`, to make the clip bind
# * `beta=0.04`, to watch the KL penalty hold the policy near the SFT model
# * `sft_mix="uniform"`, to start from every scratchpad length and watch the length distribution
#   shift as a whole
#
# The step that needs a GPU is in notebook 05: an engine that generates rollouts for a real 0.5B
# model.

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
# **Two minutes:** "RL post-training samples completions, scores them with a verifier and reweights
# the tokens of the better ones. The baseline of GRPO is the group: $G$ samples of the same prompt.
# The advantage is the reward minus the group mean, divided by the group std. Thus we need no value
# model, for the training or for the serving.
#
# "We made sure of the mechanism on a toy: a 100K-parameter transformer that can add six digits only if it
# writes partial sums first. A warm-up showed both behaviours. After it, a reward for correct answers
# alone took accuracy from 0.63 to 0.94 in the recorded run. It took the full-scratchpad share from
# 54% to 96%. Completion length rose with it, because thinking was what made the answers correct.
#
# "The logged KL to the SFT model has spikes, but they are not a sign of drift. The cause is that
# $k_3$ is noisy where the policy no longer takes a path that the reference liked. The same
# dynamics, at scale, are why RL-trained reasoning models produce long outputs. That is a serving
# problem as much as a training one (notebook 04)."
#
# **Drill 1.** *Why does GRPO not need a critic, and what does that cost?* The group mean is the
# baseline. That saves a policy-sized value model in memory and compute. But GRPO needs ${G > 1}$
# samples per prompt, that is, $G$ times the generation per step. That is why the rollouts use most
# of the time of an RL step.
#
# **Drill 2.** *Reward went up and so did response length. Is that reward hacking?* Not by itself.
# Find out if length *causes* correctness: look at the accuracy for each length, as in the table of
# the warm-up example. Also find out if the model can satisfy the verifier and not do the work.
#
# Also examine the loss normalisation. With the per-sequence mean (`"grpo"`), long incorrect answers get
# too small a penalty. That alone increases the length.
#
# **Drill 3.** *`clip_frac` is always 0. Is the clip broken?* No. With one gradient step per rollout
# batch (`num_iterations=1`, the default of TRL), the importance ratio is exactly 1. The clip acts
# only in two cases. The trainer uses a batch again, or the batch came from weights that differ from
# the weights of the trainer.
