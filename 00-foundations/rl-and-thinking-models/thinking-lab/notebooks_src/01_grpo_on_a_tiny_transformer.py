# %% [markdown]
# # 01 · GRPO on a tiny transformer: a model discovers that thinking pays
#
# **Tier:** T0 with torch on a laptop CPU (the whole training run takes about a minute; this notebook
# was checked on a shared 4-core container at ~40 s). T1 (any GPU) runs the same code faster, which
# this model does not need. Without torch every exercise still runs and the training cells show a
# recorded run, labelled illustrative.
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
#   answers, drives the thinking rate to ~100%. The reward and the completion length rise
#   *together*. This is the smallest honest version of what DeepSeek-R1-Zero saw at scale (PRIMER §5 "Thinking models").

# %%
import math, random
from thinklab import env
from thinklab.report import plot, table
from thinklab.rollout import aggregate as ref_aggregate, group_advantages as ref_advantages, k3 as ref_k3
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
    from thinklab.tinyrl import train as T             # imports torch
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

# %% [markdown]
# ## Exercise 1.2 — group-relative advantages, exactly as TRL computes them
#
# For one group of rewards return `(r - mean) / (std + 1e-4)`, where `std` uses Bessel's correction
# (divide by n − 1), as `torch.std` and TRL's `nanstd` do. A group whose rewards are all equal gets
# all-zero advantages.

# %% exercise
def my_advantages(rewards: list) -> list:
    ### BEGIN SOLUTION
    m = sum(rewards) / len(rewards)
    s = math.sqrt(sum((r - m) ** 2 for r in rewards) / (len(rewards) - 1)) if len(rewards) > 1 else 0.0
    return [(r - m) / (s + 1e-4) for r in rewards]
    ### END SOLUTION

# %% check
assert all(abs(a - e) < 1e-5 for a, e in zip(my_advantages([1, 0, 0, 1]), [0.865875, -0.865875, -0.865875, 0.865875]))
assert all(abs(a - e) < 1e-4 for a, e in zip(my_advantages([1, 0, 0, 0]), [1.4997, -0.4999, -0.4999, -0.4999]))
assert my_advantages([1, 1, 1, 1]) == [0.0] * 4
assert all(abs(a - b) < 1e-12 for a, b in zip(my_advantages([0.3, 0.9, 0.1, 0.5]), ref_advantages([0.3, 0.9, 0.1, 0.5])))
print("✅ [1,0,0,1] -> ±0.8659; one right out of four -> +1.4997 for it, -0.4999 for the others; all-equal -> 0")

# %% [markdown]
# Note the asymmetry in the second case. When one completion in four is right, it gets a large
# positive advantage and each wrong one a small negative one. The rare success is what the update
# is about.
#
# ## Exercise 1.3 — predict the reward before RL starts, and where it can end
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
if plot(RUN["rl"], "step", ["reward", "length", "frac_zero_std"], title=LABEL):
    pass

# %% [markdown]
# Read the curves together:
#
# * **reward** climbs from the SFT mix toward the ceiling of Exercise 1.3;
# * **length**, the mean completion in tokens, climbs from about 7 (half 4, half 10) to 10. The
#   model now always writes the scratchpad. Longer completions were never rewarded directly. Length
#   grew because it is *instrumentally* useful, the toy version of R1's response-length growth;
# * **frac_zero_std**, the share of groups whose 8 rewards are all equal, *rises* as the policy
#   gets good. Those groups have zero advantage and teach nothing. That is why DAPO resamples them
#   ("dynamic sampling") and why TRL logs this number;
# * **kl** to the SFT model grows then settles. With `beta = 0` nothing pulls the policy back. On a
#   real model that is where reward hacking and drift would show up.
#
# `clip_frac` is 0 at every step. With `num_iterations = 1` the rollouts are used for exactly one
# gradient step, so π_θ = π_old when the loss is computed, the ratio is exactly 1 and the clip never
# binds. Clipping matters when a batch is reused (μ > 1) or generated by a stale or different copy
# of the weights (notebook 05).
#
# ## Exercise 1.4 — three ways to average a token loss, and who they favour
#
# `per_token` holds one list of per-token losses per completion. Implement TRL's three
# normalisations:
#
# * `"grpo"`: mean over each completion's tokens, then mean over completions;
# * `"dr_grpo"`: sum of all tokens ÷ (number of completions × `max_len`);
# * `"dapo"`: sum of all tokens ÷ total number of tokens.

# %% exercise
def my_aggregate(per_token: list, kind: str, max_len: int = 10) -> float:
    ### BEGIN SOLUTION
    if kind == "grpo":
        return sum(sum(s) / len(s) for s in per_token) / len(per_token)
    if kind == "dr_grpo":
        return sum(map(sum, per_token)) / (len(per_token) * max_len)
    if kind == "dapo":
        return sum(map(sum, per_token)) / sum(map(len, per_token))
    raise ValueError(kind)
    ### END SOLUTION

# %% check
short, long_ = [-1.0] * 2, [-1.0] * 8                      # same advantage, lengths 2 and 8
batch = [short, long_]
assert my_aggregate(batch, "grpo") == -1.0 and my_aggregate(batch, "dapo") == -1.0
assert my_aggregate(batch, "dr_grpo", 10) == -0.5
for kind in ("grpo", "dr_grpo", "dapo"):
    assert abs(my_aggregate([[0.1, -0.3], [0.2, 0.2, 0.5]], kind, 10) - ref_aggregate([[0.1, -0.3], [0.2, 0.2, 0.5]], kind, 10)) < 1e-12
# the weight each token of each completion gets in the gradient (d loss / d token):
w = {k: (1 / 2 / 2, 1 / 8 / 2) if k == "grpo" else (1 / 20, 1 / 20) if k == "dr_grpo" else (1 / 10, 1 / 10)
     for k in ("grpo", "dr_grpo", "dapo")}
print("✅ per-token weight (short, long):", {k: tuple(round(x, 4) for x in v) for k, v in w.items()})
print("   'grpo' gives a token of the 8-token completion 1/4 the weight of one in the 2-token completion:")
print("   long wrong answers are punished less per token, the length bias Dr. GRPO and DAPO remove")

# %% [markdown]
# ## Exercise 1.5 — the k3 estimator of the KL to the reference model
#
# GRPO penalises drift from the reference model per token with Schulman's k3 estimator:
# `k3 = exp(d) - d - 1` with `d = log π_ref(token) - log π_θ(token)`. It is never negative, is
# zero when the two agree, and is unbiased for KL(π_θ ‖ π_ref) when tokens are sampled from π_θ.

# %% exercise
def my_k3(logp: float, ref_logp: float) -> float:
    ### BEGIN SOLUTION
    d = ref_logp - logp
    return math.exp(d) - d - 1
    ### END SOLUTION

# %% check
assert abs(my_k3(0.0, 0.1) - 0.0051709) < 1e-7 and abs(my_k3(0.0, -0.1) - 0.0048374) < 1e-7
assert abs(my_k3(0.0, 0.5) - 0.1487213) < 1e-7 and my_k3(-1.3, -1.3) == 0.0
rng = random.Random(1)
assert all(my_k3(a, b) >= 0 and abs(my_k3(a, b) - ref_k3(a, b)) < 1e-12
           for a, b in ((rng.uniform(-5, 0), rng.uniform(-5, 0)) for _ in range(200)))
print("✅ k3 at d = 0.1 / -0.1 / 0.5: 0.0051709 / 0.0048374 / 0.1487213; never negative")

# %% [markdown]
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
# behaviours, rewarding correct answers alone took accuracy from ~0.6 to ~0.95. Completion length
# rose with it, because thinking was what made answers right. The same dynamics, at scale, are why
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
