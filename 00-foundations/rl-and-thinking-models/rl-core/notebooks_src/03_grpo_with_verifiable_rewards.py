# %% [markdown]
# # 03 · GRPO with verifiable rewards
#
# **Tier:** T0: CPU only, numpy, no network, under a minute. Every formula is the same as in TRL's
# `GRPOTrainer` (v1.14.0, verify), line for line. Thus each setting here maps onto a real `GRPOConfig`. A real
# GRPO step on a small transformer is the `thinking-lab` notebook `01_grpo_on_a_tiny_transformer`. One step in
# which vLLM generates the rollouts is its notebook `05_rl_rollouts_with_an_engine`.
#
# ## The one-minute version
# **GRPO** does not use the value model of PPO. For each prompt, sample a **group** of $G$ completions. Score
# each completion with a **verifier**. Give every token of completion $i$ the advantage
# $A_i = (r_i - \operatorname{mean} r)/(\operatorname{std} r + 10^{-4})$. The group is the baseline.
#
# - A group that is all correct or all incorrect has zero advantage and teaches nothing. As the training
#   succeeds, more groups go silent.
# - The per-token loss is PPO's clipped surrogate
#   $-\min\bigl(\rho A, \operatorname{clip}(\rho, 1 - \varepsilon, 1 + \varepsilon) A\bigr)$ with
#   $\rho = \pi/\pi_{\text{old}}$, plus $\beta\,k_3$, an always-positive KL estimator. With one gradient step per
#   batch (TRL's default $\mu$ = 1), the ratio is 1 and the clip never binds.
# - Then there are the details that decide what the model learns:
#   - An average of the loss **per sequence** under-weights long completions. It bends the update toward longer,
#     truncated answers. This is the **length bias**. DAPO and Dr. GRPO remove it with a token-level or constant
#     normaliser.
#   - Division by the std up-weights nearly-solved prompts.
#   - The clip caps how fast rare tokens can increase.
# - Also, the cost: an inference engine generates $G$ completions per prompt, and these completions dominate the
#   step.
#
# Primer: `../PRIMER.md` §4 (and §1, §8 for the compute).

# %%
import math

import numpy as np

from rlcore import Policy, SeqTask, ThinkTask, grpo, pg, workload

task = SeqTask("brackets", 8)
seqs = task.all_sequences()
ok = np.array([task.verify(s) for s in seqs])
ref = Policy.for_task(task)
for _ in range(2):                                   # the weak SFT reference of notebooks 01–02 (29.7% right)
    pg.sft_step(ref, [task.as_trajectory(s) for s in seqs if task.verify(s)], lr=1.0)


def diversity(pol):
    """P(correct) and the effective number of distinct correct answers, exp(entropy) over them."""
    p = pol.sequence_probs(task, seqs)
    pv = p[ok > 0] / p[ok > 0].sum()
    return float(p @ ok), float(np.exp(-(pv * np.log(pv)).sum()))

# %% [markdown]
# ## Worked example 1 — the group is the baseline
# TRL's code is `advantages = rewards - mean_grouped_rewards`, then `/ (std_rewards + 1e-4)`. The std has
# Bessel's correction.

# %%
for r in ([1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 1, 1], [1] * 7 + [0], [1] * 4 + [0] * 4):
    print(f"rewards {r!s:26} → advantages {grpo.group_advantages([r])[0].round(4).tolist()}")

# %% [markdown]
# Read the last two rows. The single miss on a prompt that the model nearly always solves gets −2.47. A miss on a
# 50/50 prompt gets −0.94.
#
# Division by the std of the group up-weights prompts that are nearly solved or nearly
# hopeless. This is the "difficulty bias" of Dr. GRPO, and `scale_rewards="none"` removes it. The all-correct
# group gets zero: no signal, independent of the number of tokens that it cost to generate.
#
# ## Worked example 2 — the clipped ratio, and when it binds
# $\rho = \pi(\text{token})/\pi_{\text{old}}(\text{token})$. For a positive advantage, the objective does not
# increase more when $\rho > 1 + \varepsilon$. For a negative advantage, the same occurs when
# $\rho < 1 - \varepsilon$. Where the clip binds, the gradient of the token is zero.

# %%
ratio = np.array([0.7, 0.9, 1.0, 1.1, 1.3])
for adv in (1.0, -1.0):
    obj, d = grpo.clipped_surrogate(ratio, np.full(5, adv), 0.2)
    print(f"A = {adv:+.0f}: ratio {ratio.tolist()} → objective {obj.round(2).tolist()}, gradient {d.tolist()}")
for p_old in (0.01, 0.5, 0.9):
    print(f"π_old = {p_old:4}: one update can raise it to at most {min(1, p_old * 1.2):.4f} with ε = 0.2, "
          f"{min(1, p_old * 1.28):.4f} with ε_high = 0.28")

# %% [markdown]
# The cap is multiplicative. Thus a token that the model rarely selects can increase only by a small quantity in
# one update. But a likely token has only 1 as its cap. This asymmetry is the argument of DAPO that a symmetric
# clip drives the entropy down. The solution of DAPO is **clip-higher** ($\varepsilon_{\text{low}}$ 0.2,
# $\varepsilon_{\text{high}}$ 0.28).
#
# Note the TRL default `num_iterations=1`. If you sample and update with the same weights, $\rho \equiv 1$ and the
# clip is inert. The clip is important only with $\mu > 1$ or with stale (off-policy) rollouts.
#
# ## Worked example 3 — the KL estimator k3
# GRPO puts the KL in the loss. It estimates the KL per token from samples of $\pi$:
# $k_3 = \pi_{\text{ref}}/\pi - \log(\pi_{\text{ref}}/\pi) - 1$. Compare it with the simple
# $k_1 = \log(\pi/\pi_{\text{ref}})$ on two small distributions whose true KL we know.

# %%
p_, q_ = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])
x = np.random.default_rng(0).choice(3, 100000, p=p_)
true_kl = float(np.sum(p_ * np.log(p_ / q_)))
for name, fn in (("k1", grpo.k1), ("k3", grpo.k3)):
    est = fn(np.log(p_[x]), np.log(q_[x]))
    print(f"{name}: mean {est.mean():.4f} (true {true_kl:.4f})  std {est.std():.3f}  min {est.min():+.3f}")
print("k3 at log(π_ref/π) = 0.1, −0.1, 0.5:", [round(float(grpo.k3(0.0, d)), 7) for d in (0.1, -0.1, 0.5)])

# %% [markdown]
# Both are unbiased. $k_3$ is never negative and has a fraction of the spread. The low spread is important when
# the loss sums $k_3$ over thousands of tokens. TRL's default is $\beta$ = 0: no reference model in memory at all
# (DAPO also does not use the KL). The DeepSeek-R1 recipe used $\beta$ = 0.001 (per TRL's docs, verify).
#
# ## Worked example 4 — GRPO on the bracket task
# The run uses eight samples per prompt, two groups per step and $\mu$ = 4 gradient steps per batch. Thus the clip
# is live.

# %%
pol = ref.copy()
cfg = grpo.GRPOConfig(num_generations=8, num_iterations=4, lr=20.0)
hist = grpo.train_grpo(pol, task, np.random.default_rng(0), cfg, steps=100, prompts=(0, 0), log_every=20)
for h in hist:
    print(f"step {h['step']:3d}  correct {h['correct']:.2f}  zero-std groups {h['frac_reward_zero_std']:.2f}  "
          f"clipped tokens {h['clipped']:.2f}")
print("reference: P(correct) %.3f, %.1f effective correct answers" % diversity(ref))
print("after GRPO: P(correct) %.3f, %.1f effective correct answers" % diversity(pol))
for ec in (0.0, 0.05):                               # the other lever: pay for entropy in the reward
    p_ec = ref.copy()
    pg.train_reinforce(p_ec, task, np.random.default_rng(0), steps=300, batch=16, lr=0.5, entropy_coef=ec)
    print(f"REINFORCE, entropy bonus {ec}: P(correct) %.3f, %.1f effective correct answers" % diversity(p_ec))

# %% [markdown]
# The success rate goes to ~100% within a few dozen steps. After that, almost every group is all-correct, the
# advantages are zero and the policy learns nothing more. Also, the policy has concentrated on a few of the 14
# correct strings. RL sharpens: pass@1 goes up, and the diversity goes down. An entropy bonus buys the diversity
# back at a small cost in accuracy.
#
# Here, the bonus is a sequence-level $-c \log \pi(y)$ that the code adds to the reward. Thus the baseline also
# applies to the bonus, as to the reward. TRL's `entropy_coef` adds the mean per-token entropy to the loss
# instead. Thus the two coefficients are not interchangeable. In this toy, the collapse occurs with or without
# clip-higher. DAPO reports the entropy effect of clip-higher on real models (primer §4).
#
# ## Worked example 5 — the length bias of averaging per sequence
# A thinking policy answers with probability 0.1 per step. The code truncates the completions that reach 16
# thinking tokens (no answer, reward 0). How do the per-token losses become one number? The answer is TRL's
# `loss_type`:

# %%
lengths = [10, 50]
for lt in ("grpo", "dapo", "dr_grpo"):
    wts = grpo.token_weights(lengths, lt, max_len=100)
    print(f"{lt:8}: weight per token of a 10-token completion {wts[0][0]:.4f}, of a 50-token one {wts[1][0]:.4f}")

# %% [markdown]
# Under `"grpo"` (mean per sequence, then over sequences), each token of the short answer counts five times as
# much. For a correct answer, this favours short answers. For an incorrect answer, the loss punishes a long
# incorrect answer *less per token*. This includes a truncated answer. Average the update over many groups at a
# policy that does not change. Compare its direction with the exact gradient of accuracy:

# %%
think = ThinkTask(e0=0.8, q=0.15, max_think=16)
tp = Policy.for_task(think)
tp.theta[:, 1] = math.log(0.1 / 0.9)
base = think.expected(tp.stop_probs(think))
exact = np.zeros_like(tp.theta)
for i in range(16):
    for j in range(2):
        up, dn = tp.copy(), tp.copy()
        up.theta[i, j] += 1e-5
        dn.theta[i, j] -= 1e-5
        exact[i, j] = (think.expected(up.stop_probs(think))["accuracy"] - think.expected(dn.stop_probs(think))["accuracy"]) / 2e-5
print(f"policy: accuracy {base['accuracy']:.3f}, mean thinking {base['length']:.1f}, truncated {base['truncated']:.3f}")
for lt in ("grpo", "dapo", "dr_grpo"):
    g = grpo.expected_update(tp, think, np.random.default_rng(0), lt, n_groups=600)
    stepped = tp.copy()
    stepped.step(g, 1e-3 / np.linalg.norm(g))
    after = think.expected(stepped.stop_probs(think))
    cos = (g * exact).sum() / np.linalg.norm(g) / np.linalg.norm(exact)
    print(f"{lt:8}: cos(update, true gradient) {cos:.2f}   along it, truncation "
          f"{'rises' if after['truncated'] > base['truncated'] else 'falls'} and mean length +{(after['length'] - base['length']) * 1e3:.2f} per 1e-3 step")

# %% [markdown]
# Token-level (`"dapo"`, TRL's default) and constant (`"dr_grpo"`) normalisers point along the true gradient and
# decrease truncation. The per-sequence mean points in a different direction. It increases the thinking length
# two to four times as fast and pushes *into* truncation. This occurs because the loss spreads the penalty of the
# truncated completions over 16 tokens. That is the length bias. It is also why DAPO masks truncated completions
# (`mask_truncated_completions`) and shapes overlong ones.
#
# ## Worked example 6 — dynamic sampling: paying for groups that carry signal
# A dataset has nine prompts. The model nearly always solves three, sometimes solves three, and rarely solves
# three. Each step trains on four prompts, sampled at random.

# %%
mixed = ThinkTask(prompts=[(0.02, 0.1)] * 3 + [(0.8, 0.15)] * 3 + [(0.995, 0.01)] * 3, max_think=16)
for ds in (False, True):
    h = grpo.train_grpo(Policy.for_task(mixed), mixed, np.random.default_rng(0),
                        grpo.GRPOConfig(num_generations=8, dynamic_sampling=ds, lr=20.0), steps=60,
                        batch_prompts=4, log_every=1)
    silent = np.mean([x["frac_reward_zero_std"] for x in h])
    made = np.mean([x["groups_generated"] for x in h])
    trained_silent = 0.0 if ds else silent
    print(f"dynamic_sampling={ds!s:5}: groups generated per step {made:4.1f}, of which silent {silent:.0%}; "
          f"silent groups in the trained batch {trained_silent:.0%}")

# %% [markdown]
# Without dynamic sampling, more than half of every batch contributes nothing after the engine generates it. Thus
# the effective batch size changes from step to step. DAPO over-samples and keeps only informative groups, so
# every trained batch is full. DAPO pays for this in rollouts (here, about twice as many groups generated per
# step). TRL has no flag for dynamic sampling, and it logs `frac_reward_zero_std`. In DAPO's ablation, dynamic
# sampling was the largest single gain (AIME 2024 avg@32: from 42 to 50, with simple GRPO at 30, primer §4,
# verify).
#
# ## Worked example 7 — where the compute goes
# One synchronous RL step has 512 prompts × 16 samples = 8,192 completions of a 7.6B model on 64 H100s. The
# lengths are lognormal (median 4,000 tokens), with a cap at 20,480. This is DAPO's batch shape. Generation is
# decode-bound and waits for the longest completion. Training is a compute-bound $6 \cdot N$ FLOPs per token,
# plus two passes that calculate scores.

# %%
m7 = workload.Model("7.6B policy", 7.6, 28, 4, 128)
out_len = np.minimum(4000 * np.exp(0.7 * np.random.default_rng(0).standard_normal(8192)), 20480)
r = workload.rl_step_time(m7, workload.GPUS["H100"], 64, 500, out_len)
print(f"mean completion {out_len.mean():,.0f} tokens, longest {out_len.max():,.0f}")
print(f"generate {r['generate_s']:.0f} s, train {r['train_s']:.0f} s → rollouts are {r['rollout_fraction']:.0%} of the step; "
      f"the generation batch is {r['batch_occupancy']:.0%} full on average (the long tail decodes almost alone)")

# %% [markdown]
# The model is optimistic (perfect overlap within each phase, 50% MFU on decode FLOPs, 40% on training). But it
# still puts half of the step into generation. Most of that half is a wait for a few long completions. verl
# reports ~70% for rollouts in DAPO-32B training (primer §8, verify).
#
# Because generation takes this much of the step, RL frameworks embed vLLM or SGLang. For the same reason, they
# move to one-step-off-policy and fully asynchronous rollouts. The price is stale samples that the clipped ratio (and
# importance-sampling corrections) must absorb.
#
# ## Exercise 3.1 — group advantages, as TRL computes them
# `rewards` has shape (n_groups, G). Subtract the mean of each group. Divide by the std of each
# group **with Bessel's correction** plus 1e-4.

# %% exercise
def my_group_advantages(rewards):
    ### BEGIN SOLUTION
    r = np.asarray(rewards, float)
    return (r - r.mean(1, keepdims=True)) / (r.std(1, ddof=1, keepdims=True) + 1e-4)
    ### END SOLUTION

# %% check
batch = [[1, 0, 0, 1], [1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5]]
assert np.allclose(my_group_advantages(batch), grpo.group_advantages(batch))
assert np.allclose(my_group_advantages(batch)[0], [0.865875, -0.865875, -0.865875, 0.865875], atol=1e-6)
print("✅ ±0.866 for a 2-of-4 group; zeros for a group that all scored the same")

# %% [markdown]
# ## Exercise 3.2 — k3
# Write k3 from per-token log-probabilities. Make sure that it is never negative on `d_grid`.

# %% exercise
def my_k3(logp, ref_logp):
    ### BEGIN SOLUTION
    d = np.asarray(ref_logp) - np.asarray(logp)
    return np.exp(d) - d - 1
    ### END SOLUTION

# %% check
d_grid = np.linspace(-5, 5, 101)
assert np.allclose(my_k3(0.0, d_grid), grpo.k3(0.0, d_grid)) and my_k3(0.0, d_grid).min() >= 0
assert round(float(my_k3(0.0, 0.1)), 7) == 0.0051709
print("✅ k3 = e^d − d − 1 ≥ 0 for every d (e^d ≥ 1 + d), zero only when π = π_ref")

# %% [markdown]
# ## Exercise 3.3 — which tokens get a gradient?
# Return a boolean array: True where the gradient of the clipped surrogate with respect to $\rho$ is nonzero.

# %% exercise
def has_gradient(ratio, adv, eps_low=0.2, eps_high=0.2):
    ### BEGIN SOLUTION
    ratio, adv = np.asarray(ratio, float), np.asarray(adv, float)
    return ~(((adv > 0) & (ratio > 1 + eps_high)) | ((adv < 0) & (ratio < 1 - eps_low)) | (adv == 0))
    ### END SOLUTION

# %% check
rr = np.array([0.5, 0.79, 0.81, 1.0, 1.19, 1.21, 1.3, 1.5])
for a in (1.0, -1.0):
    for hi in (0.2, 0.28):
        _, d = grpo.clipped_surrogate(rr, np.full(8, a), 0.2, hi)
        assert np.array_equal(has_gradient(rr, np.full(8, a), 0.2, hi), d != 0)
print("✅ a token stops learning once its ratio has moved past the clip in the direction its advantage wants")

# %% [markdown]
# ## Exercise 3.4 — the three normalisers
# The completion lengths are `lengths` (one group, $G$ of them). Return a list of per-token weight arrays for each
# `loss_type`: "grpo" ($1/(G\,\lvert o_i \rvert)$), "dapo" ($1/\sum \lvert o \rvert$) and "dr_grpo"
# ($1/(G \cdot \mathtt{max\_len})$).

# %% exercise
def my_token_weights(lengths, loss_type, max_len):
    ### BEGIN SOLUTION
    G = len(lengths)
    if loss_type == "grpo":
        return [np.full(n, 1 / (G * n)) for n in lengths]
    if loss_type == "dapo":
        return [np.full(n, 1 / sum(lengths)) for n in lengths]
    return [np.full(n, 1 / (G * max_len)) for n in lengths]
    ### END SOLUTION

# %% check
for lt in ("grpo", "dapo", "dr_grpo"):
    for a, b in zip(my_token_weights([3, 7, 20], lt, 32), grpo.token_weights([3, 7, 20], lt, 32)):
        assert np.allclose(a, b)
print("✅ per-sequence, token-level, constant: the same losses, three different ideas of what one token is worth")

# %% [markdown]
# ## Exercise 3.5 — DAPO's soft overlong punishment
# $L_{\max}$ = 100, cache 20. The punishment is 0 up to 80 tokens. It decreases linearly to −1 at 100. It is −1
# above 100.

# %% exercise
def my_overlong(n, max_len=100, cache=20):
    ### BEGIN SOLUTION
    if n <= max_len - cache:
        return 0.0
    return ((max_len - cache) - n) / cache if n <= max_len else -1.0
    ### END SOLUTION

# %% check
assert [my_overlong(n) for n in (80, 90, 100, 101)] == [0.0, -0.5, -1.0, -1.0]
assert all(my_overlong(n) == grpo.soft_overlong_penalty(n, 100, 20) for n in range(0, 130))
print("✅ a graded penalty inside the last 20 tokens, so 'nearly too long' is a gradient, not a cliff")

# %% [markdown]
# ## Exercise 3.6 — write the configs
# Fill two `GRPOConfig`s:
#
# - `dapo_cfg`: DAPO's recipe. It has clip-higher (0.2/0.28), token-level loss, no KL, overlong filtering, soft
#   overlong punishment with a cache of 4,096, dynamic sampling, and $G$ = 16.
# - `r1_cfg`: GRPO as DeepSeekMath defined it and R1 used it. It has per-sequence mean, division by the group std,
#   $\varepsilon$ = 0.2 on both sides, and $\beta$ = 0.001. (The R1 paper itself writes one ratio per whole
#   completion, primer §4.)
#
# (TRL's name for DAPO's overlong filtering is `mask_truncated_completions`.)

# %% exercise
### BEGIN SOLUTION
dapo_cfg = grpo.GRPOConfig(num_generations=16, beta=0.0, epsilon=0.2, epsilon_high=0.28, loss_type="dapo",
                           mask_truncated_completions=True, overlong_cache=4096, dynamic_sampling=True)
r1_cfg = grpo.GRPOConfig(beta=0.001, epsilon=0.2, epsilon_high=None, loss_type="grpo", scale_rewards="group")
### END SOLUTION

# %% check
assert (dapo_cfg.epsilon, dapo_cfg.epsilon_high, dapo_cfg.loss_type, dapo_cfg.beta) == (0.2, 0.28, "dapo", 0.0)
assert dapo_cfg.mask_truncated_completions and dapo_cfg.dynamic_sampling and dapo_cfg.overlong_cache == 4096
assert dapo_cfg.num_generations == 16
assert (r1_cfg.beta, r1_cfg.loss_type, r1_cfg.scale_rewards, r1_cfg.epsilon_high) == (0.001, "grpo", "group", None)
print("✅ TRL's defaults are neither: beta=0.0, loss_type='dapo', no clip-higher, no masking — set them explicitly "
      "when you mean to reproduce a paper")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "For tasks where we can do a check of the result (math answers, unit tests, output
# formats), we use RL with verifiable rewards and GRPO. Each prompt gets a group of $G$ samples. The group mean is the
# baseline. Thus there is no value model. The memory holds the policy, a reference only if $\beta > 0$, and the
# rollout engine.
#
# "Groups that are all correct or all incorrect carry no gradient. Thus we monitor frac_reward_zero_std. We
# resample or curate prompts to stay near 50% solvable.
#
# "We use the token-level loss because a per-sequence average under-weights long completions and moves the model
# into long, truncated answers. We mask truncated completions and add a soft overlong penalty near the cap. The clip is
# important only with several updates per batch or off-policy rollouts.
#
# "Most of each step is generation. Generation is decode-bound and waits for the longest completion. Thus the
# rollout side is an inference-serving problem: KV capacity, batching, and async rollouts with bounded
# staleness."
#
# **Drill questions**
# 1. *Why does GRPO not need a value model?* Answer: the baseline is the mean reward of $G$ samples of the same
#    prompt. PPO learns ${V(s)}$ to get per-token baselines. ${V(s)}$ is a second network, as large as the policy.
# 2. *The loss decreases to zero, and the accuracy stays at a plateau of 95% on the training set. What occurs?*
#    Answer: most groups are all-correct, thus the advantages are zero: no signal. Use harder prompts, dynamic
#    sampling, or a curriculum.
# 3. *The responses continue to become longer, and more of them hit max_completion_length. Which setting do you
#    examine first?* Answer: `loss_type`. A per-sequence ("grpo") average penalises long incorrect answers less
#    per token. Use "dapo"/"dr_grpo", mask truncated completions, and add the soft overlong penalty.
