# %% [markdown]
# # 01 · Policy gradients on a toy task
#
# **Tier:** T0: CPU only, numpy, no network, well under a minute. A table of softmaxes replaces the network.
# Thus you can also calculate each expectation *exactly*: enumerate all 256 possible outputs. The same loop on a
# real (small) transformer is the `thinking-lab` notebook `01_grpo_on_a_tiny_transformer`.
#
# ## The one-minute version
# A language model is a **policy**. At each position, it selects a token from $\pi(a \mid s)$. A completion is a
# trajectory. Its log-probability is the sum of the log-probabilities of its tokens.
#
# - RL for LLMs is **sample, score, reweight**. It samples completions and scores each one (here with a
#   verifier). Then it increases the log-probability of the completions that scored above a **baseline**, and
#   decreases the log-probability of the other completions. The formula is
#   $\nabla \mathbb{E}[R] = \mathbb{E}[(R - b)\,\nabla \log \pi(y)]$ (REINFORCE). The baseline changes the noise,
#   but never the direction.
# - A **KL penalty** to the frozen reference, $R - \beta \log(\pi/\pi_{\text{ref}})$, gives the optimum a closed
#   form: $\pi^* \propto \pi_{\text{ref}} \exp(R/\beta)$. In other words, RL reweights what the model already does.
# - Also, RL optimises **exactly the reward you wrote**. If the verifier has a loophole, RL uses it. Anything
#   that the reward does not charge for (thinking length) grows.
#
# After this notebook, you can do these tasks:
#
# - Derive the gradient.
# - Tell why the baseline is important.
# - Calculate the KL-regularised optimum.
# - Predict what RL does to a buggy verifier and to a length that the reward does not charge for.
#
# Primer: `../PRIMER.md` §1–§2.

# %%
import math

import numpy as np

from rlcore import ANSWER, Policy, SeqTask, ThinkTask, pg

task = SeqTask("brackets", length=8)
seqs = task.all_sequences()


def bar(x, width=40):
    return "#" * int(round(x * width))

# %% [markdown]
# ## Worked example 1 — a task with a verifier
# The policy writes 8 tokens. Each token is `(` or `)`. A deterministic check tells if the string is balanced.
# There is no reward model, and there is nothing to learn about the reward. The verifier *is* the specification.

# %%
for s in ([0, 1, 0, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]):
    print(task.render(s), "balanced" if task.verify(s) else "not balanced")
print(f"a uniformly random policy is right {task.random_success_rate():.1%} of the time "
      f"({int(task.random_success_rate() * 256)} of 256 strings)")

# %% [markdown]
# ## Worked example 2 — the policy, a sample and its log-probability
# The state of the policy is (position, depth). There is also one "broken" state for prefixes that already went
# below zero. The policy needs only what the verifier will look at. `sample` draws tokens one at a time, as an
# engine does. `token_logprobs` is what an engine returns as logprobs. The log-probability of the completion is
# their sum.

# %%
rng = np.random.default_rng(0)
base = Policy.for_task(task)                              # all zeros: uniform, a "pretrained" model that knows nothing
traj = base.sample(task, rng, 1)[0]
print(task.render(traj.actions), "reward", traj.reward)
print("per-token logprobs:", base.token_logprobs(traj).round(3).tolist())
print(f"log π(y) = {base.token_logprobs(traj).sum():.3f} = 8·log(1/2) = {8 * math.log(0.5):.3f}")

# %% [markdown]
# ## Worked example 3 — pretrain → SFT → RL
# This is post-training on a small scale. **SFT** is maximum likelihood on demonstrations. Two gradient steps on
# the 14 balanced strings give a weak reference model. This model is correct 29.7% of the time. Then do **RL**:
# sample 16 completions, score them, reweight, and repeat.

# %%
demos = [task.as_trajectory(s) for s in seqs if task.verify(s)]
ref = Policy.for_task(task)
for _ in range(2):
    pg.sft_step(ref, demos, lr=1.0)
print(f"SFT reference: P(balanced) = {pg.expected(ref, task, task.verify):.3f}")

pol = ref.copy()
hist = pg.train_reinforce(pol, task, np.random.default_rng(0), steps=150, batch=16, lr=0.5, log_every=25)
for step, reward, correct, _ in hist:
    print(f"step {step:3d}  batch reward {reward:.2f}  {bar(reward)}")
print(f"after RL: P(balanced) = {pg.expected(pol, task, task.verify):.3f} exactly; "
      f"KL(π‖π_ref) = {pg.kl_seq(pol, ref, task):.2f} nats")

# %% [markdown]
# Each step has the same three moves: sample (inference), score (the verifier), reweight (one gradient step). In
# a production system, the first move is an inference engine that samples thousands of long completions. Most of
# the compute goes to this move (primer §1, notebook 03).
#
# ## Worked example 4 — why the baseline
# The REINFORCE estimate is unbiased with any baseline $b$ that does not depend on the sample. The reason is
# $\mathbb{E}[\nabla \log \pi(y)] = 0$. What changes is the variance.
#
# Measure the variance at the start of a ThinkTask run. Then add a constant +5 to each reward. This constant
# gives no information at all. But without a baseline, it multiplies the noise.

# %%
think = ThinkTask(e0=0.8, q=0.15, max_think=16)
p0 = Policy.for_task(think)
for off in (0.0, 5.0):
    v = {b: pg.grad_variance(p0, think, np.random.default_rng(1), b, batch=8, trials=150, offset=off)
         for b in ("none", "mean")}
    print(f"reward offset {off:+.0f}:  variance without baseline {v['none']:.3f}   with the batch mean {v['mean']:.3f}")

# %% [markdown]
# GRPO uses the batch-mean baseline (its group *is* the batch for one prompt, notebook 03). PPO learns a baseline
# instead. That baseline is a value model as large as the policy (notebook 02). Leave-one-out (RLOO) uses the
# mean of the *other* samples. Its advantage is exactly $\frac{n}{n-1}$ times the mean-baseline advantage, in the
# same direction.
#
# ## Worked example 5 — the KL penalty and its closed form
# Maximise $\mathbb{E}[R] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$. Over all distributions, the answer is
# $\pi^* = \pi_{\text{ref}} \exp(R/\beta)/Z$. Multiply the reference probability of each completion by
# $\exp(R/\beta)$, and then renormalise. With a small $\beta$, the policy goes after the reward. With a large
# $\beta$, the policy stays home.

# %%
ref_p = ref.sequence_probs(task, seqs)
R = np.array([task.verify(s) for s in seqs])
print(f"{'beta':>6} {'E[R] = P(balanced)':>19} {'KL(π*‖π_ref)':>13}")
for beta in (10.0, 1.0, 0.3, 0.1):
    pi, er, kl = pg.kl_optimal(ref_p, R, beta)
    print(f"{beta:6.1f} {er:19.3f} {kl:13.2f}")
leashed = ref.copy()
pg.train_reinforce(leashed, task, np.random.default_rng(0), steps=200, batch=16, lr=0.5, ref=ref, beta=0.3)
_, er, kl = pg.kl_optimal(ref_p, R, 0.3)
print(f"REINFORCE with β = 0.3 reaches P = {pg.expected(leashed, task, task.verify):.3f}, KL {pg.kl_seq(leashed, ref, task):.2f}"
      f"  (closed form: {er:.3f}, {kl:.2f})")

# %% [markdown]
# $\pi^*$ can move mass only among the completions that $\pi_{\text{ref}}$ already produces. A completion with
# $\pi_{\text{ref}}(y) = 0$ stays at 0 for every $\beta$. This is the precise sense in which RL "sharpens" and
# does not invent. It is also the formula that DPO inverts in notebook 02.
#
# ## Worked example 6 — reward hacking: RL finds the verifier's bug
# `buggy_verify` returns *pass* as soon as the depth goes negative (think of a test harness that counts an early
# `sys.exit(0)` as a success). The SFT reference passes this check 72.7% of the time. But it is correct only
# 29.7% of the time.

# %%
buggy = SeqTask("brackets", 8, verifier="buggy")
print(f"reference: passes the buggy check {pg.expected(ref, task, task.buggy_verify):.3f}, "
      f"truly balanced {pg.expected(ref, task, task.verify):.3f}")
for beta in (0.0, 0.3):
    hacked = ref.copy()
    pg.train_reinforce(hacked, buggy, np.random.default_rng(0), steps=200, batch=16, lr=0.5, ref=ref, beta=beta)
    print(f"RL on the buggy check, β = {beta}: passes {pg.expected(hacked, task, task.buggy_verify):.3f}, "
          f"truly balanced {pg.expected(hacked, task, task.verify):.3f}, KL {pg.kl_seq(hacked, ref, task):.2f}")
    first_close = sum(p for p, s in zip(hacked.sequence_probs(task, seqs), seqs) if s[0] == 1)
    print(f"   P(the output starts with ')') = {first_close:.2f} (reference: "
          f"{sum(p for p, s in zip(ref.sequence_probs(task, seqs), seqs) if s[0] == 1):.2f})")
pi_b, _, _ = pg.kl_optimal(ref_p, np.array([task.buggy_verify(s) for s in seqs]), 0.3)
print(f"closed-form π* for the buggy reward at β = 0.3: truly balanced {pi_b @ R:.3f}")

# %% [markdown]
# The reward went *up* while the correctness went *down*. At depth 0, `)` is an instant pass, thus every state
# learns it. With $\beta$ = 0, every policy that always passes is optimal. Gradient ascent reaches the policy
# that is easiest to reach: the loophole.
#
# The KL-regularised optimum multiplies *every* string that passes by the same $\exp(1/\beta)$. Thus it keeps the
# share of honest strings that the reference has among the strings that pass. RL with $\beta$ = 0.3 goes toward
# that optimum. The KL penalty is a leash, not a solution. The solution is the verifier (and evals that measure
# the true objective, not the training reward).
#
# The authors of R1 kept the rewards rule-based for exactly this reason. A learned reward model is a larger
# loophole (notebook 02).
#
# ## Worked example 7 — RL lengthens whatever it does not pay for
# In `ThinkTask`, the policy writes "think" tokens until it answers.
# $P(\text{correct} \mid L) = 1 - 0.8 \cdot 0.85^L$. Start from a policy that answers at once half of the time.
# First, reward only the correctness. Then charge 0.02 per thinking token.

# %%
for cost in (0.0, 0.02):
    t = ThinkTask(e0=0.8, q=0.15, max_think=16, cost=cost)
    p = Policy.for_task(t)
    h = pg.train_reinforce(p, t, np.random.default_rng(0), steps=300, batch=16, lr=2.0, log_every=100)
    ex = t.expected(p.stop_probs(t))
    print(f"cost {cost}: mean thinking length by step {[round(float(x[3]), 1) for x in h]} -> {ex['length']:.1f} tokens, "
          f"accuracy {ex['accuracy']:.3f} (optimum L* = {t.optimal_length():.1f})")

# %% [markdown]
# With no cost, the thinking length grows toward the cap. This is the toy version of the growth in response
# length that DeepSeek-R1-Zero reported (primer §5). With a per-token cost, the length grows only while a token of
# thought buys more accuracy than it costs. Lengths move slowly in this table-of-softmaxes policy (each position
# learns only after the policy reaches it). The direction is the lesson.
#
# ## Exercise 1.1 — the softmax gradient
# For one state with logits `theta_row`, return $\partial \log \pi(a)/\partial \theta$ as a vector:
# `onehot(a) − softmax(theta_row)`.

# %% exercise
def grad_logprob_row(theta_row, a):
    ### BEGIN SOLUTION
    p = np.exp(theta_row - theta_row.max())
    p /= p.sum()
    g = -p
    g[a] += 1.0
    return g
    ### END SOLUTION

# %% check
row = np.array([0.5, -1.0, 2.0])
eps = 1e-6
num = np.array([(np.log(np.exp(row + eps * np.eye(3)[j])[1] / np.exp(row + eps * np.eye(3)[j]).sum())
                 - np.log(np.exp(row - eps * np.eye(3)[j])[1] / np.exp(row - eps * np.eye(3)[j]).sum())) / (2 * eps)
                for j in range(3)])
assert np.allclose(grad_logprob_row(row, 1), num, atol=1e-8)
assert abs(grad_logprob_row(row, 1).sum()) < 1e-12
print("✅ ∂log π(a)/∂θ = onehot(a) − π: it sums to zero, so raising one token lowers the others")

# %% [markdown]
# ## Exercise 1.2 — the REINFORCE estimator
# For a batch of trajectories, return the batch-mean-baseline estimate
#
# $$
# \frac{1}{N} \sum_i (R_i - \operatorname{mean} R)\, \nabla \log \pi(y_i).
# $$
#
# Use `policy.grad_logprob(traj)` for $\nabla \log \pi(y_i)$.

# %% exercise
def my_reinforce(policy, trajs):
    ### BEGIN SOLUTION
    r = np.array([t.reward for t in trajs])
    a = r - r.mean()
    return sum(ai * policy.grad_logprob(t) for ai, t in zip(a, trajs)) / len(trajs)
    ### END SOLUTION

# %% check
batch = ref.sample(task, np.random.default_rng(5), 16)
assert np.allclose(my_reinforce(ref, batch), pg.reinforce_grad(ref, batch, "mean"))
flat = [t for t in batch]
for t in flat:
    t.reward = 1.0
assert np.allclose(my_reinforce(ref, flat), 0)
print("✅ REINFORCE with a baseline; a batch where every sample scores the same teaches nothing")

# %% [markdown]
# ## Exercise 1.3 — predict the pass rates before sampling
# The policy is uniformly random. How many of the 256 strings are balanced (`n_balanced`)? How many pass the
# buggy check (`n_buggy`)? You can use these facts about counts:
#
# - The number of balanced strings of ${2n}$ brackets is the Catalan number $\binom{2n}{n}/(n + 1)$.
# - The number of strings of length ${2n}$ that *never* go below zero is $\binom{2n}{n}$.
#
# The buggy check passes every string that does go below zero, and also the balanced strings.

# %% exercise
### BEGIN SOLUTION
n_balanced = math.comb(8, 4) // 5                 # Catalan(4) = C(8,4)/(4+1) = 14
n_buggy = (256 - math.comb(8, 4)) + n_balanced     # ever-negative strings + balanced ones
### END SOLUTION

# %% check
assert n_balanced == sum(task.verify(s) for s in seqs) and n_buggy == sum(task.buggy_verify(s) for s in seqs)
print(f"✅ {n_balanced}/256 balanced; {n_buggy}/256 pass the buggy check — the loophole is 14× larger than the target")

# %% [markdown]
# ## Exercise 1.4 — the KL-regularised optimum
# Write `kl_opt(ref_p, R, beta)`. It must return `(pi, expected_R, kl)`, with
# $\pi^* = \pi_{\text{ref}} \exp(R/\beta)/Z$. Calculate $\mathrm{KL}(\pi^* \,\|\, \pi_{\text{ref}})$ directly. Then
# select `beta_for_08`. This is the largest $\beta$ in `grid` whose $\pi^*$ is balanced at least 80% of the time.

# %% exercise
def kl_opt(ref_p, R, beta):
    ### BEGIN SOLUTION
    w = ref_p * np.exp(R / beta)
    pi = w / w.sum()
    return pi, float(pi @ R), float(np.sum(pi * np.log(pi / ref_p)))
    ### END SOLUTION


grid = [3.0, 1.0, 0.5, 0.3, 0.2, 0.1]
### BEGIN SOLUTION
beta_for_08 = max(b for b in grid if kl_opt(ref_p, R, b)[1] >= 0.8)
### END SOLUTION

# %% check
for b in grid:
    mine, lib = kl_opt(ref_p, R, b), pg.kl_optimal(ref_p, R, b)
    assert np.allclose(mine[0], lib[0]) and abs(mine[1] - lib[1]) < 1e-12 and abs(mine[2] - lib[2]) < 1e-9
assert beta_for_08 == 0.3
print(f"✅ β = {beta_for_08}: P(balanced) {kl_opt(ref_p, R, 0.3)[1]:.3f} at a KL of {kl_opt(ref_p, R, 0.3)[2]:.2f} nats; "
      "β buys reward with KL, exactly")

# %% [markdown]
# ## Exercise 1.5 — predict the optimal thinking length
# Let $P(\text{correct} \mid L) = 1 - e_0 (1 - q)^L$, and let the cost per thinking token be $c$. Then
# $\text{reward} = P(\text{correct} \mid L) - c L$. Set the derivative to zero. Solve for $L^*$ (a real number).
# Then find the best *integer* $L$: calculate the reward at the integers around $L^*$. Use $e_0$ =
# 0.8, $q$ = 0.1, $c$ = 0.01.

# %% exercise
e0, q, c = 0.8, 0.1, 0.01
### BEGIN SOLUTION
L_star = math.log(c / (e0 * -math.log(1 - q))) / math.log(1 - q)
L_int = max(range(int(L_star) - 2, int(L_star) + 3), key=lambda L: 1 - e0 * (1 - q) ** L - c * L)
### END SOLUTION

# %% check
t = ThinkTask(e0=e0, q=q, max_think=64, cost=c)
assert abs(L_star - t.optimal_length()) < 1e-9 and L_int == 20
print(f"✅ L* = {L_star:.2f} → think 20 tokens: past that, a token of thought buys less than it costs")

# %% [markdown]
# ## Exercise 1.6 — what the leash preserves
# Under the buggy reward `Rb`, $\pi^* = \pi_{\text{ref}} \exp(R_b/\beta)/Z$ multiplies every string that passes by
# the same factor. Predict `limit_true` from the reference alone. `limit_true` is the probability that $\pi^*$
# gives to a truly balanced string as $\beta \to 0$. Compare it with the unregularised run of worked example 6
# (4.9%). Then set `fix` to the change that removes the problem, not only bounds it. The options are
# `"raise beta"`, `"fix the verifier"` or `"train longer"`.

# %% exercise
Rb = np.array([task.buggy_verify(s) for s in seqs])
### BEGIN SOLUTION
limit_true = float(ref_p @ R) / float(ref_p @ Rb)     # balanced share of the passing mass, unchanged by π*
fix = "fix the verifier"
### END SOLUTION

# %% check
assert abs(limit_true - pg.kl_optimal(ref_p, Rb, 0.01)[0] @ R) < 1e-6 and fix == "fix the verifier"
print(f"✅ π* keeps {limit_true:.1%} of the passing mass honest as β → 0 — the reference's share; unregularised RL drifted to 4.9% "
      "because the loophole is the easiest pass to reach. KL bounds the damage; only the checker removes it.")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We treat the model as a policy over tokens, and a completion as a trajectory.
# Each RL step samples completions from the current model with an inference engine. Then it scores them, with a
# verifier where the task permits it. Then it takes a gradient step. This step increases the log-probability of
# completions that are better than the batch baseline, and decreases the log-probability of the others.
#
# "The baseline is there for the variance. With a baseline, a constant in the reward is invisible. Without a
# baseline, the constant is only noise.
#
# "We keep a frozen reference and penalise the KL to it. The optimum is the reference, reweighted by
# $\exp(\text{reward}/\beta)$. Thus RL sharpens behaviour that the SFT model already has. It will not find what
# the model never samples.
#
# "We design against two failure modes. First, the policy optimises exactly the reward that we wrote. Thus any
# loophole in the verifier becomes the behaviour. Second, anything that the reward does not charge for (thinking
# length) grows. We use evals of the true objective as the gate, not the training reward."
#
# **Drill questions**
# 1. *Why does the subtraction of a baseline not bias the gradient?* Answer: because
#    $\mathbb{E}_\pi[\nabla \log \pi(y)] = \sum_y \nabla \pi(y) = \nabla 1 = 0$. Thus
#    $\mathbb{E}[b\,\nabla \log \pi] = 0$ for any $b$ that does not depend on $y$.
# 2. *The training reward went to 99%, and the eval accuracy decreased. What is the first hypothesis?* Answer:
#    reward hacking. The reward and the objective disagree at some point, and the policy found that point.
#    Examine the high-reward samples that fail the evals. Repair the verifier. A larger KL coefficient only
#    decreases the speed of the hack.
# 3. *What does the KL penalty buy, exactly?* Answer: a bound on how far the policy moves, with a closed-form
#    optimum $\pi_{\text{ref}} \exp(R/\beta)/Z$. The policy buys reward with KL, at a rate that $\beta$ sets.
#    Nothing outside the support of $\pi_{\text{ref}}$ appears.
