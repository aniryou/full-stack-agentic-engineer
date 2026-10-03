# %% [markdown]
# # 02 · Preferences, reward models and DPO
#
# **Tier:** T0: CPU only, numpy, no network, well under a minute. The bracket task of notebook 01 and a
# catalogue of 200 synthetic answers replace a model and a preference dataset. The notebook calculates every
# optimum exactly. Real DPO and reward-model training use TRL's `DPOTrainer` and `RewardTrainer` (primer §3).
#
# ## The one-minute version
# When no program can do a check of an answer, people compare two answers: "A is better than B".
#
# - **Bradley–Terry** reads that statement as a noisy comparison of rewards,
#   $P(A \succ B) = \sigma(r(A) - r(B))$. Thus a **reward model** is logistic regression on pairs.
# - **RLHF** then maximises $\mathbb{E}[r] - \beta\,\mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})$ with PPO. PPO is a
#   clipped policy-gradient step, plus a **value model** for the baseline (GAE).
# - **DPO** uses the fact that the KL-regularised optimum has a closed form,
#   $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$. DPO inverts it, $r = \beta \log(\pi^*/\pi_{\text{ref}}) + \text{const}$,
#   and substitutes $r$ into Bradley–Terry. The constant cancels. Then the policy trains directly on pairs with a
#   classification loss.
# - Its **implicit reward** $\beta \log(\pi/\pi_{\text{ref}})$ is what TRL logs as `rewards/chosen`. DPO gives up
#   two things. It sees only the pairs (no exploration). Second, it optimises the *margin*, thus the likelihood of
#   the chosen answer can decrease.
# - Also, any bias in the annotators (for example, a taste for long answers) becomes the reward. Harder
#   optimisation turns that bias into padding.
#
# Primer: `../PRIMER.md` §3.

# %%
import math

import numpy as np

from rlcore import Policy, SeqTask, pg, pref

task = SeqTask("brackets", 8)
seqs = task.all_sequences()
demos = [task.as_trajectory(s) for s in seqs if task.verify(s)]
ref = Policy.for_task(task)
for _ in range(2):                                   # the same weak SFT reference as notebook 01 (29.7% right)
    pg.sft_step(ref, demos, lr=1.0)
ref_p = ref.sequence_probs(task, seqs)
ok = np.array([task.verify(s) for s in seqs])

# %% [markdown]
# ## Worked example 1 — Bradley–Terry, and why a reward model has no zero point
# A hidden linear reward $r = 1.5\,x_1 - 0.5\,x_2$ is a function of two features. Annotators compare random pairs
# with $P(A \succ B) = \sigma(r_A - r_B)$. A fit of the same logistic model to "chosen minus rejected" recovers
# the weights.

# %%
rng = np.random.default_rng(0)
xa, xb = rng.standard_normal((4000, 2)), rng.standard_normal((4000, 2))
w_true = np.array([1.5, -0.5])
a_wins = rng.random(4000) < pref.bt_prob(xa @ w_true, xb @ w_true)
chosen, rejected = np.where(a_wins[:, None], xa, xb), np.where(a_wins[:, None], xb, xa)
w = pref.fit_bradley_terry(chosen, rejected, l2=0.0, steps=1500)
print("fitted weights", w.round(2), "true", w_true)
print(f"training loss {pref.bt_nll(w, chosen, rejected):.3f} (chance: ln 2 = {math.log(2):.3f})")
print("σ(2 − 1) =", round(float(pref.bt_prob(2, 1)), 4), "= σ(12 − 11) =", round(float(pref.bt_prob(12, 11)), 4))

# %% [markdown]
# Only differences are identifiable. If you add 10 to every reward, every probability stays the same. That is why
# TRL's `RewardTrainer` has `center_rewards_coefficient`, which pins the free constant near zero. It is also why
# reward-model scores from different runs are not comparable.
#
# ## Worked example 2 — RLHF in two stages: fit a reward model, then RL against it with a KL penalty
# The preferences are over bracket strings, from a true reward $r = 3 \cdot \text{balanced}$. The notebook samples
# the pairs from the reference. A real dataset gets its pairs in the same way. Stage 1 fits a one-feature
# reward model. Stage 2 runs REINFORCE with $\beta$ = 1 against it. The target is the closed form
# $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$.

# %%
R = 3.0 * ok
pistar, er_star, kl_star = pg.kl_optimal(ref_p, R, beta=1.0)
rng = np.random.default_rng(0)
ia, ib = rng.choice(256, 4096, p=ref_p), rng.choice(256, 4096, p=ref_p)
wins = rng.random(4096) < pref.bt_prob(R[ia], R[ib])
pairs = [(seqs[i], seqs[j]) if w_ else (seqs[j], seqs[i]) for i, j, w_ in zip(ia, ib, wins)]
print(f"{np.mean(ok[ia] == ok[ib]):.0%} of the pairs are ties in the true reward (their labels are coin flips)")

feat = lambda s: [task.verify(s)]
w_rm = pref.fit_bradley_terry([feat(c) for c, _ in pairs], [feat(r) for _, r in pairs], l2=0.0, steps=3000, lr=3.0)
print(f"stage 1, reward model: r̂ = {w_rm[0]:.2f} · balanced   (true 3)")
rm_task = SeqTask("brackets", 8, reward_fn=lambda s: w_rm[0] * task.verify(s))    # score with the reward model
rlhf = ref.copy()
pg.train_reinforce(rlhf, rm_task, np.random.default_rng(1), steps=400, batch=16, lr=0.3, ref=ref, beta=1.0)
print(f"stage 2, RL with β = 1: P(balanced) {pg.expected(rlhf, task, task.verify):.3f}; "
      f"closed form π*: {pistar @ ok:.3f} (reference {ref_p @ ok:.3f})")

# %% [markdown]
# ## Worked example 3 — DPO: skip the reward model
# DPO uses the same 4,096 pairs. It uses no reward model, and it does not sample. It has one classification loss
# on the policy's own log-probability ratios. The cell prints TRL's metric names while the policy trains.

# %%
data = pref.encode_pairs(task, pairs)
dpo = ref.copy()
for epoch in range(151):
    m = pref.dpo_step(dpo, ref, data, beta=1.0, lr=5.0)
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k} {v:+.3f}" for k, v in m.items()))
p_dpo = dpo.sequence_probs(task, seqs)
print(f"DPO: P(balanced) {p_dpo @ ok:.3f} vs π* {pistar @ ok:.3f};  KL(π_DPO‖π*) = {np.sum(p_dpo * np.log(p_dpo / pistar)):.4f}")
ir = np.array([pref.implicit_reward(dpo, ref, task, s, 1.0) for s in seqs])
print(f"implicit reward β·log(π/π_ref): balanced minus unbalanced = {ir[ok > 0].mean() - ir[ok == 0].mean():.2f} (true gap 3)")

# %% [markdown]
# Read off three things:
#
# - DPO gets to the same policy as RM + RL. It *is* the same objective, solved in closed form.
# - The implicit reward recovers the true reward gap (up to the constant that Bradley–Terry cannot see).
# - `rewards/chosen` is **negative**: the log-ratios of the chosen strings decreased.
#
# DPO pushes the *margin*. When the two answers of a pair share most of their tokens (here: the same states), DPO
# also wins with a decrease of both log-ratios. The rejected one decreases faster. Monitor `rewards/chosen` in
# real runs. A chosen log-probability that decreases is the usual first sign of over-training.
#
# ## Worked example 4 — PPO's value side, in brief: GAE
# PPO gives every token its own advantage from a learned value model ${V(s)}$:
#
# $$
# \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad A_t = \delta_t + \gamma\lambda\, A_{t+1}.
# $$
#
# In RLHF, the reward-model score is on the last token, and $-\beta \log(\pi/\pi_{\text{ref}})$ is on every
# token. The value model is typically as large as the policy. GRPO saves the memory of this value model
# (notebook 03).

# %%
r_tok, v_tok = [0.0, 0.0, 1.0], [0.5, 0.6, 0.8]
for lam in (0.0, 0.5, 1.0):
    print(f"λ = {lam}: advantages {pref.gae(r_tok, v_tok, gamma=1.0, lam=lam).round(3).tolist()}")

# %% [markdown]
# $\lambda$ = 0 trusts the value model (one-step TD error: low variance, biased if $V$ is incorrect). $\lambda$ = 1
# trusts the sampled return (Monte Carlo minus $V$: unbiased, noisy). PPO then uses the same clipped ratio that
# GRPO uses (notebook 03).
#
# ## Worked example 5 — an annotator who likes long answers
# There are 40 answers to one prompt. Each answer has five versions, padded with 0–800 filler tokens. Filler
# decreases the true quality by 0.3 per 100 tokens. The reference rarely pads. Annotators prefer higher quality
# **and** longer answers: $P(A \succ B) = \sigma(\Delta\text{quality} + 0.6\,\Delta\text{length}/100)$.
#
# Fit a reward model on 3,000 such pairs. Then optimise it harder and harder (smaller $\beta$, through the closed
# form). Monitor the true quality.

# %%
cat = pref.response_catalogue()
ref_c = np.exp(-cat["padding"] / 150)
ref_c /= ref_c.sum()
ch, rj = pref.length_biased_prefs(cat, 3000, length_weight=0.6, ref_probs=ref_c)
X = np.stack([cat["quality"], cat["length"] / 100], axis=1)
w_len = pref.fit_bradley_terry(X[ch], X[rj], steps=2000)
print(f"reward model: r̂ = {w_len[0]:.2f}·quality + {w_len[1]:.2f}·length/100  — it learned the annotators, faithfully")
print(f"{'beta':>5} {'KL':>6} {'RM score':>9} {'true quality':>13} {'mean length':>12}")
print(f"{'ref':>5} {0:6.2f} {ref_c @ (X @ w_len):9.2f} {ref_c @ cat['quality']:13.3f} {ref_c @ cat['length']:12.0f}")
for beta in (10, 3, 1, 0.5, 0.3, 0.1):
    pi, er, kl = pg.kl_optimal(ref_c, X @ w_len, beta)
    print(f"{beta:5} {kl:6.2f} {er:9.2f} {pi @ cat['quality']:13.3f} {pi @ cat['length']:12.0f}")
pig, _, klg = pg.kl_optimal(ref_c, cat["quality"], 0.1)
print(f"optimising true quality instead (β = 0.1): quality {pig @ cat['quality']:.3f}, length {pig @ cat['length']:.0f}")

# %% [markdown]
# The reward-model score increases monotonically. The true quality increases, gets to a peak around a KL of
# 0.4–1.7 nats, and then decreases below the reference. At the same time, the answers become four times longer.
# That shape (the proxy goes up, the gold goes up and then down) is reward-model **over-optimisation**. Length is
# its most common form in practice.
#
# The defences:
#
# - Cap the KL budget (stop early).
# - Control for length when you collect or fit preferences.
# - Prefer verifiable rewards where they exist (notebook 03).
# - Evaluate on the true objective.
#
# ## Exercise 2.1 — the Bradley–Terry gradient
# For weights `w` and feature differences `d = x_chosen − x_rejected` (one row per pair), return the gradient of
# the mean log-likelihood $\sum \log \sigma(w \cdot d)/n$. (Hint: $\partial \log \sigma(z)/\partial z = 1 - \sigma(z)$.)

# %% exercise
def bt_grad(w, d):
    ### BEGIN SOLUTION
    z = d @ w
    return (d * (1 - pref.sigmoid(z))[:, None]).mean(axis=0)
    ### END SOLUTION

# %% check
d = chosen - rejected
eps = 1e-6
num = [(-pref.bt_nll(w + eps * e, chosen, rejected) + pref.bt_nll(w - eps * e, chosen, rejected)) / (2 * eps) for e in np.eye(2)]
assert np.allclose(bt_grad(w, d), num, atol=1e-7)
assert np.allclose(bt_grad(np.zeros(2), d), d.mean(axis=0) / 2)
print("✅ the BT gradient: each pair pulls w along its feature difference, weighted by how wrong the model is")

# %% [markdown]
# ## Exercise 2.2 — DPO's loss by hand
# The policy increased the log-ratio $\log(\pi/\pi_{\text{ref}})$ of the chosen answer by +1. It decreased the
# log-ratio of the rejected answer by −1. With $\beta$ = 0.1, calculate the DPO loss `loss_a`. Then calculate
# `loss_b` for a pair where both log-ratios are 0.3.

# %% exercise
### BEGIN SOLUTION
loss_a = -math.log(1 / (1 + math.exp(-0.1 * (1.0 - (-1.0)))))
loss_b = -math.log(1 / (1 + math.exp(-0.1 * (0.3 - 0.3))))
### END SOLUTION

# %% check
assert round(loss_a, 6) == 0.598139 and abs(loss_b - math.log(2)) < 1e-12
assert np.isclose(loss_a, pref.dpo_loss(1.0, -1.0, 0.1))
print(f"✅ margin 2, β 0.1: loss {loss_a:.6f}; zero margin: ln 2 = {loss_b:.6f} — where every DPO run starts")

# %% [markdown]
# ## Exercise 2.3 — which pairs teach DPO the most?
# The gradient of DPO on a pair is
#
# $$
# \beta\,\sigma(-m)\,\bigl(\nabla \log \pi(y^+) - \nabla \log \pi(y^-)\bigr)
# $$
#
# with $m = \beta\,(\Delta^+ - \Delta^-)$. Write `pair_weight(m, beta)` $= \beta\,\sigma(-m)$. Then set
# `most_informative` to the index of the pair in `margins` that gets the largest update.

# %% exercise
margins = [2.0, 0.0, -1.5, 0.5]                      # m for four pairs
def pair_weight(m, beta):
    ### BEGIN SOLUTION
    return beta / (1 + math.exp(m))
    ### END SOLUTION


### BEGIN SOLUTION
most_informative = int(np.argmax([pair_weight(m, 0.1) for m in margins]))
### END SOLUTION

# %% check
assert np.isclose(pair_weight(0.0, 0.1), 0.05) and most_informative == 2
assert pair_weight(10, 0.1) < 1e-5
print("✅ mis-ranked pairs (m < 0) get up to β; pairs already ranked confidently get almost nothing — DPO is "
      "self-paced, like logistic regression")

# %% [markdown]
# ## Exercise 2.4 — generalised advantage estimation
# Write `my_gae(rewards, values, gamma, lam)`. Set $V$ after the last token to 0.

# %% exercise
def my_gae(rewards, values, gamma, lam):
    ### BEGIN SOLUTION
    adv, run = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        run = rewards[t] + gamma * nxt - values[t] + gamma * lam * run
        adv[t] = run
    return np.array(adv)
    ### END SOLUTION

# %% check
for lam in (0.0, 0.5, 0.95, 1.0):
    assert np.allclose(my_gae(r_tok, v_tok, 1.0, lam), pref.gae(r_tok, v_tok, 1.0, lam))
assert np.allclose(my_gae(r_tok, v_tok, 1.0, 1.0), [0.5, 0.4, 0.2])
print("✅ GAE: λ interpolates between the value model's one-step view and the sampled return")

# %% [markdown]
# ## Exercise 2.5 — choose the KL budget
# From the length-biased experiment, select the `best_beta` in `betas` that maximises **true** quality. Report
# the KL that it spends (`kl_budget`). This is the early-stopping point that a gold eval can give you.

# %% exercise
betas = [10, 5, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.1]
### BEGIN SOLUTION
gold = {b: pg.kl_optimal(ref_c, X @ w_len, b)[0] @ cat["quality"] for b in betas}
best_beta = max(gold, key=gold.get)
kl_budget = pg.kl_optimal(ref_c, X @ w_len, best_beta)[2]
### END SOLUTION

# %% check
assert best_beta == 0.7 and 0.4 < kl_budget < 1.2
print(f"✅ β = {best_beta}: true quality peaks at a KL of {kl_budget:.2f} nats; past it the RM is being gamed")

# %% [markdown]
# ## Exercise 2.6 — IPO's fixed margin
# IPO replaces $-\log \sigma(\beta \cdot \text{margin})$ with $(\text{margin} - 1/(2\beta))^2$, where
# $\text{margin} = \Delta^+ - \Delta^-$. For $\beta$ = 0.1, what margin does IPO drive a pair toward
# (`ipo_target`)? Also, what is the gradient of the DPO loss with respect to the margin as the margin $\to \infty$
# (`dpo_grad_at_inf`)?

# %% exercise
### BEGIN SOLUTION
ipo_target = 1 / (2 * 0.1)
dpo_grad_at_inf = 0.0
### END SOLUTION

# %% check
assert ipo_target == 5.0 and pref.ipo_loss(5.0, 0.0, 0.1) == 0.0 and dpo_grad_at_inf == 0.0
assert pref.dpo_loss(50.0, 0.0, 0.1) < pref.dpo_loss(40.0, 0.0, 0.1)          # DPO keeps rewarding a wider margin
print("✅ IPO stops at a margin of 1/(2β) = 5; DPO's loss keeps (slowly) paying for more, which on deterministic "
      "preferences pushes log-ratios without bound")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Where no program can grade the output, we collect pairwise preferences. We model
# them with Bradley–Terry, $P(A \succ B) = \sigma(r_A - r_B)$. The classic path fits a reward model and runs PPO
# against it, with a KL penalty to the SFT model. This path keeps four networks in memory: policy, reference,
# reward model and value model.
#
# "DPO gets the same KL-regularised optimum in closed form: $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$. Thus
# $r$ is $\beta \log(\pi/\pi_{\text{ref}})$, up to a constant that cancels in the pairwise likelihood. We train
# the policy directly on pairs with a classification loss. This needs two networks, and the policy does not
# sample. We monitor rewards/margins and rewards/chosen. We expect the chosen log-ratio to decrease, but a
# collapse means over-training.
#
# "The main risk on each path is the preference data itself. The biases of the annotators become the reward,
# length above all. Harder optimisation turns them into the behaviour. Thus we control for length, cap the KL
# budget, and evaluate on held-out human or verifiable judgements. Where a program can do a check of the answers,
# we use verifiable rewards instead."
#
# **Drill questions**
# 1. *Why does DPO need no reward model?* Answer: the optimum of the KL-regularised objective gives
#    $r = \beta \log(\pi^*/\pi_{\text{ref}}) + \beta \log Z(x)$. In the Bradley–Terry difference $r(y^+) - r(y^-)$,
#    the $Z$ term cancels. Thus you can write the likelihood in terms of the policy alone.
# 2. *rewards/chosen decreases during DPO. Is it a bug?* Answer: not necessarily. DPO optimises the margin. A
#    decrease of both log-ratios (the rejected one faster) wins it. A collapse of the chosen
#    log-probability or a degradation of the generations is a cause for concern. If that occurs, decrease the
#    learning rate or the epochs, and increase $\beta$.
# 3. *After RLHF, answers are 3× longer, and the win-rate against the old model is up. Do you ship it?* Answer:
#    not on that evidence. Length is the classic reward-model exploit, and judges share the bias. Compare at
#    matched length or with a length-controlled judge. Also examine the task accuracy.
