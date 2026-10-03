# %% [markdown]
# # 04 · Test-time compute
#
# **Tier:** T0: CPU only, numpy, no network, well under a minute. A population of 400 synthetic questions
# replaces a benchmark. Every number is exact or a seeded simulation. The `thinking-lab` notebook `03_test_time_compute_for_real`
# does the same measurements on a real thinking model (Qwen3-0.6B in vLLM, best-of-n and majority vote). That
# notebook is T1, and at T0 it replays bundled outputs.
#
# ## The one-minute version
# There are two ways to spend more inference on one question.
#
# - **Sequential**: think longer. This is worth the cost when the model is on the correct track and needs more
#   steps.
# - **Parallel**: sample $n$ answers and select one. This is worth the cost when an attempt can go into a dead
#   end, *if* something can select. The selector can be one of these:
#   - A **verifier** (tests, a checker) finds any correct sample.
#   - **Majority vote** (self-consistency) needs no checker, but wins only when the correct answer is the most
#     common one.
#   - A **reward model** selects through noise.
#
# The best split of a token budget depends on three things: why the questions are hard, the budget, and if you
# have a verifier. Also, after a point, each added token buys less. Measure with the **unbiased pass@k**
# estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ (never $1 - (1 - c/n)^k$). When you need reliability and not a
# lucky hit, also measure with **pass^k** (all $k$ tries succeed). Primer: `../PRIMER.md` §6.

# %%
import math

import numpy as np

from rlcore import ttc

qs = ttc.question_set()                              # 400 questions: two kinds of difficulty (see ttc.py)
print(f"questions: mean chance an attempt is on a workable approach {qs['a'].mean():.2f}; "
      f"median tokens to crack once on it {1 / np.median(qs['q']):,.0f}")

# %% [markdown]
# ## Worked example 1 — estimating pass@k without fooling yourself
# Draw $n$ = 10 samples per question. Count the $c$ correct samples. The obvious estimate of pass@5 is
# $1 - (1 - c/n)^5$. Its expectation over the binomial draw is *below* the truth. The expectation of the unbiased
# estimator *is* the truth.

# %%
print(f"{'p':>4} {'truth 1-(1-p)^5':>16} {'E[unbiased]':>12} {'E[plug-in]':>11}")
for p in (0.05, 0.1, 0.3, 0.6):
    print(f"{p:4} {1 - (1 - p) ** 5:16.4f} {ttc.expected_estimate(ttc.pass_at_k, 10, 5, p):12.4f} "
          f"{ttc.expected_estimate(ttc.plugin_pass_at_k, 10, 5, p):11.4f}")
print("n=10, c=3:", {k: round(ttc.pass_at_k(10, 3, k), 6) for k in (1, 5, 8)}, " pass^5:", ttc.pass_hat_k(10, 3, 5))
print("n=16, c=4, k=4: pass@4", round(ttc.pass_at_k(16, 4, 4), 6), " pass^4", round(ttc.pass_hat_k(16, 4, 4), 6))

# %% [markdown]
# The plug-in estimate is concave in ${c/n}$. Thus, by Jensen, it is biased low. The bias is worst for small $p$,
# where pass@k is of most interest. Estimate pass@k from $n \ge k$ samples with the product form that HumanEval's
# code uses.
#
# Also note the pair in the last line. A model that solves a task 1 time in 4 has pass@4 = 0.73, but
# pass^4 = 0.0005. For an agent that must succeed every time a user asks, the measure is pass^k (τ-bench). More
# samples at test time do nothing for it.
#
# ## Worked example 2 — majority vote needs the right answer to be the most common
# There is one question. A sample is correct with $p$ = 0.4. Where the incorrect 60% goes decides everything.

# %%
cases = {"one dominant misconception [0.5, 0.1]": [0.5, 0.1], "a narrow misconception [0.42, 0.18]": [0.42, 0.18],
         "two equal wrong answers [0.3, 0.3]": [0.3, 0.3], "four scattered wrong answers [0.15]*4": [0.15] * 4}
for name, wrong in cases.items():
    print(f"{name:40}", [round(ttc.majority_accuracy(0.4, wrong, n), 3) for n in (1, 5, 15, 31, 101)])
print("(columns: n = 1, 5, 15, 31, 101 votes)")

# %% [markdown]
# With scattered mistakes, the vote turns a 40% sampler into a ~90% answerer. When one incorrect answer out-polls
# the correct one, the vote converges on the *incorrect* answer. At 50% against 40%, accuracy decreases from
# 0.400 to 0.278 by 31 votes and to 0.144 by 101. This is worse than one sample at every $n$ above 1.
#
# A *narrow* misconception (42% against 40%) shows why this is easy to miss. At small $n$, the correct answer
# often beats the split-up remainder, so the vote still helps (0.449 at 15 votes). Its accuracy goes below that
# of one sample only after about 130 votes, and goes to 0 in the limit (exercise 4.4). Self-consistency works on
# math because incorrect derivations rarely agree on the same incorrect number.
#
# ## Worked example 3 — best-of-n: a verifier vs a reward model
# Of $n$ samples, select the one with the highest score. A perfect verifier gives $1 - (1 - p)^n$. A reward model
# sees correctness through noise.

# %%
rng = np.random.default_rng(0)
print(f"{'scorer':>22}", "  ".join(f"n={n:<3}" for n in (1, 4, 16, 64)))
print(f"{'verifier, exact':>22}", "  ".join(f"{1 - 0.7 ** n:.3f}" for n in (1, 4, 16, 64)))
for noise, name in ((0.0, "verifier"), (0.5, "RM, noise 0.5"), (1.0, "RM, noise 1.0")):
    print(f"{name:>22}", "  ".join(f"{ttc.best_of_n_accuracy(0.3, n, noise, rng):.3f}" for n in (1, 4, 16, 64)))
print("(the last three rows are Monte Carlo estimates, 20,000 trials each: the verifier row differs from the exact "
      "line only by sampling noise, about ±0.01)")

# %% [markdown]
# A noisy scorer turns "more samples" into "more chances for the scorer to make an error". The gains become flat
# well below the line of the verifier. A scorer with a *systematic* bias (the reward model of notebook 02 that
# likes long answers) is worse. Then best-of-n selects for the bias. This is why RL with verifiable rewards and
# test-time search both depend on checkers.
#
# ## Worked example 4 — thinking longer: diminishing returns
# This is the accuracy of one sample as the thinking budget increases.

# %%
prev = None
for L in (0, 500, 1000, 2000, 4000, 8000, 16000):
    acc = ttc.accuracy(qs, L)
    gain = "" if prev is None else f"  +{(acc - prev[1]) / (L - prev[0]) * 1000:.3f} per 1K tokens"
    print(f"L = {L:6,d}: accuracy {acc:.3f}{gain}")
    prev = (L, acc)

# %% [markdown]
# Each time the budget doubles, the gain is smaller. Single-sample accuracy gets to a plateau near the share of
# attempts that start on a workable approach (0.65 here). Longer thinking cannot rescue a dead end, but another
# sample can. A model also "overthinks" when it continues after it cracks the question (exercise 4.6).
#
# ## Worked example 5 — splitting a budget: n samples × L tokens
# Per question, $n \cdot (L + 50\ \text{answer tokens}) \le \text{budget}$.

# %%
only_slow = ttc.question_set(viable=None)            # every attempt is workable; questions differ only in length
for label, pop in (("questions only slow (a = 1)", only_slow), ("slow or dead-end (a ~ Beta(2,1))", qs)):
    for method in ("verifier", "vote"):
        rows = [ttc.allocate(pop, b, method=method)[1] for b in (1000, 4000, 16000)]
        print(f"{label:34} {method:8}: best (n, L) at 1K / 4K / 16K tokens: "
              + ", ".join(f"({n}, {L}) {acc:.3f}" for n, L, acc in rows))

# %% [markdown]
# There are three lessons:
#
# - When the only difficulty is length, more samples buy nothing that longer thinking does not. With a
#   memoryless crack rate, $n$ short attempts equal one long attempt minus the answer overheads. Thus one long
#   sample wins.
# - When attempts can end in a dead end, a verifier makes parallel samples the better buy as the budget grows.
# - With only a vote, parallel samples pay only when each sample is correct sufficiently often to out-poll the
#   incorrect answers. Here this occurs at 16K tokens, not before.
#
# ## Worked example 6 — a smaller model with more samples
# There is a large model and a small model. The small model costs a fifth as much per token and is weaker on both
# axes. Give the small model the same compute, that is five times the tokens.

# %%
big = ttc.question_set(median_q=1 / 1000, viable=(4.0, 1.0), seed=1)
small = ttc.question_set(median_q=1 / 2500, viable=(1.5, 1.0), seed=1)
print(f"{'budget (large-model tokens)':>28} {'large':>22} {'small, 5x tokens':>22}")
for method in ("verifier", "vote"):
    for b in (1000, 4000):
        _, bb = ttc.allocate(big, b, method=method, ns=(1, 2, 4, 8, 16, 32))
        _, bs = ttc.allocate(small, 5 * b, method=method, ns=(1, 2, 4, 8, 16, 32))
        print(f"{method:>9} {b:18,d} {f'n={bb[0]}, L={bb[1]}: {bb[2]:.3f}':>22} {f'n={bs[0]}, L={bs[1]}: {bs[2]:.3f}':>22}")

# %% [markdown]
# With a verifier, the small model wins at equal compute. Its many low-cost samples cover the dead ends. With only
# a vote, the large model wins. Its single samples are correct sufficiently often, and the samples of the small model
# are not. The result for "a smaller model with more samples" is a property of the task *and* of the presence of
# a checker. Measure it on your own evals (the 07.2 evals notebook has the intervals that such comparisons need).
#
# ## Exercise 4.1 — the unbiased pass@k
# Write $1 - \binom{n-c}{k}/\binom{n}{k}$ in the stable product form
#
# $$
# 1 - \prod_{i=n-c+1}^{n} \left(1 - \frac{k}{i}\right),
# $$
#
# Return 1.0 when ${n - c < k}$.

# %% exercise
def my_pass_at_k(n, c, k):
    ### BEGIN SOLUTION
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
    ### END SOLUTION

# %% check
for n, c, k in [(10, 3, 1), (10, 3, 5), (10, 3, 8), (16, 4, 4), (64, 16, 8), (200, 7, 100)]:
    assert abs(my_pass_at_k(n, c, k) - (1 - math.comb(n - c, k) / math.comb(n, k))) < 1e-12
print("✅ pass@5 from 3/10 correct is", round(my_pass_at_k(10, 3, 5), 6), "— the product form never builds a huge binomial")

# %% [markdown]
# ## Exercise 4.2 — pass^k for reliability
# An agent flow succeeds on 12 of 16 runs of a task in your eval. A user makes five separate requests of this
# kind. Calculate these two values, both from $c$ = 12, $n$ = 16:
#
# - `reliability`: the chance that all five succeed (pass^5).
# - `any_success`: the chance that at least one of five samples succeeds (pass@5).

# %% exercise
### BEGIN SOLUTION
reliability = math.comb(12, 5) / math.comb(16, 5)
any_success = my_pass_at_k(16, 12, 5)
### END SOLUTION

# %% check
assert abs(reliability - ttc.pass_hat_k(16, 12, 5)) < 1e-12 and any_success == 1.0
print(f"✅ 75% per try: pass@5 = {any_success:.0%} but pass^5 = {reliability:.1%} — sampling hides unreliability")

# %% [markdown]
# ## Exercise 4.3 — majority of n between two answers
# A sample is correct with $p$. Otherwise, it gives the *same* incorrect answer. For odd $n$,
# $P(\text{majority right})$ is the binomial tail ${P(X > n/2)}$. Write it. Compare it with the exact
# multi-answer function.

# %% exercise
def majority_two(p, n):
    ### BEGIN SOLUTION
    return sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n // 2 + 1, n + 1))
    ### END SOLUTION

# %% check
for p in (0.3, 0.5, 0.6, 0.8):
    for n in (1, 3, 9, 21):
        assert abs(majority_two(p, n) - ttc.majority_accuracy(p, [1 - p], n)) < 1e-12
print(f"✅ p = 0.6 → {majority_two(0.6, 21):.3f} with 21 votes; p = 0.4 → {majority_two(0.4, 21):.3f}: voting amplifies "
      "whichever answer is ahead")

# %% [markdown]
# ## Exercise 4.4 — predict the limit
# As $n \to \infty$, majority vote is correct with probability $\to 1$ if the share $p$ of the correct answer
# beats the share of every incorrect answer. The probability is $\to 0$ if some incorrect answer beats it. Write
# `wins_in_the_limit(p, wrong)`. Compare it with 41 votes on three cases.

# %% exercise
def wins_in_the_limit(p, wrong):
    ### BEGIN SOLUTION
    return p > max(wrong)
    ### END SOLUTION

# %% check
for p, wrong in [(0.3, [0.2] * 3 + [0.1]), (0.4, [0.42, 0.18]), (0.25, [0.5, 0.25])]:
    big_n = ttc.majority_accuracy(p, wrong, 41) if len(wrong) <= 2 else ttc.accuracy(
        {"a": np.ones(1), "q": np.zeros(1), "e0": 1 - p}, 0, 41, "vote", wrong=[w / (1 - p) for w in wrong], trials=4000)
    assert (big_n > 0.5) == wins_in_the_limit(p, wrong), (p, wrong, big_n)
print("✅ in the limit a 30% answer wins against scattered 20% mistakes and a 40% answer loses to one 42% mistake "
      f"(at 41 votes it is still right {ttc.majority_accuracy(0.4, [0.42, 0.18], 41):.2f} of the time: the limit is slow)")

# %% [markdown]
# ## Exercise 4.5 — spend a budget
# Use the 400 questions `qs`, a perfect verifier and 50 answer tokens per sample. Find
# `best` = $(n, L, \text{accuracy})$ for a 4,000-token budget over $n \in \{1, 2, 4, 8, 16\}$. Then find
# `crossover`. It is the smallest budget in `budgets` at which the best $n$ under majority vote is greater than 1.

# %% exercise
budgets = [2000, 4000, 8000, 12000, 16000, 32000]
### BEGIN SOLUTION
best = max(((n, 4000 // n - 50, ttc.accuracy(qs, 4000 // n - 50, n, "verifier")) for n in (1, 2, 4, 8, 16)),
           key=lambda r: r[2])
crossover = next(b for b in budgets if ttc.allocate(qs, b, method="vote")[1][0] > 1)
### END SOLUTION

# %% check
assert best == ttc.allocate(qs, 4000)[1] and best[0] == 8
assert crossover == 16000
print(f"✅ with a verifier: {best[0]} samples of {best[1]} tokens ({best[2]:.3f}); with a vote, parallel pays only from "
      f"{crossover:,} tokens")

# %% [markdown]
# ## Exercise 4.6 — thinking past the answer
# Assume that a sample on a workable approach cracks the question after $\operatorname{Geometric}(q)$ tokens. A
# model that always thinks for exactly $L$ tokens spends $L$. A model that stops as soon as it cracks the question
# spends $\mathbb{E}[\min(L_{\text{crack}}, L)] = (1 - (1 - q)^L)/q$ (and the full $L$ when it never cracks it).
# For the questions in `qs`, calculate `waste`. This is the fraction of an $L$ = 4,000 thinking budget that the
# model spends *after* it already cracked the question. Average it over the questions, and weight it by $a$ (the
# chance that the attempt was on a workable approach).

# %% exercise
L_fixed = 4000
### BEGIN SOLUTION
a, q = qs["a"], qs["q"]
used_if_stopping = a * (1 - (1 - q) ** L_fixed) / q + (1 - a) * L_fixed
waste = float(1 - used_if_stopping.mean() / L_fixed)
### END SOLUTION

# %% check
assert 0.25 < waste < 0.45
print(f"✅ {waste:.0%} of a fixed 4K-token think happens after the answer is found — the case for adaptive budgets "
      "and effort routing (notebook 05, primer §7)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Test-time compute is a budget that we allocate per request. Longer thinking helps
# when the model is on the correct track and needs steps. Several sampled answers help when an attempt can end in
# a dead end, but only if something selects the correct one.
#
# "With unit tests or a checker, we sample in parallel and use the checker on each sample. Without one,
# majority vote works when mistakes are scattered. It makes the result worse when there is a common
# misconception.
#
# "Returns decrease each time the budget doubles. One budget for all questions wastes tokens on questions that
# the model solves early. Thus we set effort per request class, not globally. We report pass@1 and, for agent
# flows, pass^k. pass^k is reliability across repeated requests, which more samples do not improve. We estimate
# pass@k with the unbiased estimator from $n \ge k$ samples, with confidence intervals."
#
# **Drill questions**
# 1. *pass@1 is 40%, pass@16 is 95%. Is the model good?* Answer: only if you have a verifier that selects the
#    correct sample in production. If not, the user sees pass@1. Also, the measure for an agent flow is pass^k.
# 2. *Self-consistency made our accuracy worse on one task. How?* Answer: on that task, an incorrect answer is
#    more common than the correct one. The cause can be a shared misconception or a biased prompt. The vote
#    converges on that incorrect answer. Examine the answer distribution, not only the accuracy.
# 3. *A small model with 16 samples or a large model with one, at the same GPU budget?* Answer: with a verifier,
#    often the small model (its samples cover dead ends). With only a vote, or with a user who reads one answer,
#    the per-sample accuracy of the larger model wins. Measure both on your evals at equal cost.
