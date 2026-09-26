# %% [markdown]
# # 04 · Test-time compute
#
# **Tier:** T0 — CPU only, numpy, no network, well under a minute. A population of 400 synthetic questions stands
# in for a benchmark; every number is exact or a seeded simulation. The same measurements on a real thinking
# model (Qwen3-0.6B in vLLM, best-of-n and majority vote) are `thinking-lab` notebook
# `03_test_time_compute_for_real` (T1; at T0 it replays bundled outputs).
#
# ## The one-minute version
# There are two ways to spend more inference on one question. **Sequential**: think longer — worth it when the
# model is on the right track and needs more steps. **Parallel**: sample n answers and pick one — worth it when an
# attempt can go down a dead end, *if* something can pick: a **verifier** (tests, a checker) finds any correct
# sample; **majority vote** (self-consistency) needs no checker but only wins when the right answer is the most
# common one; a **reward model** picks through noise. Which split of a token budget is best depends on why
# questions are hard, the budget, and whether you have a verifier — and past a point every extra token buys
# less. Measure with the **unbiased pass@k** estimator 1 − C(n−c, k)/C(n, k) (never 1 − (1 − c/n)^k), and with
# **pass^k** (all k tries succeed) when what you need is reliability, not a lucky hit. Primer: `../PRIMER.md` §6.

# %%
import math

import numpy as np

from rlcore import ttc

qs = ttc.question_set()                              # 400 questions: two kinds of difficulty (see ttc.py)
print(f"questions: mean chance an attempt is on a workable approach {qs['a'].mean():.2f}; "
      f"median tokens to crack once on it {1 / np.median(qs['q']):,.0f}")

# %% [markdown]
# ## Worked example 1 — estimating pass@k without fooling yourself
# Draw n = 10 samples per question and count c correct. The tempting estimate of pass@5 is 1 − (1 − c/n)^5. Its
# expectation over the binomial draw is *below* the truth; the unbiased estimator's expectation *is* the truth.

# %%
print(f"{'p':>4} {'truth 1-(1-p)^5':>16} {'E[unbiased]':>12} {'E[plug-in]':>11}")
for p in (0.05, 0.1, 0.3, 0.6):
    print(f"{p:4} {1 - (1 - p) ** 5:16.4f} {ttc.expected_estimate(ttc.pass_at_k, 10, 5, p):12.4f} "
          f"{ttc.expected_estimate(ttc.plugin_pass_at_k, 10, 5, p):11.4f}")
print("n=10, c=3:", {k: round(ttc.pass_at_k(10, 3, k), 6) for k in (1, 5, 8)}, " pass^5:", ttc.pass_hat_k(10, 3, 5))
print("n=16, c=4, k=4: pass@4", round(ttc.pass_at_k(16, 4, 4), 6), " pass^4", round(ttc.pass_hat_k(16, 4, 4), 6))

# %% [markdown]
# The plug-in estimate is concave in c/n, so by Jensen it is biased low — worst for small p, where pass@k is most
# interesting. Estimate pass@k from n ≥ k samples with the product form HumanEval's code uses. And note the pair
# in the last line: a model that solves a task 1 time in 4 has pass@4 = 0.73 but pass^4 = 0.0005. An agent that
# must succeed every time a user asks is judged by pass^k (τ-bench), and test-time sampling does nothing for it.
#
# ## Worked example 2 — majority vote needs the right answer to be the most common
# One question; a sample is right with p = 0.4. Where the wrong 60% goes decides everything.

# %%
cases = {"one dominant misconception [0.5, 0.1]": [0.5, 0.1], "a narrow misconception [0.42, 0.18]": [0.42, 0.18],
         "two equal wrong answers [0.3, 0.3]": [0.3, 0.3], "four scattered wrong answers [0.15]*4": [0.15] * 4}
for name, wrong in cases.items():
    print(f"{name:40}", [round(ttc.majority_accuracy(0.4, wrong, n), 3) for n in (1, 5, 15, 31, 101)])
print("(columns: n = 1, 5, 15, 31, 101 votes)")

# %% [markdown]
# With scattered mistakes, voting turns a 40% sampler into a ~90% answerer. When one wrong answer out-polls the
# right one, voting converges on the *wrong* answer: at 50% against 40% accuracy falls from 0.400 to 0.278 by 31
# votes and 0.144 by 101 — worse than one sample at every n above 1. A *narrow* misconception (42% against 40%)
# shows why this is easy to miss: at small n the right answer often beats the split-up remainder, so the vote
# still helps (0.449 at 15 votes); it falls below one sample only past about 130 votes, and to 0 in the limit
# (exercise 4.4). Self-consistency works on math because wrong derivations rarely agree on the same wrong number.
#
# ## Worked example 3 — best-of-n: a verifier vs a reward model
# Pick the highest-scoring of n samples. A perfect verifier gives 1 − (1 − p)^n. A reward model sees correctness
# through noise.

# %%
rng = np.random.default_rng(0)
print(f"{'scorer':>22}", "  ".join(f"n={n:<3}" for n in (1, 4, 16, 64)))
print(f"{'verifier, exact':>22}", "  ".join(f"{1 - 0.7 ** n:.3f}" for n in (1, 4, 16, 64)))
for noise, name in ((0.0, "verifier"), (0.5, "RM, noise 0.5"), (1.0, "RM, noise 1.0")):
    print(f"{name:>22}", "  ".join(f"{ttc.best_of_n_accuracy(0.3, n, noise, rng):.3f}" for n in (1, 4, 16, 64)))
print("(the last three rows are Monte Carlo estimates, 20,000 trials each: the verifier row differs from the exact "
      "line only by sampling noise, about ±0.01)")

# %% [markdown]
# A noisy scorer turns "more samples" into "more chances to be fooled": gains flatten well below the verifier's
# line. A scorer with a *systematic* bias (notebook 02's length-loving reward model) is worse: best-of-n then
# selects for the bias. This is why RL with verifiable rewards and test-time search both lean on checkers.
#
# ## Worked example 4 — thinking longer: diminishing returns
# Accuracy of one sample as the thinking budget grows.

# %%
prev = None
for L in (0, 500, 1000, 2000, 4000, 8000, 16000):
    acc = ttc.accuracy(qs, L)
    gain = "" if prev is None else f"  +{(acc - prev[1]) / (L - prev[0]) * 1000:.3f} per 1K tokens"
    print(f"L = {L:6,d}: accuracy {acc:.3f}{gain}")
    prev = (L, acc)

# %% [markdown]
# Each doubling buys less, and single-sample accuracy plateaus near the share of attempts that start on a
# workable approach (0.65 here): thinking cannot rescue a dead end, another sample can. A model also "overthinks"
# when it keeps going after cracking the question (exercise 4.6).
#
# ## Worked example 5 — splitting a budget: n samples × L tokens
# Per question, n·(L + 50 answer tokens) ≤ budget.

# %%
only_slow = ttc.question_set(viable=None)            # every attempt is workable; questions differ only in length
for label, pop in (("questions only slow (a = 1)", only_slow), ("slow or dead-end (a ~ Beta(2,1))", qs)):
    for method in ("verifier", "vote"):
        rows = [ttc.allocate(pop, b, method=method)[1] for b in (1000, 4000, 16000)]
        print(f"{label:34} {method:8}: best (n, L) at 1K / 4K / 16K tokens: "
              + ", ".join(f"({n}, {L}) {acc:.3f}" for n, L, acc in rows))

# %% [markdown]
# Three lessons. When the only difficulty is length, sampling buys nothing that thinking does not — with a
# memoryless crack rate, n short attempts equal one long one minus the answer overheads — so one long sample
# wins. When attempts can dead-end, a verifier makes parallel sampling the better buy as the budget grows. And
# with only a vote, parallel sampling pays only once each sample is right often enough to out-poll the wrong
# answers — at 16K tokens here, not before.
#
# ## Worked example 6 — a smaller model with more samples
# A large model and a small one that costs a fifth as much per token and is weaker on both axes. Give the small
# one the same compute — five times the tokens.

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
# With a verifier the small model wins at equal compute — its many cheap samples cover the dead ends. With only a
# vote the large model wins: its single samples are right often enough, the small model's are not. Whether "a
# smaller model with more samples" wins is a property of the task *and* of having a checker; measure it on your
# own evals (07.2 evals notebook for the intervals such comparisons need).
#
# ## Exercise 4.1 — the unbiased pass@k
# Implement 1 − C(n−c, k)/C(n, k) in the stable product form 1 − Π_{i=n−c+1}^{n} (1 − k/i), returning 1.0 when
# n − c < k.

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
# kind. Compute `reliability`, the chance all five succeed (pass^5), and `any_success`, the chance that at least
# one of five samples succeeds (pass@5), both from c = 12, n = 16.

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
# A sample is right with p, otherwise it gives the *same* wrong answer. For odd n, P(majority right) is the
# binomial tail P(X > n/2). Implement it and check against the exact multi-answer function.

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
# As n → ∞, majority vote is right with probability → 1 if the right answer's share p beats every wrong
# answer's share, and → 0 if some wrong answer beats it. Write `wins_in_the_limit(p, wrong)` and check it
# against 41 votes on three cases.

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
# For the 400 questions `qs`, with a perfect verifier and 50 answer tokens per sample, find `best` = (n, L,
# accuracy) for a 4,000-token budget over n ∈ {1, 2, 4, 8, 16}. Then find `crossover`: the smallest budget in
# `budgets` at which the best n under majority vote is greater than 1.

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
# Suppose a sample on a workable approach cracks the question after Geometric(q) tokens. A model that always
# thinks for exactly L tokens spends L; one that stops the moment it cracks the question spends E[min(L_crack, L)]
# = (1 − (1 − q)^L)/q (and the full L when it never cracks it). For the questions in `qs`, compute `waste`: the
# fraction of an L = 4,000 thinking budget spent *after* the question was already cracked, averaged over questions
# and weighted by a (the chance the attempt was on a workable approach).

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
# **The two-minute version.** "Test-time compute is a budget we allocate per request. Thinking longer helps when
# the model is on the right track and needs steps; sampling several answers helps when an attempt can dead-end
# — but only if something picks the right one. With unit tests or a checker we sample in parallel and verify;
# without one, majority vote works when mistakes are scattered and backfires when there is a common
# misconception. Returns diminish with every doubling, and a fixed budget wastes tokens on questions solved
# early, so we set effort per request class rather than globally. We report pass@1 and, for agent flows, pass^k
# — reliability across repeated requests, which sampling does not improve — and estimate pass@k with the unbiased
# estimator from n ≥ k samples, with confidence intervals."
#
# **Drill questions**
# 1. *pass@1 is 40%, pass@16 is 95%. Is the model good?* — Only if you have a verifier that picks the right
#    sample in production; otherwise the user sees pass@1, and pass^k is what an agent flow is judged on.
# 2. *Self-consistency made our accuracy worse on one task. How?* — A wrong answer is more common than the right
#    one there (a shared misconception or a biased prompt); voting converges on it. Check the answer
#    distribution, not just accuracy.
# 3. *Small model with 16 samples or large model with one, same GPU budget?* — With a verifier, often the small
#    model (its samples cover dead ends); with only voting or a user reading one answer, the larger model's
#    per-sample accuracy wins. Measure both on your evals at equal cost.
