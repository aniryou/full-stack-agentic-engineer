"""Test-time compute: spend more inference per question — think longer (sequential) or sample more (parallel).

The one idea: P(correct | L) = 1 − e0·(1 − q)^L reads as "each thinking token cracks the problem with
probability q; an uncracked answer is a guess, right with probability 1 − e0". An attempt may also start on an
approach that cannot work: thinking longer fixes slowness, another sample fixes dead ends — and sampling pays
only if a verifier, or a vote the right answer wins, picks the right sample. Measure with the unbiased
pass@k = 1 − C(n−c, k)/C(n, k), never 1 − (1 − c/n)^k, and with pass^k (all k succeed) for reliability.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al. 2021): 1 − C(n−c, k)/C(n, k),
    in the numerically stable product form used by HumanEval's `estimate_pass_at_k`."""
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (τ-bench): the chance that k draws without replacement are *all* correct, C(c, k)/C(n, k)."""
    return math.comb(c, k) / math.comb(n, k)


def plugin_pass_at_k(n: int, c: int, k: int) -> float:
    """The tempting estimator 1 − (1 − c/n)^k — biased low for k > 1 (it is concave in c/n)."""
    return 1.0 - (1.0 - c / n) ** k


def expected_estimate(estimator, n: int, k: int, p: float) -> float:
    """E[estimator] when c ~ Binomial(n, p): exact, to compare an estimator with the truth 1 − (1 − p)^k."""
    return float(sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) * estimator(n, c, k) for c in range(n + 1)))


def majority_vote(answers):
    """The most common answer; ties go to the answer seen first (lm-eval's maj@k filter behaves alike)."""
    return Counter(answers).most_common(1)[0][0]


def majority_accuracy(p: float, wrong, n: int) -> float:
    """Exact P(majority of n samples is right) when a sample is right with probability p and otherwise gives
    wrong answer i with probability wrong[i] (sum(wrong) = 1 − p). Ties are broken uniformly at random."""
    total = 0.0
    for c in range(1, n + 1):                      # c votes for the right answer; no wrong answer may beat it
        pc = math.comb(n, c) * p ** c
        for counts in itertools.product(range(min(c, n - c) + 1), repeat=len(wrong)):
            if sum(counts) != n - c:
                continue
            ways = math.factorial(n - c) / math.prod(math.factorial(x) for x in counts)
            ties = 1 + sum(1 for x in counts if x == c)
            total += pc * ways * math.prod(w ** x for w, x in zip(wrong, counts)) / ties
    return total


def best_of_n_accuracy(p: float, n: int, noise: float, rng, trials: int = 20000) -> float:
    """Best-of-n with a scorer that sees correctness through Gaussian noise (a reward model, not a checker):
    score = correct + noise·N(0, 1); pick the argmax. noise = 0 is a perfect verifier: 1 − (1 − p)^n."""
    correct = rng.random((trials, n)) < p
    score = correct + noise * rng.standard_normal((trials, n))
    return float(correct[np.arange(trials), score.argmax(axis=1)].mean())


# -- a population of questions of varying difficulty ---------------------------------------------------
def question_set(n: int = 400, median_q: float = 1 / 1500, sigma: float = 1.0, viable=(2.0, 1.0),
                 e0: float = 1.0, seed: int = 0) -> dict:
    """Two kinds of hard. a_j ~ Beta(*viable): the chance one attempt picks an approach that can work;
    q_j ~ lognormal(median_q, sigma): once on a workable approach, the chance each thinking token cracks it.
    e0 = 1 means no lucky guesses (open-ended answers). viable=None sets a_j = 1: the ThinkTask formula."""
    rng = np.random.default_rng(seed)
    a = np.ones(n) if viable is None else rng.beta(*viable, n)
    return {"a": a, "q": median_q * np.exp(sigma * rng.standard_normal(n)), "e0": e0}


def sample_accuracy(qs: dict, L) -> np.ndarray:
    """Per question, P(one sample with L thinking tokens is right) = 1 − e0·(1 − a·(1 − (1 − q)^L))."""
    return 1.0 - qs["e0"] * (1.0 - qs["a"] * (1.0 - (1.0 - qs["q"]) ** L))


def accuracy(qs: dict, L, n: int = 1, method: str = "single", wrong=(0.25, 0.25, 0.25, 0.25), trials: int = 200,
             seed: int = 0) -> float:
    """Mean accuracy over the questions at L thinking tokens per sample and n samples per question.
    "single": one sample; "verifier": any of n correct (a perfect checker picks it); "vote": majority of n,
    wrong mass split over distinct wrong answers in the proportions `wrong` (seeded Monte Carlo, `trials` votes
    per question; `majority_accuracy` is the exact version for one question)."""
    p = sample_accuracy(qs, L)
    if method == "single" or n == 1:
        return float(p.mean())
    if method == "verifier":
        return float((1.0 - (1.0 - p) ** n).mean())
    rng = np.random.default_rng(seed)
    cum = np.cumsum(np.column_stack([p] + [w * (1 - p) for w in wrong]), axis=1)[:, :-1]   # (Q, m) thresholds
    answer = (rng.random((len(p), trials, n))[..., None] > cum[:, None, None, :]).sum(-1)   # 0 = right
    counts = np.stack([(answer == k).sum(-1) for k in range(len(wrong) + 1)], axis=-1)
    top = counts.max(-1)
    wins = (counts[..., 0] == top) / (counts == top[..., None]).sum(-1)                     # ties split evenly
    return float(wins.mean())


def allocate(qs: dict, budget: int, answer_tokens: int = 50, ns=(1, 2, 4, 8, 16), method: str = "verifier"):
    """Split a per-question token budget between samples and thinking: n·(L + answer_tokens) ≤ budget.
    Returns rows (n, L, accuracy) and the best row."""
    rows = [(n, budget // n - answer_tokens, accuracy(qs, budget // n - answer_tokens, n, method))
            for n in ns if budget // n > answer_tokens]
    return rows, max(rows, key=lambda r: r[2])
