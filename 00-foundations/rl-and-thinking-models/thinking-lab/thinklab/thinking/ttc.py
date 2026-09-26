"""ttc.py — test-time compute: pass@k, pass^k, majority vote, best-of-n, cost per correct answer.

One idea: there are two ways to spend more inference compute on a question — think *longer*
(sequential) or sample *more* answers and pick one (parallel) — and both are only worth what they
cost per **correct** answer. The estimators here are the standard ones:

* **pass@k** — P(at least one of k samples is right), estimated without bias from n ≥ k samples
  with c correct: ``1 − C(n−c, k) / C(n, k)`` (Chen et al. 2021, HumanEval; the numerically stable
  product form below is the one in their ``estimate_pass_at_k``). *Not* ``1 − (1 − c/n)^k``, which
  is biased for small n.
* **pass^k** — P(all k samples are right), ``C(c, k) / C(n, k)`` (τ-bench): the reliability metric an
  agent that must succeed every time is judged by. pass@k rises with k, pass^k falls.
* **maj@k** — self-consistency: the most common final answer among k samples (lm-eval's
  ``majority_vote`` filter). Needs no verifier, only answers that can be compared.
* **best-of-n** — score n samples with a verifier or reward model and keep the best; with a
  perfect verifier it *is* pass@n, with a noisy reward model it is less.

Every function is standard library only.
"""
from __future__ import annotations

import math
import random
from collections import Counter


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (HumanEval's estimator, product form)."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k: probability that k samples drawn without replacement are *all* correct."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("need 0 <= c <= n and 1 <= k <= n")
    return math.comb(c, k) / math.comb(n, k)


def majority_vote(answers: list):
    """The most common non-empty answer; ties go to the answer seen first. ``None`` if none."""
    votes = [a for a in answers if a is not None]
    if not votes:
        return None
    counts = Counter(votes)
    top = max(counts.values())
    return next(a for a in votes if counts[a] == top)


def maj_at_k(answers: list, truth, k: int, trials: int = 200, seed: int = 0) -> float:
    """Expected accuracy of majority voting over k samples, estimated by drawing k of the n recorded
    answers without replacement ``trials`` times (exact when k == n)."""
    if k > len(answers):
        raise ValueError("k larger than the number of samples")
    if k == len(answers):
        return float(majority_vote(answers) == truth)
    rng = random.Random(seed)
    return sum(majority_vote(rng.sample(answers, k)) == truth for _ in range(trials)) / trials


def best_of_n(samples: list, score) -> object:
    """The sample with the highest ``score(sample)`` (first wins ties)."""
    return max(samples, key=score) if samples else None


def best_of_n_accuracy(correct: list, rm_scores: list, k: int, trials: int = 200, seed: int = 0) -> float:
    """Accuracy of picking the top reward-model score among k of the n recorded samples."""
    rng = random.Random(seed)
    idx = list(range(len(correct)))
    hits = 0
    for _ in range(trials):
        pick = rng.sample(idx, k)
        hits += correct[max(pick, key=lambda i: rm_scores[i])]
    return hits / trials


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a pass rate — the statistic the 07 agent lab's eval gates use
    (``agentlab.evals.gate.wilson_interval``). 60 problems at 50% → about ±12 points."""
    if n == 0:
        return (0.0, 1.0)
    p = passes / n
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def cost_per_correct(output_tokens: float, accuracy: float, price_per_m_output: float,
                     input_tokens: float = 0.0, price_per_m_input: float = 0.0, samples: int = 1) -> float:
    """Dollars per *correct* answer: (cost of all samples of one question) / P(the chosen answer is right).
    Reasoning tokens are output tokens and are billed as such."""
    if accuracy <= 0:
        return math.inf
    per_question = samples * (output_tokens * price_per_m_output + input_tokens * price_per_m_input) / 1e6
    return per_question / accuracy


def compute_optimal(options: list, token_budget: float) -> dict | None:
    """Pick the most accurate option whose expected tokens per question fit ``token_budget``.
    ``options``: dicts with at least ``tokens`` and ``accuracy`` (e.g. {"mode", "n", "budget", ...}).
    Ties go to the cheaper option."""
    fitting = [o for o in options if o["tokens"] <= token_budget]
    if not fitting:
        return None
    return max(fitting, key=lambda o: (o["accuracy"], -o["tokens"]))
