"""Distilling reasoning: a student copies how long a teacher thinks — and not what it knows.

The one idea: a reasoning trace teaches a procedure, including when to stop thinking. Here a policy is a
stopping rule — P(stop at t | still thinking) = σ(θ_t), a table like `rlcore`'s ThinkTask policy — and SFT
on traces is maximum likelihood, which for a table has a closed form: the empirical stopping rate at each
length (`LengthPolicy.fit`). So the student inherits the teacher's thinking-length distribution, tail and
all; filtering the traces (only correct ones, only short ones) reshapes it, trading accuracy for tokens;
and a trace carries the whole behaviour where an RL reward carries one bit, which is why distilling beat RL
on small models in R1's comparison (`reinforce` is the RL baseline). What does not transfer is the
knowledge: accuracy is 1 − e0·(1 − q)^L with the *student's* q.
"""
from __future__ import annotations

import numpy as np

from .tasks import ThinkToy


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class LengthPolicy:
    """Stop thinking at step t with probability σ(θ_t), t = 0..max_think−1; never stopping is truncation."""

    def __init__(self, max_think: int = 32, theta=None):
        self.max_think = max_think
        self.theta = np.zeros(max_think) if theta is None else np.array(theta, float)

    @classmethod
    def geometric(cls, mean: float, max_think: int = 32) -> "LengthPolicy":
        """Constant stopping rate 1/(mean + 1): a memoryless thinker with a long tail."""
        h = 1.0 / (mean + 1.0)
        return cls(max_think, np.full(max_think, np.log(h / (1 - h))))

    @classmethod
    def fit(cls, lengths, max_think: int = 32, prior: float = 1.0) -> "LengthPolicy":
        """SFT on traces = maximum likelihood; for a table, the empirical hazard (stops at t / reached t),
        with `prior` pseudo-observations at the untrained model's rate of 1/2 for lengths no trace reached."""
        L = np.asarray(lengths)
        reached = np.array([(L >= t).sum() for t in range(max_think)], float)
        stops = np.array([(L == t).sum() for t in range(max_think)], float)
        h = (stops + 0.5 * prior) / (reached + prior)
        return cls(max_think, np.log(h / (1 - h)))

    def length_distribution(self) -> np.ndarray:
        """P(L = t) for t < max_think, then P(truncated)."""
        h = _sigmoid(self.theta)
        survive = np.concatenate([[1.0], np.cumprod(1 - h)])
        return np.concatenate([survive[:-1] * h, [survive[-1]]])

    def sample(self, rng, n: int) -> np.ndarray:
        """n thinking lengths; max_think means the budget ran out inside the think block."""
        d = self.length_distribution()
        return np.minimum((d.cumsum() > rng.random((n, 1))).argmax(1), self.max_think)

    def expected(self, task: ThinkToy, q: float | None = None) -> dict:
        """Exact accuracy, mean length, p90 length and truncation rate; q is the solver's knowledge."""
        d = self.length_distribution()
        L = np.arange(self.max_think)
        acc = float(d[:-1] @ task.accuracy(L, q))
        mean = float(d[:-1] @ L + d[-1] * self.max_think)
        p90 = int(np.searchsorted(d.cumsum(), 0.9))
        return {"accuracy": acc, "length": mean, "p90": p90, "truncated": float(d[-1]),
                "tokens_per_correct": mean / acc if acc > 0 else float("inf")}


def traces(policy: LengthPolicy, task: ThinkToy, rng, n: int, q: float | None = None):
    """n traces from a (teacher) policy: thinking lengths and whether each final answer was right."""
    L = policy.sample(rng, n)
    p = np.where(L < policy.max_think, task.accuracy(L, q), 0.0)
    return L, rng.random(n) < p


def distil(lengths, correct, max_think: int = 32, keep: str = "all", max_len: int | None = None,
           prior: float = 1.0) -> LengthPolicy:
    """SFT a student on traces after filtering: keep="all" (plain SeqKD) or "correct" (rejection sampling);
    max_len drops traces that thought longer than a budget (budget-aware distillation)."""
    L, c = np.asarray(lengths), np.asarray(correct, bool)
    m = (c if keep == "correct" else np.ones(len(L), bool)) & (L < max_think)      # a new array: never the caller's
    if max_len is not None:
        m &= L <= max_len
    return LengthPolicy.fit(L[m], max_think, prior)


def reinforce(task: ThinkToy, rng, rollouts: int, batch: int = 16, lr: float = 2.0, q: float | None = None,
              cost: float = 0.0, policy: LengthPolicy | None = None) -> LengthPolicy:
    """The RL baseline (and how the teacher was made): sample lengths, reward = correct − cost·L, batch-mean
    baseline, closed-form gradient of log π(L) = Σ_{t<L} log(1 − σ(θ_t)) + log σ(θ_L). One bit per rollout."""
    pol = LengthPolicy(task.max_think) if policy is None else policy
    for _ in range(rollouts // batch):
        L, ok = traces(pol, task, rng, batch, q)
        r = ok - cost * L
        A = r - r.mean()
        h = _sigmoid(pol.theta)
        g = np.zeros(pol.max_think)
        for a, l in zip(A, L):
            g[:l] -= a * h[:l]                 # kept thinking at t < L: d log(1 − σ) = −σ
            if l < pol.max_think:
                g[l] += a * (1 - h[l])          # stopped at L: d log σ = 1 − σ
        pol.theta += lr * g / batch
    return pol
