"""Two toy worlds small enough to enumerate, so every distillation claim can be checked exactly.

The one idea: to measure what a student learned from a teacher you need to know the truth. `ModLang` is a
next-token language whose true distribution is written down: after tokens (a, b) the next token is
(a + b) mod V with probability 1 − eps and one of its two neighbours otherwise, so the "wrong" tokens are not
equally wrong — the dark knowledge a soft target carries. A verifier checks the rule; with a short
continuation the whole output space (V^n) can be enumerated. `ThinkToy` is the reasoning toy, mirroring
`rlcore.tasks.ThinkTask`: think L tokens, then answer, right with probability 1 − e0·(1 − q)^L.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


class ModLang:
    """x_t = (x_{t-1} + x_{t-2}) mod V with probability 1 − eps, else (that ± 1) mod V, eps/2 each.
    `skew` moves the noise toward +1 (skew = 1: all of it): a fine-tuned dialect of the same language.

    A sequence is `order` prompt tokens followed by generated tokens. The main rule is a lookup table over
    V² contexts with no smooth structure, so a narrow network cannot hold all of it: a real capacity gap.
    """

    def __init__(self, V: int = 11, eps: float = 0.2, skew: float = 0.0):
        self.V, self.eps, self.skew, self.order = V, eps, skew, 2

    def main(self, ctxs) -> np.ndarray:
        c = np.asarray(ctxs)
        return (c[..., -1] + c[..., -2]) % self.V

    def true_probs(self, ctxs) -> np.ndarray:
        """The true next-token distribution for each context, shape (N, V)."""
        m = self.main(ctxs).reshape(-1)
        P = np.zeros((len(m), self.V))
        rows = np.arange(len(m))
        P[rows, m] = 1 - self.eps
        P[rows, (m + 1) % self.V] += self.eps * (1 + self.skew) / 2
        P[rows, (m - 1) % self.V] += self.eps * (1 - self.skew) / 2
        return P

    def contexts(self) -> np.ndarray:
        """All V² contexts (a, b)."""
        return np.array(list(itertools.product(range(self.V), repeat=self.order)))

    def sample_next(self, ctxs, rng) -> np.ndarray:
        """One next token per context, drawn from the true distribution."""
        P = self.true_probs(ctxs)
        return (P.cumsum(1) > rng.random((len(P), 1))).argmax(1)

    def generate(self, prompts, n_new: int, rng) -> np.ndarray:
        """Continue each prompt by n_new tokens drawn from the true language."""
        seqs = np.array(prompts, dtype=int)
        for _ in range(n_new):
            seqs = np.column_stack([seqs, self.sample_next(seqs[:, -self.order:], rng)])
        return seqs

    def rule_ok(self, seqs) -> np.ndarray:
        """Per generated position: did the token follow the main rule given the two before it? (N, n_new)"""
        s = np.asarray(seqs)
        return s[:, 2:] == (s[:, 1:-1] + s[:, :-2]) % self.V

    def verify(self, seqs) -> np.ndarray:
        """The verifier: every generated token follows the rule."""
        return self.rule_ok(seqs).all(1)

    def orbits(self) -> list[list[tuple]]:
        """The cycles of (a, b) → (b, a + b): a rule-following continuation never leaves its prompt's cycle,
        so a greedy teacher's data covers only the cycles of its prompts. Longest first, then by first context."""
        seen, out = set(), []
        for c in (tuple(int(x) for x in r) for r in self.contexts()):
            if c not in seen:
                cyc = []
                while c not in seen:
                    seen.add(c)
                    cyc.append(c)
                    c = (c[1], (c[0] + c[1]) % self.V)
                out.append(cyc)
        return sorted(out, key=lambda o: (-len(o), o[0]))

    def continuations(self, prompt, n_new: int) -> np.ndarray:
        """Every possible continuation of one prompt: V^n_new rows (enumeration for exact expectations)."""
        tails = np.array(list(itertools.product(range(self.V), repeat=n_new)))
        return np.column_stack([np.tile(prompt, (len(tails), 1)), tails])


class ThinkToy:
    """Think L tokens, then answer: P(correct | L) = 1 − e0·(1 − q)^L — `rlcore.tasks.ThinkTask`'s formula.

    Read q as how much one thinking token helps *this* model: a student with less knowledge has a smaller q,
    so copying a teacher's thinking length does not copy its accuracy. `max_think` tokens without an answer
    is a truncation, scored 0, as in ThinkTask.
    """

    def __init__(self, e0: float = 0.8, q: float = 0.1, max_think: int = 32):
        self.e0, self.q, self.max_think = e0, q, max_think

    def accuracy(self, L, q: float | None = None):
        q = self.q if q is None else q
        return 1.0 - self.e0 * (1.0 - q) ** np.asarray(L, dtype=float)

    def optimal_length(self, cost: float, q: float | None = None) -> float:
        """argmax_L accuracy(L) − cost·L: where e0·(−ln(1 − q))·(1 − q)^L = cost (ThinkTask.optimal_length)."""
        q = self.q if q is None else q
        k = -math.log(1.0 - q)
        if cost <= 0:
            return float(self.max_think)
        return float(min(max(math.log(cost / (self.e0 * k)) / math.log(1.0 - q), 0.0), self.max_think))
