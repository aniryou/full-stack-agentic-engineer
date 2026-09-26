"""data.py — a toy language with domains, so a router has something to specialise on.

One idea: experts specialise only if the router's input carries a signal worth routing on. Here
every sequence starts with a domain tag, then follows that domain's own rule (a permutation of the
content tokens) with a little noise. The rule for the next token depends on the tag seen at
position 0, so the model needs attention to carry the tag forward — and after attention, the hidden
state the router sees says which domain it is in. Numpy only: the data exists without torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyTask:
    n_domains: int = 8
    n_content: int = 48
    seq_len: int = 16
    noise: float = 0.1          # probability that the next token is uniform instead of the rule's
    rule_seed: int = 0

    @property
    def vocab(self) -> int:
        """Tag tokens 0 .. D-1, content tokens D .. D + C - 1."""
        return self.n_domains + self.n_content

    def rules(self) -> np.ndarray:
        """``[domains, content]``: the next content token under each domain's rule."""
        rng = np.random.default_rng(self.rule_seed)
        return np.stack([rng.permutation(self.n_content) for _ in range(self.n_domains)])

    def batch(self, rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
        """(tokens ``[size, seq_len]`` int64, domain ``[size]``)."""
        rules = self.rules()
        dom = rng.integers(0, self.n_domains, size)
        x = np.empty((size, self.seq_len), dtype=np.int64)
        x[:, 0] = dom
        cur = rng.integers(0, self.n_content, size)
        for t in range(1, self.seq_len):
            x[:, t] = self.n_domains + cur
            follow = rules[dom, cur]
            cur = np.where(rng.random(size) < self.noise, rng.integers(0, self.n_content, size), follow)
        return x, dom

    def bayes_loss(self) -> float:
        """The lowest achievable cross-entropy (nats) on a content position whose domain is known:
        the rule's token has probability 1 - noise + noise/C, every other token noise/C."""
        c, n = self.n_content, self.noise
        p_rule, p_other = 1 - n + n / c, n / c
        return -(p_rule * math.log(p_rule) + (c - 1) * p_other * math.log(p_other))
