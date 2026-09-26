"""Two verifiable toy environments: a checker says right or wrong, so no reward model is needed.

The one idea: RL with verifiable rewards needs only a *verifier* — a program that scores a completion — and
the verifier is the whole specification, loopholes included (`verifier="buggy"`). `SeqTask` is token-level:
emit brackets, and a deterministic check says whether they balance. `ThinkTask` is the thinking-model toy:
emit "think" tokens until answering; P(correct | L) = 1 − e0·(1 − q)^L, and every token may cost.

Both expose one interface: `n_states`, `n_actions`, `state(prompt, prefix)`, `done(prompt, prefix)` and
`score(prompt, actions, rng) -> (reward, info)`. The reward depends only on the final state (SeqTask's
"broken" state makes it so), so a table of per-state softmaxes can represent the KL-optimal policy exactly.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

THINK, ANSWER = 0, 1


@dataclass
class Trajectory:
    """One sampled completion: the states it visited, the tokens (actions) it chose, what it scored."""
    prompt: int
    states: list[int]
    actions: list[int]
    reward: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)


class SeqTask:
    """Emit `length` brackets, "(" (token 0) or ")" (token 1); correct iff the string is balanced — never below
    zero depth, ending at zero.

    verifier="buggy" scores with a checker that has a loophole (see `buggy_verify`); reward_fn(seq) replaces the
    verifier altogether (a reward model). `info["correct"]` always reports the true verifier, so the gap between
    reward and correctness stays visible.
    """

    def __init__(self, kind: str = "brackets", length: int = 8, verifier: str = "true", reward_fn=None):
        assert kind == "brackets" and verifier in ("true", "buggy")
        self.length, self.verifier, self.reward_fn = length, verifier, reward_fn
        self.n_actions, self.n_prompts = 2, 1
        self.width = length + 2                  # per position: depth 0..length, plus "broken" (went below zero)
        self.n_states = length * self.width

    def _depth(self, prefix) -> int:
        """The compressed history the verifier cares about: the depth, or −1 once the prefix went below zero."""
        d = 0
        for a in prefix:
            d += 1 if a == 0 else -1
            if d < 0:
                return -1
        return d

    def state(self, prompt: int, prefix) -> int:
        d = self._depth(prefix)
        return len(prefix) * self.width + (d if d >= 0 else self.width - 1)

    def done(self, prompt: int, prefix) -> bool:
        return len(prefix) >= self.length

    def verify(self, seq) -> float:
        """The true check."""
        return float(self._depth(seq) == 0)

    def buggy_verify(self, seq) -> float:
        """A checker with a loophole: it returns early with *pass* the moment depth goes negative (think of a
        test harness that counts an early `sys.exit(0)` as success)."""
        d = 0
        for a in seq:
            d += 1 if a == 0 else -1
            if d < 0:
                return 1.0                          # the bug: meant to be `return 0.0`
        return float(d == 0)

    def score(self, prompt: int, actions, rng=None):
        correct = self.verify(actions)
        reward = self.buggy_verify(actions) if self.verifier == "buggy" else correct
        reward = self.reward_fn(actions) if self.reward_fn else reward
        return reward, {"correct": correct, "length": len(actions), "truncated": False}

    # -- the space is small enough to compute every expectation exactly ------------------------------
    def all_sequences(self):
        return [list(s) for s in itertools.product(range(self.n_actions), repeat=self.length)]

    def as_trajectory(self, seq, reward: float = 0.0) -> Trajectory:
        """A given sequence as a trajectory (states recomputed): SFT demonstrations, DPO pairs."""
        return Trajectory(0, [self.state(0, seq[:t]) for t in range(len(seq))], list(seq), reward)

    def render(self, seq) -> str:
        return "".join("()"[a] for a in seq)

    def random_success_rate(self) -> float:
        """Exact P(correct) for a uniformly random policy: Catalan(n) / 2^(2n)."""
        seqs = self.all_sequences()
        return sum(self.verify(s) for s in seqs) / len(seqs)


class ThinkTask:
    """Think for L tokens, then answer. P(correct | L) = 1 − e0·(1 − q)^L; reward = correct − cost·L.

    Actions are THINK (0) and ANSWER (1). The state is (prompt, thinking tokens so far). A completion that
    reaches `max_think` thinking tokens without answering is *truncated* — `max_tokens` ran out inside the
    think block — and scores 0 unless `force_answer=True`, which models budget forcing: the environment
    closes the think block and the model answers with what it has, P(correct | max_think).
    `prompts` is a list of (e0, q) pairs, one per prompt (difficulty varies by prompt); default one prompt.
    """

    def __init__(self, e0: float = 0.8, q: float = 0.1, max_think: int = 32, cost: float = 0.0,
                 prompts=None, force_answer: bool = False):
        self.prompts = list(prompts) if prompts is not None else [(e0, q)]
        self.max_think, self.cost, self.force_answer = max_think, cost, force_answer
        self.n_prompts, self.n_actions = len(self.prompts), 2
        self.n_states = self.n_prompts * (max_think + 1)

    def accuracy(self, L, prompt: int = 0):
        """P(correct | L thinking tokens) = 1 − e0·(1 − q)^L."""
        e0, q = self.prompts[prompt]
        return 1.0 - e0 * (1.0 - q) ** np.asarray(L, dtype=float)

    def state(self, prompt: int, prefix) -> int:
        return prompt * (self.max_think + 1) + len(prefix)

    def done(self, prompt: int, prefix) -> bool:
        return bool(prefix) and prefix[-1] == ANSWER or len(prefix) >= self.max_think

    def score(self, prompt: int, actions, rng):
        L = sum(1 for a in actions if a == THINK)
        answered = bool(actions) and actions[-1] == ANSWER
        truncated = not answered and not self.force_answer
        p = 0.0 if truncated else float(self.accuracy(L, prompt))
        correct = float(rng.random() < p)
        return correct - self.cost * L, {"correct": correct, "length": L, "truncated": truncated}

    def optimal_length(self, cost: float | None = None, prompt: int = 0) -> float:
        """The L that maximises accuracy(L) − cost·L: where the marginal gain e0·(−ln(1−q))·(1−q)^L = cost."""
        c = self.cost if cost is None else cost
        e0, q = self.prompts[prompt]
        k = -math.log(1.0 - q)
        if c <= 0:
            return float(self.max_think)
        return float(min(max(math.log(c / (e0 * k)) / math.log(1.0 - q), 0.0), self.max_think))

    def expected(self, stop_probs, prompt: int = 0) -> dict:
        """Exact accuracy, mean thinking length, truncation rate and reward of a stopping policy."""
        d = self.length_distribution(stop_probs)
        L = np.arange(self.max_think)
        acc = float(np.dot(d[:-1], self.accuracy(L, prompt)))
        if self.force_answer:
            acc += float(d[-1] * self.accuracy(self.max_think, prompt))
        mean_len = float(np.dot(d[:-1], L) + d[-1] * self.max_think)
        return {"accuracy": acc, "length": mean_len, "truncated": 0.0 if self.force_answer else float(d[-1]),
                "reward": acc - self.cost * mean_len}

    def length_distribution(self, stop_probs) -> np.ndarray:
        """P(L = t) for t = 0..max_think−1 and, last, P(truncated), from per-position P(answer)."""
        s = np.asarray(stop_probs, dtype=float)[: self.max_think]
        survive = np.concatenate([[1.0], np.cumprod(1.0 - s)])
        return np.concatenate([survive[:-1] * s, [survive[-1]]])
