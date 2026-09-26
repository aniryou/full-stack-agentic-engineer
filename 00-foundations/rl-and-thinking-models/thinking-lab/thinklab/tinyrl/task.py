"""task.py — a verifiable toy task where a scratchpad really helps: the last digit of a sum.

One idea: a transformer spends the same fixed amount of computation on every token it emits, so a
problem that needs K sequential steps is hard to answer *in one token* but easy to answer *after
writing the intermediate steps down*. Thinking tokens buy serial computation.

The task (``DigitSum``): the prompt is K digits and ``=``; the answer is the last digit of their sum.

    prompt       3 5 8 2 6 =
    no thinking  <think> </think> 4 <eos>                    one token must hold (3+5+8+2+6) mod 10
    scratchpad   <think> 3 8 6 8 4 </think> 4 <eos>          running sums mod 10: each step is local
    partial      <think> 3 8 6 </think> 4 <eos>              stopped after 3 of 5 steps: 2 left to do "in the head"

A *scratchpad of length j* writes the first j running sums. The answer after it needs K − j more
additions done without writing them down, so accuracy rises with j: the toy analogue of
"P(correct | thinking length)". The verifier checks format and the final answer only — an
*outcome* reward, like DeepSeek-R1's rule-based accuracy reward — so RL is free to discover that
longer scratchpads pay. Pure Python: no torch needed to build data or score completions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

EQ, THINK, END_THINK, EOS, PAD = 10, 11, 12, 13, 14
VOCAB = 15
NAMES = {EQ: "=", THINK: "<think>", END_THINK: "</think>", EOS: "<eos>", PAD: "<pad>"}


@dataclass(frozen=True)
class Problem:
    digits: tuple
    base: int = 10

    @property
    def answer(self) -> int:
        return sum(self.digits) % self.base

    @property
    def prompt(self) -> list:
        return list(self.digits) + [EQ]

    def running_sums(self) -> list:
        out, s = [], 0
        for d in self.digits:
            s = (s + d) % self.base
            out.append(s)
        return out


@dataclass(frozen=True)
class DigitSum:
    """K digits in, the last digit of their sum out. ``max_completion`` = K + 4 tokens."""
    k: int = 6
    base: int = 10

    @property
    def prompt_len(self) -> int:
        return self.k + 1

    @property
    def max_completion(self) -> int:
        return self.k + 4                      # <think> + K sums + </think> + answer + <eos>

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.max_completion

    def sample(self, rng: random.Random) -> Problem:
        return Problem(tuple(rng.randrange(self.base) for _ in range(self.k)), self.base)

    def demo(self, p: Problem, scratch: int) -> list:
        """A correct completion with a scratchpad of ``scratch`` running sums (0 = no thinking)."""
        if not 0 <= scratch <= self.k:
            raise ValueError(f"scratch must be in 0..{self.k}")
        return [THINK] + p.running_sums()[:scratch] + [END_THINK, p.answer, EOS]


def parse(completion: list) -> dict:
    """Split a sampled completion into its parts. ``ok`` is False when the format is broken."""
    toks = list(completion)
    if EOS in toks:
        toks = toks[: toks.index(EOS) + 1]
    out = {"ok": False, "scratch": None, "answer": None, "length": len(toks), "finished": EOS in toks}
    if len(toks) < 4 or toks[0] != THINK or END_THINK not in toks:
        return out
    end = toks.index(END_THINK)
    body = toks[1:end]
    tail = toks[end + 1:]
    if any(t > 9 for t in body) or len(tail) != 2 or tail[0] > 9 or tail[1] != EOS:
        return out
    out.update(ok=True, scratch=len(body), answer=tail[0])
    return out


def reward(p: Problem, completion: list) -> float:
    """The verifier: 1.0 for a well-formed completion whose final answer is right, else 0.0.
    It never looks at the scratchpad's contents — only the outcome is rewarded."""
    r = parse(completion)
    return 1.0 if r["ok"] and r["answer"] == p.answer else 0.0


def render(tokens: list) -> str:
    return " ".join(NAMES.get(t, str(t)) for t in tokens if t != PAD)


def sft_batch(task: DigitSum, n: int, rng: random.Random, scratch_weights: list | None = None) -> list:
    """``n`` (problem, completion) demonstrations; scratchpad lengths drawn from ``scratch_weights``
    (uniform over 0..K by default: the warm-up shows every length, it does not prefer one)."""
    lengths = list(range(task.k + 1))
    w = scratch_weights or [1.0] * len(lengths)
    out = []
    for _ in range(n):
        p = task.sample(rng)
        out.append((p, task.demo(p, rng.choices(lengths, weights=w)[0])))
    return out
