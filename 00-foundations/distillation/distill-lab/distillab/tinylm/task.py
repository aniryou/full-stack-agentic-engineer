"""task.py — the verifiable toy task a tiny teacher and a tiny student share: a sum, with an optional scratchpad.

One idea: to see what distillation transfers, the teacher must know something the student's data does not
show. Here the *labelled data* mostly skips the working (an answer key), while the *teacher* shows its work
most of the time — so a student that copies the teacher inherits a behaviour (write the running sums, then
answer) that the labels alone would never teach it, and the accuracy that comes with it.

The task (``SumTask``): the prompt is K digits in base b and ``=``; the answer is the last digit of their sum.

    prompt        3 1 4 0 2 4 =
    direct        <think> </think> 4 <eos>                 one token must hold (3+1+4+0+2+4) mod 5
    scratchpad    <think> 3 4 3 3 0 4 </think> 4 <eos>     running sums mod 5: each step is one local addition
    partial       <think> 3 4 3 </think> 4 <eos>           stopped after 3 of 6 steps: 3 more "in the head"

A two-layer transformer cannot add six digits inside the one token it emits for the answer, but it can if it
writes the running sums first: each scratchpad token is one more step of serial computation. So accuracy
rises with the scratchpad length j, and the length the model *chooses* is a behaviour a student can inherit.

Two mixes of demonstrations (probability of each scratchpad length j = 0..K):

    TEACHER_MIX   what the teacher was trained on: the full scratchpad 70% of the time, no scratchpad 10%,
                  a partial one otherwise — a model with a spread of "thinking lengths"
    DATA_MIX      the labelled set the student has: the answer alone 80% of the time, the full working 20%

The verifier checks the format and the final answer only (an outcome check, as for any verifiable task).
Pure Python: nothing here needs torch. The same task shape as thinking-lab's ``tinyrl.task`` (re-implemented;
labs are standalone), with the distillation mixes on top.
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
    base: int = 5

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


def full_mix(k: int) -> list:
    """Always the full scratchpad."""
    return [0.0] * k + [1.0]


def teacher_mix(k: int, full: float = 0.7, none: float = 0.1) -> list:
    """The teacher's training demonstrations: full scratchpad ``full``, none ``none``, partial lengths share the rest."""
    rest = (1.0 - full - none) / max(1, k - 1)
    return [none] + [rest] * (k - 1) + [full]


def data_mix(k: int, full: float = 0.2) -> list:
    """The labelled data the student has: the answer alone, except ``full`` of the time."""
    return [1.0 - full] + [0.0] * (k - 1) + [full]


@dataclass(frozen=True)
class SumTask:
    """K digits in base ``base`` in, the last digit of their sum out. Completions are 4 to K + 4 tokens."""
    k: int = 6
    base: int = 5

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
        """A correct completion with a scratchpad of ``scratch`` running sums (0 = answer directly)."""
        if not 0 <= scratch <= self.k:
            raise ValueError(f"scratch must be in 0..{self.k}")
        return [THINK] + p.running_sums()[:scratch] + [END_THINK, p.answer, EOS]

    def demos(self, n: int, rng: random.Random, mix: list) -> list:
        """``n`` (problem, completion) pairs with scratchpad lengths drawn from ``mix`` (weights over 0..K)."""
        if len(mix) != self.k + 1:
            raise ValueError(f"mix needs {self.k + 1} weights, got {len(mix)}")
        lengths = list(range(self.k + 1))
        out = []
        for _ in range(n):
            p = self.sample(rng)
            out.append((p, self.demo(p, rng.choices(lengths, weights=mix)[0])))
        return out


def parse(completion: list) -> dict:
    """Split a completion into its parts. ``ok`` is False when the format is broken."""
    toks = list(completion)
    if EOS in toks:
        toks = toks[: toks.index(EOS) + 1]
    out = {"ok": False, "scratch": None, "answer": None, "length": len(toks), "finished": EOS in toks}
    if len(toks) < 4 or toks[0] != THINK or END_THINK not in toks:
        return out
    end = toks.index(END_THINK)
    body, tail = toks[1:end], toks[end + 1:]
    if any(t > 9 for t in body) or len(tail) != 2 or tail[0] > 9 or tail[1] != EOS:
        return out
    out.update(ok=True, scratch=len(body), answer=tail[0])
    return out


def verify(p: Problem, completion: list) -> bool:
    """The verifier: a well-formed completion whose final answer is right. It never reads the scratchpad."""
    r = parse(completion)
    return r["ok"] and r["answer"] == p.answer


def render(tokens: list) -> str:
    return " ".join(NAMES.get(t, str(t)) for t in tokens if t != PAD)


def from_text(text: str, base: int = 5) -> list | None:
    """Token ids for a rendered sequence (``render``'s inverse), or None when a word is not a token.
    Notebook 02 uses it to feed the fake teacher's digit-sum completions to the tiny student."""
    inv = {v: k for k, v in NAMES.items()}
    out = []
    for w in text.split():
        if w in inv:
            out.append(inv[w])
        elif w.isdigit() and int(w) < base:
            out.append(int(w))
        else:
            return None
    return out


def from_teacher_text(question: str, content: str | None, base: int = 5) -> tuple | None:
    """A served teacher's answer to a ``modsum`` question (``distillab.data``) as a tiny-model training pair:
    ``(Problem, completion tokens)`` — the running sums it wrote become the scratchpad, its boxed answer the answer.
    None when the text is not in that shape (a different kind, a missing line, an out-of-range digit)."""
    import re
    q = re.search(r"\(([\d + ]+)\) mod (\d+)", question or "")
    sums = re.search(r"Running sums mod \d+: ([\d ]+)", content or "")
    ans = re.search(r"\\boxed\{(\d+)\}", content or "")
    if not (q and sums and ans) or int(q.group(2)) != base:
        return None
    digits = tuple(int(x) for x in q.group(1).split(" + "))
    scratch = [int(x) for x in sums.group(1).split()]
    answer = int(ans.group(1))
    if any(x >= base for x in (*digits, *scratch, answer)):
        return None
    return Problem(digits, base), [THINK] + scratch + [END_THINK, answer, EOS]
