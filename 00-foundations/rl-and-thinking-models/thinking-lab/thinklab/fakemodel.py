"""fakemodel.py — a *simulated* thinking model: how long it thinks, and whether it is then right.

One idea: for a serving and evaluation lab the model's words do not matter, its **behaviour** does —
how many reasoning tokens a question makes it produce (a heavy-tailed distribution that grows with
difficulty), and how its accuracy depends on how much of that thinking it was allowed to do. This
module is that behaviour and nothing else:

    reasoning length   L ~ LogNormal(median m_d, σ)           m_d grows with difficulty d
    accuracy           P(correct | L) = a_max − (a_max − a_0) · exp(−L / τ_d)

— with thinking off L = 0 and accuracy is ``a_0`` (the no-thinking accuracy); with a budget B the
model answers after min(L, B) tokens of thinking (budget forcing), so accuracy follows the curve at
min(L, B); with a ``max_tokens`` cut inside the thinking there is no answer at all. It is the lab's
counterpart of the core's ``ThinkTask`` (P(correct | L) = 1 − e0·(1 − q)^L), with a ceiling and a
difficulty knob. Wrong answers are drawn from a few *plausible distractors* with unequal weights, so
majority voting helps when a_L > the top distractor's share and hurts when it is not — as it does
for real models.

Numbers here are **illustrative** parameters for a small (0.6–1.7B-class) thinking model, not
measurements of Qwen3; the T1 notebooks replace them with a real model's outputs. The reasoning
*text* is filler made of toy tokens (one word = one token, as ``thinklab.templates.tokens`` counts).
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

from .thinking import evalset

EARLY_STOP = "Considering the limited time by the user, I have to give the solution based on the thinking directly now."

_FILLER = ("let me work through this carefully first compute the next part then check the result again "
           "so the value becomes this wait I should double check that step it seems right now combine "
           "these pieces and verify the total before answering").split()


@dataclass(frozen=True)
class ModelCard:
    """Behavioural parameters of a simulated thinking model, per difficulty 1-4 (illustrative)."""
    name: str = "sim-thinker-0.6b"
    a0: tuple = (0.70, 0.35, 0.15, 0.05)                 # accuracy with thinking off
    a_max: tuple = (0.97, 0.90, 0.80, 0.62)              # accuracy with unlimited thinking
    median_think: tuple = (180, 420, 900, 1700)          # median reasoning tokens
    tau: tuple = (90, 220, 480, 950)                     # tokens to close 63% of the gap
    sigma: float = 0.75                                  # log-normal spread: p99/p50 = e^(2.33σ) ≈ 5.7
    answer_median: int = 40                              # answer (content) tokens after thinking
    answer_nothink_median: int = 90                      # answers are longer without thinking (they explain inline)
    max_think: int = 30000
    distractor_weights: tuple = (0.55, 0.3, 0.15)

    def accuracy(self, difficulty: int, think_tokens: float, thinking: bool = True) -> float:
        d = max(1, min(4, difficulty)) - 1
        if not thinking:
            return self.a0[d]
        return self.a_max[d] - (self.a_max[d] - self.a0[d]) * math.exp(-think_tokens / self.tau[d])

    def expected_accuracy(self, difficulty: int, budget: int | None = None, samples: int = 4000, seed: int = 0) -> float:
        """E[P(correct)] over the thinking-length distribution, with a thinking budget (None = unlimited)."""
        rng = random.Random(seed)
        d = max(1, min(4, difficulty)) - 1
        tot = 0.0
        for _ in range(samples):
            L = min(self.max_think, rng.lognormvariate(math.log(self.median_think[d]), self.sigma))
            tot += self.accuracy(difficulty, L if budget is None else min(L, budget))
        return tot / samples


SMALL = ModelCard()
LARGER = ModelCard(name="sim-thinker-4b", a0=(0.85, 0.55, 0.30, 0.12), a_max=(0.99, 0.96, 0.90, 0.78),
                   median_think=(150, 350, 750, 1400), tau=(70, 170, 380, 760))
CARDS = {c.name: c for c in (SMALL, LARGER)}


@dataclass
class Sample:
    """One simulated generation, already cut to what the server will emit."""
    reasoning_tokens: list = field(default_factory=list)   # toy tokens (words)
    content_tokens: list = field(default_factory=list)
    correct: bool | None = None
    answer: str | None = None
    finish_reason: str = "stop"
    planned_think: int = 0                                 # what the model would have thought, unconstrained
    forced_stop: bool = False                              # the budget ended the thinking

    @property
    def output_tokens(self) -> int:
        return len(self.reasoning_tokens) + len(self.content_tokens)


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def _distractors(answer: str, rng: random.Random) -> list:
    if answer.lstrip("-").isdigit():
        a = int(answer)
        cands = [a + 1, a - 1, a + 10, a - 10, a * 2, a + 2]
        rng.shuffle(cands)
        return [str(c) for c in cands[:3]]
    pool = [w for w in evalset.DAYS + [n.lower() for n in evalset.NAMES] if w != answer]
    return rng.sample(pool, 3)


def _filler(n: int, rng: random.Random) -> list:
    start = rng.randrange(len(_FILLER))
    return [_FILLER[(start + i) % len(_FILLER)] for i in range(n)]


def generate(card: ModelCard, question: str, *, thinking: bool = True, budget: int | None = None,
             max_tokens: int | None = None, seed=0, sample_index: int = 0, prefilled_think: int | None = None) -> Sample:
    """Simulate one completion for ``question`` (any text; generated eval questions get checked answers).

    ``budget``: vLLM-style ``thinking_token_budget`` (None or −1 = unlimited): thinking stops at B tokens
    and the model answers from there. ``max_tokens``: a hard cut on reasoning + content — inside the
    thinking it leaves ``content = None`` and ``finish_reason = "length"``. ``prefilled_think``: the
    thinking was supplied by the caller (Qwen's two-call budget recipe); only the answer is generated."""
    rng = _rng(card.name, question, seed, sample_index)
    truth = evalset.solve(question)
    d = evalset.difficulty_of(question) if truth is not None else 2
    di = d - 1
    planned = 0
    if thinking and prefilled_think is None:
        planned = int(min(card.max_think, max(8, rng.lognormvariate(math.log(card.median_think[di]), card.sigma))))
    think = planned if (budget is None or budget < 0) else min(planned, budget)
    used_think = prefilled_think if prefilled_think is not None else think
    p = card.accuracy(d, used_think, thinking or prefilled_think is not None)
    correct = rng.random() < p
    if truth is None:
        answer = None
        body = _filler(max(4, int(rng.lognormvariate(math.log(card.answer_median), 0.5))), rng)
        content = body
    else:
        if correct:
            answer = truth
        else:
            answer = rng.choices(_distractors(truth, rng), weights=card.distractor_weights)[0]
        median = card.answer_median if (thinking or prefilled_think is not None) else card.answer_nothink_median
        body = _filler(max(3, int(rng.lognormvariate(math.log(median), 0.5)) - 6), rng)
        content = body + ["so", "the", "answer", "is", f"\\boxed{{{answer}}}"]
    reasoning = _filler(think, rng)
    forced = budget is not None and budget >= 0 and planned > budget
    if forced:
        reasoning = reasoning[: max(0, budget)]
    s = Sample(reasoning, content, (answer == truth) if truth is not None else None, answer, "stop", planned, forced)
    if max_tokens is not None and s.output_tokens > max_tokens:
        if max_tokens <= len(s.reasoning_tokens):
            s.reasoning_tokens, s.content_tokens = s.reasoning_tokens[:max_tokens], []
            s.correct, s.answer = (False if truth is not None else None), None
        else:
            s.content_tokens = s.content_tokens[: max_tokens - len(s.reasoning_tokens)]
            s.answer = evalset.extract_answer(" ".join(s.content_tokens)) if s.content_tokens else None
            s.correct = (s.answer == truth) if truth is not None else None
        s.finish_reason = "length"
    return s
