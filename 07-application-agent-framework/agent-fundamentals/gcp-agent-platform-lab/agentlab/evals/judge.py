"""LLM-as-judge, with the calibration that makes it trustworthy (Primer §4.1).

A judge scores an answer 1–5 against a rubric. Judges are cheap and scale;
they are also biased, and the biases are known:

* **position bias** – in a pairwise comparison the judge prefers whichever
  answer it read first (or last). ``pairwise`` asks twice with the order
  swapped and flags a verdict that follows the position, not the answer.
* **verbosity bias** – longer, more confident answers score higher regardless
  of correctness. Rubrics must reward the specific facts, not the word count.
* **self-preference** – a model rates its own outputs (and its own style)
  above others'. Judge with a different model family than the one under test.

None of this is fatal if the judge is *calibrated*: ``calibrate`` compares
judge scores with human labels on a sample and reports agreement, Cohen's
kappa (chance-corrected) and mean absolute error. A judge with kappa below
~0.4 is not measuring what the humans measured; fix the rubric before trusting
the number at scale.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

# ------------------------------------------------------------------- scores
@dataclass
class JudgeScore:
    score: int                  # 1..5
    reason: str
    raw: str = ""


class Judge(Protocol):
    async def score(self, question: str, answer: str, rubric: str) -> JudgeScore: ...


class PairwiseJudge(Protocol):
    async def compare(self, question: str, first: str, second: str, rubric: str) -> str: ...   # "A" | "B" | "tie"


class JudgeParseError(ValueError):
    """The judge model replied without a usable verdict."""


_SCORE_RE = re.compile(r"(?<!\d)([1-5])(?!\d)")          # a lone digit 1–5; never the "1" inside "10"
_CHOICE_RE = re.compile(r"\b(A|B|[Tt][Ii][Ee])\b")        # upper-case A/B only, so the article "a" cannot vote


def parse_score(text: str, lo: int = 1, hi: int = 5) -> int | None:
    """First standalone digit in ``[lo, hi]``; ``None`` when there is none."""
    for m in _SCORE_RE.finditer(text or ""):
        value = int(m.group(1))
        if lo <= value <= hi:
            return value
    return None


def parse_choice(text: str) -> str | None:
    """First standalone ``A``, ``B`` or ``tie``; ``None`` when there is none."""
    m = _CHOICE_RE.search(text or "")
    if m is None:
        return None
    return m.group(1) if m.group(1) in ("A", "B") else "tie"


# ------------------------------------------------------------- rubric judge
SCORE_PROMPT = """You are grading an assistant's answer. Score it from 1 (unacceptable) to 5 (excellent)
against the rubric. Reply with the digit first, then one sentence of justification.

### Question
{question}

### Answer
{answer}

### Rubric
{rubric}
"""

COMPARE_PROMPT = """You are comparing two assistant answers to the same question against the rubric.
Reply with exactly one of: A, B, tie — then one sentence of justification.

### Question
{question}

### Answer A
{first}

### Answer B
{second}

### Rubric
{rubric}
"""


class RubricJudge:
    """Prompts an LLM with a rubric and parses the first digit 1–5 (or A/B for comparisons)."""

    def __init__(self, llm: Any, score_prompt: str = SCORE_PROMPT, compare_prompt: str = COMPARE_PROMPT):
        self.llm = llm
        self.score_prompt = score_prompt
        self.compare_prompt = compare_prompt

    async def score(self, question: str, answer: str, rubric: str) -> JudgeScore:
        prompt = self.score_prompt.format(question=question, answer=answer, rubric=rubric)
        resp = await self.llm.generate([{"role": "user", "content": prompt}])
        value = parse_score(resp.text or "")
        if value is None:
            raise JudgeParseError(f"no score 1-5 in judge reply: {resp.text!r}")
        return JudgeScore(score=value, reason=(resp.text or "").strip(), raw=resp.text or "")

    async def compare(self, question: str, first: str, second: str, rubric: str) -> str:
        prompt = self.compare_prompt.format(question=question, first=first, second=second, rubric=rubric)
        resp = await self.llm.generate([{"role": "user", "content": prompt}])
        choice = parse_choice(resp.text or "")
        if choice is None:
            raise JudgeParseError(f"no A/B/tie in judge reply: {resp.text!r}")
        return choice


class KeywordJudge:
    """Deterministic baseline: score = 1 + 4 × (share of rubric keywords present in the answer)."""

    def __init__(self, rubric_keywords: Sequence[str]):
        if not rubric_keywords:
            raise ValueError("KeywordJudge needs at least one keyword")
        self.keywords = [k.lower() for k in rubric_keywords]

    def _hits(self, answer: str) -> list[str]:
        low = (answer or "").lower()
        return [k for k in self.keywords if k in low]

    async def score(self, question: str, answer: str, rubric: str = "") -> JudgeScore:
        hits = self._hits(answer)
        value = 1 + round(4 * len(hits) / len(self.keywords))
        return JudgeScore(score=value, reason=f"keywords present: {hits or 'none'}")

    async def compare(self, question: str, first: str, second: str, rubric: str = "") -> str:
        a, b = len(self._hits(first)), len(self._hits(second))
        return "A" if a > b else "B" if b > a else "tie"


# ---------------------------------------------------------------- pairwise
@dataclass
class PairwiseResult:
    preferred: str | None        # "a" | "b" | "tie" | None when the verdict is position-driven
    first_pass: str              # verdict with a shown first
    second_pass: str             # verdict with b shown first
    position_bias: bool

    def __str__(self) -> str:
        return f"preferred={self.preferred} (a-first: {self.first_pass}, b-first: {self.second_pass}){' POSITION BIAS' if self.position_bias else ''}"


async def pairwise(judge: PairwiseJudge, question: str, a: str, b: str, rubric: str = "") -> PairwiseResult:
    """Compare ``a`` and ``b`` twice with the order swapped; a verdict that flips with position is not a verdict."""
    first_pass = await judge.compare(question, a, b, rubric)     # A means a
    second_pass = await judge.compare(question, b, a, rubric)    # A means b
    if first_pass == "tie" and second_pass == "tie":
        return PairwiseResult("tie", first_pass, second_pass, position_bias=False)
    if first_pass == "A" and second_pass == "B":
        return PairwiseResult("a", first_pass, second_pass, position_bias=False)
    if first_pass == "B" and second_pass == "A":
        return PairwiseResult("b", first_pass, second_pass, position_bias=False)
    # Same letter both times (or one tie): the judge picked a slot, not an answer.
    return PairwiseResult(None, first_pass, second_pass, position_bias=True)


# ------------------------------------------------------------- calibration
def cohen_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int] | None = None, weights: str | None = None) -> float:
    """Cohen's kappa between two raters; ``weights="linear"`` penalises disagreements by distance.

    kappa = 1 − Σ w·O / Σ w·E, with observed counts O and chance-expected counts E
    (row marginal × column marginal / n). Unweighted, w is 0 on the diagonal and 1
    elsewhere, which reduces to (p_o − p_e) / (1 − p_e).
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty label sequences")
    cats = sorted(set(a) | set(b)) if labels is None else list(labels)
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[idx[x]][idx[y]] += 1
    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    span = (cats[-1] - cats[0]) or 1
    weight = (lambda i, j: abs(cats[i] - cats[j]) / span) if weights == "linear" else (lambda i, j: float(i != j))
    disagreement = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    expected = sum(weight(i, j) * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    if expected == 0:                     # every label identical: agreement is trivially perfect or undefined
        return 1.0 if disagreement == 0 else 0.0
    return 1.0 - disagreement / expected


@dataclass
class Calibration:
    n: int
    exact_agreement: float
    within_one: float
    kappa: float
    weighted_kappa: float
    mae: float

    def __str__(self) -> str:
        return (f"n={self.n} exact={self.exact_agreement:.2f} within±1={self.within_one:.2f} "
                f"kappa={self.kappa:.3f} linear-kappa={self.weighted_kappa:.3f} MAE={self.mae:.2f}")


def calibrate(judge_scores: Sequence[int], human_scores: Sequence[int]) -> Calibration:
    """How well the judge tracks human labels; kappa is the number to quote — raw agreement flatters.

    Two raters who both say "4" to everything agree 100% and have kappa 0: the
    agreement was guaranteed by the marginals, not by reading the answers.
    """
    if len(judge_scores) != len(human_scores) or not judge_scores:
        raise ValueError("need equally many judge and human scores")
    n = len(judge_scores)
    diffs = [abs(j - h) for j, h in zip(judge_scores, human_scores)]
    return Calibration(
        n=n,
        exact_agreement=sum(d == 0 for d in diffs) / n,
        within_one=sum(d <= 1 for d in diffs) / n,
        kappa=cohen_kappa(judge_scores, human_scores),
        weighted_kappa=cohen_kappa(judge_scores, human_scores, weights="linear"),
        mae=sum(diffs) / n,
    )
