"""evalset.py — a generated set of verifiable problems (arithmetic and logic), no download.

One idea: to measure what thinking buys you need problems whose answers a *program* can check —
the same property RL with verifiable rewards depends on (PRIMER §4). Every problem here is
generated from a seed, carries its exact answer, and is scored by comparing the final
``\\boxed{…}`` in the model's *content* (never its reasoning) with that answer.

Five kinds, each with a difficulty knob (1 = one step, 4 = several dependent steps):

    arith      multi-step integer arithmetic            "What is 47 * 23 - 318?"
    digitsum   last digit of a long sum                 "What is the last digit of 7+3+9+...?"
    days       weekday arithmetic                        "Today is Tuesday. What day is it in 100 days?"
    order      transitive ordering of named people       "... Who is the shortest?"
    count      letter counting                           "How many times does 'r' appear in ...?"

The prompt suffix follows DeepSeek-R1's usage recommendation for math ("Please reason step by
step, and put your final answer within \\boxed{}."; no system prompt). The set is small on
purpose — accuracy on 60 problems near 50% carries a ±12-point 95% interval, which the notebooks report
(``thinklab.thinking.ttc.wilson_interval``; the same statistic as the 07 agent lab's eval gates).
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from ..parsers import extract_answer

SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NAMES = ["Ann", "Ben", "Cal", "Dee", "Eve", "Fay", "Gus", "Hal"]
WORDS = ["strawberry", "raspberry", "mirror", "carrier", "terror", "barrel", "error", "ferry", "horror", "arrow"]
KINDS = ("arith", "digitsum", "days", "order", "count")


@dataclass(frozen=True)
class Problem:
    id: str
    kind: str
    difficulty: int
    question: str
    answer: str

    @property
    def prompt(self) -> str:
        return f"{self.question} {SUFFIX}"

    def messages(self) -> list:
        return [{"role": "user", "content": self.prompt}]

    def to_dict(self) -> dict:
        return asdict(self)


def _arith(rng, d):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    if d == 1:
        return f"What is {a} + {b}?", a + b
    if d == 2:
        return f"What is {a} * {b}?", a * b
    c = rng.randint(100, 999)
    if d == 3:
        return f"What is {a} * {b} - {c}?", a * b - c
    e = rng.randint(3, 9)
    return f"What is ({a} * {b} - {c}) * {e}?", (a * b - c) * e


def _digitsum(rng, d):
    digits = [rng.randint(1, 9) for _ in range(4 + 4 * d)]
    return f"What is the last digit of the sum {'+'.join(map(str, digits))}?", sum(digits) % 10


def _days(rng, d):
    start = rng.randrange(7)
    n = rng.randint(3, 6) if d == 1 else rng.randint(8, 30) if d == 2 else rng.randint(31, 400) if d == 3 else rng.randint(401, 5000)
    return f"Today is {DAYS[start].capitalize()}. What day of the week will it be in {n} days?", DAYS[(start + n) % 7]


def _order(rng, d):
    people = rng.sample(NAMES, 2 + d)            # people[0] is the tallest ... people[-1] the shortest
    facts = []
    for x, y in zip(people, people[1:]):
        facts.append(f"{x} is taller than {y}." if rng.random() < 0.5 else f"{y} is shorter than {x}.")
    rng.shuffle(facts)
    ask_short = rng.random() < 0.5
    q = " ".join(facts) + (" Who is the shortest?" if ask_short else " Who is the tallest?")
    return q, (people[-1] if ask_short else people[0]).lower()


def _count(rng, d):
    words = rng.sample(WORDS, d)
    letter = rng.choice("re")
    text = " ".join(words)
    return f"How many times does the letter '{letter}' appear in \"{text}\"?", text.count(letter)


_GEN = {"arith": _arith, "digitsum": _digitsum, "days": _days, "order": _order, "count": _count}


def make_problem(kind: str, difficulty: int, rng: random.Random, idx: int = 0) -> Problem:
    q, a = _GEN[kind](rng, difficulty)
    return Problem(f"{kind}-{difficulty}-{idx}", kind, difficulty, q, str(a).lower())


def make_evalset(n: int = 60, seed: int = 0, kinds: tuple = KINDS, difficulties: tuple = (1, 2, 3, 4)) -> list:
    """``n`` problems cycling through every (kind, difficulty) pair, deterministic in ``seed``."""
    rng = random.Random(seed)
    pairs = [(k, d) for d in difficulties for k in kinds]
    return [make_problem(*pairs[i % len(pairs)], rng, i) for i in range(n)]


def verify(problem: Problem, content: str | None) -> bool:
    """The verifier: the final ``\\boxed{}`` (or "Answer: …") of the *content* equals the answer."""
    got = extract_answer(content)
    return got is not None and got == problem.answer


# --- a solver for generated questions (the fake server uses it to "know" the right answer) ------
_SOLVERS = [
    (re.compile(r"What is \((\d+) \* (\d+) - (\d+)\) \* (\d+)\?"), lambda m: (int(m[1]) * int(m[2]) - int(m[3])) * int(m[4])),
    (re.compile(r"What is (\d+) \* (\d+) - (\d+)\?"), lambda m: int(m[1]) * int(m[2]) - int(m[3])),
    (re.compile(r"What is (\d+) \* (\d+)\?"), lambda m: int(m[1]) * int(m[2])),
    (re.compile(r"What is (\d+) \+ (\d+)\?"), lambda m: int(m[1]) + int(m[2])),
    (re.compile(r"last digit of the sum ([\d+]+)\?"), lambda m: sum(map(int, m[1].split("+"))) % 10),
    (re.compile(r"Today is (\w+)\. What day of the week will it be in (\d+) days\?"),
     lambda m: DAYS[(DAYS.index(m[1].lower()) + int(m[2])) % 7]),
    (re.compile(r"How many times does the letter '(\w)' appear in \"([a-z ]+)\"\?"), lambda m: m[2].count(m[1])),
]


def _solve_order(text: str):
    pairs = [(a, b) for a, b in re.findall(r"(\w+) is taller than (\w+)\.", text)]
    pairs += [(b, a) for a, b in re.findall(r"(\w+) is shorter than (\w+)\.", text)]
    if not pairs:
        return None
    people = {p for pr in pairs for p in pr}
    taller = {a for a, _ in pairs}
    shorter = {b for _, b in pairs}
    if "Who is the shortest?" in text:
        (who,) = people - taller or {None}
    else:
        (who,) = people - shorter or {None}
    return who


def solve(text: str) -> str | None:
    """The exact answer of a generated question found anywhere in ``text``; ``None`` if unknown."""
    for rx, fn in _SOLVERS:
        m = rx.search(text)
        if m:
            return str(fn(m)).lower()
    who = _solve_order(text)
    return who.lower() if who else None


def difficulty_of(text: str) -> int:
    """A rough difficulty 1-4 for any generated question (used by the simulated model)."""
    if "last digit of the sum" in text:
        m = re.search(r"sum ([\d+]+)\?", text)
        return max(1, min(4, (len(m[1].split("+")) - 4) // 4)) if m else 2
    if re.search(r"What is \(", text):
        return 4
    if re.search(r"What is \d+ \+ \d+\?", text):
        return 1
    if re.search(r"\* \d+ - \d+\?", text):
        return 3
    if "*" in text:
        return 2
    if "days?" in text:
        n = int(re.search(r"in (\d+) days", text)[1])
        return 1 if n <= 7 else 2 if n <= 30 else 3 if n <= 400 else 4
    if "taller" in text or "shorter" in text:
        return max(1, min(4, len(re.findall(r"\.", text)) - 1))
    if "appear in" in text:
        m = re.search(r"\"([a-z ]+)\"", text)
        return max(1, min(4, len(m[1].split()))) if m else 2
    return 2
