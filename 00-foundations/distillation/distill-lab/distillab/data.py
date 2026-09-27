"""data.py — a generated set of verifiable arithmetic and logic problems with a scratchpad format; no download.

One idea: distillation data is *prompts* plus a *checker*. The prompts are what you ask the teacher; the
checker (a verifier) is what lets you keep only the teacher's right answers (rejection sampling, PRIMER §3) and
score the student afterwards (PRIMER §8). Every problem here is generated from a seed, carries its exact answer
and a canonical scratchpad — the step-by-step working a good teacher writes — and is scored by comparing the
final ``\\boxed{…}`` of a response with the answer.

Five kinds, each with a difficulty knob (1 = one step, 4 = several dependent steps):

    arith     multi-step integer arithmetic            "What is (47 * 23 - 318) * 4?"
    modsum    a sum of digits modulo 5                 "What is (3 + 1 + 4 + 0 + 2 + 4) mod 5?"   (the tiny task)
    days      weekday arithmetic                       "Today is Tuesday. What day of the week will it be in 100 days?"
    order     transitive ordering of named people      "... Who is the shortest?"
    count     letter counting                          "How many times does the letter 'r' appear in ...?"

``modsum`` at difficulty 2 is the tiny transformers' task (six base-5 digits, the running sums as the
scratchpad), so the fake teacher's completions for it convert to the tiny model's tokens (notebook 02).
The prompt suffix follows DeepSeek-R1's recommendation for math ("Please reason step by step, and put your
final answer within \\boxed{}.") — the same shape as thinking-lab's built-in eval set, re-implemented here.
``split`` keeps training and evaluation problems apart, and :func:`decontaminate` removes any training problem
whose question also appears in the eval set (a teacher's outputs for an eval question are contamination).
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NAMES = ["Ann", "Ben", "Cal", "Dee", "Eve", "Fay", "Gus", "Hal"]
WORDS = ["strawberry", "raspberry", "mirror", "carrier", "terror", "barrel", "error", "ferry", "horror", "arrow"]
KINDS = ("arith", "modsum", "days", "order", "count")
MODSUM_TERMS = {1: 3, 2: 6, 3: 9, 4: 12}
MODSUM_BASE = 5


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


def _modsum(rng, d):
    digits = [rng.randrange(MODSUM_BASE) for _ in range(MODSUM_TERMS[d])]
    return f"What is ({' + '.join(map(str, digits))}) mod {MODSUM_BASE}?", sum(digits) % MODSUM_BASE


def _days(rng, d):
    start = rng.randrange(7)
    n = rng.randint(3, 6) if d == 1 else rng.randint(8, 30) if d == 2 else rng.randint(31, 400) if d == 3 else rng.randint(401, 5000)
    return f"Today is {DAYS[start].capitalize()}. What day of the week will it be in {n} days?", DAYS[(start + n) % 7]


def _order(rng, d):
    people = rng.sample(NAMES, 2 + d)            # people[0] is the tallest ... people[-1] the shortest
    facts = [f"{x} is taller than {y}." if rng.random() < 0.5 else f"{y} is shorter than {x}."
             for x, y in zip(people, people[1:])]
    rng.shuffle(facts)
    ask_short = rng.random() < 0.5
    q = " ".join(facts) + (" Who is the shortest?" if ask_short else " Who is the tallest?")
    return q, (people[-1] if ask_short else people[0]).lower()


def _count(rng, d):
    words = rng.sample(WORDS, d)
    letter = rng.choice("re")
    text = " ".join(words)
    return f"How many times does the letter '{letter}' appear in \"{text}\"?", text.count(letter)


_GEN = {"arith": _arith, "modsum": _modsum, "days": _days, "order": _order, "count": _count}


def make_problem(kind: str, difficulty: int, rng: random.Random, idx: int = 0, split: str = "eval") -> Problem:
    q, a = _GEN[kind](rng, difficulty)
    return Problem(f"{split}-{kind}-{difficulty}-{idx}", kind, difficulty, q, str(a).lower())


def make_set(n: int = 60, seed: int = 0, kinds: tuple = KINDS, difficulties: tuple = (1, 2, 3, 4),
             split: str = "eval") -> list:
    """``n`` problems cycling through every (kind, difficulty) pair, deterministic in ``seed`` and ``split``
    (the two splits use different random streams, so they overlap only by chance — see ``decontaminate``)."""
    rng = random.Random(f"{split}:{seed}")
    pairs = [(k, d) for d in difficulties for k in kinds]
    return [make_problem(*pairs[i % len(pairs)], rng, i, split) for i in range(n)]


def decontaminate(train: list, eval_set: list) -> tuple:
    """Drop training problems whose question text appears in the eval set. Returns (kept, dropped)."""
    seen = {p.question for p in eval_set}
    kept = [p for p in train if p.question not in seen]
    return kept, [p for p in train if p.question in seen]


# --- the canonical scratchpad (what a careful teacher writes) -------------------------------------------

def steps(p: Problem) -> list:
    """The working, one line per step, ending in the answer. Deterministic: the fake teacher's backbone."""
    q = p.question
    if p.kind == "arith":
        nums = list(map(int, re.findall(r"\d+", q)))
        a, b = nums[0], nums[1]
        if p.difficulty == 1:
            return [f"{a} + {b} = {a + b}"]
        out = [f"{a} * {b} = {a * b}"]
        if p.difficulty >= 3:
            out.append(f"{a * b} - {nums[2]} = {a * b - nums[2]}")
        if p.difficulty == 4:
            out.append(f"{a * b - nums[2]} * {nums[3]} = {(a * b - nums[2]) * nums[3]}")
        return out
    if p.kind == "modsum":
        digits = list(map(int, re.search(r"\(([\d + ]+)\)", q).group(1).split(" + ")))
        sums, s = [], 0
        for x in digits:
            s = (s + x) % MODSUM_BASE
            sums.append(s)
        return [f"Running sums mod {MODSUM_BASE}: {' '.join(map(str, sums))}"]
    if p.kind == "days":
        m = re.search(r"Today is (\w+)\. What day of the week will it be in (\d+) days\?", q)
        n = int(m[2])
        return [f"{n} days is {n // 7} weeks and {n % 7} days", f"{m[1]} + {n % 7} days = {p.answer.capitalize()}"]
    if p.kind == "order":
        return [f"From tallest to shortest the facts give a chain; the {'shortest' if 'shortest' in q else 'tallest'} "
                f"is {p.answer.capitalize()}"]
    if p.kind == "count":
        text = re.search(r"\"([a-z ]+)\"", q)[1]
        letter = re.search(r"letter '(\w)'", q)[1]
        per = [f"{w}: {w.count(letter)}" for w in text.split()]
        return ["; ".join(per) + f" -> {p.answer}"]
    raise ValueError(p.kind)


def scratchpad(p: Problem) -> str:
    """The canonical completion: the steps, then the boxed answer."""
    return "\n".join(steps(p) + [f"Answer: \\boxed{{{p.answer}}}"])


# --- scoring ------------------------------------------------------------------------------------------

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_answer(text: str | None) -> str | None:
    """The last ``\\boxed{…}`` in ``text`` (else a trailing "Answer: …"), lower-cased; None if absent."""
    if not text:
        return None
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip().lower()
    m = re.findall(r"Answer:\s*([^\n]+)", text)
    return m[-1].strip().rstrip(".").lower() if m else None


def verify(p: Problem, content: str | None) -> bool:
    """The verifier: the final answer of the *content* (never the reasoning) equals the problem's answer."""
    got = extract_answer(content)
    return got is not None and got == p.answer


# --- solving a question found anywhere in a text (the fake teacher uses this to "know" the answer) ------

def find(text: str) -> Problem | None:
    """Rebuild the generated problem whose question appears in ``text`` (kind, difficulty and answer),
    or None when ``text`` holds no generated question."""
    pats = [
        ("arith", 4, r"What is \((\d+) \* (\d+) - (\d+)\) \* (\d+)\?", lambda m: (int(m[1]) * int(m[2]) - int(m[3])) * int(m[4])),
        ("arith", 3, r"What is (\d+) \* (\d+) - (\d+)\?", lambda m: int(m[1]) * int(m[2]) - int(m[3])),
        ("arith", 2, r"What is (\d+) \* (\d+)\?", lambda m: int(m[1]) * int(m[2])),
        ("arith", 1, r"What is (\d+) \+ (\d+)\?", lambda m: int(m[1]) + int(m[2])),
        ("modsum", 0, r"What is \(([\d + ]+)\) mod (\d+)\?", lambda m: sum(map(int, m[1].split(" + "))) % int(m[2])),
        ("days", 0, r"Today is (\w+)\. What day of the week will it be in (\d+) days\?",
         lambda m: DAYS[(DAYS.index(m[1].lower()) + int(m[2])) % 7]),
        ("count", 0, r"How many times does the letter '(\w)' appear in \"([a-z ]+)\"\?", lambda m: m[2].count(m[1])),
    ]
    for kind, d, rx, fn in pats:
        m = re.search(rx, text)
        if m:
            q = m.group(0)
            if kind == "modsum":
                n = len(m[1].split(" + "))
                d = min((k for k, v in MODSUM_TERMS.items() if v >= n), default=4)
            elif kind == "days":
                n = int(m[2])
                d = 1 if n <= 6 else 2 if n <= 30 else 3 if n <= 400 else 4
            elif kind == "count":
                d = len(m[2].split())
            return Problem("found", kind, d, q, str(fn(m)).lower())
    m = re.search(r"((?:\w+ is (?:taller|shorter) than \w+\. )+Who is the (?:shortest|tallest)\?)", text)
    if m:
        q = m.group(1)
        pairs = [(a, b) for a, b in re.findall(r"(\w+) is taller than (\w+)\.", q)]
        pairs += [(b, a) for a, b in re.findall(r"(\w+) is shorter than (\w+)\.", q)]
        people = {x for pr in pairs for x in pr}
        pick = people - ({a for a, _ in pairs} if "shortest" in q else {b for _, b in pairs})
        if len(pick) == 1:
            return Problem("found", "order", max(1, len(pairs) - 1), q, pick.pop().lower())
    return None
