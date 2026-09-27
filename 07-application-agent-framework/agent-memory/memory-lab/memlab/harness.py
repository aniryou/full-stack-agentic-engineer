"""harness.py — measure memory the way the benchmarks do: plant facts across sessions, ask later, grade.

The one idea (PRIMER §4): a memory system is only as good as the answers it enables weeks later, so
the test is a *haystack* — sessions of chatter with facts planted in some of them, a value that changes
halfway (a knowledge update), a preference never phrased as a question — followed by questions in new
sessions. The question types copy the *shapes* of LongMemEval (Wu et al., ICLR 2025: single-session-user,
-preference, multi-session, temporal-reasoning, knowledge-update, abstention as an ``_abs`` id suffix) and
LoCoMo (Maharana et al., ACL 2024: category 5 adversarial = unanswerable). Nothing is downloaded — LoCoMo is
CC BY-NC 4.0 — the generator is seeded and the data is ours.

Every question is graded ``correct``, ``stale`` (the value before the update), ``abstained`` (said "I
don't know" to an answerable question), ``hallucinated`` (answered an unanswerable one) or ``wrong``.
Reported with Wilson intervals (restated from 07.2's ``agentlab.evals.gate.wilson_interval``), plus what
each answer cost: memory tokens injected, input tokens, model calls. A **paraphrase subset** asks the
same facts in words the hashing embedder cannot match ("Which firm pays my salary?"): the gap it shows
is what a real embedder (T1) has to close.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field

from .agent import MemoryAgent
from .consolidate import ConsolidationJob
from .extract import day
from .llm import ScriptedModel
from .memory import LocalMemory
from .records import MemoryRecord, count_tokens
from .store.sqlite import SQLiteMemoryStore

CATEGORIES = ("extraction", "preference", "multi-session", "temporal", "knowledge-update", "abstention", "adversarial")
# The benchmark shape each category copies (names from their code; facts sheet §3, §4).
SHAPES = {
    "extraction": ("single-session-user", "LoCoMo 4 single-hop"),
    "preference": ("single-session-preference", "-"),
    "multi-session": ("multi-session", "LoCoMo 1 multi-hop"),
    "temporal": ("temporal-reasoning", "LoCoMo 2 temporal"),
    "knowledge-update": ("knowledge-update", "-"),
    "abstention": ("question_id suffix _abs", "-"),
    "adversarial": ("-", "LoCoMo 5 adversarial"),
}
MODES = ("none", "full_history", "tools", "implicit", "pinned")
T0 = 1_788_220_800.0                        # 2026-09-01T00:00:00Z

CITIES = ["Lisbon", "Porto", "Madrid", "Berlin", "Oslo", "Dublin", "Vienna", "Prague", "Warsaw", "Athens"]
TRIPS = ["Rome", "Kyoto", "Lima", "Cairo", "Quebec", "Hanoi", "Nairobi", "Sydney"]
ORGS = ["Globex", "Initech", "Umbrella", "Hooli", "Vandelay", "Stark"]
NAMES = ["Ana", "Bruno", "Chen", "Dara", "Eli", "Farah", "Goran", "Hana"]
PETS = ["Rex", "Max", "Luna", "Milo", "Nala", "Otis"]
DIETS = ["vegetarian", "vegan", "pescatarian"]
DRINKS = ["green tea", "espresso", "lemonade", "cold brew"]
FILLER = ["The weather was grey this morning.", "I watched a documentary about whales.",
          "The train was late again.", "I finished a long report today.", "Lunch was a quick sandwich.",
          "I spent the evening reading.", "The meeting ran over by an hour.", "I tidied the kitchen.",
          "My phone battery died twice.", "I tried a new podcast."]


@dataclass
class Question:
    id: str
    user: str
    category: str
    text: str
    answer: str | None                 # None: the right answer is to abstain
    stale: str | None = None           # the pre-update value (knowledge updates)
    evidence: list[str] = field(default_factory=list)   # values that must be in context to answer
    paraphrase: bool = False
    asked_at: float = 0.0


@dataclass
class Session:
    id: str
    date: float
    turns: list[str]


@dataclass
class Dataset:
    seed: int
    haystacks: dict[str, list[Session]]
    questions: list[Question]

    def counts(self) -> dict[str, int]:
        return {c: sum(q.category == c for q in self.questions) for c in CATEGORIES}


def generate(seed: int = 7, users: int = 6, sessions: int = 8, days_apart: float = 3.0) -> Dataset:
    """A seeded haystack per user and ~14 questions each, asked a week after the last session."""
    rng = random.Random(seed)
    haystacks, questions = {}, []
    for u in range(users):
        user = f"u{u + 1}"
        city1, city2 = rng.sample(CITIES, 2)
        org, boss, pet, drink = rng.choice(ORGS), rng.choice(NAMES), rng.choice(PETS), rng.choice(DRINKS)
        diet, animal = rng.choice(DIETS), rng.choice(["dog", "cat"])
        trips = rng.sample(TRIPS, rng.choice([2, 3]))
        planted = [f"I live in {city1}.", f"I work at {org}.", f"my manager is {boss}.",
                   f"my {animal} is called {pet}.", f"I'm {diet}, by the way.", f"my favourite drink is {drink}."]
        planted += [f"I took a trip to {t}." for t in trips]
        rng.shuffle(planted)
        slots_for = [[] for _ in range(sessions)]
        for i, p in enumerate(planted):                      # spread over all but the last two sessions
            slots_for[i % (sessions - 2)].append(p)
        update_session = sessions - 2
        slots_for[update_session].append(f"I moved to {city2}.")
        hay = []
        for s in range(sessions):
            turns = [rng.choice(FILLER) for _ in range(rng.randint(2, 4))]
            for p in slots_for[s]:
                turns.insert(rng.randint(0, len(turns)), p[0].upper() + p[1:])
            hay.append(Session(f"{user}-s{s + 1}", T0 + s * days_apart * 86400, turns))
        haystacks[user] = hay
        asked = hay[-1].date + 7 * 86400
        other = "cat" if animal == "dog" else "dog"
        qs = [
            ("extraction", "What is my employer?", org, None, [org], False),
            ("extraction", "Which firm pays my salary?", org, None, [org], True),
            ("extraction", "Who is my manager?", boss, None, [boss], False),
            ("extraction", "Who is my boss these days?", boss, None, [boss], True),
            ("extraction", "What is my favourite drink?", drink, None, [drink], False),
            ("extraction", f"What is my {animal}'s name?", pet, None, [pet], False),
            ("preference", "Can you suggest a dinner recipe?", diet, None, [diet], False),
            ("multi-session", "How many trips have I told you about?", str(len(trips)), None, trips, False),
            ("temporal", f"When did I move to {city2}?", day(hay[update_session].date), None, [city2], False),
            ("knowledge-update", "Which city is my home city?", city2, city1, [city2], False),
            ("knowledge-update", "Where am I based these days?", city2, city1, [city2], True),
            ("abstention", "Do I have any allergies?", None, None, [], False),
            ("abstention", "What is my partner's name?", None, None, [], False),
            ("adversarial", f"What is my {other}'s name?", None, None, [], False),
        ]
        for i, (cat, text, ans, stale, ev, para) in enumerate(qs):
            qid = f"{user}-q{i + 1:02d}" + ("_abs" if cat == "abstention" else "")
            questions.append(Question(qid, user, cat, text, ans, stale, ev, para, asked))
    return Dataset(seed, haystacks, questions)


# ------------------------------------------------------------------------------------------ grading
def grade(q: Question, answer: str) -> str:
    a = answer.lower()
    abstained = "don't know" in a or "not mentioned" in a or "no information" in a
    if q.answer is None:
        return "correct" if abstained else "hallucinated"
    if q.answer.lower() in a:
        return "correct"
    if q.stale and q.stale.lower() in a:
        return "stale"
    return "abstained" if abstained else "wrong"


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """``agentlab.evals.gate.wilson_interval``, restated (that module needs pydantic): the score interval
    for a pass rate. (45, 50) -> (0.7864, 0.9565); (0, 20) -> (0.0, 0.1611)."""
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Row:
    qid: str
    category: str
    paraphrase: bool
    verdict: str
    answer: str
    in_context: bool | None          # the evidence values were all in the prompt (None: nothing to find)
    memory_tokens: int
    input_tokens: int
    model_calls: int


@dataclass
class RunResult:
    mode: str
    budget_tokens: int
    rows: list[Row]
    label: str = ""

    def rate(self, verdict: str = "correct", category: str | None = None, paraphrase: bool | None = None) -> tuple[int, int]:
        rows = [r for r in self.rows if (category is None or r.category == category)
                and (paraphrase is None or r.paraphrase == paraphrase)]
        return sum(r.verdict == verdict for r in rows), len(rows)

    @property
    def accuracy(self) -> float:
        k, n = self.rate()
        return k / n if n else 0.0

    @property
    def recall(self) -> float:
        rows = [r for r in self.rows if r.in_context is not None]
        return sum(r.in_context for r in rows) / len(rows) if rows else 0.0

    def mean(self, attr: str) -> float:
        return sum(getattr(r, attr) for r in self.rows) / len(self.rows) if self.rows else 0.0

    def table(self) -> str:
        lines = [f"mode={self.mode} budget={self.budget_tokens} {self.label}",
                 f"  {'category':17s} {'correct':>9s} {'95% Wilson':>15s}  stale halluc  abstained"]
        for c in CATEGORIES:
            k, n = self.rate("correct", c)
            if not n:
                continue
            lo, hi = wilson_interval(k, n)
            lines.append(f"  {c:17s} {k:4d}/{n:<4d} [{lo:5.2f}, {hi:5.2f}]  {self.rate('stale', c)[0]:5d} "
                         f"{self.rate('hallucinated', c)[0]:6d} {self.rate('abstained', c)[0]:10d}")
        k, n = self.rate()
        lo, hi = wilson_interval(k, n)
        kp, np_ = self.rate("correct", paraphrase=True)
        lines.append(f"  {'all':17s} {k:4d}/{n:<4d} [{lo:5.2f}, {hi:5.2f}]   paraphrase subset {kp}/{np_}")
        lines.append(f"  evidence in context {self.recall:.1%}; per question: memory tokens {self.mean('memory_tokens'):.0f}, "
                     f"input tokens {self.mean('input_tokens'):.0f}, model calls {self.mean('model_calls'):.2f}")
        return "\n".join(lines)


class _Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        self.t += 1.0
        return self.t


def build_memory(ds: Dataset, *, write: str = "facts", mode: str = "implicit", embedder=None,
                 consolidate: bool = False, tenant: str = "acme", path: str = ":memory:"):
    """Play every haystack into one store (all users share it: isolation is the partition, not the file).
    ``write="facts"``: extraction after each turn through the write policy (or, in ``tools`` mode, whatever
    the model chose to ``remember``). ``write="episodes"``: each user turn stored raw as an episode;
    ``consolidate=True`` then runs the consolidation job over the whole window."""
    clock = _Clock(T0)
    store = SQLiteMemoryStore(path, embedder, clock=clock)
    model = ScriptedModel()
    for user, sessions in ds.haystacks.items():
        mem = LocalMemory(store, tenant, user, clock=clock)
        agent = MemoryAgent(model, mem, mode="tools" if mode == "tools" else "implicit")
        for s in sessions:
            clock.t = s.date
            agent.start_session(s.id)
            history: list[dict] = []
            for text in s.turns:
                if write == "episodes":
                    store.add(MemoryRecord(tenant, user, f"On {day(s.date)} the user said: {text}", kind="episodic",
                                           session=s.id, created_at=clock(), importance=5,
                                           provenance=[f"{s.id}#{len(history) // 2 + 1}"]))
                elif mode == "tools":
                    res = agent.turn(text, history)
                    history += res.history_entries(text)
                else:
                    agent.turn_index += 1
                    agent.write_facts(text)
        if write == "episodes" and consolidate:
            end = sessions[-1].date + 86400
            clock.t = end
            ConsolidationJob(store, clock=clock).run(tenant, user, T0, end)
    return store, clock


def transcript(sessions: list[Session]) -> str:
    return "\n".join(f"[session {day(s.date)}]\n" + "\n".join(f"user: {t}" for t in s.turns) for s in sessions)


def run(ds: Dataset, mode: str = "implicit", *, budget_tokens: int = 128, k: int = 5, write: str = "facts",
        consolidate: bool = False, embedder=None, model=None, store=None, label: str = "",
        kinds: tuple[str, ...] | None = None, profile_items: int = 6) -> RunResult:
    """Ask every question in a fresh session and grade it."""
    if mode not in MODES:
        raise ValueError(f"mode is one of {MODES}")
    model = model or ScriptedModel()
    rows = []
    if mode in ("none", "full_history"):
        for q in ds.questions:
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            if mode == "full_history":
                messages.append({"role": "system", "content": transcript(ds.haystacks[q.user])})
            messages.append({"role": "user", "content": q.text})
            resp = model.generate(messages, None)
            ctx = "\n".join(m["content"] for m in messages)
            rows.append(Row(q.id, q.category, q.paraphrase, grade(q, resp.text or ""), resp.text or "",
                            all(v.lower() in ctx.lower() for v in q.evidence) if q.evidence else None,
                            count_tokens(messages[1]["content"]) if mode == "full_history" else 0,
                            resp.usage.get("prompt_tokens", 0), 1))
        return RunResult(mode, budget_tokens, rows, label)
    if store is None:
        store, clock = build_memory(ds, write=write, mode=mode, embedder=embedder, consolidate=consolidate)
    else:
        clock = store.clock
    for q in ds.questions:
        if hasattr(clock, "t"):
            clock.t = q.asked_at
        mem = LocalMemory(store, "acme", q.user, clock=clock)
        agent = MemoryAgent(model, mem, mode=mode, k=k, budget_tokens=budget_tokens, write_after_turn=False,
                            kinds=kinds, profile_items=profile_items)
        agent.start_session(f"ask-{q.id}")
        res = agent.turn(q.text)
        ctx = "\n".join(m.get("content") or "" for m in res.messages)
        rows.append(Row(q.id, q.category, q.paraphrase, grade(q, res.text), res.text,
                        all(v.lower() in ctx.lower() for v in q.evidence) if q.evidence else None,
                        res.memory_tokens + sum(count_tokens(m["content"]) for m in res.messages if m["role"] == "tool"),
                        res.input_tokens, res.model_calls))
    return RunResult(mode, budget_tokens, rows, label)


def compare_modes(ds: Dataset, modes=MODES, **kw) -> dict[str, RunResult]:
    return {m: run(ds, m, **kw) for m in modes}


def recall_vs_budget(ds: Dataset, budgets=(0, 16, 32, 64, 128, 256, 512), mode: str = "implicit", k: int = 50,
                     **kw) -> list[dict]:
    """Recall (evidence in context), accuracy and memory tokens per question as the per-turn budget grows.
    The store is built once and reused, and ``k`` is large, so only the budget decides what fits."""
    store, _ = build_memory(ds, mode=mode, **{a: v for a, v in kw.items() if a in ("write", "embedder", "consolidate")})
    out = []
    for b in budgets:
        r = run(ds, mode, budget_tokens=b, store=store, k=k)
        out.append({"budget": b, "recall": r.recall, "accuracy": r.accuracy, "memory_tokens": r.mean("memory_tokens")})
    return out


def knee(points: list[dict], key: str = "recall", frac: float = 0.95) -> dict:
    """The smallest budget whose ``key`` reaches ``frac`` of the best value: past it, tokens buy little."""
    best = max(p[key] for p in points)
    return next(p for p in sorted(points, key=lambda p: p["budget"]) if p[key] >= frac * best)


def paraphrase_gap(result: RunResult) -> tuple[float, float]:
    """Accuracy on the standard wording vs the paraphrase subset, over the same facts."""
    k1, n1 = result.rate("correct", paraphrase=False)
    k2, n2 = result.rate("correct", paraphrase=True)
    return (k1 / n1 if n1 else 0.0, k2 / n2 if n2 else 0.0)


def summary(results: dict[str, RunResult]) -> str:
    w = max([13] + [len(n) for n in results])
    lines = [f"{'mode':{w}s} {'accuracy':>9s} {'95% Wilson':>15s} {'stale':>6s} {'halluc':>7s} {'para':>6s} "
             f"{'mem tok':>8s} {'in tok':>7s} {'calls':>6s}"]
    for name, r in results.items():
        k, n = r.rate()
        lo, hi = wilson_interval(k, n)
        kp, np_ = r.rate("correct", paraphrase=True)
        lines.append(f"{name:{w}s} {k:4d}/{n:<4d} [{lo:5.2f}, {hi:5.2f}] {r.rate('stale')[0]:6d} "
                     f"{r.rate('hallucinated')[0]:7d} {kp:2d}/{np_:<3d} {r.mean('memory_tokens'):8.0f} "
                     f"{r.mean('input_tokens'):7.0f} {r.mean('model_calls'):6.2f}")
    return "\n".join(lines)


def render_rows(rows: list[Row], n: int = 10) -> str:
    return "\n".join(f"  {r.qid:14s} {r.category:16s} {r.verdict:12s} {r.answer[:60]}" for r in itertools.islice(rows, n))

