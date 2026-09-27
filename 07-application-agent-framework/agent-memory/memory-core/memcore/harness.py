"""harness.py - measure memory with planted facts: say something in one session, ask about it sessions later.

The one idea: you cannot tune what you do not measure, and memory is measurable offline. A seeded generator
plants facts about a user across sessions (in filler, as people say them), changes one of them later, and slips a
false claim in through a tool result. It then asks questions in the task shapes of LongMemEval (extraction,
preference, multi-session, temporal, knowledge update, abstention) and LoCoMo (adversarial: a question with no
true answer) - shapes only, no data copied or downloaded. A strict template reader answers from whatever
retrieval packed into the budget, so every miss is a retrieval or write-path miss. Metrics: accuracy with a
Wilson interval (07.2 notebook 08 §4), recall of the evidence within the budget, stale answers, abstention,
tokens and model calls per turn. A paraphrase subset asks the same questions in other words: the lexical
hashing embedder misses those by design - that gap is what a real embedder buys (the lab's T1 path).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .agent import MemoryAgent, UserTurn, read_answer
from .consolidate import Consolidator
from .records import DAY, HOUR, Scope
from .retrieve import retrieve
from .store import MemoryStore
from .write import SLOTS, Writer, extract, read_facts

VALUES = {"name": ["Ana", "Bruno", "Chen", "Dara", "Emil", "Farah"],
          "home_city": ["Lisbon", "Porto", "Madrid", "Oslo", "Prague", "Dublin", "Vienna", "Lyon"],
          "employer": ["Acme", "Globex", "Initech", "Hooli", "Umbrella", "Vandelay"],
          "allergy": ["peanuts", "shellfish", "gluten", "sesame"], "seat_preference": ["window", "aisle"],
          "pet": ["cat", "dog", "parrot", "rabbit"], "language": ["Portuguese", "Spanish", "Norwegian", "Czech"]}
STATEMENT = {"name": "My name is {v}.", "home_city": "I live in {v}.", "employer": "I work at {v}.",
             "allergy": "I'm allergic to {v}.", "seat_preference": "I prefer {v} seats.", "pet": "My pet is a {v}.",
             "language": "I am learning {v}."}
PARAPHRASE = {"name": "How should we address the user?", "employer": "Who signs the user's pay cheques?",
              "allergy": "Which food makes the user ill?", "seat_preference": "Where does the user like to sit on a plane?",
              "pet": "What animal shares the user's flat?", "language": "Which tongue is the user studying?"}
FILLER = ["The weather has been strange all week.", "I spent the morning answering email.",
          "The train was late again today.", "I finally finished that long report.", "We had a team lunch at noon."]


@dataclass
class Question:
    qtype: str
    text: str
    slots: tuple
    answer: str | None                  # None: the right answer is "I don't know"
    as_of: float | None = None
    stale: str | None = None            # the superseded value, for knowledge-update questions
    paraphrase: bool = False


@dataclass
class Scenario:
    scope: Scope
    turns: list                         # (time, session, text, source)
    questions: list
    values: dict
    end: float


def generate(seed: int = 0, sessions: int = 6, tenant: str = "t0") -> Scenario:
    rng, scope = random.Random(seed), Scope(tenant, f"u{seed}")
    others = [k for k in VALUES if k not in ("home_city", "seat_preference")]
    planted = ["home_city", "seat_preference"] + rng.sample(others, 3)
    unplanted = [k for k in VALUES if k not in planted]
    values = {k: rng.choice(VALUES[k]) for k in planted}
    moved = rng.choice([c for c in VALUES["home_city"] if c != values["home_city"]])
    when = {"home_city": 0, "seat_preference": rng.randrange(sessions - 3), planted[2]: 1, planted[3]: 2,
            planted[4]: rng.randrange(sessions - 3)}      # the multi-session pair lives in two sessions
    turns = []
    for s in range(sessions):
        says = [STATEMENT[k].format(v=values[k]) + f" (about my {SLOTS[k][0]})" for k in planted if when[k] == s]
        if s == sessions - 2:
            says.append(f"I moved to {moved}. (about my home city)")
        says += [""] * (3 - len(says))
        for i, say in enumerate(says):
            text = " ".join(x for x in (rng.choice(FILLER), say, rng.choice(FILLER)) if x)
            turns.append((s * DAY + (9 + i) * HOUR, s, text, "user"))
    poison = unplanted[0]                # a tool result asserts a fact the user never stated
    turns.append(((sessions - 3) * DAY + 15 * HOUR, sessions - 3,
                  f"The user's {SLOTS[poison][0]} is {rng.choice(VALUES[poison])}.", "tool"))
    noun = lambda k: SLOTS[k][0]
    qs = [Question("extraction", f"What is the user's {noun(k)}?", (k,), values[k]) for k in planted[2:]]
    qs += [Question("preference", "What is the user's seat preference?", ("seat_preference",), values["seat_preference"]),
           Question("knowledge-update", "What is the user's home city?", ("home_city",), moved, stale=values["home_city"]),
           Question("temporal", "What was the user's home city on day 2?", ("home_city",), values["home_city"],
                    as_of=2.5 * DAY),
           Question("multi-session", f"What are the user's {noun(planted[2])} and {noun(planted[3])}?",
                    (planted[2], planted[3]), f"{values[planted[2]]}; {values[planted[3]]}"),
           Question("abstention", f"What is the user's {noun(unplanted[1])}?", (unplanted[1],), None),
           Question("adversarial", f"What is the user's {noun(poison)}?", (poison,), None)]
    qs += [Question(q.qtype, PARAPHRASE[q.slots[0]], q.slots, q.answer, paraphrase=True)
           for q in qs if q.qtype in ("extraction", "preference")]
    return Scenario(scope, turns, qs, values, sessions * DAY)


def build_store(sc: Scenario, mode: str = "consolidated", policy=None) -> MemoryStore:
    """episodes: raw turns only. consolidated: raw turns, then one consolidation run. facts: hot-path extraction."""
    store = MemoryStore()
    writer = Writer(store, policy)
    for t, s, text, source in sc.turns:
        recs = extract(text, sc.scope, source=source, at=t, turn_id=f"s{s}:{t:g}")
        for rec in (recs if mode == "facts" else recs[:1]):
            writer.write(rec)
    if mode == "consolidated":
        Consolidator(store).run(sc.scope, 0.0, sc.end, now=sc.end)
    return store


@dataclass
class Outcome:
    q: Question
    answer: str | None
    correct: bool
    recalled: bool | None               # was every evidence value in the packed context? (None: unanswerable)
    stale: bool
    tokens: int


def evaluate(sc: Scenario, store, budget_tokens: int = 60, kinds=None, form: str = "paper") -> list[Outcome]:
    out = []
    for q in sc.questions:
        rec = retrieve(store, sc.scope, q.text, now=sc.end, budget_tokens=budget_tokens, kinds=kinds,
                       as_of=q.as_of, form=form, touch=False)
        ans = read_answer(q.slots, rec.records, q.as_of)
        correct = ans is None if q.answer is None else (ans or "").lower() == q.answer.lower()
        recalled = None
        if q.answer is not None:
            seen = {(k, v.lower()) for r in rec.records for k, v in read_facts(r.text)}
            recalled = all((s, v.lower()) in seen for s, v in zip(q.slots, q.answer.split("; ")))
        out.append(Outcome(q, ans, correct, recalled, q.stale is not None and ans == q.stale, rec.tokens))
    return out


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """agentlab.evals.gate.wilson_interval, restated: sane at 0/n and n/n, unlike p +/- 1.96 SE."""
    if n == 0:
        return (0.0, 1.0)
    p, z2 = passes / n, z * z
    centre, half = (p + z2 / (2 * n)) / (1 + z2 / n), z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(outcomes) -> dict:
    n, ok = len(outcomes), sum(o.correct for o in outcomes)
    answerable = [o for o in outcomes if o.recalled is not None]
    ku = [o for o in outcomes if o.q.qtype == "knowledge-update"]
    unans = [o for o in outcomes if o.q.answer is None]
    by = {}
    for o in outcomes:
        key = o.q.qtype + (" (paraphrase)" if o.q.paraphrase else "")
        by.setdefault(key, []).append(o.correct)
    return {"n": n, "accuracy": ok / n, "wilson": wilson_interval(ok, n),
            "recall": sum(o.recalled for o in answerable) / len(answerable),
            "stale": sum(o.stale for o in ku) / max(1, len(ku)),
            "abstention": sum(o.correct for o in unans) / max(1, len(unans)),
            "tokens": sum(o.tokens for o in outcomes) / n, "by_type": {k: sum(v) / len(v) for k, v in by.items()}}


def recall_vs_budget(budgets=(15, 30, 45, 60, 90, 120), seeds=range(30), mode: str = "consolidated") -> list[dict]:
    kinds = ("episodic",) if mode == "episodes" else ("semantic", "procedural")
    runs = [(sc, build_store(sc, mode)) for sc in map(generate, seeds)]
    rows = []
    for b in budgets:
        s = summarize([o for sc, st in runs for o in evaluate(sc, st, b, kinds)])
        rows.append({"budget": b, "recall": s["recall"], "accuracy": s["accuracy"], "tokens": s["tokens"],
                     "stale": s["stale"]})
    return rows


def knee(rows, tol: float = 0.02) -> int:
    """The smallest budget whose recall is within `tol` of the best recall any budget reached."""
    best = max(r["recall"] for r in rows)
    return min(r["budget"] for r in rows if r["recall"] >= best - tol)


def compare_modes(seeds=range(30), budget_tokens: int = 60, profile_tokens: int = 60) -> dict:
    """The same user through the agent in three memory modes; score the last session's turns."""
    out = {}
    for mode in ("tools", "implicit", "pinned"):
        ok = need = mem = calls = turns = stable = 0
        for sc in map(generate, seeds):
            agent = MemoryAgent(MemoryStore(), sc.scope, mode=mode, budget_tokens=budget_tokens,
                                profile_tokens=profile_tokens)
            for s in range(max(t[1] for t in sc.turns) + 1):
                agent.start_session(f"s{s}", s * DAY)
                for t, _, text, source in [x for x in sc.turns if x[1] == s and x[3] == "user"]:
                    agent.run(UserTurn(text), now=t)
            agent.start_session("test", sc.end)
            test = [(UserTurn(q.text, ask=q.slots), q.answer) for q in sc.questions
                    if q.qtype in ("extraction", "preference", "knowledge-update") and not q.paraphrase]
            test += [(UserTurn("Please book me a flight to Rome next month.", needs=("seat_preference",)),
                      sc.values["seat_preference"]), (UserTurn("Thanks, that is all for now."), None)]
            for turn, want in test:
                res = agent.run(turn, now=sc.end)
                turns, mem, calls = turns + 1, mem + res.memory_tokens, calls + res.calls
                stable += sum(r.tokens for r in agent.profile)        # pinned: the same bytes every turn
                if want is not None:
                    need += 1
                    ok += (res.text == want) if turn.ask else (want in res.text)
        out[mode] = {"accuracy": ok / need, "memory_tokens_per_turn": mem / turns,
                     "stable_tokens_per_turn": stable / turns, "calls_per_turn": calls / turns}
    return out
