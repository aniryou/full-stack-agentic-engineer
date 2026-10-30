#!/usr/bin/env python3
"""The charts of the topic's PRIMER.md, computed by memcore (tools/orchestration/FIGURE-STYLE.md §7).

Each chart is a spec builder, ``spec_<name>() -> dict``, that computes every value with the core's own functions
from the inputs the primer names: the three memories A, B and C of the §3 worked example (last used 1 h, 24 h and
72 h ago), the budgets of the §4 table (15 to 120 tokens over 30 seeded users), and the session of §5 (a
2,000-token system prompt, 400 tokens of memory, 40-token user messages, 120-token replies, 16-token blocks, the
dated price table and the 4,096-token caching minimum of ``memcore.budget``). ``tools/orchestration/svgchart.py``
renders the spec into the house style, and ``main()`` writes each chart into ``../figures/`` beside the primer.
``tests/test_figures.py`` calls ``svgchart.check_figure()`` with the same builders, so a committed chart fails the
test when the core computes other values.

The planted-facts chart is deterministic: ``harness.generate`` draws from ``random.Random(seed)`` for the fixed
seeds 0 to 29, the hashing embedder and the template reader have no randomness, and ``tests/test_primer_numbers.py``
pins the same table. Not charted: the modes comparison of §6 (a scripted model, a fixture rule), the prefill times of
§5 (a roofline, SIMULATED, already in the layout table) and the decay of §7 (one formula with one pinned point).

Run it from anywhere: ``python3 tools/make_figures.py`` (standard library + numpy, like the core). The output is
deterministic: two runs give the same bytes.
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys

CORE = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = CORE.parent / "figures"
REPO = next(p for p in CORE.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file())


def _load_svgchart():
    if "svgchart" in sys.modules:
        return sys.modules["svgchart"]
    spec = importlib.util.spec_from_file_location("svgchart", REPO / "tools" / "orchestration" / "svgchart.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svgchart"] = mod
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()

if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
from memcore import budget, harness  # noqa: E402
from memcore.records import HOUR  # noqa: E402
from memcore.retrieve import minmax, recency  # noqa: E402

# The inputs the primer names.
NOW = 100 * HOUR                            # §3: any clock; only the hours from the last access count
AGES_H = {"A": 1, "B": 24, "C": 72}         # §3 worked example: hours from the last access of A, B and C
BUDGETS = (15, 30, 45, 60, 90, 120)         # §4: the budgets of "Recall against the budget"
USERS = range(30)                           # §4: "Over 30 users", the seeds 0 to 29 of harness.generate
AT = 60                                     # §4: the 60-token budget that the prose quotes
SESSION = dict(system_tokens=2000, memory_tokens=400, user_tokens=40, output_tokens=120, block_size=16)   # §5
TURNS = 20                                  # §5: the longer session of the layout table
OUTPUT = SESSION["output_tokens"]
SESSION_TURNS = 8                           # §5: the session of the dollar table
SYSTEM_RANGE = (1000, 6000, 250)            # system prompts around the 2,000 and 4,000 of the dollar table
LAYOUT_STYLE = {"none": "muted", "before_history": "warm", "pinned": "accent", "tail": "violet"}


def _sig(v: float) -> float:
    """Nine significant digits: a shorter <desc>, far inside check_figure()'s tolerance."""
    return float(f"{float(v):.9g}")


def _pct1(x: float) -> str:
    return f"{x:.1%}"


# ------------------------------------------------------------------ §3: the recency term in the two forms

def _norm_label(name: str, v: float) -> str:
    """As the §3 table prints a normalised term: "A 1.0", "C 0", "B 0.6364"."""
    return f"{name} " + ("1.0" if v == 1 else "0" if v == 0 else f"{v:.4f}")


def spec_recency_two_forms() -> dict:
    names = list(AGES_H)
    ages = [AGES_H[n] for n in names]
    last = [NOW - h * HOUR for h in ages]
    paper3 = minmax(recency(last, NOW, "paper"))
    code3 = minmax(recency(last, NOW, "code"))
    # The paper form depends on the hours alone, so a memory at any age between A and C has a value on one curve
    # (A and C stay the minimum and the maximum). The code form depends on the rank in the set, so it has a value
    # only for the three memories of the example.
    grid = list(range(min(ages), max(ages) + 1))
    paper_curve = minmax(recency([NOW - h * HOUR for h in grid], NOW, "paper"))
    # A label sits above its point, except where a line leaves the point upward to the right: there it goes below
    # (the code form's B, and A at 0, whose label goes under the axis between the ticks).
    marks = []
    for form, values, style in (("paper", paper3, "accent"), ("code", code3, "warm")):
        for n, h, v in zip(names, ages, values):
            below = form == "code" and n != "C"
            marks.append({"kind": "point", "x": h, "y": _sig(v), "label": _norm_label(n, v), "style": style,
                          "anchor": "start", "dy": (18 if v == 0 else 16) if below else -9})
    marks += [
        {"kind": "text", "x": 36, "y": 0.1, "text": "paper form: 0.995^hours", "style": "accent", "anchor": "start"},
        {"kind": "text", "x": 40, "y": 0.93, "text": "code form: 0.99^rank, oldest first", "style": "warm",
         "anchor": "end"},
    ]
    return {
        "kind": "line",
        "title": "The normalised recency term of A, B and C against the hours from the last access, in the paper "
                 "form and the code form.",
        "x": {"label": "hours from the last access (h)", "min": 0, "max": 80, "ticks": [0, 12, 24, 36, 48, 60, 72]},
        "y": {"label": "recency term, normalised", "min": 0, "max": 1.12, "ticks": [0, 0.25, 0.5, 0.75, 1.0],
              "tick_labels": ["0", "0.25", "0.5", "0.75", "1.0"]},
        "series": [
            {"name": "paper form: 0.995^hours", "x": grid, "y": [_sig(v) for v in paper_curve], "style": "accent"},
            {"name": "code form: 0.99^rank, oldest first", "x": ages, "y": [_sig(v) for v in code3],
             "style": "warm", "markers": True},
        ],
        "marks": marks,
        "legend": "none",
        "note": "computed by memcore.retrieve.recency and memcore.retrieve.minmax",
    }


# ------------------------------------------------------------------ §4: recall against the token budget

def spec_recall_vs_budget() -> dict:
    facts = {r["budget"]: r for r in harness.recall_vs_budget(BUDGETS, seeds=USERS)}
    raw = {r["budget"]: r for r in harness.recall_vs_budget(BUDGETS, seeds=USERS, mode="episodes")}
    k = harness.knee(list(facts.values()))
    xs = list(BUDGETS)
    pct = lambda rows, key: [_sig(100 * rows[b][key]) for b in xs]
    lo = xs[0]
    return {
        "kind": "line",
        "title": "Recall for consolidated facts and raw episodes, and the accuracy of raw episodes, against the "
                 "per-turn token budget over the 30 users of the harness.",
        "x": {"label": "memory budget per turn (tokens)", "min": 10, "max": 124, "ticks": xs},
        "y": {"label": "share of questions (%)", "min": 0, "max": 108, "ticks": [0, 20, 40, 60, 80, 100]},
        "series": [
            {"name": "consolidated facts: recall", "x": xs, "y": pct(facts, "recall"), "style": "accent",
             "markers": True},
            {"name": "raw episodes: recall", "x": xs, "y": pct(raw, "recall"), "style": "warm", "markers": True},
            {"name": "raw episodes: accuracy", "x": xs, "y": pct(raw, "accuracy"), "style": "muted", "dash": True,
             "markers": True},
        ],
        "marks": [
            {"kind": "vline", "x": k, "label": f"knee: {k} tokens", "style": "accent", "anchor": "end",
             "label_pos": "bottom"},
            {"kind": "point", "x": AT, "y": _sig(100 * facts[AT]["recall"]),
             "label": f"{AT} tokens: {_pct1(facts[AT]['recall'])}", "style": "accent", "anchor": "end", "dy": -9},
            {"kind": "point", "x": AT, "y": _sig(100 * raw[AT]["recall"]),
             "label": f"{AT} tokens: {_pct1(raw[AT]['recall'])}", "style": "warm", "anchor": "start", "dy": 16},
            {"kind": "point", "x": lo, "y": _sig(100 * raw[lo]["accuracy"]),
             "label": f"{_pct1(raw[lo]['accuracy'])} at zero recall", "style": "muted", "anchor": "start", "dy": -9},
            # the series names, beside their lines right of the knee (no corner is free for a legend box)
            {"kind": "text", "x": 93, "y": 88, "text": "consolidated facts: recall", "style": "accent", "anchor": "start"},
            {"kind": "text", "x": 93, "y": 53, "text": "raw episodes: accuracy", "style": "muted", "anchor": "start"},
            {"kind": "text", "x": 93, "y": 15, "text": "raw episodes: recall", "style": "warm", "anchor": "start"},
        ],
        "legend": "none",
        "note": "computed by memcore.harness.recall_vs_budget and memcore.harness.knee",
    }


# ------------------------------------------------------------------ §5: the tokens prefilled again at each turn

def spec_uncached_per_turn() -> dict:
    series, at8 = [], {}
    for layout in budget.LAYOUTS:
        turns = budget.hits_per_turn(layout, turns=TURNS, **SESSION)
        xs = list(range(2, TURNS + 1))                 # turn 1 fills the cache: every layout prefills its whole prompt
        ys = [t.prompt - t.cached for t in turns[1:]]
        at8[layout] = ys[xs.index(8)]
        series.append({"name": "none (no memory)" if layout == "none" else layout, "x": xs, "y": ys,
                       "style": LAYOUT_STYLE[layout], "dash": layout == "none", "markers": False})
    # draw "none" last, dashed, over "pinned": the two lines are the same from turn 2
    series = [s for s in series if not s["dash"]] + [s for s in series if s["dash"]]
    return {
        "kind": "line",
        "title": "The tokens that the engine prefills again at each turn from turn 2 to turn 20, for the four "
                 "layouts of memory in the prompt.",
        "x": {"label": "turn of the session", "min": 2, "max": TURNS, "ticks": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]},
        "y": {"label": "uncached tokens per turn", "min": 0, "max": 3800, "ticks": [0, 1000, 2000, 3000]},
        "series": series,
        "marks": [
            {"kind": "point", "x": 8, "y": at8["before_history"], "label": f"turn 8: {at8['before_history']:,}",
             "style": "warm", "anchor": "end", "dy": -9},
            {"kind": "point", "x": 8, "y": at8["tail"], "label": f"turn 8: {at8['tail']:,}", "style": "violet",
             "anchor": "start", "dy": -9},
            {"kind": "point", "x": 8, "y": at8["pinned"], "label": f"turn 8: {at8['pinned']:,}", "style": "accent",
             "anchor": "start", "dy": -9},
        ],
        "legend": "top-left",
        "note": "computed by memcore.budget.hits_per_turn",
    }


# ------------------------------------------------------------------ §5: dollars per session against the system prompt

def _session_cost(layout: str, system_tokens: int) -> float:
    turns = budget.hits_per_turn(layout, turns=SESSION_TURNS, **{**SESSION, "system_tokens": system_tokens})
    return sum(budget.turn_cost(t.prompt, t.cached, OUTPUT)["total"] for t in turns)


def _crossings(layout: str) -> list[int]:
    """The system-prompt sizes at which one more turn of the session clears the provider's caching minimum: the
    prompt of turn t is the system prompt plus a fixed part, so turn t clears the minimum from one size on."""
    minimum = budget.CACHE_MIN_TOKENS[inspect.signature(budget.turn_cost).parameters["model"].default]
    ref = SESSION["system_tokens"]
    turns = budget.hits_per_turn(layout, turns=SESSION_TURNS, **SESSION)
    return sorted({minimum - (t.prompt - ref) for t in turns})


def spec_session_cost_vs_system_prompt() -> dict:
    lo, hi, step = SYSTEM_RANGE
    series, cost = [], {}
    for layout in budget.LAYOUTS:
        xs = set(range(lo, hi + 1, step)) | {2000, 4000}
        for c in _crossings(layout):                   # each step down: the size just below and at the crossing
            xs |= {x for x in (c - 1, c) if lo <= x <= hi}
        xs = sorted(xs)
        ys = [_session_cost(layout, s) for s in xs]
        cost[layout] = dict(zip(xs, ys))
        series.append({"name": layout, "x": xs, "y": [_sig(v) for v in ys], "style": LAYOUT_STYLE[layout],
                       "dash": layout == "none"})
    pt = lambda layout, s, anchor, dy: {"kind": "point", "x": s, "y": _sig(cost[layout][s]),
                                        "label": f"{cost[layout][s]:.5f}", "style": LAYOUT_STYLE[layout],
                                        "anchor": anchor, "dy": dy}
    return {
        "kind": "line",
        "title": "Dollars per 8-turn session on a hosted API against the size of the system prompt, for the four "
                 "layouts, with a 4,096-token caching minimum.",
        "x": {"label": "system prompt (tokens)", "min": lo, "max": hi, "ticks": list(range(lo, hi + 1, 1000))},
        "y": {"label": "$ per 8-turn session", "min": 0, "max": 0.06, "ticks": [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
              "tick_labels": ["0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06"]},
        "series": series,
        "marks": [
            {"kind": "vline", "x": 2000, "label": "2,000-token system prompt", "style": "muted", "anchor": "start",
             "label_pos": "bottom"},
            {"kind": "vline", "x": 4000, "label": "4,000-token system prompt", "style": "muted", "anchor": "start",
             "label_pos": "bottom"},
            {**pt("pinned", 2000, "end", -9), "style": "plain"},     # the three memory layouts share this point
            pt("none", 2000, "start", 16),
            pt("before_history", 4000, "start", -9),
            pt("pinned", 4000, "start", 26),                        # under the line of "none"
        ],
        "legend": "top-right",
        "note": "computed by memcore.budget.hits_per_turn and memcore.budget.turn_cost",
    }


FIGURES = {
    "recency-two-forms.svg": spec_recency_two_forms,
    "recall-vs-budget.svg": spec_recall_vs_budget,
    "uncached-tokens-per-turn.svg": spec_uncached_per_turn,
    "session-cost-vs-system-prompt.svg": spec_session_cost_vs_system_prompt,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
