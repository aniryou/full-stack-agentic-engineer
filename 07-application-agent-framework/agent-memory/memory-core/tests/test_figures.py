"""The charts in ../figures/ are what tools/make_figures.py computes with memcore now (FIGURE-STYLE.md §7).

Each committed chart must be exactly what svgchart.render() makes of the spec stored in its <desc>, and that spec
must hold the values the core computes today (svgchart.check_figure()). If a formula, an input or the harness
changes, a test here fails until the chart is regenerated: ``python3 tools/make_figures.py``. The primer must embed
every chart, and every decimal number or percentage a chart prints (its title, series names, marks and texts) must
stand in the primer as printed, so that the reader finds the sentence.
"""
import importlib.util
import re
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("memcore_make_figures", CORE / "tools" / "make_figures.py")
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)

PRIMER = " ".join((CORE.parent / "PRIMER.md").read_text(encoding="utf-8").split())
_SPECS: dict = {}


def spec(name: str) -> dict:
    """The spec of one chart, computed once per test session (the harness chart runs 30 users at six budgets)."""
    if name not in _SPECS:
        _SPECS[name] = mf.FIGURES[name]()
    return _SPECS[name]


def test_chart_recency_two_forms():
    mf.svgchart.check_figure(mf.FIG_DIR / "recency-two-forms.svg", lambda: spec("recency-two-forms.svg"))


def test_chart_recall_vs_budget():
    mf.svgchart.check_figure(mf.FIG_DIR / "recall-vs-budget.svg", lambda: spec("recall-vs-budget.svg"))


def test_chart_uncached_tokens_per_turn():
    mf.svgchart.check_figure(mf.FIG_DIR / "uncached-tokens-per-turn.svg", lambda: spec("uncached-tokens-per-turn.svg"))


def test_chart_session_cost_vs_system_prompt():
    mf.svgchart.check_figure(mf.FIG_DIR / "session-cost-vs-system-prompt.svg",
                             lambda: spec("session-cost-vs-system-prompt.svg"))


def test_the_builders_are_the_table():
    """FIGURES maps each chart file to the builder of the same name, so the tests above check the generator's specs."""
    assert mf.FIGURES == {"recency-two-forms.svg": mf.spec_recency_two_forms,
                          "recall-vs-budget.svg": mf.spec_recall_vs_budget,
                          "uncached-tokens-per-turn.svg": mf.spec_uncached_per_turn,
                          "session-cost-vs-system-prompt.svg": mf.spec_session_cost_vs_system_prompt}


def test_every_chart_file_is_in_the_table():
    """FIGURES names exactly the chart files in ../figures/ (a computed chart keeps its spec in a <desc>; the
    hand-drawn SVGs have none), so that no chart file goes unchecked."""
    charts = {p.name for p in mf.FIG_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts and charts == set(mf.FIGURES), f"chart files {sorted(charts)} != FIGURES {sorted(mf.FIGURES)}"


def test_every_chart_is_embedded_in_the_primer():
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in PRIMER]
    assert not missing, f"PRIMER.md does not embed: {missing}"


# A decimal number or a percentage: a bare integer such as "8" or "60" is somewhere in any primer, so it proves nothing.
NUMBER = re.compile(r"\d[\d,]*\.\d+|\d+%")


def _printed(chart: dict) -> list[str]:
    """Every text the chart prints that can carry a number: the title, the series names, the marks and texts."""
    texts = [chart.get("title", "")]
    texts += [s.get("name", "") for s in chart.get("series", [])]
    texts += [m.get("label") or m.get("text") or "" for m in chart.get("marks", [])]
    return texts


@pytest.mark.parametrize("name", sorted(mf.FIGURES))
def test_every_number_a_chart_prints_is_in_the_primer(name):
    missing = [num for text in _printed(spec(name)) for num in NUMBER.findall(text) if num not in PRIMER]
    assert not missing, f"numbers on {name} that PRIMER.md does not print: {missing}"


def test_the_charts_mark_the_primer_points():
    """The marks are the pinned points of §3, §4 and §5, labelled as the primer prints them."""
    want = {
        "recency-two-forms.svg": ("A 1.0", "B 0.6364", "C 0", "C 1.0", "B 0.4975", "A 0"),
        "recall-vs-budget.svg": ("knee: 90 tokens", "60 tokens: 90.9%", "60 tokens: 17.6%", "15.4% at zero recall"),
        "uncached-tokens-per-turn.svg": ("turn 8: 1,560", "turn 8: 600", "turn 8: 56"),
        "session-cost-vs-system-prompt.svg": ("0.04464", "0.03984", "0.03084", "0.02116"),
    }
    for name, labels in want.items():
        got = {m.get("label") for m in spec(name)["marks"]}
        for label in labels:
            assert label in got, (name, label)
