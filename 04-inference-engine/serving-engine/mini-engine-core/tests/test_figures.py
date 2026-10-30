"""The charts of ../PRIMER.md that minengine computes (tools/orchestration/FIGURE-STYLE.md §7).

Each chart in ../figures/ must be what ``tools/make_figures.py`` computes now with minengine, and what
tools/orchestration/svgchart.py renders from the spec stored in the file. If a function, a preset or a default changes
(``perf.GPUS``, ``perf.LLMS``, the efficiencies or the overhead of ``perf.step_cost``), regenerate the charts
(``python3 tools/make_figures.py``) and read the primer's captions again. Standard library + numpy + pytest.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("minengine_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart
PRIMER = (LAB.parent / "PRIMER.md").read_text(encoding="utf-8")


def check(name: str) -> None:
    svgchart.check_figure(make_figures.FIGURES_DIR / name, make_figures.FIGURES[name])


def test_step_time_vs_tokens():
    check("step-time-vs-tokens.svg")


def test_token_budget_worst_step():
    check("token-budget-worst-step.svg")


def test_spec_speedup_vs_k():
    check("spec-speedup-vs-k.svg")


def test_spec_speedup_vs_batch():
    check("spec-speedup-vs-batch.svg")


def test_every_chart_has_a_test():
    assert set(make_figures.FIGURES) == {"step-time-vs-tokens.svg", "token-budget-worst-step.svg",
                                         "spec-speedup-vs-k.svg", "spec-speedup-vs-batch.svg"}


def test_every_computed_chart_is_built():
    # every chart in ../figures/ that svgchart rendered (it stores its spec in <desc>) has a spec builder
    charts = {p.name for p in make_figures.FIGURES_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts == set(make_figures.FIGURES)


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_every_chart_is_referenced_by_the_primer(name):
    assert f"figures/{name}" in PRIMER, f"PRIMER.md does not embed figures/{name}"


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_the_alt_text_is_the_chart_title(name):
    title = make_figures.FIGURES[name]()["title"]
    assert f"![{title}](figures/{name})" in PRIMER, f"the alt text of figures/{name} is not the chart's title"


def _printed(label: str) -> bool:
    # the number part of a mark ("8,192: 224 ms" -> "224 ms") is in the prose; a list of speedups ("k = 4, 5, 6, 7
    # give 2.40, 2.46, ...") prints the factor without its "×"
    number = label.rsplit(": ", 1)[-1]
    return number in PRIMER or (number.endswith("×") and re.search(rf"{re.escape(number[:-1])}\b", PRIMER) is not None)


def test_the_marks_carry_the_primers_numbers():
    # the labels are the numbers as the primer prints them, so that the reader finds the sentence
    labels = {m.get("label") for build in make_figures.FIGURES.values() for m in build()["marks"]}
    for text in ["8,192: 224 ms", "512: 14.2 ms", "2,048: 52.0 ms", "knee: ~221 tokens", "512 tokens: 14.4 ms",
                 "1,024: 26.7 ms", "p99 ITL target: 25 ms", "best k = 6: 2.47×", "k = 4: 2.40×", "k = 4: 3.36",
                 "B = 1: 1.58×", "B = 128: 0.97×", "0.65×", "1.45×"]:
        assert text in labels, text
        assert _printed(text), f"the mark {text!r}: PRIMER.md does not print its number"


def test_every_mark_label_is_in_the_primer():
    # a later edit of the prose must not leave a mark whose number the text no longer prints
    for build in make_figures.FIGURES.values():
        for m in build()["marks"]:
            if m["kind"] in ("point", "vline", "hline") and m.get("label"):
                assert _printed(m["label"]), f"the mark {m['label']!r}: PRIMER.md does not print its number"
