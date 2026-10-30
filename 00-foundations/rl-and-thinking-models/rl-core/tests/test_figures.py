"""The charts in ../figures/ are what tools/make_figures.py computes with rlcore now (FIGURE-STYLE.md §7).

Each committed chart must be exactly what svgchart.render() makes of the spec stored in its <desc>, and that spec
must hold the values the core computes today (svgchart.check_figure()). If a formula, a preset or an input changes,
a test here fails until the chart is regenerated: ``python3 tools/make_figures.py``. The primer must embed every
chart, and every decimal number a chart prints (its title, series names and marks) must stand in the primer as
printed, so that the reader finds the sentence.
"""
import importlib.util
import re
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("rlcore_make_figures", CORE / "tools" / "make_figures.py")
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)

PRIMER_PATH = CORE.parent / "PRIMER.md"
PRIMER = " ".join(PRIMER_PATH.read_text(encoding="utf-8").split()) if PRIMER_PATH.exists() else ""
needs_primer = pytest.mark.skipif(not PRIMER, reason="PRIMER.md not beside this core")


def test_majority_vote_vs_n():
    mf.svgchart.check_figure(mf.FIG_DIR / "majority-vote-vs-n.svg", mf.spec_majority_vote_vs_n)


def test_kv_token_steps():
    mf.svgchart.check_figure(mf.FIG_DIR / "kv-token-steps.svg", mf.spec_kv_token_steps)


def test_memory_gpus_vs_output():
    mf.svgchart.check_figure(mf.FIG_DIR / "memory-gpus-vs-output.svg", mf.spec_memory_gpus_vs_output)


def test_max_tokens_vs_budget():
    mf.svgchart.check_figure(mf.FIG_DIR / "max-tokens-vs-budget.svg", mf.spec_max_tokens_vs_budget)


def test_every_chart_file_is_in_the_table():
    """FIGURES names exactly the chart files in ../figures/ (a computed chart keeps its spec in a <desc>; the
    hand-drawn SVGs have none), so that no chart file goes unchecked."""
    charts = {p.name for p in mf.FIG_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts and charts == set(mf.FIGURES), f"chart files {sorted(charts)} != FIGURES {sorted(mf.FIGURES)}"


@needs_primer
def test_every_chart_is_embedded_in_the_primer():
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in PRIMER]
    assert not missing, f"PRIMER.md does not embed: {missing}"


# A decimal number, with its × when it has one: a bare integer such as "15" or "31" is somewhere in any primer.
NUMBER = re.compile(r"\d[\d,]*\.\d+×?")


def _printed(spec: dict) -> list[str]:
    """Every text the chart prints that can carry a number: the title, the series names, the mark labels."""
    texts = [spec.get("title", "")]
    texts += [s.get("name", "") for s in spec.get("series", [])]
    texts += [m.get("label") or m.get("text") or "" for m in spec.get("marks", [])]
    return texts


@needs_primer
@pytest.mark.parametrize("name", sorted(mf.FIGURES))
def test_every_number_a_chart_prints_is_in_the_primer(name):
    """Read from the committed file (the tests above pin it to the core), so the slow builders run once."""
    spec = mf.svgchart.read_spec((mf.FIG_DIR / name).read_text(encoding="utf-8"))
    missing = [num for text in _printed(spec) for num in NUMBER.findall(text) if num not in PRIMER]
    assert not missing, f"numbers on {name} that PRIMER.md does not print: {missing}"
