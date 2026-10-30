"""The charts in ../figures/ are what tools/make_figures.py computes with distillcore now (FIGURE-STYLE.md §7).

Each committed chart must be exactly what svgchart.render() makes of the spec stored in its <desc>, and that spec
must hold the values the core computes today (svgchart.check_figure()). If a formula or an input changes, a test
here fails until the chart is regenerated: ``python3 tools/make_figures.py``. The primer must embed every chart,
and every decimal number or price a chart prints (its title, series names and marks) must stand in the primer as
printed, so that the reader finds the sentence.
"""
import importlib.util
import re
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("distillcore_make_figures", CORE / "tools" / "make_figures.py")
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)

PRIMER = " ".join((CORE.parent / "PRIMER.md").read_text(encoding="utf-8").split())


@pytest.mark.parametrize("name", sorted(mf.FIGURES))
def test_chart_is_what_the_core_computes(name):
    mf.svgchart.check_figure(mf.FIG_DIR / name, mf.FIGURES[name])


def test_every_chart_file_is_in_the_table():
    """FIGURES is not empty, and it names exactly the chart files in ../figures/ (a computed chart keeps its spec in a
    <desc>; the hand-drawn SVGs have none), so that no chart file goes unchecked."""
    charts = {p.name for p in mf.FIG_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts and charts == set(mf.FIGURES), f"chart files {sorted(charts)} != FIGURES {sorted(mf.FIGURES)}"


def test_every_chart_is_embedded_in_the_primer():
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in PRIMER]
    assert not missing, f"PRIMER.md does not embed: {missing}"


# A decimal number or a price: a bare integer such as "4" or "50" is somewhere in any primer, so it proves nothing.
NUMBER = re.compile(r"\$\d[\d,]*(?:\.\d+)?×?|\d[\d,]*\.\d+×?")


def _printed(spec: dict) -> list[str]:
    """Every text the chart prints that can carry a number: the title, the series names, the mark labels."""
    texts = [spec.get("title", "")]
    texts += [s.get("name", "") for s in spec.get("series", [])]
    texts += [m.get("label") or m.get("text") or "" for m in spec.get("marks", [])]
    return texts


@pytest.mark.parametrize("name", sorted(mf.FIGURES))
def test_every_number_a_chart_prints_is_in_the_primer(name):
    missing = [num for text in _printed(mf.FIGURES[name]()) for num in NUMBER.findall(text) if num not in PRIMER]
    assert not missing, f"numbers on {name} that PRIMER.md does not print: {missing}"
