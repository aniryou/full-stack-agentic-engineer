"""The charts of ../PRIMER.md that quantcore computes (tools/orchestration/FIGURE-STYLE.md §7).

Each chart in ../figures/ must be what ``tools/make_figures.py`` computes now with quantcore, and what
tools/orchestration/svgchart.py renders from the spec stored in the file. If a function, a seed or a default changes,
regenerate the charts (``python3 tools/make_figures.py``) and read the primer's captions again. Standard library +
numpy + pytest.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("quantcore_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart
PRIMER = (LAB.parent / "PRIMER.md").read_text(encoding="utf-8")


def check(name: str, **tolerances) -> None:
    svgchart.check_figure(make_figures.FIGURES_DIR / name, make_figures.FIGURES[name], **tolerances)


TOLERANCES = {
    # seeded Gaussian weights rounded onto integer grids; the primer prints the SQNR to 0.1 dB
    "sqnr-vs-bits.svg": {"rel_tol": 1e-4},
}


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_chart(name):
    check(name, **TOLERANCES.get(name, {}))


def test_every_computed_chart_is_built():
    # every chart in ../figures/ that svgchart rendered (it stores its spec in <desc>) has a spec builder
    charts = {p.name for p in make_figures.FIGURES_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts == set(make_figures.FIGURES)


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_every_chart_is_referenced_by_the_primer(name):
    assert f"figures/{name}" in PRIMER, f"PRIMER.md does not embed figures/{name}"
