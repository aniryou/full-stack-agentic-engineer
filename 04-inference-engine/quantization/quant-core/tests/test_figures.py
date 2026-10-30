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


def test_gemm_speedup_vs_tokens():
    check("gemm-speedup-vs-tokens.svg")


def test_sqnr_vs_bits():
    # seeded Gaussian weights rounded onto integer grids; the primer prints the SQNR to 0.1 dB
    check("sqnr-vs-bits.svg", rel_tol=1e-4)


def test_kv_cache_bytes_and_sessions():
    check("kv-cache-bytes-and-sessions.svg")


def test_visible_drop_vs_eval_items():
    check("visible-drop-vs-eval-items.svg")


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_every_chart_is_referenced_by_the_primer(name):
    assert f"figures/{name}" in PRIMER, f"PRIMER.md does not embed figures/{name}"
