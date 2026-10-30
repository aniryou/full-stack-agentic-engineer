"""The charts of PRIMER.md that gwcore computes (tools/orchestration/FIGURE-STYLE.md §7).

Each committed SVG in ../figures/ must be what tools/orchestration/svgchart.py renders from its own spec, and that spec
must be what tools/make_figures.py computes now with gwcore. If the core or a builder changes, regenerate with
``python3 tools/make_figures.py`` and look at the charts. Standard library + numpy (through gwcore) + pytest.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
PRIMER = LAB.parent / "PRIMER.md"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gwcore_make_figures", LAB / "tools" / "make_figures.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gwcore_make_figures"] = mod
    spec.loader.exec_module(mod)
    return mod


mf = _load_generator()
svgchart = mf.svgchart


def test_chain_availability_chart():
    svgchart.check_figure(mf.FIGURES_DIR / "chain-availability-common-mode.svg", mf.spec_chain_availability)


def test_semantic_cache_threshold_chart():
    svgchart.check_figure(mf.FIGURES_DIR / "semantic-cache-threshold-sweep.svg", mf.spec_cache_threshold_sweep)


def test_token_reservation_chart():
    svgchart.check_figure(mf.FIGURES_DIR / "token-reservation-sweep.svg", mf.spec_token_reservation_sweep)


def test_every_chart_is_referenced_by_the_primer():
    text = PRIMER.read_text(encoding="utf-8")
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in text]
    assert not missing, f"PRIMER.md does not embed: {missing}"


@pytest.mark.parametrize("delta", [-5e-4, 5e-4])
def test_cache_thresholds_sit_off_every_similarity(delta):
    """A threshold on the exact similarity of a probe (0.75, 0.875, 1.00 in float32) would make a count depend on the
    last bit of a dot product, so it could change from one CPU to another. A small move of every threshold of the
    chart changes no count."""
    from gwcore import cache

    at = cache.sweep_thresholds(mf.THRESHOLDS, guard=False) + cache.sweep_thresholds(mf.THRESHOLDS, guard=True)
    moved = [t + delta for t in mf.THRESHOLDS]
    near = cache.sweep_thresholds(moved, guard=False) + cache.sweep_thresholds(moved, guard=True)
    assert [(r["right"], r["wrong"]) for r in at] == [(r["right"], r["wrong"]) for r in near]
