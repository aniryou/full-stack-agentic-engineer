"""The charts of docs/01-scaling-primer.md are what scalelab.capacity computes (tools/orchestration/FIGURE-STYLE.md §7).

Each committed SVG must be what tools/orchestration/svgchart.py renders from its own spec, and that spec must be what
the builder in tools/make_figures.py computes now. After a change to the core, regenerate the charts with
``python3 tools/make_figures.py``.
"""

import importlib.util
import pathlib
import re

LAB = pathlib.Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("scalelab_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart
PRIMER = LAB / "docs" / "01-scaling-primer.md"


def _check(name):
    svgchart.check_figure(make_figures.FIG_DIR / name, make_figures.FIGURES[name])


def test_inflight_turns_chart():
    _check("inflight-turns-littles-law.svg")


def test_cost_per_conversation_chart():
    _check("cost-per-conversation-flash-lite-share.svg")


def test_pt_break_even_chart():
    _check("pt-break-even-utilisation.svg")


def test_every_chart_is_referenced_by_the_primer():
    text = PRIMER.read_text(encoding="utf-8")
    for name in make_figures.FIGURES:
        assert f"figures/{name}" in text, name


def _mark(spec, label):
    return next(m for m in spec["marks"] if label in (m.get("label"), m.get("text")))


def test_the_labelled_points_are_the_primer_numbers():
    """Each label prints a number the primer prints, and the computed value rounds to it."""
    text = PRIMER.read_text(encoding="utf-8")
    for build in make_figures.FIGURES.values():
        for m in build()["marks"]:
            for number in re.findall(r"\d[\d,.]*", m.get("label") or m.get("text") or ""):
                assert number.rstrip(".") in text, (m, number)
    little = make_figures.spec_inflight_turns()
    assert round(_mark(little, "6-second turn: 125")["y"]) == 125
    assert round(_mark(little, "12-second turn: 250")["y"]) == 250
    assert round(_mark(little, "about 85 in flight")["y"]) == 85
    assert round(_mark(little, "10 M TPM sustains ≈ 14 turns/s")["x"]) == 14
    cost = make_figures.spec_cost_per_conversation()
    assert round(_mark(cost, "all 3.5 Flash: $0.141")["y"], 3) == 0.141
    assert round(_mark(cost, "prefix cached: $0.092")["y"], 3) == 0.092
    assert round(_mark(cost, "35 % on Flash-Lite: $0.068")["y"], 3) == 0.068
    pt = make_figures.spec_pt_break_even()
    assert round(_mark(pt, "1 year: 75 %")["x"]) == 75
    assert round(_mark(pt, "3 months: 90 %")["x"]) == 90
    assert round(_mark(pt, "PT for the peak: 33 % utilised")["x"]) == 33
    assert round(_mark(pt, "PayGo ($1.50/M burndown)")["y"] - 0.1, 2) == 1.50
