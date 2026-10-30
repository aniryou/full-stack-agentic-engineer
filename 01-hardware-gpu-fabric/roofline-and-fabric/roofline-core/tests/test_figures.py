"""The charts of the topic's PRIMER.md are what this core computes now (tools/orchestration/FIGURE-STYLE.md §7).

``tools/make_figures.py`` builds each chart's spec with the core's own functions and renders it with
``tools/orchestration/svgchart.py``. Each test here recomputes one spec and compares it with the committed SVG: a
change to the core, to a device or model preset, or a hand edit of the file fails it until the chart is regenerated
(``python3 tools/make_figures.py``). Standard library + pytest.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("roofline_core_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart


def test_roofline_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "roofline-h100-bf16.svg", make_figures.spec_roofline)


def test_decode_step_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "decode-step-weights-vs-kv.svg", make_figures.spec_decode_step)


def test_allreduce_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "allreduce-time-vs-message-size.svg", make_figures.spec_allreduce)


def test_checkpoint_waste_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "checkpoint-waste-vs-interval.svg",
                          make_figures.spec_checkpoint_waste)


def test_every_chart_is_in_the_table_and_referenced_by_the_primer():
    assert set(make_figures.FIGURES) == {"roofline-h100-bf16.svg", "decode-step-weights-vs-kv.svg",
                                         "allreduce-time-vs-message-size.svg", "checkpoint-waste-vs-interval.svg"}
    text = make_figures.DOC.read_text(encoding="utf-8")
    missing = [name for name in make_figures.FIGURES if f"figures/{name}" not in text]
    assert not missing, f"PRIMER.md does not embed: {missing}"
