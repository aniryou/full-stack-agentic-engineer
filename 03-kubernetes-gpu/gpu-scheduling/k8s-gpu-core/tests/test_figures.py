"""The charts of the topic's PRIMER.md are what this core computes now (tools/orchestration/FIGURE-STYLE.md §7).

``tools/make_figures.py`` builds each chart's spec with the core's own functions and renders it with
``tools/orchestration/svgchart.py``. Each test here recomputes one spec and compares it with the committed SVG: a
change to the scheduler, the autoscaler simulator or an input the primer names, or a hand edit of the file, fails it
until the chart is regenerated (``python3 tools/make_figures.py``). Standard library + pytest.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
if not any((p / "tools" / "orchestration" / "svgchart.py").is_file() for p in LAB.parents):
    pytest.skip("tools/orchestration/svgchart.py not found (the core copied out of the repo)", allow_module_level=True)
_SPEC = importlib.util.spec_from_file_location("k8s_gpu_core_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart


def test_stranded_gpus_as_pods_arrive_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "stranded-gpus-as-pods-arrive.svg",
                          make_figures.spec_stranded_arrivals)


def test_spot_gang_wall_clock_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "spot-gang-wall-clock.svg", make_figures.spec_spot_gang)


def test_queued_provisioning_billed_nodes_chart():
    svgchart.check_figure(make_figures.FIG_DIR / "queued-provisioning-billed-nodes.svg",
                          make_figures.spec_queued_provisioning)


def test_the_marks_are_the_primers_numbers():
    text = " ".join(make_figures.DOC.read_text(encoding="utf-8").split())     # the prose wraps its lines
    stranded = make_figures.spec_stranded_arrivals()
    assert [m["label"] for m in stranded["marks"] if m["kind"] == "point"] == ["16 stranded, 100%", "0 stranded, 0%"]
    assert "The values are 16 and 100% on the left, and 0 and 0% on the right." in text
    spot = make_figures.spec_spot_gang()
    for mark in (m for m in spot["marks"] if m["kind"] == "point"):
        hours, survives = mark["label"].split(": ")[1].split(", survives ")
        assert f"| {mark['x']} | {survives} | {hours} |" in text           # the row of the §7.2 table
    queued = make_figures.spec_queued_provisioning()
    labels = [m.get("label") or m.get("text") for m in queued["marks"]]
    assert labels[0] == "gang start: 2.47 h" and "start after 2.47 h with either pool" in text
    assert labels[1] == "ordinary pool: 31.3 node-hours (251 GPU-hours)"
    assert "**31.3 node-hours (251 GPU-hours)**" in text and labels[2] == "queued pool: 1.3 node-hours"
    ordinary = queued["series"][0]
    area = sum(y * (x1 - x0) for x0, x1, y in zip(ordinary["x"], ordinary["x"][1:], ordinary["y"]))
    assert area == pytest.approx(31.333, abs=1e-3)                     # the area under the step line is the cost


def test_every_chart_is_in_the_table_and_referenced_by_the_primer():
    assert set(make_figures.FIGURES) == {"stranded-gpus-as-pods-arrive.svg", "spot-gang-wall-clock.svg",
                                         "queued-provisioning-billed-nodes.svg"}
    text = make_figures.DOC.read_text(encoding="utf-8")
    missing = [name for name in make_figures.FIGURES if f"figures/{name}" not in text]
    assert not missing, f"PRIMER.md does not embed: {missing}"
