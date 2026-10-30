"""tools/orchestration/svgchart.py: the chart helper the cores draw their pinned charts with (FIGURE-STYLE.md §7).

Standard library only.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("svgchart", ROOT / "tools" / "orchestration" / "svgchart.py")
svgchart = importlib.util.module_from_spec(SPEC)
sys.modules["svgchart"] = svgchart
SPEC.loader.exec_module(svgchart)


def test_demo_renders_deterministically_and_round_trips():
    spec = svgchart.demo_spec()
    a = svgchart.render(spec)
    b = svgchart.render(svgchart.demo_spec())
    assert a == b
    assert a.endswith("</svg>\n")
    stored = svgchart.read_spec(a)
    assert stored == spec
    assert svgchart.render(stored) == a


def test_output_is_well_formed_svg_in_the_house_style_and_fits_one_screen():
    root = ET.fromstring(svgchart.render(svgchart.demo_spec()))
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    w, h = float(root.get("width")), float(root.get("height"))
    assert 720 <= w <= 1000 and h <= 560                     # FIGURE-STYLE.md §3 and §4
    assert root.get("role") == "img" and root.get("aria-label")
    title = root.find("{http://www.w3.org/2000/svg}title")
    assert title is not None and title.text == root.get("aria-label")
    card = root.find("{http://www.w3.org/2000/svg}rect")
    assert card.get("fill") == "#161E1C"
    text = ET.tostring(root, encoding="unicode")
    assert "http" not in text.replace("http://www.w3.org/2000/svg", "")   # no external resource
    assert "<script" not in text and "<image" not in text


def test_check_figure_accepts_the_file_and_rejects_a_changed_value(tmp_path):
    p = svgchart.write_figure(tmp_path / "demo.svg", svgchart.demo_spec())
    svgchart.check_figure(p, svgchart.demo_spec)

    def other():
        s = svgchart.demo_spec()
        s["series"][0]["y"][3] += 0.5
        return s

    with pytest.raises(AssertionError, match="differs from what the core computes"):
        svgchart.check_figure(p, other)
    p.write_text(p.read_text(encoding="utf-8").replace('rx="8"', 'rx="9"', 1), encoding="utf-8")
    with pytest.raises(AssertionError, match="not what svgchart.render"):
        svgchart.check_figure(p, svgchart.demo_spec)


def test_tolerance_absorbs_last_bit_differences(tmp_path):
    p = svgchart.write_figure(tmp_path / "demo.svg", svgchart.demo_spec())

    def jitter():
        s = svgchart.demo_spec()
        s["series"][0]["y"] = [v * (1 + 1e-9) for v in s["series"][0]["y"]]
        return s

    svgchart.check_figure(p, jitter)


def test_bar_chart_log_axes_and_marks():
    bar = {"kind": "bar", "title": "Bars.", "categories": ["a", "b\nc"], "x": {"label": "x"}, "y": {"label": "y", "scale": "log"},
           "series": [{"name": "s", "y": [1, 1000], "style": "warm"}], "marks": [{"kind": "hline", "y": 10, "label": "ten"}]}
    text = svgchart.render(bar)
    ET.fromstring(text)
    assert "ten" in text and "<tspan" in text
    line = {"kind": "line", "title": "Line.", "x": {"scale": "log", "ticks": [1, 10, 100]}, "y": {"min": 0, "max": 1},
            "series": [{"name": "s", "x": [1, 10, 100], "y": [0.1, 0.5, 0.9], "step": True, "fill": True, "markers": True, "labels": True}],
            "marks": [{"kind": "band", "x0": 1, "x1": 10, "label": "band"}, {"kind": "vline", "x": 10, "label": "v"},
                      {"kind": "point", "x": 10, "y": 0.5, "label": "p"}, {"kind": "text", "x": 50, "y": 0.2, "text": "t"}],
            "legend": "bottom-right", "note": "n"}
    text = svgchart.render(line)
    ET.fromstring(text)
    for s in ("band", ">v<", ">p<", ">t<", ">n<"):
        assert s in text


def test_bad_specs_are_rejected():
    with pytest.raises(ValueError):
        svgchart.render({"kind": "pie", "title": "x", "series": [{"y": [1]}]})
    with pytest.raises(ValueError):
        svgchart.render({"kind": "line", "title": "x", "height": 700, "series": [{"x": [1], "y": [1]}]})
    with pytest.raises(ValueError):
        svgchart.render({"kind": "line", "title": "x", "series": [{"x": [1, 2], "y": [1]}]})
    with pytest.raises(ValueError):
        svgchart.render({"kind": "line", "title": "x", "x": {"scale": "log"}, "series": [{"x": [0, 1], "y": [1, 2]}]})
    with pytest.raises(ValueError):
        svgchart.render({"kind": "line", "title": "x", "series": [{"x": [1], "y": [1], "style": "red"}]})


def test_tick_helpers():
    assert svgchart.nice_ticks(0, 10) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    assert svgchart.nice_ticks(0, 1) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert svgchart.log_ticks(1, 1000) == [1.0, 10.0, 100.0, 1000.0]
    assert svgchart.log_ticks(1, 50) == [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    assert svgchart.fmt_num(1000) == "1,000" and svgchart.fmt_num(2.5) == "2.5" and svgchart.fmt_num(0.123456) == "0.123"
