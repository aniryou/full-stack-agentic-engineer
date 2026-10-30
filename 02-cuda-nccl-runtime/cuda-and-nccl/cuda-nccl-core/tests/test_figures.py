"""The primer's charts are what gpusim computes (tools/orchestration/FIGURE-STYLE.md §7).

Each chart in ../figures/ carries its spec. The test fails when the file is not what svgchart renders from that
spec, or when the spec is not what tools/make_figures.py computes now with gpusim. Regenerate the charts with
``python3 tools/make_figures.py``. Standard library + pytest.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

CORE = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cuda_nccl_core_make_figures", CORE / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_figures)

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_chart_is_what_the_core_computes(name):
    make_figures.svgchart.check_figure(make_figures.FIGDIR / name, make_figures.FIGURES[name])


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_primer_embeds_the_chart_with_its_title_as_the_alt_text(name):
    text = make_figures.PRIMER.read_text(encoding="utf-8")
    assert f"figures/{name}" in text, f"PRIMER.md does not reference figures/{name}"
    title = make_figures.FIGURES[name]()["title"]
    assert f"![{title}](figures/{name})" in text, f"the alt text of figures/{name} is not the chart's title"


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_every_number_in_a_mark_label_is_printed_in_the_primer(name):
    """A mark is labelled with the number as the primer prints it, so that the reader finds the sentence."""
    text = make_figures.PRIMER.read_text(encoding="utf-8")
    for mark in make_figures.FIGURES[name]().get("marks", []):
        label = mark.get("label") or mark.get("text") or ""
        for number in NUMBER.findall(label):
            assert number in text, f"{name}: {label!r} has {number}, which PRIMER.md does not print"
