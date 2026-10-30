"""The charts in ../figures/ are what tools/make_figures.py computes with distillcore now (FIGURE-STYLE.md §7).

Each committed chart must be exactly what svgchart.render() makes of the spec stored in its <desc>, and that spec
must hold the values the core computes today (svgchart.check_figure()). If a formula or an input changes, a test
here fails until the chart is regenerated: ``python3 tools/make_figures.py``. The primer must embed every chart,
and every number a mark prints must stand in the primer as printed, so that the reader finds the sentence.
"""
import importlib.util
import re
from pathlib import Path

CORE = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("distillcore_make_figures", CORE / "tools" / "make_figures.py")
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)

PRIMER = " ".join((CORE.parent / "PRIMER.md").read_text(encoding="utf-8").split())


def check(name: str) -> None:
    mf.svgchart.check_figure(mf.FIG_DIR / name, mf.FIGURES[name])


def test_kl_against_temperature():
    check("kl-against-temperature.svg")


def test_draft_speedup():
    check("draft-speedup.svg")


def test_break_even():
    check("break-even.svg")


def test_every_chart_is_embedded_in_the_primer():
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in PRIMER]
    assert not missing, f"PRIMER.md does not embed: {missing}"


def test_every_number_a_mark_prints_is_in_the_primer():
    number = re.compile(r"\$?\d[\d,]*(?:\.\d+)?×?")
    missing = []
    for name, build in mf.FIGURES.items():
        for mark in build().get("marks", []):
            for num in number.findall(mark.get("label") or mark.get("text") or ""):
                if num not in PRIMER:
                    missing.append(f"{name}: {num}")
    assert not missing, f"numbers on the charts that PRIMER.md does not print: {missing}"
