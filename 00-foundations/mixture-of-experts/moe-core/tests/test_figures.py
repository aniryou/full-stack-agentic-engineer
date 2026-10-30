"""The charts in ../figures/ are what tools/make_figures.py computes with moecore now (FIGURE-STYLE.md §7).

Each committed chart must be exactly what svgchart.render() makes of the spec stored in its <desc>, and that spec
must hold the values the core computes today (svgchart.check_figure()). If a formula, a config or a device preset
changes, a test here fails until the chart is regenerated: ``python3 tools/make_figures.py``. The primer must embed
every chart, and every decimal number a chart prints (its title, series names and marks) must stand in the primer
as printed, so that the reader finds the sentence.
"""
import importlib.util
import re
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("moecore_make_figures", CORE / "tools" / "make_figures.py")
mf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mf)

PRIMER_PATH = CORE.parent / "PRIMER.md"
PRIMER = " ".join(PRIMER_PATH.read_text(encoding="utf-8").split()) if PRIMER_PATH.exists() else ""
needs_primer = pytest.mark.skipif(not PRIMER, reason="PRIMER.md not beside this core")


def test_decode_step_bytes_vs_batch():
    mf.svgchart.check_figure(mf.FIG_DIR / "decode-step-bytes-vs-batch.svg", mf.spec_decode_step_bytes_vs_batch)


def test_experts_touched_vs_batch():
    mf.svgchart.check_figure(mf.FIG_DIR / "experts-touched-vs-batch.svg", mf.spec_experts_touched_vs_batch)


def test_dispatch_time_vs_tokens():
    mf.svgchart.check_figure(mf.FIG_DIR / "dispatch-time-vs-tokens.svg", mf.spec_dispatch_time_vs_tokens)


def test_every_chart_file_is_in_the_table():
    """FIGURES names exactly the chart files in ../figures/ (a computed chart keeps its spec in a <desc>; the
    hand-drawn SVGs have none), so that no chart file goes unchecked."""
    charts = {p.name for p in mf.FIG_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts and charts == set(mf.FIGURES), f"chart files {sorted(charts)} != FIGURES {sorted(mf.FIGURES)}"


@needs_primer
def test_every_chart_is_embedded_in_the_primer():
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in PRIMER]
    assert not missing, f"PRIMER.md does not embed: {missing}"


# A decimal number, with its × when it has one: a bare integer such as "8" or "64" is somewhere in any primer.
NUMBER = re.compile(r"\d[\d,]*\.\d+×?")


def _printed(spec: dict) -> list[str]:
    """Every text the chart prints that can carry a number: the title, the series names, the mark labels."""
    texts = [spec.get("title", "")]
    texts += [s.get("name", "") for s in spec.get("series", [])]
    texts += [m.get("label") or m.get("text") or "" for m in spec.get("marks", [])]
    return texts


@needs_primer
@pytest.mark.parametrize("name", sorted(mf.FIGURES))
def test_every_number_a_chart_prints_is_in_the_primer(name):
    missing = [num for text in _printed(mf.FIGURES[name]()) for num in NUMBER.findall(text) if num not in PRIMER]
    assert not missing, f"numbers on {name} that PRIMER.md does not print: {missing}"
