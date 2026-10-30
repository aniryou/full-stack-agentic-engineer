"""The kv-cache primer's charts (tools/orchestration/FIGURE-STYLE.md §7): each committed SVG is what
tools/orchestration/svgchart.py renders from its own spec, and that spec is what tools/make_figures.py computes with
kerncore now. Regenerate with `python3 tools/make_figures.py` after a change to kerncore.kv or to a spec.
"""
import importlib.util
from pathlib import Path

import pytest

GEN_PATH = Path(__file__).resolve().parents[1] / "tools" / "make_figures.py"
_spec = importlib.util.spec_from_file_location("kerncore_make_figures", GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)
svgchart = gen.svgchart


def test_kv_bytes_vs_tokens():
    svgchart.check_figure(gen.FIG_DIR / "kv-bytes-vs-tokens.svg", gen.spec_kv_bytes_vs_tokens)


def test_decode_intensity():
    svgchart.check_figure(gen.FIG_DIR / "decode-intensity.svg", gen.spec_decode_intensity)


def test_sessions_per_gpu():
    svgchart.check_figure(gen.FIG_DIR / "sessions-per-gpu.svg", gen.spec_sessions_per_gpu)


def test_every_chart_has_a_test_and_a_file():
    tested = {"kv-bytes-vs-tokens.svg", "decode-intensity.svg", "sessions-per-gpu.svg"}
    assert set(gen.FIGURES) == tested
    for name in gen.FIGURES:
        assert (gen.FIG_DIR / name).is_file(), name


@pytest.mark.parametrize("name", sorted(gen.FIGURES))
def test_the_primer_references_every_chart(name):
    assert f"figures/{name}" in gen.DOC.read_text(encoding="utf-8"), f"{gen.DOC.name} does not show {name}"
