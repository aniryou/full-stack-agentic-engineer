"""Pins the charts of PRIMER.md to capacity.py (tools/orchestration/FIGURE-STYLE.md §7).

Each committed chart must be what svgchart.render() makes of the spec stored in it, and that spec must be what
make_figures.py computes with capacity.py now. Regenerate with: python3 make_figures.py
Run: python -m pytest -q test_figures.py (standard library + pytest).
"""
import importlib.util
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent


def _load(path, name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mf = _load(HERE / "make_figures.py", "capacity_make_figures")
svgchart = mf.svgchart
PRIMER = (HERE / "PRIMER.md").read_text(encoding="utf-8")


def test_ttft_vs_prompt():
    svgchart.check_figure(mf.FIG_DIR / "ttft-vs-prompt.svg", mf.spec_ttft_vs_prompt)


def test_sessions_vs_context():
    svgchart.check_figure(mf.FIG_DIR / "sessions-vs-context.svg", mf.spec_sessions_vs_context)


def test_decode_vs_batch():
    svgchart.check_figure(mf.FIG_DIR / "decode-vs-batch.svg", mf.spec_decode_vs_batch)


def test_every_chart_has_a_test_and_the_primer_embeds_it():
    tested = {"ttft-vs-prompt.svg", "sessions-vs-context.svg", "decode-vs-batch.svg"}
    assert set(mf.FIGURES) == tested
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in PRIMER]
    assert not missing, f"PRIMER.md does not reference: {missing}"


@pytest.mark.parametrize("name", sorted(mf.FIGURES))
def test_the_marks_are_the_primer_numbers(name):
    """Each labelled point sits on the value the primer quotes (the same rounding as test_capacity.py)."""
    spec = mf.FIGURES[name]()
    points = {m["label"]: m["y"] for m in spec["marks"] if m["kind"] == "point"}
    expected = {
        "ttft-vs-prompt.svg": {"2K prompt: ~0.1 s": (0.097, 3), "32K RAG prompt: ~1.9 s": (1.89, 2)},
        "sessions-vs-context.svg": {"fp8: ~355 sessions, 1 GPU": (355.1, 1), "bf16: ~89 sessions, 2 GPUs": (88.8, 1)},
        "decode-vs-batch.svg": {"~70 tok/s single-stream ceiling": (70, 0), "~100 sessions in fp8: ~9,000 tok/s": (8929, 0)},
    }[name]
    assert {label: round(points[label], nd) for label, (_, nd) in expected.items()} == {k: v for k, (v, _) in expected.items()}
