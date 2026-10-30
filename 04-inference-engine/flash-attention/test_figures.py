"""Pins the charts of flash-attention-deep-dive.md to fa_calculators.py (tools/orchestration/FIGURE-STYLE.md §7).

Each committed chart must be what svgchart.render() makes of the spec stored in it, and that spec must be what
make_figures.py computes with fa_calculators now. Regenerate with: python3 make_figures.py
Run: python -m pytest -q test_figures.py (standard library + numpy + pytest).
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


mf = _load(HERE / "make_figures.py", "fa_make_figures")
svgchart = mf.svgchart
DOC = mf.DOC.read_text(encoding="utf-8")


def test_naive_roofline():
    svgchart.check_figure(mf.FIG_DIR / "deep-dive-naive-roofline.svg", mf.spec_naive_roofline)


def test_hbm_bytes_vs_n():
    svgchart.check_figure(mf.FIG_DIR / "deep-dive-hbm-bytes-vs-n.svg", mf.spec_hbm_bytes_vs_n)


def test_clocks_per_score():
    svgchart.check_figure(mf.FIG_DIR / "deep-dive-clocks-per-score.svg", mf.spec_clocks_per_score)


def test_decode_kv_share():
    svgchart.check_figure(mf.FIG_DIR / "deep-dive-decode-kv-share.svg", mf.spec_decode_kv_share)


def test_every_chart_has_a_test_and_the_deep_dive_embeds_it():
    tested = {"deep-dive-naive-roofline.svg", "deep-dive-hbm-bytes-vs-n.svg", "deep-dive-clocks-per-score.svg",
              "deep-dive-decode-kv-share.svg"}
    assert set(mf.FIGURES) == tested
    missing = [name for name in mf.FIGURES if f"figures/{name}" not in DOC]
    assert not missing, f"flash-attention-deep-dive.md does not reference: {missing}"


# Each labelled mark, and the fragment of the deep dive that prints the same number.
MARKS = {
    "deep-dive-naive-roofline.svg": {
        "naive, three kernels: 62.1 FLOP/B": "(62.1 at $N$ = 4,096",
        "compulsory traffic: 2,040 FLOP/B": "These are intensities of 2,040 and 16,320 FLOP/B",
        "H100: 208 TFLOP/s (21% of peak)": "208 TFLOP/s (21% of peak)",
        "L4: 18.6 TFLOP/s (15% of peak)": "18.6 TFLOP/s (15% of peak)",
        "ridge 295": "**295 FLOP/B**",
        "ridge 403": "121 ÷ 0.30 = **403**",
        "8.7 µs (H100)": "8.7 µs (H100)",
        "71 µs (L4)": "71 µs (L4)",
    },
    "deep-dive-hbm-bytes-vs-n.svg": {
        "138.4 MB": "| 138.4 MB / 62 | 8,623 MB / 64 |",
        "8,623 MB": "| 138.4 MB / 62 | 8,623 MB / 64 |",
        "69.2 MB": "| 69.2 MB / 124 | 4,312 MB / 127 |",
        "4,312 MB": "| 69.2 MB / 124 | 4,312 MB / 127 |",
        "4.2 MB": "| 4.2 MB / 2,040 | 33.7 MB / 16,320 |",
        "33.7 MB": "| 4.2 MB / 2,040 | 33.7 MB / 16,320 |",
    },
    "deep-dive-decode-kv-share.svg": {
        "KV bytes = weight bytes: ≈ 122,500 tokens": "KV bytes equal weight bytes at approximately 122,500 cached tokens",
        "1 × 2k: 1.6%": "| 1 × 2k | 2,048 | 0.27 GB | 0.08 ms | 1.6% |",
        "1 × 32k: 21%": "| 1 × 32k | 32,768 | 4.29 GB | 1.28 ms | 21% |",
        "32 × 2k: 35%": "| 32 × 2k | 65,536 | 8.59 GB | 2.56 ms | 35% |",
        "1 × 128k: 52%": "| 1 × 128k | 131,072 | 17.2 GB | 5.13 ms | 52% |",
        "8 × 32k: 68%": "| 8 × 32k | 262,144 | 34.4 GB | 10.26 ms | 68% |",
    },
}


@pytest.mark.parametrize("name", sorted(MARKS))
def test_the_marks_are_the_deep_dive_numbers(name):
    """Every labelled mark of a chart is one the deep dive prints, and the deep dive still prints it there."""
    spec = mf.FIGURES[name]()
    labels = {m["label"] for m in spec["marks"] if m.get("label")}
    assert labels == set(MARKS[name])
    for label, fragment in MARKS[name].items():
        assert fragment in DOC, f"{name}: the deep dive no longer prints {fragment!r}"


def test_the_clocks_chart_shows_the_section_4_3_table():
    """The bars are the §4.3 table, and the EX2 share under each GPU is the table's percentage."""
    spec = mf.spec_clocks_per_score()
    bars = {s["name"]: s["y"] for s in spec["series"]}
    assert bars["tensor cores"] == [0.25, 0.125, 0.0625]
    assert bars["EX2 on MUFU"] == [0.0625, 0.0625, 0.0625]
    assert bars["about 5 FP32 instructions"] == [0.078, 0.039, 0.039]
    assert [t.split("\n")[1] for t in spec["x"]["tick_labels"]] == [
        "EX2 = 25% of MMA time", "EX2 = 50% of MMA time", "EX2 = 100% of MMA time"]
    assert "| tensor cores | 0.25 | 0.125 | 0.0625 |" in DOC
    assert "0.0625 (25% of MMA time) | 0.0625 (**50%**) | 0.0625 (**100%**) |" in DOC
