"""Every notebook source follows the lab contract: tier, one-minute version, 3-6 exercises each
followed by a check, a design-review section; and the built notebooks are in sync with the sources."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
SRC = LAB / "notebooks_src"
SOURCES = sorted(SRC.glob("*.py"))
NAMES = ["01_quantize_a_checkpoint", "02_serve_and_compare_schemes", "03_measure_the_accuracy_cost",
         "04_kv_cache_quantization_in_vllm", "05_fp4_and_the_blackwell_path"]


def cells(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_the_five_notebooks_of_the_spec():
    assert [p.stem for p in SOURCES] == NAMES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_contract(path):
    text = path.read_text()
    assert "**Tier:**" in text and "The one-minute version" in text and "In a design review" in text
    assert "Drill 1." in text and "Drill 2." in text
    kinds = cells(text)
    exercises = [i for i, k in enumerate(kinds) if k == "exercise"]
    assert 3 <= len(exercises) <= 6
    assert all(kinds[i + 1] == "check" for i in exercises)
    assert text.count("### BEGIN SOLUTION") == text.count("### END SOLUTION") >= len(exercises)
    assert "✅" in text and "PRIMER" in text


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_labels_on_numbers_that_are_not_measurements(path):
    text = path.read_text()
    if "bench." in text or "B.compare" in text or "gemm_time" in text:
        assert "SIMULATED" in text or "simulated" in text
    if "SAMPLES" in text or "sample_results" in text:
        assert "sample output in the documented format (illustrative)" in text


def test_built_notebooks_match_sources(tmp_path):
    import nbformat
    for stem in NAMES:
        for d in ("notebooks", "solutions"):
            assert (LAB / d / f"{stem}.ipynb").exists(), f"run: python3 tools/build_notebooks.py ({d}/{stem})"
    sys.path.insert(0, str(LAB / "tools"))
    import build_notebooks as bn
    for src in SOURCES:
        want = [c for _, c in bn.parse(src.read_text())]
        nb = nbformat.read(LAB / "solutions" / f"{src.stem}.ipynb", as_version=4)
        have = [c.source for c in nb.cells[1:]]
        assert [bn.keep_solution(w) for w in want] == have, f"stale solutions/{src.stem}.ipynb: rebuild"
