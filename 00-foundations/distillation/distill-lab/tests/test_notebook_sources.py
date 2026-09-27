"""Every notebook source follows the lab contract: tier, one-minute version, 3-6 exercises each followed by a
check, and a design-review section; and every notebook is built."""
import re
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
SRC = LAB / "notebooks_src"
SOURCES = sorted(SRC.glob("*.py"))
NAMES = ["01_kd_on_a_tiny_transformer", "02_teacher_data_and_a_real_student", "03_distilling_reasoning_traces_for_real",
         "04_a_distilled_draft_in_vllm", "05_is_the_student_worth_it"]


def cells(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_the_five_notebooks():
    assert [p.stem for p in SOURCES] == NAMES
    for d in ("notebooks", "solutions"):
        assert sorted(p.stem for p in (LAB / d).glob("*.ipynb")) == NAMES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_contract(path):
    text = path.read_text()
    assert "**Tier:**" in text and "The one-minute version" in text and "In a design review" in text
    kinds = cells(text)
    exercises = [i for i, k in enumerate(kinds) if k == "exercise"]
    assert 3 <= len(exercises) <= 6
    assert all(kinds[i + 1] == "check" for i in exercises)
    assert text.count("### BEGIN SOLUTION") == text.count("### END SOLUTION") >= len(exercises)
    assert "✅" in text and "PRIMER" in text and "Drill 1" in text
