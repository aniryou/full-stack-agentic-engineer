"""Every notebook source follows the lab contract: tier, one-minute version, 3-6 exercises each
followed by a check, and a design-review section."""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "notebooks_src"
SOURCES = sorted(SRC.glob("*.py"))


def cells(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_six_notebooks():
    assert [p.stem[:2] for p in SOURCES] == ["01", "02", "03", "04", "05", "06"]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_contract(path):
    text = path.read_text()
    assert "**Tier:**" in text and "The one-minute version" in text and "In a design review" in text
    kinds = cells(text)
    exercises = [i for i, k in enumerate(kinds) if k == "exercise"]
    assert 3 <= len(exercises) <= 6
    assert all(kinds[i + 1] == "check" for i in exercises)
    assert text.count("### BEGIN SOLUTION") == text.count("### END SOLUTION") >= len(exercises)
    assert "✅" in text
