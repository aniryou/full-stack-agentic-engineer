"""Every notebook source follows the lab contract: a tier line, the one-minute version, 3-6 exercises each
followed by a check, balanced solution markers, primer citations, and a design-review section with drills."""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "notebooks_src"
SOURCES = sorted(SRC.glob("*.py"))


def kinds(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_five_notebooks_named_as_the_spec_says():
    assert [p.stem for p in SOURCES] == [
        "01_a_memory_store_on_sqlite", "02_a_memory_service_and_an_agent", "03_memory_layouts_and_the_prefix_cache",
        "04_consolidation_as_a_scheduled_job", "05_evaluate_forget_and_audit"]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_contract(path):
    text = path.read_text()
    assert "**Tier:** T0" in text and "## The one-minute version" in text and "## In a design review" in text
    k = kinds(text)
    ex = [i for i, x in enumerate(k) if x == "exercise"]
    assert 3 <= len(ex) <= 6 and all(k[i + 1] == "check" for i in ex)
    assert text.count("### BEGIN SOLUTION") == text.count("### END SOLUTION") >= len(ex)
    assert text.count("✅") >= len(ex)
    assert len(re.findall(r"\*\*Drill \d\.\*\*", text)) >= 2


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_cites_the_primer_and_labels_provenance(path):
    text = path.read_text()
    assert re.search(r"PRIMER §\d", text) and "../../PRIMER.md" in text
    if "FakeLLMServer" in text:
        assert "SIMULATED" in text or "simulated" in text
    if "samples" in text:
        assert "illustrative" in text


def test_no_emojis_except_the_check_mark():
    for p in SOURCES:
        extra = {c for c in p.read_text() if ord(c) > 0x2700 and c not in "✅"}
        assert not extra, (p.name, extra)
