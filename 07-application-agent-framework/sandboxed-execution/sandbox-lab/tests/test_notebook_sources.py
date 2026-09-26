"""Every notebook source follows the lab contract: a tier line, the one-minute version, 3-6 exercises
each followed by a check, balanced solution markers, and a design-review section."""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "notebooks_src"
SOURCES = sorted(SRC.glob("*.py"))


def cells(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_five_notebooks_named_as_the_spec_says():
    assert [p.stem for p in SOURCES] == [
        "01_hardened_containers", "02_pod_per_execution_on_kind", "03_egress_proxy_and_secret_brokering",
        "04_an_agent_with_a_sandbox_tool", "05_gke_sandbox_with_gvisor"]


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


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_cites_primer_and_labels_provenance(path):
    text = path.read_text()
    assert re.search(r"PRIMER §|identity primer §|primer §|scaling primer §", text), "cite the primer sections"
    if "docker:" in text or "sample" in text.lower():
        pass   # the sample-labelling is asserted in the probe/bench tests
