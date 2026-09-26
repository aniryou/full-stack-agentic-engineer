"""Every notebook source follows the lab contract: tier, one-minute version, 3-6 exercises each followed
by a check, a design-review section, and T0 paths that label what they print."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "notebooks_src").glob("*.py"))
NAMES = ["01_a_tiny_moe_in_torch", "02_watch_the_router", "03_batch_vs_weight_stream",
         "04_expert_parallelism_on_two_gpus", "05_moe_on_a_small_gpu"]


def kinds(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_the_five_notebooks_of_the_spec():
    assert [p.stem for p in SOURCES] == NAMES
    for d in ("notebooks", "solutions"):
        assert sorted(p.stem for p in (ROOT / d).glob("*.ipynb")) == NAMES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_contract(path):
    text = path.read_text()
    assert "**Tier:**" in text and "The one-minute version" in text and "In a design review" in text
    k = kinds(text)
    ex = [i for i, v in enumerate(k) if v == "exercise"]
    assert 3 <= len(ex) <= 6 and all(k[i + 1] == "check" for i in ex)
    assert text.count("### BEGIN SOLUTION") == text.count("### END SOLUTION") >= len(ex)
    assert "✅" in text and "PRIMER" in text
    assert text.count("**Drill") >= 2
    assert any(w in text.lower() for w in ("simulated", "illustrative", "recorded"))


def test_torch_is_only_imported_behind_a_check():
    for p in SOURCES:
        for line in p.read_text().splitlines():
            if re.match(r"^(import torch|from torch)", line):
                pytest.fail(f"{p.name}: top-level torch import breaks the numpy-only path: {line}")
