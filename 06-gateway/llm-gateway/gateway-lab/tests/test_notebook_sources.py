"""Every notebook source follows the lab contract: the names the curriculum links to, a tier line, the one-minute
version, 3-6 exercises each followed by a check, a design-review section, and honest labels."""
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "notebooks_src"
SOURCES = sorted(SRC.glob("*.py"))
NAMES = ["01_a_gateway_over_http", "02_outages_fallbacks_and_breakers", "03_semantic_cache_vs_the_prefix_cache",
         "04_streaming_limits_metering_and_chargeback", "05_guardrails_and_mcp_authorization_over_http"]


def kinds(text):
    return [m.group(1).strip() for m in re.finditer(r"^# %%(.*)$", text, re.M)]


def test_the_five_notebooks_the_curriculum_names():
    assert [p.stem for p in SOURCES] == NAMES


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_notebook_contract(path):
    text = path.read_text()
    assert "**Tier:** T0" in text and "The one-minute version" in text and "In a design review" in text
    k = kinds(text)
    ex = [i for i, x in enumerate(k) if x == "exercise"]
    assert 3 <= len(ex) <= 6 and all(k[i + 1] == "check" for i in ex)
    assert text.count("### BEGIN SOLUTION") == text.count("### END SOLUTION") >= len(ex)
    assert "✅" in text and "simulated" in text.lower()
    assert "PRIMER" in text


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.stem)
def test_t1_paths_detect_and_skip(path):
    text = path.read_text()
    if "GWLAB_VLLM_URL" in text:
        assert 'if tiers["vllm_url"]:' in text and "T1 skipped" in text
    assert "import torch" not in text
