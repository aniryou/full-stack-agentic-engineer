"""The charts of ../PRIMER.md that fleetsim computes (tools/orchestration/FIGURE-STYLE.md §7).

Each chart in ../figures/ must be what ``tools/make_figures.py`` computes now with fleetsim, and what
tools/orchestration/svgchart.py renders from the spec stored in the file. If a function, a profile, a workload or a
seed changes (``L4_8B``, ``H100_8B``, ``search_pd``, ``pd_plan``, the ``kvtier`` formulas), regenerate the
charts (``python3 tools/make_figures.py``) and read the primer's captions again. Standard library + pytest.
"""
from __future__ import annotations

import functools
import importlib.util
import re
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("fleetsim_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart
PRIMER = (LAB.parent / "PRIMER.md").read_text(encoding="utf-8")
FLAT = re.sub(r"\s+", " ", PRIMER.replace("**", ""))          # the prose without line breaks or bold markers


@functools.lru_cache(maxsize=None)
def built(name: str) -> dict:
    """Each spec once per session: two of the three charts run the simulator."""
    return make_figures.FIGURES[name]()


def check(name: str) -> None:
    svgchart.check_figure(make_figures.FIGURES_DIR / name, lambda: built(name))


def test_signals_vs_load():
    check("signals-vs-load.svg")


def test_pd_split_slo_attainment():
    check("pd-split-slo-attainment.svg")


def test_fetch_or_recompute_vs_bandwidth():
    check("fetch-or-recompute-vs-bandwidth.svg")


def test_every_chart_has_a_test():
    assert set(make_figures.FIGURES) == {"signals-vs-load.svg", "pd-split-slo-attainment.svg",
                                         "fetch-or-recompute-vs-bandwidth.svg"}


def test_every_computed_chart_is_built():
    # every chart in ../figures/ that svgchart rendered (it stores its spec in <desc>) has a spec builder
    charts = {p.name for p in make_figures.FIGURES_DIR.glob("*.svg") if "<desc>" in p.read_text(encoding="utf-8")}
    assert charts == set(make_figures.FIGURES)


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_every_chart_is_referenced_by_the_primer(name):
    assert f"figures/{name}" in PRIMER, f"PRIMER.md does not embed figures/{name}"


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_the_alt_text_is_the_chart_title(name):
    assert f"![{built(name)['title']}](figures/{name})" in PRIMER, f"the alt text of figures/{name} is not its title"


def test_the_marks_carry_the_primers_numbers():
    # the labels are the numbers as the primer prints them, so that the reader finds the sentence
    labels = {m.get("label") or m.get("text") for name in make_figures.FIGURES for m in built(name)["marks"]}
    for text in ["0.2 req/s: 0.70", "0.5 req/s: 1.00", "3.5 req/s: KV 0.75",
                 "3P5D: 0.86", "aggregated: 0.68", "pd_plan: 2.75 replicas", "1P7D collapses: 106 s TTFT p95",
                 "7P1D collapses:", "2.4 s TPOT p95", "L4 break-even: 0.50 GB/s", "H100 break-even: 4.05 GB/s",
                 "recompute on an H100: 324 ms", "25 GbE: 439 ms", "host DRAM: 27 ms"]:
        assert text in labels, text


def test_every_mark_number_is_in_the_primer():
    # a later edit of the prose must not leave a mark whose number the text no longer prints: the part after the
    # last ": " ("0.86", "KV 0.75", "106 s TTFT p95") is in the prose verbatim; so is a text mark that carries a number
    # ("2.4 s TPOT p95"), unless it is the lead line of a two-line label ("7P1D collapses:")
    for name in make_figures.FIGURES:
        for m in built(name)["marks"]:
            label = m.get("label") or ""
            number = label.rsplit(": ", 1)[-1] if ": " in label else ""
            text = m.get("text") or ""
            if text and not text.endswith(":"):
                number = text
            if re.search(r"\d", number):
                assert number in FLAT, f"{name}: the mark {label!r}: PRIMER.md does not print {number!r}"


def test_the_tiers_are_notebook_05s():
    # the §6.2 chart's two tier points use the tiers that notebook 05 assumes; if the notebook changes them, so must
    # the chart (and the primer's 27 ms and 439 ms)
    src = (LAB / "notebooks_src" / "05_kv_cache_tiers_and_agent_sessions.py").read_text(encoding="utf-8")
    for tier in (make_figures.DRAM, make_figures.GBE_25):
        m = re.search(r'Tier\("' + re.escape(tier.name) + r'",\s*([\d_.]+),\s*([\d_.]+),\s*([\d_.]+)\)', src)
        assert m, f"notebook 05 has no tier named {tier.name!r}"
        assert tuple(float(g.replace("_", "")) for g in m.groups()) == (
            float(tier.capacity_gb), float(tier.read_gb_s), float(tier.latency_s)), tier.name
