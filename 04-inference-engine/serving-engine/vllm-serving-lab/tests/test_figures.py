"""The charts of ../../vllm-internals/vllm-internals-primer.md that servelab computes (tools/orchestration/FIGURE-STYLE.md §7).

Each chart in ../../vllm-internals/figures/ that ``tools/make_figures.py`` lists must be what it computes now with
``servelab.sizing``, and what tools/orchestration/svgchart.py renders from the spec stored in the file. If a preset or
an estimate changes (``sizing.GPUS``, the bundled config, ``sizing.overhead_estimate``, ``serve_scheduler_defaults``),
regenerate the charts (``python3 tools/make_figures.py``) and read the primer's captions again. Standard library +
pytest. The test skips when the vllm-internals primer is not in this checkout.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("servelab_make_figures", LAB / "tools" / "make_figures.py")
make_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_figures)
svgchart = make_figures.svgchart
PRIMER_PATH = make_figures.DOC_DIR / "vllm-internals-primer.md"
if not PRIMER_PATH.is_file():
    pytest.skip("04-inference-engine/vllm-internals not in this checkout", allow_module_level=True)
PRIMER = PRIMER_PATH.read_text(encoding="utf-8")


def check(name: str) -> None:
    svgchart.check_figure(make_figures.FIGURES_DIR / name, make_figures.FIGURES[name])


def test_concurrency_vs_max_model_len():
    check("concurrency-vs-max-model-len.svg")


def test_profiling_activations_vs_budget():
    check("profiling-activations-vs-budget.svg")


def test_every_chart_has_a_test():
    assert set(make_figures.FIGURES) == {"concurrency-vs-max-model-len.svg", "profiling-activations-vs-budget.svg"}


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_every_chart_is_referenced_by_the_primer(name):
    assert f"figures/{name}" in PRIMER, f"vllm-internals-primer.md does not embed figures/{name}"


@pytest.mark.parametrize("name", sorted(make_figures.FIGURES))
def test_the_alt_text_is_the_chart_title(name):
    title = make_figures.FIGURES[name]()["title"]
    assert f"![{title}](figures/{name})" in PRIMER, f"the alt text of figures/{name} is not the chart's title"


def test_the_marks_carry_the_primers_numbers():
    # the labels are the numbers as the primer prints them (§4.7), so that the reader finds the sentence
    labels = {m.get("label") for build in make_figures.FIGURES.values() for m in build()["marks"]}
    for text in ["4.62×", "2.31×", "1.15×", "55.95×", "13.99×", "3.50×", "auto: about 37.8k tokens",
                 "131,072: does not start", "L4, 2,048-token profile: 0.42 GiB",
                 "H100, 8,192-token profile: 1.67 GiB"]:
        assert text in labels, text
    for fragment in ["4.62×", "2.31×", "1.15×", "55.95×", "13.99×", "3.50×", "about 37.8k tokens",
                     "**does not start**", "0.42 GiB (2,048-token profile, 256-row sampler)",
                     "1.67 GiB (8,192-token profile, 1,024-row sampler)"]:
        assert fragment in PRIMER, fragment
