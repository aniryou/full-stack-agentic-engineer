#!/usr/bin/env python3
"""make_figures.py - the charts of ../../vllm-internals/vllm-internals-primer.md that servelab computes
(tools/orchestration/FIGURE-STYLE.md §7).

Each ``spec_<name>()`` computes its values with ``servelab.sizing`` from the inputs the primer names in §4.7 (the
bundled Llama-3.1-8B-Instruct config, BF16, the L4 and H100 80 GB rows of ``sizing.GPUS``, vLLM v0.30.0's defaults
as ``sizing.size`` models them: ``gpu_memory_utilization`` 0.92, 16-token blocks, and the per-GPU scheduler defaults
2,048 / 256 and 8,192 / 1,024), and returns a chart spec for tools/orchestration/svgchart.py. The marks are the
primer's pinned points, labelled as the primer prints them. ``tests/test_figures.py`` pins every chart with
``svgchart.check_figure``. The non-KV overheads are the lab's ESTIMATES (``sizing.overhead_estimate``), never a
measurement; a real ``Available KV cache memory`` log line replaces them.

    python3 tools/make_figures.py        # (re)writes ../../vllm-internals/figures/<name>.svg for every entry of FIGURES
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
ROOT = next(p for p in LAB.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file())
if "svgchart" in sys.modules:
    svgchart = sys.modules["svgchart"]
else:
    _spec = importlib.util.spec_from_file_location("svgchart", ROOT / "tools" / "orchestration" / "svgchart.py")
    svgchart = importlib.util.module_from_spec(_spec)
    sys.modules["svgchart"] = svgchart
    _spec.loader.exec_module(svgchart)

from servelab import sizing  # noqa: E402

DOC_DIR = ROOT / "04-inference-engine" / "vllm-internals"       # the primer these charts belong to
FIGURES_DIR = DOC_DIR / "figures"

MODEL = "llama-3.1-8b-instruct"      # the worked budget of §4.7: BF16 weights and BF16 KV
L4, H100 = "L4", "H100-80GB"         # the two columns of the §4.7 table
GiB = sizing.GiB


def _log2_grid(lo: int, hi: int, per_doubling: int, extra=(), multiple: int = 1) -> list[int]:
    """Integers from lo to hi, `per_doubling` of them per factor of two (rounded to `multiple`), plus `extra`."""
    pts, i = set(extra), 0
    while True:
        v = round(lo * 2 ** (i / per_doubling) / multiple) * multiple
        if v > hi:
            break
        pts.add(v)
        i += 1
    return sorted(p for p in pts | {lo, hi} if lo <= p <= hi)


def spec_concurrency_vs_max_model_len() -> dict:
    """§4.7: vLLM's "Maximum concurrency" (num_blocks / ceil(max_model_len / 16)) against max_model_len."""
    table = [8192, 16384, 32768, 131072]                    # the lengths of the §4.7 table and its repair list
    auto = sizing.size(MODEL, L4).estimated_max_model_len    # what `--max-model-len auto` approaches on the L4
    lengths = _log2_grid(4096, 131072, 4, table + [auto], multiple=16)

    def conc(gpu: str, length: int) -> float:
        return round(sizing.size(MODEL, gpu, max_model_len=length).max_concurrency, 4)

    l4 = {n: conc(L4, n) for n in lengths}
    h100 = {n: conc(H100, n) for n in lengths}
    default_len = sizing.load_config(MODEL).max_position_embeddings   # 131,072 when --max-model-len is unset
    default_l4 = sizing.size(MODEL, L4)
    assert not default_l4.fits and default_len == 131072
    assert sizing.size(MODEL, L4, max_model_len=auto).max_concurrency == 1.0

    def x(c: float) -> str:
        return f"{c:.2f}×"

    marks = [
        {"kind": "hline", "y": 1, "label": "1×: one request of max_model_len must fit", "style": "warm",
         "anchor": "start"},
        {"kind": "point", "x": 8192, "y": l4[8192], "label": x(l4[8192]), "style": "accent", "anchor": "end",
         "dy": 16},
        {"kind": "point", "x": 16384, "y": l4[16384], "label": x(l4[16384]), "style": "accent", "anchor": "end",
         "dy": 16},
        {"kind": "point", "x": 32768, "y": l4[32768], "label": x(l4[32768]), "style": "accent", "anchor": "end",
         "dy": 16},
        {"kind": "point", "x": auto, "y": l4[auto], "label": f"auto: about {auto / 1000:.1f}k tokens",
         "style": "accent", "anchor": "start", "dy": -10},
        {"kind": "point", "x": default_len, "y": l4[default_len], "label": f"{default_len:,}: does not start",
         "style": "warm", "anchor": "end", "dy": 18},
        {"kind": "point", "x": 8192, "y": h100[8192], "label": x(h100[8192]), "style": "violet", "anchor": "start",
         "dy": -10},
        {"kind": "point", "x": 32768, "y": h100[32768], "label": x(h100[32768]), "style": "violet",
         "anchor": "start", "dy": -10},
        {"kind": "point", "x": default_len, "y": h100[default_len], "label": x(h100[default_len]), "style": "violet",
         "anchor": "end", "dy": -10},
    ]
    p2 = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    return {
        "kind": "line",
        "title": ("The maximum concurrency of Llama-3.1-8B in BF16 against max_model_len on an L4 and an H100: "
                  f"the L4 goes below 1× at about {auto / 1000:.1f}k tokens."),
        "height": 400,
        "x": {"label": "max_model_len (tokens)", "scale": "log", "ticks": [4096, 8192, 16384, 32768, 65536, 131072],
              "min": 4096, "max": 131072},
        "y": {"label": "maximum concurrency (×)", "scale": "log", "ticks": p2, "min": 0.125, "max": 128},
        "series": [
            {"name": "H100 80 GB (SXM)", "x": lengths, "y": [h100[n] for n in lengths], "style": "violet"},
            {"name": "L4 24 GB", "x": lengths, "y": [l4[n] for n in lengths], "style": "accent"},
        ],
        "marks": marks,
        "legend": "top-right",
        "note": "computed by servelab.sizing.size: num_blocks / ceil(max_model_len / 16), with the lab's estimates",
    }


def spec_profiling_activations_vs_budget() -> dict:
    """§4.7 and §11: the activation peak of the profiling pass against max_num_batched_tokens, per sampler size."""
    m = sizing.load_config(MODEL)
    budgets = _log2_grid(512, 16384, 4, [2048, 8192])
    l4_default = sizing.serve_scheduler_defaults(sizing.gpu(L4))        # (2048, 256)
    h100_default = sizing.serve_scheduler_defaults(sizing.gpu(H100))    # (8192, 1024)

    def act(tokens: int, seqs: int) -> float:
        est = sizing.overhead_estimate(m, max_num_batched_tokens=tokens, max_num_seqs=seqs)
        return round(est["activations"] / GiB, 4)

    rows = {seqs: {b: act(b, seqs) for b in budgets} for seqs in (l4_default[1], h100_default[1])}
    l4_pt, h100_pt = act(*l4_default), act(*h100_default)
    return {
        "kind": "line",
        "title": ("The estimated activation peak of the profiling pass against max_num_batched_tokens for "
                  f"Llama-3.1-8B in BF16, with {l4_default[1]:,} and {h100_default[1]:,} sampler rows."),
        "x": {"label": "max_num_batched_tokens (tokens in the profiling forward)", "scale": "log",
              "ticks": [512, 1024, 2048, 4096, 8192, 16384]},
        "y": {"label": "activations + sampler logits (GiB)", "min": 0, "max": 2.5,
              "ticks": [0, 0.5, 1, 1.5, 2, 2.5]},
        "series": [
            {"name": f"{h100_default[1]:,}-row sampler (H100 default)", "x": budgets,
             "y": [rows[h100_default[1]][b] for b in budgets], "style": "violet"},
            {"name": f"{l4_default[1]:,}-row sampler (L4 default)", "x": budgets,
             "y": [rows[l4_default[1]][b] for b in budgets], "style": "accent"},
        ],
        "marks": [
            {"kind": "point", "x": l4_default[0], "y": l4_pt,
             "label": f"L4, {l4_default[0]:,}-token profile: {l4_pt:.2f} GiB", "style": "accent", "anchor": "start",
             "dy": 18},
            {"kind": "point", "x": h100_default[0], "y": h100_pt,
             "label": f"H100, {h100_default[0]:,}-token profile: {h100_pt:.2f} GiB", "style": "violet",
             "anchor": "end", "dy": -10},
        ],
        "legend": "top-left",
        "note": "ESTIMATE: computed by servelab.sizing.overhead_estimate (MLP activations + FP32 sampler logits)",
    }


FIGURES = {
    "concurrency-vs-max-model-len.svg": spec_concurrency_vs_max_model_len,
    "profiling-activations-vs-budget.svg": spec_profiling_activations_vs_budget,
}


def main() -> None:
    for name, build in FIGURES.items():
        print(f"wrote {svgchart.write_figure(FIGURES_DIR / name, build())}")


if __name__ == "__main__":
    main()
