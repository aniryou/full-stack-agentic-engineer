#!/usr/bin/env python3
"""make_figures.py - the charts of ../PRIMER.md that quantcore computes (tools/orchestration/FIGURE-STYLE.md §7).

Each ``spec_<name>()`` computes its values with quantcore's own functions, from the inputs the primer names (the
down_proj of Llama-3.1-8B on an L4, a 128x256 Gaussian weight with seed 0, 2,000-token sessions, vLLM's GSM8K
example at 76.8%), and returns a chart spec for tools/orchestration/svgchart.py. The marks are the primer's pinned
points, labelled as the primer prints them. ``tests/test_figures.py`` pins every chart with ``svgchart.check_figure``.

    python3 tools/make_figures.py        # (re)writes ../figures/<name>.svg for every entry of FIGURES
"""
from __future__ import annotations

import importlib.util
import math
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

import numpy as np  # noqa: E402

from quantcore import cost as C  # noqa: E402
from quantcore import eval as E  # noqa: E402
from quantcore import formats as F  # noqa: E402
from quantcore import granularity as G  # noqa: E402
from quantcore import kvquant as K  # noqa: E402

FIGURES_DIR = LAB.parent / "figures"

L4 = C.GPUS["L4"]
M8 = C.MODELS["llama-3.1-8b"]
DOWN_PROJ = (14336, 4096)          # K, N of Llama-3.1-8B's down_proj (PRIMER §1)


def spec_gemm_speedup() -> dict:
    """§1: the speedup of W4A16 and FP8 W8A8 over BF16 for one down_proj GEMM on an L4, against the tokens per step."""
    k, n = DOWN_PROJ
    cross = {s: C.crossover_tokens(k, n, L4, s) for s in ("bf16", "w8a8-fp8", "w4a16")}
    tokens = {int(round(2 ** (i / 8))) for i in range(0, 97)}                      # 1 to 4,096 tokens per step
    for x in cross.values():
        tokens |= {math.floor(x), math.ceil(x)}                                   # the corners of each roofline
    tokens = sorted(tokens)
    t = lambda m, s: C.gemm_time(m, k, n, L4, s)["t"]
    speedup = lambda s: [round(t(m, "bf16") / t(m, s), 4) for m in tokens]
    w4 = speedup("w4a16")
    at = dict(zip(tokens, w4))
    knee = cross["w4a16"]                                                         # where W4A16 turns compute-bound
    return {
        "kind": "line",
        "title": ("The speedup over BF16 of the down_proj GEMM of Llama-3.1-8B on an L4, for W4A16 and FP8 W8A8, "
                  "against the tokens per step."),
        "x": {"label": "tokens per step", "scale": "log", "ticks": [1, 4, 16, 64, 256, 1024, 4096]},
        "y": {"label": "speedup over BF16 (×)", "min": 0, "max": 4.5, "ticks": [0, 1, 2, 3, 4]},
        "series": [
            {"name": "W4A16 (INT4 g128)", "x": tokens, "y": w4, "style": "accent"},
            {"name": "FP8 W8A8", "x": tokens, "y": speedup("w8a8-fp8"), "style": "violet"},
        ],
        "marks": [
            {"kind": "vline", "x": round(cross["bf16"], 3), "label": f"BF16 compute-bound: {cross['bf16']:.0f}",
             "style": "warm"},
            {"kind": "point", "x": 1, "y": at[1], "label": f"1 token: {at[1]:.1f}×", "style": "accent"},
            {"kind": "point", "x": round(knee, 3), "y": round(t(knee, "bf16") / t(knee, "w4a16"), 4),
             "label": f"W4A16 compute-bound: {knee:.0f}", "style": "accent", "anchor": "end", "dy": 18},
            {"kind": "point", "x": 256, "y": at[256], "label": f"256 tokens: {at[256]:.1f}×", "style": "accent",
             "anchor": "end", "dy": 18},
        ],
        "legend": "bottom-left",
        "note": "computed by quantcore.cost.gemm_time",
    }


def spec_sqnr_vs_bits() -> dict:
    """§2: SQNR of per-channel INT2 to INT8 on 128x256 Gaussian weights (seed 0), the rule against the grid."""
    w = np.random.default_rng(0).standard_normal((128, 256))
    crest = float(np.mean(np.abs(w).max(1) / np.sqrt((w ** 2).mean(1))))
    bits = list(range(2, 9))
    rule = [round(float(F.sqnr_rule_db(b, crest)), 3) for b in bits]
    grid = [round(G.error(w, G.fake_quant(w, fmt=f"int{b}", convention="full"))["sqnr_db"], 3) for b in bits]
    r, g = dict(zip(bits, rule)), dict(zip(bits, grid))
    return {
        "kind": "line",
        "title": ("The SQNR of per-channel INT2 to INT8 on 128×256 Gaussian weights: the rule of about 6 dB per bit, "
                  "and what the grid measures."),
        "x": {"label": "integer bits b (one scale per channel)", "ticks": bits, "min": 2, "max": 8},
        "y": {"label": "SQNR (dB)", "min": 0, "max": 48, "ticks": [0, 10, 20, 30, 40]},
        "series": [
            {"name": f"the rule, crest factor {crest:.2f}", "x": bits, "y": rule, "style": "plain", "dash": True},
            {"name": "the grid measures", "x": bits, "y": grid, "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "point", "x": 8, "y": g[8], "label": f"INT8: rule {r[8]:.1f} dB, grid {g[8]:.1f}",
             "style": "accent", "anchor": "end"},
            {"kind": "point", "x": 4, "y": g[4], "label": f"INT4: rule {r[4]:.1f}, grid {g[4]:.1f}",
             "style": "accent", "dy": 18},
        ],
        "legend": "top-left",
        "note": "computed by quantcore.formats.sqnr_rule_db",
    }


def spec_kv_sessions() -> dict:
    """§6: KV bytes per token of Llama-3.1-8B at 16, 8, 5 and 3 bits per element, and 2,000-token sessions on an L4
    with FP8 weights (the core's round memory inputs)."""
    kinds = [("BF16", K.kv_bits_per_element(16)),
             ("FP8", K.kv_bits_per_element(8)),
             ("4-bit, groups of 32", K.kv_bits_per_element(4, 32, 16, 16)),
             ("2-bit, groups of 32", K.kv_bits_per_element(2, 32, 16, 16))]
    labels, sessions = [], []
    for name, b in kinds:
        labels.append(f"{name} · {b:.0f} bits\n{M8.kv_bytes_per_token(b):,.0f} B per token")
        sessions.append(C.sessions(L4, M8, "w8a8-fp8", 2000, b))
    return {
        "kind": "bar",
        "title": ("KV bytes per token for Llama-3.1-8B at 16, 8, 5 and 3 bits per element, and 2,000-token sessions "
                  "on an L4 with FP8 weights."),
        "categories": [name for name, _ in kinds],
        "x": {"tick_labels": labels},
        "y": {"label": "2,000-token sessions on an L4", "min": 0, "max": 260, "ticks": [0, 50, 100, 150, 200, 250]},
        "series": [{"name": "sessions, FP8 weights", "y": sessions, "style": "accent"}],
        "legend": "none",
        "note": "computed by quantcore.cost.sessions",
    }


def spec_visible_drop() -> dict:
    """§8: the smallest accuracy drop an unpaired comparison can tell from noise (2 x diff_stderr), against the items."""
    p = 0.768                                         # vLLM's FP8 GSM8K example: 250 items at 76.8%
    items = sorted({int(round(10 ** (2 + i / 20))) for i in range(53)} | {250, 40000})      # 100 to 40,000 items
    drop = [round(100 * 2 * E.diff_stderr(p, p, n), 4) for n in items]
    n77 = next(n for n in range(100, 100000) if 2 * E.diff_stderr(0.77, 0.77, n) <= 0.01)   # 1 point at 77%
    d250 = 100 * 2 * E.diff_stderr(p, p, 250)
    return {
        "kind": "line",
        "title": ("The smallest accuracy drop that an unpaired comparison of two models can tell from noise, against "
                  "the number of eval items per model."),
        "x": {"label": "eval items per model", "scale": "log", "ticks": [100, 1000, 10000], "max": 40000},
        "y": {"label": "smallest visible drop (points)", "scale": "log", "ticks": [0.5, 1, 2, 5, 10, 20],
              "min": 0.5, "max": 20},
        "series": [{"name": f"2 × the unpaired standard error, at {100 * p:.1f}%", "x": items, "y": drop,
                    "style": "accent"}],
        "marks": [
            {"kind": "point", "x": 250, "y": round(d250, 4), "label": f"250 items: ~{d250:.1f} points",
             "style": "accent"},
            {"kind": "point", "x": n77, "y": round(100 * 2 * E.diff_stderr(0.77, 0.77, n77), 4),
             "label": f"{n77:,} items per model: 1 point at 77%", "style": "warm", "anchor": "end", "dy": 18},
        ],
        "legend": "top-right",
        "note": "computed by quantcore.eval.diff_stderr",
    }


FIGURES = {
    "gemm-speedup-vs-tokens.svg": spec_gemm_speedup,
    "sqnr-vs-bits.svg": spec_sqnr_vs_bits,
    "kv-cache-bytes-and-sessions.svg": spec_kv_sessions,
    "visible-drop-vs-eval-items.svg": spec_visible_drop,
}


def main() -> None:
    for name, build in FIGURES.items():
        print(f"wrote {svgchart.write_figure(FIGURES_DIR / name, build())}")


if __name__ == "__main__":
    main()
