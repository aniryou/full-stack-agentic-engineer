#!/usr/bin/env python3
"""make_figures.py - the charts of ../PRIMER.md that minengine computes (tools/orchestration/FIGURE-STYLE.md §7).

Each ``spec_<name>()`` computes its values with minengine's own functions, from the inputs the primer names (an H100
SXM and Llama-3.1-8B from ``perf.GPUS`` and ``perf.LLMS``, the default efficiencies and the 2 ms step overhead of
``perf.step_cost``, 64 decodes at 2,000 tokens of context, a Llama-3.2-1B draft at alpha = 0.7 and k = 4, alpha = 0.8
with c = 0.1), and returns a chart spec for tools/orchestration/svgchart.py. The marks are the primer's pinned
points, labelled as the primer prints them. ``tests/test_figures.py`` pins every chart with ``svgchart.check_figure``.
Every value is SIMULATED (a roofline model), never a measurement.

    python3 tools/make_figures.py        # (re)writes ../figures/<name>.svg for every entry of FIGURES
"""
from __future__ import annotations

import importlib.util
import inspect
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

from minengine import perf, spec  # noqa: E402

FIGURES_DIR = LAB.parent / "figures"

H100 = perf.GPUS["H100-SXM"]
LLAMA_8B = perf.LLMS["llama-3.1-8b"]
LLAMA_1B = perf.LLMS["llama-3.2-1b"]          # the draft of §7 (same tokenizer)
_STEP_DEFAULTS = inspect.signature(perf.step_cost).parameters           # read, not copied: the primer's §3 assumptions
DEFAULT_EFF = (_STEP_DEFAULTS["flop_eff"].default, _STEP_DEFAULTS["bw_eff"].default)
OVERHEAD_S = _STEP_DEFAULTS["overhead_s"].default
ASSUMPTIONS = f"{DEFAULT_EFF[0]:.0%} FLOPs, {DEFAULT_EFF[1]:.0%} bandwidth, {1e3 * OVERHEAD_S:g} ms"
EXERCISE_2_4_BUDGETS = [256, 512, 1024, 2048, 4096, 8192]                 # notebook 02, exercise 2.4's candidates


def _ms(seconds: float) -> float:
    return round(1e3 * seconds, 4)


def _log_grid(lo: int, hi: int, per_doubling: int, extra=()) -> list[int]:
    """Integers from lo to hi, about `per_doubling` of them per factor of two, plus the points the primer names."""
    pts, i = set(extra), 0
    while True:
        v = round(lo * 2 ** (i / per_doubling))
        if v > hi:
            break
        pts.add(v)
        i += 1
    return sorted(p for p in pts | {lo, hi} if lo <= p <= hi)


def spec_step_time_vs_tokens() -> dict:
    """§3: the step time of Llama-3.1-8B on an H100 against the tokens in one step, with its two roofline terms."""
    table = [1, 64, 256, 512, 2048, 8192]                                       # the columns of the §3 table
    tokens = _log_grid(1, 8192, 4, table)
    cost = {n: perf.step_cost(H100, LLAMA_8B, [(0, n)]) for n in tokens}
    overhead = {n: c["t"] - max(c["t_memory"], c["t_compute"]) for n, c in cost.items()}
    step = {n: _ms(c["t"]) for n, c in cost.items()}
    knee = perf.knee_tokens(H100, LLAMA_8B, *DEFAULT_EFF)

    def label(n: int) -> str:
        t = step[n]
        ms = f"{t:.0f}" if t >= 100 else f"{t:.1f}"
        return f"{n:,} token: {ms} ms" if n == 1 else f"{n:,}: {ms} ms"

    place = {1: ("start", -10), 64: ("middle", -12), 256: ("start", 20), 512: ("start", 18),
             2048: ("start", 16), 8192: ("end", -4)}
    marks = [{"kind": "vline", "x": round(knee, 3), "label": f"knee: ~{knee:.0f} tokens", "style": "warm"}]
    for n in table:
        anchor, dy = place[n]
        marks.append({"kind": "point", "x": n, "y": step[n], "label": label(n), "style": "accent",
                      "anchor": anchor, "dy": dy})
    return {
        "kind": "line",
        "title": ("The simulated step time of Llama-3.1-8B on an H100 against the tokens in the step, "
                  "with the two roofline terms."),
        "x": {"label": "tokens in the step", "scale": "log", "ticks": [1, 4, 16, 64, 256, 1024, 4096], "max": 8192},
        "y": {"label": "step time (ms)", "scale": "log", "ticks": [2, 5, 10, 20, 50, 100, 200], "min": 2, "max": 300},
        "series": [                                                             # the step time last, on top
            {"name": "bytes ÷ bandwidth + overhead", "x": tokens,
             "y": [_ms(cost[n]["t_memory"] + overhead[n]) for n in tokens], "style": "muted", "dash": True},
            {"name": "FLOPs ÷ peak + overhead", "x": tokens,
             "y": [_ms(cost[n]["t_compute"] + overhead[n]) for n in tokens], "style": "violet", "dash": True},
            {"name": "step time", "x": tokens, "y": [step[n] for n in tokens], "style": "accent"},
        ],
        "marks": marks,
        "legend": "top-left",
        "note": f"SIMULATED: computed by minengine.perf.step_cost and knee_tokens ({ASSUMPTIONS})",
    }


def spec_token_budget_worst_step() -> dict:
    """§3: the worst step (64 decodes at 2,000 tokens of context plus one full prefill chunk) against the budget."""
    decodes = [(2000, 1)] * 64
    # the candidates, not a continuous curve: the primer's rule ("the largest budget whose worst step meets the ITL
    # SLO") picks from this list, and a continuous budget would pass up to ~950 tokens
    budgets = [b for b in EXERCISE_2_4_BUDGETS if b <= 2048]
    worst = {b: _ms(perf.step_time(H100, LLAMA_8B, decodes + [(0, b - len(decodes))])) for b in budgets}
    alone = _ms(perf.step_time(H100, LLAMA_8B, decodes))
    slo_ms = 25                                                                  # the p99 ITL target of §3
    return {
        "kind": "line",
        "title": ("The simulated worst step of 64 decodes and one prefill chunk on an H100 against the token budget, "
                  "with the 25 ms ITL target."),
        "x": {"label": "token budget, max_num_batched_tokens (tokens)", "scale": "log",
              "ticks": budgets, "min": 200, "max": 2600},
        "y": {"label": "worst step time (ms)", "min": 0, "max": 60, "ticks": [0, 10, 20, 30, 40, 50, 60]},
        "series": [                                                             # the worst step last, on top
            {"name": "64 decodes alone", "x": [budgets[0], budgets[-1]], "y": [alone, alone], "style": "muted",
             "dash": True},
            {"name": "64 decodes + one prefill chunk", "x": budgets, "y": [worst[b] for b in budgets],
             "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": slo_ms, "label": f"p99 ITL target: {slo_ms} ms", "style": "warm", "anchor": "end"},
            {"kind": "point", "x": 512, "y": worst[512], "label": f"512 tokens: {worst[512]:.1f} ms",
             "style": "accent", "anchor": "end", "dy": 20},
            {"kind": "point", "x": 1024, "y": worst[1024], "label": f"1,024: {worst[1024]:.1f} ms",
             "style": "accent", "anchor": "end", "dy": -10},
        ],
        "legend": "top-left",
        "note": f"SIMULATED: computed by minengine.perf.step_time ({ASSUMPTIONS})",
    }


def spec_spec_speedup_vs_k() -> dict:
    """§7: E[tokens per pass] and the speedup E / (k c + 1) against the draft length k, at alpha = 0.8."""
    alpha, c = 0.8, 0.1
    ks = list(range(1, 13))
    free = [round(spec.expected_tokens(alpha, k), 4) for k in ks]               # c = 0: the speedup is E
    paid = [round(spec.speedup(alpha, k, c), 4) for k in ks]
    best = spec.best_k(alpha, c)
    return {
        "kind": "line",
        "title": ("The speedup of speculative decoding against the draft depth k at an acceptance rate of 0.8, "
                  "for a draft cost c of 0 and 0.1."),
        "x": {"label": "draft tokens per round, k", "ticks": ks, "min": 1, "max": 12},
        "y": {"label": "speedup over plain decode (×)", "min": 0, "max": 5, "ticks": [0, 1, 2, 3, 4, 5]},
        "series": [
            {"name": "c = 0: E[tokens per pass]", "x": ks, "y": free, "style": "muted", "dash": True,
             "markers": True},
            {"name": "c = 0.1", "x": ks, "y": paid, "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "point", "x": 4, "y": free[ks.index(4)], "label": f"k = 4: {free[ks.index(4)]:.2f}",
             "style": "plain", "anchor": "end", "dy": -10},
            {"kind": "point", "x": 4, "y": paid[ks.index(4)], "label": f"k = 4: {paid[ks.index(4)]:.2f}×",
             "style": "accent", "anchor": "middle", "dy": 22},
            {"kind": "point", "x": best, "y": paid[ks.index(best)],
             "label": f"best k = {best}: {paid[ks.index(best)]:.2f}×", "style": "accent", "anchor": "middle",
             "dy": 22},
        ],
        "legend": "bottom-right",
        "note": "computed by minengine.spec.speedup, expected_tokens and best_k (alpha = 0.8)",
    }


def spec_spec_speedup_vs_batch() -> dict:
    """§7: the SIMULATED speedup of a 1B draft for an 8B target on an H100 against the batch, at two contexts."""
    alpha, k = 0.7, 4
    batches = _log_grid(1, 256, 4, [1, 16, 64, 128, 256])
    kv_tokens = perf.kv_cache_blocks(H100, LLAMA_8B) * 16

    def gain(b: int, ctx: int, **kw) -> float:
        return round(perf.spec_speedup(H100, LLAMA_8B, LLAMA_1B, b, ctx, alpha, k, **kw), 4)

    def fits(b: int, ctx: int) -> bool:                                          # the KV of the round must fit
        return b * (ctx + k + 1) <= kv_tokens

    short = {b: gain(b, 200) for b in batches}
    kv_limit = kv_tokens / (2000 + k + 1)
    long_b = sorted({b for b in batches if fits(b, 2000)} | {int(kv_limit)})     # up to the last batch that fits
    long_ = {b: gain(b, 2000) for b in long_b}
    lean = [gain(b, 200, draft_overhead_s=0.0) for b in batches]
    return {
        "kind": "line",
        "title": ("The simulated speedup of a 1B draft for an 8B target on an H100 against the concurrent requests, "
                  "at 200 and 2,000 tokens of context."),
        "x": {"label": "concurrent requests (batch)", "scale": "log",
              "ticks": [1, 2, 4, 8, 16, 32, 64, 128, 256]},
        "y": {"label": "speedup over plain decode (×)", "min": 0, "max": 2.2, "ticks": [0, 0.5, 1, 1.5, 2]},
        "series": [
            {"name": "200-token context", "x": batches, "y": [short[b] for b in batches], "style": "accent"},
            {"name": "2,000-token context", "x": long_b, "y": [long_[b] for b in long_b], "style": "violet"},
            {"name": "200 tokens, drafts at roofline time", "x": batches, "y": lean, "style": "muted",
             "dash": True},
        ],
        "marks": [
            {"kind": "hline", "y": 1, "label": "no gain: 1×", "style": "warm", "anchor": "start"},
            {"kind": "vline", "x": round(kv_limit, 3), "label": "2,000 tokens: KV does not fit",
             "style": "violet", "anchor": "end", "label_pos": "top"},
            {"kind": "point", "x": 1, "y": short[1], "label": f"B = 1: {short[1]:.2f}×", "style": "accent",
             "anchor": "start", "dy": 18},
            {"kind": "point", "x": 128, "y": long_[128], "label": f"{long_[128]:.2f}×", "style": "violet",
             "anchor": "end", "dy": -8},
            {"kind": "point", "x": 128, "y": short[128], "label": f"B = 128: {short[128]:.2f}×", "style": "accent",
             "anchor": "end", "dy": 16},
            {"kind": "point", "x": 256, "y": short[256], "label": f"{short[256]:.2f}×", "style": "accent",
             "anchor": "middle", "dy": 22},
        ],
        "legend": "bottom-left",
        "note": "SIMULATED: computed by minengine.perf.spec_speedup and kv_cache_blocks (alpha = 0.7, k = 4)",
    }


FIGURES = {
    "step-time-vs-tokens.svg": spec_step_time_vs_tokens,
    "token-budget-worst-step.svg": spec_token_budget_worst_step,
    "spec-speedup-vs-k.svg": spec_spec_speedup_vs_k,
    "spec-speedup-vs-batch.svg": spec_spec_speedup_vs_batch,
}


def main() -> None:
    for name, build in FIGURES.items():
        print(f"wrote {svgchart.write_figure(FIGURES_DIR / name, build())}")


if __name__ == "__main__":
    main()
