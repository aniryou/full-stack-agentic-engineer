#!/usr/bin/env python3
"""The charts of the topic's PRIMER.md, computed by distillcore (tools/orchestration/FIGURE-STYLE.md §7).

Each chart is a spec builder, ``spec_<name>() -> dict``, that computes every value with the core's own functions
from the inputs the primer names (the five-token example's logits, the H100 preset, $11 per GPU-hour, a 2K context
and a 30 ms ITL, the model presets in ``cost.SHAPES``). ``tools/orchestration/svgchart.py`` renders the spec into
the house style, and ``main()`` writes each chart into ``../figures/`` beside the primer. ``tests/test_figures.py``
calls ``svgchart.check_figure()`` with the same builders, so a committed chart fails the test when the core
computes other values.

Only closed forms are charted: the soft-target KL against the temperature, the speedup of a draft against its
acceptance rate, and the break-even of a student. The trained students and drafts move with the CPU (see the
opening note of the primer), so they are never charted.

Run it from anywhere: ``python3 tools/make_figures.py`` (standard library + numpy, like the core). The output is
deterministic: two runs give the same bytes.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

CORE = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = CORE.parent / "figures"
REPO = next(p for p in CORE.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file())


def _load_svgchart():
    if "svgchart" in sys.modules:
        return sys.modules["svgchart"]
    spec = importlib.util.spec_from_file_location("svgchart", REPO / "tools" / "orchestration" / "svgchart.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svgchart"] = mod
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()

if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
from distillcore import cost as K, draft as Dr, losses as L  # noqa: E402

# The inputs the primer names.
Z = np.array([[4.0, 3.0, 1.0, 0.0, -1.0]])      # §2: the teacher's logits of the five-token example
V = np.array([[3.0, 3.5, 0.0, 0.5, -1.0]])      # §2: the student's logits
H100 = K.GPUS["h100"]
GPU_PRICE = 11.0                                 # §9: ~$11 per GPU-hour on demand (verify)
CONTEXT, ITL_S = 2048, 0.030                     # §9: a 2K context, a 30 ms ITL
SEQKD = dict(prompts=100_000, samples=1, tokens_per_sample=2000)   # §3: the fixed cost of a SeqKD run


def _sig(v: float) -> float:
    """Nine significant digits: a shorter <desc>, far inside check_figure()'s tolerance."""
    return float(f"{float(v):.9g}")


# ------------------------------------------------------------------ §2: soft targets and temperature

def spec_kl_against_temperature() -> dict:
    temps = sorted({round(10 ** (i / 20), 4) for i in range(61)} | {2.0, 4.0})   # 1 to 1000, and the table's 2 and 4
    kl = [_sig(L.kd(V, Z, T, scale=False)[0]) for T in temps]
    t2kl = [_sig(L.kd(V, Z, T)[0]) for T in temps]
    limit = L.logit_mse(V, Z)[0]
    k1, k4, s4 = L.kd(V, Z, 1.0, scale=False)[0], L.kd(V, Z, 4.0, scale=False)[0], L.kd(V, Z, 4.0)[0]
    return {
        "kind": "line",
        "title": "The soft term of the five-token example against the temperature T: the KL decreases approximately "
                 "as 1/T², and T²·KL goes to its high-T limit.",
        "x": {"label": "temperature T (log scale)", "scale": "log", "ticks": [1, 2, 4, 10, 100, 1000],
              "tick_labels": ["1", "2", "4", "10", "100", "1000"]},    # as the §2 table prints T
        "y": {"label": "nats (log scale)", "scale": "log", "min": 1e-7, "max": 1, "ticks": [1e-6, 1e-4, 1e-2, 1],
              "tick_labels": ["10⁻⁶", "10⁻⁴", "10⁻²", "1"]},
        "series": [
            {"name": "T²·KL(p_T ‖ q_T)", "x": temps, "y": t2kl, "style": "accent"},
            {"name": "KL(p_T ‖ q_T)", "x": temps, "y": kl, "style": "warm"},
        ],
        "marks": [
            {"kind": "hline", "y": _sig(limit), "label": f"logit matching: {limit:.5f}", "style": "muted"},
            {"kind": "point", "x": 1.0, "y": _sig(k1), "label": f"T = 1: {k1:.5f}", "style": "plain", "anchor": "start", "dy": -14},
            {"kind": "point", "x": 4.0, "y": _sig(s4), "label": f"T = 4: {s4:.5f}", "style": "accent", "anchor": "start", "dy": -10},
            {"kind": "point", "x": 4.0, "y": _sig(k4), "label": f"T = 4: {k4:.5f}", "style": "warm", "anchor": "start", "dy": -8},
        ],
        "legend": "bottom-right",
        "note": "computed by distillcore.losses.kd and losses.logit_mse",
    }


# ------------------------------------------------------------------ §7: a distilled draft for speculative decoding

def spec_draft_speedup() -> dict:
    c = Dr.draft_cost(K.SHAPES["qwen3-0.6b"].params(), K.SHAPES["qwen3-4b"].params())
    alphas = [round(0.40 + 0.01 * i, 2) for i in range(61)]
    styles = {4: "accent"}                       # §7 gives numbers for k = 4 only
    series = [{"name": f"k = {k}", "x": alphas, "y": [_sig(Dr.speedup(a, k, c)) for a in alphas], "style": st}
              for k, st in styles.items()]
    marks = []
    for a in (0.6, 0.7, 0.8):
        sp = Dr.speedup(a, 4, c)
        marks.append({"kind": "point", "x": a, "y": _sig(sp), "label": f"{sp:.2f}× at α = {a}", "style": "accent",
                      "anchor": "end"})
    marks.append({"kind": "text", "x": 0.62, "y": 3.9, "text": f"c ≈ {c:.3f} by weight bytes", "style": "muted",
                  "anchor": "start"})
    return {
        "kind": "line",
        "title": f"Speedup of a draft against its acceptance rate α at c ≈ {c:.3f}, for k = 4 drafted tokens.",
        "x": {"label": "acceptance rate α", "min": 0.4, "max": 1.0, "ticks": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        "y": {"label": "speedup (×)", "min": 0, "max": 4.5, "ticks": [0, 1, 2, 3, 4]},
        "series": series,
        "marks": marks,
        "legend": "top-left",
        "note": "computed by distillcore.draft.speedup and draft.draft_cost",
    }


# ------------------------------------------------------------------ §9: the economics of a student

def spec_break_even() -> dict:
    teacher = K.serving(K.SHAPES["qwen2.5-32b"], H100, GPU_PRICE, CONTEXT, ITL_S, n_gpus=2)["usd_per_m"]
    student = K.serving(K.SHAPES["qwen2.5-1.5b"], H100, GPU_PRICE, CONTEXT, ITL_S)["usd_per_m"]
    fixed = K.fixed_cost(**SEQKD, teacher_usd_per_m=teacher, student_params=K.SHAPES["qwen2.5-1.5b"].params(),
                         g=H100, gpu_price=GPU_PRICE)["total_usd"]
    be = K.break_even(fixed, teacher, student, 50e6)
    be_m = be["tokens"] / 1e6
    xs = list(range(0, 401, 25))                   # millions of output tokens
    return {
        "kind": "line",
        "title": "Cumulative cost against output tokens: the teacher line from zero, the student line from the one-off "
                 f"bill with a lower slope, and the crossing at {be_m:.1f} million tokens.",
        "x": {"label": "output tokens (millions)", "min": 0, "max": 400},
        "y": {"label": "cumulative cost ($)", "min": 0, "max": 400},
        "series": [
            {"name": f"teacher, 32B on two H100s: ${teacher:.3f} per million", "x": xs,
             "y": [_sig(teacher * x) for x in xs], "style": "warm"},
            {"name": f"student, 1.5B: ${fixed:,.2f} once, then ${student:.3f} per million", "x": xs,
             "y": [_sig(fixed + student * x) for x in xs], "style": "accent"},
        ],
        "marks": [
            {"kind": "vline", "x": _sig(be_m), "label": f"break-even: {be_m:.1f} million tokens", "style": "muted",
             "anchor": "start"},
            {"kind": "point", "x": 0, "y": _sig(fixed), "label": f"one-off bill: ${fixed:,.2f}", "style": "accent",
             "anchor": "start", "dy": -10},
            {"kind": "point", "x": _sig(be_m), "y": _sig(teacher * be_m), "style": "plain"},
            {"kind": "text", "x": _sig(be_m * 0.98), "y": 30, "text": f"{be['days']:.1f} days at 50 million output tokens a day",
             "style": "muted", "anchor": "end"},
        ],
        "legend": "top-left",
        "note": "computed by distillcore.cost.serving, cost.fixed_cost and cost.break_even",
    }


FIGURES = {
    "kl-against-temperature.svg": spec_kl_against_temperature,
    "draft-speedup.svg": spec_draft_speedup,
    "break-even.svg": spec_break_even,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
