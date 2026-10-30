#!/usr/bin/env python3
"""The charts of the topic's PRIMER.md, computed by rlcore (tools/orchestration/FIGURE-STYLE.md §7).

Each chart is a spec builder, ``spec_<name>() -> dict``, that computes every value with the core's own functions
from the inputs the primer names: one question with p = 0.4 and four ways to spread the incorrect 60% (§6), a
1,500-token prompt (§7), the capacity primer's example (8.33 requests/s, 1,500 tokens in, the ``MISTRAL_SMALL``
preset in FP8 on the ``H100`` preset, a 40 ms TPOT, §7), and lognormal thinking lengths with median 1,500 and
sigma = 1, a 300-token answer and e0 = 0.7 (§7). ``tools/orchestration/svgchart.py`` renders the spec into the house
style, and ``main()`` writes each chart into ``../figures/`` beside the primer. ``tests/test_figures.py`` calls
``svgchart.check_figure()`` with the same builders, so a committed chart fails the test when the core computes other
values.

Only exact enumerations and closed forms are charted: the majority vote, the KV a request holds, the sizing
formulas and the budget outcome. The trained toy policies and the Monte Carlo numbers are not charted.

Run it from anywhere: ``python3 tools/make_figures.py`` (standard library + numpy, like the core). The output is
deterministic: two runs give the same bytes.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

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
from rlcore import ttc, workload as w  # noqa: E402

# The inputs the primer names.
P_VOTE = 0.4                                     # §6: one question, a sample is correct with p = 0.4
PROMPT = 1500                                    # §7: P = 1,500 prompt tokens
H100, MODEL = w.GPUS["H100"], w.MISTRAL_SMALL    # §7: the capacity primer's example, FP8 on H100s
RPS = 10_000 * 0.10 * 0.5 / 60                   # §7: 8.33 requests/s
ANSWER = 300                                     # §7: a 300-token answer
MEDIAN, SIGMA, E0 = 1500, 1.0, 0.7               # §7: L_req ~ lognormal(1,500, sigma = 1), forced answers e0 = 0.7


def _sig(v: float) -> float:
    """Nine significant digits: a shorter <desc>, far inside check_figure()'s tolerance."""
    return float(f"{float(v):.9g}")


# ------------------------------------------------------------------ §6: the majority vote against n

def spec_majority_vote_vs_n() -> dict:
    rows = [  # the four rows of the §6 table, named as the table names them
        ("one dominant misconception (0.5 / 0.1)", [0.5, 0.1], "warm", False, 151),
        ("a narrow misconception (0.42 / 0.18)", [0.42, 0.18], "accent", False, 151),
        ("two equal incorrect answers (0.3 / 0.3)", [0.3, 0.3], "plain", True, 151),
        ("four scattered incorrect answers (0.15 each)", [0.15] * 4, "violet", False, 45),
    ]
    acc = lambda wrong, n: _sig(ttc.majority_accuracy(P_VOTE, wrong, n))
    series = []
    for name, wrong, style, dash, n_max in rows:
        ns = list(range(1, n_max + 1, 2))        # odd n: no ties between the right answer and one wrong answer
        series.append({"name": name, "x": ns, "y": [acc(wrong, n) for n in ns], "style": style, "dash": dash})
    narrow = rows[1][1]
    # §6: "the vote drops below one sample's accuracy only past about 130": the first odd n below p
    cross = next(n for n in range(1, 400, 2) if ttc.majority_accuracy(P_VOTE, narrow, n) < P_VOTE)
    assert 120 <= cross <= 140, f"the narrow vote now drops below p at n = {cross}, not about 130"
    single = ttc.majority_accuracy(P_VOTE, narrow, 1)
    marks = [
        {"kind": "hline", "y": _sig(single), "style": "muted"},
        {"kind": "point", "x": 15, "y": acc(narrow, 15), "style": "accent", "anchor": "middle", "dy": -10,
         "label": f"{ttc.majority_accuracy(P_VOTE, narrow, 15):.3f} at 15"},
        {"kind": "point", "x": cross, "y": acc(narrow, cross), "style": "accent", "anchor": "end", "dy": 18,
         "label": f"below one sample ({single:.3f}) past about 130"},
    ]
    for (_, wrong, style, _, _), anchor in zip((rows[0], rows[2], rows[3]), ("end", "start", "start")):  # n = 31
        marks.append({"kind": "point", "x": 31, "y": acc(wrong, 31), "style": style, "anchor": anchor, "dy": 16,
                      "label": f"{ttc.majority_accuracy(P_VOTE, wrong, 31):.3f} at 31"})
    return {
        "kind": "line",
        "title": "The accuracy of a majority vote against the number of samples, for one question where a sample "
                 "is correct with p = 0.4, for four ways to spread the incorrect 60%.",
        "height": 400,
        "x": {"label": "samples per question n (log scale)", "scale": "log", "min": 1, "max": 151,
              "ticks": [1, 3, 5, 15, 31, 63, 131]},
        "y": {"label": "P(the vote is correct)", "min": 0, "max": 1, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        "series": series,
        "marks": marks,
        "legend": "top-left",
        "note": "computed by rlcore.ttc.majority_accuracy (exact, p = 0.4, odd n)",
    }


# ------------------------------------------------------------------ §7: the KV a request holds, step by step

def spec_kv_token_steps() -> dict:
    held = lambda t: w.kv_token_steps(PROMPT, t) - w.kv_token_steps(PROMPT, t - 1)   # P + t at decode step t
    short, long_ = 300, 3000
    s_short, s_long = w.kv_token_steps(PROMPT, short), w.kv_token_steps(PROMPT, long_)
    area = lambda L: {"x": [1, L, L], "y": [held(1), held(L), 0]}        # the request frees its KV after step L
    return {
        "kind": "line",
        # the alt text of the figure in PRIMER.md §7, which the replaced hand-drawn figure carried
        "title": "The KV a request holds at each decode step, drawn against time: a 300-token output on a "
                 "1,500-token prompt is a thin sliver, a 3,000-token output an area 18.2 times larger.",
        "x": {"label": "decode step t (at step t the request holds P + t tokens of KV, for L steps)", "min": 0,
              "max": 3200, "ticks": [0, short, 1000, 2000, long_]},
        "y": {"label": "KV tokens held", "min": 0, "max": 5000, "ticks": [0, 1500, 3000, 4500]},
        "series": [
            {"name": f"L = {long_:,}: {s_long:,} KV-token-steps = {s_long / s_short:.1f}×", **area(long_),
             "style": "warm", "fill": True},
            {"name": f"L = {short:,}: {s_short:,} KV-token-steps", **area(short), "style": "accent", "fill": True},
        ],
        "marks": [
            {"kind": "hline", "y": PROMPT, "style": "muted"},
            {"kind": "text", "x": 1750, "y": 1150, "style": "muted", "anchor": "middle",
             "text": f"P = {PROMPT:,} prompt tokens, held from the first decode step"},
            {"kind": "text", "x": 1750, "y": 650, "style": "muted", "anchor": "middle",
             "text": "the area is memory × time per request"},
            {"kind": "point", "x": short, "y": held(short), "style": "accent", "anchor": "end", "dy": -10,
             "label": f"peak {held(short):,}"},
            {"kind": "point", "x": long_, "y": held(long_), "style": "warm", "anchor": "end", "dy": -10,
             "label": f"peak {held(long_):,} against {held(short):,} tokens: only {held(long_) / held(short):.1f}×"},
        ],
        "legend": "top-left",
        "note": "computed by rlcore.workload.kv_token_steps: the area is P·L + L(L + 1)/2 (P = 1,500)",
    }


# ------------------------------------------------------------------ §7: the fleet against the output length

def spec_memory_gpus_vs_output() -> dict:
    outs = list(range(ANSWER, 6001, 100))
    conv = lambda out: w.plan(MODEL, H100, RPS, PROMPT, out)["gpus"]
    steady = lambda out: w.plan_steady(MODEL, H100, RPS, PROMPT, out)["gpus"]
    c3, s3, c0, s0 = conv(3000), steady(3000), conv(ANSWER), steady(ANSWER)
    ratio = round(c3["memory"] / c0["memory"])                      # §7: ten times the output, eighteen times the GPUs
    assert ratio == round(s3["memory"] / s0["memory"]), "the two conventions no longer agree on the ratio"
    pt = lambda x, y, st, anchor, dy, label: {"kind": "point", "x": x, "y": _sig(y), "style": st, "anchor": anchor,
                                              "dy": dy, "label": label}
    return {
        "kind": "line",
        "title": "GPUs for memory against the output tokens per request in the capacity primer's example "
                 "(8.33 requests/s, 1,500 tokens in, FP8 on H100s), at the SLO's TPOT and at the step the fleet runs at.",
        "height": 400,
        "x": {"label": "output tokens per request (thinking + a 300-token answer, log scale)", "scale": "log",
              "min": 150, "max": 6000, "ticks": [300, 600, 1000, 1500, 3000, 6000]},
        "y": {"label": "GPUs needed for memory (log scale)", "scale": "log", "min": 0.1, "max": 20,
              "ticks": [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]},
        "series": [
            {"name": "at the SLO's TPOT (plan)", "x": outs, "y": [_sig(conv(o)["memory"]) for o in outs],
             "style": "accent"},
            {"name": "at the step the fleet runs at (plan_steady)", "x": outs,
             "y": [_sig(steady(o)["memory"]) for o in outs], "style": "plain", "dash": True},
        ],
        "marks": [
            {"kind": "hline", "y": _sig(c0["prefill"]), "label": f"prefill: {c0['prefill']:.2f} GPUs", "style": "warm",
             "anchor": "end"},
            pt(ANSWER, c0["memory"], "accent", "end", 4, f"300 tokens: {c0['memory']:.3f}"),
            pt(ANSWER, s0["memory"], "plain", "end", 4, f"{s0['memory']:.2f}"),
            pt(3000, c3["memory"], "accent", "end", -9, f"3,000 tokens: {c3['memory']:.2f}"),
            pt(3000, s3["memory"], "plain", "start", 16, f"{s3['memory']:.2f}"),
            {"kind": "text", "x": 1000, "y": 3, "style": "muted", "anchor": "middle",
             "text": f"{3000 // ANSWER}× the output: {ratio}× the GPUs on both lines"},
        ],
        "legend": "bottom-right",
        "note": "computed by rlcore.workload.plan and plan_steady (40 ms TPOT, memory only)",
    }


# ------------------------------------------------------------------ §7: max_tokens against a thinking budget

def spec_max_tokens_vs_budget() -> dict:
    p99 = w.lognormal_quantile(MEDIAN, SIGMA, 0.99)
    max_model_len = int(-(-(PROMPT + p99 + ANSWER) // 1024) * 1024)        # §7: a multiple of 1,024
    max_tokens = max_model_len - PROMPT
    limits = sorted({round(2 ** (10 + i / 8)) for i in range(41)} | {2048, 4096, 8192, 16384, max_tokens})
    cut = lambda c: w.budget_outcome(MEDIAN, SIGMA, ANSWER, E0, max_tokens=c)
    forced = lambda c: w.budget_outcome(MEDIAN, SIGMA, ANSWER, E0, budget=c - ANSWER)
    pt = lambda y, st, anchor, dy: {"kind": "point", "x": 4096, "y": _sig(y), "style": st, "anchor": anchor, "dy": dy,
                                    "label": f"{y:.3f}"}
    return {
        "kind": "line",
        "title": "Accuracy against the output limit when max_tokens truncates and when a thinking budget forces "
                 "the answer, for lognormal thinking lengths with median 1,500 and a 300-token answer.",
        "height": 380,
        "x": {"label": "output limit (tokens, log scale)", "scale": "log", "min": 1024, "max": 32768,
              "ticks": [1024, 2048, 4096, 8192, 16384, 32768]},
        "y": {"label": "share of requests", "min": 0, "max": 1.3, "ticks": [0, 0.25, 0.5, 0.75, 1.0]},
        "series": [
            {"name": "thinking budget (limit − 300): accuracy", "x": limits,
             "y": [_sig(forced(c)["accuracy"]) for c in limits], "style": "accent"},
            {"name": "max_tokens: accuracy", "x": limits, "y": [_sig(cut(c)["accuracy"]) for c in limits],
             "style": "plain"},
            {"name": "max_tokens: truncated, no answer", "x": limits, "y": [_sig(cut(c)["truncated"]) for c in limits],
             "style": "warm", "dash": True},
        ],
        "marks": [
            {"kind": "vline", "x": max_tokens, "label": f"max_tokens = {max_tokens:,}", "style": "muted",
             "anchor": "end", "label_pos": "top"},
            pt(forced(4096)["accuracy"], "accent", "end", -9),
            pt(cut(4096)["accuracy"], "plain", "start", 16),
            pt(cut(4096)["truncated"], "warm", "end", 16),
        ],
        "legend": "top-left",
        "note": "computed by rlcore.workload.budget_outcome (median 1,500, sigma = 1, e0 = 0.7)",
    }


FIGURES = {
    "majority-vote-vs-n.svg": spec_majority_vote_vs_n,
    "kv-token-steps.svg": spec_kv_token_steps,
    "memory-gpus-vs-output.svg": spec_memory_gpus_vs_output,
    "max-tokens-vs-budget.svg": spec_max_tokens_vs_budget,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
