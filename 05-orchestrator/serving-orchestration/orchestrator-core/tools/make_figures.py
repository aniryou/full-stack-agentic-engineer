#!/usr/bin/env python3
"""make_figures.py - the charts of ../PRIMER.md that fleetsim computes (tools/orchestration/FIGURE-STYLE.md §7).

Each ``spec_<name>()`` computes its values with fleetsim's own functions, from the inputs the primer and its notebooks
name (the ``L4_8B`` and ``H100_8B`` engine profiles, the 131,072 B/token KV of an 8B model, the workloads and seeds of
notebooks 03 to 05, the SLOs of the primer's tables, the tier bandwidths that §6 assumes), and returns a chart spec for
tools/orchestration/svgchart.py. The marks are the primer's numbers, labelled as the primer prints them.
``tests/test_figures.py`` pins every chart with ``svgchart.check_figure``. The fleet numbers are SIMULATED (a model of
an engine, under fixed seeds), never a measurement; the §6.2 chart is closed-form arithmetic.

    python3 tools/make_figures.py        # (re)writes ../figures/<name>.svg for every entry of FIGURES
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from decimal import ROUND_HALF_UP, Decimal

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

from fleetsim import (H100_8B, L4_8B, LLAMA_8B_KV, Fleet, PowerOfTwo, Tier, breakeven_gb_s, chat,  # noqa: E402
                      onload_s, pd_plan, rag, recompute_s, search_pd)

FIGURES_DIR = LAB.parent / "figures"


def _r(v: float, nd: int = 4) -> float:
    """A stored value: rounded, so the spec in the SVG stays short (the test compares within 1e-6)."""
    return round(float(v), nd)


def _as_table(v: float) -> str:
    """A value as the §4.2 table prints it: notebook 03's three significant digits, then two decimals (0.995 -> 1.00).

    The label follows the table, not the plotted value: the "GPU util" at 0.5 req/s is 0.995, so its point sits just
    under 1 while its label reads 1.00, as the table and the prose print it.
    """
    return str(Decimal(f"{v:.3g}").quantize(Decimal("0.01"), ROUND_HALF_UP))


# ------------------------------------------------------------------ §4.2: which signal

LOADS = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]                    # notebook 03's load test


def _chats(rate, duration=300, seed=5):
    """Notebook 03's chat load: 1,300 tokens in (a 1,000-token system prompt + 300), 150 out."""
    return chat(rate, duration, seed=seed, system=1000, user=300, output=150)


def _load_test() -> dict:
    """The per-replica averages of one L4 replica from 60 s to 300 s, at each offered load (notebook 03)."""
    out = {}
    for rate in LOADS:
        res = Fleet(L4_8B, 1, PowerOfTwo(seed=1), sample_s=15.0).run(_chats(rate))
        tl = [row for row in res.timeline if 60 <= row["t"] <= 300]
        out[rate] = {k: sum(row[k] for row in tl) / len(tl) for k in ("gpu_util", "kv")}
    return out


def spec_signals_vs_load() -> dict:
    """§4.2: "GPU util" and KV usage of one L4 replica against the offered chat load."""
    sweep = _load_test()
    util = [_r(sweep[x]["gpu_util"]) for x in LOADS]
    kv = [_r(sweep[x]["kv"]) for x in LOADS]
    return {
        "kind": "line",
        "title": "The simulated GPU utilisation and KV usage of one L4 replica against the offered chat load.",
        "x": {"label": "offered load (req/s)", "min": 0, "max": 3.5,
              "ticks": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]},
        "y": {"label": "per-replica value (0 to 1)", "min": 0, "max": 1.1, "ticks": [0, 0.25, 0.5, 0.75, 1]},
        "series": [
            {"name": "\"GPU util\" (what nvidia-smi reports)", "x": LOADS, "y": util, "style": "warm",
             "markers": True},
            {"name": "KV usage", "x": LOADS, "y": kv, "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "point", "x": 0.2, "y": util[0], "label": f"0.2 req/s: {_as_table(util[0])}", "style": "warm",
             "anchor": "start", "dy": 18},
            {"kind": "point", "x": 0.5, "y": util[1], "label": f"0.5 req/s: {_as_table(util[1])}", "style": "warm",
             "anchor": "start", "dy": -10},
            {"kind": "point", "x": 3.5, "y": kv[-1], "label": f"3.5 req/s: KV {_as_table(kv[-1])}", "style": "accent",
             "anchor": "end", "dy": -10},
        ],
        "legend": "bottom-right",
        "note": "SIMULATED (seed 5): computed by fleetsim.sim.Fleet, timeline averages from 60 s to 300 s, notebook 03",
    }


# ------------------------------------------------------------------ §5.3: sizing the P:D ratio

def _heavier():
    """Notebook 04's traffic: 1.2 req/s of RAG prompts of ~6,060 tokens (4 retrieved chunks), 250-token answers."""
    return rag(1.2, 150, seed=11, system=400, docs=4, doc=1400, corpus=5000, zipf=0.0, output=250)


def spec_pd_split_slo() -> dict:
    """§5.3: the SLO attainment of every xPyD split of eight L4s, with aggregated serving and the analytic plan."""
    rows = {r["split"]: r for r in search_pd(L4_8B, _heavier, 8, ttft_slo=3.0, tpot_slo=0.08,
                                             make_router=lambda: PowerOfTwo(seed=1))}
    plan = pd_plan(L4_8B, rate=1.2, isl=6060, osl=250, itl_slo_s=0.08)
    xs = list(range(1, 8))
    att = [_r(rows[f"{n}P{8 - n}D"]["slo_attainment"]) for n in xs]
    agg = _r(rows["8 aggregated"]["slo_attainment"])
    one, seven = rows["1P7D"], rows["7P1D"]
    n_prefill = _r(plan["prefill_replicas"])
    return {
        "kind": "line",
        "title": ("The simulated SLO attainment of every xPyD split of eight L4 replicas, "
                  "against aggregated serving and the analytic plan."),
        "height": 400,
        "x": {"label": "prefill replicas of eight; the rest decode", "min": 0.6, "max": 7.4, "ticks": xs,
              "tick_labels": [f"{n}P{8 - n}D" for n in xs]},
        "y": {"label": "SLO attainment (TTFT ≤ 3 s, TPOT ≤ 80 ms)", "min": 0, "max": 1,
              "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1]},
        "series": [{"name": "xPyD split", "x": xs, "y": att, "style": "accent", "markers": True}],
        "marks": [
            {"kind": "hline", "y": agg, "label": f"aggregated: {agg:.2f}", "style": "muted"},
            {"kind": "vline", "x": n_prefill, "label": f"pd_plan: {n_prefill:.2f} replicas", "style": "violet",
             "anchor": "end"},
            {"kind": "point", "x": 3, "y": att[2], "label": f"3P5D: {att[2]:.2f}", "style": "accent",
             "anchor": "start", "dy": -10},
            {"kind": "point", "x": 1, "y": att[0], "label": f"1P7D collapses: {one['ttft_p95']:.0f} s TTFT p95",
             "style": "warm", "anchor": "start", "dy": 3},
            {"kind": "point", "x": 7, "y": att[-1], "style": "warm"},     # its label in two lines, clear of 6P2D
            {"kind": "text", "x": 7, "y": att[-1] + 0.11, "text": "7P1D collapses:", "style": "warm", "anchor": "end"},
            {"kind": "text", "x": 7, "y": att[-1] + 0.06, "text": f"{seven['tpot_p95']:.1f} s TPOT p95", "style": "warm",
             "anchor": "end"},
        ],
        "legend": "none",
        "note": "SIMULATED (seed 11): computed by fleetsim.disagg.search_pd and pd_plan, 100 Gb/s link, notebook 04",
    }


# ------------------------------------------------------------------ §6.2: fetch or recompute

TOKENS = 10_000                                                      # "For ten thousand tokens on the H100"
DRAM = Tier("host DRAM (PCIe)", 256, 50.0, 0.0005)                   # notebook 05's tiers, assumed (verify)
GBE_25 = Tier("remote store, 25 GbE", 10_000, 3.0, 0.002)
BANDWIDTHS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]               # GB/s


def spec_fetch_or_recompute() -> dict:
    """§6.2: the time to get the KV of 10,000 tokens: a fetch against the tier bandwidth, a recompute on two GPUs."""
    ms = lambda s: _r(1e3 * s)                                       # noqa: E731
    fetch = [ms(onload_s(TOKENS, LLAMA_8B_KV, Tier("fetch", 1e9, bw))) for bw in BANDWIDTHS]
    l4 = ms(recompute_s(TOKENS, L4_8B.compute_tok_s))
    h100 = ms(recompute_s(TOKENS, H100_8B.compute_tok_s))
    be_l4 = _r(breakeven_gb_s(LLAMA_8B_KV, L4_8B.compute_tok_s))
    be_h100 = _r(breakeven_gb_s(LLAMA_8B_KV, H100_8B.compute_tok_s))
    dram, gbe = ms(onload_s(TOKENS, LLAMA_8B_KV, DRAM)), ms(onload_s(TOKENS, LLAMA_8B_KV, GBE_25))
    return {
        "kind": "line",
        "title": ("The time to get the KV of ten thousand tokens: a fetch against the tier bandwidth, "
                  "and a recompute on an L4 and an H100."),
        "x": {"label": "tier bandwidth (GB/s)", "scale": "log", "min": 0.1, "max": 100},
        "y": {"label": "time for ten thousand tokens (ms)", "scale": "log", "min": 10, "max": 20000,
              "ticks": [10, 100, 1000, 10000]},
        "series": [
            {"name": "fetch, no latency: tokens × KV bytes/token ÷ bandwidth", "x": BANDWIDTHS, "y": fetch, "style": "accent"},
        ],
        "marks": [                                                   # a fetch wins to the right of each crossing
            {"kind": "hline", "y": l4, "label": "recompute on an L4", "style": "violet"},
            {"kind": "hline", "y": h100, "label": f"recompute on an H100: {h100:.0f} ms", "style": "warm"},
            {"kind": "point", "x": be_l4, "y": l4, "label": f"L4 break-even: {be_l4:.2f} GB/s", "style": "violet",
             "anchor": "start", "dy": -10},
            {"kind": "point", "x": be_h100, "y": h100, "label": f"H100 break-even: {be_h100:.2f} GB/s",
             "style": "warm", "anchor": "end", "dy": 18},
            {"kind": "point", "x": GBE_25.read_gb_s, "y": gbe, "label": f"25 GbE: {gbe:.0f} ms", "style": "accent",
             "anchor": "start", "dy": -10},
            {"kind": "point", "x": DRAM.read_gb_s, "y": dram, "label": f"host DRAM: {dram:.0f} ms",
             "style": "accent", "anchor": "end", "dy": 18},
        ],
        "legend": "bottom-left",
        "note": "computed by fleetsim.kvtier.onload_s, recompute_s and breakeven_gb_s (8B model, 131,072 B/token)",
    }


FIGURES = {
    "signals-vs-load.svg": spec_signals_vs_load,
    "pd-split-slo-attainment.svg": spec_pd_split_slo,
    "fetch-or-recompute-vs-bandwidth.svg": spec_fetch_or_recompute,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIGURES_DIR / name, build())
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
