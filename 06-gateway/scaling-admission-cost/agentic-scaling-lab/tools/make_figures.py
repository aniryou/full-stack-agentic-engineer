#!/usr/bin/env python3
"""The charts of docs/01-scaling-primer.md, computed by scalelab.capacity (tools/orchestration/FIGURE-STYLE.md §7).

Each builder ``spec_<name>()`` computes its values with the core's own functions from the inputs the primer names
(the anchor scenario of §3.1, a 12-second turn, the Flash-Lite share, the PT terms) and returns a chart spec for
``tools/orchestration/svgchart.py``. ``python3 tools/make_figures.py`` writes every chart into ``docs/figures/``;
``tests/test_figures.py`` checks each committed file against the same builder, so a chart fails the test when the
core computes other values.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from scalelab.capacity import PT_USD_PER_GSU_HOUR, Scenario, plan  # noqa: E402

FIG_DIR = LAB / "docs" / "figures"


def _repo_root(start: pathlib.Path) -> pathlib.Path:
    """The folder that holds tools/orchestration/svgchart.py, found by a walk up from this file."""
    for d in [start, *start.parents]:
        if (d / "tools" / "orchestration" / "svgchart.py").is_file():
            return d
    raise RuntimeError("tools/orchestration/svgchart.py not found above " + str(start))


def _load_svgchart():
    path = _repo_root(LAB) / "tools" / "orchestration" / "svgchart.py"
    loaded = sys.modules.get("svgchart")
    if loaded is not None and pathlib.Path(getattr(loaded, "__file__", "")).resolve() == path.resolve():
        return loaded
    spec = importlib.util.spec_from_file_location("svgchart", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["svgchart"] = module
    spec.loader.exec_module(module)
    return module


svgchart = _load_svgchart()


def _r(v: float, nd: int = 6) -> float:
    return round(float(v), nd)


# --- §1.3: Little's law, turns in flight against the arrival rate for a 6-second and a 12-second turn ---------------

SLOW_TURN_S = 12.0  # §1.3: "a 12-second turn means 250"


def spec_inflight_turns() -> dict:
    base = Scenario()
    per_day = list(range(0, 360_001, 20_000))       # conversations a day; the average level of plan() is the rate
    fast = [plan(Scenario(conversations_per_day=c)) for c in per_day]
    slow = [plan(Scenario(conversations_per_day=c, turn_seconds=SLOW_TURN_S)) for c in per_day]
    xs = [_r(p["rates"]["average"]["turns_per_s"]) for p in fast]
    peak_fast = plan(base)
    peak_slow = plan(Scenario(turn_seconds=SLOW_TURN_S))
    x_peak = peak_fast["rates"]["peak"]["turns_per_s"]
    budget = peak_fast["concurrency"]          # the token budget of the 10 M TPM baseline (§3.3)
    return {
        "kind": "line",
        "title": "Turns in flight against the arrival rate of turns, for a 6-second turn and a 12-second turn (Little's law).",
        "x": {"label": "arrival rate (turns per second)", "min": 0, "max": 25},
        "y": {"label": "turns in flight", "min": 0},
        "series": [
            {"name": f"{base.turn_seconds:g}-second turn", "x": xs,
             "y": [_r(p["concurrency"]["average"]["inflight_turns"]) for p in fast], "style": "accent"},
            {"name": "12-second turn", "x": xs,
             "y": [_r(p["concurrency"]["average"]["inflight_turns"]) for p in slow], "style": "violet"},
        ],
        "marks": [
            {"kind": "vline", "x": _r(budget["max_turns_per_s_for_baseline"]), "label": "10 M TPM sustains ≈ 14 turns/s",
             "style": "warm"},
            {"kind": "point", "x": _r(budget["max_turns_per_s_for_baseline"]), "y": _r(budget["max_inflight_for_baseline"]),
             "label": "about 85 in flight", "style": "accent", "anchor": "start", "dy": 16},
            {"kind": "point", "x": _r(x_peak), "y": _r(peak_slow["concurrency"]["peak"]["inflight_turns"]),
             "label": "12-second turn: 250", "style": "violet", "anchor": "end"},
            {"kind": "point", "x": _r(x_peak), "y": _r(peak_fast["concurrency"]["peak"]["inflight_turns"]),
             "label": "6-second turn: 125", "style": "accent", "anchor": "start", "dy": 16},
        ],
        "legend": "top-left",
        "note": "computed by scalelab.capacity.plan",
    }


# --- §3.4: cost per conversation against the share of calls on Flash-Lite, with and without the cached prefix -------

def spec_cost_per_conversation() -> dict:
    base = Scenario()
    shares = [i / 20 for i in range(21)]              # 0 % to 100 % of model calls on the lite model
    cached = [plan(Scenario(lite_share=s))["cost"]["per_conversation_routed_cached"] for s in shares]
    uncached = [plan(Scenario(lite_share=s, cached_tokens=0))["cost"]["per_conversation_routed_cached"] for s in shares]
    anchor = plan(base)["cost"]
    xs = [_r(100 * s, 4) for s in shares]
    return {
        "kind": "line",
        "title": "Cost per conversation against the share of model calls on 3.5 Flash-Lite, with no caching and with the prefix cached.",
        "x": {"label": "share of model calls on 3.5 Flash-Lite (%)", "min": 0, "max": 100},
        "y": {"label": "cost per conversation ($)", "min": 0, "ticks": [0, 0.05, 0.1, 0.15],
              "tick_labels": ["0", "0.05", "0.10", "0.15"]},
        "series": [
            {"name": "no caching", "x": xs, "y": [_r(v) for v in uncached], "style": "muted", "dash": True},
            {"name": "prefix cached, 90 % hit rate", "x": xs, "y": [_r(v) for v in cached], "style": "accent"},
        ],
        "marks": [
            {"kind": "point", "x": 0, "y": _r(anchor["per_conversation_no_cache"]), "label": "all 3.5 Flash: $0.141",
             "style": "plain", "anchor": "start", "dy": -10},
            {"kind": "point", "x": 0, "y": _r(anchor["per_conversation_cached"]), "label": "prefix cached: $0.092",
             "style": "accent", "anchor": "start", "dy": -10},
            {"kind": "point", "x": _r(100 * base.lite_share, 4), "y": _r(anchor["per_conversation_routed_cached"]),
             "label": "35 % on Flash-Lite: $0.068", "style": "accent", "anchor": "start", "dy": -10},
        ],
        "legend": "top-right",
        "note": "computed by scalelab.capacity.plan",
    }


# --- §3.5: the price of a used burndown token on PT against the utilisation of the GSUs, per term, against PayGo ----

TERMS = {"1w": ("1 week", "muted"), "1m": ("1 month", "violet"), "3m": ("3 months", "yellow"), "1y": ("1 year", "accent")}


def spec_pt_break_even() -> dict:
    base = Scenario()
    p = plan(base)["pt"]
    paygo = p["usd_per_million_burndown"]["paygo"]
    util = [u / 100 for u in range(30, 101, 2)]       # the share of the bought GSU-hours that serves traffic
    series = []
    for term in PT_USD_PER_GSU_HOUR:
        name, style = TERMS[term]
        full = p["usd_per_million_burndown"][term]      # the price per M burndown tokens at 100 % utilisation
        series.append({"name": name, "x": [_r(100 * u, 4) for u in util], "y": [_r(full / u) for u in util],
                       "style": style, "dash": term == "1w"})
    peak_util = p["gsus"]["average"] / p["gsus"]["peak"]          # PT bought for the peak, used at the average load
    return {
        "kind": "line",
        "title": "The price per M burndown tokens on Provisioned Throughput against the utilisation of the GSUs, for each term, and the PayGo price.",
        "height": 420,
        "x": {"label": "utilisation of the GSUs (%)", "min": 30, "max": 100, "ticks": list(range(30, 101, 10))},
        "y": {"label": "$ per M burndown tokens", "min": 0, "max": 5},
        "series": series,
        "marks": [
            {"kind": "hline", "y": _r(paygo), "style": "warm"},
            {"kind": "text", "x": 35, "y": _r(paygo + 0.1), "text": "PayGo ($1.50/M burndown)", "style": "warm",
             "anchor": "start"},
            {"kind": "vline", "x": _r(100 * peak_util, 4), "label": "PT for the peak: 33 % utilised", "style": "warm"},
            {"kind": "point", "x": _r(100 * p["breakeven_utilisation"]["1y"], 4), "y": _r(paygo),
             "label": "1 year: 75 %", "style": "accent", "anchor": "end", "dy": 36},
            {"kind": "point", "x": _r(100 * p["breakeven_utilisation"]["3m"], 4), "y": _r(paygo),
             "label": "3 months: 90 %", "style": "yellow", "anchor": "end", "dy": 36},
        ],
        "legend": "top-right",
        "note": "computed by scalelab.capacity.plan",
    }


FIGURES = {
    "inflight-turns-littles-law.svg": spec_inflight_turns,
    "cost-per-conversation-flash-lite-share.svg": spec_cost_per_conversation,
    "pt-break-even-utilisation.svg": spec_pt_break_even,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(LAB)}")


if __name__ == "__main__":
    main()
