#!/usr/bin/env python3
"""The charts of the topic's PRIMER.md that gwcore computes (tools/orchestration/FIGURE-STYLE.md §7).

Each ``spec_<name>()`` computes every value with gwcore's own functions, from the inputs the primer names (the
targets' availabilities and the common-mode failures of §1.6 and §2.5, the bundled labelled traffic of §3.3, the
§4.3 workload that ``ratelimit.compare_buckets()`` runs), and returns a chart spec for
``tools/orchestration/svgchart.py``. The marks are the primer's pinned points, labelled with the number as the primer
prints it. ``main()`` writes every chart into ``../figures/`` beside the primer; ``tests/test_figures.py`` pins the
committed files to these builders.

    cd 06-gateway/llm-gateway/gateway-core && python3 tools/make_figures.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from gwcore import cache, ratelimit as RL, routing as R  # noqa: E402

ROOT = next(p for p in LAB.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file())
FIGURES_DIR = LAB.parent / "figures"


def _load_svgchart():
    if "svgchart" in sys.modules:
        return sys.modules["svgchart"]
    spec = importlib.util.spec_from_file_location("svgchart", ROOT / "tools" / "orchestration" / "svgchart.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svgchart"] = mod
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()


def _pct(x: float, digits: int) -> str:
    """A share as the primer prints it: ``0.99995`` with 3 digits is ``99.995 %``."""
    return f"{x:.{digits}%}".replace("%", " %")


# ------------------------------------------------------------------ §2.5: the common-mode floor of a chain

PRIMARY, FALLBACK = 0.995, 0.99            # §2.5: primary 99.5 %, fallback 99.0 % (and a third target at 99.0 %)
GATEWAY_COMMON_MODE = 0.0005               # §1.6: a gateway at 99.95 % in series with the chain
SHARED_COMMON_MODE = 0.001                 # §2.5: a 0.1 % common-mode failure


def spec_chain_availability() -> dict:
    chains = [[PRIMARY], [PRIMARY, FALLBACK], [PRIMARY, FALLBACK, FALLBACK]]
    n = [len(c) for c in chains]

    def failed(cm: float) -> list:
        return [100 * (1 - R.chain_availability(c, cm)) for c in chains]

    a = {cm: [R.chain_availability(c, cm) for c in chains] for cm in (0.0, GATEWAY_COMMON_MODE, SHARED_COMMON_MODE)}
    return {
        "kind": "line",
        "title": "Failed requests of a fallback chain against its number of targets, with no common-mode failure "
                 "and with one of 0.05 % or 0.1 %.",
        "height": 400,
        "x": {"label": "targets in the chain (99.5 %, then 99.0 % each)", "ticks": n, "tick_labels": ["1", "2", "3"],
              "min": 0.9, "max": 3.1},
        "y": {"label": "failed requests (%), log scale", "scale": "log", "min": 1e-5, "max": 1.0,
              "ticks": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
              "tick_labels": ["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"]},
        "series": [
            {"name": "independent failures", "x": n, "y": failed(0.0), "style": "accent", "markers": True},
            {"name": "0.05 % common-mode (the §1.6 gateway)", "x": n, "y": failed(GATEWAY_COMMON_MODE),
             "style": "yellow", "markers": True},
            {"name": "0.1 % common-mode", "x": n, "y": failed(SHARED_COMMON_MODE), "style": "warm", "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": 100 * SHARED_COMMON_MODE,
             "label": f"common-mode floor: {_pct(a[SHARED_COMMON_MODE][2], 1)}", "style": "warm"},
            {"kind": "point", "x": 2, "y": failed(0.0)[1], "label": f"2 targets: {_pct(a[0.0][1], 3)}",
             "style": "accent"},
            {"kind": "point", "x": 3, "y": failed(0.0)[2], "label": f"3 targets: {_pct(a[0.0][2], 5)}",
             "style": "accent", "anchor": "end", "dy": 18},
            {"kind": "point", "x": 2, "y": failed(SHARED_COMMON_MODE)[1],
             "label": f"0.1 % common-mode: {_pct(a[SHARED_COMMON_MODE][1], 3)}", "style": "warm"},
            {"kind": "point", "x": 2, "y": failed(GATEWAY_COMMON_MODE)[1],
             "label": f"the §1.6 gateway: {_pct(a[GATEWAY_COMMON_MODE][1], 3)}", "style": "yellow", "dy": 18},
        ],
        "legend": "bottom-left",
        "note": "computed by gwcore.routing.chain_availability",
    }


# ------------------------------------------------------------------ §3.3: hits and false hits against the threshold

# Every second hundredth from 0.50 to 0.98, the primer's 0.95, and 0.99. None of them sits on a similarity of the
# bundled sample (an exact 0.75 or 0.875, or 1.00 for an identical query in float32), so the counts do not depend on
# the last bit of a dot product; tests/test_figures.py checks that.
THRESHOLDS = sorted({round(0.50 + 0.02 * i, 2) for i in range(25)} | {0.95, 0.99})


def spec_cache_threshold_sweep() -> dict:
    off = cache.sweep_thresholds(THRESHOLDS, guard=False)
    on = cache.sweep_thresholds(THRESHOLDS, guard=True)
    at = lambda rows, th: next(r for r in rows if r["threshold"] == th)        # noqa: E731
    pct = lambda x: f"{100 * x:.1f} %"                                         # noqa: E731
    return {
        "kind": "line",
        "title": "Hits and false hits of the semantic cache against the threshold, without and with the entity guard, "
                 "on the bundled sample.",
        "height": 380,
        "x": {"label": "threshold τ (cosine similarity)", "min": 0.5, "max": 1.0,
              "ticks": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "tick_labels": ["0.50", "0.60", "0.70", "0.80", "0.90", "1.00"]},
        "y": {"label": "share of lookups (%)", "min": 0, "max": 105, "ticks": [0, 20, 40, 60, 80, 100]},
        "series": [
            {"name": "Hits, no guard", "x": THRESHOLDS, "y": [100 * r["hit_rate"] for r in off], "style": "accent"},
            {"name": "Hits, entity guard", "x": THRESHOLDS, "y": [100 * r["hit_rate"] for r in on], "style": "accent",
             "dash": True},
            {"name": "False hits, no guard", "x": THRESHOLDS, "y": [100 * r["false_hit_rate"] for r in off],
             "style": "warm"},
            {"name": "False hits, entity guard", "x": THRESHOLDS, "y": [100 * r["false_hit_rate"] for r in on],
             "style": "warm", "dash": True},
        ],
        "marks": [
            {"kind": "point", "x": 0.6, "y": 100 * at(off, 0.6)["false_hit_rate"],
             "label": f"0.60: {pct(at(off, 0.6)['false_hit_rate'])} false hits", "style": "warm", "dy": -10},
            {"kind": "point", "x": 0.6, "y": 100 * at(on, 0.6)["false_hit_rate"],
             "label": f"with the guard: {pct(at(on, 0.6)['false_hit_rate'])}", "style": "warm", "dy": 18},
            {"kind": "point", "x": 0.9, "y": 100 * at(off, 0.9)["hit_rate"],
             "label": f"0.90: {pct(at(off, 0.9)['hit_rate'])} hits", "style": "accent", "dy": -10},
        ],
        "legend": "top-right",
        "note": "computed by gwcore.cache.sweep_thresholds",
    }


# ------------------------------------------------------------------ §4.3: the size of the reservation

RESERVATIONS = [1024, 2048, 4096, 8192, 16384]   # the sweep of §4.3 (1,024, 2,048, 4,096) up to the 16,384 cap


def spec_token_reservation_sweep() -> dict:
    kw = RL.compare_buckets.__kwdefaults__            # the §4.3 experiment's workload and limit, as the core runs it
    work = RL.workload(kw["seconds"], kw["rate"], kw["prompt"], kw["new_median"], kw["sigma"], seed=kw["seed"])
    runs = [RL.simulate(work, "reserve", limit=kw["limit"], reserve_out=ro) for ro in RESERVATIONS]
    served = [100 * r["utilisation"] for r in runs]
    peak = [100 * r["peak_ratio"] for r in runs]

    def over(i: int) -> str:
        s = runs[i]["seconds_over_limit"]
        return f"{s} seconds over" if s else "none over"

    return {
        "kind": "line",
        "title": "Tokens served and the peak window, as a share of the limit, against the output reservation of the "
                 "token limiter.",
        "height": 380,
        "x": {"label": "output reservation (tokens), log scale", "scale": "log", "ticks": RESERVATIONS},
        "y": {"label": "% of the limit", "min": 0, "max": 130, "ticks": [0, 25, 50, 75, 100, 125]},
        "series": [
            {"name": "tokens served ÷ limit", "x": RESERVATIONS, "y": served, "style": "accent", "markers": True},
            {"name": "peak window ÷ limit", "x": RESERVATIONS, "y": peak, "style": "warm", "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": 100, "label": f"the {kw['limit']:,} tokens-per-minute limit", "style": "muted"},
            # the §4.3 sweep: what each reservation serves, and the seconds its peak window spends over the limit
            {"kind": "point", "x": RESERVATIONS[0], "y": served[0], "label": f"{_pct(runs[0]['utilisation'], 1)} served",
             "style": "accent", "dy": 26},
            {"kind": "point", "x": RESERVATIONS[1], "y": served[1], "label": f"{_pct(runs[1]['utilisation'], 1)} served",
             "style": "accent", "anchor": "end", "dy": 20},
            {"kind": "point", "x": RESERVATIONS[2], "y": served[2], "label": f"{_pct(runs[2]['utilisation'], 1)} served",
             "style": "accent", "anchor": "end", "dy": 20},
            {"kind": "point", "x": RESERVATIONS[0], "y": peak[0], "label": over(0), "style": "warm"},
            {"kind": "point", "x": RESERVATIONS[1], "y": peak[1], "label": over(1), "style": "warm"},
            {"kind": "point", "x": RESERVATIONS[2], "y": peak[2], "label": over(2), "style": "warm"},
        ],
        "legend": "bottom-left",
        "note": "computed by gwcore.ratelimit.simulate",
    }


FIGURES = {
    "chain-availability-common-mode.svg": spec_chain_availability,
    "semantic-cache-threshold-sweep.svg": spec_cache_threshold_sweep,
    "token-reservation-sweep.svg": spec_token_reservation_sweep,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIGURES_DIR / name, build())
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
