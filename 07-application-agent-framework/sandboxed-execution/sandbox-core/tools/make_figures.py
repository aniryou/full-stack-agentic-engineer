#!/usr/bin/env python3
"""The charts of the topic's PRIMER.md, computed by sandboxcore (tools/orchestration/FIGURE-STYLE.md §7).

Each chart is a spec builder, ``spec_<name>() -> dict``, that computes every value with the core's own functions
from the inputs the primer names (§6: 5 executions/s, a 2 s run and a 3 s cold start; a wait target of 0.2; the
simulation's seed 1 and 30,000 arrivals). ``tools/orchestration/svgchart.py`` renders the spec into the house
style, and ``main()`` writes each chart into ``../figures/`` beside the primer. ``tests/test_figures.py`` calls
``svgchart.check_figure()`` with the same builders, so a committed chart fails the test when the core computes
other values.

Only closed forms (Erlang C) and the two simulated figures that ``tests/test_pool.py`` pins are charted; the
latencies of §2 and §6 are measurements or ``(verify)`` inputs, so they are never charted.

Run it from anywhere: ``python3 tools/make_figures.py`` (standard library only, like the core). The output is
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
from sandboxcore import pool  # noqa: E402

# The inputs the primer names (§6, the worked example; FACTS §12).
LAM = 5.0           # executions/s
T_EXEC = 2.0        # s, the run
T_COLD = 3.0        # s, the cold start of a replacement
TARGET = 0.2        # the wait target: P(wait) <= 0.2
SIM_N, SIM_SEED = 30_000, 1     # the simulation of §6 (SIMULATED, seed 1, 30,000 arrivals)


def _sig(v: float) -> float:
    """Nine significant digits: a shorter <desc>, far inside check_figure()'s tolerance."""
    return float(f"{float(v):.9g}")


# ------------------------------------------------------------------ §6: how many slots for a wait target

def spec_pool_wait_vs_slots() -> dict:
    a_rau = pool.mean_occupancy(LAM, T_EXEC, T_COLD)                          # 25 Erlangs: busy + warming
    a_reuse = pool.mean_occupancy(LAM, T_EXEC, T_COLD, replace_after_use=False)   # 10 Erlangs: busy only
    cs = list(range(8, 41))
    rau = [_sig(pool.erlang_c(a_rau, c)) for c in cs]
    reuse = [_sig(pool.erlang_c(a_reuse, c)) for c in cs]
    c_rau = pool.slots_for_wait_target(LAM, T_EXEC, T_COLD, TARGET)
    c_reuse = pool.slots_for_wait_target(LAM, T_EXEC, T_COLD, TARGET, replace_after_use=False)
    p_rau, p_reuse = pool.erlang_c(a_rau, c_rau), pool.erlang_c(a_reuse, c_reuse)
    c_first = int(a_rau) + 1
    p_first = pool.erlang_c(a_rau, c_first)
    sim_floor = pool.simulate(LAM, T_EXEC, T_COLD, int(a_rau), n=SIM_N, mode="replace_after_use", seed=SIM_SEED)
    sim_sized = pool.simulate(LAM, T_EXEC, T_COLD, c_rau, n=SIM_N, mode="replace_after_use", seed=SIM_SEED)
    f_floor, f_sized = round(sim_floor.frac_waited, 2), round(sim_sized.frac_waited, 2)
    return {
        "kind": "line",
        "title": "The share of requests that wait for a warm sandbox against the pool size, for a replace-after-use "
                 f"pool and a reuse pool at {LAM:g} executions/s.",
        "x": {"label": "pool slots c", "min": cs[0], "max": cs[-1], "ticks": list(range(10, 41, 5))},
        "y": {"label": "P(wait), share of requests", "min": 0, "max": 1.05, "ticks": [0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        "series": [
            {"name": f"replace-after-use, a = {a_rau:g}", "x": cs, "y": rau, "style": "accent", "markers": True},
            {"name": f"reuse, a = {a_reuse:g}", "x": cs, "y": reuse, "style": "violet", "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": TARGET, "label": f"wait target {TARGET:g}", "style": "warm"},
            {"kind": "vline", "x": a_rau, "label": f"{a_rau:g} slots occupied on average", "style": "muted",
             "anchor": "start", "label_pos": "bottom"},
            {"kind": "point", "x": c_first, "y": _sig(p_first), "label": f"{c_first} slots: {p_first:.3f}",
             "style": "accent", "anchor": "start"},
            {"kind": "point", "x": c_rau, "y": _sig(p_rau), "label": f"{c_rau} slots: {p_rau:.3f}", "style": "accent",
             "anchor": "start", "dy": -12},
            {"kind": "point", "x": c_reuse, "y": _sig(p_reuse), "label": f"{c_reuse} slots: {p_reuse:.3f}",
             "style": "violet", "anchor": "start", "dy": -12},
            {"kind": "point", "x": int(a_rau), "y": f_floor, "label": f"simulated: {round(100 * f_floor)}% wait",
             "style": "yellow", "anchor": "end"},
            {"kind": "point", "x": c_rau, "y": f_sized, "label": f"simulated: {round(100 * f_sized)}% wait",
             "style": "yellow", "anchor": "end", "dy": 4},
        ],
        "legend": "top-right",
        "note": "computed by sandboxcore.pool.erlang_c, pool.slots_for_wait_target and pool.simulate (seed 1)",
    }


FIGURES = {
    "pool-wait-vs-slots.svg": spec_pool_wait_vs_slots,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
