#!/usr/bin/env python3
"""The charts of the layer 02 primer that gpusim computes (tools/orchestration/FIGURE-STYLE.md §7).

Each builder ``spec_<name>()`` computes its values with gpusim from the inputs the primer names (an L4's
compute capability, 8 H100s with alpha = 2 us and B = 450 GB/s, 384 kernels with 5 us launches) and returns a
spec for ``tools/orchestration/svgchart.py``. The marks are the primer's pinned points, labelled with the
number as the primer prints it. ``main()`` writes every chart into ``../figures/`` beside the primer;
``tests/test_figures.py`` calls the same builders, so a chart that no longer matches the core fails the tests.

    python3 tools/make_figures.py          # from cuda-nccl-core/, or from anywhere
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

CORE = pathlib.Path(__file__).resolve().parents[1]
TOPIC = CORE.parent                      # 02-cuda-nccl-runtime/cuda-and-nccl/
FIGDIR = TOPIC / "figures"
PRIMER = TOPIC / "PRIMER.md"

if str(CORE) not in sys.path:            # import gpusim from this checkout
    sys.path.insert(0, str(CORE))

from gpusim import collectives as C      # noqa: E402
from gpusim import occupancy as occ      # noqa: E402
from gpusim import tiling                # noqa: E402


def _load_svgchart():
    """tools/orchestration/svgchart.py, found by walking up to the repository root."""
    root = next((p for p in CORE.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file()), None)
    if root is None:
        raise RuntimeError("tools/orchestration/svgchart.py not found above " + str(CORE))
    spec = importlib.util.spec_from_file_location("svgchart", root / "tools" / "orchestration" / "svgchart.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()

# ------------------------------------------------------------------ the inputs the primer names

# §2.3: an L4, blocks of 256 threads, 64 registers per thread, 48 KB of shared memory per block
L4_CC = occ.DEVICES["L4"]["cc"]
THREADS = 256
REGS = 64
SMEM = 48 * 1024

# §4.2: 384 kernels, an eager launch of 5 us, one graph launch of 10 us; 2 us and 20 us kernels
N_KERNELS = 384
LAUNCH_US = 5.0
GRAPH_LAUNCH_US = 10.0
SHORT_US, LONG_US = 2.0, 20.0

# §5.4 to §5.6: 8 H100s, alpha = 2 us per step, B = 450 GB/s per direction (the illustrative NVLink 4 numbers
# of layer 01); a 70B-class model at TP = 8 (hidden 8,192, 80 layers), decode at batch 32, prefill of 8,192 tokens
P = 8
ALPHA = 2e-6
BW = 450e9
LAYERS, HIDDEN = 80, 8192
DECODE_TOKENS, PREFILL_TOKENS = 32, 8192
GIB = 2 ** 30

KIB = 1024
SIZES = [round(2 ** (e / 4)) for e in range(64, 121)]       # 64 KiB to 1 GiB, four points per octave
SIZE_TICKS = [2 ** e for e in range(16, 31, 2)]
SWEEP_ROWS = [2 ** 24, 2 ** 27]                             # two rows of the §5.5 sweep: 16 MiB and 128 MiB


def _r(v: float, nd: int = 4) -> float:
    """A value as the spec stores it (the chart needs no more digits)."""
    return round(float(v), nd)


def _bytes_label(n: int) -> str:
    for unit, scale in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if n >= scale and n % scale == 0:
            return f"{n // scale} {unit}"
    return f"{n} B"


def _steps(xs, ys) -> tuple[list, list]:
    """The first x of every run of equal y, and the last point: what a step series needs."""
    px, py = [xs[0]], [ys[0]]
    for x, y in zip(xs[1:], ys[1:]):
        if y != py[-1]:
            px.append(x)
            py.append(y)
    if px[-1] != xs[-1]:
        px.append(xs[-1])
        py.append(ys[-1])
    return px, py


# ------------------------------------------------------------------ §2.3 occupancy against registers per thread

def spec_occupancy_vs_registers() -> dict:
    sm = occ.sm_limits(L4_CC)
    regs = list(range(16, sm.max_regs_per_thread + 1))
    pct = lambda o: _r(o.occupancy * 100, 3)
    plain = [pct(occ.occupancy(THREADS, r, 0, L4_CC)) for r in regs]
    smem = [pct(occ.occupancy(THREADS, r, SMEM, L4_CC)) for r in regs]
    px, py = _steps(regs, plain)
    sx, sy = _steps(regs, smem)
    at64 = occ.occupancy(THREADS, REGS, 0, L4_CC)
    at64s = occ.occupancy(THREADS, REGS, SMEM, L4_CC)
    wpb = at64.warps_per_block
    yticks = [_r(100 * w / sm.max_warps, 3) for w in range(0, sm.max_warps + 1, wpb)]
    return {
        "kind": "line",
        "title": (f"Occupancy of one L4 SM against registers per thread for blocks of {THREADS} threads: "
                  f"{at64.blocks_per_sm} blocks and {at64.occupancy:.0%} at {REGS} registers, and "
                  f"{at64s.occupancy:.0%} when each block also uses {SMEM // KIB} KB of shared memory."),
        "x": {"label": "registers per thread", "min": regs[0], "max": regs[-1],
              "ticks": [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 255]},
        "y": {"label": f"occupancy (% of {sm.max_warps} warps)", "min": 0, "max": 100,
              "ticks": yticks, "tick_labels": [f"{t:.0f}%" for t in yticks]},
        "series": [
            {"name": "no shared memory", "x": px, "y": py, "style": "accent", "step": True},
            {"name": f"{SMEM // KIB} KB shared memory per block", "x": sx, "y": sy, "style": "warm",
             "step": True, "dash": True},
        ],
        "marks": [
            {"kind": "point", "x": REGS, "y": pct(at64), "style": "accent", "dy": -10,
             "label": f"{REGS} registers: {at64.blocks_per_sm} blocks, {at64.occupancy:.0%}"},
            {"kind": "point", "x": REGS, "y": pct(at64s), "style": "warm", "dy": 20,
             "label": f"{REGS} registers and {SMEM // KIB} KB: {at64s.occupancy:.0%}"},
        ],
        "legend": "top-right",
        "note": "computed by gpusim.occupancy.occupancy (cuda-nccl-core), CC 8.9",
    }


# ------------------------------------------------------------------ §4.2 step time against kernel length

def spec_step_time_vs_kernel_length() -> dict:
    ks = [i / 2 for i in range(1, 51)]                        # 0.5 us to 25 us
    run = lambda k, graph: tiling.step_time([k] * N_KERNELS, launch_us=LAUNCH_US, graph=graph,
                                            graph_launch_us=GRAPH_LAUNCH_US)["time_us"]
    eager = [_r(run(k, False), 3) for k in ks]
    graph = [_r(run(k, True), 3) for k in ks]
    e2, g2 = run(SHORT_US, False), run(SHORT_US, True)
    e20, g20 = run(LONG_US, False), run(LONG_US, True)
    return {
        "kind": "line",
        "title": (f"Time of {N_KERNELS} dependent kernels against the kernel length, with an eager launch of "
                  f"{LAUNCH_US:.0f} µs per kernel or one CUDA Graph launch of {GRAPH_LAUNCH_US:.0f} µs: "
                  f"{e2:,.0f} µs against {g2:,.0f} µs for {SHORT_US:.0f} µs kernels, and no gain for "
                  f"{LONG_US:.0f} µs kernels."),
        "x": {"label": "kernel length (µs, log scale)", "scale": "log", "min": ks[0], "max": ks[-1],
              "ticks": [0.5, 1, 2, 5, 10, 20]},
        "y": {"label": f"time of {N_KERNELS} kernels (µs, log scale)", "scale": "log", "min": 100, "max": 20000,
              "ticks": [100, 200, 500, 1000, 2000, 5000, 10000, 20000]},
        "series": [
            {"name": f"eager: one {LAUNCH_US:.0f} µs launch per kernel", "x": ks, "y": eager, "style": "warm"},
            {"name": f"CUDA Graph: one {GRAPH_LAUNCH_US:.0f} µs launch", "x": ks, "y": graph, "style": "accent",
             "dash": True},
        ],
        "marks": [
            {"kind": "vline", "x": LAUNCH_US, "style": "muted", "label_pos": "bottom",
             "label": f"kernel = launch, {LAUNCH_US:.0f} µs"},
            {"kind": "point", "x": SHORT_US, "y": _r(e2, 3), "style": "warm", "anchor": "middle", "dy": -11,
             "label": f"{SHORT_US:.0f} µs kernels: eager {e2:,.0f} µs"},
            {"kind": "point", "x": SHORT_US, "y": _r(g2, 3), "style": "accent", "dy": 17,
             "label": f"graph {g2:,.0f} µs, {e2 / g2:.1f}× faster"},
            {"kind": "point", "x": LONG_US, "y": _r(e20, 3), "style": "warm", "anchor": "end", "dy": -12,
             "label": f"{LONG_US:.0f} µs kernels: eager {e20:,.0f} µs, graph {g20:,.0f} µs"},
        ],
        "legend": "top-left",
        "note": "computed by gpusim.tiling.step_time (cuda-nccl-core)",
    }


# ------------------------------------------------------------------ §5.4 busbw against message size

def spec_allreduce_busbw_vs_size() -> dict:
    ring_t = [C.model_time("all_reduce", "ring", s, P, ALPHA, BW) for s in SIZES]
    nvls_t = [C.model_time("all_reduce", "switch", s, P, ALPHA, BW, None) for s in SIZES]
    ring_bus = [_r(C.busbw("all_reduce", s, t, P) / 1e9, 3) for s, t in zip(SIZES, ring_t)]
    ring_alg = [_r(C.algbw(s, t) / 1e9, 3) for s, t in zip(SIZES, ring_t)]
    nvls_bus = [_r(C.busbw("all_reduce", s, t, P) / 1e9, 3) for s, t in zip(SIZES, nvls_t)]
    t_ring = C.model_time("all_reduce", "ring", GIB, P, ALPHA, BW)
    k_nvls = C.pipeline_chunks("all_reduce", "switch", GIB, P, ALPHA, BW)
    t_nvls = C.model_time("all_reduce", "switch", GIB, P, ALPHA, BW, k_nvls)
    bus_ring = C.busbw("all_reduce", GIB, t_ring, P) / 1e9
    alg_ring = C.algbw(GIB, t_ring) / 1e9
    bus_nvls = C.busbw("all_reduce", GIB, t_nvls, P) / 1e9
    ceiling = C.BUSBW_FACTOR["all_reduce"](P) * BW / 1e9
    rows = [(s, C.busbw("all_reduce", s, C.model_time("all_reduce", "ring", s, P, ALPHA, BW), P) / 1e9)
            for s in SWEEP_ROWS]
    return {
        "kind": "line",
        "title": (f"All-reduce bandwidth against message size on {P} H100s in the alpha-beta model: the ring busbw "
                  f"goes up to the {BW / 1e9:.0f} GB/s link rate and is {bus_ring:.0f} GB/s at 1 GiB, and the NVLS "
                  f"busbw is {bus_nvls:.0f} GB/s at 1 GiB, above the link rate."),
        "height": 380,
        "x": {"label": "message size S (bytes, log scale)", "scale": "log", "min": SIZES[0], "max": SIZES[-1],
              "ticks": SIZE_TICKS, "tick_labels": [_bytes_label(s) for s in SIZE_TICKS]},
        "y": {"label": "bandwidth (GB/s)", "min": 0, "max": 850, "ticks": [0, 100, 200, 300, 400, 500, 600, 700, 800]},
        "series": [
            {"name": "ring busbw", "x": SIZES, "y": ring_bus, "style": "accent"},
            {"name": "ring algbw", "x": SIZES, "y": ring_alg, "style": "muted", "dash": True},
            {"name": "NVLS busbw (in-switch)", "x": SIZES, "y": nvls_bus, "style": "violet"},
        ],
        "marks": [
            {"kind": "hline", "y": BW / 1e9, "style": "warm", "label": f"link rate B = {BW / 1e9:.0f} GB/s"},
            {"kind": "hline", "y": ceiling, "style": "muted", "label": f"NVLS ceiling 2(p−1)/p · B = {ceiling:.0f} GB/s"},
            {"kind": "point", "x": GIB, "y": _r(bus_nvls, 3), "style": "violet", "anchor": "end", "dy": 36,
             "label": f"k* = {k_nvls}: busbw {bus_nvls:.0f} GB/s"},
            {"kind": "point", "x": GIB, "y": _r(bus_ring, 3), "style": "accent", "anchor": "end", "dy": 16,
             "label": f"1 GiB: busbw {bus_ring:.0f} GB/s"},
            {"kind": "point", "x": GIB, "y": _r(alg_ring, 3), "style": "muted", "anchor": "end", "dy": -9,
             "label": f"algbw {alg_ring:.0f} GB/s"},
        ] + [
            {"kind": "point", "x": s, "y": _r(v, 3), "style": "accent", "anchor": "start", "dy": dy,
             "label": f"{_bytes_label(s)}: {v:.2f} GB/s"}
            for (s, v), dy in zip(rows, (18, 30))
        ],
        "legend": "bottom-right",
        "note": "computed by gpusim.collectives.model_time, algbw, busbw and pipeline_chunks (cuda-nccl-core)",
    }


# ------------------------------------------------------------------ §5.5 ring all-reduce time against message size

def spec_ring_allreduce_regimes() -> dict:
    a, c = C.cost_terms("all_reduce", "ring", P)
    ring = [_r(C.model_time("all_reduce", "ring", s, P, ALPHA, BW) * 1e6) for s in SIZES]
    lat = [_r(a * ALPHA * 1e6) for _ in SIZES]
    bwt = [_r(c * s / BW * 1e6) for s in SIZES]
    xover = C.crossover_bytes("all_reduce", "ring", P, ALPHA, BW)
    decode = C.tp_comm(LAYERS, DECODE_TOKENS, HIDDEN, P, ALPHA, BW, "ring")
    prefill = C.tp_comm(LAYERS, PREFILL_TOKENS, HIDDEN, P, ALPHA, BW, "ring")
    s_dec, t_dec = decode["message_bytes"], decode["per_call_s"] * 1e6
    s_pre, t_pre = prefill["message_bytes"], prefill["per_call_s"] * 1e6
    t_gib = C.model_time("all_reduce", "ring", GIB, P, ALPHA, BW) * 1e3
    return {
        "kind": "line",
        # the alt text of this figure in PRIMER §5.5, which keeps its line (tests/test_figures.py compares them)
        "title": (f"Ring all-reduce time against message size on {P} H100s in the alpha-beta model: flat near the "
                  f"latency term below the {xover / 1e6:.1f} MB crossover, where the {_bytes_label(s_dec)} "
                  f"all-reduce of decode sits, and growing with the bandwidth term above it, where the "
                  f"{_bytes_label(s_pre)} all-reduce of prefill sits."),
        "height": 380,
        "x": {"label": "message size S (bytes, log scale)", "scale": "log", "min": SIZES[0], "max": SIZES[-1],
              "ticks": SIZE_TICKS, "tick_labels": [_bytes_label(s) for s in SIZE_TICKS]},
        "y": {"label": "time (µs, log scale)", "scale": "log", "min": 10, "max": 10000,
              "ticks": [10, 100, 1000, 10000]},
        "series": [
            {"name": "ring all-reduce", "x": SIZES, "y": ring, "style": "accent"},
            {"name": f"latency term 2(p−1)·α, {a} steps", "x": SIZES, "y": lat, "style": "muted", "dash": True},
            {"name": "bandwidth term 2(p−1)/p · S/B", "x": SIZES, "y": bwt, "style": "yellow", "dash": True},
        ],
        "marks": [
            {"kind": "vline", "x": xover, "style": "warm", "label": f"S* = p·α·B = {xover / 1e6:.1f} MB"},
            {"kind": "text", "x": xover / 1.25, "y": 2000, "text": "latency-bound", "anchor": "end"},
            {"kind": "text", "x": xover * 1.25, "y": 2000, "text": "bandwidth-bound", "anchor": "start"},
            {"kind": "point", "x": s_dec, "y": _r(t_dec), "style": "accent", "anchor": "middle", "dy": -12,
             "label": f"decode, {_bytes_label(s_dec)}: {t_dec:.1f} µs"},
            {"kind": "point", "x": s_pre, "y": _r(t_pre), "style": "accent", "anchor": "end",
             "label": f"prefill, {_bytes_label(s_pre)}: {t_pre:.0f} µs"},
            {"kind": "point", "x": GIB, "y": _r(t_gib * 1e3), "style": "accent", "anchor": "end",
             "label": f"1 GiB: {t_gib:.2f} ms"},
        ],
        "legend": "top-left",
        "note": "computed by gpusim.collectives.model_time, cost_terms, crossover_bytes and tp_comm (cuda-nccl-core)",
    }


FIGURES = {
    "occupancy-vs-registers.svg": spec_occupancy_vs_registers,
    "step-time-vs-kernel-length.svg": spec_step_time_vs_kernel_length,
    "allreduce-busbw-vs-size.svg": spec_allreduce_busbw_vs_size,
    "ring-allreduce-regimes.svg": spec_ring_allreduce_regimes,
}


def main() -> None:
    for name, build in FIGURES.items():
        path = svgchart.write_figure(FIGDIR / name, build())
        print(f"wrote {path.relative_to(TOPIC.parents[1])}")


if __name__ == "__main__":
    main()
