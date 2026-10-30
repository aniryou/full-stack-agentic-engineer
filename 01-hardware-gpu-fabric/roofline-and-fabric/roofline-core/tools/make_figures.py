#!/usr/bin/env python3
"""make_figures.py: the charts of the topic's PRIMER.md that this core computes (FIGURE-STYLE.md §7).

Each chart is a spec (a plain dict) that one builder below computes with the core's own functions, from the
inputs the primer names: a device of ``roofline.specs``, a model preset of ``roofline.llm``, a link of
``roofline.fabric``, the Llama 3 failure count of §7.1. No value is copied out of the prose; the marks are the
primer's pinned points, labelled the way the primer prints them (``tests/test_primer_numbers.py`` holds the same
format strings). ``tools/orchestration/svgchart.py`` renders each spec into ``../figures/<name>.svg`` beside the
primer, and ``tests/test_figures.py`` checks that every committed chart is what the core computes now.

    python3 tools/make_figures.py          # rewrite every chart (the output is the same bytes on every machine)

Standard library only.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

LAB = pathlib.Path(__file__).resolve().parents[1]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from roofline import fabric, llm, reliability, specs  # noqa: E402
from roofline import roofline as rl  # noqa: E402

TOPIC = LAB.parent
FIG_DIR = TOPIC / "figures"
DOC = TOPIC / "PRIMER.md"
REPO = next(p for p in LAB.parents if (p / "tools" / "orchestration" / "svgchart.py").is_file())


def _load_svgchart():
    path = REPO / "tools" / "orchestration" / "svgchart.py"
    spec = importlib.util.spec_from_file_location("svgchart", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()

H100, L4 = specs.DEVICES["h100-sxm"], specs.DEVICES["l4"]
M8, M70 = llm.PRESETS["llama-3.1-8b"], llm.PRESETS["llama-3.1-70b"]
GIB = 1 << 30


def sig(v: float, digits: int = 6) -> float:
    """A value rounded to `digits` significant digits: the chart needs no more, and the file stays small."""
    return float(f"{v:.{digits}g}")


# ---------------------------------------------------------------- §2: the roofline

def spec_roofline() -> dict:
    """attainable(I) for the H100 and the L4 in dense bf16, with the kernels of §2 and the LLM steps of §3."""
    def line(dev):
        xs = sorted({0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, sig(dev.ridge())})
        return xs, [sig(rl.attainable(x, dev) / 1e12) for x in xs]

    hx, hy = line(H100)
    lx, ly = line(L4)
    n = 1 << 26                                          # the size does not change an intensity
    kernels = [  # (kernel, label, anchor, dy) -- the rows of the §2 table, labelled with its intensity column
        (rl.elementwise(n, 2), "vector add: {:.2f}", "start", 17),
        (rl.reduction(n, 4), "sum of n fp32: {:.2f}", "start", 4),
        (rl.gemm(1, 4096, 4096), "GEMV, 1 token: {:.1f}", "end", -8),
        (rl.gemm(64, 4096, 4096), "GEMM 64 × 4096 × 4096: {:.0f}", "end", -24),
        (rl.gemm(4096, 4096, 4096), "GEMM 4096³: {:,.0f}", "middle", 20),
    ]
    marks = []
    for k, fmt, anchor, dy in kernels:
        marks.append({"kind": "point", "x": sig(k.intensity), "y": sig(rl.attainable(k.intensity, H100) / 1e12),
                      "label": fmt.format(k.intensity), "style": "plain", "anchor": anchor, "dy": dy})
    steps = [  # (step, label, anchor, dy) -- §3.2 and two rows of the §3.3 table (2K context)
        (llm.decode(M8, H100, 1, 2048), "decode, batch 1: {:.1f}", "start", 17),
        (llm.decode(M8, H100, 64, 2048), "decode, batch 64: {:.1f}", "end", -8),
        (llm.prefill(M8, H100, 2048), "prefill, 2,048 tokens: {:,.0f}", "middle", -10),
    ]
    for s, fmt, anchor, dy in steps:
        marks.append({"kind": "point", "x": sig(s.intensity), "y": sig(rl.attainable(s.intensity, H100) / 1e12),
                      "label": fmt.format(s.intensity), "style": "accent", "anchor": anchor, "dy": dy})
    marks += [
        {"kind": "vline", "x": sig(H100.ridge()), "label": f"H100 ridge: {H100.ridge():.0f} FLOP/B",
         "style": "warm", "anchor": "end"},
        {"kind": "text", "x": sig(L4.ridge() * 1.12), "y": sig(L4.peak() / 1e12 / 2),
         "text": f"L4 ridge: {L4.ridge():.0f} FLOP/B", "style": "muted", "anchor": "start"},
        {"kind": "text", "x": 4.0, "y": 1500.0, "text": "memory-bound", "style": "muted", "anchor": "middle"},
        {"kind": "text", "x": 3000.0, "y": 3500.0, "text": "compute-bound", "style": "muted", "anchor": "middle"},
    ]
    return {
        "kind": "line",
        "title": ("The roofline of the H100 and the L4 in dense bf16 on log axes: attainable TFLOP/s against "
                  "arithmetic intensity, with the kernels of the table and the decode and prefill steps of "
                  "section 3 on the H100."),
        "width": 960, "height": 440,
        "x": {"label": "arithmetic intensity (FLOP/byte)", "scale": "log",
              "ticks": [0.1, 1, 10, 100, 1000, 10000], "min": 0.1, "max": 10000},
        "y": {"label": "attainable TFLOP/s", "scale": "log",
              "ticks": [0.01, 0.1, 1, 10, 100, 1000, 10000], "min": 0.01, "max": 10000},
        "series": [
            {"name": f"H100: {H100.tflops['bf16']:.0f} TFLOP/s, {H100.memory_tbs:g} TB/s",
             "x": hx, "y": hy, "style": "accent"},
            {"name": f"L4: {L4.tflops['bf16']:.0f} TFLOP/s, {L4.memory_tbs:g} TB/s",
             "x": lx, "y": ly, "style": "muted", "dash": True},
        ],
        "marks": marks,
        "legend": "bottom-right",
        "note": "computed by roofline.roofline.attainable, roofline.llm.decode and prefill (roofline-core)",
    }


# ---------------------------------------------------------------- §3.3: decode, weights against KV

def spec_decode_step() -> dict:
    """decode() of Llama-3.1-8B on an H100 at 2K context, and its weight stream and KV read (decode_split())."""
    cap = llm.max_batch_by_memory(M8, H100, 2048)
    xs = sorted(set(range(1, 8)) | set(range(8, cap + 1, 8)) | {cap})
    step, weights, kv = [], [], []
    for b in xs:
        s, sp = llm.decode(M8, H100, b, 2048), llm.decode_split(M8, H100, b, 2048)
        step.append(sig(s.time * 1e3))
        weights.append(sig(sp.gemms.time * 1e3))
        kv.append(sig(sp.attention.time * 1e3))
    place = {1: ("start", -10), 8: ("start", -12), 32: ("end", -10), 64: ("end", -10), 128: ("end", -10),
             cap: ("end", -6)}
    marks = []
    for b in (1, 8, 32, 64, 128, cap):
        s = llm.decode(M8, H100, b, 2048)
        anchor, dy = place[b]
        name = f"batch {b} (HBM limit)" if b == cap else f"batch {b}"
        marks.append({"kind": "point", "x": b, "y": sig(s.time * 1e3), "label": f"{name}: {s.time * 1e3:.2f} ms",
                      "style": "accent", "anchor": anchor, "dy": dy})
    return {
        "kind": "line",
        "title": ("One decode step of Llama-3.1-8B on an H100 in bf16 at 2K context against the batch: the weight "
                  "stream takes the same time at every batch, and the KV read grows with the batch up to the "
                  "HBM limit."),
        "width": 880, "height": 380,
        "x": {"label": "batch (sequences of 2K tokens)", "scale": "log",
              "ticks": [1, 2, 4, 8, 16, 32, 64, 128, 256], "min": 1, "max": 256},
        "y": {"label": "step time (ms)", "min": 0, "max": 24, "ticks": [0, 4, 8, 12, 16, 20, 24]},
        "series": [
            {"name": "step time", "x": xs, "y": step, "style": "accent"},
            {"name": "weight stream", "x": xs, "y": weights, "style": "plain", "dash": True},
            {"name": "KV read", "x": xs, "y": kv, "style": "warm"},
        ],
        "marks": marks,
        "legend": "top-left",
        "note": "computed by roofline.llm.decode and decode_split (roofline-core)",
    }


# ---------------------------------------------------------------- §5.2: the α-β model

def spec_allreduce() -> dict:
    """An all-reduce over 8 GPUs against its size: a ring and recursive doubling on NVLink 4, a ring on a NIC."""
    nv, nic = fabric.LINKS["nvlink4"], fabric.LINKS["ib-ndr"]
    p = 8
    xs = [sig(10 ** (4 + i / 4)) for i in range(21)] + [GIB]          # 10 KB to 1 GB, four points a decade, 1 GiB
    xs = sorted(set(xs))
    ser = [
        ("NVLink 4, ring", [sig(fabric.ring_allreduce_time(n, p, nv) * 1e3) for n in xs], "accent", False),
        ("NVLink 4, recursive doubling",
         [sig(fabric.recursive_doubling_allreduce_time(n, p, nv) * 1e3) for n in xs], "violet", True),
        ("400 Gb/s NIC, ring", [sig(fabric.ring_allreduce_time(n, p, nic) * 1e3) for n in xs], "warm", False),
    ]
    x_nv, x_nic = fabric.allreduce_crossover_bytes(p, nv), fabric.allreduce_crossover_bytes(p, nic)
    t_gib = fabric.ring_allreduce_time(GIB, p, nv)
    dec, pre = fabric.tp_allreduce_bytes(M70, 1), fabric.tp_allreduce_bytes(M70, 4096)
    marks = [
        {"kind": "point", "x": sig(x_nv), "y": sig(fabric.ring_allreduce_time(x_nv, p, nv) * 1e3),
         "label": f"crossover: {x_nv / 1e6:.1f} MB", "style": "accent", "anchor": "start", "dy": 16},
        {"kind": "point", "x": sig(x_nic), "y": sig(fabric.ring_allreduce_time(x_nic, p, nic) * 1e3),
         "label": f"crossover: {x_nic / 1e6:.1f} MB", "style": "warm", "anchor": "end", "dy": -9},
        {"kind": "point", "x": GIB, "y": sig(t_gib * 1e3), "label": f"1 GiB: {t_gib * 1e3:.2f} ms",
         "style": "accent", "anchor": "start", "dy": 4},
        {"kind": "vline", "x": dec, "label": f"70B decode message: {dec // 1024} KiB", "style": "muted"},
        {"kind": "vline", "x": pre, "label": f"70B prefill message: {pre / 1e6:.0f} MB", "style": "muted",
         "anchor": "end"},
    ]
    return {
        "kind": "line",
        "title": ("All-reduce time over 8 GPUs against the message size on log axes: a ring and recursive "
                  "doubling over NVLink 4, and a ring through 400 Gb/s NICs."),
        "width": 880, "height": 400,
        "x": {"label": "message size (bytes)", "scale": "log",
              "ticks": [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10],
              "tick_labels": ["10 KB", "100 KB", "1 MB", "10 MB", "100 MB", "1 GB", "10 GB"],
              "min": 1e4, "max": 1e10},
        "y": {"label": "all-reduce time (ms)", "scale": "log", "ticks": [0.001, 0.01, 0.1, 1, 10, 100],
              "min": 0.001, "max": 100},
        "series": [{"name": n, "x": xs, "y": ys, "style": st, "dash": d} for n, ys, st, d in ser],
        "marks": marks,
        "legend": "bottom-right",
        "note": "computed by roofline.fabric.ring_allreduce_time and recursive_doubling_allreduce_time (roofline-core)",
    }


# ---------------------------------------------------------------- §7.2: Young/Daly

def spec_checkpoint_waste() -> dict:
    """wasted_fraction() against the checkpoint interval for the 16,384-GPU job of §7.1, at its Young interval."""
    m_gpu = reliability.component_mtbf_from_observation(419, 54 * 24, 16384)      # the Llama 3 report, §7.1
    ms = reliability.cluster_mtbf(m_gpu, 16384) * 3600
    restart = 600                                                                  # the assumed 10-minute restart
    w = reliability.wasted_fraction
    xs_min = [sig(2 * 90 ** (i / 48)) for i in range(49)]                          # 2 to 180 min, log-spaced
    cases = [("δ = 60 s", 60, 0, "accent", False), ("δ = 10 s", 10, 0, "violet", False),
             ("δ = 60 s, R = 10 min", 60, restart, "warm", True)]
    series = [{"name": name, "x": xs_min, "y": [sig(100 * w(t * 60, d, ms, r)) for t in xs_min],
               "style": st, "dash": dash} for name, d, r, st, dash in cases]
    t60, t10 = reliability.young_daly_interval(60, ms), reliability.young_daly_interval(10, ms)
    marks = [
        {"kind": "point", "x": sig(t60 / 60), "y": sig(100 * w(t60, 60, ms)),
         "label": f"τ* = {t60 / 60:.1f} min: {w(t60, 60, ms):.1%}", "style": "accent", "anchor": "middle", "dy": 18},
        {"kind": "point", "x": sig(t10 / 60), "y": sig(100 * w(t10, 10, ms)),
         "label": f"τ* = {t10 / 60:.1f} min: {w(t10, 10, ms):.1%}", "style": "violet", "anchor": "middle", "dy": 18},
        {"kind": "point", "x": sig(t60 / 60), "y": sig(100 * w(t60, 60, ms, restart)),
         "label": f"with R: {w(t60, 60, ms, restart):.1%}", "style": "warm", "anchor": "middle", "dy": -10},
        {"kind": "vline", "x": 60, "label": "hourly", "style": "muted", "anchor": "end"},
        {"kind": "point", "x": 60, "y": sig(100 * w(3600, 60, ms)),
         "label": f"{w(3600, 60, ms):.1%}", "style": "accent", "anchor": "end", "dy": -6},
        {"kind": "point", "x": 60, "y": sig(100 * w(3600, 10, ms)),
         "label": f"{w(3600, 10, ms):.1%}", "style": "violet", "anchor": "start", "dy": 14},
    ]
    return {
        "kind": "line",
        "title": ("The share of wall-clock time that a 16,384-GPU job loses against its checkpoint interval, for "
                  "a 60 s and a 10 s checkpoint and for a 60 s checkpoint with a 10-minute restart."),
        "width": 880, "height": 400,
        "x": {"label": "checkpoint interval τ (min)", "scale": "log", "ticks": [2, 5, 10, 20, 60, 180],
              "min": 2, "max": 180},
        "y": {"label": "waste (% of wall-clock time)", "min": 0, "max": 30, "ticks": [0, 5, 10, 15, 20, 25, 30]},
        "series": series,
        "marks": marks,
        "legend": "bottom-right",
        "note": "computed by roofline.reliability.wasted_fraction and young_daly_interval (roofline-core)",
    }


FIGURES = {
    "roofline-h100-bf16.svg": spec_roofline,
    "decode-step-weights-vs-kv.svg": spec_decode_step,
    "allreduce-time-vs-message-size.svg": spec_allreduce,
    "checkpoint-waste-vs-interval.svg": spec_checkpoint_waste,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
