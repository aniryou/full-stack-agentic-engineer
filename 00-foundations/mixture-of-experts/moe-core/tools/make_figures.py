#!/usr/bin/env python3
"""The charts of the topic's PRIMER.md, computed by moecore (tools/orchestration/FIGURE-STYLE.md §7).

Each chart is a spec builder, ``spec_<name>() -> dict``, that computes every value with the core's own functions
from the inputs the primer names (the configs in ``sizing.MODELS``, the H200 preset in ``touched.DEVICES``, a 1K
context, the illustrative links in ``ep.LINKS``, 8 GPUs). ``tools/orchestration/svgchart.py`` renders the spec into
the house style, and ``main()`` writes each chart into ``../figures/`` beside the primer. ``tests/test_figures.py``
calls ``svgchart.check_figure()`` with the same builders, so a committed chart fails the test when the core
computes other values.

Only closed forms and roofline bounds are charted: the bytes of a decode step against the batch, the share of the
experts a layer touches under uniform routing, and the alpha-beta time of a dispatch all-to-all. The Zipf-skewed
Monte Carlo numbers and the toy trainer are not charted.

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
from moecore import ep as E, sizing as S, touched as T  # noqa: E402

# The inputs the primer names.
MIX, Q3, DS = S.MODELS["mixtral-8x7b"], S.MODELS["qwen3-30b-a3b"], S.MODELS["deepseek-v3"]
H200 = T.DEVICES["h200"]
CONTEXT = 1024                                   # §5 and the one-minute version: an H200 at 1K context
EP_GPUS = 8                                      # §6.2: one MoE layer on 8 GPUs, a direct all-to-all
NVLINK, IB = E.LINKS["nvlink4"], E.LINKS["ib-ndr"]


def _sig(v: float) -> float:
    """Nine significant digits: a shorter <desc>, far inside check_figure()'s tolerance."""
    return float(f"{float(v):.9g}")


def _first_batch(cfg, touched: float) -> int:
    """The smallest batch at which one layer of `cfg` touches at least `touched` experts (uniform routing)."""
    b = 1
    while T.experts_touched(cfg.n_experts, cfg.top_k, b) < touched:
        b += 1
    return b


def _log2_grid(hi: int, per_octave: int, extra=()) -> list[int]:
    """Integer batches spread evenly on a log axis from 1 to `hi`, plus the batches the marks name."""
    n = per_octave * (hi.bit_length() - 1)
    return sorted({round(2 ** (i / per_octave)) for i in range(n + 1)} | set(extra))


# ------------------------------------------------------------------ the one-minute version and §8: bytes per step

def spec_decode_step_bytes_vs_batch() -> dict:
    dense = S.dense_equivalent(MIX)
    batches = list(range(1, 65))
    step = lambda cfg, b: T.decode_step(cfg, H200, b, CONTEXT)
    gb = lambda cfg, b: _sig(step(cfg, b).bytes / 1e9)
    s1, s64 = step(MIX, 1), step(MIX, 64)
    hbm = S.weight_bytes(MIX) / 1e9
    marks = [
        {"kind": "hline", "y": _sig(hbm), "label": f"all weights in HBM: {hbm:.1f} GB", "style": "warm",
         "anchor": "start"},
        {"kind": "point", "x": 1, "y": gb(MIX, 1), "style": "accent", "anchor": "start", "dy": 20,
         "label": f"batch 1: {T.experts_touched(8, 2, 1):.2f} of 8 experts per layer, {s1.time * 1e3:.2f} ms"},
        {"kind": "point", "x": 64, "y": gb(MIX, 64), "style": "accent", "anchor": "end", "dy": -10,
         "label": f"batch 64: {T.experts_touched(8, 2, 64):.2f} of 8, {s64.bytes / 1e9:.1f} GB, {s64.time * 1e3:.2f} ms"},
    ]
    for b in (4, 16):
        moe, den = step(MIX, b), step(dense, b)
        marks += [
            {"kind": "point", "x": b, "y": gb(MIX, b), "label": f"{moe.bytes / 1e9:.1f} GB", "style": "accent",
             "anchor": "end"},
            {"kind": "point", "x": b, "y": gb(dense, b), "label": f"{den.bytes / 1e9:.1f} GB", "style": "muted",
             "anchor": "end"},
            {"kind": "text", "x": _sig(b * 1.07), "y": _sig((moe.bytes + den.bytes) / 2e9),
             "text": f"{moe.time / den.time:.1f}× slower", "style": "muted", "anchor": "start"},
        ]
    return {
        "kind": "line",
        "title": "Bytes one decode step reads against the batch for Mixtral-8x7B and for a dense model of its active "
                 "size, on an H200 at 1K context.",
        "height": 400,
        "x": {"label": "batch (sequences in one decode step, log scale)", "scale": "log",
              "ticks": [1, 2, 4, 8, 16, 32, 64]},
        "y": {"label": "GB one decode step reads (weights + KV)", "min": 0, "max": 120, "ticks": [0, 25, 50, 75, 100]},
        "series": [
            {"name": f"Mixtral-8x7B: {MIX.total() / 1e9:.2f}B total, {MIX.active() / 1e9:.2f}B active",
             "x": batches, "y": [gb(MIX, b) for b in batches], "style": "accent"},
            {"name": f"dense model of the active size ({dense.total() / 1e9:.1f}B)",
             "x": batches, "y": [gb(dense, b) for b in batches], "style": "muted"},
        ],
        "marks": marks,
        "legend": "bottom-right",
        "note": "computed by moecore.touched.decode_step and sizing.dense_equivalent (H200, 1K context, simulated)",
    }


# ------------------------------------------------------------------ §5: which experts a step touches

def spec_experts_touched_vs_batch() -> dict:
    models = [(MIX, "accent"), (Q3, "plain"), (DS, "muted")]
    b_mix, b_q3 = _first_batch(MIX, 7.5), _first_batch(Q3, 120)      # §5: 7.5 of 8 and 120 of 128
    batches = _log2_grid(256, 8, extra=(8, 16, 32, 128, 256, b_mix, b_q3))
    share = lambda cfg, b: _sig(100 * T.experts_touched(cfg.n_experts, cfg.top_k, b) / cfg.n_experts)
    series = [{"name": f"{cfg.name}: E = {cfg.n_experts}, k = {cfg.top_k}", "x": batches,
               "y": [share(cfg, b) for b in batches], "style": st} for cfg, st in models]
    n = lambda cfg, b: T.experts_touched(cfg.n_experts, cfg.top_k, b)
    # §5: "Mixtral touches 7.5 of its 8 experts per layer from batch 10, Qwen3 120 of 128 from batch 43", and the
    # DeepSeek-V3 counts at 8, 32 and 128 tokens; §3.7 and §5: Qwen3 touches 82.4 of 128 at batch 16
    marks = [
        {"kind": "point", "x": b_mix, "y": share(MIX, b_mix), "style": "accent"},
        {"kind": "text", "x": b_mix, "y": 103, "text": f"7.5 of 8 from batch {b_mix}", "style": "accent",
         "anchor": "end"},
        {"kind": "point", "x": 16, "y": share(Q3, 16), "style": "plain", "anchor": "end", "dy": -6,
         "label": f"{n(Q3, 16):.1f} of 128 at batch 16"},
        {"kind": "point", "x": b_q3, "y": share(Q3, b_q3), "style": "plain"},
        {"kind": "text", "x": b_q3, "y": 103, "text": f"120 of 128 from batch {b_q3}", "style": "plain",
         "anchor": "end"},
        {"kind": "point", "x": 8, "y": share(DS, 8), "style": "muted", "anchor": "start", "dy": 16,
         "label": f"{n(DS, 8):.1f} of 256 at batch 8"},
        {"kind": "point", "x": 32, "y": share(DS, 32), "style": "muted", "anchor": "start", "dy": 16,
         "label": f"{n(DS, 32):.1f} at batch 32"},
        {"kind": "point", "x": 128, "y": share(DS, 128), "style": "muted"},
        {"kind": "text", "x": 128, "y": 88, "text": f"{n(DS, 128):.1f} at batch 128", "style": "muted",
         "anchor": "middle"},
    ]
    return {
        "kind": "line",
        "title": "The share of the experts that one layer touches, against the batch, for Mixtral-8x7B, "
                 "Qwen3-30B-A3B and DeepSeek-V3.",
        "height": 400,
        "x": {"label": "batch (tokens in the step, log scale)", "scale": "log", "ticks": [1, 4, 16, 64, 256]},
        "y": {"label": "experts touched per layer (% of E)", "min": 0, "max": 108, "ticks": [0, 25, 50, 75, 100],
              "tick_labels": ["0%", "25%", "50%", "75%", "100%"]},
        "series": series,
        "marks": marks,
        "legend": "bottom-right",
        "note": "computed by moecore.touched.experts_touched (uniform routing)",
    }


# ------------------------------------------------------------------ §6.2: dispatch and combine

def spec_dispatch_time_vs_tokens() -> dict:
    tokens = _log2_grid(8192, 4, extra=(8, 256, 4096))
    t_us = lambda n, link: E.a2a_time(E.dispatch_bytes(n, MIX.top_k, MIX.d_model), EP_GPUS, link) * 1e6
    series = [{"name": name, "x": tokens, "y": [_sig(t_us(n, link)) for n in tokens], "style": st}
              for name, link, st in (("NVLink 4", NVLINK, "accent"), ("400 Gb/s IB", IB, "warm"))]
    pt = lambda n, link, st, anchor, dy, label: {"kind": "point", "x": n, "y": _sig(t_us(n, link)), "style": st,
                                                  "anchor": anchor, "dy": dy, "label": label}
    marks = [
        {"kind": "hline", "y": NVLINK.alpha_us, "label": f"α = {NVLINK.alpha_us:g} µs (NVLink 4)", "style": "muted",
         "anchor": "end"},
        pt(8, NVLINK, "accent", "start", 18, f"8 tokens: {t_us(8, NVLINK):.1f} µs"),
        pt(8, IB, "warm", "end", -9, f"8 tokens: {t_us(8, IB):.1f} µs"),
        pt(256, NVLINK, "accent", "start", 16, f"256 tokens: {t_us(256, NVLINK):.0f} µs direct"),
        pt(4096, NVLINK, "accent", "end", -9, f"{t_us(4096, NVLINK):,.1f} µs"),
        pt(4096, IB, "warm", "end", -9, f"4,096 tokens: {t_us(4096, IB):,.1f} µs"),
    ]
    return {
        "kind": "line",
        "title": "Time of one direct Mixtral dispatch all-to-all on 8 GPUs against the tokens per GPU, on NVLink 4 "
                 "and on 400 Gb/s InfiniBand.",
        "height": 400,
        "x": {"label": "tokens per GPU (log scale)", "scale": "log", "min": 1, "max": 8192, "ticks": [1, 8, 64, 512, 4096],
              "tick_labels": ["1", "8", "64", "512", "4,096"]},
        "y": {"label": "dispatch time per MoE layer (µs, log scale)", "scale": "log", "min": 1, "max": 10_000,
              "ticks": [1, 10, 100, 1000, 10_000], "tick_labels": ["1", "10", "100", "1,000", "10,000"]},
        "series": series,
        "marks": marks,
        "legend": "top-left",
        "note": "computed by moecore.ep.a2a_time and ep.dispatch_bytes (hidden 4,096, top-2, BF16, simulated)",
    }


FIGURES = {
    "decode-step-bytes-vs-batch.svg": spec_decode_step_bytes_vs_batch,
    "experts-touched-vs-batch.svg": spec_experts_touched_vs_batch,
    "dispatch-time-vs-tokens.svg": spec_dispatch_time_vs_tokens,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
