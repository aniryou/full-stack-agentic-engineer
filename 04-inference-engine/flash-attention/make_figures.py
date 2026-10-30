#!/usr/bin/env python3
"""make_figures.py: the charts of flash-attention-deep-dive.md, computed by fa_calculators.py
(tools/orchestration/FIGURE-STYLE.md §7).

    python3 make_figures.py        # writes figures/<name>.svg beside the deep dive

Each chart is a spec builder, ``spec_<name>()``, that computes every value with fa_calculators from the inputs the
deep dive names (the device presets in ``fa_calculators.DEVICES`` and ``PER_SM_CLOCK``, one head at d = 128 in bf16,
FA2's 128-row Q tile, the 8B-shaped model of §6.5 with 32 layers, 8 KV heads and 8.03 B parameters), never from a
number copied out of the prose. The marks are the deep dive's numbers, labelled as the deep dive prints them.
test_figures.py pins each chart with svgchart.check_figure(): it fails when fa_calculators computes other values, or
when a file is not what svgchart.render() makes of the spec stored in it.

  deep-dive-naive-roofline.svg     §1.4: attainable TFLOP/s against intensity on an H100 and an L4, with the naive
                                   schedule and the compulsory traffic at N = 4,096
                                   (fa_calculators.roofline, naive_traffic, compulsory_bytes)
  deep-dive-hbm-bytes-vs-n.svg     §3.4: HBM bytes of one head against N, naive, FA2 order (no L2) and compulsory
                                   (fa_calculators.naive_traffic, flash_traffic, compulsory_bytes)
  deep-dive-clocks-per-score.svg   §4.3: clocks per score per SM on each unit, A100, H100, B200
                                   (fa_calculators.clocks_per_score)
  deep-dive-decode-kv-share.svg    §6.5: the share of a decode step's bytes that the KV read takes, against the
                                   cached tokens in the batch (fa_calculators.decode_kv_share,
                                   decode_kv_crossover_tokens)

Standard library + numpy (through fa_calculators). The renderer is tools/orchestration/svgchart.py, found by walking
up from this file to the repository root.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
DOC = HERE / "flash-attention-deep-dive.md"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fa_calculators as fc  # noqa: E402


def _repo_root() -> pathlib.Path:
    """The folder that holds tools/orchestration/svgchart.py, found by walking up from this file."""
    for p in (HERE, *HERE.parents):
        if (p / "tools" / "orchestration" / "svgchart.py").is_file():
            return p
    raise FileNotFoundError("tools/orchestration/svgchart.py not found above " + str(HERE))


def _load_svgchart():
    if "svgchart" in sys.modules:
        return sys.modules["svgchart"]
    spec = importlib.util.spec_from_file_location("svgchart", _repo_root() / "tools" / "orchestration" / "svgchart.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svgchart"] = mod
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()

# --- the inputs the deep dive names ------------------------------------------------------------------------------
D = 128                                  # head dimension of every worked example (bf16, b = 2)
N_4K, N_32K = 4096, 32768                # §1.4 and §3.4: the two sequence lengths of the worked numbers
BLOCK_M, BLOCK_N = 128, 64               # §3.4: FA2 order, outer Q, B_r = 128 (FA2's 128 x 64 tile, §4.5)
N_LAYERS, N_KV_HEADS = 32, 8             # §6.5: the 8B-shaped model, 32 layers, 8 KV heads, d = 128
PARAMS = 8.03e9                          # §6.5: "It has 8.03 B parameters" (bf16: 2 bytes each)
BF16 = 2


def _r(v: float) -> float:
    """Six significant digits: the spec stays short, and check_figure() compares within 1e-6."""
    return float(f"{v:.6g}")


def _mb(v: float) -> str:
    """Bytes as the §3.4 table prints them: 138.4 MB, 8,623 MB."""
    mb = v / 1e6
    return f"{mb:.1f} MB" if mb < 1000 else f"{mb:,.0f} MB"


# --- 1. §1.4: the naive schedule and the compulsory traffic on the roofline --------------------------------------
def spec_naive_roofline() -> dict:
    naive = fc.naive_traffic(N_4K, D)
    fused_bytes = fc.compulsory_bytes(N_4K, D)
    fused_flops = fc.attention_flops(N_4K, N_4K, D)
    i_naive, i_fused = naive["intensity"], fused_flops / fused_bytes
    lo, hi = 10.0, 10_000.0
    series, marks = [], []
    for key, name, style in (("H100", "H100 SXM roof", "plain"), ("L4", "L4 roof", "violet")):
        dev = fc.DEVICES[key]
        xs = [lo, dev.ridge, hi]
        # the roof at intensity I: roofline() of I FLOPs per byte, on one byte
        ys = [fc.roofline(x, 1.0, key)["attainable_tflops"] for x in xs]
        series.append({"name": name, "x": [_r(x) for x in xs], "y": [_r(y) for y in ys], "style": style})
        at_naive = fc.roofline(naive["flops"], naive["bytes"], key)
        at_fused = fc.roofline(fused_flops, fused_bytes, key)
        marks += [
            {"kind": "point", "x": _r(i_naive), "y": _r(at_naive["attainable_tflops"]),
             "label": f"{key}: {at_naive['attainable_tflops']:.3g} TFLOP/s ({at_naive['fraction_of_peak']:.0%} of peak)",
             "style": "warm", "anchor": "start", "dy": 16},
            {"kind": "point", "x": _r(dev.ridge), "y": _r(dev.bf16_tflops), "label": f"ridge {dev.ridge:.0f}",
             "style": style, "anchor": "start", "dy": 18},
            {"kind": "point", "x": _r(i_fused), "y": _r(at_fused["attainable_tflops"]),
             "label": f"{at_fused['t_floor_s'] * 1e6:.2g} µs ({key})", "style": "accent", "anchor": "start", "dy": -9},
        ]
    marks = [
        {"kind": "vline", "x": _r(i_naive), "label": f"naive, three kernels: {i_naive:.1f} FLOP/B", "style": "warm"},
        {"kind": "vline", "x": _r(i_fused), "label": f"compulsory traffic: {i_fused:,.0f} FLOP/B", "style": "accent",
         "anchor": "end"},
    ] + marks
    return {
        "kind": "line",
        "title": ("Naive attention at 62.1 FLOP/B sits below the ridge of the H100 (295) and the L4 (403), and the "
                  "compulsory traffic of a fused schedule at 2,040 FLOP/B is past both."),
        "height": 400,
        "x": {"label": "arithmetic intensity (FLOP/B), one head, N = 4,096, d = 128, bf16", "scale": "log",
              "ticks": [10, 100, 1000, 10000], "min": lo, "max": hi},
        "y": {"label": "attainable TFLOP/s", "scale": "log", "ticks": [1, 10, 100, 1000],
              "min": 1, "max": 3000},
        "series": series,
        "marks": marks,
        "legend": "bottom-right",
        "note": "computed by fa_calculators.roofline, naive_traffic, compulsory_bytes; peaks are datasheet values (verify)",
    }


# --- 2. §3.4: HBM bytes of one head against the sequence length -------------------------------------------------
def spec_hbm_bytes_vs_n() -> dict:
    ns = [1024 * 2 ** k for k in range(7)]                       # 1k .. 64k
    naive = [fc.naive_traffic(n, D)["bytes"] for n in ns]
    fa2 = [fc.flash_traffic(n, D, BLOCK_M, BLOCK_N, schedule="fa2")["bytes"] for n in ns]
    comp = [fc.compulsory_bytes(n, D) for n in ns]
    marks = []
    above_left, below_right = ("end", -9), ("start", 16)
    for n in (N_4K, N_32K):
        # each label on the side of its point away from the next line, as the §3.4 table prints the bytes
        for b, style, (anchor, dy) in (
                (fc.naive_traffic(n, D)["bytes"], "warm", above_left),
                (fc.flash_traffic(n, D, BLOCK_M, BLOCK_N, schedule="fa2")["bytes"], "plain", below_right),
                (fc.compulsory_bytes(n, D), "accent", above_left if n == N_4K else below_right)):
            marks.append({"kind": "point", "x": n, "y": _r(b / 1e6), "label": _mb(b), "style": style,
                          "anchor": anchor, "dy": dy})
    return {
        "kind": "line",
        "title": ("HBM bytes of one head against the sequence length, d = 128, bf16: the naive schedule and the FA2 "
                  "order without L2 increase as N squared, and the compulsory traffic increases as N."),
        "x": {"label": "sequence length N (tokens)", "scale": "log", "ticks": ns,
              "tick_labels": [f"{n // 1024}k" for n in ns], "min": 1024, "max": ns[-1]},
        "y": {"label": "HBM bytes per head (MB)", "scale": "log", "ticks": [1, 10, 100, 1000, 10000, 100000],
              "min": 0.5, "max": 100000},
        "series": [
            {"name": "naive, three kernels", "x": ns, "y": [_r(b / 1e6) for b in naive], "style": "warm"},
            {"name": "FA2 order, B_r = 128, no L2", "x": ns, "y": [_r(b / 1e6) for b in fa2], "style": "plain"},
            {"name": "compulsory: Q, K, V in, O and LSE out", "x": ns, "y": [_r(b / 1e6) for b in comp],
             "style": "accent"},
        ],
        "marks": marks,
        "legend": "top-left",
        "note": "computed by fa_calculators.naive_traffic, flash_traffic(schedule=\"fa2\"), compulsory_bytes",
    }


# --- 3. §4.3: clocks per score per SM on each unit ---------------------------------------------------------------
GPUS = ("A100", "H100", "B200")


def spec_clocks_per_score() -> dict:
    per = {g: fc.clocks_per_score(g, d=D) for g in GPUS}
    return {
        "kind": "bar",
        "title": ("Clocks per score per SM when each unit runs alone, d = 128: the tensor-core bar halves from A100 to "
                  "H100 to B200 while the EX2 bar stays at 0.0625."),
        "categories": list(GPUS),
        "x": {"tick_labels": [f"{g}\nEX2 = {per[g]['ex2_vs_mma']:.0%} of MMA time" for g in GPUS]},
        "y": {"label": "SM clocks per score, d = 128", "min": 0, "max": 0.3,
              "ticks": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
        "series": [
            {"name": "tensor cores", "y": [per[g]["mma"] for g in GPUS], "style": "accent"},
            {"name": "EX2 on MUFU", "y": [per[g]["ex2"] for g in GPUS], "style": "warm"},
            # rounded to the three decimals the §4.3 table prints (0.078, 0.039), so the bar labels match it
            {"name": "about 5 FP32 instructions", "y": [round(per[g]["fp32"], 3) for g in GPUS], "style": "plain"},
        ],
        "legend": "top-right",
        "note": "computed by fa_calculators.clocks_per_score; the B200 row is an estimate (verify)",
    }


# --- 4. §6.5: the share of a decode step's bytes that the KV read takes ------------------------------------------
ROWS = ((1, 2048), (1, 32768), (32, 2048), (1, 131072), (8, 32768))       # the §6.5 table (8 x 32k = 32 x 8k)


def _share(tokens: int) -> float:
    """The KV read as a percentage of the step's bytes (fa_calculators.decode_kv_share)."""
    return 100.0 * fc.decode_kv_share(tokens, N_LAYERS, N_KV_HEADS, D, PARAMS * BF16, BF16)


def _pct(v: float) -> str:
    """A share as the §6.5 table prints it: 1.6%, 21%."""
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def _ctx(tokens: int) -> str:
    return f"{tokens // 1024}k"


def spec_decode_kv_share() -> dict:
    xs = [round(1024 * 2 ** (k / 4)) for k in range(4 * 9 + 1)]           # 1k .. 512k, four points per doubling
    cross = fc.decode_kv_crossover_tokens(N_LAYERS, N_KV_HEADS, D, PARAMS * BF16, BF16)
    marks = [{"kind": "vline", "x": _r(cross), "label": f"KV bytes = weight bytes: ≈ {round(cross, -2):,.0f} tokens",
              "style": "warm", "anchor": "end", "label_pos": "top"}]
    for (batch, ctx), anchor, dy in zip(ROWS, ("start", "end", "end", "start", "end"), (-9, -9, -9, 16, -9)):
        tokens = batch * ctx
        label = f"{batch} × {_ctx(ctx)}: {_pct(_share(tokens))}"
        marks.append({"kind": "point", "x": tokens, "y": _r(_share(tokens)), "label": label, "style": "accent",
                      "anchor": anchor, "dy": dy})
    return {
        "kind": "line",
        "title": ("The share of the bytes of one decode step that the KV read takes, against the cached tokens in "
                  "the batch, for the 8B-shaped model of section 6.5 in bf16."),
        "x": {"label": "cached tokens in the batch (batch × context)", "scale": "log",
              "ticks": [1024 * 2 ** k for k in range(10)], "tick_labels": [_ctx(1024 * 2 ** k) for k in range(10)],
              "min": 1024, "max": 1024 * 2 ** 9},
        "y": {"label": "attention share of the step's bytes (%)", "min": 0, "max": 100,
              "ticks": [0, 20, 40, 60, 80, 100]},
        "series": [{"name": "KV read / (KV read + weights read)", "x": xs, "y": [_r(_share(x)) for x in xs],
                    "style": "accent"}],
        "marks": marks,
        "legend": "none",
        "note": "computed by fa_calculators.decode_kv_share, decode_kv_crossover_tokens; weights: 8.03 B parameters in bf16",
    }


FIGURES = {
    "deep-dive-naive-roofline.svg": spec_naive_roofline,
    "deep-dive-hbm-bytes-vs-n.svg": spec_hbm_bytes_vs_n,
    "deep-dive-clocks-per-score.svg": spec_clocks_per_score,
    "deep-dive-decode-kv-share.svg": spec_decode_kv_share,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
