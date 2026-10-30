"""make_figures.py: the charts of PRIMER.md, computed by capacity.py (tools/orchestration/FIGURE-STYLE.md §7).

    python3 make_figures.py        # writes figures/<name>.svg beside PRIMER.md

Every value comes from a capacity.py function and the inputs that PRIMER.md names: the H100 preset, the 24B dense
model preset, fp8 at 50% MFU for prefill, the bank example's workload (10,000 staff, 10% active at peak, one request
per 2 minutes, 1,500 tokens in, 300 out). The marks are the primer's numbers, labelled as the primer prints them.
test_figures.py pins each chart with svgchart.check_figure(): it fails when capacity.py computes other values, or
when a file is not what svgchart.render() makes of the spec stored in it. Standard library only.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
sys.path.insert(0, str(HERE))
import capacity as c  # noqa: E402


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

# --- the inputs PRIMER.md names --------------------------------------------------------------------------------
MODEL = c.MISTRAL_SMALL                   # the 24B dense model of the primer and the bank example
H100 = c.GPUS["H100"]
STAFF, PEAK_ACTIVE, REQ_PER_MIN = 10_000, 0.10, 0.5   # the bank: 10% active at peak, ~1 request / 2 min
IN_TOK, OUT_TOK = 1500, 300                           # the bank: 1,500 tokens in, 300 out
AVG_CTX = IN_TOK + OUT_TOK // 2           # the KV holds the prompt and the answer so far (as worked_example.py)
RPS = STAFF * PEAK_ACTIVE * REQ_PER_MIN / 60
K_TICKS = [1000 * 2 ** k for k in range(8)]            # 1K .. 128K tokens, as the primer counts them
K_LABELS = [f"{t // 1000}K" for t in K_TICKS]


def _r(v: float) -> float:
    """Six significant digits: the spec stays short, and check_figure() compares within 1e-6."""
    return float(f"{v:.6g}")


def _tokens_grid(start: int = 1000) -> list[int]:
    """Four points per doubling from `start` to 128K tokens (the curves are smooth on a log axis)."""
    n = round(4 * math.log2(128_000 / start))
    return [round(start * 2 ** (k / 4)) for k in range(n + 1)]


def _spare_gb(dtype: str) -> float:
    return c.usable_hbm_gb(H100) - c.weight_memory_gb(MODEL.params_b, dtype)


def _live_sessions() -> float:
    """The bank example's live sessions: Little's law over the request duration (TTFT + 300 tokens at 40 ms)."""
    return c.concurrency(RPS, c.request_duration_s(MODEL.active_b, IN_TOK, OUT_TOK, H100))


# --- 1. prefill: TTFT against the prompt length (the formulas, step 4) -----------------------------------------
def spec_ttft_vs_prompt() -> dict:
    xs = _tokens_grid()
    with_attn = [_r(c.ttft_s(MODEL.active_b, s, H100, model=MODEL)) for s in xs]
    weights = [_r(c.ttft_s(MODEL.active_b, s, H100)) for s in xs]
    return {
        "kind": "line",
        "title": "TTFT on one H100 against the prompt length, for the 24B model in fp8 at 50% MFU, with and without the attention term.",
        "x": {"label": "prompt length (tokens)", "scale": "log", "ticks": K_TICKS, "tick_labels": K_LABELS},
        "y": {"label": "TTFT (s)", "scale": "log", "ticks": [0.1, 1, 10], "tick_labels": ["0.1", "1", "10"],
              "min": 0.02, "max": 20},
        "series": [
            {"name": "weights + attention", "x": xs, "y": with_attn, "style": "accent"},
            {"name": "weights only", "x": xs, "y": weights, "style": "muted", "dash": True},
        ],
        "marks": [
            {"kind": "hline", "y": 1, "label": "good chat UX: TTFT < 1 s", "style": "warm", "anchor": "start"},
            {"kind": "point", "x": 2000, "y": _r(c.ttft_s(MODEL.active_b, 2000, H100)), "label": "2K prompt: ~0.1 s",
             "style": "muted", "anchor": "start", "dy": 16},   # the prose's ~0.1 s is the weights-only TTFT
            {"kind": "point", "x": 32000, "y": _r(c.ttft_s(MODEL.active_b, 32000, H100, model=MODEL)),
             "label": "32K RAG prompt: ~1.9 s", "style": "accent", "anchor": "end", "dy": -9},
        ],
        "legend": "top-left",
        "note": "computed by capacity.ttft_s (H100, fp8, 50% MFU)",
    }


# --- 2. the bank example: sessions per GPU against the context length ------------------------------------------
def spec_sessions_vs_context() -> dict:
    xs = _tokens_grid(500)              # from 500 tokens: room for the bank labels left of 128K
    per = {dt: [_r(c.max_concurrent_sessions(_spare_gb(dt), MODEL, s, dt)) for s in xs] for dt in ("bf16", "fp8")}
    bank = {dt: _r(c.max_concurrent_sessions(_spare_gb(dt), MODEL, AVG_CTX, dt)) for dt in ("bf16", "fp8")}
    return {
        "kind": "line",
        "title": "Concurrent sessions on one H100 against the context length, for the 24B model in bf16 and in fp8, with the ~100 live sessions of the bank example.",
        "x": {"label": "context length per session (tokens)", "scale": "log", "ticks": [500, *K_TICKS],
              "tick_labels": ["500", *K_LABELS]},
        "y": {"label": "concurrent sessions per H100", "scale": "log", "ticks": [1, 10, 100, 1000],
              "tick_labels": ["1", "10", "100", "1,000"], "min": 1, "max": 2000},
        "series": [
            {"name": "fp8 (weights and KV)", "x": xs, "y": per["fp8"], "style": "accent"},
            {"name": "bf16 (weights and KV)", "x": xs, "y": per["bf16"], "style": "plain"},
        ],
        "marks": [
            {"kind": "hline", "y": _r(_live_sessions()), "label": "~100 live sessions", "style": "warm"},
            {"kind": "point", "x": AVG_CTX, "y": bank["fp8"], "label": "fp8: ~355 sessions, 1 GPU", "style": "accent",
             "anchor": "start", "dy": -9},
            {"kind": "point", "x": AVG_CTX, "y": bank["bf16"], "label": "bf16: ~89 sessions, 2 GPUs", "style": "plain",
             "anchor": "end", "dy": 18},
        ],
        "legend": "bottom-left",
        "note": "computed by capacity.max_concurrent_sessions; the hline by capacity.concurrency",
    }


# --- 3. decode: throughput against the batch size (the formulas, step 3; numbers to carry) ---------------------
def _batches(limit: float) -> list[int]:
    """Powers of two up to the last whole batch that fits, then that batch."""
    top = math.floor(limit)
    xs = [b for b in (2 ** k for k in range(10)) if b < top]
    return xs + [top]


def spec_decode_vs_batch() -> dict:
    roof = c.roofline_batch(H100)
    series, ends = [], {}
    for dt, style in (("fp8", "accent"), ("bf16", "plain")):
        # a batch ends where the KV cache fills the spare HBM, or at the compute roofline, whichever comes first
        xs = _batches(min(c.max_concurrent_sessions(_spare_gb(dt), MODEL, AVG_CTX, dt), roof))
        pts = [c.decode_aggregate(MODEL, H100, b, AVG_CTX, dt) for b in xs]
        series.append({"name": f"aggregate, {dt}", "x": xs, "y": [_r(a) for a, _ in pts], "style": style})
        series.append({"name": f"per user, {dt}", "x": xs, "y": [_r(u) for _, u in pts], "style": style, "dash": True})
        ends[dt] = (xs[-1], pts[-1][0])
    live = int(_live_sessions())
    tpot_target = 1000 / 40                 # TPOT <= 40 ms is >= 25 tok/s per user
    return {
        "kind": "line",
        "title": "Decode throughput on one H100 against the batch size, for the 24B model at the context of the bank example: the total and the rate per user.",
        "x": {"label": "batch size (sequences)", "scale": "log", "ticks": [1, 2, 4, 8, 16, 32, 64, 128, 256], "max": 400},
        "y": {"label": "decode throughput (tok/s)", "scale": "log", "ticks": [10, 100, 1000, 10000],
              "tick_labels": ["10", "100", "1,000", "10,000"], "min": 10, "max": 40000},
        "series": series,
        "marks": [
            {"kind": "vline", "x": _r(roof), "label": "compute roofline: ~batch 300", "style": "warm", "anchor": "end"},
            {"kind": "hline", "y": tpot_target, "style": "muted"},
            {"kind": "text", "x": 16, "y": _r(tpot_target / 1.6), "text": "≥25 tok/s/user (TPOT ≤ 40 ms)", "style": "muted"},
            {"kind": "text", "x": _r(ends["bf16"][0] * 1.1), "y": _r(ends["bf16"][1] / 1.15), "text": "memory limit, bf16",
             "style": "plain", "anchor": "start"},
            {"kind": "point", "x": 1, "y": _r(c.decode_tok_s_single(c.weight_memory_gb(MODEL.params_b, "bf16"), H100)),
             "label": "~70 tok/s single-stream ceiling", "style": "plain", "anchor": "start", "dy": 18},
            {"kind": "point", "x": live, "y": _r(c.decode_aggregate(MODEL, H100, live, AVG_CTX, "fp8")[0]),
             "label": "~100 sessions in fp8: ~9,000 tok/s", "style": "accent", "anchor": "end", "dy": -9},
        ],
        "legend": "top-left",
        "note": "computed by capacity.decode_aggregate, roofline_batch, max_concurrent_sessions, decode_tok_s_single",
    }


FIGURES = {
    "ttft-vs-prompt.svg": spec_ttft_vs_prompt,
    "sessions-vs-context.svg": spec_sessions_vs_context,
    "decode-vs-batch.svg": spec_decode_vs_batch,
}


def main() -> None:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
