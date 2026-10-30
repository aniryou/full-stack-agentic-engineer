#!/usr/bin/env python3
"""The kv-cache primer's charts, computed by kerncore (tools/orchestration/FIGURE-STYLE.md §7).

    python3 tools/make_figures.py        # writes every chart into ../kv-cache/figures/

Each chart is a spec builder, ``spec_<name>()``, that computes every value with ``kerncore.kv`` from the inputs the
primer names (the attention shapes in ``kv.MODELS``, the H100's 80 GB, ~1,000 TFLOP/s and 3.35 TB/s, fp16 and fp8),
never from a number copied out of the prose. The marks are the primer's pinned points, labelled as the primer prints
them. ``tests/test_figures.py`` calls ``svgchart.check_figure()`` with the same builders, so a committed chart fails
the test when kerncore computes other values.

  kv-bytes-vs-tokens.svg     §4: the KV cache against the tokens it holds (seq_len × batch), GQA against full MHA,
                             next to the fixed weights            (kv.kv_cache_bytes, kv.weight_bytes)
  decode-intensity.svg       §3: decode attention's FLOP per byte against g, fp16 and fp8, under the H100's ridge
                                                                  (kv.decode_intensity, kv.ridge)
  sessions-per-gpu.svg       §7 (numbers of §4): sessions on one 80 GB H100 against the context length, fp16 and fp8
                                                                  (kv.sessions_per_gpu)

Standard library + numpy (through kerncore). The renderer is tools/orchestration/svgchart.py, found by walking up
from this file to the repository root.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORE = HERE.parents[1]                                   # 04-inference-engine/kernel-core
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from kerncore import kv                                  # noqa: E402
from kerncore.kv import GB, GiB                          # noqa: E402


def _repo_root() -> Path:
    for p in HERE.parents:
        if (p / "tools" / "orchestration" / "svgchart.py").is_file():
            return p
    raise RuntimeError("tools/orchestration/svgchart.py not found above " + str(HERE))


ROOT = _repo_root()


def _load_svgchart():
    if "svgchart" in sys.modules:
        return sys.modules["svgchart"]
    spec = importlib.util.spec_from_file_location("svgchart", ROOT / "tools" / "orchestration" / "svgchart.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["svgchart"] = mod
    spec.loader.exec_module(mod)
    return mod


svgchart = _load_svgchart()

DOC = ROOT / "04-inference-engine" / "kv-cache" / "kv-cache-primer.md"
FIG_DIR = DOC.parent / "figures"

# The inputs the primer names (its §3, §4 and Verify list): the two attention shapes, the H100, the cache dtypes.
L3, L2 = kv.MODELS["llama-3-8b"], kv.MODELS["llama-2-13b"]
H100_HBM = 80 * GB                    # "an 80 GB H100"
H100_TFLOPS, H100_TB_S = 1000, 3.35   # "approximately 1,000 TFLOP/s in fp16", "about 3.35 TB/s from HBM"
FP16, FP8 = kv.DTYPE_BYTES["fp16"], kv.DTYPE_BYTES["fp8"]
NOTE = "computed by kerncore.kv.{}"


def _k(tokens: int) -> str:
    """8192 -> '8K', as the primer names a context length."""
    return f"{tokens // 1024}K"


def _per_token(shape, bytes_per: int = FP16) -> int:
    return kv.kv_bytes_per_token(shape.n_layers, shape.n_kv_heads, shape.head_dim, bytes_per)


def _cache(shape, seq_len: int, batch: int = 1, bytes_per: int = FP16) -> int:
    return kv.kv_cache_bytes(shape.n_layers, shape.n_kv_heads, shape.head_dim, seq_len, batch, bytes_per)


# ---------------------------------------------------------------------------------------------------------------
# §4: the cache against the tokens it holds, for the two shapes, next to the weights
# ---------------------------------------------------------------------------------------------------------------
def spec_kv_bytes_vs_tokens() -> dict:
    tokens = [2 ** i for i in range(11, 19)]                       # 2K .. 256K tokens in the cache
    gqa = [_cache(L3, t) / GiB for t in tokens]
    mha = [_cache(L2, t) / GiB for t in tokens]
    w = kv.weight_bytes(L3, FP16)
    one_8k = _cache(L3, 8192)
    one_128k = _cache(L3, 131_072)
    users_32 = _cache(L3, 8192, batch=32)
    return {
        "kind": "line",
        "title": (f"The fp16 KV cache grows with the tokens that it holds: {kv.fmt_bytes(_per_token(L3))} per token "
                  f"for {L3.name}, {kv.fmt_bytes(_per_token(L2))} for {L2.name}."),
        "width": 900,
        "height": 440,
        "x": {"label": "tokens in the cache: seq_len × batch", "scale": "log",
              "ticks": tokens, "tick_labels": [_k(t) for t in tokens]},
        "y": {"label": "KV cache in fp16 (GiB, log scale)", "scale": "log",
              "ticks": [0.25, 1, 4, 16, 64, 256], "tick_labels": ["0.25", "1", "4", "16", "64", "256"],
              "min": 0.125, "max": 256},
        "series": [
            {"name": f"{L2.name}, full MHA: {kv.fmt_bytes(_per_token(L2))} per token", "x": tokens, "y": mha,
             "style": "violet"},
            {"name": f"{L3.name}, GQA: {kv.fmt_bytes(_per_token(L3))} per token", "x": tokens, "y": gqa,
             "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": w / GiB, "label": f"{L3.name} weights: ~{w / GB:.0f} GB ({w / GiB:.0f} GiB)",
             "style": "warm", "anchor": "start"},
            {"kind": "point", "x": 8192, "y": one_8k / GiB, "style": "accent", "anchor": "start", "dy": 17,
             "label": f"1 user, {_k(8192)}: {one_8k / GiB:.0f} GiB ({one_8k / GB:.2f} GB)"},
            {"kind": "point", "x": 131_072, "y": one_128k / GiB, "style": "accent", "anchor": "end", "dy": -9,
             "label": f"1 user, {_k(131_072)}: {one_128k / GiB:.0f} GiB ({one_128k / GB:.1f} GB)"},
            {"kind": "point", "x": 8192 * 32, "y": users_32 / GiB, "style": "accent", "anchor": "end", "dy": -9,
             "label": f"32 users, {_k(8192)} each: {users_32 / GiB:.0f} GiB ({users_32 / GB:.1f} GB)"},
        ],
        "legend": "top-left",
        "note": NOTE.format("kv_cache_bytes"),
    }


# ---------------------------------------------------------------------------------------------------------------
# §3: decode attention's arithmetic intensity against the query heads per KV head, under the H100's ridge
# ---------------------------------------------------------------------------------------------------------------
def spec_decode_intensity() -> dict:
    heads = L3.n_heads                                             # 32 query heads: g = 1 (MHA) .. 32 (MQA)
    groups = [heads // kv_heads for kv_heads in (heads, 16, 8, 4, 2, 1)]
    mha_g = heads // kv.kv_heads_for("MHA", heads)
    gqa_g = L3.group
    mqa_g = heads // kv.kv_heads_for("MQA", heads)
    ridge = kv.ridge(H100_TFLOPS, H100_TB_S)
    labels = []                        # g = 4 keeps one line: under it sits the axis title; its point label names it
    for g in groups:
        if g == mha_g:
            labels.append(f"{g}\nfull MHA")
        elif g == mqa_g:
            labels.append(f"{g}\nMQA")
        else:
            labels.append(str(g))
    return {
        "kind": "line",
        "title": (f"Decode attention does 2g/b FLOP per byte: {kv.decode_intensity(mha_g, FP16):.0f} for full MHA "
                  f"and {kv.decode_intensity(gqa_g, FP16):.0f} for {L3.name} in fp16, against the "
                  f"~{round(ridge, -2):.0f} that the H100 needs."),
        "width": 880,
        "height": 380,
        "x": {"label": "query heads that share one KV head, g", "scale": "log",
              "ticks": groups, "tick_labels": labels},
        "y": {"label": "FLOP per byte (log scale)", "scale": "log", "ticks": [1, 10, 100, 1000],
              "min": 0.5, "max": 1000},
        "series": [
            {"name": "fp16 cache, b = 2", "x": groups, "y": [kv.decode_intensity(g, FP16) for g in groups],
             "style": "accent", "markers": True},
            {"name": "fp8 cache, b = 1", "x": groups, "y": [kv.decode_intensity(g, FP8) for g in groups],
             "style": "yellow", "dash": True, "markers": True},
        ],
        "marks": [
            {"kind": "hline", "y": ridge, "style": "warm", "anchor": "start",
             "label": f"the H100 needs ~{round(ridge, -2):.0f} FLOP per byte loaded"},
            {"kind": "point", "x": mha_g, "y": kv.decode_intensity(mha_g, FP16), "style": "accent",
             "anchor": "start", "dy": 17, "label": f"full MHA, fp16: about {kv.decode_intensity(mha_g, FP16):.0f} FLOP per byte"},
            {"kind": "point", "x": gqa_g, "y": kv.decode_intensity(gqa_g, FP16), "style": "accent",
             "anchor": "start", "dy": 17, "label": f"g = {gqa_g} ({L3.name}): {kv.decode_intensity(gqa_g, FP16):.0f} FLOP/B"},
        ],
        "legend": "bottom-right",
        "note": NOTE.format("decode_intensity and ridge"),
    }


# ---------------------------------------------------------------------------------------------------------------
# §4, drawn in §7: sessions of one context length that fit on one 80 GB H100 next to the weights, fp16 and fp8
# ---------------------------------------------------------------------------------------------------------------
def spec_sessions_per_gpu() -> dict:
    contexts = [2 ** i for i in range(10, 18)]                     # 1K .. 128K tokens per session
    w = kv.weight_bytes(L3, FP16)
    fp16 = [kv.sessions_per_gpu(H100_HBM, w, c, _per_token(L3, FP16)) for c in contexts]
    fp8 = [kv.sessions_per_gpu(H100_HBM, w, c, _per_token(L3, FP8)) for c in contexts]
    s16 = kv.sessions_per_gpu(H100_HBM, w, 8192, _per_token(L3, FP16))
    s8 = kv.sessions_per_gpu(H100_HBM, w, 8192, _per_token(L3, FP8))
    return {
        "kind": "line",
        "title": (f"On an {H100_HBM / GB:.0f} GB H100 with {L3.name}, fewer sessions fit as the context grows: "
                  f"{s16} of {_k(8192)} tokens in fp16, {s8} in fp8."),
        "width": 880,
        "height": 380,
        "x": {"label": "context length of each session (tokens)", "scale": "log",
              "ticks": contexts, "tick_labels": [_k(c) for c in contexts]},
        "y": {"label": "sessions, an upper bound (log scale)", "scale": "log", "ticks": [1, 10, 100, 1000],
              "min": 1, "max": 1000},
        "series": [
            {"name": "fp8 cache", "x": contexts, "y": fp8, "style": "yellow", "dash": True, "markers": True},
            {"name": "fp16 cache", "x": contexts, "y": fp16, "style": "accent", "markers": True},
        ],
        "marks": [
            {"kind": "point", "x": 8192, "y": s8, "style": "yellow", "anchor": "start", "dy": -10,
             "label": f"{s8} sessions of {_k(8192)} tokens with an fp8 cache"},
            {"kind": "point", "x": 8192, "y": s16, "style": "accent", "anchor": "end", "dy": 18,
             "label": f"{s16} sessions of {_k(8192)} tokens with an fp16 cache"},
        ],
        "legend": "top-right",
        "note": NOTE.format("sessions_per_gpu"),
    }


FIGURES = {
    "kv-bytes-vs-tokens.svg": spec_kv_bytes_vs_tokens,
    "decode-intensity.svg": spec_decode_intensity,
    "sessions-per-gpu.svg": spec_sessions_per_gpu,
}


def main() -> int:
    for name, build in FIGURES.items():
        out = svgchart.write_figure(FIG_DIR / name, build())
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
