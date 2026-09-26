"""quantcore reproduces the numbers the repo already publishes, with its own code.

- minengine.quant / mini-engine-core notebook 06 / serving-engine PRIMER section 8: the six-scheme error table, the
  FP8 grid, the outlier channel, SmoothQuant's 6.6x, and the SIMULATED Llama-3.1-8B-on-L4 table;
- vllm-internals section 8.1: the down_proj roofline table and the W4A16 crossover (~120 tokens on an L4, ~85 on an H100);
- servelab.sizing / facts: Qwen2.5-0.5B weights and the two INT4 bits-per-weight conventions.
Layouts differ: minengine stores w as (d_in, d_out); quantcore uses (out, in), so w.T is the same matrix.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from quantcore import cost as C
from quantcore import formats as F
from quantcore import granularity as G
from quantcore import smoothquant as S


def _replay_notebook_06():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((256, 128)) * 0.02
    x = rng.standard_normal(10000)
    wo = rng.standard_normal((64, 32)) * 0.02
    X, W = rng.standard_normal((32, 64)), rng.standard_normal((64, 16)) * 0.05
    return w, x, wo, X, W


def test_minengine_error_table():
    w = _replay_notebook_06()[0].T
    rows = [("int8", "tensor", None, 8, 0.0103, 39.8), ("int8", "channel", None, 8, 0.0071, 43.0),
            ("fp8", "tensor", None, 8, 0.0263, 31.6), ("int4", "channel", None, 4, 0.1280, 17.9),
            ("int4", "group", 128, 4.125, 0.1184, 18.5), ("int4", "group", 32, 4.5, 0.0979, 20.2)]
    for fmt, gran, g, bpw, rel, sqnr in rows:
        e = G.error(w, G.fake_quant(w, fmt=fmt, granularity=gran, group_size=g or 128))
        assert round(e["rel"], 4) == rel and round(e["sqnr_db"], 1) == sqnr
        assert F.bits_per_weight(int(bpw), g) == bpw


def test_minengine_fp8_error_outlier_and_smoothquant():
    _, x, wo, X, W = _replay_notebook_06()
    rel = np.abs(F.to_float(x) - x) / np.abs(x)
    assert f"{np.median(rel):.3%}" == "2.224%" and f"{rel[np.abs(x) > 2 ** -6].max():.3%}" == "5.877%"
    wo[:, 3] *= 100
    Wo, rest = wo.T, [c for c in range(32) if c != 3]
    t = G.fake_quant(Wo, fmt="int8", granularity="tensor")
    assert round(G.error(Wo, t)["rel"], 3) == 0.034 and round(G.error(Wo[rest], t[rest])["rel"], 3) == 0.624
    X[:, 5] *= 60
    naive, smooth = S.w8a8_error(X, W.T), S.w8a8_error(X, W.T, alpha=0.5)
    assert round(naive, 4) == 0.0250 and round(smooth, 4) == 0.0038 and round(naive / smooth, 1) == 6.6


def test_serving_engine_s8_simulated_table():
    """serving-engine PRIMER section 8 'What each buys' (minengine.perf: 80% BW, 60% FLOPs, 2 ms; L4 as minengine
    defines it: 121 TFLOP/s bf16, 2x that for FP8, 300 GB/s, 24 GB). INT8 weight-only = our w8a16 row."""
    L4 = C.GPU("L4", 8.9, 24, 0.30, {"bf16": 121, "fp8": 242})
    m = C.MODELS["llama-3.1-8b"]
    expected = [("bf16", 16, 16.1, 65.1, 82.0, 360, 17), ("w8a16-fp8", 16, 9.1, 36.0, 53.0, 360, 43),
                ("w4a16", 16, 5.7, 21.9, 38.9, 360, 56), ("w8a8-fp8", 16, 9.1, 36.0, 53.0, 181, 43),
                ("w8a8-fp8", 8, 9.1, 35.7, 44.2, 181, 87), ("w4a16", 8, 5.7, 21.6, 30.1, 360, 113)]
    for s, kv, gb, d1, d32, pf, sess in expected:
        assert round(m.weight_bytes(C.SCHEMES[s].w_bits) / 1e9, 1) == gb
        assert round(1e3 * C.step_cost(L4, m, s, [(1000, 1)], kv)["t"], 1) == d1
        assert round(1e3 * C.step_cost(L4, m, s, [(1000, 1)] * 32, kv)["t"], 1) == d32
        assert round(1e3 * C.step_cost(L4, m, s, [(0, 1800)], kv)["t"]) == pf
        assert C.sessions(L4, m, s, 2000, kv) == sess
    assert C.kv_blocks(L4, m, "bf16") == 2164                                # serving-engine section 4, core inputs
    assert round(m.weight_bytes(16) / 1e9) == 16 and round(C.MODELS["llama-3.1-70b"].weight_bytes(16) / 1e9) == 141
    assert round(C.MODELS["llama-3.1-70b"].weight_bytes(4.125) / 1e9, 1) == 39.5  # notebook 06 exercise 6.3


def test_vllm_internals_down_proj_table():
    """vllm-internals section 8.1: K 14,336 x N 4,096, W4A16 at 4.16 bits per weight, ideal roofline."""
    L4, H100 = C.GPUS["L4"], C.GPUS["H100-SXM"]
    t = lambda M, s: round(1e6 * C.gemm_time(M, 14336, 4096, L4, s, w_bits=4.16 if s == "w4a16" else None)["t"])
    assert [t(1, s) for s in ("bf16", "w4a16", "w8a8-fp8")] == [392, 102, 196]
    assert [t(256, s) for s in ("bf16", "w4a16", "w8a8-fp8")] == [423, 248, 215]
    assert [t(2048, s) for s in ("bf16", "w4a16", "w8a8-fp8")] == [1988, 1988, 992]
    assert round(C.crossover_tokens(14336, 4096, L4, w_bits=4.16)) == 120
    assert round(C.crossover_tokens(14336, 4096, H100, w_bits=4.16)) == 85


def test_servelab_weight_bytes_and_int4_conventions():
    q = C.MODELS["qwen2.5-0.5b"]
    assert [round(q.weight_bytes(b) / 1e9, 3) for b in (16, 8, 4.125)] == [0.988, 0.630, 0.457]
    assert F.bits_per_weight(4, 128) == 4.125 and F.bits_per_weight(4, 128, zero_point_bits=4) == 4 + 2.5 * 8 / 128
    assert round(q.weight_bytes(16) / q.weight_bytes(4.125), 2) == 2.16      # the tied 16-bit embedding is 27.6%
    assert round(q.embed_params / q.params, 3) == 0.276


def test_live_cross_check_with_minengine_when_present():
    """If the sibling core is on disk, compare against minengine.quant directly on fresh data."""
    root = Path(__file__).resolve().parents[3] / "serving-engine" / "mini-engine-core"
    if not (root / "minengine").exists():
        pytest.skip("mini-engine-core not found next to this topic")
    sys.path.insert(0, str(root))
    try:
        from minengine import quant as mq
    finally:
        sys.path.remove(str(root))
    w = np.random.default_rng(7).standard_normal((256, 64))
    for kw in (dict(fmt="int8", granularity="channel"), dict(fmt="int4", granularity="group", group_size=32),
               dict(fmt="fp8", granularity="tensor")):
        np.testing.assert_allclose(G.fake_quant(w.T, **kw).T, mq.fake_quant(w, **kw), atol=1e-12)
