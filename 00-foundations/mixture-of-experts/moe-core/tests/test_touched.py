"""Which experts a batch touches, and the decode roofline -- reproducing layer 01's PRIMER §3.6.

roofline-core (01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/roofline/llm.py) computes
experts_touched(), decode() and decode_crossover_batch() for Mixtral-8x7B and Qwen3-30B-A3B; its
PRIMER §3.6 prints the table below. moecore is a standalone package, so this test pins the same
numbers from moecore's own code, and (when the repo is checked out) checks that the table in that
primer still says them.
"""
from pathlib import Path

import numpy as np
import pytest

from moecore import touched as T
from moecore.sizing import MODELS

H200 = T.DEVICES["h200"]
MIX, Q3, L8 = MODELS["mixtral-8x7b"], MODELS["qwen3-30b-a3b"], MODELS["llama-3.1-8b"]
ROOFLINE_PRIMER = Path(__file__).resolve().parents[4] / "01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md"

# layer 01 PRIMER §3.6: batch -> (Mixtral experts/layer, step GB on H200, step ms, Qwen3 experts/layer)
TABLE = {1: ("2.00", "25.6", "5.34", "8.0"), 4: ("5.47", "65.1", "13.57", "29.1"),
         16: ("7.92", "94.4", "19.66", "82.4"), 64: ("8.00", "101.7", "21.20", "125.9")}


def test_closed_form_hand_computed():
    assert T.experts_touched(8, 2, 1) == 2.0
    assert T.experts_touched(8, 2, 16) == pytest.approx(8 * (1 - 0.75 ** 16))            # 7.92
    assert T.experts_touched(0, 0, 100) == 1.0                                            # dense
    ds = [round(T.experts_touched(256, 8, t), 1) for t in (1, 8, 32, 128, 256)]
    assert ds == [8.0, 57.4, 163.3, 251.6, 255.9]                                        # DeepSeek-V3


def test_monte_carlo_agrees_with_the_closed_form_under_uniform_routing():
    for e, k, t in [(8, 2, 4), (128, 8, 16), (256, 8, 32)]:
        mc, _ = T.touched_mc(e, k, t, s=0.0, trials=300)
        assert mc == pytest.approx(T.experts_touched(e, k, t), rel=0.02)


def test_skew_touches_fewer_experts_and_heats_one():
    uni, hot_u = T.touched_mc(128, 8, 16, s=0.0)
    zipf, hot_z = T.touched_mc(128, 8, 16, s=1.0)
    assert zipf < uni and hot_z > 2 * hot_u


def test_samples_are_k_distinct_experts_per_token():
    idx = T.sample_routes(16, 4, 500, T.zipf_popularity(16, 1.2), np.random.default_rng(0))
    assert idx.shape == (500, 4) and all(len(set(r)) == 4 for r in idx)


def test_reproduces_the_roofline_primer_moe_table():
    """Same bytes to the byte as roofline.llm.decode(): 25,631,531,008 B at batch 1 (25,497,182,208 weights + KV)."""
    s1 = T.decode_step(MIX, H200, 1, 1024)
    assert s1.bytes == 25_631_531_008 and T.streamed_weight_bytes(MIX, 1) == 25_497_182_208
    for b, (mix_e, gb, ms, q_e) in TABLE.items():
        s = T.decode_step(MIX, H200, b, 1024)
        assert f"{T.experts_touched(8, 2, b):.2f}" == mix_e and f"{s.bytes / 1e9:.1f}" == gb
        assert f"{s.time * 1e3:.2f}" == ms and s.bound == "memory"
        assert f"{T.experts_touched(128, 8, b):.1f}" == q_e


@pytest.mark.skipif(not ROOFLINE_PRIMER.exists(), reason="layer 01 not checked out beside this core")
def test_the_roofline_primer_still_prints_these_rows():
    text = " ".join(ROOFLINE_PRIMER.read_text(encoding="utf-8").split())
    for b, (mix_e, gb, ms, q_e) in TABLE.items():
        mix = f"{mix_e} experts/layer" if b == 1 else mix_e
        assert f"| {b} | {mix} | {gb} GB | {ms} ms | {q_e}{' / 128' if b == 1 else ''} |" in text


def test_decode_crossover_scales_with_total_over_active():
    """207 / 754 / 2,055 at c = 0 on an H200, as roofline.llm.decode_crossover_batch() gives them."""
    x = {m.name: T.decode_crossover_batch(m, H200, 0) for m in (L8, MIX, Q3)}
    assert x == {"Llama-3.1-8B": 207, "Mixtral-8x7B": 754, "Qwen3-30B-A3B": 2055}
    for m in (MIX, Q3):   # streamed / multiplied weights (no input embedding) sets the ratio
        ratio = (m.total() - m.vocab * m.d_model) / (m.active() - m.vocab * m.d_model)
        assert x[m.name] / x[L8.name] == pytest.approx(ratio, rel=0.01)


def test_kv_share_is_lower_for_moe():
    """Batch 64, 1K context: the same 8.6 GB of KV is 8% of a Mixtral step but 36% of a Llama-8B step."""
    kv = 64 * 1025 * 131_072
    assert kv == 8_598_323_200
    assert round(T.kv_share(MIX, 64, 1024), 3) == 0.085 and round(T.kv_share(L8, 64, 1024), 2) == 0.36


def test_skewed_touched_count_feeds_the_step():
    """Fewer experts touched -> fewer bytes streamed; a caller can pass a measured or simulated count."""
    full, skew = T.decode_step(Q3, H200, 16, 1024), T.decode_step(Q3, H200, 16, 1024, touched=50.0)
    assert skew.bytes < full.bytes
    assert full.bytes - skew.bytes == pytest.approx(48 * (T.experts_touched(128, 8, 16) - 50) * Q3.expert_params() * 2)


def test_attention_flops_per_position_follow_the_attention_type():
    """MHA/GQA: 4 x heads x head_dim per layer per cached position (q.k and the weighted V sum).
    Absorbed MLA (DeepSeek-V3) works in the latent: 2 x heads x (576 + 512), about 4x more."""
    assert MIX.attn_flops_per_position() == 4 * 32 * 128
    ds = MODELS["deepseek-v3"]
    assert ds.attn_flops_per_position() == 128 * 2 * (576 + 512) == 278_528
    s0, s4k = T.decode_step(ds, H200, 32, 0, 1), T.decode_step(ds, H200, 32, 4096, 1)
    assert s4k.flops - s0.flops == 32 * 61 * 278_528 * 4096
    assert s4k.bound == "memory"                                   # still bytes-bound at batch 32
