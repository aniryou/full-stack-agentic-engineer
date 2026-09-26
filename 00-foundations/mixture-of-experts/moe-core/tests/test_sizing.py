"""Sizing: memory by total, prefill by active, decode by bytes streamed; reproduces the capacity primer."""
import pytest

from moecore import ep as E
from moecore import sizing as S
from moecore import touched as T

M, D = S.MODELS, T.DEVICES


def test_reproduces_the_capacity_primers_mistral_large_3_numbers():
    """00-foundations/gpu-capacity-planning (PRIMER "When one GPU won't do", capacity.py): 675 B total,
    41 B active; ~675 GB in FP8; decode at large batch streams all of it: 675e9 / (8 x 4.8e12) = 17.6 ms
    ("~18 ms floor across 8xH200"); prefill FLOPs = capacity.prefill_flops(41, tokens) = 2 x 41e9 x tokens."""
    assert S.decode_floor(675e9, 8, D["h200"]) * 1e3 == pytest.approx(17.578125)
    assert S.prefill_flops(41e9, 2048) == 2 * 41 * 1e9 * 2048 == pytest.approx(1.679e14, rel=1e-3)


def test_mxfp4_experts_make_gpt_oss_fit():
    """Experts at 4.25 bits (MXFP4: 4-bit values + an 8-bit scale per 32), the rest bf16: 120b fits one 80 GB
    GPU, 20b fits 16 GB (gpt-oss README)."""
    assert round(S.weight_bytes(M["gpt-oss-120b"], 16, 4.25) / 1e9, 1) == 65.2
    assert round(S.weight_bytes(M["gpt-oss-20b"], 16, 4.25) / 1e9, 1) == 13.8
    assert S.weight_bytes(M["mixtral-8x7b"]) == 2 * M["mixtral-8x7b"].total()     # 93.4 GB in bf16


def test_sessions_hand_computed():
    """Qwen3-30B-A3B, bf16, one H100, 4K context: (72e9 - 61.06e9) / (4096 x 98,304 B) = 27 sequences."""
    q = M["qwen3-30b-a3b"]
    assert S.sessions(q, D["h100-sxm"], 1, 4096) == int((72e9 - 2 * q.total()) // (4096 * 98_304)) == 27
    assert S.sessions(q, D["l4"], 1, 4096) == 0                                   # 61 GB of weights, 24 GB card
    assert S.sessions(q, D["l4"], 4, 4096, layout="ep") < S.sessions(q, D["l4"], 4, 4096, layout="tp")


def test_plans():
    nv = E.LINKS["nvlink4"]
    mix = S.plan(M["mixtral-8x7b"], D["h100-sxm"], 64, 4096, 50, nv)
    assert (mix.gpus, round(mix.step_ms, 2)) == (2, 19.64)
    dense = S.plan(M["llama-3.1-70b"], D["h100-sxm"], 64, 4096, 50, nv, layout="tp")
    assert (dense.gpus, round(dense.step_ms, 2)) == (4, 19.26)
    assert S.plan(M["llama-3.1-70b"], D["h100-sxm"], 64, 4096, 50, nv, layout="ep") is None   # replicas don't fit
    ds = S.plan(M["deepseek-v3"], D["h200"], 256, 4096, 50, nv, bits=8, usd_per_gpu_hr=3.0)
    assert (ds.gpus, round(ds.step_ms, 2)) == (8, 23.38)
    assert ds.usd_per_mtok == pytest.approx(S.usd_per_mtok(8, 3.0, 256 / (ds.step_ms / 1e3)))


def test_cost_and_offload_hand_computed():
    assert S.usd_per_mtok(8, 3.0, 10_000) == pytest.approx(8 * 3.0 / 3600 / 10_000 * 1e6)   # $0.67
    assert round(S.offload_step_s(4) * 1e3, 1) == 171.8                           # 4 GiB over ~25 GB/s, every step


def test_kv_gb():
    assert S.kv_gb(M["mixtral-8x7b"], 32_768) == pytest.approx(32_768 * 131_072 / 1e9)   # 4.29 GB per 32K sequence
