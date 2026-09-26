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


def test_one_small_gpu_budgets_like_vllm_and_reproduces_the_lab():
    """kv_room_gib() = 0.92 x reported GiB - 1.5 GiB - weights: the budget moelab.offload.fit uses (the lab
    prints these same numbers: `python -m moelab fit --model olmoe-1b-7b --gpu T4`)."""
    olmoe, q3, t4, l4 = M["olmoe-1b-7b"], M["qwen3-30b-a3b"], D["t4"], D["l4"]
    assert S.kv_room_gib(olmoe, t4) == pytest.approx(0.92 * 15.0 - 1.5 - S.weight_bytes(olmoe) / 2 ** 30)
    assert S.kv_tokens(olmoe, t4) == 0                                     # fp16 OLMoE on a T4: no KV room
    assert S.min_offload_gib(olmoe, t4, 4 * 4096) == 3.0                   # lab: --cpu-offload-gb 3
    assert S.kv_tokens(olmoe, t4, offload_gib=3.0) == 19_761               # lab: 4.8 x 4096
    assert round(S.offload_step_s(3.0, 12) * 1e3) == 268                   # lab: +268 ms per step at 12 GB/s
    assert S.kv_tokens(q3, l4, 16, 4.25) == 21_593                         # lab: int4-experts, 5.3 x 4096
    assert S.kv_tokens(q3, l4, 8) == 0 and S.kv_tokens(q3, l4) == 0        # FP8 and bf16 do not fit
    assert S.sessions(q3, l4, 1, 4096, 16, 4.25) == 7                      # the round fleet budget is looser
    assert D["h100-sxm"].memory_gib() == pytest.approx(80e9 / 2 ** 30)     # no reported value: nominal


def test_a_dense_model_of_the_active_size():
    """Mixtral's dense twin: same attention and embeddings, one MLP two experts wide. It decodes exactly
    as fast at batch 1 and holds 26 GB instead of 93 GB; from batch 4 the MoE streams 2.5x its bytes."""
    mix = M["mixtral-8x7b"]
    dense = S.dense_equivalent(mix)
    assert dense.n_experts == 0 and dense.expert_ff == 2 * 14_336
    assert mix.active() - dense.total() == mix.moe_layers * mix.router_params() == 1_048_576
    h200 = D["h200"]
    step = lambda cfg, b: T.decode_step(cfg, h200, b, 1024).time
    assert step(mix, 1) / step(dense, 1) == pytest.approx(1.0, abs=1e-3)
    assert 2.4 < step(mix, 4) / step(dense, 4) < 2.6 and 3.3 < step(mix, 16) / step(dense, 16) < 3.5
    with pytest.raises(ValueError):
        S.dense_equivalent(M["deepseek-v3"])
