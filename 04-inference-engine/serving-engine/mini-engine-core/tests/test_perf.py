"""The roofline step-time model, pinned to hand-computed numbers; the simulator's knob effects."""
from dataclasses import replace

import numpy as np
import pytest

from minengine import perf, spec

H100, L4 = perf.GPUS["H100-SXM"], perf.GPUS["L4"]
LLAMA, QWEN = perf.LLMS["llama-3.1-8b"], perf.LLMS["qwen2.5-0.5b"]
EXACT = dict(flop_eff=1.0, bw_eff=1.0, overhead_s=0.0)


def test_decode_step_is_the_weight_read():
    c = perf.step_cost(H100, LLAMA, [(0, 1)], **EXACT)
    assert c["bound"] == "memory"
    streamed = (8.03e9 - 128256 * 4096) * 2                          # untied: the input embedding is a gather
    assert np.isclose(c["t"], (streamed + 131072) / 3.35e12)        # ~4.48 ms: bytes / bandwidth
    batch = perf.step_cost(H100, LLAMA, [(0, 1)] * 64, **EXACT)
    assert batch["t"] < 1.01 * c["t"]                               # 64 decodes cost ~ the same as one


def test_prefill_is_compute_bound():
    c = perf.step_cost(H100, LLAMA, [(0, 2048)], **EXACT)
    embed = 128256 * 4096
    flops = 2 * (8.03e9 - 2 * embed) * 2048 + 2 * embed + 4 * 32 * 32 * 128 * (2048 * 2049 / 2)
    assert c["bound"] == "compute" and np.isclose(c["flops"], flops)
    assert np.isclose(c["t"], flops / 989e12)


def test_knee_tokens_hand_computed():
    assert np.isclose(perf.knee_tokens(H100, LLAMA), 989e12 * 2 / (2 * 3.35e12))      # ~295
    assert np.isclose(perf.knee_tokens(L4, LLAMA), 121e12 * 2 / (2 * 300e9))          # ~403


def test_kv_cache_blocks_hand_computed():
    assert perf.kv_cache_blocks(L4, LLAMA) == int((24e9 * 0.9 - 16.06e9 - 1e9) // (16 * 131072))   # 2164


def test_quantized_weights_keep_the_embedding_and_lm_head_in_16bit():
    embed = 128256 * 4096
    matmul = 8.03e9 - 2 * embed
    int4 = replace(LLAMA, bytes_per_param=4.125 / 8)
    assert np.isclose(int4.weight_bytes, matmul * 4.125 / 8 + 2 * embed * 2)          # 3.60 + 2.10 = 5.70 GB
    assert np.isclose(int4.streamed_bytes, matmul * 4.125 / 8 + embed * 2)            # the LM head is read every step
    assert np.isclose(LLAMA.weight_bytes, 8.03e9 * 2) and np.isclose(LLAMA.streamed_bytes, (8.03e9 - embed) * 2)


def test_verify_pass_needs_logits_at_every_position():
    one = perf.step_cost(H100, LLAMA, [(1000, 5)], **EXACT)
    five = perf.step_cost(H100, LLAMA, [(1000, 5, 5)], **EXACT)
    assert np.isclose(five["flops"] - one["flops"], 4 * 2 * 128256 * 4096)           # 4 more LM-head rows


def test_speculation_cost_model_matches_the_formula_at_batch_1_and_stops_paying_at_high_load():
    draft = perf.LLMS["llama-3.2-1b"]
    c = perf.step_time(H100, draft, [(200, 1)], overhead_s=0.0005) / perf.step_time(H100, LLAMA, [(200, 1)])
    one = perf.spec_speedup(H100, LLAMA, draft, 1, 200, 0.7, 4)
    assert abs(one - spec.speedup(0.7, 4, c)) < 0.02                  # batch 1: verify ~ one target step
    assert 1.5 < one < 1.7 and 0.18 < c < 0.2                         # ~1.6x with a 0.19 draft (SIMULATED)
    at = lambda batch, ctx: perf.spec_speedup(H100, LLAMA, draft, batch, ctx, 0.7, 4)
    assert at(128, 200) < 1 < at(64, 200)                             # short contexts: a slow-down at high load
    assert at(128, 2000) > 1.3                                        # long context: KV-bound, keeps paying


def test_goodput_counts_only_requests_that_meet_both_slos():
    r = perf.SimResult("hand", ttft=np.array([0.1, 0.3, 0.1, 2.0]), tpot=np.array([0.02, 0.02, 0.09, 0.02]),
                       itl=np.array([0.02]), e2e=np.zeros(4), duration=2.0, output_tokens=400, steps=1,
                       preemptions=0, hit_rate=0.0, peak_kv_usage=0.0)
    assert r.goodput(ttft_slo=0.5, tpot_slo=0.05) == 2 / 2.0                           # requests 0 and 1 of 4
    assert r.throughput == 200.0


def test_fp8_compute_needs_fp8_hardware():
    fp8 = replace(LLAMA, bytes_per_param=1.0, compute_scale=2.0)
    assert perf.step_time(H100, fp8, [(0, 2048)], **EXACT) < 0.51 * perf.step_time(H100, LLAMA, [(0, 2048)], **EXACT)
    with pytest.raises(ValueError):
        perf.step_time(perf.GPUS["A100-80GB"], fp8, [(0, 1)])


def _sim(**kw):
    w = perf.Workload(n_requests=60, rate=6, prompt_len=(1500, 3000), output_len=(40, 120), seed=3)
    return perf.simulate(H100, LLAMA, w, **kw)


def test_simulation_is_labelled_and_complete():
    r = _sim(label="baseline")
    assert r.summary().startswith("SIMULATED")
    w = perf.Workload(n_requests=60, rate=6, prompt_len=(1500, 3000), output_len=(40, 120), seed=3)
    assert r.output_tokens == sum(n for _, _, n in w.requests()) and len(r.ttft) == 60


def test_smaller_token_budget_trades_ttft_for_smoother_itl():
    whole, small = _sim(enable_chunked_prefill=False), _sim(max_num_batched_tokens=256)
    assert small.pct("itl", 99) < whole.pct("itl", 99)
    assert small.pct("ttft", 50) > whole.pct("ttft", 50)


def test_prefix_caching_cuts_ttft_for_a_shared_system_prompt():
    w = perf.Workload(n_requests=60, rate=6, prompt_len=(2000, 2200), output_len=(40, 80), shared_prefix=1800)
    on = perf.simulate(H100, LLAMA, w)
    off = perf.simulate(H100, LLAMA, w, enable_prefix_caching=False)
    assert on.hit_rate > 0.7 and on.pct("ttft", 50) < 0.5 * off.pct("ttft", 50)


def test_a_small_kv_cache_forces_preemptions():
    assert _sim(num_blocks=600).preemptions > 0 and _sim().preemptions == 0


def test_tensor_parallel_and_lora_sizing():
    assert perf.tp_allreduces(80, 8192, 64) == (160, 64 * 8192 * 2)       # a 70B model, 64 decodes: 1 MiB each
    llama8b = [(4096, 4096), (4096, 1024), (4096, 1024), (4096, 4096), (4096, 14336), (4096, 14336), (14336, 4096)]
    assert perf.lora_params(llama8b, 16, 32) == 41_943_040                # ~84 MB in bf16, ~0.5% of the base
