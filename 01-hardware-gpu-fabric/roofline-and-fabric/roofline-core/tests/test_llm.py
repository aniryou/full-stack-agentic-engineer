"""LLM inference on the roofline: parameter counts, KV bytes, prefill/decode step costs."""
import math

import pytest

from roofline import llm, specs

H100, L4, H200 = specs.get("h100-sxm"), specs.get("l4"), specs.get("h200")
M8, M70 = llm.PRESETS["llama-3.1-8b"], llm.PRESETS["llama-3.1-70b"]
MIX, Q3 = llm.PRESETS["mixtral-8x7b"], llm.PRESETS["qwen3-30b-a3b"]


def test_param_counts_match_published_sizes():
    assert M8.params() / 1e9 == pytest.approx(8.03, abs=0.01)
    assert M70.params() / 1e9 == pytest.approx(70.55, abs=0.01)
    assert MIX.params() / 1e9 == pytest.approx(46.7, abs=0.05)
    assert MIX.active_params() / 1e9 == pytest.approx(12.9, abs=0.05)
    assert Q3.params() / 1e9 == pytest.approx(30.5, abs=0.05)
    assert Q3.active_params() / 1e9 == pytest.approx(3.35, abs=0.05)
    assert llm.PRESETS["qwen2.5-1.5b"].params() / 1e9 == pytest.approx(1.54, abs=0.01)
    # by hand, one Llama-3.1-8B layer: q,o 4096^2 each; k,v 4096*1024 each; MLP 3*4096*14336
    assert M8.attn_params() + M8.expert_params() == 2 * 4096 ** 2 + 2 * 4096 * 1024 + 3 * 4096 * 14336


def test_kv_bytes_per_token():
    assert M8.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2 == 131_072          # 128 KiB
    assert M70.kv_bytes_per_token() == 2 * 80 * 8 * 128 * 2 == 327_680         # 320 KiB
    assert M8.kv_bytes_per_token(kv_bytes=1) == 65_536


def test_prefill_is_compute_bound():
    s = llm.prefill(M8, H100, 2048)
    assert s.bound == "compute"
    assert s.flops == pytest.approx(2.969e13, rel=1e-3)
    assert s.t_compute * 1e3 == pytest.approx(30.0, abs=0.1)                 # ideal TTFT ~30 ms
    assert s.intensity == pytest.approx(1941, abs=1)
    # intensity ~ tokens in the step: the crossover sits near the ridge (295) in tokens
    assert llm.prefill(M8, H100, 300).bound == "memory"
    assert llm.prefill(M8, H100, 320).bound == "compute"


def test_decode_batch_one_is_a_weight_stream():
    s = llm.decode(M8, H100, 1, 1024)
    assert s.bound == "memory" and s.intensity == pytest.approx(1.03, abs=0.01)
    assert s.bytes / 1e9 == pytest.approx(15.14, abs=0.01)
    assert s.time * 1e3 == pytest.approx(4.52, abs=0.01) and round(s.tokens_per_s) == 221
    assert llm.decode(M8, L4, 1, 1024).time * 1e3 == pytest.approx(50.5, abs=0.1)


def test_batching_amortizes_weights_until_kv_dominates():
    sweep = {b: llm.decode(M8, H100, b, 2048) for b in (1, 64, 256)}
    assert round(sweep[64].tokens_per_s) == 6659 and sweep[64].time * 1e3 == pytest.approx(9.61, abs=0.01)
    assert sweep[256].intensity == pytest.approx(49.2, abs=0.1)
    assert all(s.bound == "memory" for s in sweep.values())


def test_kv_reads_cap_decode_intensity():
    assert llm.decode_intensity_limit(M8, 4096) == pytest.approx(32.0, abs=0.05)
    assert llm.max_context_for_compute_bound(M8, H100) == pytest.approx(392, abs=0.5)
    assert llm.decode_crossover_batch(M8, H100, 0) == 297                  # ~ the ridge
    assert llm.decode_crossover_batch(M8, H100, 128) == 440
    assert llm.decode_crossover_batch(M8, H100, 1024) is None              # never compute-bound


def test_memory_capacity_caps_the_batch_first():
    # (0.9 x 80e9 - 8.03e9 x 2) / (4096 x 131072) = 104.2
    assert llm.max_batch_by_memory(M8, H100, 4096) == math.floor((72e9 - M8.params() * 2) / (4096 * 131072)) == 104
    assert llm.max_batch_by_memory(M8, L4, 2048) == 20
    assert llm.best_batch_under_itl(M8, H100, 4096, 0.020) == 96


def test_quantization_moves_bytes_not_flops():
    base = llm.decode(M8, H100, 64, 2048)
    fp8 = llm.decode(M8, H100, 64, 2048, **llm.SCHEMES["fp8"])
    w4 = llm.decode(M8, H100, 64, 2048, **llm.SCHEMES["w4a16"])
    w4kv8 = llm.decode(M8, H100, 64, 2048, **llm.SCHEMES["w4a16-kv8"])
    assert base.time / fp8.time == pytest.approx(2.0, abs=0.01)             # every byte halves
    assert base.time / w4.time == pytest.approx(1.54, abs=0.01)             # KV untouched
    assert base.time / w4kv8.time == pytest.approx(2.61, abs=0.01)
    assert llm.decode(M8, H100, 1, 2048).time / llm.decode(M8, H100, 1, 2048, **llm.SCHEMES["w4a16"]).time \
        == pytest.approx(3.80, abs=0.01)
    assert llm.decode_crossover_batch(M8, H100, 0, **llm.SCHEMES["w4a16"]) == 75   # ~ ridge x 0.5 / 2


def test_moe_streams_more_experts_as_the_batch_grows():
    assert llm.experts_touched(8, 2, 1) == pytest.approx(2.0)
    assert llm.experts_touched(8, 2, 16) == pytest.approx(8 * (1 - 0.75 ** 16))      # 7.92
    assert llm.experts_touched(128, 8, 16) == pytest.approx(82.4, abs=0.05)
    one, sixteen = llm.decode(MIX, H200, 1, 1024), llm.decode(MIX, H200, 16, 1024)
    assert one.bytes / 1e9 == pytest.approx(25.6, abs=0.05)                # ~ active params x 2
    assert sixteen.bytes / 1e9 == pytest.approx(94.4, abs=0.05)            # ~ all experts
    assert one.time * 1e3 == pytest.approx(5.34, abs=0.01) and sixteen.time * 1e3 == pytest.approx(19.66, abs=0.01)
