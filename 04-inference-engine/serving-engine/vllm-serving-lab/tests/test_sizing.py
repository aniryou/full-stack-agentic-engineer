"""Sizing: the per-token KV formula, block arithmetic, fit check and log calibration, pinned by hand."""
import importlib.util
import math
from pathlib import Path

import pytest

from servelab import SAMPLES_DIR as FIX
from servelab import sizing
from servelab.sizing import GiB, kv_bytes_per_token, load_config, param_count, size


def test_kv_bytes_per_token_by_hand():
    # 2 (K,V) x layers x kv_heads x head_dim x bytes
    assert kv_bytes_per_token(load_config("qwen2.5-0.5b-instruct")) == 2 * 24 * 2 * 64 * 2 == 12_288
    assert kv_bytes_per_token(load_config("llama-3.1-8b-instruct")) == 2 * 32 * 8 * 128 * 2 == 131_072
    # Qwen3-0.6B: explicit head_dim 128 although hidden/heads = 64 -> read head_dim, never derive it
    assert kv_bytes_per_token(load_config("qwen3-0.6b")) == 2 * 28 * 8 * 128 * 2 == 114_688
    m = load_config("llama-3.1-8b-instruct")
    assert kv_bytes_per_token(m, kv_cache_dtype="fp8") == 65_536                    # fp8 KV halves it
    assert kv_bytes_per_token(m, tensor_parallel_size=2) == 65_536                  # TP splits KV heads
    q = load_config("qwen2.5-0.5b-instruct")                                        # 2 KV heads, TP 4:
    assert kv_bytes_per_token(q, tensor_parallel_size=4) == 2 * 24 * 1 * 64 * 2     # one replicated head/GPU


def test_param_counts_match_published_counts():
    published = {"llama-3.1-8b-instruct": 8_030_261_248, "llama-3.2-1b-instruct": 1_235_814_400,
                 "mistral-7b-instruct-v0.3": 7_248_023_552, "qwen2.5-0.5b-instruct": 494_032_768,
                 "qwen2.5-7b-instruct": 7_615_616_512}
    for name, n in published.items():
        assert param_count(load_config(name)).total == n, name
    mix = param_count(load_config("mixtral-8x7b-instruct-v0.1"))
    assert round(mix.total / 1e9, 1) == 46.7 and round(mix.active / 1e9, 1) == 12.9   # MoE: memory vs compute


def test_quantization_shrinks_linear_layers_only():
    m = load_config("qwen2.5-0.5b-instruct")
    p = param_count(m)
    assert sizing.weight_bytes(m) == p.total * 2
    assert sizing.weight_bytes(m, quantization="fp8") == p.linear * 1 + (p.total - p.linear) * 2
    awq = sizing.weight_bytes(m, quantization="awq")
    assert awq == pytest.approx(p.linear * (0.5 + 2.5 / 128) + (p.total - p.linear) * 2)
    assert awq / (p.total * 2) > 0.45   # embeddings (28% of this small model) stay 16-bit: ~2x, not 4x


def test_block_arithmetic_by_hand():
    r = size("qwen2.5-0.5b-instruct", "T4", max_model_len=4096, kv_budget_bytes=1 * GiB)
    assert r.bytes_per_block == 16 * 12_288 == 196_608
    assert r.num_blocks == 1_073_741_824 // 196_608 == 5461
    assert r.blocks_per_request == 4096 // 16 == 256
    assert r.max_concurrency == pytest.approx(5461 / 256)            # 21.33x in vLLM's log
    assert r.kv_cache_tokens == 87_376 == 5461 * 16
    r2 = size("qwen2.5-0.5b-instruct", "T4", max_model_len=4100, kv_budget_bytes=1 * GiB)
    assert r2.blocks_per_request == 257                               # partial block still costs a block


def test_memory_model_and_utilization():
    r = size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096,
             overhead_bytes={"measured": int(1.0 * GiB)})
    assert r.requested_bytes == math.ceil(15.0 * GiB * 0.92)
    assert r.kv_budget_bytes == r.requested_bytes - r.weights_bytes - int(1.0 * GiB)
    lower = size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096, gpu_memory_utilization=0.5,
                 overhead_bytes={"measured": int(1.0 * GiB)})
    assert lower.num_blocks < r.num_blocks
    assert any("--dtype half" in n for n in size("qwen2.5-0.5b-instruct", "T4").notes)


def test_does_not_fit_reports_like_vllm():
    r = size("llama-3.1-8b-instruct", "L4")                     # 131,072-token default context, bf16
    assert not r.fits
    msg = r.error()
    assert "max seq len (131072)" in msg and "estimated maximum model length is" in msg
    ok = size("llama-3.1-8b-instruct", "L4", max_model_len=r.estimated_max_model_len)
    assert ok.fits and ok.max_concurrency == pytest.approx(1.0, abs=0.01)
    assert sizing.max_model_len_for("llama-3.1-8b-instruct", "L4", concurrency=4) == (r.num_blocks // 4) * 16


def test_agrees_with_capacity_planning_formulas():
    root = Path(__file__).resolve().parents[4]
    path = root / "00-foundations" / "gpu-capacity-planning" / "capacity.py"
    if not path.exists():
        pytest.skip("00-foundations/gpu-capacity-planning not in this checkout")
    spec = importlib.util.spec_from_file_location("capacity", path)
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    m = load_config("mistral-small-24b-instruct-2501")
    assert kv_bytes_per_token(m) == cap.kv_per_token_kb(cap.MISTRAL_SMALL) * 1024 == 163_840
    assert kv_bytes_per_token(m, kv_cache_dtype="fp8") == cap.kv_per_token_kb(cap.MISTRAL_SMALL, "fp8") * 1024
    # concurrency: same spare memory and context -> same answer (block-aligned, so no rounding)
    spare, ctx = 4096 * 16 * 163_840, 8192
    ours = size(m, gpu_memory_bytes=80 * GiB, max_model_len=ctx, kv_budget_bytes=spare).max_concurrency
    theirs = cap.max_concurrent_sessions(spare / GiB, cap.MISTRAL_SMALL, ctx)
    assert ours == pytest.approx(theirs) == 8.0


def test_parse_startup_log_and_calibrate():
    log = sizing.parse_startup_log((FIX / "vllm_startup_log.txt").read_text())
    assert log == {"model_loading_gib": 0.93, "available_kv_gib": 11.84, "kv_cache_tokens": 1_034_592,
                   "max_concurrency": 252.59, "max_model_len": 4096, "init_engine_s": 21.07}
    # replaying the logged KV budget through the block arithmetic reproduces the logged capacity
    replay = size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096,
                  kv_budget_bytes=int(round(log["available_kv_gib"] * GiB)))
    assert replay.kv_cache_tokens == log["kv_cache_tokens"]
    pred = size("qwen2.5-0.5b-instruct", "T4", dtype="half", max_model_len=4096)
    cal = sizing.calibrate(pred, log)
    assert cal["implied_overhead_gib"] == pytest.approx(pred.requested_bytes / GiB - 0.93 - 11.84)
