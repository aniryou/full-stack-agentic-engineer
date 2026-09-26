"""The roofline step model reproduces vllm-internals §8.1/§8.2; the emulator and the fake server agree."""
import pytest

from quantlab import bench as B
from quantlab.fakeserver import FakeServer

# vllm-internals §8.1: Llama-3.1-8B down_proj (K 14,336, N 4,096) on an L4, microseconds at 100% of peak
DOWN_PROJ_L4 = {1: (392, 102, 196), 256: (423, 248, 215), 2048: (1988, 1988, 992)}


@pytest.mark.parametrize("M", list(DOWN_PROJ_L4))
def test_gemm_table(M):
    got = tuple(round(B.gemm_time(M, 14336, 4096, "L4", s) * 1e6) for s in ("bf16", "w4a16", "fp8"))
    assert got == DOWN_PROJ_L4[M]


def test_where_w4a16_turns_compute_bound_and_where_its_edge_is_gone():
    for gpu, near in (("L4", 120), ("H100-80GB", 85)):                            # vllm-internals: ~120 / ~85
        assert abs(B.compute_bound_tokens(14336, 4096, gpu, "w4a16") - near) <= 10
    assert B.crossover_tokens(14336, 4096, "L4") > B.compute_bound_tokens(14336, 4096, "L4", "w4a16")


def test_decode_floors():
    """Streamed weight bytes / bandwidth: vllm-internals §8.2 (L4 50.0 / 26.8 / 15.6 ms; H100 4.48 / 2.40 / 1.40)."""
    assert [round(B.decode_floor_ms("llama-3.1-8b-instruct", "L4", s), 1) for s in ("bf16", "fp8", "w4a16")] == [50.0, 26.8, 15.6]
    assert [round(B.decode_floor_ms("llama-3.1-8b-instruct", "H100-80GB", s), 2) for s in ("bf16", "fp8", "w4a16")] == [4.48, 2.40, 1.40]


def test_scheme_paths():
    assert B.profile("llama-3.1-8b-instruct", "T4", "fp8").peak_flops == 65e12            # weight-only on Turing
    assert B.profile("llama-3.1-8b-instruct", "L4", "fp8").peak_flops == 242.5e12
    assert B.profile("llama-3.1-8b-instruct", "B200", "nvfp4").peak_flops == 9000e12
    with pytest.raises(ValueError):
        B.profile("llama-3.1-8b-instruct", "B200", "w8a8-int8")


def test_simulation_orders_the_schemes():
    rows = {r["scheme"]: r for r in B.compare(["bf16", "fp8", "w4a16"], users=8, n_requests=24)}
    assert rows["w4a16"]["tpot_ms_mean"] < rows["fp8"]["tpot_ms_mean"] < rows["bf16"]["tpot_ms_mean"]   # decode: bytes
    assert rows["fp8"]["ttft_ms_mean"] < rows["w4a16"]["ttft_ms_mean"] * 0.7                          # prefill: FLOPs
    assert all(r["source"].startswith("SIMULATED") for r in rows.values())


def test_prefill_flops_count_the_lm_head_once_per_sampled_sequence():
    """2 x linear x tokens + 2 x vocab x d per sampled sequence + 4 L H hd per attended pair
    (the same accounting as quantcore.cost.step_cost; vLLM computes logits only where it samples)."""
    p = B.profile("llama-3.1-8b-instruct", "L4", "bf16", overhead_s=0.0)
    linear, head = 6_979_321_856, 128_256 * 4_096                        # Llama-3.1-8B, untied
    assert (p.params_per_token - p.head_params, p.head_params, p.attn_flops_per_pair) == (linear, head, 4 * 32 * 32 * 128)
    n = 1800
    flops = 2 * linear * n + 2 * head * 1 + 4 * 32 * 32 * 128 * n * (n + 1) / 2
    assert p.step_time([], n) == pytest.approx(flops / (p.peak_flops * p.compute_eff))      # compute-bound
    mid_prompt = p.step_time([], n, sampled=0)                                                # a chunk that samples nothing
    assert mid_prompt == pytest.approx((flops - 2 * head) / (p.peak_flops * p.compute_eff))
    engine = B.Engine(p, max_num_batched_tokens=512)
    engine.add(B.Seq(0, 0.0, 1024, 4))
    first, second = engine.schedule(), None
    engine.commit(first, first.time)
    second = engine.schedule()
    assert first.time == pytest.approx(p.step_time([], 512, 0, sampled=0))                  # chunk 1 of 2: no logits
    assert second.time == pytest.approx(p.step_time([], 512, 512, sampled=1))               # chunk 2 samples one token


def test_fake_server_matches_the_emulator_over_http():
    """Deterministic quantities exactly; wall-clock TPOT only one-sided, because a busy CPU can only
    make the fake server's sleeps longer, never shorter."""
    prof = B.profile("qwen2.5-0.5b-instruct", "L4", "w4a16")
    with FakeServer(prof) as url:
        assert B.is_simulated(url) and B.server_kind(url) == "simulated"
        live = B.run_http(url, "quantlab-fake", users=2, n_requests=4, prompt_len=64, output_len=12)
    sim = B.summarize(B.from_seqs(B.closed_loop(2, 4, 64, 12, prof)), "sim")
    assert live["source"].startswith("SIMULATED") and live["requests"] == 4
    assert 0.8 * sim["tpot_ms_mean"] <= live["tpot_ms_mean"] < 5 * sim["tpot_ms_mean"]


def test_an_unidentified_server_is_not_labelled_simulated_or_vllm():
    assert B.server_kind("http://127.0.0.1:9/") is None                   # nothing listens on the discard port
    assert not B.is_simulated("https://127.0.0.1:9/")
