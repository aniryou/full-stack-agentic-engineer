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


def test_fake_server_matches_the_emulator_over_http():
    prof = B.profile("qwen2.5-0.5b-instruct", "L4", "w4a16")
    with FakeServer(prof) as url:
        assert B.is_simulated(url)
        live = B.run_http(url, "quantlab-fake", users=2, n_requests=4, prompt_len=64, output_len=12)
    sim = B.summarize(B.from_seqs(B.closed_loop(2, 4, 64, 12, prof)), "sim")
    assert live["source"].startswith("SIMULATED") and live["requests"] == 4
    assert live["tpot_ms_mean"] == pytest.approx(sim["tpot_ms_mean"], rel=0.5)
