"""What a scheme runs as on each GPU, and what it buys."""
import pytest

from quantcore import cost as C


def test_what_a_checkpoint_runs_as():
    runs = lambda g, s: C.supported(C.GPUS[g], s)["runs_as"]
    assert runs("A100-80GB", "w8a8-fp8") == "w8a16-fp8" and runs("T4", "w8a8-fp8") == "w8a16-fp8"
    assert runs("L4", "w8a8-fp8") == "w8a8-fp8" and runs("H100-SXM", "w4a4-nvfp4") == "w4a16"
    assert runs("B200", "w8a8-int8") is None and runs("B200", "w4a4-nvfp4") == "w4a4-nvfp4"
    assert C.supported(C.GPUS["H100-SXM"], "w4a16")["kernel"] == "Machete"
    assert not C.supported(C.GPUS["T4"], "bf16", kv_bits=8)["kv"] and C.supported(C.GPUS["A100-80GB"], "bf16", 8)["kv"]


def test_unsupported_steps_refuse():
    with pytest.raises(ValueError, match="does not run"):
        C.step_cost(C.GPUS["T4"], C.MODELS["llama-3.1-8b"], "w4a16", [(0, 1)], kv_bits=8)


def test_weight_only_cuts_bytes_not_flops():
    L4, m = C.GPUS["L4"], C.MODELS["llama-3.1-8b"]
    b, w4, f8 = (C.step_cost(L4, m, s, [(0, 1800)]) for s in ("bf16", "w4a16", "w8a8-fp8"))
    assert w4["flops"] == b["flops"] and w4["t"] == b["t"] and f8["t"] < 0.55 * b["t"]
    d = lambda s: C.step_cost(L4, m, s, [(1000, 1)])["t"]
    assert d("w4a16") < 0.35 * d("bf16")


def test_gemm_bounds_and_crossover():
    L4 = C.GPUS["L4"]
    assert C.gemm_time(1, 4096, 4096, L4, "w4a16")["bound"] == "memory"
    m = C.crossover_tokens(4096, 4096, L4)
    assert C.gemm_time(int(m) - 1, 4096, 4096, L4, "w4a16")["bound"] == "memory"
    assert C.gemm_time(int(m) + 2, 4096, 4096, L4, "w4a16")["bound"] == "compute"


def test_kv_blocks_hand_computed():
    L4, m = C.GPUS["L4"], C.MODELS["llama-3.1-8b"]
    assert C.kv_blocks(L4, m, "bf16") == int((0.9 * 24e9 - m.weight_bytes(16) - 1e9) // (16 * 131072))
    assert C.sessions(L4, m, "bf16", 2000) == C.kv_blocks(L4, m, "bf16") // 125


def test_choose_the_least_aggressive_scheme_that_meets_the_targets():
    rows = C.table(C.GPUS["L4"], C.MODELS["llama-3.1-8b"])
    pick = C.choose(rows, min_sessions=48, max_prefill_ms=300)       # minengine notebook 06, exercise 6.5
    assert (pick["scheme"], pick["kv_bits"]) == ("w8a8-fp8", 8)
    assert C.choose(rows, min_sessions=40)["scheme"] == "w8a16-fp8"
    assert C.choose(rows, min_sessions=500) is None
    t4 = C.table(C.GPUS["T4"], C.MODELS["llama-3.1-8b"], kv=(16,))
    assert "does not fit" in t4[0]["note"] and "prefill_ms" not in t4[0]


def test_cost_per_million_hand_computed():
    assert C.cost_per_million(3.6, 1000, 1.0) == pytest.approx(1.0)
    assert C.cost_per_million(0.70, 500, 0.5) == pytest.approx(0.70 / (500 * 3600 * 0.5) * 1e6)
