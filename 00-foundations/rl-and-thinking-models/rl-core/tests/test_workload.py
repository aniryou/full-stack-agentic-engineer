"""The serving workload. The no-thinking baseline reproduces the capacity primer's bank example
(00-foundations/gpu-capacity-planning: PRIMER.md, capacity.py, worked_example.py B) number for number, and
the cost function reproduces the 06 scaling lab's cost_per_call; then the thinking version."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from rlcore import workload as w

REPO = Path(__file__).resolve().parents[4]
H100, SMALL = w.GPUS["H100"], w.MISTRAL_SMALL
RPS = 10_000 * 0.10 * 0.5 / 60                                  # the bank: 8.33 requests/s


def _load(rel, name):
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                                     # dataclasses look their module up here
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True   # leave no cache in the other topic's dir
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


def test_reproduces_the_capacity_primers_bank_example():
    """worked_example.py B: 8.33 RPS, 12.07 s, 100.6 live, 95.3 / 381.3 sessions per GPU, 9,156 and 20,615 tok/s."""
    p = w.plan(SMALL, H100, RPS, 1500, 300, tpot_ms=40, dtype="fp8")
    assert round(RPS, 2) == 8.33 and round(p["duration_s"], 2) == 12.07 and round(p["concurrency"], 1) == 100.6
    assert p["avg_ctx"] == 1650 and round(w.ttft_s(24, 1500, H100), 4) == 0.0728
    assert round(w.sessions_per_gpu(SMALL, H100, 1650, "bf16"), 1) == 95.3
    assert round(w.sessions_per_gpu(SMALL, H100, 1650, "fp8"), 1) == 381.3
    assert round(w.decode_tok_s(SMALL, H100, 100, 1650, "fp8")) == 9156 == round(p["decode_tok_s_per_gpu"])
    assert round(w.prefill_tok_s(24, H100)) == 20615
    assert round(p["gpus"]["memory"], 3) == 0.264 and p["gpus_needed"] == 1


def test_matches_capacity_py_function_by_function():
    cap = _load("00-foundations/gpu-capacity-planning/capacity.py", "capacity")
    for dt in ("bf16", "fp8"):
        assert w.weight_gb(SMALL, dt) == cap.weight_memory_gb(24, dt)
        assert w.kv_per_token_kb(SMALL, dt) == cap.kv_per_token_kb(cap.MISTRAL_SMALL, dt)
        assert w.kv_per_session_gb(SMALL, 1650, dt) == cap.kv_per_session_gb(cap.MISTRAL_SMALL, 1650, dt)
        spare = cap.usable_hbm_gb(cap.GPUS["H100"]) - cap.weight_memory_gb(24, dt)
        assert math.isclose(w.sessions_per_gpu(SMALL, H100, 1650, dt),
                            cap.max_concurrent_sessions(spare, cap.MISTRAL_SMALL, 1650, dt))
    agg, _ = cap.decode_aggregate(cap.MISTRAL_SMALL, cap.GPUS["H100"], 100, 1650, "fp8")
    assert math.isclose(w.decode_tok_s(SMALL, H100, 100, 1650, "fp8"), agg)
    assert math.isclose(w.request_duration_s(24, 1500, 300, H100), cap.request_duration_s(24, 1500, 300, cap.GPUS["H100"]))
    assert math.isclose(w.prefill_tok_s(24, H100), cap.prefill_tok_s(24, cap.GPUS["H100"]))


def test_thinking_multiplies_memory_not_just_tokens():
    """10× the output (2,700 thinking + 300 answer): 120.07 s, 1,000.6 live, 209.7 sessions/GPU → 4.77 GPUs
    for memory, 18× the baseline's 0.264; decode_aggregate alone would have put all 1,000 on one GPU."""
    t = w.plan(SMALL, H100, RPS, 1500, 3000, tpot_ms=40, dtype="fp8")
    assert round(t["duration_s"], 2) == 120.07 and round(t["concurrency"], 1) == 1000.6 and t["avg_ctx"] == 3000
    assert round(t["sessions_per_gpu"], 1) == 209.7 and round(t["gpus"]["memory"], 2) == 4.77
    assert t["binding"] == "memory" and t["gpus_needed"] == 5 and t["batch"] == 209
    assert round(t["gpus"]["memory"] / 0.2638, 0) == 18
    assert 1000 * w.kv_per_session_gb(SMALL, 3000, "fp8") > 80           # the uncapped batch cannot exist


def test_a_tight_itl_slo_binds_before_hbm():
    t = w.plan(SMALL, H100, RPS, 1500, 3000, tpot_ms=20, dtype="fp8")
    assert t["itl_batch"] == 187 < t["sessions_per_gpu"] and t["binding"] == "itl_slots"
    assert w.decode_step_s(SMALL, H100, 187, 3000, "fp8") <= 0.020 < w.decode_step_s(SMALL, H100, 188, 3000, "fp8")


def test_steady_plan_closes_littles_law_on_the_step_the_fleet_runs_at():
    """plan() prices every token at the SLO's TPOT, so a 20 ms SLO needs fewer GPUs (3) than a 40 ms one (5).
    At 3 GPUs the fleet settles at 139.1 per GPU and 16.7 ms a step — under both SLOs — so both need 3, and the
    tight SLO's need is the larger. By hand (memory-bound step, FP8): c = rps·(TTFT + 3000·(24 + 0.2289·c/N)/3350)."""
    loose = w.plan_steady(SMALL, H100, RPS, 1500, 3000, tpot_ms=40)
    tight = w.plan_steady(SMALL, H100, RPS, 1500, 3000, tpot_ms=20)
    assert loose["gpus_needed"] == tight["gpus_needed"] == 3
    assert tight["gpus"]["itl_slots"] > loose["gpus"]["itl_slots"] and tight["binding"] == "itl_slots"
    kv, ttft = w.kv_per_session_gb(SMALL, 3000, "fp8"), w.ttft_s(24, 1500, H100)
    a, slope = RPS * (ttft + 3000 * 24 / 3350), RPS * 3000 * kv / 3350
    c = a / (1 - slope / 3)                                              # the fixed point, solved by hand
    assert loose["concurrency"] == pytest.approx(c, rel=1e-6) and round(c, 1) == 417.3
    assert loose["step_s"] == pytest.approx((24 + kv * c / 3) / 3350, rel=1e-6) and round(loose["step_s"] * 1e3, 1) == 16.7
    assert slope / 2 < 1 and a / (1 - slope / 2) / 2 > loose["sessions_per_gpu"]  # 2 GPUs settle past HBM: infeasible
    for b in (50, 100, 150, 200):                                        # N(b) falls as b grows
        assert w.gpus_for_batch(SMALL, H100, RPS, 1500, 3000, b) > w.gpus_for_batch(SMALL, H100, RPS, 1500, 3000, b + 1)
    base = w.plan_steady(SMALL, H100, RPS, 1500, 300)
    assert round(loose["gpus"]["memory"] / base["gpus"]["memory"]) == 18      # the 18× survives the convention


def test_a_gpu_without_fp8_refuses_an_fp8_plan():
    t4 = w.GPUS["T4"]
    with pytest.raises(ValueError):
        w.plan(w.QWEN3_0_6B, t4, 1.0, 500, 3000, dtype="fp8")
    assert w.plan(w.QWEN3_0_6B, t4, 1.0, 500, 3000, dtype="fp16")["gpus_needed"] >= 1


def test_kv_working_set_grows_with_the_square_of_output():
    assert w.kv_token_steps(1500, 300) == 1500 * 300 + 300 * 301 // 2 == 495_150
    assert round(w.kv_token_steps(1500, 3000) / w.kv_token_steps(1500, 300), 1) == 18.2
    assert round(w.kv_token_steps(1500, 1500) / w.kv_token_steps(1500, 300), 1) == 6.8


def test_heavy_tailed_thinking_lengths():
    assert round(w.lognormal_quantile(1500, 1.0, 0.5)) == 1500
    assert round(w.lognormal_mean(1500, 1.0)) == 2473 and round(w.lognormal_quantile(1500, 1.0, 0.99)) == 15361


def test_max_tokens_truncates_where_a_budget_answers():
    cut = w.budget_outcome(1500, 1.0, answer_tokens=300, e0=0.7, max_tokens=8192)
    forced = w.budget_outcome(1500, 1.0, answer_tokens=300, e0=0.7, budget=8192 - 300)
    assert round(cut["truncated"], 3) == 0.048 and round(cut["accuracy"], 3) == 0.952
    assert math.isclose(forced["accuracy"], cut["accuracy"] + 0.3 * cut["truncated"])
    assert math.isclose(forced["tokens"], cut["tokens"])
    assert w.budget_outcome(1500, 1.0)["accuracy"] == 1.0


def test_dropped_thinking_is_re_prefilled_never_reused():
    """Turn N's KV holds prompt + thinking + answer; the next prompt renders only the answer, so the hit
    ends at the end of turn N's prompt (rounded down to a block) and the answer is prefilled again."""
    drop = w.turn_prefills(1000, [(100, 800, 200)] * 3)
    keep = w.turn_prefills(1000, [(100, 800, 200)] * 3, keep_thinking=True)
    assert drop == [(1104, 0), (1408, 1104), (1712, 1408)]
    assert keep == [(1104, 0), (2208, 2096), (3312, 3200)]
    assert drop[1][0] - drop[1][1] == 200 + 100 + 4                  # answer + user + header prefilled again


def test_reproduces_the_scaling_labs_cost_per_call():
    assert round(w.api_cost(5000, 350, 2700), 6) == 0.007005 and round(w.api_cost(5000, 3500, 2700), 6) == 0.035355
    lab = _load("06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/capacity.py", "scalelab_capacity")
    assert math.isclose(w.api_cost(5000, 350, 2700), lab.cost_per_call("gemini-3.5-flash", 5000, 350, 2700))
    assert w.cost_per_correct(0.01, 0.5) == 0.02


def test_rollouts_wait_for_the_longest_completion():
    lengths = [100] * 63 + [1000]
    r = w.rl_step_time(SMALL, H100, 1, 200, lengths)
    assert r["batch_occupancy"] < 0.2                                 # 63 finish at step 100; one runs to 1,000
    even = w.rl_step_time(SMALL, H100, 1, 200, [int(sum(lengths) / 64)] * 64)
    assert even["generate_s"] < r["generate_s"] / 2 and math.isclose(even["train_s"], r["train_s"], rel_tol=1e-3)
