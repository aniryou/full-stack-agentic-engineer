"""The economics of a student: roofline serving cost, the fixed cost, break-even, the cascade (primer §1, §9)."""
import math

import pytest

from distillcore import cost as K

S, G = K.SHAPES, K.GPUS


def test_parameter_and_kv_counts():
    assert S["qwen2.5-32b"].params() == 32_762_757_120 and S["qwen2.5-1.5b"].params() == 1_543_569_408
    assert S["qwen2.5-0.5b"].params() == 493_961_216 and S["llama-3.1-8b"].params() == 8_029_995_008
    assert S["qwen2.5-32b"].kv_bytes_per_token() == 262_144 and S["qwen2.5-1.5b"].kv_bytes_per_token() == 28_672


def test_a_student_streams_a_twentieth_of_the_bytes():
    h = G["h100"]
    t32, t15 = K.decode_step(S["qwen2.5-32b"], h, 1, 1024), K.decode_step(S["qwen2.5-1.5b"], h, 1, 1024)
    assert round(t32 * 1e3, 3) == 19.175 and round(t15 * 1e3, 3) == 0.930
    with pytest.raises(KeyError):
        K.decode_step(S["qwen2.5-1.5b"], G["t4"], 1, 1024)          # a T4 has no bf16 path
    assert round(K.decode_step(S["qwen2.5-1.5b"], G["t4"], 1, 1024, precision="fp16") * 1e3, 3) == 9.739


def test_serving_teacher_and_student_under_one_itl_budget():
    h = G["h100"]
    t, s = K.serving(S["qwen2.5-32b"], h, 11, 2048, 0.030), K.serving(S["qwen2.5-1.5b"], h, 11, 2048, 0.030)
    assert t["batch"] == 12 and round(t["step_s"] * 1e3, 2) == 21.02 and round(t["usd_per_m"], 3) == 5.352
    assert s["batch"] == 1173 and round(s["tok_s"]) == 54575 and round(s["usd_per_m"], 4) == 0.0560
    assert round(t["usd_per_m"] / s["usd_per_m"]) == 96
    miss = K.serving(S["qwen2.5-32b"], h, 11, 2048, 0.010)
    assert miss["batch"] == 0 and miss["usd_per_m"] == math.inf    # batch 1 already takes 19.3 ms


def test_a_capacity_starved_teacher_is_not_a_fair_baseline():
    """One H100 leaves the 32B 6.47 GB for KV (batch 12, 96×); on two (ideal TP) it batches 146 and the ratio is 16×."""
    h, t, s = G["h100"], S["qwen2.5-32b"], S["qwen2.5-1.5b"]
    assert round(80 * 0.9 - K.weight_gb(t.params()), 2) == 6.47
    two = K.tp_group(h, 2)
    assert (two.memory_gb, two.tb_s, two.tflops["bf16"]) == (160, 6.7, 2 * 989.4) and K.tp_group(h, 1) is h
    tp2, st = K.serving(t, h, 11, 2048, 0.030, n_gpus=2), K.serving(s, h, 11, 2048, 0.030)
    assert tp2["batch"] == 146 and round(tp2["step_s"] * 1e3, 2) == 21.25 and round(tp2["usd_per_m"], 3) == 0.890
    assert math.isclose(tp2["usd_per_m"], 11 * 2 / (tp2["tok_s"] * 3600) * 1e6)     # two GPUs' price
    assert round(tp2["usd_per_m"] / st["usd_per_m"]) == 16
    assert round(K.serving(t, h, 11, 2048, 0.030, n_gpus=4)["usd_per_m"] / st["usd_per_m"]) == 11


def test_the_fixed_cost_is_mostly_teacher_tokens():
    h = G["h100"]
    f = K.fixed_cost(100_000, 1, 2000, 9.00, S["qwen2.5-1.5b"].params(), h, 11, mfu=0.4)
    assert f["tokens"] == 2e8 and f["generation_usd"] == 1800
    assert round(f["gpu_hours"], 3) == 1.300 and round(f["train_usd"], 2) == 14.30
    assert K.training_flops(1.5e9, 2e8) == 1.8e18


def test_break_even_and_the_cascade():
    b = K.break_even(1084.72, 5.352, 0.056, 50e6)
    assert round(b["days"], 2) == 4.10 and round(b["saving_per_day"], 1) == 264.8
    assert K.break_even(100, 1.0, 1.0, 1e6)["tokens"] == math.inf
    c = K.cascade(1.0, 10.0, (0.95, 0.30), (0.97, 0.85), hard=0.3, catch=0.8, false_alarm=0.1)
    assert math.isclose(c["to_teacher"], 0.31) and math.isclose(c["cost"], 4.1)
    assert math.isclose(c["accuracy"], 0.3 * (0.8 * 0.85 + 0.2 * 0.30) + 0.7 * (0.1 * 0.97 + 0.9 * 0.95))
    router = K.cascade(1.0, 10.0, (0.95, 0.30), (0.97, 0.85), hard=0.3, catch=0.8, false_alarm=0.1, student_first=False)
    assert math.isclose(router["cost"], 0.69 + 3.1) and router["accuracy"] == c["accuracy"]
