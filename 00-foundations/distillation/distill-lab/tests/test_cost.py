"""The economics, pinned to the fact sheet's worked numbers (computed there with the repo's roofline)."""
import math

import pytest

from distillab import cost as C
from distillab import metrics as M


def test_teacher_and_student_on_an_h100():
    rows = {n: C.serving(C.shape(n), C.GPUS["H100"], context=2048, itl_s=0.030)
            for n in ("qwen2.5-32b-instruct", "qwen2.5-1.5b-instruct", "qwen2.5-0.5b-instruct")}
    t, s, s2 = rows.values()
    assert (t["batch"], t["tok/s"]) == (12, 571) and t["$/M"] == pytest.approx(5.352, abs=5e-4)   # HBM caps the 32B
    assert (s["batch"], s["tok/s"]) == (1173, 54575) and s["$/M"] == pytest.approx(0.0560, abs=5e-5)
    assert (s2["batch"], s2["tok/s"]) == (2821, 131218) and s2["$/M"] == pytest.approx(0.0233, abs=5e-5)
    assert t["$/M"] / s["$/M"] == pytest.approx(95.6, abs=0.5)
    # at a 10 ms ITL the 32B misses at batch 1 (19.2 ms): no throughput, reported instead of dividing by zero
    miss = C.serving(C.shape("qwen2.5-32b-instruct"), C.GPUS["H100"], context=2048, itl_s=0.010)
    assert miss["batch"] == 0 and miss["$/M"] == math.inf and miss["step ms"] == pytest.approx(19.26, abs=0.01)
    assert 1e3 * C.decode_step(C.shape("qwen2.5-32b-instruct"), C.GPUS["H100"], 1, 1024) == pytest.approx(19.175, abs=1e-3)


def test_price_the_teacher_on_the_gpus_you_would_give_it():
    """On two H100s (ideal TP) the 32B batches 146 at $0.890/M: 16× the 1.5B student, not the one-GPU 96×."""
    h, t, s = C.GPUS["H100"], C.shape("qwen2.5-32b-instruct"), C.shape("qwen2.5-1.5b-instruct")
    tp2 = C.serving(t, h, context=2048, itl_s=0.030, n_gpus=2)
    st = C.serving(s, h, context=2048, itl_s=0.030)
    assert (tp2["batch"], tp2["tok/s"], tp2["gpu"]) == (146, 6870, "2xH100") and tp2["$/M"] == pytest.approx(0.8896, abs=5e-5)
    assert tp2["$/M"] == pytest.approx(C.cost_per_million_tokens(11.0, 6869.80, n_gpus=2), rel=1e-4)
    assert round(tp2["$/M"] / st["$/M"]) == 16 and C.tp_group(h, 1) is h and C.tp_group(h, 2).memory_gb == 160


def test_t4_has_no_bf16_and_small_models_decode_fast():
    m = C.shape("qwen2.5-0.5b-instruct")
    with pytest.raises(KeyError):
        C.decode_step(m, C.GPUS["T4"], 1, 1024)
    assert 1e3 * C.decode_step(m, C.GPUS["T4"], 1, 1024, precision="fp16") == pytest.approx(3.127, abs=5e-3)
    assert 1e3 * C.decode_step(C.shape("qwen3-4b"), C.GPUS["T4"], 1, 1024, precision="fp16") == pytest.approx(25.612, abs=0.05)


def test_fixed_cost_break_even_and_cascade():
    f = C.fixed_cost(teacher_tokens=2e8, teacher_price_per_m=9.0, student_params=1.5e9, train_tokens=2e8,
                     gpu=C.GPUS["H100"], mfu=0.4)
    assert f["teacher generation $"] == 1800 and f["training $"] == pytest.approx(13.90, abs=0.01)
    assert C.fixed_cost(teacher_tokens=2e8, teacher_price_per_m=9.0, student_params=1.5e9, train_tokens=2e8,
                        gpu=C.GPUS["H100"], mfu=0.3)["training GPU-hours"] == pytest.approx(1.685, abs=1e-3)
    be = C.break_even(1000.0, 5.0, 1.0, 50e6)
    assert be["million tokens"] == 250 and be["days"] == 5
    assert C.break_even(10, 1.0, 1.0, 1e6)["days"] == math.inf
    up = C.cascade(0.2, 0.05, 5.0, acc_student_kept=0.9, acc_teacher_routed=0.8, student_first=False)
    esc = C.cascade(0.2, 0.05, 5.0, acc_student_kept=0.9, acc_teacher_routed=0.8)
    assert up["$/M"] == pytest.approx(0.8 * 0.05 + 0.2 * 5) and esc["$/M"] == pytest.approx(0.05 + 0.2 * 5)
    assert up["accuracy"] == pytest.approx(0.88) and C.cost_per_correct(1.0, 0.5) == 2.0


def test_measured_throughput_from_two_scrapes():
    before = M.parse('vllm:generation_tokens_total{model_name="m"} 1000\nvllm:request_success_total{finished_reason="stop"} 3\n')
    after = M.parse('vllm:generation_tokens_total{model_name="m"} 61000\nvllm:request_success_total{finished_reason="stop"} 53\n')
    r = C.from_metrics(before, after, 0.35, seconds=60)
    assert r["output_tok_s"] == 1000 and r["requests"] == 50
    assert r["$/M"] == pytest.approx(0.35 / 3.6)
