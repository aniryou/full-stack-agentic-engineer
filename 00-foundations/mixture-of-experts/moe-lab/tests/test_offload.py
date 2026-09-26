"""Fitting an MoE on one small GPU: sizes, KV room, the offload toll, CPU experts, and the start-up log."""
from pathlib import Path

import pytest

from moelab import configs as C
from moelab import hooks, offload

OLMOE, T4, L4 = C.get("olmoe-1b-7b"), C.gpu("T4"), C.gpu("L4")
FIX = Path(hooks.FIXTURES)


def test_olmoe_16_bit_does_not_start_on_a_t4_but_int4_does():
    f16, i4 = offload.fit(OLMOE, T4, "fp16"), offload.fit(OLMOE, T4, "int4")
    assert not f16.fits and f16.kv_tokens == 0 and f16.kv_gib < 0
    assert i4.fits and i4.sessions > 10
    assert offload.fit(OLMOE, L4, "bf16").fits


def test_qwen3_int4_on_an_l4_leaves_about_3_gib_of_kv():
    f = offload.fit(C.get("qwen3-30b-a3b"), L4, "int4")
    assert f.kv_gib == pytest.approx(3.2, abs=0.05) and f.fits


def test_capability_rules():
    assert not offload.runnable(C.get("gpt-oss-20b"), T4, "mxfp4")[0]        # sm80+ and bf16
    assert offload.runnable(C.get("gpt-oss-20b"), L4, "mxfp4")[0]
    assert not offload.runnable(OLMOE, T4, "bf16")[0]                        # T4 has no bf16
    assert not offload.runnable(OLMOE, T4, "fp8")[0] and offload.runnable(OLMOE, L4, "fp8")[0]
    assert not offload.runnable(C.get("gpt-oss-20b"), L4, "int4")[0]         # ships MXFP4


def test_offloading_frees_exactly_what_it_moves():
    base, off = offload.fit(OLMOE, T4, "fp16"), offload.fit(OLMOE, T4, "fp16", offload_gib=6)
    assert off.kv_gib - base.kv_gib == pytest.approx(6.0)
    assert offload.min_offload_gib(OLMOE, T4, "fp16", kv_tokens=16_384) == 3.0
    assert offload.min_offload_gib(C.get("granite-3b-a800m"), T4, "fp16", kv_tokens=16_384) == 0.0


def test_uva_toll_is_per_step_and_routing_independent():
    assert offload.offload_step_s(4, 25) * 1e3 == pytest.approx(171.8, abs=0.1)     # ~170 ms (fact sheet §7)
    assert offload.offload_step_s(0, 25) == 0.0


def test_cpu_experts_grow_with_batch_and_cross_the_uva_toll():
    c = [offload.cpu_expert_step_s(OLMOE, b) for b in (1, 4, 16, 64)]
    assert c == sorted(c) and c[0] < offload.offload_step_s(3, T4.pcie_gbs) < c[-1]


def test_parse_startup_log_fixtures():
    t4 = offload.parse_startup_log((FIX / "vllm_startup_olmoe_t4_offload.log").read_text())
    assert t4["moe_default_config"] and t4["moe_config_file"] is None
    assert t4["max_model_len"] == 4096 and t4["kv_cache_tokens"] > 0
    off = offload.min_offload_gib(OLMOE, T4, "fp16", kv_tokens=4 * 4096)    # the log is the recommended offload
    pred = offload.fit(OLMOE, T4, "fp16", offload_gib=off)
    cmp_ = offload.compare_with_log(pred, t4)
    assert abs(cmp_["kv_error_gib"]) < 0.01 and cmp_["log_weights_gib"] == pytest.approx(pred.weights_gib - off, abs=0.01)
    assert f"--cpu-offload-gb {off:g} " in (FIX / "vllm_startup_olmoe_t4_offload.log").read_text()


def test_parse_startup_log_tuned_config_line():
    text = ("INFO [fused_moe.py:1151] Using configuration from /x/configs/E=64,N=1024,device_name=NVIDIA_H200.json "
            "for MoE layer.\nINFO Available KV cache memory: 100.50 GiB\n")
    log = offload.parse_startup_log(text)
    assert log["moe_config_file"].endswith("NVIDIA_H200.json") and not log["moe_default_config"]
    assert log["available_kv_gib"] == 100.5


def test_fit_line_is_readable():
    line = offload.fit(C.get("gpt-oss-20b"), T4, "mxfp4").line()
    assert "no: MXFP4" in line
