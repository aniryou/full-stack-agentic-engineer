"""Expert parallelism on two GPUs: flags, bytes on the wire, alpha-beta, per-GPU memory, the slowest GPU."""
from pathlib import Path

import numpy as np
import pytest

from moelab import configs as C
from moelab import ep, hooks

OLMOE, T4 = C.get("olmoe-1b-7b"), C.gpu("T4")
LINK = ep.LINKS["pcie-2xT4"]
FIX = Path(hooks.FIXTURES)


def test_flags_and_ep_size_follow_vllm():
    assert ep.vllm_flags("tp") == ["--tensor-parallel-size", "2"]
    assert ep.vllm_flags("tp_ep")[-1] == "--enable-expert-parallel"
    assert ep.vllm_flags("dp_ep")[:2] == ["--data-parallel-size", "2"]
    assert ep.ep_size(2, 1, True) == 2 and ep.ep_size(1, 2, True) == 2 and ep.ep_size(2, 2, False) == 1
    assert ep.ep_size(2, 4, True) == 8                                       # EP = TP x DP


def test_all_to_all_bytes_layer_02_and_deepep():
    assert ep.a2a_bytes(256, 2, 4096) == 4 * 2**20                            # 02 PRIMER §5.6
    d, c = ep.a2a_bytes(128, 8, 7168, 1, 128), ep.a2a_bytes(128, 8, 7168, 2)
    assert d == 7_569_408 and c == 14_680_064
    assert d / 77e-6 / 1e9 == pytest.approx(98.3, abs=0.05)                   # DeepEP EP8 LL dispatch: 98 GB/s
    assert c / 114e-6 / 1e9 == pytest.approx(128.8, abs=0.05)                 # combine: 127 reported


def test_alpha_beta_matches_layer_02():
    nv = ep.LINKS["nvlink-8gpu"]
    four = 4 * 2**20
    assert ep.collective_time("all_to_all", four, 8, nv, "pairwise") * 1e6 == pytest.approx(22.2, abs=0.1)
    assert ep.collective_time("all_to_all", four, 8, nv, "direct") * 1e6 == pytest.approx(10.2, abs=0.1)
    s = 1 << 20
    assert ep.collective_time("all_reduce", s, 2, LINK) == pytest.approx(2 * LINK.alpha_s + s / (LINK.bw_gbs * 1e9))
    assert ep.collective_time("all_reduce", s, 1, LINK) == 0.0


def test_comm_per_layer_dp_ep_has_no_attention_collective():
    tp = ep.moe_layer_comm("tp", 32, 2048, LINK)
    dp = ep.moe_layer_comm("dp_ep", 32, 2048, LINK)
    assert tp == pytest.approx(2 * dp)                                      # AG + RS = one all-reduce's cost at p=2
    with pytest.raises(ValueError):
        ep.moe_layer_comm("pp", 32, 2048, LINK)


def test_per_gpu_weights():
    tp = ep.per_gpu_weight_bytes(OLMOE, "tp")
    assert tp == ep.per_gpu_weight_bytes(OLMOE, "tp_ep")
    assert tp / C.GiB == pytest.approx(6.45, abs=0.01)
    assert ep.per_gpu_weight_bytes(OLMOE, "dp_ep") > tp                     # attention replicated under DP
    q = C.get("qwen1.5-moe-a2.7b")
    assert ep.per_gpu_weight_bytes(q, "tp") / C.GiB > 0.92 * T4.memory_gib - 1.5   # needs 24 GB cards


def test_tp_is_balanced_ep_is_not_at_batch_1():
    tp = ep.ep_decode_step(OLMOE, T4, "tp", 1, 512, LINK)
    tpe = ep.ep_decode_step(OLMOE, T4, "tp_ep", 1, 512, LINK)
    assert tp.imbalance == pytest.approx(1.0) and tpe.imbalance > 1.1
    assert tpe.time > tp.time
    big = ep.ep_decode_step(OLMOE, T4, "tp_ep", 128, 512, LINK)
    assert big.imbalance == pytest.approx(1.0, abs=0.02)                    # every expert touched on both GPUs


def test_step_is_the_sum_of_per_layer_maxima():
    st = ep.ep_decode_step(OLMOE, T4, "tp_ep", 8, 512, LINK)
    lay = np.array(st.layer_s)
    assert lay.shape == (OLMOE.layers, 2)
    assert st.compute_s == pytest.approx(lay.max(1).sum() + max(st.head_s))
    assert st.compute_s >= lay.sum(0).max()                                 # meeting per layer never helps


def test_skewed_routing_from_the_fixture_runs():
    ids = hooks.load_fixture().stacked()
    st = ep.ep_decode_step(OLMOE, T4, "dp_ep", 64, 512, LINK, ids=ids)
    assert st.time > 0 and len(st.rank_bytes) == 2
    rows = ep.compare_layouts(OLMOE, T4, LINK, batches=(1, 32))
    assert [r["batch"] for r in rows] == [1, 32] and "simulated" in ep.format_rows(rows)


def test_parse_bench_serve_on_every_fixture():
    files = sorted(FIX.glob("bench_serve_*.txt"))
    assert len(files) == 6
    for f in files:
        r = ep.parse_bench_serve(f.read_text())
        assert {"successful_requests", "mean_ttft_ms", "median_itl_ms", "p99_tpot_ms",
                "output_token_throughput_tok_s", "request_throughput_req_s"} <= set(r)
        assert f.read_text().startswith("# sample output in the documented format (illustrative)")


def test_parse_bench_serve_on_a_hand_written_block():
    text = "\n".join(["noise before", "{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="),
                      "{:<40} {:<10}".format("Successful requests:", 10),
                      "{:<40} {:<10.2f}".format("Mean ITL (ms):", 12.5),
                      "{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"),
                      "=" * 50, "{:<40} {:<10.2f}".format("Mean ITL (ms):", 99.0)])
    assert ep.parse_bench_serve(text) == {"successful_requests": 10, "mean_itl_ms": 12.5}


def test_bench_command_uses_v0_30_flags():
    cmd = ep.bench_command("m", 16)
    assert cmd[:3] == ["vllm", "bench", "serve"]
    for flag in ("--dataset-name", "--random-input-len", "--random-output-len", "--num-prompts", "--max-concurrency",
                 "--ignore-eos", "--base-url"):
        assert flag in cmd
    assert cmd[cmd.index("--num-prompts") + 1] == "128"
