"""Expert parallelism: all-to-all bytes and time, the slowest rank, EPLB, TP vs EP, wide-EP memory."""
import numpy as np
import pytest

from moecore import ep as E
from moecore import touched as T
from moecore.sizing import MODELS

NV = E.LINKS["nvlink4"]


def test_reproduces_the_cuda_nccl_primer_ep_numbers():
    """02-cuda-nccl-runtime PRIMER §5.6 (gpusim.collectives.model_time, alpha 2 us, B 450 GB/s, 8 GPUs):
    Mixtral-like, hidden 4,096, top-2, 256 tokens per GPU -> 4 MiB per direction, 22 us pairwise, 10 us direct."""
    s = E.dispatch_bytes(256, 2, 4096)
    assert s == 4 * 2 ** 20
    assert round(E.a2a_time(s, 8, NV, "pairwise") * 1e6) == 22
    assert round(E.a2a_time(s, 8, NV, "direct") * 1e6) == 10
    assert E.a2a_time(s, 1, NV) == 0.0


def test_the_formula_explains_deepeps_published_low_latency_numbers():
    """DeepEP V1 low-latency, EP8: 128 tokens, hidden 7168, top-8, FP8 dispatch (a 4-byte scale per 128
    channels), BF16 combine; 77 us / 114 us reported as 98 / 127 GB/s."""
    d = E.dispatch_bytes(128, 8, 7168, elem_bytes=1, scale_block=128)
    c = E.dispatch_bytes(128, 8, 7168, elem_bytes=2)
    assert d == 7_569_408 and c == 14_680_064
    assert round(d / 77e-6 / 1e9, 1) == 98.3 and round(c / 114e-6 / 1e9, 1) == 128.8


def test_exchange_counts_every_assignment_once():
    rng = np.random.default_rng(0)
    idx = T.sample_routes(64, 8, 1024, rng=rng)
    m = E.exchange(idx, np.repeat(np.arange(8), 128), E.placement(64, 8), 8)
    assert (m.sum(axis=1) == 128 * 8).all()                             # each rank sends tokens x k rows
    assert np.trace(m) / m.sum() == pytest.approx(1 / 8, abs=0.02)      # (p-1)/p leaves the GPU


def test_the_slowest_rank_sets_the_layer_time():
    m = np.array([[10, 10], [10, 70]])                                  # rank 1 holds a hot expert
    lt = E.layer_time(m, 1e6, 1024, T.DEVICES["h100-sxm"], NV)
    assert lt.compute[1] == 4 * lt.compute[0] and lt.imbalance == pytest.approx(1.6)
    assert lt.total == pytest.approx(2 * lt.comm + lt.compute[1])


def test_placement_and_eplb_rebalance():
    assert E.placement(8, 4).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert E.placement(8, 4, "round_robin").tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    load = np.array([40, 10, 10, 10, 10, 10, 5, 5])
    by_linear = np.bincount(E.placement(8, 4), weights=load)          # [50, 20, 20, 10]
    assert by_linear.max() / by_linear.mean() == pytest.approx(2.0)
    packed = E.rebalance(load, 4)                                       # heaviest first, 2 slots per rank
    assert packed.tolist() == [45, 20, 20, 15]                          # the hot expert's rank still waits
    twin = E.rebalance(load, 4, redundant=4)                            # 12 copies, 3 slots per rank
    assert twin.tolist() == [30, 30, 20, 20] and twin.max() / twin.mean() == pytest.approx(1.2)


def test_eplb_redundant_expert_memory_matches_vllms_note():
    """vLLM expert_parallel_deployment.md: ~2.4 GB for one redundant expert per EP rank (DeepSeek-V3, FP8):
    58 MoE layers x 44,040,192 B = 2,554,331,136 B = 2.38 GiB."""
    ds = MODELS["deepseek-v3"]
    extra = E.wide_ep_weights(ds, 32, 1, redundant=32) - E.wide_ep_weights(ds, 32, 1)
    assert extra == pytest.approx(2_554_331_136) and round(extra / 2 ** 30, 2) == 2.38


def test_wide_ep_replicates_everything_but_the_routed_experts():
    ds = MODELS["deepseek-v3"]
    replicated = ds.total() - ds.moe_layers * ds.n_experts * ds.expert_params()
    assert replicated == 17_116_766_720                                 # 17.1 B: MLA, dense layers, shared experts, embeddings
    assert round(E.wide_ep_weights(ds, 8, 1) / 1e9, 1) == 98.9         # FP8 per GPU at EP 8
    assert round(E.wide_ep_weights(ds, 32, 1) / 1e9, 1) == 37.6         # frees HBM for KV


def test_tp_vs_ep_communication_for_one_mixtral_layer():
    """Batch 64 on 8 GPUs: TP all-reduces 64 x 4096 x 2 B = 512 KiB (ring, 30 us); EP with DP attention
    ships 8 tokens x 2 x 4096 x 2 B = 128 KiB each way (direct, 2 x 2.25 us)."""
    tp_b, tp_t = E.moe_comm(64, 2, 4096, 8, NV, "tp")
    ep_b, ep_t = E.moe_comm(8, 2, 4096, 8, NV, "ep")
    assert tp_b == 2 * 7 / 8 * 524_288 and round(tp_t * 1e6, 1) == 30.0
    assert ep_b == 2 * 7 / 8 * 131_072 and round(ep_t * 1e6, 2) == 4.51


def test_one_gpu_decode_on_equals_the_single_device_roofline():
    mix, h200 = MODELS["mixtral-8x7b"], T.DEVICES["h200"]
    for layout in ("ep", "tp"):
        st = E.decode_on(mix, h200, 16, 1024, 1, layout)
        assert st["bytes_per_gpu"] == pytest.approx(T.decode_step(mix, h200, 16, 1024).bytes)
        assert st["comm"] == 0.0


def test_ep_across_a_slow_fabric_costs_more():
    mix, h100 = MODELS["mixtral-8x7b"], T.DEVICES["h100-sxm"]
    fast = E.decode_on(mix, h100, 64, 4096, 8, "ep", NV)
    slow = E.decode_on(mix, h100, 64, 4096, 8, "ep", E.LINKS["ib-ndr"])
    assert slow["comm"] > 3 * fast["comm"] and slow["roofline"] == fast["roofline"]
