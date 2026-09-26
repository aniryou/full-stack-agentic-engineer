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


def test_each_rank_is_its_own_roofline_and_the_slowest_sets_the_layer():
    """Two ranks, two experts each; rank 1's expert 2 is hot. Rows: [2, 8]; touched: [2, 1]."""
    idx = np.array([[0], [1], [2], [2], [2], [2], [2], [2], [2], [2]])
    origin, where = np.array([0] * 5 + [1] * 5), np.array([0, 0, 1, 1])
    dev = T.Device("toy", {"bf16": 1.0}, 1e-3, 1)                    # 1 TFLOP/s, 1 GB/s: ridge 1,000
    lt = E.layer_time(idx, origin, where, 2, 1e6, 1024, dev, NV)
    assert lt.rows.tolist() == [2, 8] and lt.imbalance == pytest.approx(1.6)
    assert E.touched_per_rank(idx, where, 2).tolist() == [2, 1]
    np.testing.assert_allclose(lt.compute, [2 * 2e6 / 1e12, 8 * 2e6 / 1e12])      # rows x 2P / peak
    np.testing.assert_allclose(lt.memory, [2 * 2e6 / 1e9, 1 * 2e6 / 1e9])          # touched x P x 2 B / BW
    np.testing.assert_allclose(lt.rank, lt.memory)                  # 8 rows << ridge: bytes, not FLOPs
    assert lt.total == pytest.approx(2 * lt.comm + lt.memory[0])    # rank 0 reads two experts: it is slowest
    # the busiest port: rank 1 receives 3 remote rows in dispatch (and sends them back in combine)
    assert lt.comm == pytest.approx(2e-6 + 3 * 1024 * 2 / 450e9)
    assert lt.comm_even == pytest.approx(2e-6 + 3 / 2 * 1024 * 2 / 450e9)    # 3 remote rows over 2 ports


def test_skew_barely_moves_decode_gemms_but_slows_prefill():
    """Qwen3-30B-A3B on 8 H100s (NVLink), Zipf s = 1.0 skew, vLLM's linear placement. At 128 tokens per
    GPU each expert sees ~64 rows, far below the ridge (295): every rank streams its 16 experts in
    45 us whatever the skew, and the skew's cost is the hot rank's port. At 4,096 tokens per GPU the
    GEMMs are compute-bound and the busiest rank's 1.65x rows set the layer."""
    q3, h100 = MODELS["qwen3-30b-a3b"], T.DEVICES["h100-sxm"]
    pop = T.zipf_popularity(128, 1.0)[np.random.default_rng(0).permutation(128)]
    res = {}
    for tpg in (128, 4096):
        idx = T.sample_routes(128, 8, 8 * tpg, pop, np.random.default_rng(1))
        res[tpg] = E.layer_time(idx, np.repeat(np.arange(8), tpg), E.placement(128, 8), 8,
                                q3.expert_params(), 2048, h100, NV)
    dec, pre = res[128], res[4096]
    assert dec.imbalance > 1.6 and pre.imbalance > 1.6                     # the same skew in rows
    assert dec.bound == "memory" and np.ptp(dec.rank) == 0                 # decode: balanced expert time
    assert dec.rank.max() == pytest.approx(16 * q3.expert_params() * 2 / h100.bandwidth())
    assert dec.comm > 1.4 * dec.comm_even and 1.1 < dec.penalty < 1.2     # what skew costs: the exchange
    assert pre.bound == "compute" and pre.penalty > 1.5                    # prefill: the rows
    assert pre.rank.max() / pre.rank.mean() == pytest.approx(pre.imbalance)


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
    and all-to-all kernels ships 8 tokens x 2 x 4096 x 2 B = 128 KiB each way (direct, 2 x 2.25 us);
    vLLM's default allgather_reducescatter moves TP's volume (all-gather 448 KiB + reduce-scatter 448 KiB)."""
    tp_b, tp_t = E.moe_comm(64, 2, 4096, 8, NV, "tp")
    ep_b, ep_t = E.moe_comm(8, 2, 4096, 8, NV, "a2a")
    ag_b, ag_t = E.moe_comm(8, 2, 4096, 8, NV, "agrs")
    assert tp_b == 2 * 7 / 8 * 524_288 and round(tp_t * 1e6, 1) == 30.0
    assert ep_b == 2 * 7 / 8 * 131_072 and round(ep_t * 1e6, 2) == 4.51
    assert ag_b == tp_b == 2 * 7 * 8 * 4096 * 2 and ag_t == pytest.approx(tp_t)
    with pytest.raises(ValueError):
        E.moe_comm(8, 2, 4096, 8, NV, "ep")


def test_fp8_dispatch_and_a_two_level_fabric():
    """DeepSeek-V3 dispatches in FP8 (+ a 4-byte scale per 128) and combines in BF16. Across two 8-GPU
    nodes (EP 16) a GPU's 7 node peers are on NVLink and only 8/16 of the traffic crosses the NIC."""
    ib = E.LINKS["ib-ndr"]
    _, bf16 = E.moe_comm(32, 8, 7168, 16, ib, "a2a")
    _, fp8 = E.moe_comm(32, 8, 7168, 16, ib, "a2a", dispatch_elem=1, scale_block=128)
    assert fp8 < bf16
    s = E.dispatch_bytes(32, 8, 7168)
    two = E.a2a_time(s, 16, ib, per_node=8, intra=NV)
    assert two == pytest.approx(max(2e-6 + 7 / 16 * s / 450e9, 5e-6 + 8 / 16 * s / 50e9))
    assert E.a2a_time(s, 16, NV) < two < E.a2a_time(s, 16, ib)
    assert E.a2a_time(s, 8, ib, per_node=8, intra=NV) == E.a2a_time(s, 8, ib)   # one node: no split


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
