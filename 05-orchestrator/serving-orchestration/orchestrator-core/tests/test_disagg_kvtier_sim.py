"""P/D and KV-tier math pinned to hand-computed numbers; the simulator is deterministic and obeys Little's law."""
import pytest

from fleetsim import (L4_8B, H100_8B, LLAMA_8B_KV, Autoscaler, Fleet, HPA, PowerOfTwo, Tier, TieredKV, agentic,
                      breakeven_gb_s, chat, decode_step_s, expand, kv_bytes, max_decode_batch, onload_s, pd_plan,
                      percentile, rag, recompute_s, run_pd, simulate_sessions, transfer_s, working_set_gb)


def test_kv_transfer_bytes_and_time():
    assert kv_bytes(4096, LLAMA_8B_KV) == 536_870_912                        # 4,096 x 128 KiB = 512 MiB
    assert transfer_s(4096, LLAMA_8B_KV, 400) == pytest.approx(0.010737, abs=1e-6)   # 400 Gb/s RDMA: 10.7 ms
    assert transfer_s(4096, LLAMA_8B_KV, 10) == pytest.approx(0.429497, abs=1e-6)    # 10 GbE: 0.43 s
    assert transfer_s(4096, LLAMA_8B_KV, 10, overlap=0.5) == pytest.approx(0.214748, abs=1e-6)


def test_decode_batch_is_capped_by_itl_slo_and_by_kv():
    # L4, 4,100 tokens of context: each request adds 4,101 x 0.437 us = 1.79 ms per step on top of 56.3 ms
    assert decode_step_s(L4_8B, 5, 4100) == pytest.approx(0.003 + 0.053333 + 5 * 4101 * 4.3691e-7, abs=1e-5)
    assert max_decode_batch(L4_8B, 4100, itl_slo_s=0.1) == 8            # KV: 2,193 // 257 blocks = 8 requests
    assert max_decode_batch(L4_8B, 500, itl_slo_s=0.07) == 62           # ITL: 56.3 ms + b x 0.219 ms <= 70 ms


def test_pd_plan_numbers():
    plan = pd_plan(L4_8B, rate=1.0, isl=4000, osl=200, itl_slo_s=0.1)
    assert plan["prefill_tok_s_demand"] == 4000
    assert plan["prefill_replicas"] == pytest.approx(4000 / (3781.25 * 0.7))       # 1.51
    assert plan["decode_batch"] == 8
    assert plan["decode_replicas"] == pytest.approx(200 / (8 / decode_step_s(L4_8B, 8, 4100)))


def test_breakeven_bandwidth_and_onload_vs_recompute():
    assert breakeven_gb_s(LLAMA_8B_KV, L4_8B.compute_tok_s) == pytest.approx(0.4956, abs=1e-4)
    assert breakeven_gb_s(LLAMA_8B_KV, H100_8B.compute_tok_s) == pytest.approx(4.0509, abs=1e-4)
    dram = Tier("DRAM", 256, 50.0, 0.0005)
    assert onload_s(10_000, LLAMA_8B_KV, dram) == pytest.approx(0.0005 + 1.31072e9 / 50e9)   # 26.7 ms
    assert recompute_s(10_000, H100_8B.compute_tok_s) == pytest.approx(0.32356, abs=1e-5)    # 12x slower
    assert working_set_gb(200, 30_000, LLAMA_8B_KV) == pytest.approx(786.432)


def test_tiers_demote_on_evict_and_drop_off_the_end():
    kv = TieredKV([Tier("HBM", 2.0, 3000), Tier("DRAM", 2.0, 50)], kv_bytes_per_token=1_000_000)   # 1 MB/token
    kv.put("a", 1500)
    kv.put("b", 1500)                            # HBM holds 2 GB: "a" (LRU) is demoted to DRAM
    assert kv.lookup("a") == (1, 1500) and kv.lookup("b") == (0, 1500)
    kv.put("c", 1500)                            # "b" -> DRAM, which pushes "a" off the last tier
    assert kv.lookup("a") == (None, 0) and kv.lookup("b")[0] == 1 and kv.lookup("c")[0] == 0


def test_sticky_routing_or_a_shared_tier_keeps_agent_kv_reusable():
    common = dict(kv_bytes_per_token=LLAMA_8B_KV, prefill_tok_s=H100_8B.compute_tok_s, replicas=4, seed=3)
    tiers = [Tier("HBM", 8, 3350), Tier("DRAM", 64, 50, 0.0005)]
    sticky = simulate_sessions(300, tiers=tiers, sticky=True, **common)
    spray = simulate_sessions(300, tiers=tiers, sticky=False, **common)
    shared = simulate_sessions(300, tiers=tiers[:1], sticky=False, shared=Tier("shared", 500, 20, 0.002), **common)
    assert sticky["recompute_token_share"] < spray["recompute_token_share"]
    assert shared["recompute_token_share"] == pytest.approx(sticky["recompute_token_share"], abs=0.01)


def test_percentile_interpolates_like_numpy():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_simulation_is_deterministic_and_complete():
    def run():
        return Fleet(L4_8B, 3, PowerOfTwo(seed=2)).run(agentic(0.3, 60, seed=9)).summary()
    a, b = run(), run()
    assert a == b and a["requests"] == len(expand(agentic(0.3, 60, seed=9)))


def test_littles_law_holds_in_the_simulator():
    reqs = chat(2.0, 600, seed=3)
    res = Fleet(L4_8B, 2, PowerOfTwo(seed=1), sample_s=5.0).run(reqs)
    lam = len(reqs) / 600
    w = sum(r.t_done - r.arrival for r in res.requests) / len(reqs)
    in_system = [row["waiting"] + row["running"] for row in res.timeline if 60 <= row["t"] <= 600]
    assert sum(in_system) / len(in_system) == pytest.approx(lam * w, rel=0.15)     # L = lambda x W


def test_disaggregation_removes_the_prefill_stall_from_decode():
    def reqs():
        return rag(0.5, 120, seed=11, docs=4, doc=1400, corpus=5000, zipf=0.0, output=150)
    agg = run_pd(L4_8B, reqs(), 0, 4, router=PowerOfTwo(seed=1)).summary()
    pd = run_pd(L4_8B, reqs(), 1, 3, router=PowerOfTwo(seed=1)).summary()
    assert agg["itl_p99"] > 0.5 and pd["itl_p99"] < 0.1           # a 2,048-token chunk stalls decodes 0.54 s


def test_scale_to_zero_holds_requests_until_a_replica_is_ready():
    hpa = HPA(0, 4)
    auto = Autoscaler(hpa, metric="inflight", target=10, kind="external")
    reqs = chat([(0, 0.0), (400, 2.0), (460, 0.0)], 900, seed=1)
    res = Fleet(L4_8B, 1, PowerOfTwo(seed=1), autoscaler=auto, cold_start_s=60).run(reqs, horizon=1200)
    assert any(row["replicas"] == 0 for row in res.timeline)                      # idle: scaled to zero
    first = min(reqs, key=lambda r: r.arrival)
    assert first.t_first - first.arrival >= 45                                    # the first request eats a cold start
    assert all(r.t_done is not None for r in reqs)
