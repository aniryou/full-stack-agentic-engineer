"""The emulated engine: budgets, chunked prefill, prefix hits, LRU eviction, preemption, roofline, speculation."""
import random

import pytest

from servelab.fake_engine import EngineConfig, FakeEngine, profile, seq_timings, simulate, tiny_profile


def prompt(n, seed=0):
    rng = random.Random(seed)
    return [rng.randint(1, 50_000) for _ in range(n)]


def test_prefix_hit_is_block_aligned_and_leaves_one_token_to_compute():
    e = FakeEngine(tiny_profile(block_size=4), EngineConfig(max_num_batched_tokens=128))
    a, b = simulate(e, [(0.0, prompt(64), 2), (1.0, prompt(64), 2)])
    assert a.cached_tokens == 0 and b.cached_tokens == 60        # (64-1)//4 = 15 blocks; the last token is recomputed
    c, d = simulate(FakeEngine(tiny_profile(block_size=4), EngineConfig(max_num_batched_tokens=128)),
                    [(0.0, prompt(66), 2), (1.0, prompt(66), 2)])
    assert d.cached_tokens == 64                                   # a partial last block is never shared
    assert seq_timings(b)["ttft"] < seq_timings(a)["ttft"]


def test_no_hits_when_prefix_caching_is_off():
    e = FakeEngine(tiny_profile(), EngineConfig(enable_prefix_caching=False))
    a, b = simulate(e, [(0.0, prompt(64), 2), (1.0, prompt(64), 2)])
    assert b.cached_tokens == 0


def test_step_budgets_and_chunked_prefill():
    e = FakeEngine(tiny_profile(num_blocks=512, max_model_len=2048),
                   EngineConfig(max_num_seqs=3, max_num_batched_tokens=32))
    for i in range(6):
        e.add_request(prompt(100, seed=i), 5, 0.0)
    t, first_steps = 0.0, 0
    while e.has_work():
        plan = e.schedule(t)
        assert plan.num_tokens <= 32 and len(e.running) <= 3
        t += e.step_time(plan)
        res = e.commit(plan, t)
        first_steps += 1
        if any(ev.first for ev in res.events):
            break
    assert first_steps == 4                     # a 100-token prompt needs ceil(100/32) = 4 chunks of the budget


def test_chunked_prefill_off_requires_a_budget_that_fits_a_whole_prompt():
    with pytest.raises(ValueError):
        FakeEngine(tiny_profile(max_model_len=512), EngineConfig(enable_chunked_prefill=False, max_num_batched_tokens=256))


def test_lru_keeps_recently_used_prefixes():
    cfg = EngineConfig(max_num_batched_tokens=256)
    # small second request: allocations come from never-used blocks first, A's cached blocks survive
    e = FakeEngine(tiny_profile(num_blocks=16, block_size=4), cfg)
    a1, _, a2 = simulate(e, [(0.0, prompt(30, 1), 2), (1.0, prompt(12, 2), 2), (2.0, prompt(30, 1), 2)])
    assert a2.cached_tokens == 28
    # a request that needs every block evicts A's cached prefix
    e = FakeEngine(tiny_profile(num_blocks=16, block_size=4), cfg)
    _, _, a2 = simulate(e, [(0.0, prompt(30, 1), 2), (1.0, prompt(62, 2), 2), (2.0, prompt(30, 1), 2)])
    assert a2.cached_tokens == 0


def test_preemption_by_recompute_when_blocks_run_out():
    e = FakeEngine(tiny_profile(num_blocks=16, block_size=4), EngineConfig(max_num_batched_tokens=64))
    seqs = simulate(e, [(0.0, prompt(20, s), 20) for s in range(5)])
    assert e.total_preemptions > 0
    assert all(len(s.output) == 20 and s.finish_reason == "length" for s in seqs)   # nobody loses output
    assert sum(s.preemptions for s in seqs) == e.total_preemptions


def test_request_validation():
    e = FakeEngine(tiny_profile(max_model_len=100))
    with pytest.raises(ValueError, match="maximum context length"):
        e.add_request(prompt(90), 20, 0.0)


def test_step_time_is_the_roofline():
    p = tiny_profile(num_blocks=256)
    e = FakeEngine(p, EngineConfig(max_num_batched_tokens=4096))
    s = e.add_request(prompt(8), 4, 0.0)
    plan = e.schedule(0.0)                                   # an 8-token prefill: memory-bound
    mem = (p.weight_bytes + 8 * p.kv_bytes_per_token) / p.mem_bw
    assert e.step_time(plan) == pytest.approx(p.overhead_s + mem)
    big = FakeEngine(p, EngineConfig(max_num_batched_tokens=4096))
    big.add_request(prompt(400), 4, 0.0)
    plan = big.schedule(0.0)                                 # a 400-token prefill: compute-bound
    assert big.step_time(plan) == pytest.approx(p.overhead_s + 2 * p.active_params * 400 / p.flops)
    assert s.status == "running"
    t4 = profile("t4-qwen2.5-0.5b")
    assert t4.decode_step_s(1, 512) < t4.decode_step_s(64, 512) < 64 * t4.decode_step_s(1, 512) / 8


def test_speculative_decoding_matches_the_formula():
    k, a = 4, 0.7
    e = FakeEngine(tiny_profile(num_blocks=2048, max_model_len=8192),
                   EngineConfig(num_speculative_tokens=k, spec_acceptance=a, seed=1))
    (s,) = simulate(e, [(0.0, prompt(8), 6000)])
    per_step = (len(s.output) - 1) / (len(s.emit_times) - 1)            # tokens per verify step
    assert per_step == pytest.approx((1 - a ** (k + 1)) / (1 - a), rel=0.03)


def test_prefix_counters_exclude_readmitted_preempted_requests():
    """vLLM records a preempted request's re-admission in preempted_* stats, not in
    vllm:prefix_cache_queries/hits (v1/core/kv_cache_manager.py, v1/metrics/stats.py)."""
    e = FakeEngine(tiny_profile(num_blocks=16, block_size=4), EngineConfig(max_num_batched_tokens=64))
    for s in range(5):
        e.add_request(prompt(20, s), 20, 0.0)
    t, queries, hits, pre_q = 0.0, 0, 0, 0
    while e.has_work():
        plan = e.schedule(t)
        queries, hits, pre_q = queries + plan.prefix_queries, hits + plan.prefix_hits, pre_q + plan.preempted_prefix_queries
        t += e.step_time(plan)
        e.commit(plan, t)
    assert e.total_preemptions > 0 and pre_q > 0
    assert queries == 5 * 20                     # each request's first admission, once
    assert hits == 0                             # distinct prompts: nothing to share on first admission
