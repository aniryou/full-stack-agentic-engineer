"""Continuous batching: budget, admission order, chunking, preemption - and no leaked blocks."""
import random

import pytest

from minengine import Engine, SamplingParams, TinyLM, encode
from minengine.kv import KVCacheManager
from minengine.scheduler import Request, Scheduler, SchedulerConfig


def _drive(sched, reqs, seed=0):
    """Run requests to completion with fake sampling, checking every invariant after every step."""
    rng = random.Random(seed)
    for r in reqs:
        sched.add_request(r)
    trace, B = [], sched.kv.block_size
    while sched.has_unfinished():
        published = set(sched.kv.cached)
        out = sched.schedule()
        filled = {b for r, n in out.scheduled for b in sched.kv.tables[r.request_id][:(r.num_computed_tokens + n) // B]}
        assert all(b in filled for name, b in sched.kv.cached.items() if name not in published)   # only what this step computes
        assert out.num_batched_tokens <= sched.cfg.max_num_batched_tokens
        assert len(sched.running) <= sched.cfg.max_num_seqs
        assert len({id(r) for r, _ in out.scheduled}) == len(out.scheduled)
        trace.append([(r.request_id, n) for r, n in out.scheduled])
        sampled = {r.request_id: rng.randrange(256) for r, n in out.scheduled
                   if r.num_computed_tokens + n == r.num_tokens}
        sched.update(out, sampled)
        sched.kv.check()
        assert len(trace) < 5000
    return trace


def _random_requests(n, seed=0, shared=12):
    rng = random.Random(seed)
    base = [rng.randrange(50) for _ in range(shared)]
    return [Request(f"r{i}", (base if rng.random() < 0.5 else []) + [rng.randrange(50) for _ in range(rng.randrange(1, 30))],
                    SamplingParams(max_tokens=rng.randrange(1, 20))) for i in range(n)]


@pytest.mark.parametrize("blocks,budget,chunked", [(40, 16, True), (16, 8, True), (60, 64, False), (16, 32, True)])
def test_budget_seq_limit_and_block_accounting_hold_every_step(blocks, budget, chunked):
    kv = KVCacheManager(blocks, block_size=4)
    sched = Scheduler(SchedulerConfig(budget, 4, chunked, max_model_len=64), kv)
    reqs = _random_requests(25, seed=blocks)
    _drive(sched, reqs)
    assert all(r.status.finished for r in reqs)
    assert all(len(r.output_token_ids) == r.params.max_tokens for r in reqs)   # 64-token cap never binds here
    assert kv.num_free_blocks == blocks and not kv.tables          # nothing leaked


def test_chunked_prefill_splits_a_long_prompt_around_decodes():
    kv = KVCacheManager(64, block_size=4)
    sched = Scheduler(SchedulerConfig(16, 4, max_model_len=64), kv)
    trace = _drive(sched, [Request("a", [1] * 3, SamplingParams(max_tokens=8)),
                           Request("b", [2] * 40, SamplingParams(max_tokens=2))])
    assert trace[0] == [("a", 3), ("b", 13)]                        # budget 16: a's prompt, b's first chunk
    assert trace[1] == [("a", 1), ("b", 15)] and trace[2] == [("a", 1), ("b", 12)]


def test_without_chunking_a_prompt_runs_whole_and_budget_must_cover_max_model_len():
    with pytest.raises(ValueError):
        SchedulerConfig(max_num_batched_tokens=32, enable_chunked_prefill=False, max_model_len=64)
    sched = Scheduler(SchedulerConfig(64, 4, False, max_model_len=64), KVCacheManager(64, 4))
    trace = _drive(sched, [Request("a", [1] * 5, SamplingParams(max_tokens=3)),
                           Request("b", [2] * 60, SamplingParams(max_tokens=1))])
    assert trace[0] == [("a", 5)] and trace[1] == [("a", 1), ("b", 60)]   # b waits for a whole-prompt slot


def test_fcfs_admission_and_newest_is_preempted():
    kv = KVCacheManager(6, block_size=4)
    sched = Scheduler(SchedulerConfig(32, 4, max_model_len=24, admit_whole_prompt=False), kv)
    reqs = [Request(f"r{i}", [i] * 7, SamplingParams(max_tokens=9)) for i in range(3)]
    for r in reqs:
        sched.add_request(r)
    first = sched.schedule()
    assert [r.request_id for r, _ in first.scheduled] == ["r0", "r1", "r2"]
    sched.update(first, {r.request_id: 0 for r, _ in first.scheduled})
    seen = []
    for _ in range(4):                                              # blocks run out as sequences grow
        out = sched.schedule()
        seen += [r.request_id for r in out.preempted]
        sched.update(out, {r.request_id: 0 for r, n in out.scheduled if r.num_computed_tokens + n == r.num_tokens})
    assert seen and seen[0] == "r2"                                 # the newest pays first
    assert sched.waiting[0].request_id == "r2" and reqs[2].num_computed_tokens == 0


def test_preemption_by_recompute_does_not_change_outputs():
    model = TinyLM()
    prompts = ["The engine runs a loop. Each step", "When memory runs out, the", "A request that finishes"]
    ref = [model.generate_dense(encode(p), 14) for p in prompts]
    eng = Engine(model, num_blocks=16, block_size=4, max_num_batched_tokens=16, max_num_seqs=3, admit_whole_prompt=False)
    outs = eng.generate(prompts, SamplingParams(max_tokens=14, temperature=0))
    assert eng.scheduler.num_preemptions > 0
    assert [o.token_ids for o in outs] == ref


def test_a_victim_already_in_this_batch_gives_its_tokens_back():
    class ByPriority(Scheduler):              # vLLM's priority policy: evict the least important, newest last
        def pick_victim(self):
            return max(self.running, key=lambda r: (r.priority, r.arrival_time))
    kv, rng = KVCacheManager(10, block_size=4), random.Random(3)
    sched = ByPriority(SchedulerConfig(16, 4, max_model_len=40, admit_whole_prompt=False), kv)
    reqs = _random_requests(16, seed=7, shared=0)
    for i, r in enumerate(reqs):
        r.priority, r.arrival_time = rng.randrange(3), i
        sched.add_request(r)
    in_batch_victims = 0
    while sched.has_unfinished():
        before = list(sched.running)
        out = sched.schedule()
        assert out.num_batched_tokens <= 16 and not ({id(v) for v in out.preempted} & {id(r) for r, _ in out.scheduled})
        in_batch_victims += sum(any(before.index(v) < before.index(r) for r, _ in out.scheduled if r in before)
                                for v in out.preempted if v in before)
        sched.update(out, {r.request_id: 0 for r, n in out.scheduled if r.num_computed_tokens + n == r.num_tokens})
        kv.check()
    assert in_batch_victims > 0 and all(r.status.finished for r in reqs) and kv.num_free_blocks == 10


def test_limits_are_enforced_up_front():
    sched = Scheduler(SchedulerConfig(max_model_len=16), KVCacheManager(8, 4))
    with pytest.raises(ValueError):
        sched.add_request(Request("big", [1] * 16))            # a prompt must leave room for one token
    with pytest.raises(ValueError):
        Scheduler(SchedulerConfig(max_model_len=64), KVCacheManager(8, 4))   # 64 tokens > a 32-token cache


def test_preemption_does_not_change_seeded_sampling_either():
    model = TinyLM()
    prompts = ["The engine runs a loop. Each step", "When memory runs out, the", "A request that finishes"]
    params = [SamplingParams(max_tokens=14, temperature=0.9, top_p=0.95, seed=s) for s in range(3)]
    roomy = Engine(model, num_blocks=64, block_size=4).generate(prompts, params)
    tight = Engine(model, num_blocks=16, block_size=4, max_num_batched_tokens=16, admit_whole_prompt=False)
    outs = tight.generate(prompts, params)
    assert tight.scheduler.num_preemptions > 0
    assert [o.token_ids for o in outs] == [o.token_ids for o in roomy]   # one draw per sampled token, same logits


def test_whole_prompt_check_and_watermark_reduce_preemptions():
    """Without the whole-prompt check, chunked prefill admits prompts memory cannot finish and preempts
    them a few steps later; a watermark keeps headroom for running requests to grow into."""
    def run(whole, watermark):
        rng = random.Random(1)
        reqs = [Request(f"r{i}", [rng.randrange(50) for _ in range(rng.randrange(20, 40))],
                        SamplingParams(max_tokens=rng.randrange(5, 25))) for i in range(30)]
        sched = Scheduler(SchedulerConfig(8, 8, max_model_len=64, admit_whole_prompt=whole, watermark_blocks=watermark),
                          KVCacheManager(24, block_size=4))
        _drive(sched, reqs)
        assert all(r.status.finished for r in reqs) and sched.kv.num_free_blocks == 24
        return sched.num_preemptions
    no_check, check, check_and_watermark = run(False, 0), run(True, 0), run(True, 4)
    assert check_and_watermark < check < no_check / 2                   # 0 < 8 < 20 on this workload
