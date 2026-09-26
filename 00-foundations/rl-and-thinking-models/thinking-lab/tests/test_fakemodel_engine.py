"""The simulated model's behaviour and the engine emulator's arithmetic, pinned to hand-computed values."""
import math

import pytest

from thinklab import engine as E
from thinklab import fakemodel as F
from thinklab.thinking.evalset import make_evalset


def test_accuracy_curve_shape():
    c = F.SMALL
    assert c.accuracy(3, 0, thinking=False) == c.a0[2]
    assert c.accuracy(3, 0) == pytest.approx(c.a0[2])
    assert c.accuracy(3, 480) == pytest.approx(c.a_max[2] - (c.a_max[2] - c.a0[2]) * math.exp(-1))
    accs = [c.expected_accuracy(3, b, samples=2000) for b in (64, 256, 1024, None)]
    assert accs == sorted(accs) and accs[-1] < c.a_max[2]


def test_generate_modes_and_truncation():
    q = make_evalset(20, seed=0)[15].prompt                      # a difficulty-4 arithmetic question
    s = F.generate(F.SMALL, q, thinking=True, seed=1)
    assert s.planned_think == len(s.reasoning_tokens) > 0 and s.finish_reason == "stop"
    off = F.generate(F.SMALL, q, thinking=False, seed=1)
    assert off.reasoning_tokens == [] and off.content_tokens
    b = F.generate(F.SMALL, q, thinking=True, budget=16, seed=1)
    assert len(b.reasoning_tokens) <= 16 and b.forced_stop == (b.planned_think > 16) and b.answer is not None
    cut = F.generate(F.SMALL, q, thinking=True, max_tokens=5, seed=1)
    assert cut.finish_reason == "length" and cut.content_tokens == [] and cut.correct is False
    assert F.generate(F.SMALL, q, seed=1).reasoning_tokens == s.reasoning_tokens          # deterministic


def test_thinking_helps_on_average():
    qs = [p.prompt for p in make_evalset(200, seed=5)]
    on = sum(F.generate(F.SMALL, q, seed=0, sample_index=i).correct for i, q in enumerate(qs))
    off = sum(F.generate(F.SMALL, q, thinking=False, seed=0, sample_index=i).correct for i, q in enumerate(qs))
    assert on > off + 40


def test_step_time_formula_by_hand():
    p = E.profile("t4-qwen3-0.6b")
    assert p.weight_bytes == 596_049_920 * 2 and p.kv_capacity_tokens == 6_969 * 16
    # batch 1, 512 tokens of context: memory-bound
    mem = (1_192_099_840 + 512 * 114_688) / (320e9 * 0.8)
    assert p.decode_step_s(1, 512) == pytest.approx(0.004 + mem)
    # a large prefill is compute-bound: 2 x params x tokens / (65 TFLOPS x 0.5)
    assert p.step_s(4096, 0, 4096) == pytest.approx(0.004 + 2 * 596_049_920 * 4096 / 32.5e12)


def test_simulate_conserves_tokens_and_preempts_when_the_pool_is_small():
    prof = E.profile("tiny", num_blocks=40)                      # 160 KV token slots
    reqs = [(0.0, 10, 50, 40), (0.0, 10, 60, 50), (0.0, 10, 40, 0), (0.001, 10, 30, 0)]
    eng = E.simulate(prof, reqs)
    assert len(eng.finished) == 4 and all(r.generated == r.output_tokens for r in eng.finished)
    assert eng.preemptions > 0 and eng.free == prof.num_blocks
    first = {r.rid: r for r in eng.finished}
    assert first[0].first_content > first[0].first_token          # 40 reasoning tokens before the answer
    assert all(t >= 0 for r in eng.finished for t in r.itls)


def test_max_model_len_is_enforced():
    eng = E.Engine(E.profile("tiny", max_model_len=100))
    with pytest.raises(ValueError):
        eng.add(60, 50, 0.0)


def test_itl_grows_with_context_not_just_batch():
    p = E.profile("t4-qwen3-0.6b")
    assert p.decode_step_s(64, 64 * 3000) > 3 * p.decode_step_s(64, 64 * 200)


def test_long_recompute_after_preemption_is_readmitted_and_abort_frees_blocks():
    prof = E.profile("tiny", num_blocks=600)                     # 2,400 slots; budget 2,048 tokens per step
    eng = E.simulate(prof, [(0.0, 10, 2300, 2000), (0.0, 10, 2300, 0)])
    assert len(eng.finished) == 2 and eng.preemptions >= 1     # the preempted one re-prefills > 2,048 tokens alone
    e2 = E.Engine(prof)
    r = e2.add(10, 50, 0.0)
    e2.commit(e2.schedule(0.0), 0.01)
    e2.abort(r, 0.02)
    assert e2.free == prof.num_blocks and not e2.has_work() and r.state == "aborted"
