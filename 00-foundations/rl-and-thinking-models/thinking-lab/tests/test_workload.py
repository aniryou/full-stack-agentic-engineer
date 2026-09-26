"""Length statistics, the serving shape, the capacity primer's arithmetic (by hand), and the modes in virtual time."""
import math

import pytest

from thinklab import engine as E
from thinklab import workload as W
from thinklab.thinking.evalset import make_evalset


def test_kv_token_steps_and_quadratic_growth():
    assert W.kv_token_steps(0, 3) == 6 and W.kv_token_steps(1500, 300) == 495_150
    assert W.kv_token_steps(1500, 3000) / W.kv_token_steps(1500, 300) == pytest.approx(18.18, abs=0.01)


def test_quantiles_and_lognormal_fit():
    st = W.length_stats([1, 2, 3, 4, 5])
    assert (st.p50, st.mean, st.max) == (3, 3, 5) and st.p90 == pytest.approx(4.6)
    import random
    rng = random.Random(0)
    med, sig = W.fit_lognormal([rng.lognormvariate(math.log(500), 0.8) for _ in range(20000)])
    assert med == pytest.approx(500, rel=0.03) and sig == pytest.approx(0.8, rel=0.03)


def test_capacity_primer_numbers_by_hand():
    """The capacity primer's worked example (fact sheet §11): 8.33 RPS, 1,500 in / 300 out, TPOT 40 ms, fp8."""
    rps = 10_000 * 0.10 * 0.5 / 60
    base, think = W.capacity_primer_view(rps, 1500, 300), W.capacity_primer_view(rps, 1500, 3000)
    assert base["ttft_s"] == pytest.approx(0.0728, abs=1e-4) and base["duration_s"] == pytest.approx(12.07, abs=0.01)
    assert base["concurrency"] == pytest.approx(100.6, abs=0.1) and base["sessions_per_gpu"] == pytest.approx(355.1, abs=0.1)
    assert base["gpus_for_memory"] == pytest.approx(0.283, abs=0.001)
    assert think["duration_s"] == pytest.approx(120.07, abs=0.01) and think["concurrency"] == pytest.approx(1000.6, abs=0.1)
    assert think["kv_per_session_gb"] == pytest.approx(0.2458, abs=1e-4) and think["sessions_per_gpu"] == pytest.approx(195.3, abs=0.1)
    assert think["gpus_for_memory"] == pytest.approx(5.12, abs=0.01) and think["decode_tok_s_needed"] == 25_000


def test_derive_shape_is_consistent_with_the_engine_profile():
    p = E.profile("t4-qwen3-0.6b")
    s = W.derive_shape(p, 60, [3000] * 10)
    ctx = 60 + 1500
    assert s.batch_by_memory == p.kv_capacity_tokens // ctx and s.bound == "memory (KV)"
    assert s.itl_ms == pytest.approx(1e3 * p.decode_step_s(s.batch_by_memory, s.batch_by_memory * ctx))
    short = W.derive_shape(p, 60, [100] * 10)
    assert short.bound == "max_num_seqs" and short.itl_ms < s.itl_ms


def test_modes_in_virtual_time():
    qs = [x.prompt for x in make_evalset(120, seed=2)]
    rows = {r["mode"]: r for r in W.simulate_modes(E.profile("t4-qwen3-0.6b"), qs,
            {"off": {"thinking": False}, "on": {}, "b512": {"budget": 512}}, rate=2.0)}
    assert rows["on"]["out mean"] > 5 * rows["off"]["out mean"]
    assert rows["on"]["ITL p50 ms"] > rows["b512"]["ITL p50 ms"] >= rows["off"]["ITL p50 ms"]
    assert rows["on"]["accuracy"] > rows["b512"]["accuracy"] > rows["off"]["accuracy"]
    assert rows["on"]["KV peak"] > rows["b512"]["KV peak"]
