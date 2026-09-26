"""Warm-pool sizing and cost: the closed-form numbers pinned to FACTS §12, plus a simulator sanity check."""
import math

from sandboxcore import pool


def test_little_and_warming_and_pool_size():
    # FACTS §12: λ=5/s, t_exec=2 s, t_cold=3 s -> 10 busy + 15 warming = 25
    assert pool.busy_sandboxes(5, 2) == 10
    assert pool.warming_sandboxes(5, 3) == 15
    assert pool.pool_size(5, 2, 3) == 25


def test_erlang_c_table_matches_facts():
    # offered load a = λ·t_exec = 10; hand-verified P(wait) and E[Wq] (FACTS §12)
    table = {11: (0.682, 1.364), 12: (0.449, 0.449), 13: (0.285, 0.190),
             14: (0.174, 0.087), 16: (0.057, 0.019)}
    for c, (pwait, ewq) in table.items():
        assert math.isclose(pool.erlang_c(10, c), pwait, abs_tol=0.001)
        assert math.isclose(pool.expected_wait_s(5, 2, c), ewq, abs_tol=0.001)


def test_erlang_c_is_one_when_undersized():
    assert pool.erlang_c(10, 10) == 1.0
    assert pool.expected_wait_s(5, 2, 10) == math.inf


def test_servers_for_wait_target():
    # smallest c with P(wait) <= 0.2 at a=10 is 14 (0.174), since 13 is 0.285
    assert pool.servers_for_wait_target(5, 2, 0.2) == 14


def test_scaling_primer_link():
    # §3.2: 27.1 tool calls/s at peak; a fifth are run_code -> ~5.4/s
    assert round(27.1 * 0.2, 2) == 5.42


def test_cost_per_execution_and_actions():
    # 2 s exec + 3 s cold, node at $0.0001/s, warm pool absorbs the cold start (warm_share=0)
    warm = pool.cost_per_execution(2, 3, 1e-4, warm_share=0.0)
    payg = pool.cost_per_execution(2, 3, 1e-4, warm_share=1.0)
    assert math.isclose(warm, 2 * 1e-4)
    assert math.isclose(payg, 5 * 1e-4)          # pay-per-use also pays the cold start
    assert math.isclose(pool.actions_cost(1.3, warm), 1.3 * warm)   # scaling primer: 1.3 tool calls/turn


def test_simulator_tracks_the_closed_form_for_a_warm_pool():
    # SIMULATED: a warm pool with c=12 at a=10 should sit near the M/M/c prediction
    sim = pool.simulate(5, 2, 3, 12, n=30000, warm_pool=True, seed=3)
    assert sim.cold_starts == 0                  # warm pool: no cold start on the request path
    assert sim.max_busy <= 12
    assert 0.3 <= sim.frac_waited <= 0.6         # near P(wait)=0.449
