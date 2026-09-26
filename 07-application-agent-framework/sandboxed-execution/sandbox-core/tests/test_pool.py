"""Warm-pool sizing and cost: the closed forms (FACTS §12) and the simulator that checks them."""
import math

import pytest

from sandboxcore import pool


def test_littles_law_is_the_mean_occupancy_not_a_pool_size():
    # FACTS §12: λ=5/s, t_exec=2 s, t_cold=3 s -> 10 busy + 15 warming = 25 slots held ON AVERAGE
    assert pool.busy_sandboxes(5, 2) == 10
    assert pool.warming_sandboxes(5, 3) == 15
    assert pool.mean_occupancy(5, 2, 3) == 25
    # ...and 25 slots is the offered load itself: the queue never clears, every request eventually waits
    assert pool.erlang_c(25, 25) == 1.0
    assert pool.expected_wait_s(5, 5, 25) == math.inf


def test_replace_after_use_pool_is_sized_with_erlang_c_on_exec_plus_cold():
    # a = λ(t_exec + t_cold) = 25 Erlangs; the primer's §6 table
    table = {26: (0.782, 3.912), 28: (0.457, 0.762), 30: (0.250, 0.250), 31: (0.180, 0.150),
             33: (0.088, 0.055), 35: (0.040, 0.020)}
    for c, (pwait, ewq) in table.items():
        assert math.isclose(pool.erlang_c(25, c), pwait, abs_tol=0.001)
        assert math.isclose(pool.expected_wait_s(5, 5, c), ewq, abs_tol=0.001)
    assert pool.slots_for_wait_target(5, 2, 3, 0.2) == 31      # 30 gives 0.250
    assert pool.slots_for_wait_target(5, 2, 3, 0.05) == 35


def test_reuse_pool_erlang_c_table_matches_facts():
    # a reuse (session) pool holds a slot for t_exec only: a = λ·t_exec = 10 (FACTS §12, hand-verified)
    table = {11: (0.682, 1.364), 12: (0.449, 0.449), 13: (0.285, 0.190),
             14: (0.174, 0.087), 16: (0.057, 0.019)}
    for c, (pwait, ewq) in table.items():
        assert math.isclose(pool.erlang_c(10, c), pwait, abs_tol=0.001)
        assert math.isclose(pool.expected_wait_s(5, 2, c), ewq, abs_tol=0.001)
    assert pool.slots_for_wait_target(5, 2, 3, 0.2, replace_after_use=False) == 14   # 13 gives 0.285


def test_erlang_c_is_stable_at_large_offered_loads():
    # the iterative recurrence does not overflow where a**c / c! would (the scaling primer's peak rate
    # with 6 s holds: a = 162.6 Erlangs)
    assert pool.servers_for_wait_target(27.1, 6.0, 0.05) == 186
    assert 0.0 < pool.erlang_c(162.6, 170) < 1.0


def test_scaling_primer_rate_drives_the_pool():
    # scaling primer §3.2: 27.1 tool calls/s at peak; if a fifth are run_code, λ = 5.42/s
    lam = 27.1 * 0.2
    assert math.isclose(pool.mean_occupancy(lam, 2, 3), 27.1)
    assert pool.slots_for_wait_target(lam, 2, 3, 0.2) == 34


def test_cost_per_execution_charges_the_cold_start_unless_the_sandbox_is_reused():
    # 2 s exec + 3 s cold, node at $0.0001/s (a (verify) input)
    one_shot = pool.cost_per_execution(2, 3, 1e-4)
    reused = pool.cost_per_execution(2, 3, 1e-4, reuse=True)
    assert math.isclose(one_shot, 5e-4)          # a replace-after-use pool still pays each replacement's warm-up
    assert math.isclose(reused, 2e-4)            # only a reused (stateful) sandbox amortises it
    # fleet view: 31 slots at λ=5 cost 31/5 = 6.2 sandbox-seconds per execution (1.2 s of it idle headroom)
    assert math.isclose(pool.pool_cost_per_execution(5, 31, 1e-4), 6.2e-4)
    assert math.isclose(pool.pool_cost_per_execution(5, 31, 1e-4) - one_shot, (31 - 25) / 5 * 1e-4)
    # scaling primer §1: 1.3 tool calls/turn
    assert math.isclose(pool.actions_cost(1.3, one_shot), 6.5e-4)


def test_simulated_reuse_pool_tracks_mmc():
    # SIMULATED: a reuse pool with c=12 at a=10 sits near the M/M/c prediction P(wait)=0.449
    sim = pool.simulate(5, 2, 3, 12, n=30000, mode="reuse", seed=3)
    assert sim.cold_on_path == 0 and sim.rewarms == 0
    assert sim.max_busy <= 12
    assert 0.38 <= sim.frac_waited <= 0.52


def test_simulated_replace_after_use_needs_more_than_the_mean_occupancy():
    # SIMULATED: 25 slots (the Little's-law mean) is unstable; 31 (Erlang C, P(wait) 0.18) mostly serves warm
    at_mean = pool.simulate(5, 2, 3, 25, n=30000, mode="replace_after_use", seed=1)
    sized = pool.simulate(5, 2, 3, 31, n=30000, mode="replace_after_use", seed=1)
    assert at_mean.frac_waited > 0.8 and at_mean.mean_wait_s > 3
    assert sized.cold_on_path == 0 and sized.rewarms == 30000        # every cold start was off the path
    assert 0.08 <= sized.frac_waited <= pool.erlang_c(25, 31) + 0.02  # fixed cold start: at or below Erlang C


def test_primer_section_6_simulated_figures():
    # PRIMER §6 quotes these (SIMULATED, seed 1, n = 30,000 — notebook 05's worked example 4)
    at_mean = pool.simulate(5, 2, 3, 25, n=30000, mode="replace_after_use", seed=1)
    sized = pool.simulate(5, 2, 3, 31, n=30000, mode="replace_after_use", seed=1)
    cold = pool.simulate(5, 2, 3, 31, n=30000, mode="cold_on_demand", seed=1)
    assert round(at_mean.frac_waited, 2) == 0.91 and round(at_mean.mean_wait_s, 1) == 6.8
    assert round(sized.frac_waited, 2) == 0.14 and round(sized.mean_wait_s, 2) == 0.06
    assert cold.frac_waited == 1.0 and cold.mean_wait_s >= 3.0


def test_simulated_cold_on_demand_pays_the_cold_start_on_every_request():
    sim = pool.simulate(5, 2, 3, 31, n=20000, mode="cold_on_demand", seed=1)
    assert sim.cold_on_path == 20000 and sim.frac_waited == 1.0
    assert sim.mean_wait_s >= 3.0


def test_simulate_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        pool.simulate(5, 2, 3, 10, mode="warm")
