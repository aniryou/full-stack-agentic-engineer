"""Cold start, failure arithmetic and the cost of a token."""
import math

import pytest

from roofline import cost, llm, reliability, storage

M70 = llm.PRESETS["llama-3.1-70b"]


# -- storage -------------------------------------------------------------------------
def test_checkpoint_bytes_and_tier_times():
    b = storage.checkpoint_bytes(M70.params(), 2)
    assert b / 1e9 == pytest.approx(141.1, abs=0.05)
    assert storage.read_time(b, 0.1) / 60 == pytest.approx(23.5, abs=0.05)    # one stream
    assert storage.read_time(16e9, 20) == pytest.approx(0.8)


def test_parallel_streams_hit_a_cap():
    assert storage.parallel_gbs(16, 0.1, 12.5) == pytest.approx(1.6)
    assert storage.parallel_gbs(128, 0.1, 12.5) == 12.5                     # NIC-bound


def test_streaming_approaches_the_slower_hop():
    n = 141.1e9
    seq = storage.load_time(n, 12.5, 50, gpus=8)
    streamed = storage.load_time(n, 12.5, 50, gpus=8, streamed=True)
    assert seq == pytest.approx(n / 12.5e9 + n / 400e9)
    assert streamed == pytest.approx(n / 12.5e9 + storage.GIB * 0.25 / 400e9)
    assert streamed < seq


def test_cold_start_breakdown_sums_to_total():
    cs = storage.cold_start(141.1e9, fetch_gbs=0.1, h2d_gbs=50, gpus=8,
                            provision_s=120, image_s=60, init_s=90)
    assert cs.total == pytest.approx(120 + 60 + 1411 + 141.1e9 / 400e9 + 90)
    assert "fetch weights" in cs.table()


# -- reliability ---------------------------------------------------------------------
def test_llama3_interruption_rate():
    # Llama 3 paper: 419 unexpected interruptions in 54 days on 16,384 H100s
    m = reliability.component_mtbf_from_observation(419, 54 * 24, 16384)
    assert m == pytest.approx(16384 * 1296 / 419)                           # 50,677 GPU-hours
    assert m / reliability.HOURS_PER_YEAR == pytest.approx(5.79, abs=0.01)
    assert reliability.cluster_mtbf(m, 16384) == pytest.approx(3.09, abs=0.01)
    assert reliability.failures_per_day(16384, m) == pytest.approx(7.76, abs=0.01)
    assert reliability.p_survive(24, 1024, m) == pytest.approx(math.exp(-1024 * 24 / m))   # 0.616


def test_young_daly_interval():
    assert reliability.young_daly_interval(60, 11_135) == pytest.approx(math.sqrt(2 * 60 * 11_135))   # 1156 s
    daly = reliability.young_daly_interval(60, 11_135, higher_order=True)
    r = 60 / (2 * 11_135)
    assert daly == pytest.approx(math.sqrt(2 * 60 * 11_135) * (1 + math.sqrt(r) / 3 + r / 9) - 60)
    assert reliability.young_daly_interval(100, 10, higher_order=True) == 10          # delta >= 2M


def test_waste_is_minimised_at_the_young_interval():
    d, m = 60, 11_135
    tau = reliability.young_daly_interval(d, m)
    w = reliability.wasted_fraction(tau, d, m)
    assert w == pytest.approx(math.sqrt(2 * d / m))                                   # 10.4%
    assert w < reliability.wasted_fraction(tau * 0.5, d, m)
    assert w < reliability.wasted_fraction(tau * 2, d, m)
    assert reliability.wasted_fraction(3600, d, m) == pytest.approx(60 / 3600 + 1800 / m)   # 17.8%


def test_replicas_are_failure_domains():
    m = 16384 * 1296 / 419
    a8 = reliability.replica_availability(m, 8, 48)
    assert a8 == pytest.approx((m / 8) / (m / 8 + 48))                                # 0.99248
    assert reliability.p_at_least(8, 8, a8) == pytest.approx(a8 ** 8)                 # 94.1%
    assert reliability.p_at_least(8, 9, a8) == pytest.approx(a8 ** 9 + 9 * a8 ** 8 * (1 - a8))
    assert reliability.replicas_for(8, a8, 0.999) == 10                               # 16 spare GPUs
    assert reliability.replicas_for(64, reliability.replica_availability(m, 1, 48), 0.999) == 66


# -- cost ------------------------------------------------------------------------------
def test_cost_per_million_tokens():
    assert cost.cost_per_million_tokens(2.0, 1000) == pytest.approx(2 / 3.6)          # $0.556
    assert cost.cost_per_million_tokens(2.0, 1000, utilisation=0.5) == pytest.approx(2 * 2 / 3.6)
    assert cost.cost_per_million_tokens(11.0, 6659) == pytest.approx(0.459, abs=0.001)


def test_utilisation_of_a_peak_sized_fleet():
    assert cost.utilisation([0.2] * 8 + [1.0] * 8 + [0.6] * 8) == pytest.approx(0.6)


def test_rent_versus_own_breakeven():
    fixed, energy = cost.owned_cost_per_hour(300_000, 4, 10.2, pue=1.3, usd_per_kwh=0.10,
                                             opex_per_year=30_000)
    assert fixed == pytest.approx(300_000 / 35_040 + 30_000 / 8760)                  # $11.99/hr
    assert energy == pytest.approx(10.2 * 1.3 * 0.10)                                 # $1.33/hr
    assert cost.breakeven_utilisation(11.0, fixed / 8, energy / 8) == pytest.approx(0.138, abs=0.001)
    assert cost.breakeven_utilisation(3.7, fixed / 8, energy / 8) == pytest.approx(0.424, abs=0.001)
    assert cost.breakeven_utilisation(0.1, 1.0, 0.2) == math.inf
