"""The timer and the α-β fit, driven by a fake clock and exact synthetic data."""
import pytest

from gpubench.timing import Timing, bench, fit_alpha_beta


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bench_amortises_calls_until_min_time():
    clock, calls = FakeClock(), []

    def fn():
        calls.append(1)
        clock.t += 0.001                      # every call "takes" 1 ms

    t = bench(fn, clock=clock, min_time=0.01, repeats=4, warmup=2)
    assert t.inner >= 10 and len(t.samples) == 4
    assert all(s == pytest.approx(0.001) for s in t.samples)
    assert len(calls) >= 2 + 4 * t.inner       # warm-up calls happened too


def test_setup_runs_outside_the_clock():
    clock = FakeClock()

    def setup():
        clock.t += 1.0                         # e.g. dropping the page cache: must not be timed

    def fn():
        clock.t += 0.002

    t = bench(fn, clock=clock, setup=setup, max_inner=1, repeats=3, min_time=0.0)
    assert t.inner == 1 and t.samples == pytest.approx((0.002, 0.002, 0.002))


def test_timing_statistics():
    t = Timing((3.0, 1.0, 2.0))
    assert (t.best, t.median, t.worst) == (1.0, 2.0, 3.0)
    assert Timing.from_dict(t.to_dict()) == t


def test_alpha_beta_recovers_exact_parameters():
    alpha, beta = 10e-6, 12e9                   # 10 µs fixed cost, 12 GB/s link
    sizes = [2 ** k for k in range(10, 30, 2)]
    fit = fit_alpha_beta(sizes, [alpha + s / beta for s in sizes])
    assert fit.alpha == pytest.approx(alpha, rel=1e-6) and fit.beta == pytest.approx(beta, rel=1e-6)
    assert fit.n_half == pytest.approx(120_000, rel=1e-6)          # α·β = 10e-6 × 12e9
    assert fit.bandwidth(fit.n_half) == pytest.approx(beta / 2)     # half the link at n½
    assert fit.r2 == pytest.approx(1.0)
