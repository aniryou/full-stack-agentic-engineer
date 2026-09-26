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


def test_transfer_series_keep_pipelined_and_latency_sweeps_apart():
    from gpubench import transfer
    from gpubench.accounting import transfer_cost
    from gpubench.measure import Measurement

    def m(op, n, pinned, mode, t):
        return Measurement(op, {"nbytes": n, "pinned": pinned, "mode": mode}, transfer_cost(n), Timing((t,)), "torch", "cuda:0")

    ms = [m("h2d", n, True, "pipelined", 2e-6 + n / 25e9) for n in (4096, 1 << 20, 64 << 20)]
    ms += [m("h2d", n, True, "latency", 9e-6 + n / 25e9) for n in (4096, 65536, 1 << 20)]
    groups = transfer.series(ms)
    assert set(groups) == {("h2d", True, "pipelined"), ("h2d", True, "latency")}
    fits = {transfer.series_label(k): transfer.fit(v) for k, v in groups.items()}
    assert fits["h2d pinned latency"].alpha == pytest.approx(9e-6, rel=1e-6)
    assert fits["h2d pinned pipelined"].alpha == pytest.approx(2e-6, rel=1e-6)
    assert transfer.series_label(("memcpy", None, None)) == "memcpy"


def test_host_link_prefers_nvidia_smi_then_the_spec_then_says_it_assumed(monkeypatch):
    from gpubench import inventory, transfer
    monkeypatch.setattr(inventory, "query_gpus", lambda: ([{"pcie.link.gen.max": 3, "pcie.link.width.current": 8,
                                                             "pcie.link.width.max": 16}], ""))
    assert transfer.host_link("Tesla T4") == {"gen": 3, "width": 8, "source": "nvidia-smi"}
    monkeypatch.setattr(inventory, "query_gpus", lambda: (None, "no nvidia-smi"))
    assert transfer.host_link("Tesla T4")["gen"] == 3                          # spec table: PCIe Gen3 x16
    assert transfer.host_link(None)["source"].startswith("assumed")
