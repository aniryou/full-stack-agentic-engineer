"""Token buckets, reserve -> stream -> reconcile, RPM and TPM together, and the per-request bucket's flaw."""
import pytest

from gwlab.gateway.config import Limits, Tenant
from gwlab.gateway.ratelimit import Bucket, Limiter, over_admission


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_bucket_refills_goes_into_debt_and_says_when_to_retry():
    c = Clock()
    b = Bucket(100, 10, c)                          # 100 burst, 10 per second
    assert b.can_take(100)
    b.take(100)
    assert not b.can_take(1) and b.retry_after(30) == pytest.approx(3.0)
    c.t = 3.0
    assert b.available() == pytest.approx(30)
    c.t = 20.0
    assert b.available() == 100                      # capped at the burst
    b.take(250)                                      # larger than the burst: admitted from full, into debt
    assert b.available() == -150 and b.retry_after(50) == pytest.approx(20.0)


def make(mode="reserve", tpm=6000, rpm=60, **kw):
    c = Clock()
    lim = Limiter(Limits(mode=mode, minute_s=60, default_output_estimate=500, hard_output_cap=1000,
                         per_request_output_guess=50, **kw), {"t": Tenant("t", ["chat"], rpm=rpm, tpm=tpm)}, c)
    return lim, c


def test_reserve_then_reconcile():
    lim, _ = make()
    d = lim.admit("t", 100, None)                    # no cap from the caller: reserve the default estimate
    assert d.ok and d.reservation.charged == 600 and lim.state()["t"]["tpm_available"] == 5400
    assert lim.reconcile(d.reservation, 100, 80) == 420           # refund what was not used
    assert lim.state()["t"]["tpm_available"] == pytest.approx(5820)
    d2 = lim.admit("t", 100, 4000)                   # a caller cap above the hard cap is clipped
    assert d2.reservation.reserved_output == 1000


def test_streaming_past_the_reservation_debits_and_the_hard_cap_cuts():
    lim, _ = make()
    res = lim.admit("t", 100, 200).reservation
    assert lim.on_output(res, 150) and lim.state()["t"]["tpm_available"] == pytest.approx(5700)
    assert lim.on_output(res, 350) and lim.state()["t"]["tpm_available"] == pytest.approx(5550)    # 150 more debited
    assert not lim.on_output(res, 1000)                                   # the hard cap
    lim.reconcile(res, 100, 1000)
    assert lim.state()["t"]["tpm_available"] == pytest.approx(4900)


def test_rpm_and_tpm_are_both_enforced_and_refusals_carry_retry_after():
    lim, c = make(rpm=2)
    assert lim.admit("t", 10, 10).ok and lim.admit("t", 10, 10).ok
    d = lim.admit("t", 10, 10)
    assert not d.ok and d.limit == "rpm" and d.headers["Retry-After"] == "30"
    c.t = 30.0
    assert lim.admit("t", 10, 10).ok
    lim2, _ = make(tpm=1000)
    assert lim2.admit("t", 400, 500).ok
    d = lim2.admit("t", 400, 500)
    assert not d.ok and d.limit == "tpm" and "x-ratelimit-remaining-tokens" in d.headers


def test_global_tpm_sits_above_the_tenants():
    c = Clock()
    lim = Limiter(Limits(global_tpm=1000, default_output_estimate=100),
                  {n: Tenant(n, ["chat"], rpm=100, tpm=10_000) for n in ("a", "b")}, c)
    assert lim.admit("a", 400, 100).ok and not lim.admit("b", 450, 100).ok     # 500 left globally; b wants 550
    assert lim.admit("b", 400, 100).ok and not lim.admit("b", 1, 1).ok           # b's own bucket has plenty
    assert lim.rejections[("b", "global_tpm")] == 2


def test_per_request_mode_charges_a_guess_and_never_reconciles():
    lim, _ = make(mode="per_request")
    res = lim.admit("t", 100, 2000).reservation
    assert res.charged == 150                                            # prompt + the fixed guess
    assert lim.on_output(res, 900) and lim.reconcile(res, 100, 900) == 0
    assert lim.state()["t"]["tpm_available"] == pytest.approx(5850)      # 850 tokens it never counted


def test_over_admission_by_hand():
    assert over_admission(100, 900, 100) == 5.0              # outputs 9x the guess: 5x the TPM gets through
    assert over_admission(1000, 100, 100) == 1.0
    lim, _ = make(mode="per_request", tpm=15_000, rpm=10_000)
    admitted = 0
    while lim.admit("t", 100, None).ok:                      # a full bucket admits TPM / (P + g) = 15000 / 150 ...
        admitted += 1
    assert admitted == 100                                   # ... requests that really cost P + L = 1000 each:
    assert admitted * (100 + 900) == pytest.approx(over_admission(100, 900, 50) * 15_000)   # 6.7x the TPM
