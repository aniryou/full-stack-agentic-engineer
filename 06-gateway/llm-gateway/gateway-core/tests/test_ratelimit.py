"""§4: per-request charges vs reserve -> stream -> reconcile."""
import random

import pytest

from gwcore import ratelimit as RL


def test_token_bucket_rule_hand_computed():
    b = RL.TokenBucket(rate=10, capacity=60, now=0)
    assert b.acquire(50, 0) == 0 and b.tokens == 10
    assert b.acquire(30, 1) == pytest.approx(1.0)          # 10 + 10 refilled = 20; the 10 missing take 1 s
    assert b.acquire(100, 2) == pytest.approx(6.0)         # larger than the burst: wait for a full bucket (60) ...
    assert b.tokens == pytest.approx(-40)                  # ... then go into debt
    assert not b.try_acquire(1, 8) and b.try_acquire(1, 12.5)


def test_window_meter_slides():
    m = RL.WindowMeter(60)
    m.add(0, 100)
    m.add(30, 50)
    assert m.used(59) == 150 and m.used(60) == 50 and m.used(90) == 0


def test_reserve_debit_reconcile():
    lim = RL.ReserveLimiter(10_000)
    assert lim.admit("a", 6_000, 0) and not lim.admit("b", 5_000, 0)       # 6,000 held: 4,000 headroom
    lim.debit("a", 1_000, 1)                                                # streamed tokens move reserved -> used
    assert lim.meter.used(1) == 1_000 and lim.outstanding == 5_000 and lim.headroom(1) == 4_000
    assert lim.finish("a") == 5_000 and lim.headroom(1) == 9_000           # reconcile: the unused 5,000 come back
    assert lim.retry_after(1) == 60


def test_reserve_invariant_bounds_what_the_provider_sees():
    """used-in-window + outstanding never exceeds the limit when reservations are upper bounds, so a provider
    counting the same tokens in the same window never sees more than the limit -- checked on random traffic."""
    rng = random.Random(0)
    for trial in range(20):
        lim, live, t = RL.ReserveLimiter(50_000), {}, 0.0
        for rid in range(300):
            t += rng.expovariate(2.0)
            for r in list(live):                               # stream some tokens of every live request
                n = min(live[r], rng.randint(0, 400))
                live[r] -= n
                lim.debit(r, n, t)
                if not live[r]:
                    lim.finish(r)
                    del live[r]
            cap = rng.choice([1_000, 4_000, 8_000])
            if lim.admit(rid, cap, t):
                live[rid] = rng.randint(1, cap)                # actual <= reservation
            assert lim.meter.used(t) + lim.outstanding <= 50_000 + 1e-9
            assert lim.overrun == 0


def test_admit_all_is_all_or_nothing():
    tenant, provider = RL.ReserveLimiter(10_000), RL.ReserveLimiter(5_000)
    assert not RL.admit_all([tenant, provider], "r", [6_000, 6_000], 0)
    assert tenant.outstanding == 0                              # nothing half-committed
    assert RL.admit_all([tenant, provider], "r", [4_000, 4_000], 0) and tenant.outstanding == provider.outstanding == 4_000


def test_overadmission_closed_form():
    old, new = RL.lognormal_mean(300, 1.0), RL.lognormal_mean(1_500, 1.0)
    assert round(old, 1) == 494.6 and round(new) == 2473           # the RL primer's §7 lognormal (median 1,500, sigma 1)
    assert round(RL.overadmission_ratio(1_500, old, new), 2) == 1.99


def test_compare_buckets_pinned():
    res = RL.compare_buckets()
    old, new = res["per-request, old outputs"], res["per-request, thinking outputs"]
    est, cap = res["reserve prompt + 4,096"], res["reserve prompt + cap 16,384"]
    assert old["admitted"] == new["admitted"] == 5055                 # the per-request bucket cannot see the change
    assert round(old["peak_ratio"], 2) == 1.03 and old["seconds_over_limit"] == 116
    assert round(new["peak_ratio"], 2) == 1.97 and new["seconds_over_limit"] == 604 and round(new["utilisation"], 2) == 1.87
    assert est["seconds_over_limit"] == 0 and round(est["peak_ratio"], 2) == 0.77 and round(est["utilisation"], 3) == 0.704
    assert est["admitted"] == 1920 and est["overrun_tokens"] == 910_498
    assert cap["seconds_over_limit"] == 0 and round(cap["utilisation"], 3) == 0.265 and cap["truncated"] == 5
