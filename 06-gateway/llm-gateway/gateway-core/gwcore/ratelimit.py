"""Streaming-aware rate limits (PRIMER §4).

The one idea: a request's token cost is unknown at admission and heavy-tailed -- most of it is output
that has not been generated yet -- so a bucket that charges per request (or a fixed estimate) admits
whatever the outputs turn out to be, and over-runs a provider's tokens per minute the day outputs grow.
Instead **reserve** an upper bound at admission (prompt + the output cap), **debit** real tokens as they
stream, and **reconcile** (release the unused reservation) at the end. Admission then checks
``used in window + still reserved + this reservation <= limit``, which bounds what the provider sees.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class TokenBucket:
    """``scalelab.resilience.TokenBucket``'s rule on an explicit clock: refill ``rate``/s up to ``capacity``;
    ``acquire`` returns the wait before the tokens are taken; a request larger than the burst waits for a
    full bucket and goes into debt. ``try_acquire`` is the gateway's version: no wait, admit or refuse."""

    def __init__(self, rate: float, capacity: float, now: float = 0.0):
        self.rate, self.capacity, self.tokens, self.last = rate, capacity, capacity, now

    def _refill(self, now: float) -> None:
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    def wait_for(self, tokens: float, now: float) -> float:
        self._refill(now)
        return max(0.0, (min(tokens, self.capacity) - self.tokens) / self.rate)

    def acquire(self, tokens: float, now: float) -> float:
        wait = self.wait_for(tokens, now)
        if wait:
            self._refill(now + wait)
        self.tokens -= tokens
        return wait

    def try_acquire(self, tokens: float, now: float) -> bool:
        if self.wait_for(tokens, now) > 0:
            return False
        self.tokens -= tokens
        return True


class WindowMeter:
    """Tokens counted in a sliding window -- how a provider's tokens-per-minute sees you (modelled). Each event is a
    mutable ``[t, n]`` so a refund can shrink the events it corrects instead of adding a negative one."""

    def __init__(self, window: float = 60.0):
        self.window, self.events, self.total = window, deque(), 0.0

    def add(self, t: float, n: float) -> list:
        ev = [t, n]
        self.events.append(ev)
        self.total += n
        return ev

    def used(self, now: float) -> float:
        while self.events and self.events[0][0] <= now - self.window:
            self.total -= self.events.popleft()[1]
        return self.total

    def shrink(self, ev: list, n: float, now: float) -> float:
        """Take up to ``n`` off one event still in the window (never below zero); returns what was taken."""
        if ev[0] <= now - self.window:
            return 0.0                                    # already expired: it no longer counts, nothing to correct
        take = min(n, ev[1])
        ev[1] -= take
        self.total -= take
        return take


class ReserveLimiter:
    """Reserve -> stream -> reconcile against one limit (tokens per window, or requests with reserve 1)."""

    def __init__(self, limit: float, window: float = 60.0):
        self.limit, self.meter, self.held, self.outstanding, self.overrun = limit, WindowMeter(window), {}, 0.0, 0.0
        self.debits: dict = {}                            # rid -> its meter events, oldest first (for refunds)

    def headroom(self, now: float) -> float:
        return self.limit - self.meter.used(now) - self.outstanding

    def admit(self, rid, reserve: float, now: float) -> bool:
        if reserve > self.headroom(now):
            return False
        self.held[rid] = reserve
        self.outstanding += reserve
        return True

    def debit(self, rid, n: float, now: float) -> None:
        """Tokens actually processed: move them from reserved to used. Beyond the reservation is overrun."""
        self.debits.setdefault(rid, []).append(self.meter.add(now, n))
        take = min(n, self.held.get(rid, 0.0))
        self.held[rid] = self.held.get(rid, 0.0) - take
        self.outstanding -= take
        self.overrun += n - take

    def refund(self, rid, n: float, now: float) -> float:
        """This request's streamed estimate over-counted by ``n`` (usage says so): shrink its own debits, earliest
        first, never below zero. A negative event at ``now`` would outlive the debits it corrects and let the window
        read less than was processed; this way ``used`` never drops below the provider-true count."""
        left = n
        for ev in self.debits.get(rid, []):
            if left <= 0:
                break
            left -= self.meter.shrink(ev, left, now)
        return n - left

    def finish(self, rid) -> float:
        """Reconcile: release what was reserved and not used. Returns the refund."""
        self.debits.pop(rid, None)
        left = self.held.pop(rid, 0.0)
        self.outstanding -= left
        return left

    def retry_after(self, now: float) -> int:
        self.meter.used(now)
        return max(1, math.ceil(self.meter.events[0][0] + self.meter.window - now)) if self.meter.events else 1


def admit_all(limiters: list, rid, reserves: list, now: float) -> bool:
    """Hierarchical limits (key < tenant < org < provider key): check every level, then commit every level --
    one atomic step (a single Lua script when the state is in Redis)."""
    if any(r > lim.headroom(now) for lim, r in zip(limiters, reserves)):
        return False
    return all(lim.admit(rid, r, now) for lim, r in zip(limiters, reserves))


def lognormal_mean(median: float, sigma: float) -> float:
    return median * math.exp(sigma ** 2 / 2)


def overadmission_ratio(prompt: float, est_output: float, true_mean_output: float) -> float:
    """A per-request charge sized for ``est_output`` lets through (prompt + true mean) / (prompt + estimate)
    times the tokens it was sized for, once the bucket binds."""
    return (prompt + true_mean_output) / (prompt + est_output)


def workload(seconds: float, rate: float, prompt: int, median_out: float, sigma: float, seed: int = 0) -> list:
    """Poisson arrivals with lognormal output lengths: [(arrival s, prompt tokens, output tokens)]."""
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * seconds)
    t = np.sort(rng.uniform(0, seconds, n))
    out = np.maximum(1, np.round(median_out * np.exp(sigma * rng.standard_normal(n)))).astype(int)
    return [(float(a), prompt, int(o)) for a, o in zip(t, out)]


def simulate(arrivals: list, kind: str, *, limit: float, est: float = 0.0, burst_s: float = 6.0,
             reserve_out: float = 0.0, cap: int | None = None, tok_per_s: float = 50.0) -> dict:
    """Admit on a 1 s grid; admitted streams process the prompt at once and ``tok_per_s`` output tokens per
    second. ``kind="per_request"``: a ``TokenBucket`` (rate limit/60, a ``burst_s`` burst) charging ``est``
    per request, never reconciled. ``kind="reserve"``: reserve prompt + ``reserve_out``, debit, release.
    The provider is modelled as a ``WindowMeter`` over the tokens processed (simulated). The run continues past the
    last arrival until every admitted stream drains (``end``); ``seconds_over_limit`` counts the whole run, and
    ``first_over``/``last_over`` say when it started and stopped."""
    rate = limit / 60
    bucket, lim, provider = TokenBucket(rate, rate * burst_s), ReserveLimiter(limit), WindowMeter()
    live, done, admitted, refused, truncated, served, peak, over_s = {}, [], 0, 0, 0, 0.0, 0.0, 0
    first_over = last_over = None
    horizon = int(arrivals[-1][0]) + 1 if arrivals else 0
    i, t = 0, 0
    while i < len(arrivals) or live:
        t += 1
        while i < len(arrivals) and arrivals[i][0] < t:
            _, p, o = arrivals[i]
            ok = bucket.try_acquire(est, t) if kind == "per_request" else lim.admit(i, p + reserve_out, t)
            if ok:
                admitted += 1
                truncated += cap is not None and o > cap
                live[i] = [p, min(o, cap) if cap else o, True]
            else:
                refused += 1
            i += 1
        for rid, st in list(live.items()):
            n = st[0] if st[2] else 0                     # the prompt, in its first second
            step = min(st[1], tok_per_s)
            st[1] -= step
            n, st[2] = n + step, False
            provider.add(t, n)
            if kind == "reserve":
                lim.debit(rid, n, t)
            served += n if t <= horizon else 0
            if st[1] <= 0:
                lim.finish(rid)
                del live[rid]
        used = provider.used(t)
        peak, over_s = max(peak, used), over_s + (used > limit)
        if used > limit:
            first_over, last_over = (t if first_over is None else first_over), t
    return {"admitted": admitted, "refused": refused, "truncated": truncated, "peak_ratio": peak / limit,
            "seconds_over_limit": over_s, "first_over": first_over, "last_over": last_over, "end": t,
            "utilisation": served / (limit / 60 * horizon) if horizon else 0.0, "overrun_tokens": lim.overrun}


def compare_buckets(*, limit: float = 1_000_000, prompt: int = 1_500, old_median: float = 300,
                    new_median: float = 1_500, sigma: float = 1.0, rate: float = 12.0, seconds: float = 600,
                    cap: int = 16_384, estimate_out: int = 4_096, seed: int = 0) -> dict:
    """The §4 experiment: a per-request bucket sized for yesterday's outputs, then a thinking rollout."""
    est = prompt + lognormal_mean(old_median, sigma)
    old = workload(seconds, rate, prompt, old_median, sigma, seed)
    new = workload(seconds, rate, prompt, new_median, sigma, seed)
    return {
        "per-request, old outputs": simulate(old, "per_request", limit=limit, est=est),
        "per-request, thinking outputs": simulate(new, "per_request", limit=limit, est=est),
        f"reserve prompt + {estimate_out:,}": simulate(new, "reserve", limit=limit, reserve_out=estimate_out),
        f"reserve prompt + cap {cap:,}": simulate(new, "reserve", limit=limit, reserve_out=cap, cap=cap),
    }
