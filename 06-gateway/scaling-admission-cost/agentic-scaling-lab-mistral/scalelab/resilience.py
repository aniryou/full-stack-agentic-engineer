"""Four small mechanisms that keep a shared model pool usable under load.

TokenBucket        smooth our own traffic to a rate we chose (tokens per second), so we
                   queue for a few hundred milliseconds on our side instead of taking 429s.
backoff()          exponential backoff with *full jitter*: when a pool throttles everyone at
                   once, jitter is what stops everyone retrying at once.
CircuitBreaker     after repeated failures, fail fast for a cooling period and probe gently.
call_with_retries  puts the three together around one model call, bounded by a deadline.
"""

from __future__ import annotations

import random
from collections import deque

from .clock import CLOCK


class RateLimited(Exception):
    """A 429: the shared pool is contended. ``retry_after`` is the server's hint, if any."""

    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited")
        self.retry_after = retry_after


class CircuitOpen(Exception):
    pass


class TokenBucket:
    """Refills ``rate`` tokens per second up to ``capacity``; ``acquire`` waits for a deficit."""

    def __init__(self, rate: float, capacity: float):
        self.rate, self.capacity = rate, capacity
        self.tokens = capacity
        self.last = CLOCK.now()
        self.total_wait = 0.0

    def _refill(self) -> None:
        now = CLOCK.now()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """Take ``tokens``; returns how long we waited. A request larger than the burst goes into debt."""
        self._refill()
        deficit = tokens - self.tokens
        wait = max(0.0, deficit / self.rate) if tokens <= self.capacity else max(0.0, (self.capacity - self.tokens) / self.rate)
        if wait:
            await CLOCK.sleep(wait)
            self._refill()
        self.tokens -= tokens
        self.total_wait += wait
        return wait


def backoff(attempt: int, *, base: float = 0.5, cap: float = 8.0, jitter: bool = True, rng: random.Random | None = None) -> float:
    """Delay before retry number ``attempt`` (1-based): min(cap, base·2^(attempt-1)), jittered uniformly."""
    raw = min(cap, base * 2 ** (attempt - 1))
    return (rng or random).uniform(0, raw) if jitter else raw


class CircuitBreaker:
    """CLOSED → OPEN when, over the last ``window`` seconds, at least ``min_calls`` were made and
    ``ratio`` of them failed (and at least ``threshold`` did) → HALF_OPEN after ``cooldown``: one
    probe closes it on success or re-opens it on failure."""

    def __init__(self, threshold: int = 5, min_calls: int = 10, ratio: float = 0.5, window: float = 30.0, cooldown: float = 15.0):
        self.threshold, self.min_calls, self.ratio, self.window, self.cooldown = threshold, min_calls, ratio, window, cooldown
        self.calls: deque[tuple[float, bool]] = deque()   # (time, ok)
        self.opened_at: float | None = None
        self.trips = 0

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if CLOCK.now() - self.opened_at >= self.cooldown else "open"

    def before_call(self) -> None:
        if self.state == "open":
            raise CircuitOpen()

    def record(self, ok: bool) -> None:
        now = CLOCK.now()
        if self.state == "half_open":
            self.opened_at = None if ok else now      # the probe decides
            if ok:
                self.calls.clear()
            return
        self.calls.append((now, ok))
        while self.calls and self.calls[0][0] < now - self.window:
            self.calls.popleft()
        failures = sum(1 for _, k in self.calls if not k)
        if len(self.calls) >= self.min_calls and failures >= self.threshold and failures / len(self.calls) >= self.ratio:
            self.opened_at, self.trips = now, self.trips + 1
            self.calls.clear()


async def call_with_retries(fn, *, deadline: float, bucket: TokenBucket | None = None, tokens: float = 1.0,
                            breaker: CircuitBreaker | None = None, max_attempts: int = 4, rng: random.Random | None = None,
                            stats: dict | None = None):
    """Run ``await fn()`` with smoothing, breaker and jittered retries, never past ``deadline`` (virtual seconds).

    Retries only on RateLimited. ``stats`` (optional dict) counts attempts and rate limits.
    """
    stats = stats if stats is not None else {}
    for attempt in range(1, max_attempts + 1):
        if CLOCK.now() >= deadline:
            raise TimeoutError("deadline before attempt")
        if breaker:
            breaker.before_call()
        if bucket:
            await bucket.acquire(tokens)
        stats["attempts"] = stats.get("attempts", 0) + 1
        try:
            result = await fn()
        except RateLimited as e:
            stats["rate_limited"] = stats.get("rate_limited", 0) + 1
            if breaker:
                breaker.record(False)
            delay = max(backoff(attempt, rng=rng), (e.retry_after or 0) * (1 + (rng or random).random() * 0.25))
            if attempt == max_attempts or CLOCK.now() + delay >= deadline:
                raise
            await CLOCK.sleep(delay)
            continue
        if breaker:
            breaker.record(True)
        return result
    raise RateLimited()
