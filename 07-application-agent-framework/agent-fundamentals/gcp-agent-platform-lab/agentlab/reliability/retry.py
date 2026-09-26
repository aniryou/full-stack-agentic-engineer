"""Retries: classify first, back off with jitter, and never retry a write without a key (notebook 10).

Three ideas, each of which a design review will probe:

* **Classification before retrying.** A 503 or a timeout may succeed on the next try; a 400 or a
  404 will fail identically forever, and retrying it only burns budget and hides a bug.
* **Capped exponential backoff with jitter.** Doubling spreads load off a struggling dependency;
  the cap keeps the wait bounded; jitter stops every client from retrying in lock-step
  (the "thundering herd" that turns a blip into an outage).
* **Idempotency keys.** A timeout is *ambiguous*: the write may have landed before the response
  was lost. Retrying without a key means "maybe do it twice" — for a refund, that is an incident.
  ``IdempotentCall`` shows the server-side half of the contract.

Sleeps and random sources are injected so tests and notebooks run in milliseconds.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import random
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, MutableMapping, TypeVar

from ..agents.tools import ToolPermanentError, ToolTransientError

T = TypeVar("T")


# ------------------------------------------------------------ classification
class RetryableError(Exception):
    """A failure that a later attempt may not see (503, 429, timeout, connection reset).

    ``retry_after_s`` carries a server hint (HTTP ``Retry-After``); ``retry`` honours it
    as a floor on the next delay, because ignoring it is how clients get rate-limited harder.
    """

    def __init__(self, message: str = "retryable failure", *, status: int | None = None, retry_after_s: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after_s = retry_after_s


class PermanentError(Exception):
    """A failure that will repeat (400, 401, 403, 404, validation). Do not retry; fix the request."""

    def __init__(self, message: str = "permanent failure", *, status: int | None = None):
        super().__init__(message)
        self.status = status


# 408 request timeout, 429 rate limited, 502/503/504 upstream unavailable. 500 is deliberately
# absent: it is often a deterministic bug, and retrying deterministic failures amplifies outages.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 429, 502, 503, 504})


def classify_http(status: int, retryable: frozenset[int] = RETRYABLE_STATUSES) -> bool:
    """Return True when an HTTP status is worth retrying.

    Other 4xx codes are the caller's fault and are permanent by definition; anything not
    in ``retryable`` is treated as permanent so unknown failures fail fast and get looked at.
    """
    return status in retryable


def is_retryable(exc: BaseException) -> bool:
    """Default retry predicate: transient by type, or by HTTP status when the error carries one.

    Unknown exceptions are *not* retryable — a ``KeyError`` in your own code will not fix itself.
    """
    if isinstance(exc, (PermanentError, ToolPermanentError)):
        return False
    if isinstance(exc, (RetryableError, ToolTransientError, asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status", None)
    return isinstance(status, int) and classify_http(status)


# ------------------------------------------------------------------- policy
@dataclass(frozen=True)
class RetryPolicy:
    """How many times to try and how long to wait between tries.

    Defaults give delays of roughly 0.5, 1, 2 s (+ up to 0.3 s jitter): four attempts
    complete in under 5 s, which fits inside a typical per-tool deadline.
    """

    max_attempts: int = 4
    base_s: float = 0.5
    cap_s: float = 8.0
    jitter_s: float = 0.3
    retry_on: Callable[[BaseException], bool] = is_retryable

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_s < 0 or self.jitter_s < 0:
            raise ValueError("base_s and jitter_s must be >= 0")
        if self.cap_s < self.base_s:
            raise ValueError("cap_s must be >= base_s")


def backoff_schedule(policy: RetryPolicy, rng: random.Random) -> list[float]:
    """The delay before each retry: ``min(cap, base * 2**i) + uniform(0, jitter)``.

    Pure function of the policy and the random source, so the schedule can be asserted in
    tests and printed in notebooks. Length is ``max_attempts - 1`` (no sleep after the last try).
    """
    return [
        min(policy.cap_s, policy.base_s * (2 ** i)) + rng.uniform(0.0, policy.jitter_s)
        for i in range(policy.max_attempts - 1)
    ]


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def retry(
    fn: Callable[[], Awaitable[T] | T],
    policy: RetryPolicy | None = None,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
    on_attempt: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``fn`` until it succeeds, the error is permanent, or attempts run out.

    ``on_attempt(attempt_number, error, next_delay_s)`` fires after each failed attempt that
    will be retried — the hook a tracer or a notebook uses to show what happened. A fresh
    ``Random(0)`` is used when ``rng`` is omitted so runs are reproducible without sharing state.
    """
    policy = policy or RetryPolicy()
    delays = backoff_schedule(policy, rng if rng is not None else random.Random(0))
    attempt = 0
    while True:
        attempt += 1
        try:
            return await _maybe_await(fn())
        except Exception as exc:  # noqa: BLE001 - the predicate decides, not the type hierarchy
            if attempt >= policy.max_attempts or not policy.retry_on(exc):
                raise
            delay = max(delays[attempt - 1], float(getattr(exc, "retry_after_s", None) or 0.0))
            if on_attempt is not None:
                on_attempt(attempt, exc, delay)
            await sleep(delay)


def with_retry(policy: RetryPolicy | None = None, **retry_kwargs: Any) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator form: ``@with_retry(RetryPolicy(max_attempts=3), sleep=fake_sleep)``."""

    def decorate(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry(lambda: fn(*args, **kwargs), policy, **retry_kwargs)

        return wrapper

    return decorate


# -------------------------------------------------------------- idempotency
class IdempotentCall:
    """Apply a write at most once per key; replay the recorded outcome on any repeat.

    This is the *server-side* half of the idempotency contract (the client's half is to send
    the same key on every retry). It closes the timeout-after-success hole: the write lands,
    the response is lost, the client retries with the same key, and instead of a second
    refund the caller gets the first result back.

    In production the "apply" and the "record" must commit in one transaction; here the
    store is any mapping (an ``agentlab.agents.IdempotencyStore`` or a dict).
    """

    def __init__(self, store: MutableMapping[str, Any] | None = None):
        self.store: MutableMapping[str, Any] = store if store is not None else {}
        self.applied = 0
        self.replayed = 0

    async def __call__(self, key: str, fn: Callable[[], Awaitable[T] | T]) -> T:
        if key in self.store:
            self.replayed += 1
            return self.store[key]
        result = await _maybe_await(fn())
        self.store[key] = result
        self.applied += 1
        return result
