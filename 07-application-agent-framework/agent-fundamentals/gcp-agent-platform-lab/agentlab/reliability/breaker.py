"""Circuit breaker, bulkhead and deadlines: stop a sick dependency from taking the agent down (Primer §4.4).

* A **circuit breaker** fails fast once a dependency keeps failing, so callers stop piling
  timeouts onto it and it gets room to recover; after a cooling period one probe decides
  whether to close the circuit again.
* A **bulkhead** caps how many calls may be in flight (plus a small queue) so one slow
  system cannot consume every worker and starve the others.
* A **deadline** is a point in time, not a duration: every hop of a turn is given what is
  left of the whole budget, so the sum of hops cannot exceed what the user was promised.

Clocks are injected everywhere: state machines are tested with a fake clock, not with sleeps.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from .retry import is_retryable

T = TypeVar("T")
Clock = Callable[[], float]


# ----------------------------------------------------------- circuit breaker
class BreakerState(str, Enum):
    CLOSED = "closed"          # normal: calls flow, failures are counted
    OPEN = "open"              # tripped: calls are rejected without touching the dependency
    HALF_OPEN = "half_open"    # cooling period over: a bounded number of probe calls decide


class CircuitOpen(RuntimeError):
    """Raised instead of calling the dependency while the circuit is open."""

    def __init__(self, name: str, retry_after_s: float):
        super().__init__(f"circuit {name!r} is open; retry in {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s


@dataclass
class BreakerMetrics:
    failures: int = 0      # calls that failed against the dependency
    successes: int = 0
    opens: int = 0         # transitions into OPEN (each is an alert-worthy event)
    rejections: int = 0    # calls refused while open / half-open capacity used


class CircuitBreaker:
    """Closed → open after ``failure_threshold`` consecutive failures; open → half-open after
    ``recovery_timeout_s``; half-open → closed on a successful probe, → open on a failed one.

    ``trip_on`` decides which errors count: by default the transient class (timeouts, 5xx,
    rate limits). A 404 or a validation error says nothing about the dependency's health and
    must not open the circuit.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Clock = time.monotonic,
        name: str = "breaker",
        trip_on: Callable[[BaseException], bool] = is_retryable,
    ):
        if failure_threshold < 1 or half_open_max_calls < 1:
            raise ValueError("failure_threshold and half_open_max_calls must be >= 1")
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self.clock = clock
        self.name = name
        self.trip_on = trip_on
        self.metrics = BreakerMetrics()
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probes_in_flight = 0

    # -- state ---------------------------------------------------------------
    @property
    def state(self) -> BreakerState:
        """Current state, applying the time-based OPEN → HALF_OPEN transition lazily."""
        if self._state is BreakerState.OPEN and self.clock() - self._opened_at >= self.recovery_timeout_s:
            self._state = BreakerState.HALF_OPEN
            self._probes_in_flight = 0
        return self._state

    def retry_after_s(self) -> float:
        return max(0.0, self.recovery_timeout_s - (self.clock() - self._opened_at))

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self.clock()
        self._consecutive_failures = 0
        self.metrics.opens += 1

    def record_success(self) -> None:
        self.metrics.successes += 1
        self._consecutive_failures = 0
        if self.state is BreakerState.HALF_OPEN:
            self._state = BreakerState.CLOSED

    def record_failure(self) -> None:
        self.metrics.failures += 1
        self._consecutive_failures += 1
        if self.state is BreakerState.HALF_OPEN or self._consecutive_failures >= self.failure_threshold:
            self._trip()

    def _admit(self) -> bool:
        """Decide whether a call may proceed; returns True when it counts as a half-open probe."""
        state = self.state
        if state is BreakerState.OPEN:
            self.metrics.rejections += 1
            raise CircuitOpen(self.name, self.retry_after_s())
        if state is BreakerState.HALF_OPEN:
            if self._probes_in_flight >= self.half_open_max_calls:
                self.metrics.rejections += 1
                raise CircuitOpen(self.name, 0.0)
            self._probes_in_flight += 1
            return True
        return False

    # -- use -----------------------------------------------------------------
    async def call(self, fn: Callable[..., Awaitable[T] | T], *args: Any, **kwargs: Any) -> T:
        """Run ``fn`` through the breaker: reject when open, count the outcome otherwise."""
        probe = self._admit()
        try:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:  # noqa: BLE001 - classified by trip_on, then re-raised
            if self.trip_on(exc):
                self.record_failure()
            raise
        else:
            self.record_success()
            return result
        finally:
            if probe and self._state is BreakerState.HALF_OPEN:
                self._probes_in_flight -= 1


# ------------------------------------------------------------------ bulkhead
class BulkheadFull(RuntimeError):
    """Raised when every slot and every queue position is taken: shed the load, do not wait."""


class Bulkhead:
    """At most ``max_concurrent`` calls in flight and ``max_queue`` waiting; the rest are rejected.

    The explicit queue limit is the point: an unbounded semaphore queue converts overload into
    latency that grows without limit, which is worse than an honest "busy, try later".
    """

    def __init__(self, max_concurrent: int, max_queue: int = 0, name: str = "bulkhead"):
        if max_concurrent < 1 or max_queue < 0:
            raise ValueError("max_concurrent must be >= 1 and max_queue >= 0")
        self.max_concurrent = max_concurrent
        self.max_queue = max_queue
        self.name = name
        self._slots = asyncio.Semaphore(max_concurrent)
        self.active = 0
        self.waiting = 0
        self.peak_active = 0
        self.rejections = 0

    async def run(self, fn: Callable[..., Awaitable[T] | T], *args: Any, **kwargs: Any) -> T:
        if self._slots.locked() and self.waiting >= self.max_queue:
            self.rejections += 1
            raise BulkheadFull(f"{self.name}: {self.max_concurrent} active and {self.waiting} queued")
        self.waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self.waiting -= 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            result = fn(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        finally:
            self.active -= 1
            self._slots.release()


# ----------------------------------------------------------------- deadlines
class DeadlineExceeded(RuntimeError):
    """The turn's time budget is gone; whatever is left of the plan is abandoned."""


class Deadline:
    """An absolute point in time, measured on an injectable clock.

    Pass the *deadline* down the call chain, never the original duration: a hop that starts
    late gets less time, and no hop can outlive the turn that owns it.
    """

    def __init__(self, seconds: float, clock: Clock = time.monotonic):
        self.seconds = seconds
        self.clock = clock
        self.started_at = clock()

    def remaining(self) -> float:
        return max(0.0, self.seconds - (self.clock() - self.started_at))

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def sub(self, seconds: float) -> "Deadline":
        """A child deadline for one hop: the requested time, capped by what remains."""
        return Deadline(min(seconds, self.remaining()), self.clock)


async def with_deadline(awaitable: Awaitable[T], deadline: Deadline) -> T:
    """Await ``awaitable`` for at most ``deadline.remaining()`` seconds."""
    remaining = deadline.remaining()
    if remaining <= 0.0:
        if inspect.iscoroutine(awaitable):
            awaitable.close()  # never started: avoid the "coroutine was never awaited" warning
        raise DeadlineExceeded(f"deadline of {deadline.seconds:.2f}s already passed")
    try:
        return await asyncio.wait_for(awaitable, timeout=remaining)
    except asyncio.TimeoutError:
        raise DeadlineExceeded(f"exceeded {deadline.seconds:.2f}s deadline") from None


def per_hop_budget(total_s: float, hops: int, reserve_s: float = 0.0) -> float:
    """Split a turn budget evenly across sequential hops after reserving time for the answer.

    The reserve matters: a turn that spends every millisecond on tools has nothing left to
    produce a fallback message, which is the worst kind of timeout — silent.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    per_hop = (total_s - reserve_s) / hops
    if per_hop <= 0:
        raise ValueError(f"no time left for {hops} hops after reserving {reserve_s}s of {total_s}s")
    return per_hop
