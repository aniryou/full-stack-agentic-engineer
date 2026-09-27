"""In-memory implementations of every port.

They deliberately mimic the *semantics* of the GCP services, not just the
interfaces: the queue dedups by task name and honours delays, the store does
optimistic concurrency and lease expiry, the LLM charges tokens. This lets the
notebooks demonstrate crash/resume, duplicate delivery and backoff without a
Google Cloud project.
"""

from __future__ import annotations

import copy
import heapq
import itertools
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from ...core.engine import SimulatedCrash
from ...core.models import LLMResponse, LLMUsage, Run, StepTask
from ...core.ports import ConflictError, LeaseHeldError


# ---------------------------------------------------------------------------
class FakeClock:
    """Controllable clock so tests can fast-forward through delays and timeouts."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 1, 1, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float = 0, **kw: Any) -> datetime:
        self._now += timedelta(seconds=seconds, **kw)
        return self._now


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
class InMemoryStateStore:
    """Dict-backed store with the same concurrency contract as Firestore."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}
        self._effects: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.write_count = 0

    # -- runs ----------------------------------------------------------------
    def create(self, run: Run) -> Run:
        with self._lock:
            if run.run_id in self._runs:
                raise ConflictError(f"run {run.run_id} already exists")
            run.version = 1
            self._runs[run.run_id] = run.to_doc()
            self.write_count += 1
            return Run.from_doc(copy.deepcopy(self._runs[run.run_id]))

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            doc = self._runs.get(run_id)
            return Run.from_doc(copy.deepcopy(doc)) if doc else None

    def save(self, run: Run, *, expected_version: int) -> Run:
        with self._lock:
            doc = self._runs.get(run.run_id)
            if doc is None:
                raise ConflictError(f"run {run.run_id} does not exist")
            if doc["version"] != expected_version:
                raise ConflictError(
                    f"run {run.run_id}: expected version {expected_version}, found {doc['version']}"
                )
            run.version = expected_version + 1
            self._runs[run.run_id] = run.to_doc()
            self.write_count += 1
            return Run.from_doc(copy.deepcopy(self._runs[run.run_id]))

    def acquire_lease(self, run_id: str, owner: str, ttl: timedelta, now: datetime) -> Run:
        with self._lock:
            run = self.get(run_id)
            if run is None:
                raise ConflictError(f"run {run_id} does not exist")
            if run.lease and run.lease.owner != owner and not run.lease.expired(now):
                raise LeaseHeldError(f"run {run_id} leased by {run.lease.owner} until {run.lease.expires_at}")
            from ...core.models import Lease

            run.lease = Lease(owner=owner, expires_at=now + ttl)
            return self.save(run, expected_version=run.version)

    def release_lease(self, run_id: str, owner: str) -> None:
        with self._lock:
            run = self.get(run_id)
            if run and run.lease and run.lease.owner == owner:
                run.lease = None
                self.save(run, expected_version=run.version)

    def list_runs(self, *, status: str | None = None, limit: int = 100) -> list[Run]:
        with self._lock:
            out = [Run.from_doc(copy.deepcopy(d)) for d in self._runs.values() if status is None or d["status"] == status]
            return out[:limit]

    # -- effects -------------------------------------------------------------
    def effect_get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            v = self._effects.get(key)
            return copy.deepcopy(v) if v is not None else None

    def effect_put(self, key: str, value: dict[str, Any]) -> bool:
        with self._lock:
            if key in self._effects:
                return False
            self._effects[key] = copy.deepcopy(value)
            return True

    # -- fan-in --------------------------------------------------------------
    def record_child_result(self, parent_run_id: str, child_key: str, result: Any, error: str | None) -> Run:
        with self._lock:
            parent = self.get(parent_run_id)
            if parent is None:
                raise ConflictError(f"parent {parent_run_id} does not exist")
            fi = parent.fan_in
            if fi is None:
                return parent
            if child_key in fi.results or child_key in fi.failures:
                return parent  # duplicate completion notification
            if error is None:
                fi.results[child_key] = result
            else:
                fi.failures[child_key] = error
            fi.completed += 1
            return self.save(parent, expected_version=parent.version)

    # -- introspection helpers for notebooks ---------------------------------
    def dump(self, run_id: str) -> str:
        return json.dumps(self._runs[run_id], indent=2, default=str)


# ---------------------------------------------------------------------------
class InMemoryTaskQueue:
    """Priority queue keyed by ``not_before`` with Cloud Tasks-style name dedup."""

    def __init__(self, clock: FakeClock | SystemClock) -> None:
        self._clock = clock
        self._heap: list[tuple[datetime, int, StepTask]] = []
        self._names: set[str] = set()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self.enqueued: list[StepTask] = []
        self.rejected_duplicates: list[str] = []
        self.delivery_log: list[tuple[str, str]] = []  # (dedup_key, outcome)

    def enqueue(self, task: StepTask, *, delay: timedelta | None = None) -> bool:
        with self._lock:
            if task.dedup_key in self._names:
                self.rejected_duplicates.append(task.dedup_key)
                return False
            self._names.add(task.dedup_key)
            when = self._clock.now() + (delay or timedelta(0))
            heapq.heappush(self._heap, (when, next(self._seq), task))
            self.enqueued.append(task)
            return True

    def pop_due(self) -> StepTask | None:
        with self._lock:
            if not self._heap or self._heap[0][0] > self._clock.now():
                return None
            _, _, task = heapq.heappop(self._heap)
            return task

    def pending(self) -> list[StepTask]:
        with self._lock:
            return [t for _, _, t in sorted(self._heap)]

    def next_due_at(self) -> datetime | None:
        with self._lock:
            return self._heap[0][0] if self._heap else None

    def forget(self, dedup_key: str) -> None:
        """Cloud Tasks frees a name only after the task completes (plus a tombstone period)."""
        with self._lock:
            self._names.discard(dedup_key)

    def __len__(self) -> int:
        return len(self._heap)


# ---------------------------------------------------------------------------
class InMemoryEventBus:
    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self._subs: list[Callable[[str, dict[str, Any]], None]] = []

    def publish(self, topic: str, payload: dict[str, Any], attributes: dict[str, str] | None = None) -> str:
        msg = {"topic": topic, "payload": copy.deepcopy(payload), "attributes": dict(attributes or {})}
        self.published.append(msg)
        for fn in self._subs:
            fn(topic, payload)
        return f"msg-{len(self.published)}"

    def subscribe(self, fn: Callable[[str, dict[str, Any]], None]) -> None:
        self._subs.append(fn)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [m["payload"] for m in self.published if m["payload"].get("type") == event_type]


# ---------------------------------------------------------------------------
class FakeLLM:
    """Scripted model. Routes prompts to canned answers, charges fake tokens.

    ``routes`` maps a regex (searched in the prompt) to either a string or a
    callable ``(prompt) -> str``. First match wins; ``default`` otherwise.
    ``fail_times`` makes the first N calls raise, to exercise retry paths.
    """

    PRICE_PER_1K = (0.00025, 0.001)  # (input, output) — illustrative Flash-class pricing

    def __init__(
        self,
        routes: dict[str, str | Callable[[str], str]] | None = None,
        default: str | Callable[[str], str] = "OK",
        fail_times: int = 0,
    ) -> None:
        self.routes = routes or {}
        self.default = default
        self.fail_times = fail_times
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("simulated model outage (503)")
        answer: str | Callable[[str], str] = self.default
        for pattern, value in self.routes.items():
            if re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL):
                answer = value
                break
        text = answer(prompt) if callable(answer) else answer
        if json_mode and not isinstance(text, str):
            text = json.dumps(text)
        in_tok = max(1, (len(prompt) + len(system or "")) // 4)
        out_tok = max(1, len(text) // 4)
        cost = in_tok / 1000 * self.PRICE_PER_1K[0] + out_tok / 1000 * self.PRICE_PER_1K[1]
        return LLMResponse(text=text, usage=LLMUsage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost), model="fake-flash")


# ---------------------------------------------------------------------------
class LocalRunner:
    """Drives an Engine against the in-memory queue the way Cloud Tasks would.

    Cloud Tasks retries a task when the worker returns non-2xx; ``lease-held``
    is the one outcome we map to that, with a short delay.
    """

    def __init__(self, engine: Any, queue: InMemoryTaskQueue, clock: FakeClock, *, lease_retry_s: float = 1.0, crash_redelivery_s: float = 30.0) -> None:
        self.engine = engine
        self.queue = queue
        self.clock = clock
        self.lease_retry_s = lease_retry_s
        self.crash_redelivery_s = crash_redelivery_s
        self.trace: list[tuple[str, str]] = []

    def step(self, *, auto_advance: bool = False) -> bool:
        """Deliver one due task. Returns False if nothing is due.

        With ``auto_advance`` the clock jumps to the next scheduled task (a
        delayed retry or timer) instead of returning False.
        """
        task = self.queue.pop_due()
        if task is None and auto_advance:
            nxt = self.queue.next_due_at()
            if nxt is not None and nxt > self.clock.now():
                self.clock._now = nxt
                task = self.queue.pop_due()
        if task is None:
            return False
        try:
            outcome = self.engine.execute_task(task)
        except SimulatedCrash:
            # The worker died mid-request. Cloud Tasks never got a response, so it
            # redelivers the same task after the dispatch deadline / backoff.
            self.queue.forget(task.dedup_key)
            self.queue.enqueue(task, delay=timedelta(seconds=self.crash_redelivery_s))
            self.trace.append((task.dedup_key, "crashed"))
            raise
        self.trace.append((task.dedup_key, outcome))
        self.queue.delivery_log.append((task.dedup_key, outcome))
        if outcome == "lease-held":
            self.queue.forget(task.dedup_key)
            self.queue.enqueue(task, delay=timedelta(seconds=self.lease_retry_s))
        else:
            self.queue.forget(task.dedup_key)
        return True

    def run_until_idle(self, *, max_tasks: int = 1000, auto_advance: bool = True) -> int:
        """Drain the queue. With ``auto_advance`` the clock jumps to the next delayed task."""
        n = 0
        while n < max_tasks:
            if self.step():
                n += 1
                continue
            nxt = self.queue.next_due_at()
            if nxt is None or not auto_advance:
                break
            if nxt > self.clock.now():
                self.clock._now = nxt  # jump straight to the next scheduled task
        return n
