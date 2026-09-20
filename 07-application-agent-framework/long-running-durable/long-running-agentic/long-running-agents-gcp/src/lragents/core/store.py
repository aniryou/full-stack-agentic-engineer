"""Run persistence.

The store is the *only* thing that has to be durable. Two guarantees matter:

1. **Optimistic concurrency** – ``save`` succeeds only if the run's ``version``
   equals the stored version, then bumps it. Two workers racing on the same run
   cannot both win; the loser gets ``VersionConflict`` and must reload.
2. **Leases** – ``acquire_lease`` is an atomic *compare-and-set* on the lease
   field. It gives a worker exclusive right to advance a run for ``ttl`` seconds.
   Leases expire, so a dead worker never wedges a run forever.

``InMemoryRunStore`` implements both and deliberately deep-copies on every
read/write to simulate the serialisation boundary of a real database: bugs that
rely on in-place mutation surface here instead of in production.

``firestore_store.FirestoreRunStore`` implements the same protocol with
Firestore transactions.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Callable, Protocol, TypeVar

from .models import Lease, Run, RunStatus


class VersionConflict(Exception):
    """Another writer updated the run since we read it."""


class RunNotFound(KeyError):
    pass


class LeaseHeld(Exception):
    """Someone else holds an unexpired lease on this run."""


class RunStore(Protocol):
    def create(self, run: Run) -> Run: ...
    def get(self, run_id: str) -> Run: ...
    def save(self, run: Run) -> Run: ...
    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> Run: ...
    def release_lease(self, run_id: str, owner: str) -> None: ...
    def list_runs(self, status: RunStatus | None = None) -> list[Run]: ...


class InMemoryRunStore:
    """Dictionary-backed store with the same semantics as the Firestore one."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._docs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._clock = clock
        self.save_count = 0

    # -- helpers -----------------------------------------------------------
    def _load(self, run_id: str) -> Run:
        try:
            return Run.from_dict(copy.deepcopy(self._docs[run_id]))
        except KeyError as e:
            raise RunNotFound(run_id) from e

    def _store(self, run: Run) -> None:
        run.updated_at = self._clock()
        self._docs[run.run_id] = copy.deepcopy(run.to_dict())

    # -- protocol ----------------------------------------------------------
    def create(self, run: Run) -> Run:
        with self._lock:
            if run.run_id in self._docs:
                # idempotent create: return what is there
                return self._load(run.run_id)
            run.version = 1
            self._store(run)
            return self._load(run.run_id)

    def get(self, run_id: str) -> Run:
        with self._lock:
            return self._load(run_id)

    def save(self, run: Run) -> Run:
        with self._lock:
            current = self._load(run.run_id)
            if current.version != run.version:
                raise VersionConflict(
                    f"{run.run_id}: expected v{run.version}, stored v{current.version}"
                )
            run.version += 1
            self._store(run)
            self.save_count += 1
            return self._load(run.run_id)

    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> Run:
        with self._lock:
            run = self._load(run_id)
            now = self._clock()
            if run.lease and run.lease.owner != owner and not run.lease.expired(now):
                raise LeaseHeld(f"{run_id} leased by {run.lease.owner} until {run.lease.expires_at:.0f}")
            run.lease = Lease(owner=owner, expires_at=now + ttl_s)
            run.version += 1
            self._store(run)
            return self._load(run_id)

    def release_lease(self, run_id: str, owner: str) -> None:
        with self._lock:
            run = self._load(run_id)
            if run.lease and run.lease.owner == owner:
                run.lease = None
                run.version += 1
                self._store(run)

    def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        with self._lock:
            runs = [self._load(k) for k in self._docs]
        return [r for r in runs if status is None or r.status == status]


T = TypeVar("T")


def transact(store: RunStore, run_id: str, mutate: Callable[[Run], T], max_retries: int = 10) -> tuple[Run, T]:
    """Read-modify-write with optimistic retry.

    ``mutate`` must be a pure function of the run it receives (no external side
    effects!) because it may be re-run on conflict. This is the in-memory analogue
    of a Firestore transaction function.
    """
    last: Exception | None = None
    for _ in range(max_retries):
        run = store.get(run_id)
        out = mutate(run)
        try:
            return store.save(run), out
        except VersionConflict as e:   # someone else won, reload and retry
            last = e
    raise RuntimeError(f"transact({run_id}) gave up after {max_retries} conflicts") from last
