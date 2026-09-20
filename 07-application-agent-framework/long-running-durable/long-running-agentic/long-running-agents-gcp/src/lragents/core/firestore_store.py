"""Firestore-backed ``RunStore`` and ``IdempotencyStore``.

Why Firestore for agent checkpoints?
* Serverless, no connection pool to manage from Cloud Run, pay per operation.
* Transactions give us the two primitives we need: compare-and-set on
  ``version`` (optimistic concurrency) and compare-and-set on ``lease``.
* A run maps to one document. Journals are stored inline here for clarity;
  once a run's journal risks the 1 MiB document limit, move steps to a
  ``runs/{id}/steps`` subcollection and keep only the head pointer in the run doc.

Two Firestore facts every design should respect:
* A single document should not be written more than ~1 time per second sustained
  (hot-document limit). One run = one worker = one write stream, so a run is fine;
  a *global* counter document is not (see the fan-in pattern for the sharded fix).
* Transactions retry automatically on contention; keep the transaction body free
  of external side effects because it may run more than once.

Run locally against the emulator:
    gcloud emulators firestore start --host-port=localhost:8080
    export FIRESTORE_EMULATOR_HOST=localhost:8080
"""

from __future__ import annotations

import time
from typing import Any

from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from .models import Lease, Run, RunStatus
from .store import LeaseHeld, RunNotFound, VersionConflict


class FirestoreRunStore:
    def __init__(
        self,
        client: firestore.Client | None = None,
        collection: str = "agent_runs",
        project: str | None = None,
        database: str | None = None,
    ) -> None:
        self.db = client or firestore.Client(project=project, database=database)
        self.col = self.db.collection(collection)

    # -- protocol ----------------------------------------------------------
    def create(self, run: Run) -> Run:
        ref = self.col.document(run.run_id)
        run.version = 1
        try:
            ref.create(run.to_dict())          # atomic "create if absent"
        except AlreadyExists:
            pass                               # idempotent: a retry of the same start
        return self.get(run.run_id)

    def get(self, run_id: str) -> Run:
        snap = self.col.document(run_id).get()
        if not snap.exists:
            raise RunNotFound(run_id)
        return Run.from_dict(snap.to_dict())

    def save(self, run: Run) -> Run:
        ref = self.col.document(run.run_id)
        expected = run.version

        @firestore.transactional
        def _save(tx: firestore.Transaction) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                raise RunNotFound(run.run_id)
            if snap.get("version") != expected:
                raise VersionConflict(f"{run.run_id}: expected v{expected}, stored v{snap.get('version')}")
            run.version = expected + 1
            run.updated_at = time.time()
            tx.set(ref, run.to_dict())

        _save(self.db.transaction())
        return self.get(run.run_id)

    def acquire_lease(self, run_id: str, owner: str, ttl_s: float) -> Run:
        ref = self.col.document(run_id)

        @firestore.transactional
        def _acquire(tx: firestore.Transaction) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                raise RunNotFound(run_id)
            data = snap.to_dict()
            now = time.time()
            lease = data.get("lease")
            if lease and lease["owner"] != owner and now < lease["expires_at"]:
                raise LeaseHeld(f"{run_id} leased by {lease['owner']}")
            tx.update(
                ref,
                {
                    "lease": {"owner": owner, "expires_at": now + ttl_s},
                    "version": data["version"] + 1,
                    "updated_at": now,
                },
            )

        _acquire(self.db.transaction())
        return self.get(run_id)

    def release_lease(self, run_id: str, owner: str) -> None:
        ref = self.col.document(run_id)

        @firestore.transactional
        def _release(tx: firestore.Transaction) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                return
            data = snap.to_dict()
            lease = data.get("lease")
            if lease and lease["owner"] == owner:
                tx.update(ref, {"lease": None, "version": data["version"] + 1, "updated_at": time.time()})

        _release(self.db.transaction())

    def list_runs(self, status: RunStatus | None = None) -> list[Run]:
        q = self.col
        if status is not None:
            q = q.where(filter=FieldFilter("status", "==", status.value))
        return [Run.from_dict(s.to_dict()) for s in q.stream()]


class FirestoreIdempotencyStore:
    """``idempotency_keys/{key}`` documents. Keys are set once; reads are cheap.

    Add a TTL policy on ``expires_at`` in the console/Terraform so old keys age out.
    """

    def __init__(self, client: firestore.Client | None = None, collection: str = "idempotency_keys", ttl_days: int = 30) -> None:
        self.db = client or firestore.Client()
        self.col = self.db.collection(collection)
        self.ttl_s = ttl_days * 86400

    def get(self, key: str) -> Any | None:
        snap = self.col.document(_safe_id(key)).get()
        return snap.to_dict().get("result") if snap.exists else None

    def put(self, key: str, result: Any) -> None:
        self.col.document(_safe_id(key)).set(
            {"key": key, "result": result, "created_at": time.time(), "expires_at": time.time() + self.ttl_s}
        )

    def __contains__(self, key: str) -> bool:
        return self.col.document(_safe_id(key)).get().exists


def _safe_id(key: str) -> str:
    # Firestore doc ids cannot contain '/'; keys like "run_x:3" are fine but be defensive.
    return key.replace("/", "__")
