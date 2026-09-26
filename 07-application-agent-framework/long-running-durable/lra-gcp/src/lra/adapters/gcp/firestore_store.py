"""Firestore-backed :class:`StateStore`.

Why Firestore for run state:

* Document = run. One read hydrates the whole checkpoint; one transactional
  write commits it. No joins, no schema migrations for ``state``.
* Transactions give us compare-and-set on ``version`` (optimistic
  concurrency) and atomic fan-in counters without a separate lock service.
* It scales to zero cost with the workload, which matters for agents that
  sleep for days.

Limits to design around (see the primer): 1 MiB per document, ~1 write/sec
sustained per document, and composite indexes for multi-field queries.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from ...core.models import Lease, Run
from ...core.ports import ConflictError, LeaseHeldError

log = logging.getLogger("lra.gcp.firestore")


class FirestoreStateStore:
    def __init__(
        self,
        client: Any | None = None,
        *,
        project: str | None = None,
        database: str | None = None,
        runs_collection: str = "agent_runs",
        effects_collection: str = "agent_effects",
    ) -> None:
        from google.cloud import firestore  # lazy: keep the core importable without GCP libs

        if client is None:
            kwargs: dict[str, Any] = {}
            if project:
                kwargs["project"] = project
            if database:
                kwargs["database"] = database
            client = firestore.Client(**kwargs)
        self._fs = firestore
        self.db = client
        self.runs = client.collection(runs_collection)
        self.effects = client.collection(effects_collection)

    # ---------------------------------------------------------------- runs
    def create(self, run: Run) -> Run:
        from google.api_core import exceptions as gexc

        run.version = 1
        try:
            self.runs.document(run.run_id).create(run.to_doc())
        except gexc.AlreadyExists as e:
            raise ConflictError(f"run {run.run_id} already exists") from e
        return run

    def get(self, run_id: str) -> Run | None:
        snap = self.runs.document(run_id).get()
        return Run.from_doc(snap.to_dict()) if snap.exists else None

    def save(self, run: Run, *, expected_version: int) -> Run:
        ref = self.runs.document(run.run_id)
        transaction = self.db.transaction()

        @self._fs.transactional
        def _commit(tx: Any) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                raise ConflictError(f"run {run.run_id} does not exist")
            current = snap.get("version")
            if current != expected_version:
                raise ConflictError(f"run {run.run_id}: expected version {expected_version}, found {current}")
            run.version = expected_version + 1
            tx.set(ref, run.to_doc())

        _commit(transaction)
        return run

    def acquire_lease(self, run_id: str, owner: str, ttl: timedelta, now: datetime) -> Run:
        ref = self.runs.document(run_id)
        transaction = self.db.transaction()
        holder: dict[str, Run] = {}

        @self._fs.transactional
        def _claim(tx: Any) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                raise ConflictError(f"run {run_id} does not exist")
            run = Run.from_doc(snap.to_dict())
            if run.lease and run.lease.owner != owner and not run.lease.expired(now):
                raise LeaseHeldError(f"run {run_id} leased by {run.lease.owner} until {run.lease.expires_at}")
            run.lease = Lease(owner=owner, expires_at=now + ttl)
            run.version += 1
            tx.set(ref, run.to_doc())
            holder["run"] = run

        _claim(transaction)
        return holder["run"]

    def release_lease(self, run_id: str, owner: str) -> None:
        ref = self.runs.document(run_id)
        transaction = self.db.transaction()

        @self._fs.transactional
        def _release(tx: Any) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                return
            doc = snap.to_dict()
            lease = doc.get("lease")
            if lease and lease.get("owner") == owner:
                tx.update(ref, {"lease": None, "version": doc["version"] + 1})

        try:
            _release(transaction)
        except Exception:  # noqa: BLE001 - lease expiry is the safety net
            log.exception("release_lease failed for %s", run_id)

    def list_runs(self, *, status: str | None = None, limit: int = 100) -> list[Run]:
        from google.cloud.firestore_v1.base_query import FieldFilter

        q = self.runs
        if status:
            q = q.where(filter=FieldFilter("status", "==", status))
        q = q.limit(limit)
        return [Run.from_doc(s.to_dict()) for s in q.stream()]

    # ------------------------------------------------------------- effects
    def effect_get(self, key: str) -> dict[str, Any] | None:
        snap = self.effects.document(_safe_id(key)).get()
        return snap.to_dict().get("value") if snap.exists else None

    def effect_put(self, key: str, value: dict[str, Any]) -> bool:
        from google.api_core import exceptions as gexc

        try:
            self.effects.document(_safe_id(key)).create({"key": key, "value": value})
            return True
        except gexc.AlreadyExists:
            return False

    # -------------------------------------------------------------- fan-in
    def record_child_result(self, parent_run_id: str, child_key: str, result: Any, error: str | None) -> Run:
        ref = self.runs.document(parent_run_id)
        transaction = self.db.transaction()
        holder: dict[str, Run] = {}

        @self._fs.transactional
        def _record(tx: Any) -> None:
            snap = ref.get(transaction=tx)
            if not snap.exists:
                raise ConflictError(f"parent {parent_run_id} does not exist")
            parent = Run.from_doc(snap.to_dict())
            fi = parent.fan_in
            if fi is not None and child_key not in fi.results and child_key not in fi.failures:
                if error is None:
                    fi.results[child_key] = result
                else:
                    fi.failures[child_key] = error
                fi.completed += 1
                parent.version += 1
                tx.set(ref, parent.to_doc())
            holder["run"] = parent

        _record(transaction)
        return holder["run"]


def _safe_id(key: str) -> str:
    # Firestore document ids cannot contain '/', and must be < 1500 bytes.
    return key.replace("/", "_")[:1400]
