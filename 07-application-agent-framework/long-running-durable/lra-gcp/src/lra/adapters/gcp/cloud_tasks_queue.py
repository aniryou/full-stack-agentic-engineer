"""Cloud Tasks-backed :class:`TaskQueue`.

Why Cloud Tasks (not Pub/Sub) for *step dispatch*:

* **Named tasks dedup**: creating a task whose name already exists fails with
  ``AlreadyExists``. We name tasks ``run--kind--step--attempt``, so a crashed
  worker that re-enqueues the same step cannot create a second copy.
* **Scheduled delivery** (``schedule_time``) gives us backoff, timers and
  "wake me in 3 days" for free — no sleeping containers.
* **Rate limiting & concurrency** per queue protect downstream APIs
  (and your model quota) without code.
* **Push with OIDC**: Cloud Tasks mints an identity token for the worker's
  service account; Cloud Run verifies it, so the worker is not publicly
  callable.

Pub/Sub remains the right tool for *fan-out notifications* (progress
events, audit, DLQ analytics) — see :mod:`pubsub_bus`.

Gotchas encoded here:

* Task names are only free ~1h after the task completes/deletes. Attempt
  numbers in the name avoid collisions.
* ``dispatch_deadline`` caps at 30 min; steps longer than that must be split
  or moved to Cloud Run Jobs / Batch.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any

from ...core.models import StepTask

log = logging.getLogger("lra.gcp.tasks")

_ID_RE = re.compile(r"[^A-Za-z0-9_-]")


class CloudTasksQueue:
    def __init__(
        self,
        *,
        project: str,
        location: str,
        queue: str,
        target_url: str,
        service_account_email: str,
        audience: str | None = None,
        dispatch_deadline_s: int = 1800,
        client: Any | None = None,
    ) -> None:
        from google.cloud import tasks_v2  # lazy import

        self._tasks = tasks_v2
        self.client = client or tasks_v2.CloudTasksClient()
        self.queue_path = self.client.queue_path(project, location, queue)
        self.project, self.location, self.queue = project, location, queue
        self.target_url = target_url
        self.service_account_email = service_account_email
        self.audience = audience or target_url
        self.dispatch_deadline_s = dispatch_deadline_s

    def task_name(self, task: StepTask) -> str:
        task_id = _ID_RE.sub("-", task.dedup_key)[:500]
        return self.client.task_path(self.project, self.location, self.queue, task_id)

    def build_request(self, task: StepTask, delay: timedelta | None = None) -> dict[str, Any]:
        """Pure function returning the CreateTask body (unit-testable without GCP)."""
        from google.protobuf import duration_pb2, timestamp_pb2

        body: dict[str, Any] = {
            "name": self.task_name(task),
            "http_request": {
                "http_method": self._tasks.HttpMethod.POST,
                "url": self.target_url,
                "headers": {"Content-Type": "application/json"},
                "body": task.model_dump_json().encode("utf-8"),
                "oidc_token": {
                    "service_account_email": self.service_account_email,
                    "audience": self.audience,
                },
            },
            "dispatch_deadline": duration_pb2.Duration(seconds=self.dispatch_deadline_s),
        }
        if delay and delay.total_seconds() > 0:
            from datetime import datetime, timezone

            ts = timestamp_pb2.Timestamp()
            ts.FromDatetime(datetime.now(timezone.utc) + delay)
            body["schedule_time"] = ts
        return body

    def enqueue(self, task: StepTask, *, delay: timedelta | None = None) -> bool:
        from google.api_core import exceptions as gexc

        req = self.build_request(task, delay)
        try:
            self.client.create_task(parent=self.queue_path, task=req)
            return True
        except gexc.AlreadyExists:
            log.info("task %s already queued; dedup", task.dedup_key)
            return False
