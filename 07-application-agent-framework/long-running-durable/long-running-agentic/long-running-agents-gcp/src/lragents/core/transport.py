"""How a run gets its next wake-up.

The agent never sleeps in-process. It checkpoints, then asks a *dispatcher* to
wake it later. On GCP that is:

* **Cloud Tasks** for "run step N of run X (optionally not before time T)":
  per-task scheduling, retries with backoff, rate limiting, and *named-task
  de-duplication* so a double-enqueue is rejected instead of double-run.
* **Pub/Sub** for fan-out ("here are 40 subtasks") and for decoupling schedulers
  from the service (Cloud Scheduler → topic → push subscription → Cloud Run).

Both are at-least-once. The in-memory dispatcher reproduces the two behaviours
that bite in production — duplicate delivery and retry-on-failure — so the tests
can prove the handlers are idempotent.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Envelope:
    run_id: str
    step_index: int
    kind: str = "step"                       # step | poll | subtask | aggregate | tick | resume
    payload: dict[str, Any] = field(default_factory=dict)
    not_before: float = 0.0                  # epoch seconds; 0 = now
    attempt: int = 0

    @property
    def name(self) -> str:
        """Stable task name → de-duplication key."""
        extra = "-".join(str(self.payload[k]) for k in ("subtask_id", "attempt") if k in self.payload)
        raw = f"{self.run_id}-{self.kind}-{self.step_index}-{extra}".rstrip("-")
        return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:500]

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str | bytes) -> "Envelope":
        return cls(**json.loads(s))


class Dispatcher(Protocol):
    def enqueue(self, env: Envelope) -> str: ...


class DuplicateTask(Exception):
    pass


class InMemoryDispatcher:
    """Deterministic queue with Cloud-Tasks-like semantics.

    * named-task de-duplication (``enqueue`` of a seen name is a no-op)
    * ``not_before`` scheduling against an injectable clock
    * retry with attempt counting when the handler raises
    * ``duplicate_next`` to inject an at-least-once redelivery
    """

    def __init__(self, clock: Callable[[], float] = time.time, max_attempts: int = 5) -> None:
        self.clock = clock
        self.max_attempts = max_attempts
        self.queue: list[Envelope] = []
        self.seen_names: set[str] = set()
        self.dead_letter: list[tuple[Envelope, str]] = []
        self.delivered: list[Envelope] = []
        self._dup_next = False

    def enqueue(self, env: Envelope) -> str:
        if env.name in self.seen_names:
            return f"dedup:{env.name}"
        self.seen_names.add(env.name)
        self.queue.append(env)
        return env.name

    def duplicate_next(self) -> None:
        self._dup_next = True

    def pending(self) -> int:
        return len(self.queue)

    def deliver_one(self, handler: Callable[[Envelope], Any]) -> bool:
        now = self.clock()
        for i, env in enumerate(self.queue):
            if env.not_before <= now:
                self.queue.pop(i)
                break
        else:
            return False
        copies = 2 if self._dup_next else 1
        self._dup_next = False
        for _ in range(copies):
            self.delivered.append(env)
            try:
                handler(env)
            except Exception as e:  # noqa: BLE001 — mimic the platform: any failure → retry
                env.attempt += 1
                if env.attempt >= self.max_attempts:
                    self.dead_letter.append((env, repr(e)))
                else:
                    self.queue.append(env)       # retry (no backoff in the fake; the clock is fake anyway)
        return True

    def drain(self, handler: Callable[[Envelope], Any], max_deliveries: int = 200) -> int:
        n = 0
        while n < max_deliveries and self.deliver_one(handler):
            n += 1
        return n


class CloudTasksDispatcher:
    """Wake the Cloud Run service via an HTTP task with an OIDC identity token."""

    def __init__(self, project: str, location: str, queue: str, target_url: str, service_account_email: str, dispatch_deadline_s: int = 1800) -> None:
        from google.cloud import tasks_v2

        self._tasks_v2 = tasks_v2
        self.client = tasks_v2.CloudTasksClient()
        self.parent = self.client.queue_path(project, location, queue)
        self.project, self.location, self.queue = project, location, queue
        self.target_url = target_url
        self.sa = service_account_email
        self.dispatch_deadline_s = dispatch_deadline_s

    def enqueue(self, env: Envelope) -> str:
        from google.api_core.exceptions import AlreadyExists
        from google.protobuf import duration_pb2, timestamp_pb2

        t = self._tasks_v2
        task = t.Task(
            name=self.client.task_path(self.project, self.location, self.queue, env.name),
            http_request=t.HttpRequest(
                http_method=t.HttpMethod.POST,
                url=f"{self.target_url}/internal/tasks/{env.kind}",
                headers={"Content-Type": "application/json"},
                body=env.to_json().encode(),
                oidc_token=t.OidcToken(service_account_email=self.sa, audience=self.target_url),
            ),
            dispatch_deadline=duration_pb2.Duration(seconds=self.dispatch_deadline_s),
        )
        if env.not_before > time.time():
            task.schedule_time = timestamp_pb2.Timestamp(seconds=int(env.not_before))
        try:
            created = self.client.create_task(request=t.CreateTaskRequest(parent=self.parent, task=task))
            return created.name
        except AlreadyExists:
            return f"dedup:{env.name}"          # de-dup window hit: already scheduled


class PubSubDispatcher:
    """Publish envelopes to a topic; a push subscription delivers them to Cloud Run."""

    def __init__(self, project: str, topic: str, ordering: bool = False) -> None:
        from google.cloud import pubsub_v1

        opts = pubsub_v1.types.PublisherOptions(enable_message_ordering=ordering)
        self.publisher = pubsub_v1.PublisherClient(publisher_options=opts)
        self.topic_path = self.publisher.topic_path(project, topic)
        self.ordering = ordering

    def enqueue(self, env: Envelope) -> str:
        kwargs: dict[str, Any] = {"kind": env.kind, "run_id": env.run_id, "step_index": str(env.step_index)}
        if self.ordering:
            kwargs["ordering_key"] = env.run_id
        future = self.publisher.publish(self.topic_path, env.to_json().encode(), **kwargs)
        return future.result(timeout=30)


def decode_pubsub_push(body: dict[str, Any]) -> Envelope:
    """Unwrap the Pub/Sub push envelope {"message": {"data": base64, "attributes": {...}}}."""
    data = base64.b64decode(body["message"]["data"]).decode()
    return Envelope.from_json(data)
