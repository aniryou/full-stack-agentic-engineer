"""Runtime configuration.

``LRA_BACKEND=memory`` (default) builds an engine on the in-memory adapters —
what the notebooks, tests and ``scripts/local_demo.py`` use.
``LRA_BACKEND=gcp`` builds the Firestore/Cloud Tasks/Pub/Sub/Gemini engine
used by the Cloud Run services.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

from .core.engine import Engine, WorkflowRegistry
from .core.workflow import Workflow


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


@dataclass
class Settings:
    backend: str = field(default_factory=lambda: _env("LRA_BACKEND", "memory") or "memory")
    project: str = field(default_factory=lambda: _env("GOOGLE_CLOUD_PROJECT", "") or "")
    location: str = field(default_factory=lambda: _env("LRA_LOCATION", "asia-southeast1") or "asia-southeast1")
    tasks_queue: str = field(default_factory=lambda: _env("LRA_TASKS_QUEUE", "agent-steps") or "agent-steps")
    tasks_service_account: str = field(default_factory=lambda: _env("LRA_TASKS_SA", "") or "")
    worker_url: str = field(default_factory=lambda: _env("LRA_WORKER_URL", "") or "")
    firestore_database: str | None = field(default_factory=lambda: _env("LRA_FIRESTORE_DB"))
    gemini_model: str = field(default_factory=lambda: _env("LRA_GEMINI_MODEL", "gemini-3.1-flash-lite") or "gemini-3.1-flash-lite")
    gemini_location: str = field(default_factory=lambda: _env("LRA_GEMINI_LOCATION", "global") or "global")
    lease_ttl_s: int = field(default_factory=lambda: int(_env("LRA_LEASE_TTL_S", "120") or "120"))
    progress_topic: str = field(default_factory=lambda: _env("LRA_PROGRESS_TOPIC", "agent-events") or "agent-events")

    def validate_for_gcp(self) -> None:
        missing = [k for k in ("project", "worker_url", "tasks_service_account") if not getattr(self, k)]
        if missing:
            raise RuntimeError(f"LRA_BACKEND=gcp requires settings: {missing} (see .env.example)")


def build_engine(workflows: Iterable[Workflow] | WorkflowRegistry, settings: Settings | None = None) -> Engine:
    settings = settings or Settings()
    if settings.backend == "gcp":
        settings.validate_for_gcp()
        from .adapters.gcp import build_gcp_engine

        return build_gcp_engine(workflows, settings)

    from .adapters.memory import FakeLLM, InMemoryEventBus, InMemoryStateStore, InMemoryTaskQueue, SystemClock

    clock = SystemClock()
    return Engine(
        store=InMemoryStateStore(),
        queue=InMemoryTaskQueue(clock),
        bus=InMemoryEventBus(),
        llm=FakeLLM(),
        clock=clock,
        workflows=workflows,
    )
