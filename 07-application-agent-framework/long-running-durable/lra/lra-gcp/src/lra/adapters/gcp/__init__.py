"""Google Cloud adapters and the factory that wires them together.

Pattern -> service mapping (see docs/primer.md §3 for the full table):

=====================  =============================================
Concern                Service
=====================  =============================================
Run state/checkpoints  Firestore (transactions, optimistic concurrency)
Step dispatch/timers   Cloud Tasks (named-task dedup, schedule_time, OIDC)
Progress/alerts        Pub/Sub (+ dead-letter topic)
Model                  Gemini on Vertex AI (google-genai)
Compute                Cloud Run services (api, worker); Jobs for >30 min steps
Repair cron            Cloud Scheduler -> worker /internal/reap
Fixed control flow     Cloud Workflows (see workflows/)
Managed agent runtime  Agent Engine + ADK (see examples/adk_agent_engine)
=====================  =============================================
"""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable

from ...core.engine import Engine, WorkflowRegistry
from ...core.workflow import Workflow
from ..memory import SystemClock
from .cloud_tasks_queue import CloudTasksQueue
from .firestore_store import FirestoreStateStore
from .gemini_llm import GeminiLLM
from .pubsub_bus import PubSubEventBus

__all__ = ["CloudTasksQueue", "FirestoreStateStore", "GeminiLLM", "PubSubEventBus", "build_gcp_engine"]


def build_gcp_engine(workflows: Iterable[Workflow] | WorkflowRegistry, settings: "Settings") -> Engine:  # noqa: F821
    """Assemble a production Engine from :class:`lra.config.Settings`."""
    store = FirestoreStateStore(project=settings.project, database=settings.firestore_database)
    queue = CloudTasksQueue(
        project=settings.project,
        location=settings.location,
        queue=settings.tasks_queue,
        target_url=f"{settings.worker_url.rstrip('/')}/tasks/step",
        service_account_email=settings.tasks_service_account,
        audience=settings.worker_url,
    )
    bus = PubSubEventBus(project=settings.project)
    llm = GeminiLLM(project=settings.project, location=settings.gemini_location, model=settings.gemini_model)
    return Engine(
        store=store,
        queue=queue,
        bus=bus,
        llm=llm,
        clock=SystemClock(),
        workflows=workflows,
        lease_ttl=timedelta(seconds=settings.lease_ttl_s),
    )
