"""Observability helpers.

Long-running agents fail in slow, quiet ways (a run parked WAITING for a
webhook that never comes; a reflection loop burning budget). The signals
that catch those are *per-run* and *per-step*, not per-request:

* Structured logs with ``run_id``, ``step``, ``attempt`` on every line so
  Cloud Logging can group a run's story across replicas and days.
* A span per step (``lra.step``) with token/cost attributes, exported to
  Cloud Trace, so a run reads as one trace even when it spans weeks.
* Metrics derived from the ``agent-events`` topic (steps/min, retries,
  waits > SLA, cost per run) via a Pub/Sub -> BigQuery subscription.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator


class CloudLoggingJsonFormatter(logging.Formatter):
    """Emits one JSON object per line in the shape Cloud Logging parses natively."""

    LEVEL_MAP = {"DEBUG": "DEBUG", "INFO": "INFO", "WARNING": "WARNING", "ERROR": "ERROR", "CRITICAL": "CRITICAL"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "severity": self.LEVEL_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
        }
        for key in ("run_id", "step", "attempt", "workflow", "worker"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        trace_id = getattr(record, "trace_id", None)
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if trace_id and project:
            payload["logging.googleapis.com/trace"] = f"projects/{project}/traces/{trace_id}"
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingJsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def setup_tracing(service_name: str = "lra-worker") -> Any:
    """Configure OpenTelemetry -> Cloud Trace if the exporter is installed; no-op otherwise."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # pragma: no cover
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    try:
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

        provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
    except Exception:  # noqa: BLE001 - exporter missing or no credentials: keep local spans
        pass
    trace.set_tracer_provider(provider)
    return trace.get_tracer("lra")


@contextmanager
def step_span(tracer: Any, run_id: str, step: str, attempt: int) -> Iterator[Any]:
    """Wrap one step execution in a span; safe when tracing is disabled."""
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span("lra.step") as span:
        span.set_attribute("lra.run_id", run_id)
        span.set_attribute("lra.step", step)
        span.set_attribute("lra.attempt", attempt)
        yield span
