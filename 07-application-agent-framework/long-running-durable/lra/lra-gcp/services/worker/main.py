"""Worker service (Cloud Run): executes one step per request.

Endpoints
---------
POST /tasks/step      Cloud Tasks HTTP target. Body = StepTask JSON.
POST /internal/reap   Cloud Scheduler target (every 1–5 min). Repairs stuck runs.
POST /pubsub/push     Optional: Pub/Sub push subscription on ``agent-events``
                      (e.g. to mirror progress into a UI store). Demonstrates
                      the envelope handling; not required by the engine.
GET  /healthz

Scaling notes: set ``--concurrency`` low (4–8) because a step may hold a
model call open for tens of seconds; use ``--timeout`` >= the longest step
and <= 30 min (Cloud Tasks dispatch deadline). Steps longer than that
belong in Cloud Run Jobs, triggered from a step that returns ``Wait`` and is
resumed by the job's completion callback.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `common` importable in the container

from fastapi import Depends, FastAPI, Request, Response  # noqa: E402

from common import OUTCOME_STATUS, decode_pubsub_push, engine, verify_cloud_tasks  # noqa: E402
from lra.core.models import StepTask  # noqa: E402

log = logging.getLogger("lra.worker")
app = FastAPI(title="lra-worker", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tasks/step", dependencies=[Depends(verify_cloud_tasks)])
def run_step(task: StepTask, response: Response) -> dict[str, str]:
    outcome = engine().execute_task(task)
    response.status_code = OUTCOME_STATUS.get(outcome, 500)
    log.info("step %s -> %s", task.dedup_key, outcome, extra={"run_id": task.run_id, "step": task.step, "attempt": task.attempt})
    return {"outcome": outcome}


@app.post("/internal/reap", dependencies=[Depends(verify_cloud_tasks)])
def reap() -> dict[str, list[str]]:
    return engine().reap()


@app.post("/pubsub/push")
async def pubsub_push(request: Request) -> dict[str, str]:
    payload, attrs = decode_pubsub_push(await request.json())
    # Always ack (2xx) unless you *want* redelivery: a poison message on a
    # push subscription with a dead-letter topic will land in the DLQ after
    # max_delivery_attempts.
    log.info("event %s for run %s", attrs.get("type"), attrs.get("run_id"), extra={"run_id": attrs.get("run_id")})
    return {"status": "ok", "type": payload.get("type", "")}
