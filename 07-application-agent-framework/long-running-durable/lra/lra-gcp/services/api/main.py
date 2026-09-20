"""Control-plane API (Cloud Run): the public surface for runs.

POST /runs                     start a run (idempotent via ``run_id`` or Idempotency-Key header)
GET  /runs/{run_id}            status, current step, budget, history
POST /runs/{run_id}/events     deliver an external event (approval, webhook) -> resume
POST /runs/{run_id}/cancel     cooperative cancel (compensates if the workflow defines it)
GET  /workflows                registered workflows and versions

Auth: put this behind IAP / API Gateway or require an IAM identity token; the
service itself is workflow-agnostic. Approval links embed ``run_id`` and a
gate key — treat the pair as a capability and keep it unguessable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from common import engine  # noqa: E402
from lra.core.models import Budget, Event  # noqa: E402

app = FastAPI(title="lra-api", version="0.1.0")


class StartRun(BaseModel):
    workflow: str
    input: dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = None
    budget: Budget | None = None


class DeliverEvent(BaseModel):
    key: str
    payload: dict[str, Any] = Field(default_factory=dict)
    event_id: str | None = None


@app.get("/workflows")
def workflows() -> dict[str, Any]:
    reg = engine().registry
    return {"workflows": [{"name": n, "version": reg.latest(n).version, "start": reg.latest(n).start, "steps": list(reg.latest(n).steps)} for n in reg.names()]}


@app.post("/runs", status_code=201)
def start_run(body: StartRun, idempotency_key: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        run = engine().start(body.workflow, body.input, run_id=body.run_id or idempotency_key, budget=body.budget)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return _view(run)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = engine().store.get(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return _view(run, full=True)


@app.post("/runs/{run_id}/events")
def deliver_event(run_id: str, body: DeliverEvent) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"run_id": run_id, "key": body.key, "payload": body.payload}
    if body.event_id:
        kwargs["event_id"] = body.event_id
    try:
        run = engine().resume(Event(**kwargs))
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    if run is None:
        # Not an error: duplicate webhook, or the run is no longer waiting on this key.
        current = engine().store.get(run_id)
        return {"applied": False, "status": current.status.value if current else "unknown"}
    return {"applied": True, **_view(run)}


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    run = engine().cancel(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return _view(run)


def _view(run: Any, full: bool = False) -> dict[str, Any]:
    out = {
        "run_id": run.run_id,
        "workflow": f"{run.workflow}@{run.workflow_version}",
        "status": run.status.value,
        "current_step": run.current_step,
        "wait": run.wait.model_dump(mode="json") if run.wait else None,
        "budget": run.budget.model_dump(mode="json"),
        "error": run.error,
        "result": run.result,
        "version": run.version,
    }
    if full:
        out["state"] = run.state
        out["history"] = [h.model_dump(mode="json") for h in run.history]
        out["fan_in"] = run.fan_in.model_dump(mode="json") if run.fan_in else None
    return out
