"""Cloud Run service exposing the patterns over HTTP.

Wire-up is chosen by ``LRAGENTS_BACKEND``:

* ``memory`` (default) – in-memory store/dispatcher + ScriptedLLM; the test suite
  and notebooks drive the in-memory dispatcher by POSTing envelopes back at the
  handlers, exactly as Cloud Tasks would.
* ``gcp`` – Firestore + Cloud Tasks + Gemini on Vertex AI.

Routes
  POST /runs                         start a durable-loop run           → 202 {run_id}
  GET  /runs/{id}                    inspect (status, journal, usage)
  POST /runs/{id}/approve            human decision {token, approved, approver, comment}
  POST /internal/tasks/step          Cloud Tasks target: advance one step  (Envelope JSON)
  POST /internal/tasks/poll          Cloud Tasks target: poll an async tool
  POST /internal/pubsub/push         Pub/Sub push: subtask / aggregate / tick envelopes
  POST /internal/callbacks/{id}/{ticket}  external system webhook (async tool done)
  POST /internal/scheduler/tick      Cloud Scheduler: expire stale approvals, heartbeat agents
  GET  /healthz

Internal routes must only be reachable with a valid OIDC token minted for the
service (Cloud Tasks / Pub/Sub push with a service account; ``--no-allow-unauthenticated``
on Cloud Run). ``verify_oidc`` is enforced when LRAGENTS_BACKEND=gcp.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from ..core import (
    Budget,
    Decision,
    FaultInjector,
    InMemoryDispatcher,
    InMemoryIdempotencyStore,
    InMemoryRunStore,
    LeaseHeld,
    PaymentGateway,
    PriceCard,
    RunNotFound,
    ScriptedLLM,
    Tool,
    ToolRegistry,
    decode_pubsub_push,
)
from ..core.transport import Envelope
from ..patterns import ApprovalError, DurableAgentLoop, approve, expire_stale_approvals


# ----------------------------------------------------------------- wiring
@dataclass
class Services:
    loop: DurableAgentLoop
    dispatcher: Any
    gateway: PaymentGateway
    backend: str


def default_tools(gateway: PaymentGateway) -> ToolRegistry:
    def lookup(args, ctx):
        return {"price": 42.0, "sku": args.get("sku", "?")}

    def charge_card(args, ctx):
        return gateway.charge(float(args["amount"]), idempotency_key=ctx.idempotency_key)

    return ToolRegistry([
        Tool("lookup_price", "Look up the price of a SKU.", lookup,
             {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}),
        Tool("charge_card", "Charge the customer's card. Requires human approval.", charge_card,
             {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]},
             requires_approval=True),
    ])


def build_services(backend: str | None = None) -> Services:
    backend = backend or os.environ.get("LRAGENTS_BACKEND", "memory")
    gateway = PaymentGateway()
    tools = default_tools(gateway)
    if backend == "gcp":
        from ..core.firestore_store import FirestoreIdempotencyStore, FirestoreRunStore
        from ..core.llm import GeminiLLM
        from ..core.transport import CloudTasksDispatcher

        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        dispatcher = CloudTasksDispatcher(project, location, os.environ.get("TASKS_QUEUE", "agent-steps"),
                                          os.environ["SERVICE_URL"], os.environ["TASKS_INVOKER_SA"])
        loop = DurableAgentLoop(store=FirestoreRunStore(project=project), llm=GeminiLLM(), tools=tools,
                                dispatcher=dispatcher, idempotency=FirestoreIdempotencyStore(),
                                price=PriceCard(float(os.environ.get("PRICE_IN_PER_M", 0)), float(os.environ.get("PRICE_OUT_PER_M", 0))),
                                worker_id=os.environ.get("K_REVISION", "cloud-run"))
    else:
        dispatcher = InMemoryDispatcher()
        llm = ScriptedLLM([Decision.call("lookup_price", sku="ABC"), Decision.call("charge_card", amount=42.0),
                           Decision.final("Charged 42.00 for ABC.")])
        loop = DurableAgentLoop(store=InMemoryRunStore(), llm=llm, tools=tools, dispatcher=dispatcher,
                                idempotency=InMemoryIdempotencyStore(), faults=FaultInjector())
    return Services(loop=loop, dispatcher=dispatcher, gateway=gateway, backend=backend)


# ------------------------------------------------------------------- auth
async def verify_oidc(request: Request) -> None:
    """Validate the Google-signed identity token Cloud Tasks / Pub/Sub attach."""
    if os.environ.get("LRAGENTS_BACKEND", "memory") != "gcp":
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        from google.auth.transport import requests as ga_requests
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(auth[7:], ga_requests.Request(), audience=os.environ["SERVICE_URL"])
        if claims.get("email") not in {os.environ.get("TASKS_INVOKER_SA"), os.environ.get("PUBSUB_INVOKER_SA")}:
            raise HTTPException(403, "wrong service account")
    except ValueError as e:
        raise HTTPException(401, f"invalid token: {e}") from e


# ------------------------------------------------------------------ app
class StartRun(BaseModel):
    goal: str
    max_steps: int = 25
    max_cost_usd: float = 2.0


class Approval(BaseModel):
    token: str
    approved: bool
    approver: str
    comment: str = ""


def create_app(services: Services | None = None) -> FastAPI:
    svc = services or build_services()
    app = FastAPI(title="lragents", version="0.1.0")
    app.state.services = svc

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "backend": svc.backend}

    @app.post("/runs", status_code=202)
    def start(body: StartRun):
        run = svc.loop.start(body.goal, Budget(max_steps=body.max_steps, max_cost_usd=body.max_cost_usd))
        return {"run_id": run.run_id, "status": run.status}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return svc.loop.store.get(run_id).to_dict()
        except RunNotFound:
            raise HTTPException(404, "no such run")

    @app.post("/runs/{run_id}/approve")
    def approve_run(run_id: str, body: Approval):
        try:
            run = approve(svc.loop, run_id, body.token, body.approved, body.approver, body.comment)
        except RunNotFound:
            raise HTTPException(404, "no such run")
        except ApprovalError as e:
            raise HTTPException(403, str(e))
        return {"run_id": run.run_id, "status": run.status}

    @app.post("/internal/tasks/{kind}", dependencies=[Depends(verify_oidc)])
    def task(kind: str, env: dict, request: Request):
        """Return 2xx only when the step is durably done; any exception → 5xx → Cloud Tasks retries."""
        envelope = Envelope(**env)
        try:
            run = svc.loop.handle(envelope)
        except LeaseHeld as e:                                 # another worker is on it: ask for a retry later
            raise HTTPException(status_code=429, detail=str(e))
        return {"run_id": run.run_id, "status": run.status, "next_index": run.next_index(),
                "retry_count": request.headers.get("X-CloudTasks-TaskRetryCount")}

    @app.post("/internal/pubsub/push", dependencies=[Depends(verify_oidc)])
    def pubsub(body: dict):
        envelope = decode_pubsub_push(body)
        run = svc.loop.handle(envelope)
        return {"run_id": run.run_id, "status": run.status}

    @app.post("/internal/callbacks/{run_id}/{ticket}")
    def callback(run_id: str, ticket: str, body: dict):
        run = svc.loop.resume_with_event(run_id, ticket, body.get("result"))
        return {"run_id": run.run_id, "status": run.status}

    @app.post("/internal/scheduler/tick", dependencies=[Depends(verify_oidc)])
    def tick():
        expired = expire_stale_approvals(svc.loop, ttl_s=float(os.environ.get("APPROVAL_TTL_S", 72 * 3600)))
        return {"expired": [r.run_id for r in expired]}

    return app


app = create_app() if os.environ.get("LRAGENTS_AUTOCREATE", "0") == "1" else None

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
