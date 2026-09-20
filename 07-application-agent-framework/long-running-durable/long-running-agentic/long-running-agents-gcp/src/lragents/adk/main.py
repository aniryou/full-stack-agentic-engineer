"""Cloud Run entry point for the ADK agent: ADK's FastAPI app + a wake endpoint.

* ``get_fast_api_app`` serves the standard ADK API (sessions, /run, /run_sse)
  and, with ``trigger_sources=["pubsub"]``, a push endpoint for Pub/Sub messages.
* ADK's built-in Pub/Sub trigger starts a *new* session per message. A
  long-running run must wake an *existing* paused session, so ``/wake`` builds
  a ``Runner`` on the same session database and resumes the paused invocation
  with a ``FunctionResponse`` for the pending interrupt. Point
  Cloud Scheduler → Pub/Sub → push subscription at ``/wake``.

Env: SESSION_SERVICE_URI (e.g. postgresql+asyncpg://user:pw@/adk?host=/cloudsql/PROJ:REGION:INST),
     ARTIFACT_SERVICE_URI (gs://bucket/artifacts), PORT, GOOGLE_CLOUD_PROJECT.
"""

from __future__ import annotations

import base64
import json
import os

import uvicorn
from fastapi import Body
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_URI = os.environ.get("SESSION_SERVICE_URI")

app = get_fast_api_app(
    agents_dir=os.path.dirname(HERE),                  # the folder that CONTAINS the `adk` agent package
    session_service_uri=SESSION_URI,
    artifact_service_uri=os.environ.get("ARTIFACT_SERVICE_URI"),
    web=False,                                         # never ship the dev UI
    trigger_sources=["pubsub"],
    trace_to_cloud=bool(os.environ.get("GOOGLE_CLOUD_PROJECT")),
)


def _session_service():
    if SESSION_URI:
        from google.adk.sessions import DatabaseSessionService   # needs google-adk[db]
        return DatabaseSessionService(db_url=SESSION_URI)
    return InMemorySessionService()


_runner: Runner | None = None


def get_runner() -> Runner:
    global _runner
    if _runner is None:
        from .agent import app as adk_app                     # same App object the ADK API serves
        _runner = Runner(app=adk_app, session_service=_session_service())
    return _runner


@app.post("/wake")
async def wake(payload: dict = Body(default={})):
    """Resume a paused session.

    Body: {"user_id", "session_id", "invocation_id", "interrupt_id", "response"},
    or a Pub/Sub push envelope whose base64 ``data`` is that JSON.
    Idempotent: resuming an invocation that already completed yields no new work.
    """
    if "message" in payload:                                  # Pub/Sub push envelope
        payload = json.loads(base64.b64decode(payload["message"]["data"]).decode())
    msg = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
        id=payload["interrupt_id"], name="adk_request_input", response=payload.get("response", {"ok": True})))])
    n, interrupts = 0, []
    async for ev in get_runner().run_async(user_id=payload["user_id"], session_id=payload["session_id"],
                                           invocation_id=payload.get("invocation_id"), new_message=msg):
        n += 1
        for p in (ev.content.parts if ev.content else []) or []:
            if p.function_call and p.function_call.name == "adk_request_input":
                interrupts.append(p.function_call.id)
    return {"resumed": True, "events": n, "parked_on": interrupts}


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
