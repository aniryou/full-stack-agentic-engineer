"""Shared plumbing for the Cloud Run services.

Both services are thin: all behaviour lives in ``lra.core.engine``. What they
add is the *boundary* concerns of running on Cloud Run:

* Building one Engine per process from environment settings.
* Verifying that step tasks really come from Cloud Tasks (OIDC token minted
  for our service account) and that Pub/Sub pushes come from our subscription.
* Mapping engine outcomes to HTTP codes Cloud Tasks understands
  (2xx = ack, 5xx = retry with the queue's backoff).
"""

from __future__ import annotations

import base64
import json
import logging
import os
from functools import lru_cache
from typing import Any

from fastapi import Header, HTTPException, Request

from lra.config import Settings, build_engine
from lra.core.engine import Engine
from lra.examples import ALL_WORKFLOWS
from lra.observability import setup_logging, setup_tracing

log = logging.getLogger("lra.services")


@lru_cache(maxsize=1)
def engine() -> Engine:
    settings = Settings()
    if settings.backend == "gcp":
        setup_logging()
        setup_tracing(os.environ.get("K_SERVICE", "lra"))
    return build_engine(ALL_WORKFLOWS, settings)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


# --- Cloud Tasks -> worker authentication -----------------------------------
async def verify_cloud_tasks(
    request: Request,
    authorization: str | None = Header(default=None),
    x_cloudtasks_queuename: str | None = Header(default=None),
) -> None:
    """Reject anything that is not an OIDC token for our audience from our SA.

    Cloud Run (with ingress=internal + require-auth) already checks the token
    signature and audience at the edge; we re-verify here so the service is
    safe even if someone deploys it with ``--allow-unauthenticated``.
    """
    s = settings()
    if s.backend != "gcp":
        return  # local mode: no IAM in the loop
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        from google.auth.transport import requests as ga_requests
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(token, ga_requests.Request(), audience=s.worker_url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(401, f"invalid token: {e}") from e
    if claims.get("email") != s.tasks_service_account:
        raise HTTPException(403, "token is not for the tasks service account")
    if not x_cloudtasks_queuename:
        raise HTTPException(403, "not a Cloud Tasks request")


def decode_pubsub_push(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Unwrap a Pub/Sub push envelope -> (payload, attributes)."""
    msg = body.get("message") or {}
    data = msg.get("data")
    payload = json.loads(base64.b64decode(data).decode("utf-8")) if data else {}
    return payload, dict(msg.get("attributes") or {})


OUTCOME_STATUS = {
    # 2xx: Cloud Tasks deletes the task. Everything the engine handled itself is an ack.
    "ok": 200, "done": 200, "waiting": 200, "fanout": 200, "retry": 200, "failed": 200,
    "compensating": 200, "compensated": 200, "budget": 200, "cancelled": 200, "stale": 200,
    "unknown-run": 200, "version-missing": 200,
    # 503: Cloud Tasks retries per queue RetryConfig — the lease holder will finish or expire.
    "lease-held": 503,
}
