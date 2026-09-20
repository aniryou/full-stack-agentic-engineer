"""A tickets MCP server that is a proper OAuth 2.1 **resource server**.

What "proper" means here (MCP authorization spec, 2025-11-25 / 2026-07-28):

* Publishes **Protected Resource Metadata** (RFC 9728) at
  ``/.well-known/oauth-protected-resource[/mcp]`` naming its authorization server, and returns
  ``401`` with ``WWW-Authenticate: Bearer resource_metadata="…"`` when no/invalid token is
  presented — the ``mcp`` SDK does this once ``token_verifier`` + ``AuthSettings`` are set.
* **Validates audience**: a token is accepted only if it was issued *for this server*
  (``aud`` == canonical server URI, the RFC 8707 ``resource``). Tokens minted for another API
  are rejected even though they are signed by the same issuer.
* **Scopes → tools**: ``tickets:read`` for read-only tools, ``tickets:write`` for
  ``refund_ticket``; tool annotations (``readOnlyHint``/``destructiveHint``) tell clients and
  gateways the same thing (VPC-SC's ``mcp.tool.isReadOnly`` conditions key off this).
* **No token passthrough**: when the server calls an upstream API it obtains a *separate*
  token for that audience; the inbound token never leaves this process.
* **Delegation-aware**: if the inbound token is delegated (``act`` claim), tools scope their
  data to the user (``sub``) and the audit record carries both identities.
* Optional **DPoP** (RFC 9449): with ``require_dpop=True`` the server insists on
  ``Authorization: DPoP <token>`` + a ``DPoP`` proof bound to the token's ``cnf.jkt``.

Run locally with :func:`serve` (uvicorn) or embed via :func:`build_app`.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..audit.log import AuditLog
from ..identity.tokens import DPoP, TokenError, TokenIssuer

SCOPE_READ = "tickets:read"
SCOPE_WRITE = "tickets:write"

_presented_jkt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentsec_presented_jkt", default=None
)


class DelegatedAccessToken(AccessToken):
    """The SDK's AccessToken plus who is acting for whom."""

    subject: str
    actor: str | None = None
    authority: str = "own"


@dataclass
class JwtTokenVerifier:
    """``TokenVerifier`` for the ``mcp`` SDK: signature, issuer, expiry, **audience**, binding."""

    issuer: TokenIssuer
    audience: str
    require_dpop: bool = False
    audit: AuditLog | None = None

    async def verify_token(self, token: str) -> AccessToken | None:
        jkt = _presented_jkt.get()
        try:
            claims = self.issuer.verify(
                token,
                audience=self.audience,
                presented_jkt=jkt,
                allow_unbound=False,
            )
        except TokenError as e:
            if self.audit:
                self.audit.record(
                    event_type="mcp.auth",
                    agent="mcp-server",
                    authority="n/a",
                    decision="deny",
                    reasons=[type(e).__name__, str(e)],
                )
            return None
        if self.require_dpop and not jkt:
            return None
        return DelegatedAccessToken(
            token=token,
            client_id=claims.actor or claims.subject,
            scopes=sorted(claims.scopes),
            expires_at=int(claims.raw["exp"]),
            resource=self.audience,
            subject=claims.subject,
            actor=claims.actor,
            authority="delegated" if claims.is_delegated else "own",
        )


class DPoPMiddleware:
    """Validate ``Authorization: DPoP`` + ``DPoP`` proof, then hand a Bearer token to the SDK.

    The verified key thumbprint is published through a context variable so the
    :class:`JwtTokenVerifier` can require ``cnf.jkt`` to match. A DPoP-bound token presented as
    a plain Bearer (no proof) therefore fails verification — exactly what prevents replay.
    """

    def __init__(self, app: Any, *, require: bool = False, proof_max_age: int = 300):
        self.app = app
        self.require = require
        self.proof_max_age = proof_max_age
        # jti -> first-seen time. Proofs older than proof_max_age are rejected by DPoP.verify
        # anyway, so entries can be pruned after that window; this keeps the cache bounded.
        self._seen_jti: dict[str, float] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        if len(self._seen_jti) > 10_000 or (
            self._seen_jti and now - min(self._seen_jti.values()) > self.proof_max_age
        ):
            self._seen_jti = {
                j: t for j, t in self._seen_jti.items() if now - t <= self.proof_max_age
            }

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        proof = headers.get("dpop")
        token_var = _presented_jkt.set(None)
        try:
            if auth.lower().startswith("dpop ") and proof:
                token = auth[5:].strip()
                url = self._request_url(scope, headers)
                try:
                    with self._lock:
                        now = time.time()
                        self._prune(now)
                        seen = set(self._seen_jti)
                        claims = DPoP.verify(
                            proof,
                            method=scope["method"],
                            url=url,
                            access_token=token,
                            seen_jti=seen,
                            max_age=self.proof_max_age,
                        )
                        self._seen_jti[claims["jti"]] = now
                except TokenError as e:
                    await self._reject(send, f"invalid_dpop_proof: {e}")
                    return
                _presented_jkt.set(claims["jkt"])
                # Re-present as Bearer for the SDK's BearerAuthBackend.
                scope = dict(scope)
                scope["headers"] = [
                    (k, v) for k, v in scope["headers"] if k.lower() != b"authorization"
                ] + [(b"authorization", f"Bearer {token}".encode())]
            elif self.require and scope.get("path", "").endswith("/mcp"):
                await self._reject(send, "dpop_required")
                return
            await self.app(scope, receive, send)
        finally:
            _presented_jkt.reset(token_var)

    @staticmethod
    def _request_url(scope: dict[str, Any], headers: dict[str, str]) -> str:
        host = headers.get("host", "localhost")
        scheme = scope.get("scheme", "http")
        return f"{scheme}://{host}{scope.get('path', '/')}"

    @staticmethod
    async def _reject(send: Any, reason: str) -> None:
        body = json.dumps({"error": "invalid_token", "error_description": reason}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (
                        b"www-authenticate",
                        f'DPoP error="invalid_token", error_description="{reason}"'.encode(),
                    ),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


# ---- domain -----------------------------------------------------------------------------------
TICKETS: dict[str, dict[str, Any]] = {
    "T-1": {"owner": "u-ana", "event": "Jazz Festival", "price": 60.0, "status": "valid"},
    "T-2": {"owner": "u-ana", "event": "Museum pass", "price": 35.0, "status": "valid"},
    "T-3": {"owner": "u-ben", "event": "F1 grandstand", "price": 120.0, "status": "valid"},
}


@dataclass
class UpstreamPaymentsClient:
    """Illustrates *no token passthrough*: the MCP server is an OAuth client to the upstream API.

    It obtains a token for the upstream **audience** under its **own** identity (``subject`` is
    the MCP server's SPIFFE ID), recording the calling agent as ``act`` and the end user as
    ``on_behalf_of`` — a fresh delegation hop, never the inbound token forwarded. Locally the
    token is minted by the shared STS; on Google Cloud the server would run a client-credentials
    or token-exchange grant against the upstream authorization server with its Agent Identity.
    """

    issuer: TokenIssuer
    audience: str = "https://payments.acme.example"
    server_identity: str = (
        "spiffe://agents.global.org-123456789012.system.id.goog/resources/run/"
        "projects/987654321098/locations/us-central1/services/tickets-mcp"
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    def refund(
        self, *, ticket_id: str, amount: float, on_behalf_of: str, actor: str
    ) -> dict[str, Any]:
        upstream_token = self.issuer.mint(
            subject=self.server_identity,
            audience=self.audience,
            scope="payments:refund",
            ttl=60,
            extra={"act": {"sub": actor}, "on_behalf_of": on_behalf_of, "authority": "delegated"},
        )
        self.calls.append(
            {
                "ticket_id": ticket_id,
                "amount": amount,
                "aud": self.audience,
                "subject": self.server_identity,
                "token": upstream_token,
            }
        )
        return {"payment_ref": f"P-{int(time.time())}", "amount": amount}


def free_port(host: str = "127.0.0.1") -> int:
    import socket

    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@dataclass
class TicketsServer:
    mcp: FastMCP
    verifier: JwtTokenVerifier
    audit: AuditLog
    payments: UpstreamPaymentsClient
    resource_url: str

    def app(self, *, require_dpop: bool = False) -> Any:
        return build_app(self.mcp, require_dpop=require_dpop)


def build_server(
    issuer: TokenIssuer,
    *,
    resource_url: str,
    audit: AuditLog | None = None,
    require_dpop: bool = False,
    payments: UpstreamPaymentsClient | None = None,
) -> TicketsServer:
    """Build the tickets MCP server bound to ``resource_url`` (its canonical URI / audience)."""
    audit = audit or AuditLog()
    payments = payments or UpstreamPaymentsClient(issuer)
    verifier = JwtTokenVerifier(
        issuer=issuer, audience=resource_url, require_dpop=require_dpop, audit=audit
    )
    mcp = FastMCP(
        "tickets",
        instructions="Ticket lookup and refunds for Acme Tickets. Requires tickets:read / tickets:write.",
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=issuer.issuer,  # type: ignore[arg-type]
            resource_server_url=resource_url,  # type: ignore[arg-type]
            required_scopes=[SCOPE_READ],
        ),
        stateless_http=True,
        json_response=True,
    )

    def _caller() -> DelegatedAccessToken:
        tok = get_access_token()
        if not isinstance(tok, DelegatedAccessToken):
            raise PermissionError("unauthenticated")
        return tok

    def _require(scope: str) -> DelegatedAccessToken:
        tok = _caller()
        if scope not in tok.scopes:
            audit.record(
                event_type="mcp.tool",
                agent=tok.actor or tok.subject,
                user=tok.subject if tok.actor else None,
                authority=tok.authority,
                decision="deny",
                reasons=[f"insufficient_scope: {scope}"],
            )
            raise PermissionError(f"insufficient_scope: {scope} required")
        return tok

    @mcp.tool(
        annotations=ToolAnnotations(title="Get ticket", readOnlyHint=True, destructiveHint=False)
    )
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        """Get one ticket by ID."""
        tok = _require(SCOPE_READ)
        t = TICKETS.get(ticket_id)
        if not t:
            return {"error": "not_found"}
        if tok.authority == "delegated" and t["owner"] != tok.subject:
            audit.record(
                event_type="mcp.tool",
                tool="get_ticket",
                agent=tok.actor or tok.subject,
                user=tok.subject,
                authority=tok.authority,
                decision="deny",
                reasons=["not owner"],
            )
            return {"error": "forbidden", "reason": "ticket belongs to another user"}
        audit.record(
            event_type="mcp.tool",
            tool="get_ticket",
            agent=tok.actor or tok.subject,
            user=tok.subject if tok.actor else None,
            authority=tok.authority,
            decision="allow",
        )
        return {"ticket": {"id": ticket_id, **t}}

    @mcp.tool(
        annotations=ToolAnnotations(title="List tickets", readOnlyHint=True, destructiveHint=False)
    )
    def list_tickets() -> dict[str, Any]:
        """List tickets visible to the caller (own tickets when delegated)."""
        tok = _require(SCOPE_READ)
        if tok.authority == "delegated":
            items = {k: v for k, v in TICKETS.items() if v["owner"] == tok.subject}
        else:
            items = dict(TICKETS)
        audit.record(
            event_type="mcp.tool",
            tool="list_tickets",
            agent=tok.actor or tok.subject,
            user=tok.subject if tok.actor else None,
            authority=tok.authority,
            decision="allow",
        )
        return {"tickets": [{"id": k, **v} for k, v in items.items()]}

    @mcp.tool(
        annotations=ToolAnnotations(
            title="Refund ticket", readOnlyHint=False, destructiveHint=True, idempotentHint=False
        )
    )
    def refund_ticket(ticket_id: str, amount: float) -> dict[str, Any]:
        """Refund a ticket (destructive). Requires tickets:write and a delegated user token."""
        tok = _require(SCOPE_WRITE)
        if tok.authority != "delegated":
            return {"error": "forbidden", "reason": "refunds require a user-delegated token"}
        t = TICKETS.get(ticket_id)
        if not t or t["owner"] != tok.subject:
            return {"error": "forbidden"}
        if amount > t["price"]:
            return {"error": "exceeds_price"}
        ref = payments.refund(
            ticket_id=ticket_id,
            amount=amount,
            on_behalf_of=tok.subject,
            actor=tok.actor or tok.subject,
        )
        t["status"] = "refunded"
        audit.record(
            event_type="mcp.tool",
            tool="refund_ticket",
            agent=tok.actor or tok.subject,
            user=tok.subject,
            authority=tok.authority,
            decision="allow",
            extra={"amount": amount},
        )
        return {"refund": ref, "ticket_id": ticket_id}

    return TicketsServer(
        mcp=mcp, verifier=verifier, audit=audit, payments=payments, resource_url=resource_url
    )


def build_app(mcp: FastMCP, *, require_dpop: bool = False) -> Starlette:
    """Starlette app: MCP endpoint + RFC 9728 metadata + DPoP handling + a health route."""
    app = mcp.streamable_http_app()

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True})

    app.add_route("/healthz", health, methods=["GET"])
    return DPoPMiddleware(app, require=require_dpop)  # type: ignore[return-value]


def serve(app: Any, host: str = "127.0.0.1", port: int = 8765) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


class ServerThread:
    """Run the app on a background uvicorn server (tests, notebooks)."""

    def __init__(self, app: Any, host: str = "127.0.0.1", port: int = 0):
        import socket

        import uvicorn

        if port == 0:
            with socket.socket() as s:
                s.bind((host, 0))
                port = s.getsockname()[1]
        self.host, self.port = host, port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on")
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    def start(self) -> ServerThread:
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started and time.time() < deadline:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError("MCP server did not start")
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)

    def __enter__(self) -> ServerThread:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()
