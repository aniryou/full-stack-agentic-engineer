"""service.py — the memory service over HTTP: scope from a verified token, idempotent writes, a forget by key, an audit line each.

The one idea (PRIMER §8, the 06/07 split): the partition a request reads or writes is decided by who
the caller *is*, never by what the request *says*. The service verifies a bearer token and takes
``tenant`` and ``sub`` (the user) from its claims; a body that tries to name a tenant or user is
rejected (400 ``scope_in_body``), so a prompt-injected agent cannot ask for someone else's memory. The
token carries the agent as its actor (``act``), so every audit event records both identities
(identity primer §3.5 delegation, §9 audit).

The token here is an **HMAC stand-in** for a real verifier — the identity core's RS256 ``Issuer``
(06-gateway) or your IdP's JWKS; it has the same claims and the same failure modes (bad signature,
expired, wrong audience, missing scope), not the same cryptography.

Endpoints (JSON):

    POST   /v1/memories                 write one memory; header Idempotency-Key makes a retry a replay
    POST   /v1/memories/search          {"query", "k", "budget_tokens", "kinds", "as_of"} or {"profile": true, "k"}
    DELETE /v1/memories?subject=...     forget by subject or deletion key; returns the DeletionReport
    POST   /v1/memories/{id}/promote    a reviewer (scope memory.review) promotes a quarantined record
    GET    /healthz, /metrics

An Idempotency-Key reused with a *different* body is refused with 422, as the IETF
``Idempotency-Key`` header draft recommends (verify).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict

from aiohttp import web

from .audit import AuditEvent, AuditLog, args_digest
from .memory import LocalMemory, WritePolicy
from .store.sqlite import SQLiteMemoryStore

SCOPES = ("memory.read", "memory.write", "memory.forget")
BODY_SCOPE_FIELDS = ("tenant", "user", "user_id", "sub", "agent")
WRITE_FIELDS = ("text", "kind", "slot", "value", "source", "confidence", "importance", "provenance", "session",
                "scope", "ttl_s", "deletion_key", "valid_from")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class TokenVerifier:
    """Mint and verify the lab's stand-in tokens: ``base64url(claims).base64url(HMAC-SHA256)``."""

    def __init__(self, key: bytes, audience: str = "memlab", issuer: str = "memlab-standin-issuer", clock=time.time):
        self.key, self.audience, self.issuer, self.clock = key, audience, issuer, clock

    def mint(self, tenant: str, user: str, agent: str = "agent:memlab", scopes=SCOPES, ttl_s: float = 900) -> str:
        now = self.clock()
        claims = {"iss": self.issuer, "aud": self.audience, "sub": user, "tenant": tenant,
                  "act": {"sub": agent}, "scope": " ".join(scopes), "iat": int(now), "exp": int(now + ttl_s)}
        payload = _b64(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
        sig = _b64(hmac.new(self.key, payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{sig}"

    def verify(self, token: str, need: str | None = None) -> dict:
        try:
            payload, sig = token.split(".")
        except ValueError:
            raise PermissionError("malformed token") from None
        good = _b64(hmac.new(self.key, payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise PermissionError("bad signature")
        claims = json.loads(_unb64(payload))
        if claims.get("aud") != self.audience or claims.get("iss") != self.issuer:
            raise PermissionError("wrong audience or issuer")
        if claims.get("exp", 0) <= self.clock():
            raise PermissionError("token expired")
        if need and need not in claims.get("scope", "").split():
            raise PermissionError(f"missing scope {need}")
        return claims


def _err(status: int, error: str, message: str) -> web.Response:
    return web.json_response({"ok": False, "error": error, "message": message}, status=status)


class MemoryService:
    """The aiohttp app around one store. ``start()`` runs it on a background thread and returns the base URL."""

    def __init__(self, store: SQLiteMemoryStore, verifier: TokenVerifier, *, policy: WritePolicy | None = None,
                 audit: AuditLog | None = None, host: str = "127.0.0.1", port: int = 0):
        self.store, self.verifier = store, verifier
        self.policy = policy or WritePolicy()
        self.audit = audit or AuditLog()
        self.host, self.port = host, port
        self.counters: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()
        with store._lock:
            store.con.execute("CREATE TABLE IF NOT EXISTS idempotency (tenant TEXT NOT NULL, key TEXT NOT NULL, "
                              "request_hash TEXT NOT NULL, status INTEGER NOT NULL, response TEXT NOT NULL, "
                              "record_id TEXT, created_at REAL NOT NULL, PRIMARY KEY (tenant, key))")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    # -- helpers ---------------------------------------------------------------------------------
    def _count(self, name: str) -> None:
        self.counters[name] = self.counters.get(name, 0) + 1

    def _principal(self, request: web.Request, need: str) -> dict:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise PermissionError("missing bearer token")
        return self.verifier.verify(auth[7:], need)

    def _memory(self, claims: dict) -> LocalMemory:
        return LocalMemory(self.store, claims["tenant"], claims["sub"], agent=claims.get("act", {}).get("sub", "?"),
                           policy=self.policy, audit=self.audit, clock=self.store.clock)

    # -- routes ----------------------------------------------------------------------------------
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/healthz", self._health)
        app.router.add_get("/metrics", self._metrics)
        app.router.add_post("/v1/memories", self._write)
        app.router.add_post("/v1/memories/search", self._search)
        app.router.add_delete("/v1/memories", self._forget)
        app.router.add_post("/v1/memories/{id}/promote", self._promote)
        return app

    async def _health(self, request):
        return web.json_response({"ok": True, "records": self.store.stats()["records"]})

    async def _metrics(self, request):
        lines = ["# TYPE memlab_requests_total counter"]
        lines += [f'memlab_requests_total{{route="{k}"}} {v}' for k, v in sorted(self.counters.items())]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def _write(self, request):
        try:
            claims = self._principal(request, "memory.write")
        except PermissionError as e:
            self._count("write_401")
            return _err(401, "unauthorized", str(e))
        body = await request.json()
        leaked = [f for f in BODY_SCOPE_FIELDS if f in body]
        if leaked:
            self._count("write_400")
            return _err(400, "scope_in_body", f"scope comes from the token, never the body: remove {leaked}")
        unknown = [f for f in body if f not in WRITE_FIELDS]
        if unknown or not isinstance(body.get("text"), str) or not body["text"].strip():
            return _err(400, "invalid_body", f"need a non-empty 'text'; unknown fields {unknown}")
        if body.get("source") == "human" and "memory.review" not in claims.get("scope", "").split():
            return _err(403, "forbidden", "source 'human' needs the memory.review scope")
        key = request.headers.get("Idempotency-Key")
        req_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key:
            with self.store._lock:
                row = self.store.con.execute("SELECT request_hash, status, response FROM idempotency "
                                             "WHERE tenant=? AND key=?", (claims["tenant"], key)).fetchone()
            if row:
                if row["request_hash"] != req_hash:
                    return _err(422, "idempotency_key_reused", "this Idempotency-Key was used with a different body")
                self._count("write_replay")
                resp = json.loads(row["response"])
                resp["replayed"] = True
                return web.json_response(resp, status=row["status"], headers={"Idempotent-Replayed": "true"})
        mem = self._memory(claims)
        fields = {k: body[k] for k in WRITE_FIELDS if k in body and k != "text"}
        if "provenance" in fields:
            fields["provenance"] = tuple(fields["provenance"])
        out = mem.remember(body["text"], idempotency_key=f"svc:{claims['tenant']}:{key}" if key else None, **fields)
        status = 201 if out.get("ok") else 422
        if key:
            with self.store._lock:
                self.store.con.execute("INSERT OR IGNORE INTO idempotency VALUES (?,?,?,?,?,?,?)",
                                       (claims["tenant"], key, req_hash, status, json.dumps(out), out.get("id"),
                                        self.store.clock()))
        self._count("write")
        return web.json_response(out, status=status)

    async def _search(self, request):
        try:
            claims = self._principal(request, "memory.read")
        except PermissionError as e:
            self._count("search_401")
            return _err(401, "unauthorized", str(e))
        body = await request.json()
        leaked = [f for f in BODY_SCOPE_FIELDS if f in body]
        if leaked:
            return _err(400, "scope_in_body", f"scope comes from the token, never the body: remove {leaked}")
        mem = self._memory(claims)
        k, budget = int(body.get("k", 5)), body.get("budget_tokens")
        if body.get("profile"):
            items = mem.profile(k, budget)
        else:
            kinds = body.get("kinds")
            items = mem.recall(str(body.get("query", "")), k, budget, as_of=body.get("as_of"),
                               session=body.get("session"), kinds=tuple(kinds) if kinds else None)
        self._count("search")
        return web.json_response({"ok": True, "items": items, "scope": mem.scope})

    async def _forget(self, request):
        try:
            claims = self._principal(request, "memory.forget")
        except PermissionError as e:
            self._count("forget_401")
            return _err(401, "unauthorized", str(e))
        subject = request.query.get("subject") or request.query.get("deletion_key")
        if not subject:
            return _err(400, "invalid_request", "name a subject or a deletion_key")
        mem = self._memory(claims)
        key = mem.deletion_key_for(subject)
        if not key.startswith(f"{claims['tenant']}/{claims['sub']}/"):
            return _err(403, "forbidden", "a deletion key outside your own partition")
        rep = mem.forget(subject)
        with self.store._lock:
            ids = rep.deleted_ids
            n = 0
            for rid in ids:
                n += self.store.con.execute("DELETE FROM idempotency WHERE tenant=? AND record_id=?",
                                            (claims["tenant"], rid)).rowcount
        rep.counts["idempotency"] = n
        self._count("forget")
        return web.json_response({"ok": True, "report": asdict(rep)})

    async def _promote(self, request):
        try:
            claims = self._principal(request, "memory.review")
        except PermissionError as e:
            return _err(403, "forbidden", str(e))
        rec = self.store.get(request.match_info["id"])
        if rec is None or rec.tenant != claims["tenant"]:
            return _err(404, "not_found", "no such record in your tenant")
        if rec.status != "quarantined":
            return _err(409, "conflict", f"record is {rec.status}, not quarantined")
        self.store.set_fields(rec.id, status="active")
        self.audit.record(AuditEvent("memory.write", claims.get("act", {}).get("sub", "?"), "delegated", user=rec.user,
                                     decision="promote", reasons=["quarantined record reviewed and promoted"],
                                     args_hash=args_digest({"id": rec.id}), approver=claims["sub"],
                                     provenance=rec.provenance, tenant=rec.tenant))
        self._count("promote")
        return web.json_response({"ok": True, "id": rec.id, "status": "active", "approver": claims["sub"]})

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> str:
        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._runner = web.AppRunner(self.app(), access_log=None)
            self._loop.run_until_complete(self._runner.setup())
            site = web.TCPSite(self._runner, self.host, self.port)
            self._loop.run_until_complete(site.start())
            self.port = site._server.sockets[0].getsockname()[1]
            self._ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(self._runner.cleanup())
            self._loop.close()
        self._thread = threading.Thread(target=run, daemon=True, name="memlab-service")
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("memory service did not start")
        return self.url

    def stop(self) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class MemoryClient:
    """A small synchronous client (urllib): returns ``(status, json)``; never raises on 4xx."""

    def __init__(self, url: str, token: str, timeout_s: float = 10):
        self.url, self.token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def request(self, method: str, path: str, body: dict | None = None, headers: dict | None = None):
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json", **(headers or {})}
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def write(self, text: str, idempotency_key: str | None = None, **fields):
        return self.request("POST", "/v1/memories", {"text": text, **fields},
                            {"Idempotency-Key": idempotency_key} if idempotency_key else None)

    def search(self, query: str = "", k: int = 5, budget_tokens: int | None = None, **extra):
        body = {"query": query, "k": k, **extra}
        if budget_tokens is not None:
            body["budget_tokens"] = budget_tokens
        return self.request("POST", "/v1/memories/search", body)

    def forget(self, subject: str):
        return self.request("DELETE", "/v1/memories?subject=" + urllib.request.quote(subject))

    def promote(self, rec_id: str):
        return self.request("POST", f"/v1/memories/{rec_id}/promote", {})


class RemoteMemory:
    """``LocalMemory``'s four methods over HTTP, so the agent cannot tell (and need not care) which it has."""

    def __init__(self, client: MemoryClient, scope: str = "?"):
        self.client, self.scope = client, scope

    def remember(self, text: str, *, idempotency_key: str | None = None, **fields) -> dict:
        fields = {k: v for k, v in fields.items() if k in WRITE_FIELDS and v is not None}
        if "provenance" in fields:
            fields["provenance"] = list(fields["provenance"])
        status, body = self.client.write(text, idempotency_key, **fields)
        return body if status < 500 else {"ok": False, "error": "server_error"}

    def recall(self, query: str, k: int = 5, budget_tokens: int | None = None, *, kinds=None, **_) -> list[dict]:
        status, body = self.client.search(query, k, budget_tokens, **({"kinds": list(kinds)} if kinds else {}))
        return body.get("items", []) if status == 200 else []

    def profile(self, max_items: int = 6, budget_tokens: int | None = None) -> list[dict]:
        status, body = self.client.search("", max_items, budget_tokens, profile=True)
        return body.get("items", []) if status == 200 else []

    def forget(self, subject: str, **_) -> dict:
        status, body = self.client.forget(subject)
        return body.get("report", body)

    def render(self, items: list[dict]) -> str:
        from .extract import render_memory
        return render_memory([(i["text"], i.get("date")) for i in items], self.scope)
