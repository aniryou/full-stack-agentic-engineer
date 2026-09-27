"""A fake OAuth authorization server and a fake MCP server, over HTTP, for the gateway's client side (PRIMER §8).

The one idea: the MCP server is an OAuth 2.1 *resource server* (identity primer §7.1); whoever calls it —
here, the gateway on an agent's behalf — must discover its authorization server and obtain a token for the
right principal, resource and scopes. These fakes implement the server side of what the client must handle:

  MCP server (`rs`): 401 + `WWW-Authenticate: Bearer resource_metadata="…", scope="…"`; protected-resource
    metadata (RFC 9728) at the path-inserted well-known URI; 403 `insufficient_scope` for a tool that needs
    more; DPoP-bound tokens with a server nonce (401 `DPoP error="use_dpop_nonce"` + `DPoP-Nonce`).
  Authorization server (`as_`): metadata at RFC 8414 or OIDC locations (configurable, to exercise the client's
    fallback order); a Client ID Metadata Document fetch with the draft's rules (HTTPS URL with a path — loopback
    HTTP allowed in dev mode only, labelled — a 5 KB read cap, `client_id` must equal the URL, errors never
    cached); PKCE S256 only; `resource` required in both requests; RFC 9207 `iss` in the redirect; rotating
    refresh tokens with **reuse detection** (a replayed old refresh token revokes the whole grant — OAuth 2.1
    §4.3.1); DPoP at the token endpoint with a nonce (400 `use_dpop_nonce`).

There is no browser: `/authorize` approves the `login_hint` principal at once and answers with the redirect
(labelled). Tokens are HS256 JWTs signed with a secret the two fakes share, the same simplification as 07.2's
`agentlab.auth.oauth`; a real AS signs asymmetrically and publishes JWKS.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import json
import secrets
import time
import urllib.parse
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

from .dpop import ProofError, verify_proof

CIMD_MAX_BYTES = 5 * 1024
AUTO_CONSENT_LABEL = "no browser: the fake AS approves the login_hint principal immediately"
TOOLS = {"search_notes": "mcp:read", "delete_note": "mcp:write"}


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def jwt_encode(claims: dict, key: bytes) -> str:
    h = _b64(json.dumps({"alg": "HS256", "typ": "at+jwt"}, separators=(",", ":")).encode())
    c = _b64(json.dumps(claims, separators=(",", ":")).encode())
    return f"{h}.{c}." + _b64(hmac.new(key, f"{h}.{c}".encode(), hashlib.sha256).digest())


def jwt_decode(token: str, key: bytes) -> dict:
    h, c, s = token.split(".")
    if not hmac.compare_digest(_b64(hmac.new(key, f"{h}.{c}".encode(), hashlib.sha256).digest()), s):
        raise ValueError("bad signature")
    return json.loads(_unb64(c))


def cimd_url_ok(url: str, allow_loopback_http: bool) -> tuple[bool, str]:
    """The CIMD draft's client_id rules: https, a path, no userinfo/fragment, no dot segments; SSRF: no
    special-use addresses (loopback allowed only for a loopback AS in development)."""
    u = urllib.parse.urlsplit(url)
    host = u.hostname or ""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    loop = host == "localhost" or (ip is not None and ip.is_loopback)
    if u.scheme != "https" and not (allow_loopback_http and loop and u.scheme == "http"):
        return False, "client_id must be an https URL"
    if not u.path or u.path == "/" or "/./" in u.path or "/../" in u.path:
        return False, "client_id URL needs a path without dot segments"
    if u.username or u.fragment:
        return False, "no userinfo or fragment"
    if ip is not None and not loop and (ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast):
        return False, "special-use address (SSRF)"
    if loop and not allow_loopback_http:
        return False, "loopback client_id only for a loopback AS in development"
    return True, ""


@dataclass
class Family:
    """One grant: its rotating refresh tokens and the access tokens issued under it."""
    sub: str
    client_id: str
    resource: str
    scope: str
    jkt: str | None
    active_refresh: str
    used: set = field(default_factory=set)
    revoked: bool = False


@dataclass
class AsOptions:
    metadata: str = "rfc8414"           # rfc8414 | oidc | oidc_suffix : where the AS publishes its metadata
    advertise_pkce: bool = True         # code_challenge_methods_supported present?
    cimd: bool = True                   # client_id_metadata_document_supported
    dpop: bool = False                  # require DPoP at the token endpoint and bind tokens (cnf.jkt)
    iss_param: bool = True              # RFC 9207 iss in the authorization response
    allow_loopback_cimd: bool = True    # development only (labelled)
    access_ttl_s: int = 300
    wrong_issuer: bool = False          # publish a mismatching `issuer` (the client must reject it)


class McpServers:
    """The fake AS and the fake MCP server, each on its own port, sharing one signing key.

        servers = McpServers(AsOptions(dpop=True)); await servers.start(); servers.resource  # the MCP URL
    """

    def __init__(self, options: AsOptions | None = None, prm_in_challenge: bool = True, host: str = "127.0.0.1"):
        self.opt = options or AsOptions()
        self.prm_in_challenge = prm_in_challenge
        self.host = host
        self.key = secrets.token_bytes(32)
        self.codes: dict = {}
        self.families: dict = {}           # family id -> Family
        self.refresh_index: dict = {}      # refresh token -> family id
        self.cimd_cache: dict = {}
        self.as_nonce = secrets.token_urlsafe(12)
        self.rs_nonce = secrets.token_urlsafe(12)
        self.seen_jti: set = set()
        self.log: list = []
        self.stats = {"cimd_fetches": 0, "tokens": 0, "refreshes": 0, "reuse_detected": 0, "tool_calls": 0}
        self._runners: list = []
        self.as_url = self.rs_url = ""

    @property
    def issuer(self) -> str:
        return self.as_url + "/tenant1"

    @property
    def resource(self) -> str:
        return self.rs_url + "/mcp"

    async def start(self) -> str:
        self.session = aiohttp.ClientSession()
        for app, attr in ((self._as_app(), "as_url"), (self._rs_app(), "rs_url")):
            r = web.AppRunner(app, access_log=None)
            await r.setup()
            site = web.TCPSite(r, self.host, 0)
            await site.start()
            setattr(self, attr, f"http://{self.host}:{site._server.sockets[0].getsockname()[1]}")
            self._runners.append(r)
        return self.resource

    async def stop(self) -> None:
        for r in self._runners:
            await r.cleanup()
        await self.session.close()

    def rotate_nonces(self) -> None:
        self.as_nonce, self.rs_nonce = secrets.token_urlsafe(12), secrets.token_urlsafe(12)

    # ======================================================================================== AS
    def _as_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/.well-known/oauth-authorization-server/tenant1", self.as_meta_8414)
        app.router.add_get("/.well-known/openid-configuration/tenant1", self.as_meta_oidc)
        app.router.add_get("/tenant1/.well-known/openid-configuration", self.as_meta_oidc_suffix)
        app.router.add_get("/tenant1/authorize", self.authorize)
        app.router.add_post("/tenant1/token", self.token)
        return app

    def metadata(self) -> dict:
        m = {"issuer": self.issuer + ("-evil" if self.opt.wrong_issuer else ""),
             "authorization_endpoint": self.issuer + "/authorize", "token_endpoint": self.issuer + "/token",
             "response_types_supported": ["code"], "grant_types_supported": ["authorization_code", "refresh_token"],
             "token_endpoint_auth_methods_supported": ["none"], "scopes_supported": list(TOOLS.values()),
             "client_id_metadata_document_supported": self.opt.cimd,
             "authorization_response_iss_parameter_supported": self.opt.iss_param}
        if self.opt.advertise_pkce:
            m["code_challenge_methods_supported"] = ["S256"]
        if self.opt.dpop:
            m["dpop_signing_alg_values_supported"] = ["ES256", "HS256"]      # HS256: the lab's stand-in only
        return m

    async def as_meta_8414(self, request):
        return web.json_response(self.metadata()) if self.opt.metadata == "rfc8414" else web.Response(status=404)

    async def as_meta_oidc(self, request):
        return web.json_response(self.metadata()) if self.opt.metadata == "oidc" else web.Response(status=404)

    async def as_meta_oidc_suffix(self, request):
        return web.json_response(self.metadata()) if self.opt.metadata == "oidc_suffix" else web.Response(status=404)

    async def _client_metadata(self, client_id: str) -> dict:
        ok, why = cimd_url_ok(client_id, self.opt.allow_loopback_cimd)
        if not ok:
            raise ValueError(why)
        hit = self.cimd_cache.get(client_id)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        self.stats["cimd_fetches"] += 1
        async with self.session.get(client_id, allow_redirects=False) as r:
            raw = await r.content.read(CIMD_MAX_BYTES + 1)
            if r.status != 200 or len(raw) > CIMD_MAX_BYTES:
                raise ValueError(f"client metadata fetch failed ({r.status}, {len(raw)} bytes)")   # never cached
        doc = json.loads(raw)
        if doc.get("client_id") != client_id or not doc.get("redirect_uris") or not doc.get("client_name"):
            raise ValueError("client metadata document must carry client_id (= its URL), client_name, redirect_uris")
        if any(m in doc.get("token_endpoint_auth_method", "none") for m in ("client_secret",)) or "client_secret" in doc:
            raise ValueError("no shared-secret auth for a metadata-document client")
        self.cimd_cache[client_id] = (time.monotonic() + 300, doc)
        return doc

    async def authorize(self, request):
        q = request.query
        redirect = q.get("redirect_uri", "")
        try:
            doc = await self._client_metadata(q.get("client_id", ""))
        except (ValueError, aiohttp.ClientError) as e:
            return web.json_response({"error": "invalid_client", "error_description": str(e)}, status=400)
        if redirect not in doc["redirect_uris"]:
            return web.json_response({"error": "invalid_request", "error_description": "redirect_uri not registered"}, status=400)
        if q.get("code_challenge_method") != "S256" or not q.get("code_challenge"):
            return web.json_response({"error": "invalid_request", "error_description": "PKCE S256 required"}, status=400)
        if not q.get("resource"):
            return web.json_response({"error": "invalid_target", "error_description": "resource is required"}, status=400)
        code = secrets.token_urlsafe(16)
        self.codes[code] = {"client_id": q["client_id"], "redirect_uri": redirect, "challenge": q["code_challenge"],
                            "resource": q["resource"], "scope": q.get("scope", ""), "sub": q.get("login_hint", "anonymous"),
                            "exp": time.time() + 60}
        params = {"code": code, "state": q.get("state", "")}
        if self.opt.iss_param:
            params["iss"] = self.issuer
        self.log.append(("authorize", q.get("login_hint"), q.get("scope")))
        raise web.HTTPFound(redirect + "?" + urllib.parse.urlencode(params))

    def _dpop_at_token(self, request) -> tuple[str | None, web.Response | None]:
        """The AS side of DPoP: returns (jkt, None) or (None, error response). No proof and DPoP off -> (None, None)."""
        proof = request.headers.get("DPoP")
        if not self.opt.dpop:
            return None, None
        if not proof:
            return None, web.json_response({"error": "invalid_dpop_proof", "error_description": "DPoP proof required"}, status=400)
        try:
            _, jkt = verify_proof(proof, method="POST", url=self.issuer + "/token", nonce=self.as_nonce, seen_jti=self.seen_jti)
            return jkt, None
        except ProofError as e:
            return None, web.json_response({"error": e.error, "error_description": e.description}, status=400,
                                           headers={"DPoP-Nonce": self.as_nonce, "Cache-Control": "no-store"})

    def _issue(self, fam_id: str, fam: Family) -> dict:
        now = int(time.time())
        claims = {"iss": self.issuer, "sub": fam.sub, "aud": fam.resource, "scope": fam.scope, "client_id": fam.client_id,
                  "iat": now, "exp": now + self.opt.access_ttl_s, "jti": secrets.token_hex(8), "fam": fam_id}
        if fam.jkt:
            claims["cnf"] = {"jkt": fam.jkt}
        refresh = secrets.token_urlsafe(24)
        fam.active_refresh = refresh
        self.refresh_index[refresh] = fam_id
        self.stats["tokens"] += 1
        return {"access_token": jwt_encode(claims, self.key), "token_type": "DPoP" if fam.jkt else "Bearer",
                "expires_in": self.opt.access_ttl_s, "refresh_token": refresh, "scope": fam.scope}

    async def token(self, request):
        f = await request.post()
        jkt, err = self._dpop_at_token(request)
        if err is not None:
            return err
        headers = {"DPoP-Nonce": self.as_nonce, "Cache-Control": "no-store"} if self.opt.dpop else {"Cache-Control": "no-store"}
        if f.get("grant_type") == "authorization_code":
            c = self.codes.pop(f.get("code", ""), None)
            if not c or c["exp"] < time.time() or c["client_id"] != f.get("client_id") or c["redirect_uri"] != f.get("redirect_uri"):
                return web.json_response({"error": "invalid_grant", "error_description": "unknown or mismatched code"}, status=400)
            verifier = f.get("code_verifier", "")
            if _b64(hashlib.sha256(verifier.encode("ascii")).digest()) != c["challenge"]:
                return web.json_response({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status=400)
            if f.get("resource") != c["resource"]:
                return web.json_response({"error": "invalid_target", "error_description": "resource must match"}, status=400)
            fam_id = secrets.token_hex(6)
            fam = self.families[fam_id] = Family(c["sub"], c["client_id"], c["resource"], c["scope"], jkt, "")
            self.log.append(("token", c["sub"], c["scope"]))
            return web.json_response(self._issue(fam_id, fam), headers=headers)
        if f.get("grant_type") == "refresh_token":
            rt = f.get("refresh_token", "")
            fam_id = self.refresh_index.get(rt)
            fam = self.families.get(fam_id) if fam_id else None
            if fam is None or fam.revoked:
                return web.json_response({"error": "invalid_grant", "error_description": "unknown or revoked refresh token"}, status=400)
            if rt != fam.active_refresh:                  # an already-rotated token came back: someone has a copy
                fam.revoked = True
                self.stats["reuse_detected"] += 1
                self.log.append(("reuse_detected", fam.sub, fam_id))
                return web.json_response({"error": "invalid_grant", "error_description": "refresh token reuse: grant revoked"},
                                         status=400)
            if f.get("client_id") != fam.client_id or f.get("resource") != fam.resource:
                return web.json_response({"error": "invalid_grant", "error_description": "client or resource mismatch"}, status=400)
            if fam.jkt and jkt != fam.jkt:
                return web.json_response({"error": "invalid_dpop_proof", "error_description": "key does not match the grant"}, status=400)
            fam.used.add(rt)
            self.stats["refreshes"] += 1
            self.log.append(("refresh", fam.sub, fam_id))
            return web.json_response(self._issue(fam_id, fam), headers=headers)
        return web.json_response({"error": "unsupported_grant_type"}, status=400)

    # ======================================================================================== RS (the MCP server)
    def _rs_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/.well-known/oauth-protected-resource/mcp", self.prm)
        app.router.add_get("/.well-known/oauth-protected-resource", self.prm)
        app.router.add_post("/mcp", self.mcp)
        return app

    async def prm(self, request):
        doc = {"resource": self.resource, "authorization_servers": [self.issuer], "scopes_supported": list(TOOLS.values()),
               "bearer_methods_supported": ["header"]}
        if self.opt.dpop:
            doc["dpop_bound_access_tokens_required"] = True
        return web.json_response(doc)

    def _challenge(self, status: int, error: str | None = None, scope: str = "mcp:read", scheme: str = "Bearer") -> web.Response:
        parts = []
        if error:
            parts.append(f'error="{error}"')
        if scope:
            parts.append(f'scope="{scope}"')
        if self.prm_in_challenge:
            parts.append(f'resource_metadata="{self.rs_url}/.well-known/oauth-protected-resource/mcp"')
        headers = {"WWW-Authenticate": f"{scheme} " + ", ".join(parts)}
        if scheme == "DPoP":
            headers["DPoP-Nonce"] = self.rs_nonce
        return web.json_response({"error": error or "unauthorized"}, status=status, headers=headers)

    async def mcp(self, request):
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        if not token:
            return self._challenge(401)
        try:
            claims = jwt_decode(token, self.key)
        except (ValueError, KeyError):
            return self._challenge(401, "invalid_token")
        fam = self.families.get(claims.get("fam"))
        if claims.get("aud") != self.resource or claims.get("exp", 0) < time.time() or fam is None or fam.revoked:
            return self._challenge(401, "invalid_token")      # wrong audience, expired, or its grant was revoked
        if "cnf" in claims:
            if scheme != "DPoP":
                return self._challenge(401, "invalid_token", scheme="DPoP")
            try:
                _, jkt = verify_proof(request.headers.get("DPoP", ""), method="POST", url=self.resource, access_token=token,
                                      nonce=self.rs_nonce, seen_jti=self.seen_jti)
            except ProofError as e:
                return self._challenge(401, e.error, scope="", scheme="DPoP")
            if jkt != claims["cnf"]["jkt"]:
                return self._challenge(401, "invalid_token", scheme="DPoP")
        rpc = await request.json()
        granted = set(claims.get("scope", "").split())
        if rpc.get("method") == "tools/list":
            return web.json_response({"jsonrpc": "2.0", "id": rpc.get("id"), "result": {"tools": [
                {"name": n, "description": f"needs {s}", "inputSchema": {"type": "object"}} for n, s in TOOLS.items()]}})
        if rpc.get("method") == "tools/call":
            name = (rpc.get("params") or {}).get("name")
            need = TOOLS.get(name)
            if need is None:
                return web.json_response({"jsonrpc": "2.0", "id": rpc.get("id"), "error": {"code": -32602, "message": "unknown tool"}})
            if need not in granted:
                return self._challenge(403, "insufficient_scope", scope=" ".join(sorted(granted | {need})))
            self.stats["tool_calls"] += 1
            return web.json_response({"jsonrpc": "2.0", "id": rpc.get("id"), "result": {
                "content": [{"type": "text", "text": f"{name} ran for {claims['sub']}"}], "isError": False}})
        return web.json_response({"jsonrpc": "2.0", "id": rpc.get("id"), "result": {}})
