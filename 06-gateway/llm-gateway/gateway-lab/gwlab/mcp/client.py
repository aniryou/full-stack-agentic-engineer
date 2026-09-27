"""The gateway's MCP client: discovery, a Client ID Metadata Document, PKCE, tokens per principal, refresh,
step-up and DPoP nonces — the client side of the MCP authorization spec (revision 2026-07-28, verify).

The one idea (PRIMER §8): when agents reach MCP servers *through* the gateway (07.2 notebook 05's egress
gateway), the gateway is the OAuth client. For every (principal, MCP server) it must:

    1. call without a token -> 401 + WWW-Authenticate (resource_metadata, scope)
    2. fetch protected-resource metadata (RFC 9728): the header's URL, else the path-inserted then the root
       well-known URI
    3. fetch the authorization server's metadata: RFC 8414 with path insertion, then OIDC with path insertion,
       then OIDC with the path appended; the document's `issuer` must equal the issuer it was built from
    4. refuse unless `code_challenge_methods_supported` lists S256; register by Client ID Metadata Document when
       the AS advertises `client_id_metadata_document_supported` (Dynamic Client Registration is deprecated)
    5. authorize with PKCE S256 and `resource` -> code (+ `iss`, checked per RFC 9207) -> token (`resource` again)
    6. cache the token per (principal, resource, scopes) — the Python SDK's `TokenStorage` keeps one per
       server, which a multi-tenant gateway cannot do
    7. on 401 invalid_token: refresh (rotation; a reused refresh token revokes the grant) or re-authorize
    8. on 403 insufficient_scope: step up with the union of old and new scopes, a bounded number of times
    9. with DPoP (RFC 9449, not the MCP spec): answer `use_dpop_nonce` from the AS (400) or the RS (401) by
       retrying with the `DPoP-Nonce` it sent; keep one nonce per server.

07.2's `agentlab.auth.oauth` already runs discovery -> PKCE -> code -> token for one agent; this adds what a
gateway needs on top: CIMD, rotation with reuse detection, the per-principal cache, and nonces.
"""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
import urllib.parse
from dataclasses import dataclass

import aiohttp

from .dpop import default_signer, make_proof

MAX_STEP_UPS = 2


def pkce_challenge(verifier: str) -> str:
    """S256: base64url(SHA-256(ascii(verifier))) without padding (RFC 7636 §4.2)."""
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode()


def make_verifier() -> str:
    return secrets.token_urlsafe(32)          # 43 characters from the unreserved set


def parse_www_authenticate(header: str | None) -> tuple[str, dict]:
    """`Bearer error="insufficient_scope", scope="a b", resource_metadata="…"` -> ("Bearer", {...})."""
    if not header:
        return "", {}
    scheme, _, rest = header.strip().partition(" ")
    params, i = {}, 0
    while i < len(rest):
        eq = rest.find("=", i)
        if eq < 0:
            break
        name = rest[i:eq].strip(" ,")
        if rest[eq + 1:eq + 2] == '"':
            end = rest.find('"', eq + 2)
            params[name], i = rest[eq + 2:end], end + 1
        else:
            end = rest.find(",", eq)
            end = len(rest) if end < 0 else end
            params[name], i = rest[eq + 1:end].strip(), end + 1
    return scheme, params


def prm_urls(resource_url: str) -> list[str]:
    """Protected-resource metadata locations to try when the 401 names none: path-inserted, then root."""
    u = urllib.parse.urlsplit(resource_url)
    base = f"{u.scheme}://{u.netloc}"
    path = u.path.rstrip("/")
    out = [f"{base}/.well-known/oauth-protected-resource{path}"] if path else []
    return out + [f"{base}/.well-known/oauth-protected-resource"]


def as_metadata_urls(issuer: str) -> list[str]:
    """Authorization-server metadata locations, in the order MCP 2026-07-28 requires a client to try them."""
    u = urllib.parse.urlsplit(issuer)
    base, path = f"{u.scheme}://{u.netloc}", u.path.rstrip("/")
    if path:
        return [f"{base}/.well-known/oauth-authorization-server{path}", f"{base}/.well-known/openid-configuration{path}",
                f"{base}{path}/.well-known/openid-configuration"]
    return [f"{base}/.well-known/oauth-authorization-server", f"{base}/.well-known/openid-configuration"]


def union_scopes(previous: str, challenge: str) -> str:
    """Step-up: the new request asks for everything asked before plus what the challenge names."""
    return " ".join(sorted(set(previous.split()) | set(challenge.split())))


def iss_ok(supported: bool, iss: str | None, expected: str) -> bool:
    """RFC 9207 as MCP applies it: advertised and absent -> reject; present -> exact string match; neither -> ok."""
    if iss is not None:
        return iss == expected
    return not supported


class AuthError(Exception):
    pass


@dataclass
class TokenSet:
    access_token: str
    token_type: str
    scope: str
    refresh_token: str | None
    expires_at: float
    issuer: str
    token_endpoint: str


class McpClient:
    def __init__(self, session: aiohttp.ClientSession, client_id: str | None, redirect_uri: str | None = None,
                 signer=None, use_dpop: bool = True):
        self.session, self.client_id = session, client_id
        self._redirect = redirect_uri
        self.signer = signer
        self.use_dpop = use_dpop
        self.tokens: dict = {}          # (principal, resource, frozenset(scopes)) -> TokenSet
        self.nonces: dict = {}          # server origin -> the latest DPoP-Nonce it sent
        self.meta: dict = {}            # resource -> (prm, as metadata)
        self.log: list = []

    @property
    def redirect_uri(self) -> str:
        if self._redirect:
            return self._redirect
        u = urllib.parse.urlsplit(self.client_id)
        return f"{u.scheme}://{u.netloc}/oauth/callback"

    def _step(self, *what):
        self.log.append(what)

    @staticmethod
    def _origin(url: str) -> str:
        u = urllib.parse.urlsplit(url)
        return f"{u.scheme}://{u.netloc}"

    def _dpop_headers(self, method: str, url: str, access_token: str | None = None) -> dict:
        if not self.signer:
            return {}
        return {"DPoP": make_proof(self.signer, method, url, access_token, self.nonces.get(self._origin(url)))}

    def _remember_nonce(self, url: str, headers) -> None:
        if headers.get("DPoP-Nonce"):
            self.nonces[self._origin(url)] = headers["DPoP-Nonce"]

    # ------------------------------------------------------------------ token cache
    def cached(self, principal: str, resource: str) -> TokenSet | None:
        best = None
        for (p, r, scopes), ts in self.tokens.items():
            if p == principal and r == resource and (best is None or len(scopes) > len(best[0])):
                best = (scopes, ts)
        return best[1] if best else None

    def forget(self, principal: str, resource: str) -> None:
        for k in [k for k in self.tokens if k[0] == principal and k[1] == resource]:
            del self.tokens[k]

    def _store(self, principal: str, resource: str, ts: TokenSet) -> None:
        for k in [k for k in self.tokens if k[0] == principal and k[1] == resource]:
            del self.tokens[k]
        self.tokens[(principal, resource, frozenset(ts.scope.split()))] = ts

    # ------------------------------------------------------------------ the call
    async def call(self, principal: str, resource: str, rpc: dict) -> tuple[int, dict]:
        """POST a JSON-RPC message to the MCP server as `principal`, obtaining or renewing tokens as needed."""
        step_ups, nonce_retries, reauth = 0, 0, 0
        while True:
            ts = self.cached(principal, resource)
            headers = {"Content-Type": "application/json"}
            if ts:
                headers["Authorization"] = f"{ts.token_type} {ts.access_token}"
                if ts.token_type == "DPoP":
                    headers.update(self._dpop_headers("POST", resource, ts.access_token))
            async with self.session.post(resource, json=rpc, headers=headers) as r:
                self._remember_nonce(resource, r.headers)
                status = r.status
                www = r.headers.get("WWW-Authenticate")
                body = await r.json(content_type=None)
            if status < 400:
                return status, body
            scheme, ch = parse_www_authenticate(www)
            self._step("rs", status, ch.get("error") or "no token", ch.get("scope", ""))
            if status == 401 and ch.get("error") == "use_dpop_nonce" and nonce_retries < 2:
                nonce_retries += 1
                continue                                   # the nonce is stored; sign a new proof with it
            if status == 401 and (ts is None or ch.get("error") == "invalid_token") and reauth < 3:
                reauth += 1
                if ts and ts.refresh_token:
                    try:
                        await self.refresh(principal, resource, ts)
                        continue
                    except AuthError as e:                 # e.g. the grant was revoked: start over
                        self._step("refresh_failed", str(e))
                        self.forget(principal, resource)
                await self.authorize(principal, resource, www, previous_scope=ts.scope if ts else "")
                continue
            if status == 403 and ch.get("error") == "insufficient_scope" and step_ups < MAX_STEP_UPS:
                step_ups += 1
                await self.authorize(principal, resource, www, previous_scope=ts.scope if ts else "")
                continue
            return status, body

    # ------------------------------------------------------------------ discovery
    async def discover(self, resource: str, www: str | None) -> tuple[dict, dict]:
        if resource in self.meta:
            return self.meta[resource]
        _, ch = parse_www_authenticate(www)
        prm = None
        for url in ([ch["resource_metadata"]] if ch.get("resource_metadata") else prm_urls(resource)):
            async with self.session.get(url) as r:
                if r.status == 200:
                    prm = await r.json()
                    self._step("prm", url)
                    break
        if not prm or not prm.get("authorization_servers"):
            raise AuthError("no protected-resource metadata")
        issuer = prm["authorization_servers"][0]
        asm = None
        for url in as_metadata_urls(issuer):
            async with self.session.get(url) as r:
                self._step("as_metadata_try", url, r.status)
                if r.status == 200:
                    asm = await r.json()
                    break
        if asm is None:
            raise AuthError("no authorization-server metadata at any well-known location")
        if asm.get("issuer") != issuer:
            raise AuthError(f"issuer mismatch: metadata says {asm.get('issuer')!r}, expected {issuer!r}")
        if "S256" not in (asm.get("code_challenge_methods_supported") or []):
            raise AuthError("the AS does not advertise PKCE S256: refusing to proceed")
        self.meta[resource] = (prm, asm)
        return prm, asm

    async def authorize(self, principal: str, resource: str, www: str | None, previous_scope: str = "") -> TokenSet:
        prm, asm = await self.discover(resource, www)
        _, ch = parse_www_authenticate(www)
        scope = union_scopes(previous_scope, ch.get("scope") or " ".join(prm.get("scopes_supported") or []))
        if not asm.get("client_id_metadata_document_supported"):
            raise AuthError("the AS does not accept Client ID Metadata Documents (and DCR is not implemented here)")
        verifier, state = make_verifier(), secrets.token_urlsafe(8)
        params = {"response_type": "code", "client_id": self.client_id, "redirect_uri": self.redirect_uri, "scope": scope,
                  "state": state, "code_challenge": pkce_challenge(verifier), "code_challenge_method": "S256",
                  "resource": resource, "login_hint": principal}
        async with self.session.get(asm["authorization_endpoint"], params=params, allow_redirects=False) as r:
            if r.status != 302:
                raise AuthError(f"authorize: {r.status} {await r.text()}")
            loc = urllib.parse.urlsplit(r.headers["Location"])
        q = dict(urllib.parse.parse_qsl(loc.query))
        if q.get("state") != state:
            raise AuthError("state mismatch")
        if not iss_ok(bool(asm.get("authorization_response_iss_parameter_supported")), q.get("iss"), asm["issuer"]):
            raise AuthError("iss check failed (possible mix-up attack)")
        self._step("authorized", principal, scope)
        form = {"grant_type": "authorization_code", "code": q["code"], "redirect_uri": self.redirect_uri,
                "client_id": self.client_id, "code_verifier": verifier, "resource": resource}
        tok = await self._token_request(asm, form)
        ts = TokenSet(tok["access_token"], tok.get("token_type", "Bearer"), tok.get("scope", scope), tok.get("refresh_token"),
                      time.time() + tok.get("expires_in", 300), asm["issuer"], asm["token_endpoint"])
        self._store(principal, resource, ts)
        self._step("token", principal, ts.token_type, ts.scope)
        return ts

    async def refresh(self, principal: str, resource: str, ts: TokenSet) -> TokenSet:
        _, asm = self.meta[resource]
        tok = await self._token_request(asm, {"grant_type": "refresh_token", "refresh_token": ts.refresh_token,
                                              "client_id": self.client_id, "resource": resource})
        new = TokenSet(tok["access_token"], tok.get("token_type", "Bearer"), tok.get("scope", ts.scope),
                       tok.get("refresh_token") or ts.refresh_token, time.time() + tok.get("expires_in", 300),
                       ts.issuer, ts.token_endpoint)
        self._store(principal, resource, new)
        self._step("refreshed", principal)
        return new

    async def _token_request(self, asm: dict, form: dict) -> dict:
        url = asm["token_endpoint"]
        if self.use_dpop and asm.get("dpop_signing_alg_values_supported") and self.signer is None:
            self.signer = default_signer()
        for _ in range(3):
            headers = self._dpop_headers("POST", url) if (self.use_dpop and asm.get("dpop_signing_alg_values_supported")) else {}
            async with self.session.post(url, data=form, headers=headers) as r:
                self._remember_nonce(url, r.headers)
                body = await r.json(content_type=None)
                if r.status == 200:
                    return body
                if r.status == 400 and body.get("error") == "use_dpop_nonce":
                    self._step("as", 400, "use_dpop_nonce")
                    continue
                raise AuthError(f"token endpoint: {r.status} {body}")
        raise AuthError("token endpoint kept asking for a DPoP nonce")
