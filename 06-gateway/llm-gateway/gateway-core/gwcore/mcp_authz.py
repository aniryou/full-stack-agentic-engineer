"""The gateway as an MCP client (PRIMER §8).

The one idea: agents reach MCP servers through the gateway, so the gateway runs the *client* side of MCP
authorization (revision 2026-07-28, verify) for each (principal, resource): a 401 names the protected-resource
metadata (RFC 9728) -> the authorization server's metadata (RFC 8414 or OIDC discovery) -> the client is a
Client ID Metadata Document URL -> PKCE S256 with ``resource`` in both requests -> code -> token -> refresh
with rotation (a reused refresh token revokes the whole grant) -> step-up on 403 with the union of scopes.
The gateway holds every token; the agent never sees one. DPoP nonces come from RFC 9449 §8-§9, not MCP.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from urllib.parse import urlsplit


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def s256(verifier: str) -> str:
    """PKCE S256 (RFC 7636 §4.2): base64url(SHA-256(ASCII verifier)); Appendix B pins one pair."""
    return b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def make_verifier(n_bytes: int = 32) -> str:
    return b64url(secrets.token_bytes(n_bytes))


def parse_challenge(header: str | None) -> tuple:
    scheme, _, rest = (header or "").partition(" ")
    return scheme, dict(re.findall(r'([\w-]+)="([^"]*)"', rest))


def _base(url: str) -> tuple:
    u = urlsplit(url)
    return f"{u.scheme}://{u.netloc}", u.path.rstrip("/")


def prm_urls(resource: str, www_authenticate: str | None = None) -> list:
    """Where to look for protected-resource metadata: the 401's ``resource_metadata``, else the path-aware
    well-known URI, then the root one (clients MUST support both)."""
    rm = parse_challenge(www_authenticate)[1].get("resource_metadata")
    if rm:
        return [rm]
    base, path = _base(resource)
    return ([f"{base}/.well-known/oauth-protected-resource{path}"] if path else []) + [f"{base}/.well-known/oauth-protected-resource"]


def as_metadata_urls(issuer: str) -> list:
    """RFC 8414 path insertion, OIDC path insertion, then OIDC path appending (issuer without a path: two)."""
    base, path = _base(issuer)
    if path:
        return [f"{base}/.well-known/oauth-authorization-server{path}", f"{base}/.well-known/openid-configuration{path}",
                f"{base}{path}/.well-known/openid-configuration"]
    return [f"{base}/.well-known/oauth-authorization-server", f"{base}/.well-known/openid-configuration"]


def validate_cimd(url: str, doc: dict, max_bytes: int = 5120) -> list:
    """A Client ID Metadata Document's rules (draft-ietf-oauth-client-id-metadata-document): [] if valid."""
    u, errs = urlsplit(url), []
    checks = [(u.scheme == "https", "client_id must be an https URL"), (u.path not in ("", "/"), "client_id URL needs a path"),
              (not u.username and "@" not in u.netloc, "no userinfo"), (not u.fragment, "no fragment"),
              (not {".", ".."} & set(u.path.split("/")), "no dot segments"),
              (len(json.dumps(doc)) <= max_bytes, "document larger than the 5 KB read cap"),
              (doc.get("client_id") == url, "client_id must equal the document URL exactly"),
              ("client_secret" not in doc and doc.get("token_endpoint_auth_method") not in
               {"client_secret_post", "client_secret_basic", "client_secret_jwt"}, "no shared secrets")]
    errs += [msg for ok, msg in checks if not ok]
    return errs + [f"missing {f}" for f in ("client_id", "client_name", "redirect_uris") if f not in doc]


def iss_ok(expected: str, received: str | None, supported: bool) -> bool:
    """RFC 9207: a present ``iss`` must equal the issuer exactly; absent is fine only if not advertised."""
    return received == expected if received is not None else not supported


class HMACSigner:
    """DPoP proofs through a pluggable signer. A labelled stand-in: RFC 9449 §4.2 forbids MAC algorithms, so
    this is NOT DPoP-conformant; ``agentsec.identity.tokens.DPoP`` (06.6) signs with RS256 via ``cryptography``."""
    label = "HMAC stand-in (not RFC 9449: DPoP requires an asymmetric key)"

    def __init__(self, key: bytes):
        self.key, self.jkt = key, b64url(hashlib.sha256(b"jwk:" + key).digest())

    def proof(self, method: str, url: str, *, now: float, access_token: str | None = None, nonce: str | None = None) -> str:
        claims = {"jti": secrets.token_hex(12), "htm": method, "htu": url.split("?")[0].split("#")[0], "iat": int(now)}
        if access_token:
            claims["ath"] = s256(access_token)             # base64url(SHA-256(token)): the same construction as S256
        if nonce:
            claims["nonce"] = nonce
        body = b64url(json.dumps({"typ": "dpop+jwt", "alg": "HS256-STAND-IN", "jkt": self.jkt}).encode()) + "." + b64url(json.dumps(claims).encode())
        return body + "." + b64url(hmac.new(self.key, body.encode(), hashlib.sha256).digest())

    def verify(self, proof: str) -> tuple:
        body, _, sig = (proof or "..").rpartition(".")
        if not hmac.compare_digest(sig, b64url(hmac.new(self.key, body.encode(), hashlib.sha256).digest())):
            raise PermissionError("bad DPoP proof signature")
        h, c = body.split(".")
        return tuple(json.loads(base64.urlsafe_b64decode(x + "=" * (-len(x) % 4))) for x in (h, c))


class FakeAS:
    """An OAuth 2.1 authorization server in miniature: CIMD clients, PKCE, rotation with reuse detection, DPoP nonces."""

    def __init__(self, issuer: str, clock, *, documents: dict, pkce: bool = True, cimd: bool = True,
                 dpop_signer: HMACSigner | None = None, access_ttl: float = 300.0):
        self.issuer, self.clock, self.documents, self.pkce, self.cimd = issuer, clock, documents, pkce, cimd
        self.signer, self.ttl, self.nonce = dpop_signer, access_ttl, "as-nonce-1"
        self.codes, self.access, self.refresh, self.revoked, self.clients = {}, {}, {}, set(), {}

    def metadata(self) -> dict:
        m = {"issuer": self.issuer, "authorization_endpoint": self.issuer + "/authorize", "token_endpoint": self.issuer + "/token",
             "grant_types_supported": ["authorization_code", "refresh_token"], "authorization_response_iss_parameter_supported": True}
        m.update({"code_challenge_methods_supported": ["S256"]} if self.pkce else {})
        m.update({"client_id_metadata_document_supported": True} if self.cimd else {})
        return m | ({"dpop_signing_alg_values_supported": ["ES256"]} if self.signer else {})

    def _client(self, client_id: str) -> dict:
        if client_id not in self.clients:                 # fetch, validate; cache only valid documents
            doc = self.documents.get(client_id) or {}
            errs = validate_cimd(client_id, doc)
            if errs:
                raise PermissionError(f"invalid client metadata document: {errs}")
            self.clients[client_id] = doc
        return self.clients[client_id]

    def authorize(self, *, client_id, redirect_uri, code_challenge, code_challenge_method, resource, scope, subject) -> dict:
        if redirect_uri not in self._client(client_id)["redirect_uris"] or code_challenge_method != "S256":
            raise PermissionError("invalid_request")
        code = secrets.token_urlsafe(8)
        self.codes[code] = {"client_id": client_id, "challenge": code_challenge, "resource": resource, "scope": scope, "sub": subject}
        return {"code": code, "iss": self.issuer}          # the user's consent is assumed

    def token(self, form: dict, headers: dict | None = None) -> tuple:
        jkt = None
        if self.signer:                                    # RFC 9449 §8: the AS demands its nonce with a 400
            hdr, claims = self.signer.verify((headers or {}).get("DPoP"))
            if claims.get("nonce") != self.nonce:
                return 400, {"DPoP-Nonce": self.nonce}, {"error": "use_dpop_nonce"}
            jkt = hdr["jkt"]
        if form["grant_type"] == "authorization_code":
            c = self.codes.pop(form.get("code"), None)
            if not c or c["client_id"] != form.get("client_id") or s256(form.get("code_verifier", "")) != c["challenge"] \
                    or form.get("resource") != c["resource"]:
                return 400, {}, {"error": "invalid_grant"}
            return self._issue(secrets.token_hex(4), c["sub"], c["scope"], c["resource"], jkt)
        r = self.refresh.get(form.get("refresh_token"))
        if r is None or r["grant"] in self.revoked or form.get("resource") != r["resource"]:
            return 400, {}, {"error": "invalid_grant"}
        if r["used"]:                                      # OAuth 2.1 §4.3.1: reuse of a rotated token revokes the grant
            self.revoked.add(r["grant"])
            return 400, {}, {"error": "invalid_grant", "error_description": "refresh token reuse: grant revoked"}
        r["used"] = True
        return self._issue(r["grant"], r["sub"], r["scope"], r["resource"], jkt)

    def _issue(self, grant, sub, scope, resource, jkt) -> tuple:
        at, rt = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
        self.access[at] = {"grant": grant, "sub": sub, "aud": resource, "scope": set(scope.split()),
                           "exp": self.clock.now() + self.ttl, "jkt": jkt}
        self.refresh[rt] = {"grant": grant, "used": False, "sub": sub, "scope": scope, "resource": resource}
        return 200, {}, {"access_token": at, "token_type": "DPoP" if jkt else "Bearer", "expires_in": self.ttl,
                         "refresh_token": rt, "scope": scope}

    def introspect(self, token: str) -> dict | None:
        rec = self.access.get(token)
        return rec if rec and rec["grant"] not in self.revoked and rec["exp"] > self.clock.now() else None


class FakeMCPServer:
    """A resource server: 401 with ``resource_metadata``, 403 ``insufficient_scope``, DPoP nonces (RFC 9449 §9)."""

    def __init__(self, resource: str, auth: FakeAS, tool_scopes: dict, *, dpop_nonce: bool = False):
        self.resource, self.auth, self.tool_scopes, self.dpop_nonce, self.nonce = resource, auth, tool_scopes, dpop_nonce, "rs-nonce-1"
        base, path = _base(resource)
        self.prm_url = f"{base}/.well-known/oauth-protected-resource{path}"

    def prm(self) -> dict:
        return {"resource": self.resource, "authorization_servers": [self.auth.issuer],
                "scopes_supported": sorted(set(self.tool_scopes.values()))}

    def call(self, tool: str, headers: dict) -> tuple:
        scheme, _, token = headers.get("Authorization", "").partition(" ")
        rec, need = self.auth.introspect(token), self.tool_scopes[tool]
        if rec is None or rec["aud"] != self.resource:     # a token minted for another resource is refused
            return 401, {"WWW-Authenticate": f'Bearer resource_metadata="{self.prm_url}", scope="{need}"'}, {"error": "invalid_token"}
        if rec["jkt"]:
            hdr, claims = self.auth.signer.verify(headers.get("DPoP"))
            if scheme != "DPoP" or hdr["jkt"] != rec["jkt"] or claims.get("ath") != s256(token):
                return 401, {"WWW-Authenticate": 'DPoP error="invalid_token"'}, {"error": "invalid_token"}
            if self.dpop_nonce and claims.get("nonce") != self.nonce:
                return 401, {"WWW-Authenticate": 'DPoP error="use_dpop_nonce"', "DPoP-Nonce": self.nonce}, {"error": "use_dpop_nonce"}
        if need not in rec["scope"]:
            return 403, {"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{need}", '
                                             f'resource_metadata="{self.prm_url}"'}, {"error": "insufficient_scope"}
        return 200, {}, {"result": f"{tool} ran for {rec['sub']}"}


class Network:
    """The in-process internet: JSON documents by URL, endpoints (servers) by URL."""

    def __init__(self):
        self.docs, self.endpoints = {}, {}

    def add_as(self, auth: FakeAS, well_known: str | None = None) -> None:
        self.docs[well_known or as_metadata_urls(auth.issuer)[0]] = auth.metadata
        self.endpoints[auth.issuer + "/authorize"] = self.endpoints[auth.issuer + "/token"] = auth

    def add_mcp(self, server: FakeMCPServer) -> None:
        self.docs[server.prm_url] = server.prm
        self.endpoints[server.resource] = server

    def get(self, url: str) -> dict | None:
        return self.docs[url]() if url in self.docs else None


class MCPClient:
    """The gateway's MCP client: one token per (principal, resource), carrying the scopes it was granted."""

    def __init__(self, client_id: str, redirect_uri: str, net: Network, clock, *, signer: HMACSigner | None = None):
        self.client_id, self.redirect_uri, self.net, self.clock, self.signer = client_id, redirect_uri, net, clock, signer
        self.tokens, self.nonces, self.log = {}, {}, []

    def _discover(self, resource: str, challenge: str | None) -> dict:
        prm = next((d for d in map(self.net.get, prm_urls(resource, challenge)) if d), None)
        if prm is None:
            raise PermissionError("no protected-resource metadata at any well-known location")
        issuer = prm["authorization_servers"][0]
        meta = next((d for d in map(self.net.get, as_metadata_urls(issuer)) if d), None)
        if meta is None or meta["issuer"] != issuer:
            raise PermissionError("authorization server metadata missing or issuer mismatch")
        if "S256" not in meta.get("code_challenge_methods_supported", []):
            raise PermissionError("refusing: the AS does not advertise PKCE S256")
        if not meta.get("client_id_metadata_document_supported"):
            raise PermissionError("no CIMD support: pre-register, or fall back to DCR (deprecated)")
        self.log.append(f"discovered {issuer} via {resource}")
        return {"prm": prm, "meta": meta}

    def _headers(self, method: str, url: str, token: dict | None, server_key: str) -> dict:
        h = {"Authorization": f"{token['type']} {token['access']}"} if token else {}
        if self.signer:
            h["DPoP"] = self.signer.proof(method, url, now=self.clock.now(), access_token=token and token["access"],
                                          nonce=self.nonces.get(server_key))
        return h

    def _token_request(self, meta: dict, form: dict) -> dict:
        auth = self.net.endpoints[meta["token_endpoint"]]
        for _ in range(2):                                 # at most one nonce retry (RFC 9449 §8)
            status, h, body = auth.token(form, self._headers("POST", meta["token_endpoint"], None, meta["issuer"]))
            if body.get("error") == "use_dpop_nonce":
                self.nonces[meta["issuer"]] = h["DPoP-Nonce"]
                self.log.append("AS asked for a DPoP nonce")
                continue
            if status != 200:
                raise PermissionError(body)
            return body
        raise PermissionError("DPoP nonce loop")

    def _store(self, principal: str, resource: str, body: dict, meta: dict) -> dict:
        tok = {"access": body["access_token"], "refresh": body.get("refresh_token"), "type": body["token_type"],
               "scopes": set(body["scope"].split()), "exp": self.clock.now() + body["expires_in"], "meta": meta}
        self.tokens[(principal, resource)] = tok
        return tok

    def authorize(self, principal: str, resource: str, challenge: str | None = None, scopes: set | None = None) -> dict:
        d = self._discover(resource, challenge)
        meta, asked = d["meta"], parse_challenge(challenge)[1].get("scope")
        scopes = scopes or set((asked or " ".join(d["prm"].get("scopes_supported", []))).split())
        verifier = make_verifier()
        resp = self.net.endpoints[meta["authorization_endpoint"]].authorize(
            client_id=self.client_id, redirect_uri=self.redirect_uri, code_challenge=s256(verifier),
            code_challenge_method="S256", resource=resource, scope=" ".join(sorted(scopes)), subject=principal)
        if not iss_ok(meta["issuer"], resp.get("iss"), meta.get("authorization_response_iss_parameter_supported", False)):
            raise PermissionError("iss mismatch: possible mix-up attack")
        body = self._token_request(meta, {"grant_type": "authorization_code", "code": resp["code"], "code_verifier": verifier,
                                          "client_id": self.client_id, "redirect_uri": self.redirect_uri, "resource": resource})
        self.log.append(f"token for {principal} @ {resource}: {sorted(scopes)}")
        return self._store(principal, resource, body, meta)

    def refresh(self, principal: str, resource: str) -> dict:
        tok = self.tokens[(principal, resource)]
        body = self._token_request(tok["meta"], {"grant_type": "refresh_token", "refresh_token": tok["refresh"],
                                                 "client_id": self.client_id, "resource": resource})
        self.log.append(f"refreshed (rotated) for {principal}")
        return self._store(principal, resource, body, tok["meta"])

    def call(self, principal: str, resource: str, tool: str, max_steps: int = 5) -> dict:
        server = self.net.endpoints[resource]
        for _ in range(max_steps):
            tok = self.tokens.get((principal, resource))
            if tok and tok["exp"] <= self.clock.now():
                try:
                    tok = self.refresh(principal, resource)
                except PermissionError:                        # refused (revoked, reused): start over
                    tok = self.tokens.pop((principal, resource), None) and None
            status, h, body = server.call(tool, self._headers("POST", resource, tok, resource))
            if status == 200:
                return body
            params = parse_challenge(h.get("WWW-Authenticate"))[1]
            if params.get("error") == "use_dpop_nonce":
                self.nonces[resource] = h["DPoP-Nonce"]
                self.log.append("MCP server asked for a DPoP nonce")
            elif status == 401:
                self.authorize(principal, resource, h.get("WWW-Authenticate"))
            elif status == 403 and params.get("error") == "insufficient_scope":
                self.log.append(f"step-up: + {params['scope']}")
                self.authorize(principal, resource, h.get("WWW-Authenticate"), tok["scopes"] | set(params["scope"].split()))
            else:
                raise PermissionError(body)
        raise PermissionError("gave up after max_steps")
