"""A toy OAuth 2.1 authorization server and the client-side discovery chain (notebook 06).

The identity-and-security primer covers the same standards in its §3.5 and §7.1
(06-gateway/identity-security/agentic-identity-gcp-lab/docs/primer.md).

Everything is in-process and uses only stdlib crypto so the *shapes* stay in
focus:

* PKCE with S256 (OAuth 2.1 makes it mandatory for every client);
* RFC 8414 server metadata and RFC 9728 protected-resource metadata;
* RFC 8707 resource indicators: the token's ``aud`` is the MCP server's
  canonical URL, so a token stolen from one server is useless at another;
* RFC 9207 ``iss`` in the authorization response, checked by the client to
  defeat mix-up attacks;
* RFC 8693 token exchange: the MCP server trades the inbound user token for a
  downstream credential that keeps the same ``sub`` and records the server in
  ``act`` — the alternative to the forbidden token passthrough;
* ``insufficient_scope`` step-up: parse the required scope from a 403 and
  re-authorize with the union.

Tokens are HS256 JWTs (``hmac`` + base64url). A real authorization server
signs with an asymmetric key and publishes the public half as JWKS so resource
servers can verify without sharing a secret; HS256 is used here only so the
whole chain fits in one readable file.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping

from ..agents.tools import Identity

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
WELL_KNOWN_PRM = "/.well-known/oauth-protected-resource"
WELL_KNOWN_AS = "/.well-known/oauth-authorization-server"


# --------------------------------------------------------------------- errors
class OAuthError(Exception):
    """An OAuth error response; ``error`` is the RFC 6749 error code."""

    def __init__(self, error: str, description: str = ""):
        super().__init__(f"{error}: {description}" if description else error)
        self.error = error
        self.description = description


class InvalidToken(OAuthError):
    def __init__(self, description: str):
        super().__init__("invalid_token", description)


class InvalidSignature(InvalidToken):
    pass


class TokenExpired(InvalidToken):
    pass


class WrongAudience(InvalidToken):
    pass


class InvalidGrant(OAuthError):
    def __init__(self, description: str):
        super().__init__("invalid_grant", description)


class InvalidClient(OAuthError):
    def __init__(self, description: str):
        super().__init__("invalid_client", description)


class InvalidScope(OAuthError):
    def __init__(self, description: str):
        super().__init__("invalid_scope", description)


class MixUpDetected(OAuthError):
    """The ``iss`` in the authorization response is not the server we talked to (RFC 9207)."""

    def __init__(self, expected: str, got: str | None):
        super().__init__("invalid_issuer", f"expected iss {expected!r}, got {got!r}")


# ----------------------------------------------------------------- JWT / PKCE
def b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def encode_jwt(claims: Mapping[str, Any], key: bytes, kid: str = "lab-hs256") -> str:
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    signing_input = b64url_encode(json.dumps(header, separators=(",", ":")).encode()) + "." + \
        b64url_encode(json.dumps(dict(claims), separators=(",", ":"), sort_keys=True).encode())
    signature = hmac.new(key, signing_input.encode("ascii"), hashlib.sha256).digest()
    return signing_input + "." + b64url_encode(signature)


def decode_jwt(token: str, key: bytes) -> dict[str, Any]:
    """Verify the HMAC and return the claims. Signature first: never parse what you have not verified."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        raise InvalidToken("malformed token") from None
    expected = hmac.new(key, f"{header_b64}.{payload_b64}".encode("ascii"), hashlib.sha256).digest()
    if not hmac.compare_digest(expected, b64url_decode(sig_b64)):
        raise InvalidSignature("signature does not verify")
    return json.loads(b64url_decode(payload_b64))


def peek_claims(token: str) -> dict[str, Any]:
    """Read claims WITHOUT verifying. For display and tests only; never for decisions."""
    return json.loads(b64url_decode(token.split(".")[1]))


def make_verifier(n_bytes: int = 32) -> str:
    """PKCE code verifier: 43–128 chars of unguessable entropy, kept by the client."""
    return b64url_encode(secrets.token_bytes(n_bytes))


def challenge_for(verifier: str) -> str:
    """S256: ``BASE64URL(SHA256(verifier))``. The challenge travels; the verifier never does until the token call."""
    return b64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())


# ------------------------------------------------------- authorization server
@dataclass
class RegisteredClient:
    client_id: str
    redirect_uris: tuple[str, ...] = ()
    resource_url: str | None = None          # set for a resource server (an MCP server) that may exchange tokens issued to it
    delegated_scopes: tuple[str, ...] = ()   # downstream scopes that resource server may request on a user's behalf


@dataclass
class _AuthCode:
    client_id: str
    redirect_uri: str
    scope: str
    resource: str
    code_challenge: str
    subject: str
    expires_at: float
    used: bool = False


@dataclass
class AuthorizationServer:
    """The enterprise IdP, in miniature. ``clock`` is injectable so expiry is testable without sleeping."""

    issuer: str
    clock: Callable[[], float] = time.time
    token_ttl_s: int = 3600
    code_ttl_s: int = 60
    scopes_supported: tuple[str, ...] = ()
    _key: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    _clients: dict[str, RegisteredClient] = field(default_factory=dict, repr=False)
    _codes: dict[str, _AuthCode] = field(default_factory=dict, repr=False)

    # -- registration and discovery -----------------------------------------
    def register_client(self, client_id: str, redirect_uris: Iterable[str] = (), resource_url: str | None = None,
                        delegated_scopes: Iterable[str] = ()) -> RegisteredClient:
        client = RegisteredClient(client_id, tuple(redirect_uris), resource_url, tuple(delegated_scopes))
        self._clients[client_id] = client
        return client

    def registered_client(self, client_id: str) -> RegisteredClient:
        client = self._clients.get(client_id)
        if client is None:
            raise InvalidClient(f"unknown client {client_id!r}")
        return client

    def metadata(self) -> dict[str, Any]:
        """RFC 8414 authorization-server metadata (what ``{issuer}/.well-known/oauth-authorization-server`` returns)."""
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "introspection_endpoint": f"{self.issuer}/introspect",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", TOKEN_EXCHANGE_GRANT],
            "code_challenge_methods_supported": ["S256"],
            "scopes_supported": list(self.scopes_supported),
            "authorization_response_iss_parameter_supported": True,
        }

    # -- authorization code + PKCE ------------------------------------------
    def authorize(self, client_id: str, redirect_uri: str, scope: str, resource: str, code_challenge: str,
                  subject: str, code_challenge_method: str = "S256", state: str | None = None) -> dict[str, Any]:
        """The redirect to ``/authorize``. ``subject`` stands in for the user's SSO session at the IdP.

        Returns what the browser would carry back to ``redirect_uri``: the code,
        the ``iss`` (RFC 9207) and the client's ``state``.
        """
        client = self.registered_client(client_id)
        if redirect_uri not in client.redirect_uris:
            raise InvalidClient(f"redirect_uri {redirect_uri!r} is not registered for {client_id}")
        if code_challenge_method != "S256" or not code_challenge:
            raise OAuthError("invalid_request", "PKCE with S256 is required")
        if not resource or "://" not in resource:
            raise OAuthError("invalid_target", "resource must be an absolute URI (RFC 8707)")
        code = secrets.token_urlsafe(24)
        self._codes[code] = _AuthCode(client_id, redirect_uri, scope, resource, code_challenge, subject,
                                      expires_at=self.clock() + self.code_ttl_s)
        out = {"code": code, "iss": self.issuer}
        if state is not None:
            out["state"] = state
        return out

    def token(self, grant_type: str, code: str, code_verifier: str, client_id: str,
              resource: str | None = None, redirect_uri: str | None = None) -> dict[str, Any]:
        """The ``/token`` call that redeems a code. PKCE binds it to the client that started the flow."""
        if grant_type != "authorization_code":
            raise OAuthError("unsupported_grant_type", grant_type)
        record = self._codes.get(code)
        if record is None or record.used or record.expires_at < self.clock():
            raise InvalidGrant("unknown, used or expired code")
        record.used = True   # single use, even when the checks below fail (replay protection)
        if record.client_id != client_id:
            raise InvalidGrant("code was issued to another client")
        if redirect_uri is not None and redirect_uri != record.redirect_uri:
            raise InvalidGrant("redirect_uri does not match the authorization request")
        if not hmac.compare_digest(challenge_for(code_verifier), record.code_challenge):
            raise InvalidGrant("PKCE verification failed")
        if resource is not None and resource != record.resource:
            raise OAuthError("invalid_target", "resource differs from the authorization request")
        return self._issue(subject=record.subject, audience=record.resource, scope=record.scope, client_id=client_id)

    # -- token exchange (RFC 8693) ------------------------------------------
    def token_exchange(self, subject_token: str, audience: str, scope: str | None = None, *, client_id: str) -> dict[str, Any]:
        """Trade a token for a downstream credential: same ``sub``, new ``aud``, ``act`` names the exchanging party.

        Policy baked in (the AS decides these, the RFC only gives the shape):
        the caller must be the audience the subject token was issued to — you
        can exchange a token you received, not one you observed — and the new
        scope must come from the user's grant or from the downstream scopes this
        resource server is registered to delegate; it can never be widened at will.
        """
        client = self.registered_client(client_id)
        claims = self.verify(subject_token)
        if client.resource_url is None or claims.get("aud") != client.resource_url:
            raise InvalidGrant(f"{client_id} is not the audience of the subject token; only the recipient may exchange it")
        granted = set(str(claims.get("scope", "")).split())
        requested = set(scope.split()) if scope else granted
        allowed = granted | set(client.delegated_scopes)
        if not requested <= allowed:
            raise InvalidScope(f"{client_id} may not request {sorted(requested - allowed)} on the user's behalf")
        act: dict[str, Any] = {"sub": client_id}
        if claims.get("act"):
            act["act"] = claims["act"]          # delegation chains nest, newest actor outermost
        expires_in = min(self.token_ttl_s, int(claims["exp"] - self.clock()))
        out = self._issue(subject=claims["sub"], audience=audience, scope=" ".join(sorted(requested)),
                          client_id=client_id, extra={"act": act}, ttl_s=expires_in)
        out["issued_token_type"] = ACCESS_TOKEN_TYPE
        return out

    # -- verification -------------------------------------------------------
    def verify(self, token: str, audience: str | None = None) -> dict[str, Any]:
        """Signature → issuer → expiry → audience. Raises a typed ``InvalidToken`` subclass."""
        claims = decode_jwt(token, self._key)
        if claims.get("iss") != self.issuer:
            raise InvalidToken(f"issuer {claims.get('iss')!r} is not {self.issuer!r}")
        if claims.get("exp", 0) < self.clock():
            raise TokenExpired("token has expired")
        if audience is not None and claims.get("aud") != audience:
            raise WrongAudience(f"token audience {claims.get('aud')!r} is not {audience!r}")
        return claims

    def verifier(self, audience: str | None = None) -> Callable[[str], dict[str, Any]]:
        """A ``token -> claims`` callable to plug into an ``McpServer`` or a downstream service."""
        return lambda token: self.verify(token, audience)

    def introspect(self, token: str) -> dict[str, Any]:
        """RFC 7662 shape: ``{"active": false}`` for anything that does not verify."""
        try:
            return {"active": True, **self.verify(token)}
        except OAuthError:
            return {"active": False}

    def issue_access_token(self, subject: str, audience: str, scope: str, ttl_s: int | None = None,
                           extra: Mapping[str, Any] | None = None, client_id: str = "direct") -> str:
        """Mint a token directly — a test seam and the stand-in for 'the user already signed in'."""
        return self._issue(subject, audience, scope, client_id=client_id, ttl_s=ttl_s, extra=extra)["access_token"]

    # -- internals ----------------------------------------------------------
    def _issue(self, subject: str, audience: str, scope: str, client_id: str,
               extra: Mapping[str, Any] | None = None, ttl_s: int | None = None) -> dict[str, Any]:
        now = int(self.clock())
        ttl = self.token_ttl_s if ttl_s is None else ttl_s
        claims = {"iss": self.issuer, "sub": subject, "aud": audience, "scope": scope, "client_id": client_id,
                  "iat": now, "exp": now + ttl, "jti": secrets.token_hex(8), **(extra or {})}
        return {"access_token": encode_jwt(claims, self._key), "token_type": "Bearer", "expires_in": ttl, "scope": scope}


# ------------------------------------------------------------- client side
def parse_www_authenticate(header: str | None) -> dict[str, str]:
    """``Bearer resource_metadata="…", error="insufficient_scope", scope="a b"`` → dict (plus ``scheme``)."""
    if not header:
        return {}
    scheme, _, rest = header.strip().partition(" ")
    out = {"scheme": scheme}
    for part in rest.split(","):
        key, sep, value = part.strip().partition("=")
        if sep:
            out[key.strip()] = value.strip().strip('"')
    return out


@dataclass
class ProtectedResourceMetadata:
    """RFC 9728: what a resource server publishes so clients can find its authorization server."""
    resource: str
    authorization_servers: list[str]
    scopes_supported: list[str] = field(default_factory=list)
    bearer_methods_supported: list[str] = field(default_factory=lambda: ["header"])

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProtectedResourceMetadata":
        return cls(resource=d["resource"], authorization_servers=list(d.get("authorization_servers", [])),
                   scopes_supported=list(d.get("scopes_supported", [])),
                   bearer_methods_supported=list(d.get("bearer_methods_supported", ["header"])))

    def to_dict(self) -> dict[str, Any]:
        return {"resource": self.resource, "authorization_servers": self.authorization_servers,
                "scopes_supported": self.scopes_supported, "bearer_methods_supported": self.bearer_methods_supported}


@dataclass
class Grant:
    """What the client keeps after a successful flow: the token plus what it was issued for."""
    access_token: str
    resource: str
    issuer: str
    scopes: set[str]
    expires_in: int


@dataclass
class OAuthClient:
    """The agent's OAuth client identity plus how it reaches authorization servers.

    ``resolve_issuer`` stands in for fetching ``{issuer}/.well-known/oauth-authorization-server``
    over HTTPS; here it returns the in-process ``AuthorizationServer`` for an issuer URL.
    """
    client_id: str
    redirect_uri: str
    resolve_issuer: Callable[[str], AuthorizationServer]


def authorize_with_pkce(client: OAuthClient, issuer: str, resource: str, subject: str, scopes: Iterable[str]) -> Grant:
    """Authorization-code flow with PKCE and a resource indicator; checks ``iss`` before redeeming the code."""
    authz = client.resolve_issuer(issuer)
    meta = authz.metadata()
    verifier = make_verifier()
    scope = " ".join(sorted(set(scopes)))
    response = authz.authorize(client_id=client.client_id, redirect_uri=client.redirect_uri, scope=scope,
                               resource=resource, code_challenge=challenge_for(verifier), subject=subject)
    if response.get("iss") != meta["issuer"]:
        raise MixUpDetected(meta["issuer"], response.get("iss"))
    token = authz.token(grant_type="authorization_code", code=response["code"], code_verifier=verifier,
                        client_id=client.client_id, resource=resource)
    return Grant(access_token=token["access_token"], resource=resource, issuer=meta["issuer"],
                 scopes=set(token["scope"].split()), expires_in=token["expires_in"])


async def discover_and_authorize(client: OAuthClient, mcp_401_headers: Mapping[str, str], subject: str,
                                 scopes: Iterable[str], fetch_json: Callable[[str], Awaitable[dict[str, Any]]]) -> Grant:
    """The MCP authorization chain: 401 → protected-resource metadata → AS metadata → PKCE → audience-bound token.

    ``fetch_json`` is how the client GETs the metadata URL named in ``WWW-Authenticate``
    (an ``McpClient.fetch_json`` works). The token's audience is the ``resource``
    the PRM declares — never a URL the client guessed.
    """
    challenge = parse_www_authenticate(_header(mcp_401_headers, "WWW-Authenticate"))
    if "resource_metadata" not in challenge:
        raise OAuthError("invalid_request", "401 without a resource_metadata challenge; nothing to discover")
    prm = ProtectedResourceMetadata.from_dict(await fetch_json(challenge["resource_metadata"]))
    if not prm.authorization_servers:
        raise OAuthError("invalid_request", "protected resource names no authorization server")
    wanted = set(scopes) | set(challenge.get("scope", "").split())
    return authorize_with_pkce(client, prm.authorization_servers[0], prm.resource, subject, wanted)


def step_up(client: OAuthClient, mcp_403_headers: Mapping[str, str], grant: Grant, subject: str) -> Grant:
    """``insufficient_scope`` → re-authorize for the union of the current and the required scopes."""
    challenge = parse_www_authenticate(_header(mcp_403_headers, "WWW-Authenticate"))
    if challenge.get("error") != "insufficient_scope":
        raise OAuthError("invalid_request", f"not an insufficient_scope challenge: {challenge}")
    required = set(challenge.get("scope", "").split())
    return authorize_with_pkce(client, grant.issuer, grant.resource, subject, grant.scopes | required)


def identity_from_claims(claims: Mapping[str, Any], token: str | None = None) -> Identity:
    """Verified claims → the agentlab ``Identity`` tools check ``required_scope`` against."""
    return Identity(subject=str(claims["sub"]), tenant=str(claims.get("tenant", "default")),
                    scopes=set(str(claims.get("scope", "")).split()), token=token)


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = {k.lower(): v for k, v in headers.items()}
    return lowered.get(name.lower())
