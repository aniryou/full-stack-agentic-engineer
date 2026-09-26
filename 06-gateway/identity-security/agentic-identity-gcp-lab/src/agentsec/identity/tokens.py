"""A local security token service (STS) and DPoP implementation.

What this gives you, standards-faithfully but in ~300 lines:

* :class:`TokenIssuer` — signs RS256 JWTs with an ``iss``/``aud``/``exp``/``scope`` shape
  (RFC 9068 style), publishes a JWKS, and implements the two things an agent platform needs:

  - **Certificate-bound agent tokens** (RFC 8705): ``cnf = {"x5t#S256": <thumbprint>}``.
    A verifier that is handed the presented certificate rejects tokens bound to another one.
  - **RFC 8693 token exchange** for delegation: ``subject_token`` (the user) + ``actor_token``
    (the agent) → a token whose ``sub`` is the user and whose ``act`` claim names the agent,
    narrowed to one ``audience`` and ``scope``. Nested ``act`` chains express multi-hop delegation.

* :class:`DPoP` — RFC 9449 proof-of-possession: the client signs a per-request proof with a
  private key; the token carries ``cnf.jkt`` (the JWK thumbprint, RFC 7638). A stolen bearer
  token without the key fails verification, and ``jti`` replay is tracked.

On Google Cloud you do not run this yourself: Agent Identity mints certificate-bound tokens,
Google's STS performs exchanges for Workload Identity Federation, and Agent Gateway enforces
DPoP. The local versions exist so the notebooks can *show* audience mismatch, binding mismatch
and replay failing.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from .certs import AgentCertificate
from .principals import UserPrincipal

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"
ID_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:id_token"
JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"


# ----------------------------------------------------------------------------------------------
# Errors — deliberately specific so policy code and tests can distinguish failure modes.
# ----------------------------------------------------------------------------------------------
class TokenError(Exception):
    """Base class for token verification failures."""


class InvalidSignature(TokenError):
    pass


class ExpiredToken(TokenError):
    pass


class InvalidAudience(TokenError):
    pass


class InvalidIssuer(TokenError):
    pass


class InsufficientScope(TokenError):
    def __init__(self, missing: set[str]):
        super().__init__(f"insufficient_scope: missing {sorted(missing)}")
        self.missing = missing


class BindingMismatch(TokenError):
    """The token is bound (``cnf``) to a certificate/key the caller did not prove possession of."""


class ReplayDetected(TokenError):
    pass


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_uint(n: int) -> str:
    length = (n.bit_length() + 7) // 8
    return _b64url(n.to_bytes(length, "big"))


def public_jwk(key: rsa.RSAPrivateKey | rsa.RSAPublicKey, kid: str | None = None) -> dict[str, Any]:
    pub = key.public_key() if isinstance(key, rsa.RSAPrivateKey) else key
    numbers = pub.public_numbers()
    jwk: dict[str, Any] = {
        "kty": "RSA",
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
        "alg": "RS256",
        "use": "sig",
    }
    if kid:
        jwk["kid"] = kid
    return jwk


def jwk_thumbprint(jwk: dict[str, Any]) -> str:
    """RFC 7638 JWK thumbprint (SHA-256) for RSA keys — the ``cnf.jkt`` value for DPoP."""
    canonical = json.dumps(
        {"e": jwk["e"], "kty": jwk["kty"], "n": jwk["n"]}, separators=(",", ":"), sort_keys=True
    )
    return _b64url(hashlib.sha256(canonical.encode()).digest())


def scopes_of(claims: dict[str, Any]) -> set[str]:
    raw = claims.get("scope", "")
    if isinstance(raw, list):
        return set(raw)
    return set(str(raw).split())


@dataclass
class Claims:
    """Verified claims plus derived conveniences."""

    raw: dict[str, Any]

    @property
    def subject(self) -> str:
        return self.raw["sub"]

    @property
    def audience(self) -> str:
        aud = self.raw.get("aud")
        return aud[0] if isinstance(aud, list) else aud

    @property
    def scopes(self) -> set[str]:
        return scopes_of(self.raw)

    @property
    def actor(self) -> str | None:
        act = self.raw.get("act")
        return act.get("sub") if isinstance(act, dict) else None

    @property
    def actor_chain(self) -> list[str]:
        chain: list[str] = []
        act = self.raw.get("act")
        while isinstance(act, dict):
            chain.append(act.get("sub", "?"))
            act = act.get("act")
        return chain

    @property
    def is_delegated(self) -> bool:
        return self.actor is not None

    @property
    def cnf(self) -> dict[str, Any]:
        return self.raw.get("cnf", {}) or {}


# ----------------------------------------------------------------------------------------------
# Issuer / STS
# ----------------------------------------------------------------------------------------------
@dataclass
class TokenIssuer:
    """A local authorization server + STS.

    ``issuer`` should be an HTTPS URL in real deployments; locally any URI works.
    """

    issuer: str = "https://sts.agentsec.local"
    default_ttl: int = 300  # seconds — short-lived by default
    kid: str = field(default_factory=lambda: f"k{secrets.token_hex(4)}")
    _key: rsa.RSAPrivateKey = field(
        default_factory=lambda: rsa.generate_private_key(65537, 2048), repr=False
    )

    # ---- key material ------------------------------------------------------------------
    @property
    def public_key_pem(self) -> str:
        return (
            self._key.public_key()
            .public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
            .decode()
        )

    def jwks(self) -> dict[str, Any]:
        return {"keys": [public_jwk(self._key, self.kid)]}

    def metadata(self) -> dict[str, Any]:
        """RFC 8414 authorization server metadata (the subset MCP clients look for)."""
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.issuer}/authorize",
            "token_endpoint": f"{self.issuer}/token",
            "jwks_uri": f"{self.issuer}/.well-known/jwks.json",
            "response_types_supported": ["code"],
            "grant_types_supported": [
                "authorization_code",
                "client_credentials",
                TOKEN_EXCHANGE_GRANT,
            ],
            "code_challenge_methods_supported": ["S256"],
            "client_id_metadata_document_supported": True,
            "token_endpoint_auth_methods_supported": ["private_key_jwt", "none"],
            "dpop_signing_alg_values_supported": ["RS256"],
        }

    # ---- minting -----------------------------------------------------------------------
    def mint(
        self,
        *,
        subject: str,
        audience: str,
        scope: str | list[str] = "",
        ttl: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "sub": subject,
            "aud": audience,
            "iat": now,
            "nbf": now - 5,
            "exp": now + (ttl or self.default_ttl),
            "jti": uuid.uuid4().hex,
            "scope": " ".join(scope) if isinstance(scope, list) else scope,
        }
        if extra:
            claims.update(extra)
        return jwt.encode(
            claims, self._key, algorithm="RS256", headers={"kid": self.kid, "typ": "at+jwt"}
        )

    def mint_user_id_token(self, user: UserPrincipal, audience: str, ttl: int = 3600) -> str:
        """What an IdP would give the front-end after login (OIDC ID token shape)."""
        extra = {"email": user.email, "tenant": user.tenant, "groups": list(user.groups)}
        return self.mint(
            subject=user.subject,
            audience=audience,
            ttl=ttl,
            extra={k: v for k, v in extra.items() if v},
        )

    def mint_agent_token(
        self,
        cert: AgentCertificate,
        *,
        audience: str,
        scope: str | list[str] = "",
        ttl: int | None = None,
    ) -> str:
        """An agent's *own-authority* token, certificate-bound (RFC 8705 ``cnf.x5t#S256``)."""
        return self.mint(
            subject=cert.agent.spiffe_id,
            audience=audience,
            scope=scope,
            ttl=ttl,
            extra={"cnf": {"x5t#S256": cert.thumbprint}, "authority": "own"},
        )

    def mint_dpop_bound_token(
        self,
        *,
        subject: str,
        audience: str,
        dpop_public_jwk: dict[str, Any],
        scope: str | list[str] = "",
        ttl: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """A token bound to a DPoP key (RFC 9449 ``cnf.jkt``)."""
        merged = {"cnf": {"jkt": jwk_thumbprint(dpop_public_jwk)}}
        if extra:
            merged.update(extra)
        return self.mint(subject=subject, audience=audience, scope=scope, ttl=ttl, extra=merged)

    # ---- RFC 8693 token exchange --------------------------------------------------------
    def exchange(
        self,
        *,
        subject_token: str,
        subject_token_type: str = ID_TOKEN_TYPE,
        actor_token: str | None = None,
        actor_token_type: str = ACCESS_TOKEN_TYPE,
        audience: str,
        scope: str | list[str] = "",
        subject_token_audience: str | None = None,
        actor_token_audience: str | None = None,
        ttl: int | None = None,
        presented_thumbprint: str | None = None,
    ) -> dict[str, Any]:
        """Exchange a user token (+ optional agent actor token) for a delegated, narrowed token.

        Returns the RFC 8693 token response shape. The issued token's ``sub`` is the user and
        ``act.sub`` is the agent; if the subject token was itself delegated, the existing ``act``
        is nested (``act.act``) to preserve the chain.
        """
        subject = self.verify(subject_token, audience=subject_token_audience, allow_unbound=True)
        actor: Claims | None = None
        if actor_token is not None:
            actor = self.verify(
                actor_token,
                audience=actor_token_audience,
                presented_thumbprint=presented_thumbprint,
                allow_unbound=presented_thumbprint is None,
            )
        requested = set(scope.split()) if isinstance(scope, str) else set(scope)
        if subject.raw.get("scope"):
            # A delegated token can never be broader than what the subject already had.
            narrowed = requested & subject.scopes if requested else subject.scopes
        else:
            narrowed = requested
        extra: dict[str, Any] = {"authority": "delegated" if actor else "own"}
        for claim in ("email", "tenant", "groups"):
            if claim in subject.raw:
                extra[claim] = subject.raw[claim]
        if actor:
            act: dict[str, Any] = {"sub": actor.subject}
            if "act" in subject.raw:
                act["act"] = subject.raw["act"]
            extra["act"] = act
            if actor.cnf:
                extra["cnf"] = actor.cnf  # keep the delegated token bound to the agent's cert/key
        token = self.mint(
            subject=subject.subject, audience=audience, scope=sorted(narrowed), ttl=ttl, extra=extra
        )
        # RFC 9449 §5: "DPoP" only for a key-bound (cnf.jkt) token. A certificate-bound token
        # (RFC 8705, cnf.x5t#S256) is still a Bearer token type, presented over mutual TLS.
        dpop_bound = actor is not None and "jkt" in actor.cnf
        return {
            "access_token": token,
            "issued_token_type": ACCESS_TOKEN_TYPE,
            "token_type": "DPoP" if dpop_bound else "Bearer",
            "expires_in": ttl or self.default_ttl,
            "scope": " ".join(sorted(narrowed)),
        }

    # ---- verification ------------------------------------------------------------------
    def verify(
        self,
        token: str,
        *,
        audience: str | None = None,
        required_scopes: set[str] | None = None,
        presented_thumbprint: str | None = None,
        presented_jkt: str | None = None,
        allow_unbound: bool = False,
    ) -> Claims:
        """Verify signature, issuer, expiry, audience, scopes and sender-binding.

        ``presented_thumbprint`` is the ``x5t#S256`` of the TLS client certificate the caller
        presented (mTLS); ``presented_jkt`` is the thumbprint of the DPoP key whose proof was
        verified. If the token carries a ``cnf`` claim, one of them must match unless
        ``allow_unbound`` is set (only appropriate inside the STS itself).
        """
        options = {"verify_aud": audience is not None, "require": ["exp", "iat", "sub", "iss"]}
        try:
            raw = jwt.decode(
                token,
                self._key.public_key(),
                algorithms=["RS256"],
                audience=audience,
                issuer=self.issuer,
                options=options,
                leeway=5,
            )
        except jwt.ExpiredSignatureError as e:
            raise ExpiredToken(str(e)) from e
        except jwt.InvalidAudienceError as e:
            raise InvalidAudience(str(e)) from e
        except jwt.InvalidIssuerError as e:
            raise InvalidIssuer(str(e)) from e
        except jwt.PyJWTError as e:
            raise InvalidSignature(str(e)) from e
        claims = Claims(raw)
        if required_scopes:
            missing = set(required_scopes) - claims.scopes
            if missing:
                raise InsufficientScope(missing)
        cnf = claims.cnf
        if cnf and not allow_unbound:
            bound_ok = False
            if "x5t#S256" in cnf and presented_thumbprint == cnf["x5t#S256"]:
                bound_ok = True
            if "jkt" in cnf and presented_jkt == cnf["jkt"]:
                bound_ok = True
            if not bound_ok:
                raise BindingMismatch(
                    "token is sender-constrained; caller did not prove possession"
                )
        return claims


# ----------------------------------------------------------------------------------------------
# DPoP (RFC 9449)
# ----------------------------------------------------------------------------------------------
class DPoP:
    """Create and verify DPoP proofs."""

    @staticmethod
    def generate_key() -> rsa.RSAPrivateKey:
        return rsa.generate_private_key(65537, 2048)

    @staticmethod
    def proof(
        key: rsa.RSAPrivateKey,
        *,
        method: str,
        url: str,
        access_token: str | None = None,
        nonce: str | None = None,
    ) -> str:
        """Sign a proof for one HTTP request. ``ath`` binds the proof to the access token."""
        claims: dict[str, Any] = {
            "jti": uuid.uuid4().hex,
            "htm": method.upper(),
            "htu": url.split("#")[0].split("?")[0],
            "iat": int(time.time()),
        }
        if access_token:
            claims["ath"] = _b64url(hashlib.sha256(access_token.encode("ascii")).digest())
        if nonce:
            claims["nonce"] = nonce
        headers = {"typ": "dpop+jwt", "alg": "RS256", "jwk": public_jwk(key)}
        return jwt.encode(claims, key, algorithm="RS256", headers=headers)

    @staticmethod
    def verify(
        proof: str,
        *,
        method: str,
        url: str,
        access_token: str | None = None,
        expected_jkt: str | None = None,
        seen_jti: set[str] | None = None,
        max_age: int = 300,
    ) -> dict[str, Any]:
        """Verify a proof; returns its claims plus the key thumbprint under ``jkt``.

        Checks: header typ, embedded public key signs the proof, ``htm``/``htu`` match the
        request, ``iat`` is fresh, ``ath`` matches the presented access token, the key's
        thumbprint matches the token's ``cnf.jkt``, and ``jti`` has not been seen (replay).
        """
        try:
            header = jwt.get_unverified_header(proof)
        except jwt.PyJWTError as e:
            raise InvalidSignature(f"malformed DPoP proof: {e}") from e
        if header.get("typ") != "dpop+jwt" or "jwk" not in header:
            raise InvalidSignature("DPoP proof must have typ=dpop+jwt and an embedded jwk")
        jwk = header["jwk"]
        if "d" in jwk:
            raise InvalidSignature("DPoP proof jwk must not contain private key material")
        public_key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
        try:
            claims = jwt.decode(
                proof, public_key, algorithms=["RS256"], options={"verify_aud": False}
            )
        except jwt.PyJWTError as e:
            raise InvalidSignature(f"DPoP signature invalid: {e}") from e
        if claims.get("htm") != method.upper():
            raise InvalidSignature("DPoP htm mismatch")
        if claims.get("htu") != url.split("#")[0].split("?")[0]:
            raise InvalidSignature("DPoP htu mismatch")
        if abs(int(time.time()) - int(claims.get("iat", 0))) > max_age:
            raise ExpiredToken("DPoP proof too old")
        if access_token is not None:
            expected_ath = _b64url(hashlib.sha256(access_token.encode("ascii")).digest())
            if claims.get("ath") != expected_ath:
                raise BindingMismatch("DPoP ath does not match presented access token")
        jkt = jwk_thumbprint(jwk)
        if expected_jkt is not None and jkt != expected_jkt:
            raise BindingMismatch("DPoP key does not match token cnf.jkt")
        if seen_jti is not None:
            jti = claims.get("jti")
            if not jti:
                raise InvalidSignature("DPoP proof missing jti")
            if jti in seen_jti:
                raise ReplayDetected(f"DPoP proof jti {jti} already used")
            seen_jti.add(jti)
        claims["jkt"] = jkt
        return claims
