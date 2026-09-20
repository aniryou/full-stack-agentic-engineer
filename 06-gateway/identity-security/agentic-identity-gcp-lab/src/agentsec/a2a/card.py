"""A2A Agent Cards with declared security schemes, and detached-JWS card signing.

In A2A (v1.0) the Agent Card is the discovery document a client fetches before talking to an
agent. Two security-relevant parts:

* ``security_schemes`` + ``security_requirements`` declare *how* a client must authenticate
  (API key, HTTP bearer, OAuth 2, OpenID Connect, mutual TLS) and which scopes each skill needs.
* ``signatures`` let a client verify the card was not tampered with in transit or in a registry
  (``AgentCardSignature`` = detached JWS: ``protected`` header, ``signature``; payload is the
  canonical JSON of the card without ``signatures``).

We build a card for the reference agent that requires a bearer JWT issued by our STS with the
``agent:invoke`` scope, sign it with the agent's certificate key (the same key that backs its
SPIFFE identity), and verify it against the issuer's JWKS-style key set.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from a2a import types as a2a
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from google.protobuf.json_format import MessageToDict, ParseDict

from ..identity.principals import AgentIdentity

INVOKE_SCOPE = "agent:invoke"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def canonical_json(obj: Any) -> bytes:
    """RFC 8785-style canonical JSON (sufficient for card content: strings, ints, bools, lists)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def build_agent_card(
    *,
    agent: AgentIdentity,
    name: str,
    url: str,
    issuer: str,
    description: str = "Reference support agent",
    version: str = "1.0.0",
    scopes: list[str] | None = None,
) -> a2a.AgentCard:
    """An Agent Card that requires a bearer JWT (scope ``agent:invoke``) from ``issuer``."""
    scopes = scopes or [INVOKE_SCOPE]
    card = a2a.AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=[
            a2a.AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")
        ],
        provider=a2a.AgentProvider(organization="Acme", url="https://acme.example"),
        capabilities=a2a.AgentCapabilities(
            streaming=False, push_notifications=False, extended_agent_card=True
        ),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            a2a.AgentSkill(
                id="support",
                name="Customer support",
                description="Order lookups and refunds on behalf of an authenticated user.",
                tags=["support"],
                security_requirements=[
                    a2a.SecurityRequirement(schemes={"bearer": a2a.StringList(list=scopes)})
                ],
            )
        ],
    )
    card.security_schemes["bearer"].http_auth_security_scheme.CopyFrom(
        a2a.HTTPAuthSecurityScheme(
            description=f"JWT access token issued by {issuer} for audience {url}; agent identity {agent.spiffe_id}",
            scheme="bearer",
            bearer_format="JWT",
        )
    )
    card.security_schemes["mtls"].mtls_security_scheme.CopyFrom(
        a2a.MutualTlsSecurityScheme(
            description="Mutual TLS with a runtime-issued SPIFFE certificate"
        )
    )
    card.security_requirements.append(
        a2a.SecurityRequirement(schemes={"bearer": a2a.StringList(list=scopes)})
    )
    return card


def card_to_dict(card: a2a.AgentCard) -> dict[str, Any]:
    return MessageToDict(card, preserving_proto_field_name=False)


def card_from_dict(data: dict[str, Any]) -> a2a.AgentCard:
    return ParseDict(data, a2a.AgentCard())


def sign_agent_card(
    card: a2a.AgentCard, private_key: rsa.RSAPrivateKey, *, kid: str
) -> a2a.AgentCard:
    """Attach a detached JWS (RS256) over the canonical card-without-signatures."""
    payload = card_to_dict(card)
    payload.pop("signatures", None)
    protected = _b64url(canonical_json({"alg": "RS256", "kid": kid, "typ": "JOSE+JSON"}))
    signing_input = f"{protected}.{_b64url(canonical_json(payload))}".encode("ascii")
    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    sig = a2a.AgentCardSignature(protected=protected, signature=_b64url(signature))
    signed = a2a.AgentCard()
    signed.CopyFrom(card)
    signed.signatures.append(sig)
    return signed


def verify_agent_card(card: a2a.AgentCard, keys: dict[str, rsa.RSAPublicKey]) -> str:
    """Verify every signature on the card; returns the ``kid`` that verified. Raises on failure."""
    if not card.signatures:
        raise ValueError("card is unsigned")
    payload = card_to_dict(card)
    payload.pop("signatures", None)
    payload_b64 = _b64url(canonical_json(payload))
    for sig in card.signatures:
        header = json.loads(_b64url_decode(sig.protected))
        if header.get("alg") != "RS256":
            raise ValueError(f"unsupported alg {header.get('alg')}")
        kid = header.get("kid")
        key = keys.get(kid)
        if key is None:
            raise ValueError(f"unknown kid {kid!r}")
        signing_input = f"{sig.protected}.{payload_b64}".encode("ascii")
        key.verify(
            _b64url_decode(sig.signature), signing_input, padding.PKCS1v15(), hashes.SHA256()
        )
        return kid
    raise ValueError("no verifiable signature")


def required_scopes(card: a2a.AgentCard, skill_id: str | None = None) -> set[str]:
    """Scopes a client needs for the card (or one skill) under the ``bearer`` scheme."""
    reqs = list(card.security_requirements)
    if skill_id:
        for skill in card.skills:
            if skill.id == skill_id:
                reqs = list(skill.security_requirements) or reqs
    out: set[str] = set()
    for r in reqs:
        if "bearer" in r.schemes:
            out |= set(r.schemes["bearer"].list)
    return out
