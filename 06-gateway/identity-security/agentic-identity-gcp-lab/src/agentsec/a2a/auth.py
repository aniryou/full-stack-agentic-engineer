"""Inbound A2A authorization: every hop re-authorizes, delegation is exchanged not forwarded.

:func:`authorize_inbound` is what the receiving agent runs on each request: verify the bearer
token against *its own* audience, check the scopes the Agent Card advertised, and build the
:class:`AuthorityContext` it will act under. :func:`token_for_peer` is what the calling agent
runs: exchange its current authority for a token scoped to the peer (RFC 8693), so the user's
delegation propagates with the calling agent recorded as the actor, and nothing broader.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..identity.delegation import AuthorityContext
from ..identity.principals import AgentIdentity
from ..identity.tokens import Claims, InsufficientScope, TokenError, TokenIssuer
from .card import INVOKE_SCOPE


class A2AAuthError(PermissionError):
    pass


def authorize_inbound(
    headers: Mapping[str, str],
    *,
    issuer: TokenIssuer,
    audience: str,
    required: set[str] | None = None,
    this_agent: AgentIdentity,
    presented_thumbprint: str | None = None,
    max_delegation_depth: int = 3,
) -> tuple[Claims, AuthorityContext]:
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise A2AAuthError("missing bearer token")
    token = auth.split(None, 1)[1].strip()
    try:
        claims = issuer.verify(
            token,
            audience=audience,
            required_scopes=required or {INVOKE_SCOPE},
            presented_thumbprint=presented_thumbprint,
            allow_unbound=presented_thumbprint is None,
        )
    except InsufficientScope as e:
        raise A2AAuthError(f"insufficient_scope: {sorted(e.missing)}") from e
    except TokenError as e:
        raise A2AAuthError(f"invalid_token: {type(e).__name__}: {e}") from e
    if len(claims.actor_chain) > max_delegation_depth:
        raise A2AAuthError("delegation chain too deep")
    authority = AuthorityContext.from_claims(claims, agent=this_agent)
    if authority.chain and authority.chain[0] != this_agent.spiffe_id:
        # The token names the *caller* as actor; this agent becomes the next hop.
        authority = authority.with_hop(this_agent)
    return claims, authority


def token_for_peer(
    issuer: TokenIssuer,
    *,
    caller: AgentIdentity,
    current_token: str | None,
    peer_audience: str,
    scopes: set[str] | None = None,
) -> str:
    """Obtain a token to call a peer agent. Never forwards ``current_token``.

    * Delegated context (``current_token`` is the user's delegated token): exchange it with
      this agent as actor, narrowed to the peer's audience/scopes.
    * Own authority (no user token): mint the agent's own token for the peer's audience.
    """
    scopes = scopes or {INVOKE_SCOPE}
    actor_token = issuer.mint(subject=caller.spiffe_id, audience=issuer.issuer, ttl=60)
    if current_token is None:
        return issuer.mint(
            subject=caller.spiffe_id,
            audience=peer_audience,
            scope=sorted(scopes),
            ttl=120,
            extra={"authority": "own"},
        )
    resp = issuer.exchange(
        subject_token=current_token,
        actor_token=actor_token,
        actor_token_audience=issuer.issuer,
        audience=peer_audience,
        scope=sorted(scopes),
        ttl=120,
    )
    return resp["access_token"]
