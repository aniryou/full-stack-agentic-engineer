"""Authority context: *own* authority vs *delegated* (on-behalf-of) authority.

Every action an agent takes runs under exactly one authority mode. Policy decisions, the
credential used, and the audit record all depend on it — so we make it an explicit object that
travels with the request rather than an implicit property of whichever token happens to be in
hand.

On Google Cloud the same distinction shows up in Cloud Audit Logs: when an agent acts on a user's
behalf through Auth Manager, the log shows both identities; when it acts on its own authority,
only the agent's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .principals import AgentIdentity, UserPrincipal
from .tokens import Claims


class AuthorityMode(str, Enum):
    OWN = "own"
    DELEGATED = "delegated"


@dataclass(frozen=True)
class AuthorityContext:
    """Who is acting, for whom, and with what scopes."""

    mode: AuthorityMode
    agent: AgentIdentity
    user: UserPrincipal | None = None
    scopes: frozenset[str] = field(default_factory=frozenset)
    chain: tuple[str, ...] = field(default_factory=tuple)  # actor SPIFFE IDs, outermost first
    claims: dict[str, Any] | None = None

    @classmethod
    def own(cls, agent: AgentIdentity, scopes: set[str] | None = None) -> AuthorityContext:
        return cls(
            mode=AuthorityMode.OWN,
            agent=agent,
            scopes=frozenset(scopes or ()),
            chain=(agent.spiffe_id,),
        )

    @classmethod
    def delegated(
        cls, agent: AgentIdentity, user: UserPrincipal, scopes: set[str] | None = None
    ) -> AuthorityContext:
        return cls(
            mode=AuthorityMode.DELEGATED,
            agent=agent,
            user=user,
            scopes=frozenset(scopes or ()),
            chain=(agent.spiffe_id,),
        )

    @classmethod
    def from_claims(cls, claims: Claims, agent: AgentIdentity | None = None) -> AuthorityContext:
        """Reconstruct the authority from a verified token.

        A token with an ``act`` claim is delegated: ``sub`` is the user, ``act.sub`` the agent.
        A token whose ``sub`` is a SPIFFE ID and has no ``act`` is the agent's own authority.
        """
        raw = claims.raw
        if claims.is_delegated:
            actor_id = (
                AgentIdentity.parse(claims.actor) if claims.actor.startswith("spiffe://") else agent
            )
            user = UserPrincipal(
                subject=claims.subject,
                email=raw.get("email"),
                tenant=raw.get("tenant"),
                groups=tuple(raw.get("groups", ())),
            )
            return cls(
                mode=AuthorityMode.DELEGATED,
                agent=actor_id or agent,  # type: ignore[arg-type]
                user=user,
                scopes=frozenset(claims.scopes),
                chain=tuple(claims.actor_chain),
                claims=raw,
            )
        subject_agent = (
            AgentIdentity.parse(claims.subject) if claims.subject.startswith("spiffe://") else agent
        )
        if subject_agent is None:
            raise ValueError("token subject is not an agent and no agent was supplied")
        return cls(
            mode=AuthorityMode.OWN,
            agent=subject_agent,
            scopes=frozenset(claims.scopes),
            chain=(subject_agent.spiffe_id,),
            claims=raw,
        )

    def audit_identities(self) -> dict[str, str | None]:
        """The pair that must appear in every audit record."""
        return {
            "agent": self.agent.spiffe_id,
            "user": (self.user.email or self.user.subject) if self.user else None,
            "authority": self.mode.value,
        }

    def with_hop(self, next_agent: AgentIdentity) -> AuthorityContext:
        """Represent a further delegation hop (agent → sub-agent) without widening scopes."""
        return AuthorityContext(
            mode=self.mode,
            agent=next_agent,
            user=self.user,
            scopes=self.scopes,
            chain=(next_agent.spiffe_id, *self.chain),
            claims=self.claims,
        )
