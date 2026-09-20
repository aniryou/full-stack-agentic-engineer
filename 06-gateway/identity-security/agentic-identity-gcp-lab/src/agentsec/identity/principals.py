"""Principals in an agentic system: users, agents (SPIFFE identities), and principal sets.

Google Cloud Agent Identity names an agent with a SPIFFE ID and exposes it to IAM as a
``principal://`` member:

    spiffe://TRUST_DOMAIN/resources/SERVICE/RESOURCE_PATH
    principal://TRUST_DOMAIN/resources/SERVICE/RESOURCE_PATH

where ``TRUST_DOMAIN`` is ``agents.global.org-ORG_ID.system.id.goog`` (organisation) or
``agents.global.project-PROJECT_NUMBER.system.id.goog`` (project without an org), ``SERVICE``
is the hosting service shortname (``aiplatform`` for Agent Engine, ``discoveryengine`` for
Gemini Enterprise) and ``RESOURCE_PATH`` is the full resource path of the deployed agent.

Fleet-level bindings use ``principalSet://`` members:

    principalSet://TRUST_DOMAIN/attribute.platformContainer/aiplatform/projects/PROJECT_NUMBER
    principalSet://TRUST_DOMAIN/attribute.platform/aiplatform

This module models those identifiers so policy code can match an agent against IAM-style
members exactly the way IAM would.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SPIFFE_RE = re.compile(
    r"^(spiffe|principal)://(?P<td>[^/]+)/resources/(?P<service>[^/]+)/(?P<path>.+)$"
)
_PSET_RE = re.compile(
    r"^principalSet://(?P<td>[^/]+)/attribute\.(?P<attr>platformContainer|platform)/(?P<value>.+)$"
)


class PrincipalError(ValueError):
    """Raised for malformed principal identifiers."""


def org_trust_domain(org_id: str | int) -> str:
    return f"agents.global.org-{org_id}.system.id.goog"


def project_trust_domain(project_number: str | int) -> str:
    return f"agents.global.project-{project_number}.system.id.goog"


@dataclass(frozen=True)
class UserPrincipal:
    """A human controller, as authenticated by the identity provider."""

    subject: str
    email: str | None = None
    tenant: str | None = None
    groups: tuple[str, ...] = field(default_factory=tuple)

    @property
    def iam_member(self) -> str:
        return f"user:{self.email}" if self.email else f"principal://idp/subject/{self.subject}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.email or self.subject


@dataclass(frozen=True)
class AgentIdentity:
    """A first-class agent principal (SPIFFE ID + IAM ``principal://`` member)."""

    trust_domain: str
    service: str
    resource_path: str

    # ---- constructors -------------------------------------------------------------------
    @classmethod
    def for_agent_engine(
        cls,
        *,
        project_number: str | int,
        location: str,
        engine_id: str,
        org_id: str | int | None = None,
    ) -> AgentIdentity:
        td = (
            org_trust_domain(org_id) if org_id is not None else project_trust_domain(project_number)
        )
        path = f"projects/{project_number}/locations/{location}/reasoningEngines/{engine_id}"
        return cls(trust_domain=td, service="aiplatform", resource_path=path)

    @classmethod
    def for_gemini_enterprise(
        cls,
        *,
        project_number: str | int,
        engine_id: str,
        org_id: str | int | None = None,
        collection: str = "default_collection",
    ) -> AgentIdentity:
        td = (
            org_trust_domain(org_id) if org_id is not None else project_trust_domain(project_number)
        )
        path = f"projects/{project_number}/locations/global/collections/{collection}/engines/{engine_id}"
        return cls(trust_domain=td, service="discoveryengine", resource_path=path)

    @classmethod
    def parse(cls, identifier: str) -> AgentIdentity:
        m = _SPIFFE_RE.match(identifier.strip())
        if not m:
            raise PrincipalError(f"not a SPIFFE/principal agent identifier: {identifier!r}")
        return cls(trust_domain=m["td"], service=m["service"], resource_path=m["path"])

    # ---- representations ---------------------------------------------------------------
    @property
    def spiffe_id(self) -> str:
        return f"spiffe://{self.trust_domain}/resources/{self.service}/{self.resource_path}"

    @property
    def iam_principal(self) -> str:
        return f"principal://{self.trust_domain}/resources/{self.service}/{self.resource_path}"

    @property
    def project_number(self) -> str | None:
        m = re.match(r"^projects/([^/]+)/", self.resource_path)
        return m.group(1) if m else None

    @property
    def platform_container(self) -> str | None:
        """The ``attribute.platformContainer`` value IAM would derive: ``SERVICE/projects/N``."""
        pn = self.project_number
        return f"{self.service}/projects/{pn}" if pn else None

    @property
    def short_name(self) -> str:
        return self.resource_path.rsplit("/", 1)[-1]

    def __str__(self) -> str:
        return self.spiffe_id


@dataclass(frozen=True)
class PrincipalSet:
    """A ``principalSet://`` member selecting many agents by attribute."""

    trust_domain: str
    attribute: str  # "platformContainer" | "platform"
    value: str

    @classmethod
    def parse(cls, member: str) -> PrincipalSet:
        m = _PSET_RE.match(member.strip())
        if not m:
            raise PrincipalError(f"not a principalSet member: {member!r}")
        return cls(trust_domain=m["td"], attribute=m["attr"], value=m["value"])

    @property
    def iam_member(self) -> str:
        return f"principalSet://{self.trust_domain}/attribute.{self.attribute}/{self.value}"

    def matches(self, agent: AgentIdentity) -> bool:
        if agent.trust_domain != self.trust_domain:
            return False
        if self.attribute == "platform":
            return agent.service == self.value
        if self.attribute == "platformContainer":
            # Exact segment match: "aiplatform/projects/123" must not match project 1234.
            return agent.platform_container == self.value
        return False  # pragma: no cover - regex restricts attributes


def member_matches(member: str, agent: AgentIdentity) -> bool:
    """Does an IAM-style member string select this agent?

    Accepts ``principal://`` (exact), ``spiffe://`` (exact) and ``principalSet://`` members.
    Anything else (``user:``, ``serviceAccount:``, ``group:``) never matches an agent.
    """
    member = member.strip()
    if member.startswith("principalSet://"):
        return PrincipalSet.parse(member).matches(agent)
    if member.startswith(("principal://", "spiffe://")):
        try:
            return AgentIdentity.parse(member) == agent
        except PrincipalError:
            return False
    return False
