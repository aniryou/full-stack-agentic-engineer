"""Profiles: ``local`` (everything in-process with fakes) or ``gcp`` (bound to Google Cloud).

All values come from environment variables prefixed ``AGENTSEC_``; see ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .identity.principals import AgentIdentity

REPO_ROOT = Path(__file__).resolve().parents[2]


class Profile(str, Enum):
    LOCAL = "local"
    GCP = "gcp"


@dataclass(frozen=True)
class Settings:
    profile: Profile = Profile.LOCAL
    project_id: str = "demo-project"
    project_number: str = "987654321098"
    org_id: str | None = "123456789012"
    location: str = "us-central1"
    agent_engine_id: str = "support-agent"
    model: str = "gemini-2.5-flash"
    policy_path: Path = REPO_ROOT / "policies" / "support-agent.yaml"
    mcp_url: str = "http://127.0.0.1:8765/mcp"
    sts_issuer: str = "https://sts.agentsec.local"
    auth_provider_short: str = "tickets-3lo"
    model_armor_template: str | None = None  # projects/P/locations/L/templates/T
    audit_log_name: str = "agentsec-audit"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        return cls(
            profile=Profile(env("AGENTSEC_PROFILE", "local")),
            project_id=env("AGENTSEC_PROJECT_ID", cls.project_id),
            project_number=env("AGENTSEC_PROJECT_NUMBER", cls.project_number),
            org_id=env("AGENTSEC_ORG_ID", cls.org_id) or None,
            location=env("AGENTSEC_LOCATION", cls.location),
            agent_engine_id=env("AGENTSEC_AGENT_ENGINE_ID", cls.agent_engine_id),
            model=env("AGENTSEC_MODEL", cls.model),
            policy_path=Path(env("AGENTSEC_POLICY", str(cls.policy_path))),
            mcp_url=env("AGENTSEC_MCP_URL", cls.mcp_url),
            sts_issuer=env("AGENTSEC_STS_ISSUER", cls.sts_issuer),
            auth_provider_short=env("AGENTSEC_AUTH_PROVIDER", cls.auth_provider_short),
            model_armor_template=env("AGENTSEC_MODEL_ARMOR_TEMPLATE") or None,
            audit_log_name=env("AGENTSEC_AUDIT_LOG", cls.audit_log_name),
        )

    @property
    def is_local(self) -> bool:
        return self.profile is Profile.LOCAL

    def agent_identity(self) -> AgentIdentity:
        return AgentIdentity.for_agent_engine(
            project_number=self.project_number,
            location=self.location,
            engine_id=self.agent_engine_id,
            org_id=self.org_id,
        )

    @property
    def auth_provider_name(self) -> str:
        return (
            f"projects/{self.project_id}/locations/global/authProviders/{self.auth_provider_short}"
        )

    @property
    def mcp_audience(self) -> str:
        """Canonical MCP server URI (RFC 8707 resource indicator) — no trailing slash, no query."""
        return self.mcp_url.split("?")[0].rstrip("/")
