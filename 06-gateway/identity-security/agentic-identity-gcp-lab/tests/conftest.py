from __future__ import annotations

import logging

import pytest

from agentsec.config import Settings
from agentsec.identity import AgentIdentity, LocalRuntimeCA, TokenIssuer, UserPrincipal
from agentsec.logging_utils import quiet_logs

quiet_logs(logging.ERROR)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def agent(settings: Settings) -> AgentIdentity:
    return settings.agent_identity()


@pytest.fixture
def other_agent() -> AgentIdentity:
    # Same org, different project → not in the support_agents principalSet.
    return AgentIdentity.for_agent_engine(
        project_number="111111111111",
        location="us-central1",
        engine_id="marketing-agent",
        org_id="123456789012",
    )


@pytest.fixture
def issuer() -> TokenIssuer:
    return TokenIssuer()


@pytest.fixture
def ca() -> LocalRuntimeCA:
    return LocalRuntimeCA()


@pytest.fixture
def ana() -> UserPrincipal:
    return UserPrincipal(subject="u-ana", email="ana@customer.example", tenant="acme")
