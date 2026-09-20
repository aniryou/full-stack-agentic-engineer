"""A local credential broker with the shape of Google Cloud's Agent Identity **Auth Manager**.

Auth Manager is a centralised vault + broker for an agent's *outbound* credentials. You register
**auth providers** (``projects/P/locations/L/authProviders/NAME``) of three kinds:

* **3-legged OAuth** — user-delegated: the end user consents once; the agent later retrieves a
  user-scoped token without ever seeing the refresh token.
* **2-legged OAuth** — the agent's own authority against a SaaS API (client credentials).
* **API key** — the agent's own authority, key stored in the vault.

Access to a provider is governed by IAM: the agent's principal needs ``roles/agentidentity.user``
on the provider. The agent authenticates to the broker with its own SPIFFE identity, so every
retrieval — and, for 3LO, every end-user access — is attributable to the agent.

The broker's API is ``retrieveCredentials`` (returns *success* | *pending* |
*uri_consent_required* | *consent_rejected*) plus ``credentials:finalize`` for the consent
callback. :class:`LocalAuthManager` implements those semantics in memory and
:class:`LocalGcpAuthProvider` adapts it to ADK's pluggable-auth interface so the very same tool
code (``McpToolset(auth_scheme=GcpAuthProviderScheme(...))`` or ``AuthenticatedFunctionTool``)
runs offline here and against the real service on Google Cloud.
"""

from __future__ import annotations

import datetime as dt
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlencode

from ..secrets.store import SecretValue
from .principals import AgentIdentity, member_matches
from .tokens import TokenIssuer

ROLE_AUTH_PROVIDER_USER = "roles/agentidentity.user"
ROLE_AUTH_PROVIDER_ADMIN = "roles/agentidentity.admin"


class ProviderKind(str, Enum):
    THREE_LEGGED_OAUTH = "three_legged_oauth"
    TWO_LEGGED_OAUTH = "two_legged_oauth"
    API_KEY = "api_key"


class AuthManagerError(Exception):
    pass


class PermissionDenied(AuthManagerError):
    pass


class UnknownProvider(AuthManagerError):
    pass


@dataclass
class AuthProvider:
    name: str  # projects/P/locations/L/authProviders/NAME
    kind: ProviderKind
    audience: str  # the API this credential is for (token ``aud``)
    allowed_scopes: tuple[str, ...] = ()
    authorization_url: str | None = None
    token_url: str | None = None
    client_id: str | None = None
    client_secret: SecretValue | None = None
    api_key: SecretValue | None = None
    api_key_header: str = "X-API-Key"
    bindings: dict[str, set[str]] = field(default_factory=dict)  # member -> roles

    @property
    def callback_url(self) -> str:
        # Mirrors: https://agentidentitycredentials.googleapis.com/v1/{name}/oauthcallback
        return f"https://agentidentitycredentials.local/v1/{self.name}/oauthcallback"


@dataclass(frozen=True)
class RetrieveCredentialsResult:
    """The oneof returned by ``retrieveCredentials``."""

    kind: str  # "success" | "pending" | "uri_consent_required" | "consent_rejected"
    header: str | None = None  # e.g. "Authorization: Bearer" or "X-API-Key"
    token: str | None = None
    authorization_uri: str | None = None
    consent_nonce: str | None = None

    @property
    def is_success(self) -> bool:
        return self.kind == "success"


@dataclass
class _Consent:
    user_id: str
    scopes: set[str]
    nonce: str
    state: str = "pending"  # pending | granted | rejected
    granted_at: dt.datetime | None = None


@dataclass(frozen=True)
class AccessEvent:
    ts: dt.datetime
    agent: str
    user: str | None
    provider: str
    scopes: tuple[str, ...]
    outcome: str


class LocalAuthManager:
    """In-memory Auth Manager: providers, IAM bindings, consent store, token minting, audit."""

    def __init__(
        self, issuer: TokenIssuer, *, project: str = "demo-project", location: str = "global"
    ):
        self.issuer = issuer
        self.project = project
        self.location = location
        self._providers: dict[str, AuthProvider] = {}
        self._consents: dict[tuple[str, str], _Consent] = {}
        self.access_log: list[AccessEvent] = []

    # ---- admin plane ---------------------------------------------------------------------
    def provider_name(self, short: str) -> str:
        return f"projects/{self.project}/locations/{self.location}/authProviders/{short}"

    def create_provider(
        self, short: str, kind: ProviderKind, *, audience: str, **kwargs: Any
    ) -> AuthProvider:
        name = self.provider_name(short)
        if name in self._providers:
            raise AuthManagerError(f"provider exists: {name}")
        if kind is ProviderKind.API_KEY and "api_key" not in kwargs:
            raise AuthManagerError("api_key providers need api_key=SecretValue(...)")
        if kind is ProviderKind.THREE_LEGGED_OAUTH and not kwargs.get("authorization_url"):
            raise AuthManagerError("3LO providers need authorization_url/token_url/client_id")
        p = AuthProvider(name=name, kind=kind, audience=audience, **kwargs)
        self._providers[name] = p
        return p

    def add_iam_policy_binding(
        self, provider_name: str, member: str, role: str = ROLE_AUTH_PROVIDER_USER
    ) -> None:
        self._get(provider_name).bindings.setdefault(member, set()).add(role)

    def remove_iam_policy_binding(
        self, provider_name: str, member: str, role: str = ROLE_AUTH_PROVIDER_USER
    ) -> None:
        roles = self._get(provider_name).bindings.get(member)
        if roles:
            roles.discard(role)

    def list_providers(self) -> list[str]:
        return sorted(self._providers)

    def _get(self, name: str) -> AuthProvider:
        try:
            return self._providers[name]
        except KeyError as e:
            raise UnknownProvider(name) from e

    # ---- data plane ------------------------------------------------------------------------
    def _authorize(self, provider: AuthProvider, caller: AgentIdentity) -> None:
        for member, roles in provider.bindings.items():
            if ROLE_AUTH_PROVIDER_USER in roles and member_matches(member, caller):
                return
        raise PermissionDenied(
            f"{caller.spiffe_id} lacks {ROLE_AUTH_PROVIDER_USER} on {provider.name}"
        )

    def retrieve_credentials(
        self,
        *,
        auth_provider: str,
        user_id: str | None,
        caller: AgentIdentity,
        scopes: list[str] | None = None,
        continue_uri: str = "",
    ) -> RetrieveCredentialsResult:
        provider = self._get(auth_provider)
        requested = set(scopes or provider.allowed_scopes)
        try:
            self._authorize(provider, caller)
        except PermissionDenied:
            self._record(caller, user_id, provider, requested, "permission_denied")
            raise
        if (
            not requested <= set(provider.allowed_scopes)
            and provider.kind is not ProviderKind.API_KEY
        ):
            self._record(caller, user_id, provider, requested, "scope_not_allowed")
            raise PermissionDenied(
                f"scopes {sorted(requested - set(provider.allowed_scopes))} not allowed on provider"
            )

        if provider.kind is ProviderKind.API_KEY:
            self._record(caller, None, provider, (), "success")
            return RetrieveCredentialsResult(
                kind="success", header=provider.api_key_header, token=provider.api_key.reveal()
            )  # type: ignore[union-attr]

        if provider.kind is ProviderKind.TWO_LEGGED_OAUTH:
            token = self.issuer.mint(
                subject=caller.spiffe_id,
                audience=provider.audience,
                scope=sorted(requested),
                extra={"authority": "own", "client_id": provider.client_id},
            )
            self._record(caller, None, provider, requested, "success")
            return RetrieveCredentialsResult(
                kind="success", header="Authorization: Bearer", token=token
            )

        # 3-legged OAuth: user-delegated
        if not user_id:
            raise AuthManagerError("3LO providers require a user_id")
        consent = self._consents.get((provider.name, user_id))
        if consent and consent.state == "granted" and requested <= consent.scopes:
            token = self.issuer.mint(
                subject=user_id,
                audience=provider.audience,
                scope=sorted(requested),
                extra={"authority": "delegated", "act": {"sub": caller.spiffe_id}},
            )
            self._record(caller, user_id, provider, requested, "success")
            return RetrieveCredentialsResult(
                kind="success", header="Authorization: Bearer", token=token
            )
        if consent and consent.state == "rejected":
            self._record(caller, user_id, provider, requested, "consent_rejected")
            return RetrieveCredentialsResult(kind="consent_rejected")
        nonce = secrets.token_urlsafe(16)
        self._consents[(provider.name, user_id)] = _Consent(
            user_id=user_id, scopes=requested, nonce=nonce
        )
        params = {
            "client_id": provider.client_id or "",
            "response_type": "code",
            "scope": " ".join(sorted(requested)),
            "state": nonce,
            "redirect_uri": provider.callback_url,
            # PKCE (code_challenge/S256) is generated and verified by the broker itself against
            # the provider's token endpoint (Auth Manager: `enable_pkce`); the agent never sees it.
        }
        if continue_uri:
            params["continue_uri"] = continue_uri
        uri = f"{provider.authorization_url}?{urlencode(params)}"
        self._record(caller, user_id, provider, requested, "uri_consent_required")
        return RetrieveCredentialsResult(
            kind="uri_consent_required", authorization_uri=uri, consent_nonce=nonce
        )

    def finalize(
        self,
        *,
        auth_provider: str,
        user_id: str,
        consent_nonce: str,
        user_id_validation_state: str = "VALIDATED",
    ) -> None:
        """The ``credentials:finalize`` call the front-end makes after the OAuth redirect."""
        consent = self._consents.get((auth_provider, user_id))
        if not consent or consent.nonce != consent_nonce:
            raise AuthManagerError("unknown consent nonce")
        if user_id_validation_state != "VALIDATED":
            consent.state = "rejected"
            return
        consent.state = "granted"
        consent.granted_at = dt.datetime.now(dt.UTC)

    def revoke(self, *, auth_provider: str, user_id: str) -> None:
        self._consents.pop((auth_provider, user_id), None)

    def _record(
        self,
        caller: AgentIdentity,
        user: str | None,
        provider: AuthProvider,
        scopes: set[str] | tuple,
        outcome: str,
    ) -> None:
        self.access_log.append(
            AccessEvent(
                ts=dt.datetime.now(dt.UTC),
                agent=caller.spiffe_id,
                user=user,
                provider=provider.name,
                scopes=tuple(sorted(scopes)),
                outcome=outcome,
            )
        )


# ----------------------------------------------------------------------------------------------
# ADK adapter — same interface as google.adk.integrations.agent_identity.GcpAuthProvider
# ----------------------------------------------------------------------------------------------
_BRIDGE: Any = None


def make_local_gcp_auth_provider(manager: LocalAuthManager, agent: AgentIdentity):
    """Return the ADK ``BaseAuthProvider`` bound to this broker and agent identity.

    ADK keeps one provider per scheme type in a process-wide registry, so this returns a single
    bridge object and re-binds it to the given broker/agent (the runtime supplies the agent's
    identity via mTLS on Google Cloud; locally we bind it explicitly). Imported lazily so the
    identity package does not require ADK at import time.
    """
    global _BRIDGE
    from google.adk.auth.auth_credential import (
        AuthCredential,
        AuthCredentialTypes,
        HttpAuth,
        HttpCredentials,
        OAuth2Auth,
    )
    from google.adk.auth.base_auth_provider import BaseAuthProvider
    from google.adk.integrations.agent_identity import GcpAuthProviderScheme

    class LocalGcpAuthProvider(BaseAuthProvider):
        """Offline twin of ADK's ``GcpAuthProvider``; the caller identity is the running agent."""

        def __init__(self) -> None:
            self.manager: LocalAuthManager | None = None
            self.agent: AgentIdentity | None = None

        def bind(self, manager: LocalAuthManager, agent: AgentIdentity) -> LocalGcpAuthProvider:
            self.manager, self.agent = manager, agent
            return self

        @property
        def supported_auth_schemes(self):
            return (GcpAuthProviderScheme,)

        async def get_auth_credential(self, auth_config, context=None):
            scheme = auth_config.auth_scheme
            if not isinstance(scheme, GcpAuthProviderScheme):
                raise ValueError(f"Expected GcpAuthProviderScheme, got {type(scheme)}")
            if self.manager is None or self.agent is None:
                raise RuntimeError("LocalGcpAuthProvider is not bound to a broker/agent")
            user_id = getattr(context, "user_id", None)
            result = self.manager.retrieve_credentials(
                auth_provider=scheme.name,
                user_id=user_id,
                caller=self.agent,
                scopes=scheme.scopes,
                continue_uri=scheme.continue_uri or "",
            )
            if result.kind == "consent_rejected":
                raise RuntimeError("Operation failed: User consent rejected.")
            if result.kind == "uri_consent_required":
                # Signals CredentialManager to emit `adk_request_credential` to the client.
                return AuthCredential(
                    auth_type=AuthCredentialTypes.OAUTH2,
                    oauth2=OAuth2Auth(
                        auth_uri=result.authorization_uri, nonce=result.consent_nonce
                    ),
                )
            if result.kind != "success":
                raise ValueError(f"unexpected auth manager outcome: {result.kind}")
            header_name, _, header_value = (result.header or "").partition(":")
            if (
                header_name.strip().lower() == "authorization"
                and header_value.strip().lower().startswith("bearer")
            ):
                return AuthCredential(
                    auth_type=AuthCredentialTypes.HTTP,
                    http=HttpAuth(scheme="Bearer", credentials=HttpCredentials(token=result.token)),
                )
            return AuthCredential(
                auth_type=AuthCredentialTypes.HTTP,
                http=HttpAuth(
                    scheme="",
                    credentials=HttpCredentials(),
                    additional_headers={result.header: result.token},
                ),
            )

    if _BRIDGE is None:
        _BRIDGE = LocalGcpAuthProvider()
    return _BRIDGE.bind(manager, agent)
