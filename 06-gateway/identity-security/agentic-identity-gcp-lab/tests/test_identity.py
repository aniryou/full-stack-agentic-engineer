from __future__ import annotations

import datetime as dt

import jwt
import pytest

from agentsec.identity import (
    AgentIdentity,
    AuthorityContext,
    AuthorityMode,
    BindingMismatch,
    BoundaryEvaluator,
    BoundaryRule,
    DPoP,
    ExpiredToken,
    InsufficientScope,
    InvalidAudience,
    LocalAuthManager,
    LocalRuntimeCA,
    PermissionDenied,
    PrincipalSet,
    ProviderKind,
    ReplayDetected,
    TokenError,
    TokenIssuer,
    UserPrincipal,
    build_boundary,
    jwk_thumbprint,
    member_matches,
    public_jwk,
)
from agentsec.secrets import SecretValue

# ---------------------------------------------------------------------------------------------
# Principals
# ---------------------------------------------------------------------------------------------


def test_agent_identity_formats(agent: AgentIdentity):
    assert agent.spiffe_id.startswith(
        "spiffe://agents.global.org-123456789012.system.id.goog/resources/aiplatform/projects/987654321098/"
    )
    assert agent.iam_principal == agent.spiffe_id.replace("spiffe://", "principal://")
    assert AgentIdentity.parse(agent.iam_principal) == agent
    assert AgentIdentity.parse(agent.spiffe_id) == agent
    assert agent.platform_container == "aiplatform/projects/987654321098"
    assert agent.short_name == "support-agent"


def test_project_trust_domain_when_no_org():
    a = AgentIdentity.for_agent_engine(project_number=42, location="europe-west1", engine_id="x")
    assert a.trust_domain == "agents.global.project-42.system.id.goog"


def test_principal_set_matching_is_exact_on_segments(
    agent: AgentIdentity, other_agent: AgentIdentity
):
    ps = PrincipalSet.parse(
        "principalSet://agents.global.org-123456789012.system.id.goog/attribute.platformContainer/aiplatform/projects/987654321098"
    )
    assert ps.matches(agent)
    assert not ps.matches(other_agent)
    # prefix of the project number must NOT match (the classic substring bug)
    prefix = PrincipalSet.parse(
        "principalSet://agents.global.org-123456789012.system.id.goog/attribute.platformContainer/aiplatform/projects/98765432109"
    )
    assert not prefix.matches(agent)
    platform = PrincipalSet.parse(
        "principalSet://agents.global.org-123456789012.system.id.goog/attribute.platform/aiplatform"
    )
    assert platform.matches(agent) and platform.matches(other_agent)
    other_org = PrincipalSet.parse(
        "principalSet://agents.global.org-999.system.id.goog/attribute.platform/aiplatform"
    )
    assert not other_org.matches(agent)


def test_member_matches_never_matches_non_agent_members(agent: AgentIdentity):
    assert member_matches(agent.iam_principal, agent)
    assert not member_matches("user:ana@customer.example", agent)
    assert not member_matches("serviceAccount:x@p.iam.gserviceaccount.com", agent)
    assert not member_matches(
        "principal://agents.global.org-123456789012.system.id.goog/resources/aiplatform/projects/987654321098/locations/us-central1/reasoningEngines/other",
        agent,
    )


# ---------------------------------------------------------------------------------------------
# Certificates & bound tokens
# ---------------------------------------------------------------------------------------------


def test_runtime_ca_issues_spiffe_certificates(ca: LocalRuntimeCA, agent: AgentIdentity):
    cert = ca.issue(agent, ttl=dt.timedelta(hours=24))
    assert cert.spiffe_id == agent.spiffe_id
    assert ca.verify(cert.certificate)
    assert cert.is_valid_at()
    assert (cert.not_after - dt.datetime.now(dt.UTC)) < dt.timedelta(hours=25)
    other = LocalRuntimeCA()
    assert not other.verify(cert.certificate)


def test_cert_bound_token_rejects_replay_from_other_runtime(
    issuer: TokenIssuer, ca: LocalRuntimeCA, agent: AgentIdentity
):
    cert = ca.issue(agent)
    stolen = ca.issue(agent)  # a different certificate (e.g. attacker's container)
    token = issuer.mint_agent_token(cert, audience="https://api.acme.example", scope="x:read")
    claims = issuer.verify(
        token, audience="https://api.acme.example", presented_thumbprint=cert.thumbprint
    )
    assert claims.subject == agent.spiffe_id and claims.cnf["x5t#S256"] == cert.thumbprint
    with pytest.raises(BindingMismatch):
        issuer.verify(
            token, audience="https://api.acme.example", presented_thumbprint=stolen.thumbprint
        )
    with pytest.raises(BindingMismatch):
        issuer.verify(token, audience="https://api.acme.example")  # plain bearer use


def test_audience_expiry_and_scope_checks(issuer: TokenIssuer):
    token = issuer.mint(subject="u1", audience="https://a.example", scope="s1 s2", ttl=1)
    assert issuer.verify(token, audience="https://a.example", required_scopes={"s1"}).scopes == {
        "s1",
        "s2",
    }
    with pytest.raises(InvalidAudience):
        issuer.verify(token, audience="https://b.example")
    with pytest.raises(InsufficientScope):
        issuer.verify(token, audience="https://a.example", required_scopes={"s3"})
    expired = issuer.mint(
        subject="u1", audience="https://a.example", ttl=-10
    )  # beyond the 5s leeway
    with pytest.raises(ExpiredToken):
        issuer.verify(expired, audience="https://a.example")
    other_issuer = TokenIssuer(
        issuer=issuer.issuer
    )  # same name, different key → signature must fail
    with pytest.raises(Exception):
        other_issuer.verify(token, audience="https://a.example")


def test_token_exchange_records_actor_and_narrows_scope(
    issuer: TokenIssuer, ca: LocalRuntimeCA, agent: AgentIdentity, ana: UserPrincipal
):
    cert = ca.issue(agent)
    id_token = issuer.mint_user_id_token(ana, audience="https://app.acme.example")
    subject_token = issuer.mint(
        subject=ana.subject,
        audience="https://app.acme.example",
        scope="tickets:read tickets:write",
        extra={"email": ana.email},
    )
    agent_token = issuer.mint_agent_token(cert, audience=issuer.issuer)
    resp = issuer.exchange(
        subject_token=subject_token,
        subject_token_audience="https://app.acme.example",
        actor_token=agent_token,
        actor_token_audience=issuer.issuer,
        presented_thumbprint=cert.thumbprint,
        audience="https://tickets.example/mcp",
        scope="tickets:read payments:refund",  # payments:refund is not in the subject token → dropped
    )
    claims = issuer.verify(
        resp["access_token"],
        audience="https://tickets.example/mcp",
        presented_thumbprint=cert.thumbprint,
    )
    assert claims.subject == ana.subject and claims.actor == agent.spiffe_id
    assert claims.scopes == {"tickets:read"}
    assert claims.is_delegated and claims.raw["authority"] == "delegated"
    assert claims.cnf["x5t#S256"] == cert.thumbprint  # delegation stays bound to the agent's cert
    ctx = AuthorityContext.from_claims(claims)
    assert (
        ctx.mode is AuthorityMode.DELEGATED and ctx.user.email == ana.email and ctx.agent == agent
    )
    # nested chain
    sub2 = AgentIdentity.for_agent_engine(
        project_number="987654321098",
        location="us-central1",
        engine_id="refunds-subagent",
        org_id="123456789012",
    )
    sub_token = issuer.mint(subject=sub2.spiffe_id, audience=issuer.issuer)
    resp2 = issuer.exchange(
        subject_token=resp["access_token"],
        subject_token_audience="https://tickets.example/mcp",
        actor_token=sub_token,
        actor_token_audience=issuer.issuer,
        audience="https://payments.example",
        scope="tickets:read",
    )
    chain = issuer.verify(
        resp2["access_token"], audience="https://payments.example", allow_unbound=True
    ).actor_chain
    assert chain == [sub2.spiffe_id, agent.spiffe_id]
    assert jwt.get_unverified_header(id_token)["typ"] == "at+jwt"


def test_exchange_honours_may_act(
    issuer: TokenIssuer, agent: AgentIdentity, other_agent: AgentIdentity, ana: UserPrincipal
):
    """RFC 8693 §4.4: a subject token's may_act names the only actor allowed to act for it."""
    subject_token = issuer.mint(
        subject=ana.subject,
        audience="https://app.acme.example",
        scope="tickets:read",
        extra={"may_act": {"sub": agent.spiffe_id}},
    )
    common = dict(
        subject_token=subject_token,
        subject_token_audience="https://app.acme.example",
        actor_token_audience=issuer.issuer,
        audience="https://tickets.example/mcp",
        scope="tickets:read",
    )
    named = issuer.mint(subject=agent.spiffe_id, audience=issuer.issuer)
    resp = issuer.exchange(actor_token=named, **common)
    assert issuer.verify(resp["access_token"], audience="https://tickets.example/mcp").actor == (
        agent.spiffe_id
    )
    stranger = issuer.mint(subject=other_agent.spiffe_id, audience=issuer.issuer)
    with pytest.raises(TokenError, match="may_act"):
        issuer.exchange(actor_token=stranger, **common)


def test_exchange_token_type_follows_binding_kind(
    issuer: TokenIssuer, ca: LocalRuntimeCA, agent: AgentIdentity, ana: UserPrincipal
):
    """RFC 9449 §5: token_type is "DPoP" only for a cnf.jkt binding. A certificate-bound token
    (RFC 8705, cnf.x5t#S256) is presented as Bearer over mTLS, and an unbound one is Bearer."""
    subject_token = issuer.mint(
        subject=ana.subject, audience="https://app.acme.example", scope="tickets:read"
    )
    common = dict(
        subject_token=subject_token,
        subject_token_audience="https://app.acme.example",
        actor_token_audience=issuer.issuer,
        audience="https://tickets.example/mcp",
        scope="tickets:read",
    )
    # Certificate-bound actor → delegated token keeps cnf.x5t#S256, token_type Bearer.
    cert = ca.issue(agent)
    resp = issuer.exchange(
        actor_token=issuer.mint_agent_token(cert, audience=issuer.issuer),
        presented_thumbprint=cert.thumbprint,
        **common,
    )
    assert resp["token_type"] == "Bearer"
    cnf = issuer.verify(
        resp["access_token"],
        audience="https://tickets.example/mcp",
        presented_thumbprint=cert.thumbprint,
    ).cnf
    assert "x5t#S256" in cnf and "jkt" not in cnf
    # DPoP-bound actor → delegated token keeps cnf.jkt, token_type DPoP.
    key = DPoP.generate_key()
    dpop_actor = issuer.mint_dpop_bound_token(
        subject=agent.spiffe_id, audience=issuer.issuer, dpop_public_jwk=public_jwk(key)
    )
    resp = issuer.exchange(actor_token=dpop_actor, **common)
    assert resp["token_type"] == "DPoP"
    assert issuer.verify(
        resp["access_token"],
        audience="https://tickets.example/mcp",
        presented_jkt=jwk_thumbprint(public_jwk(key)),
    ).cnf == {"jkt": jwk_thumbprint(public_jwk(key))}
    # Unbound actor → Bearer.
    plain_actor = issuer.mint(subject=agent.spiffe_id, audience=issuer.issuer)
    assert issuer.exchange(actor_token=plain_actor, **common)["token_type"] == "Bearer"


def test_dpop_proof_binding_and_replay(issuer: TokenIssuer):
    key = DPoP.generate_key()
    jwk = public_jwk(key)
    token = issuer.mint_dpop_bound_token(
        subject="u1", audience="https://mcp.example/mcp", dpop_public_jwk=jwk, scope="tickets:read"
    )
    proof = DPoP.proof(key, method="POST", url="https://mcp.example/mcp", access_token=token)
    seen: set[str] = set()
    claims = DPoP.verify(
        proof, method="POST", url="https://mcp.example/mcp", access_token=token, seen_jti=seen
    )
    assert claims["jkt"] == jwk_thumbprint(jwk)
    issuer.verify(token, audience="https://mcp.example/mcp", presented_jkt=claims["jkt"])
    with pytest.raises(ReplayDetected):
        DPoP.verify(
            proof, method="POST", url="https://mcp.example/mcp", access_token=token, seen_jti=seen
        )
    with pytest.raises(BindingMismatch):
        DPoP.verify(proof, method="POST", url="https://mcp.example/mcp", access_token=token + "x")
    with pytest.raises(BindingMismatch):
        DPoP.verify(
            proof,
            method="POST",
            url="https://mcp.example/mcp",
            access_token=token,
            expected_jkt="nope",
        )
    other = DPoP.generate_key()
    wrong_key_proof = DPoP.proof(
        other, method="POST", url="https://mcp.example/mcp", access_token=token
    )
    with pytest.raises(BindingMismatch):
        issuer.verify(
            token,
            audience="https://mcp.example/mcp",
            presented_jkt=DPoP.verify(
                wrong_key_proof, method="POST", url="https://mcp.example/mcp", access_token=token
            )["jkt"],
        )


# ---------------------------------------------------------------------------------------------
# Auth Manager twin
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def auth_manager(issuer: TokenIssuer, agent: AgentIdentity) -> LocalAuthManager:
    am = LocalAuthManager(issuer)
    p = am.create_provider(
        "crm-3lo",
        ProviderKind.THREE_LEGGED_OAUTH,
        audience="https://crm.example",
        allowed_scopes=("crm.read",),
        authorization_url="https://idp.example/auth",
        token_url="https://idp.example/token",
        client_id="c1",
    )
    am.add_iam_policy_binding(p.name, agent.iam_principal)
    am.create_provider(
        "weather-key",
        ProviderKind.API_KEY,
        audience="https://weather.example",
        api_key=SecretValue("k-123", name="weather"),
    )
    am.add_iam_policy_binding(
        am.provider_name("weather-key"),
        "principalSet://agents.global.org-123456789012.system.id.goog/attribute.platformContainer/aiplatform/projects/987654321098",
    )
    am.create_provider(
        "erp-2lo",
        ProviderKind.TWO_LEGGED_OAUTH,
        audience="https://erp.example",
        allowed_scopes=("erp.read",),
        client_id="erp-client",
        token_url="https://erp.example/token",
    )
    return am


def test_three_legged_consent_flow(
    auth_manager: LocalAuthManager, issuer: TokenIssuer, agent: AgentIdentity
):
    name = auth_manager.provider_name("crm-3lo")
    r = auth_manager.retrieve_credentials(
        auth_provider=name,
        user_id="u-ana",
        caller=agent,
        scopes=["crm.read"],
        continue_uri="https://app/validate",
    )
    assert r.kind == "uri_consent_required" and "state=" in r.authorization_uri and r.consent_nonce
    auth_manager.finalize(auth_provider=name, user_id="u-ana", consent_nonce=r.consent_nonce)
    r2 = auth_manager.retrieve_credentials(
        auth_provider=name, user_id="u-ana", caller=agent, scopes=["crm.read"]
    )
    assert r2.is_success and r2.header == "Authorization: Bearer"
    claims = issuer.verify(r2.token, audience="https://crm.example")
    assert (
        claims.subject == "u-ana"
        and claims.actor == agent.spiffe_id
        and claims.scopes == {"crm.read"}
    )
    outcomes = [(e.outcome, e.agent, e.user) for e in auth_manager.access_log]
    assert outcomes[-1] == ("success", agent.spiffe_id, "u-ana")  # attributable to agent AND user


def test_auth_manager_enforces_iam_and_scopes(
    auth_manager: LocalAuthManager, agent: AgentIdentity, other_agent: AgentIdentity
):
    name = auth_manager.provider_name("crm-3lo")
    with pytest.raises(PermissionDenied):
        auth_manager.retrieve_credentials(
            auth_provider=name, user_id="u-ana", caller=other_agent, scopes=["crm.read"]
        )
    with pytest.raises(PermissionDenied):
        auth_manager.retrieve_credentials(
            auth_provider=name, user_id="u-ana", caller=agent, scopes=["crm.write"]
        )
    # API key via principalSet binding; the value is never exposed in logs
    r = auth_manager.retrieve_credentials(
        auth_provider=auth_manager.provider_name("weather-key"), user_id=None, caller=agent
    )
    assert r.is_success and r.header == "X-API-Key" and r.token == "k-123"
    with pytest.raises(PermissionDenied):
        auth_manager.retrieve_credentials(
            auth_provider=auth_manager.provider_name("erp-2lo"), user_id=None, caller=agent
        )  # no binding
    assert "k-123" not in repr(auth_manager.access_log)


def test_consent_rejection(auth_manager: LocalAuthManager, agent: AgentIdentity):
    name = auth_manager.provider_name("crm-3lo")
    r = auth_manager.retrieve_credentials(
        auth_provider=name, user_id="u-ben", caller=agent, scopes=["crm.read"]
    )
    auth_manager.finalize(
        auth_provider=name,
        user_id="u-ben",
        consent_nonce=r.consent_nonce,
        user_id_validation_state="REJECTED",
    )
    assert (
        auth_manager.retrieve_credentials(
            auth_provider=name, user_id="u-ben", caller=agent, scopes=["crm.read"]
        ).kind
        == "consent_rejected"
    )


# ---------------------------------------------------------------------------------------------
# Credential Access Boundaries
# ---------------------------------------------------------------------------------------------


def test_credential_access_boundary_shape_and_evaluation():
    boundary = build_boundary(
        BoundaryRule(
            bucket="acme-invoices", prefix="tenant-a/", roles=("roles/storage.objectViewer",)
        )
    )
    js = boundary.to_json()["accessBoundary"]["accessBoundaryRules"][0]
    assert js["availableResource"] == "//storage.googleapis.com/projects/_/buckets/acme-invoices"
    assert js["availablePermissions"] == ["inRole:roles/storage.objectViewer"]
    assert "tenant-a/" in js["availabilityCondition"]["expression"]
    ev = BoundaryEvaluator(boundary)
    assert ev.allows("gs://acme-invoices/tenant-a/2026-01.pdf", "storage.objects.get")
    assert not ev.allows("gs://acme-invoices/tenant-b/2026-01.pdf", "storage.objects.get")
    assert not ev.allows("gs://acme-invoices/tenant-a/2026-01.pdf", "storage.objects.delete")
    assert not ev.allows("gs://other-bucket/tenant-a/x", "storage.objects.get")
    with pytest.raises(ValueError):
        build_boundary(*[BoundaryRule(bucket=f"b{i}") for i in range(11)])
