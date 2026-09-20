"""agentlab.auth.oauth: PKCE, the authorization-code flow, audience binding, expiry, token exchange, discovery and step-up."""
import pytest

from agentlab.agents import SideEffect, tool
from agentlab.auth import (AuthorizationServer, InvalidGrant, InvalidScope, InvalidSignature, InvalidToken, MixUpDetected,
                         OAuthClient, OAuthError, ProtectedResourceMetadata, TokenExpired, WrongAudience, authorize_with_pkce,
                         challenge_for, discover_and_authorize, identity_from_claims, make_verifier, parse_www_authenticate,
                         peek_claims, step_up)
from agentlab.mcp import Forbidden, InProcessTransport, McpClient, McpServer, Unauthorized

ISSUER = "https://idp.example"
RESOURCE = "https://orders.mcp.example/mcp"
REDIRECT = "https://agent.example/callback"


class Clock:
    def __init__(self, now: float = 1_700_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_as(clock=None) -> AuthorizationServer:
    authz = AuthorizationServer(ISSUER, clock=clock or Clock(), scopes_supported=("orders:read", "orders:write"))
    authz.register_client("agent-app", [REDIRECT])
    return authz


def oauth_client(authz: AuthorizationServer) -> OAuthClient:
    return OAuthClient("agent-app", REDIRECT, resolve_issuer={authz.issuer: authz}.__getitem__)


def run_flow(authz: AuthorizationServer, scope="orders:read", resource=RESOURCE, verifier=None, subject="alice"):
    verifier = verifier or make_verifier()
    response = authz.authorize("agent-app", REDIRECT, scope, resource, challenge_for(verifier), subject)
    return response, verifier


# ------------------------------------------------------------------- PKCE
def test_pkce_challenge_is_s256_of_the_verifier():
    v = make_verifier()
    assert 43 <= len(v) <= 128 and challenge_for(v) != v
    assert challenge_for("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"   # RFC 7636 appendix B


def test_authorization_code_happy_path_binds_audience_and_scope():
    authz = make_as()
    response, verifier = run_flow(authz)
    assert response["iss"] == ISSUER
    token = authz.token("authorization_code", response["code"], verifier, "agent-app", resource=RESOURCE)
    claims = authz.verify(token["access_token"], audience=RESOURCE)
    assert claims["sub"] == "alice" and claims["aud"] == RESOURCE and claims["scope"] == "orders:read"
    assert claims["iss"] == ISSUER and claims["exp"] > claims["iat"] and claims["jti"]
    assert authz.metadata()["code_challenge_methods_supported"] == ["S256"]
    assert authz.metadata()["authorization_response_iss_parameter_supported"] is True


def test_wrong_verifier_rejected_and_code_is_single_use():
    authz = make_as()
    response, verifier = run_flow(authz)
    with pytest.raises(InvalidGrant):
        authz.token("authorization_code", response["code"], make_verifier(), "agent-app")
    with pytest.raises(InvalidGrant):                       # the failed attempt burned the code
        authz.token("authorization_code", response["code"], verifier, "agent-app")
    response, verifier = run_flow(authz)
    authz.token("authorization_code", response["code"], verifier, "agent-app")
    with pytest.raises(InvalidGrant):                       # replaying a redeemed code fails too
        authz.token("authorization_code", response["code"], verifier, "agent-app")


def test_resource_indicator_must_be_absolute_and_stable():
    authz = make_as()
    with pytest.raises(OAuthError) as e:
        authz.authorize("agent-app", REDIRECT, "orders:read", "orders", challenge_for(make_verifier()), "alice")
    assert e.value.error == "invalid_target"
    response, verifier = run_flow(authz)
    with pytest.raises(OAuthError):
        authz.token("authorization_code", response["code"], verifier, "agent-app", resource="https://other.example/mcp")


# ------------------------------------------------------------ verification
def test_wrong_audience_rejected():
    authz = make_as()
    token = authz.issue_access_token("alice", RESOURCE, "orders:read")
    with pytest.raises(WrongAudience):
        authz.verify(token, audience="https://ledger.example/api")
    assert authz.verify(token, audience=RESOURCE)["sub"] == "alice"


def test_expired_token_rejected_without_sleeping():
    clock = Clock()
    authz = make_as(clock)
    token = authz.issue_access_token("alice", RESOURCE, "orders:read", ttl_s=60)
    clock.now += 61
    with pytest.raises(TokenExpired):
        authz.verify(token)
    assert authz.introspect(token) == {"active": False}


def test_tampered_or_foreign_tokens_rejected():
    authz = make_as()
    token = authz.issue_access_token("alice", RESOURCE, "orders:read")
    header, payload, sig = token.split(".")
    with pytest.raises(InvalidSignature):
        authz.verify(f"{header}.{payload[:-2]}xx.{sig}")
    other = AuthorizationServer("https://evil.example")
    with pytest.raises(InvalidSignature):                   # different key: the signature fails before the issuer is even read
        authz.verify(other.issue_access_token("alice", RESOURCE, "orders:read"))
    with pytest.raises(InvalidToken):
        authz.verify("not.a.jwt")


# ----------------------------------------------------------- token exchange
def test_token_exchange_keeps_sub_adds_act_and_narrows_scope():
    authz = make_as()
    authz.register_client("orders-mcp", resource_url=RESOURCE)
    user_token = authz.issue_access_token("alice", RESOURCE, "orders:read orders:write")
    out = authz.token_exchange(user_token, audience="https://ledger.example/api", scope="orders:read", client_id="orders-mcp")
    claims = authz.verify(out["access_token"], audience="https://ledger.example/api")
    assert claims["sub"] == "alice" and claims["act"] == {"sub": "orders-mcp"} and claims["scope"] == "orders:read"
    assert out["issued_token_type"].endswith(":access_token")
    # a second hop nests the actors, newest outermost
    authz.register_client("ledger-svc", resource_url="https://ledger.example/api")
    hop2 = authz.token_exchange(out["access_token"], audience="https://core.example", client_id="ledger-svc")
    assert peek_claims(hop2["access_token"])["act"] == {"sub": "ledger-svc", "act": {"sub": "orders-mcp"}}


def test_token_exchange_policy_only_recipient_may_exchange_and_never_widen():
    authz = make_as()
    authz.register_client("orders-mcp", resource_url=RESOURCE, delegated_scopes=["ledger:read"])
    authz.register_client("bystander", resource_url="https://bystander.example")
    user_token = authz.issue_access_token("alice", RESOURCE, "orders:read")
    with pytest.raises(InvalidGrant):                       # not the token's audience: observed, not received
        authz.token_exchange(user_token, audience="https://ledger.example/api", client_id="bystander")
    with pytest.raises(InvalidScope):                       # neither granted by the user nor registered for delegation
        authz.token_exchange(user_token, audience="https://ledger.example/api", scope="ledger:admin", client_id="orders-mcp")
    out = authz.token_exchange(user_token, audience="https://ledger.example/api", scope="ledger:read", client_id="orders-mcp")
    assert peek_claims(out["access_token"])["scope"] == "ledger:read"   # the registered downstream scope is fine


# ------------------------------------------------------------ client side
def test_parse_www_authenticate_and_prm_shape():
    parsed = parse_www_authenticate('Bearer resource_metadata="https://o.example/mcp/.well-known/oauth-protected-resource", error="insufficient_scope", scope="a b"')
    assert parsed == {"scheme": "Bearer", "resource_metadata": "https://o.example/mcp/.well-known/oauth-protected-resource",
                      "error": "insufficient_scope", "scope": "a b"}
    prm = ProtectedResourceMetadata.from_dict({"resource": RESOURCE, "authorization_servers": [ISSUER]})
    assert prm.to_dict()["bearer_methods_supported"] == ["header"] and prm.authorization_servers == [ISSUER]


def test_iss_mismatch_is_detected():
    class Impostor(AuthorizationServer):
        """Publishes our IdP's issuer in its metadata, but its authorization responses carry its own ``iss``."""
        def metadata(self):
            return {**super().metadata(), "issuer": ISSUER}

    impostor = Impostor("https://rogue.example")
    impostor.register_client("agent-app", [REDIRECT])
    client = OAuthClient("agent-app", REDIRECT, resolve_issuer=lambda iss: impostor)
    with pytest.raises(MixUpDetected):
        authorize_with_pkce(client, ISSUER, RESOURCE, "alice", {"orders:read"})


@tool
def get_order(order_id: str) -> dict:
    """Look up an order."""
    return {"id": order_id}


@tool(side_effect=SideEffect.IRREVERSIBLE, requires_confirmation=False, required_scope="orders:write")
def cancel_order(order_id: str) -> dict:
    """Cancel an order."""
    return {"id": order_id, "status": "cancelled"}


async def test_discover_and_authorize_walks_the_chain_and_step_up_adds_scope():
    authz = make_as()
    server = McpServer("orders", [get_order, cancel_order], resource_url=RESOURCE, token_verifier=authz.verify,
                       authorization_servers=[ISSUER], scopes_supported=["orders:read", "orders:write"])
    mcp = McpClient(InProcessTransport(server))
    client = oauth_client(authz)
    with pytest.raises(Unauthorized) as e:
        await mcp.call_tool("get_order", {"order_id": "ORD-1"})
    grant = await discover_and_authorize(client, e.value.headers, subject="alice", scopes={"orders:read"}, fetch_json=mcp.fetch_json)
    assert grant.resource == RESOURCE and grant.issuer == ISSUER and grant.scopes == {"orders:read"}
    mcp.bearer_token = grant.access_token
    assert (await mcp.call_tool("get_order", {"order_id": "ORD-1"}))["isError"] is False
    with pytest.raises(Forbidden) as e:
        await mcp.call_tool("cancel_order", {"order_id": "ORD-1"})
    assert e.value.required_scope == "orders:write"
    grant = step_up(client, e.value.headers, grant, subject="alice")
    assert grant.scopes == {"orders:read", "orders:write"}
    mcp.bearer_token = grant.access_token
    assert (await mcp.call_tool("cancel_order", {"order_id": "ORD-1"}))["structuredContent"]["status"] == "cancelled"
    assert peek_claims(grant.access_token)["sub"] == "alice"


def test_identity_from_claims_maps_scopes_and_keeps_the_token():
    authz = make_as()
    token = authz.issue_access_token("alice", RESOURCE, "orders:read orders:write", extra={"tenant": "bank-sg"})
    identity = identity_from_claims(authz.verify(token), token=token)
    assert identity.subject == "alice" and identity.tenant == "bank-sg"
    assert identity.has_scope("orders:write") and not identity.has_scope("ledger:admin") and identity.token == token
