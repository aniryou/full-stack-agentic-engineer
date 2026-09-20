"""MCP server as OAuth 2.1 resource server (audience, scopes, PRM, DPoP) and A2A card security."""

from __future__ import annotations

import json

import httpx
import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from agentsec.a2a import (
    A2AAuthError,
    authorize_inbound,
    build_agent_card,
    card_from_dict,
    card_to_dict,
    required_scopes,
    sign_agent_card,
    token_for_peer,
    verify_agent_card,
)
from agentsec.agents import LocalStack, Step, build_support_agent
from agentsec.audit import AuditLog
from agentsec.config import Settings
from agentsec.identity import AgentIdentity, DPoP, TokenIssuer, UserPrincipal, public_jwk
from agentsec.mcp import (
    SCOPE_READ,
    SCOPE_WRITE,
    ServerThread,
    build_server,
    delegated_token_minter,
    free_port,
    make_mcp_toolset,
)
from agentsec.runtime import run_turn, seed_session

JSONRPC_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
    "MCP-Protocol-Version": "2025-11-25",
}


@pytest.fixture(scope="module")
def issuer() -> TokenIssuer:
    return TokenIssuer()


@pytest.fixture(scope="module")
def server(issuer: TokenIssuer):
    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    srv = build_server(issuer, resource_url=url, audit=AuditLog())
    with ServerThread(srv.app(), port=port) as thread:
        yield srv, thread.url


@pytest.fixture(scope="module")
def dpop_server(issuer: TokenIssuer):
    port = free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    srv = build_server(issuer, resource_url=url, audit=AuditLog(), require_dpop=True)
    with ServerThread(srv.app(require_dpop=True), port=port) as thread:
        yield srv, thread.url


def _rpc(
    url: str, token: str | None, extra_headers: dict | None = None, body: dict | None = None
) -> httpx.Response:
    headers = dict(HEADERS)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    headers.update(extra_headers or {})
    return httpx.post(url, headers=headers, json=body or JSONRPC_LIST, timeout=10)


def test_protected_resource_metadata_and_401_challenge(server):
    srv, url = server
    base = url.rsplit("/mcp", 1)[0]
    prm = httpx.get(f"{base}/.well-known/oauth-protected-resource/mcp", timeout=5)
    assert prm.status_code == 200
    doc = prm.json()
    assert (
        doc["resource"].rstrip("/") == url
        and doc["authorization_servers"][0].rstrip("/") == srv.verifier.issuer.issuer
    )
    assert SCOPE_READ in doc["scopes_supported"]
    r = _rpc(url, None)
    assert r.status_code == 401
    www = r.headers["www-authenticate"]
    assert www.startswith("Bearer") and "resource_metadata=" in www


def test_audience_and_scope_enforcement(server, issuer: TokenIssuer):
    srv, url = server
    wrong_aud = issuer.mint(subject="u-ana", audience="https://other.example/mcp", scope=SCOPE_READ)
    assert _rpc(url, wrong_aud).status_code == 401  # signed by our AS but not for this server
    no_scope = issuer.mint(subject="u-ana", audience=url, scope="")
    r = _rpc(url, no_scope)
    assert r.status_code == 403 and "insufficient_scope" in r.headers["www-authenticate"]
    ok = issuer.mint(subject="u-ana", audience=url, scope=SCOPE_READ)
    r = _rpc(url, ok)
    assert r.status_code == 200
    tools = {t["name"]: t for t in r.json()["result"]["tools"]}
    assert tools["get_ticket"]["annotations"]["readOnlyHint"] is True
    assert tools["refund_ticket"]["annotations"]["destructiveHint"] is True
    # tool-level scope: read token cannot refund
    call = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "refund_ticket", "arguments": {"ticket_id": "T-1", "amount": 10}},
    }
    r = _rpc(url, ok, body=call)
    assert r.status_code == 200 and r.json()["result"]["isError"] is True
    assert "insufficient_scope" in json.dumps(r.json())


def test_dpop_required_server_rejects_bearer_and_replay(dpop_server, issuer: TokenIssuer):
    srv, url = dpop_server
    key = DPoP.generate_key()
    token = issuer.mint_dpop_bound_token(
        subject="u-ana", audience=url, dpop_public_jwk=public_jwk(key), scope=SCOPE_READ
    )
    assert _rpc(url, token).status_code == 401  # bearer presentation of a bound token
    proof = DPoP.proof(key, method="POST", url=url, access_token=token)
    r = _rpc(url, None, {"Authorization": f"DPoP {token}", "DPoP": proof})
    assert r.status_code == 200
    r = _rpc(url, None, {"Authorization": f"DPoP {token}", "DPoP": proof})  # replayed proof
    assert r.status_code == 401 and "already used" in r.headers["www-authenticate"]
    other = DPoP.generate_key()
    bad = DPoP.proof(other, method="POST", url=url, access_token=token)
    assert _rpc(url, None, {"Authorization": f"DPoP {token}", "DPoP": bad}).status_code == 401


async def test_adk_toolset_uses_delegated_audience_bound_tokens(server, issuer: TokenIssuer):
    srv, url = server
    settings = Settings(mcp_url=url, sts_issuer=issuer.issuer)
    stack = LocalStack.create(settings)
    stack.issuer = issuer  # share the AS with the server
    minter = delegated_token_minter(
        issuer, agent=stack.agent_id, audience=url, scopes=[SCOPE_READ, SCOPE_WRITE]
    )
    toolset = make_mcp_toolset(url, token_minter=minter)
    agent = build_support_agent(model=stack.llm, extra_tools=[toolset])
    runner = Runner(
        app_name="t", agent=agent, plugins=[stack.plugin], session_service=InMemorySessionService()
    )
    try:
        await seed_session(
            runner,
            user_id="u-ana",
            session_id="s",
            user={"subject": "u-ana", "email": "ana@customer.example"},
            scopes=[SCOPE_READ, SCOPE_WRITE, "agent:invoke"],
        )
        stack.script(
            Step.call("tickets_list_tickets"),
            Step.call(
                "tickets_get_ticket", ticket_id="T-3"
            ),  # Ben's ticket → forbidden at the server
            Step.call("tickets_refund_ticket", ticket_id="T-2", amount=35.0),
            Step.say("ok"),
        )
        r = await run_turn(runner, user_id="u-ana", session_id="s", message="tickets")
        by_name = {t["name"]: t["response"] for t in r.tool_responses}
        listed = by_name["tickets_list_tickets"]["structuredContent"]["tickets"]
        assert {t["id"] for t in listed} == {
            "T-1",
            "T-2",
        }  # row-level filtering by delegated subject
        assert by_name["tickets_get_ticket"]["structuredContent"]["error"] == "forbidden"
        assert by_name["tickets_refund_ticket"]["structuredContent"]["ticket_id"] == "T-2"
        assert by_name["tickets_list_tickets"]["content"][0]["text"].startswith(
            "<untrusted_content"
        )
        # upstream call used a separate token for the payments audience (no passthrough)
        call = srv.payments.calls[-1]
        claims = issuer.verify(call["token"], audience="https://payments.acme.example")
        assert (
            claims.subject == srv.payments.server_identity
            and claims.actor == stack.agent_id.spiffe_id
        )
        assert claims.raw["on_behalf_of"] == "u-ana"
        assert (
            call["token"] != minter
        )  # trivially different objects; the point is the audience differs
    finally:
        await toolset.close()


def test_agent_card_sign_verify_and_tamper(issuer: TokenIssuer, agent: AgentIdentity, ca):
    cert = ca.issue(agent)
    card = build_agent_card(
        agent=agent, name="support", url="https://agents.acme.example/support", issuer=issuer.issuer
    )
    assert required_scopes(card) == {"agent:invoke"}
    assert card.security_schemes["bearer"].http_auth_security_scheme.scheme == "bearer"
    signed = sign_agent_card(card, cert.private_key, kid=cert.thumbprint)
    keys = {cert.thumbprint: cert.private_key.public_key()}
    assert verify_agent_card(signed, keys) == cert.thumbprint
    # round-trip through JSON (what a registry stores) still verifies
    assert verify_agent_card(card_from_dict(card_to_dict(signed)), keys) == cert.thumbprint
    tampered = card_to_dict(signed)
    tampered["supportedInterfaces"][0]["url"] = "https://evil.example/support"
    with pytest.raises(Exception):
        verify_agent_card(card_from_dict(tampered), keys)
    with pytest.raises(ValueError):
        verify_agent_card(card, keys)  # unsigned


def test_inbound_authorization_and_delegation_hop(
    issuer: TokenIssuer, agent: AgentIdentity, ana: UserPrincipal
):
    peer = AgentIdentity.for_agent_engine(
        project_number="987654321098",
        location="us-central1",
        engine_id="refunds-agent",
        org_id="123456789012",
    )
    peer_url = "https://agents.acme.example/refunds"
    user_token = issuer.mint(
        subject=ana.subject,
        audience="https://app.acme.example",
        scope="agent:invoke tickets:read",
        extra={"email": ana.email},
    )
    token = token_for_peer(issuer, caller=agent, current_token=user_token, peer_audience=peer_url)
    claims, authority = authorize_inbound(
        {"Authorization": f"Bearer {token}"}, issuer=issuer, audience=peer_url, this_agent=peer
    )
    assert claims.actor == agent.spiffe_id and authority.user.email == ana.email
    assert authority.chain == (peer.spiffe_id, agent.spiffe_id)  # hop recorded, nothing widened
    assert authority.scopes == {"agent:invoke"}
    with pytest.raises(A2AAuthError):
        authorize_inbound(
            {"Authorization": f"Bearer {user_token}"},
            issuer=issuer,
            audience=peer_url,
            this_agent=peer,
        )  # forwarded user token: wrong audience
    own = token_for_peer(issuer, caller=agent, current_token=None, peer_audience=peer_url)
    _, own_auth = authorize_inbound(
        {"Authorization": f"Bearer {own}"}, issuer=issuer, audience=peer_url, this_agent=peer
    )
    assert own_auth.mode.value == "own" and own_auth.user is None
