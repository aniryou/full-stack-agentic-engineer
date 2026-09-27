"""§8: the gateway as an MCP client, against a fake authorization server and MCP server."""
import pytest

from gwcore.mcp_authz import (FakeAS, FakeMCPServer, HMACSigner, MCPClient, Network, as_metadata_urls, iss_ok, prm_urls, s256,
                              validate_cimd)
from gwcore.providers import Clock

CID = "https://gateway.example.com/oauth/client.json"
DOCS = {CID: {"client_id": CID, "client_name": "llm-gateway", "redirect_uris": ["https://gateway.example.com/cb"]}}
RES = "https://mcp.example.com/mcp"
TOOLS = {"list_tickets": "tickets:read", "close_ticket": "tickets:write"}


def world(*, dpop=False, pkce=True, well_known=2):
    clock, net = Clock(), Network()
    signer = HMACSigner(b"gateway-dpop-key") if dpop else None
    auth = FakeAS("https://auth.example.com/tenant1", clock, documents=DOCS, pkce=pkce, dpop_signer=signer)
    server = FakeMCPServer(RES, auth, TOOLS, dpop_nonce=dpop)
    net.add_as(auth, well_known=as_metadata_urls(auth.issuer)[well_known])
    net.add_mcp(server)
    return clock, auth, server, MCPClient(CID, "https://gateway.example.com/cb", net, clock, signer=signer)


def test_rfc7636_appendix_b():
    assert s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_discovery_orders():
    assert prm_urls("https://mcp.example.com/public/mcp") == [
        "https://mcp.example.com/.well-known/oauth-protected-resource/public/mcp", "https://mcp.example.com/.well-known/oauth-protected-resource"]
    assert prm_urls(RES, 'Bearer resource_metadata="https://x.example/prm"') == ["https://x.example/prm"]
    assert as_metadata_urls("https://auth.example.com/tenant1") == [
        "https://auth.example.com/.well-known/oauth-authorization-server/tenant1",
        "https://auth.example.com/.well-known/openid-configuration/tenant1",
        "https://auth.example.com/tenant1/.well-known/openid-configuration"]
    assert len(as_metadata_urls("https://auth.example.com")) == 2


def test_cimd_rules_and_iss_table():
    assert validate_cimd(CID, DOCS[CID]) == []
    assert "client_id URL needs a path" in validate_cimd("https://gw.example.com/", {**DOCS[CID], "client_id": "https://gw.example.com/"})
    assert "no shared secrets" in validate_cimd(CID, {**DOCS[CID], "client_secret": "s"})
    assert "client_id must equal the document URL exactly" in validate_cimd(CID, {**DOCS[CID], "client_id": CID + "?x"})
    assert iss_ok("https://a", "https://a", True) and not iss_ok("https://a", "https://A", True)
    assert not iss_ok("https://a", None, True) and iss_ok("https://a", None, False)


def test_full_flow_with_step_up_and_refresh_rotation():
    clock, auth, server, client = world()
    assert client.call("alice", RES, "list_tickets") == {"result": "list_tickets ran for alice"}
    assert client.tokens[("alice", RES)]["scopes"] == {"tickets:read"}              # the 401's scope, not everything
    assert client.call("alice", RES, "close_ticket")["result"].startswith("close_ticket")
    assert client.tokens[("alice", RES)]["scopes"] == {"tickets:read", "tickets:write"}   # step-up = the union
    old_refresh = client.tokens[("alice", RES)]["refresh"]
    clock.sleep(301)                                                                # the access token expires
    client.call("alice", RES, "list_tickets")
    assert client.tokens[("alice", RES)]["refresh"] != old_refresh                  # rotated
    assert any("step-up" in line for line in client.log) and any("refreshed" in line for line in client.log)


def test_tokens_are_per_principal():
    _, _, _, client = world()
    client.call("alice", RES, "list_tickets")
    client.call("bob", RES, "list_tickets")
    assert client.tokens[("alice", RES)]["access"] != client.tokens[("bob", RES)]["access"]


def test_refresh_reuse_revokes_the_whole_grant():
    clock, auth, _, client = world()
    client.call("alice", RES, "list_tickets")
    tok = client.tokens[("alice", RES)]
    stolen = tok["refresh"]
    client.refresh("alice", RES)                                                    # the legitimate rotation
    status, _, body = auth.token({"grant_type": "refresh_token", "refresh_token": stolen, "client_id": CID, "resource": RES})
    assert status == 400 and "reuse" in body["error_description"]
    assert auth.introspect(client.tokens[("alice", RES)]["access"]) is None         # the active token died with the grant


def test_resource_must_match_and_pkce_must_be_advertised():
    _, auth, _, client = world()
    client.call("alice", RES, "list_tickets")
    rt = client.tokens[("alice", RES)]["refresh"]
    assert auth.token({"grant_type": "refresh_token", "refresh_token": rt, "client_id": CID,
                       "resource": "https://other.example.com/mcp"})[0] == 400
    _, _, _, client = world(pkce=False)
    with pytest.raises(PermissionError, match="S256"):
        client.call("alice", RES, "list_tickets")


def test_dpop_nonces_from_the_as_and_the_resource_server():
    _, auth, _, client = world(dpop=True)
    assert client.call("alice", RES, "list_tickets")["result"].startswith("list_tickets")
    assert client.nonces == {auth.issuer: "as-nonce-1", RES: "rs-nonce-1"}
    assert client.tokens[("alice", RES)]["type"] == "DPoP"
    assert "not RFC 9449" in HMACSigner.label


def test_a_refused_refresh_starts_a_fresh_authorization():
    clock, auth, _, client = world()
    client.call("alice", RES, "list_tickets")
    auth.revoked.add(auth.access[client.tokens[("alice", RES)]["access"]]["grant"])   # the AS revoked the grant
    clock.sleep(301)
    assert client.call("alice", RES, "list_tickets")["result"].startswith("list_tickets")
    assert sum("discovered" in line for line in client.log) == 2
