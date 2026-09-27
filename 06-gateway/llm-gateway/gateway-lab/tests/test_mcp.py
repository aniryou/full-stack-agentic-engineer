"""The MCP client flow over HTTP: discovery order, CIMD, PKCE, iss, step-up, refresh rotation, DPoP nonces."""
import asyncio
import time

import aiohttp
import pytest
from aiohttp import web

from gwlab.mcp import McpClient, McpServers, as_metadata_urls, parse_www_authenticate, pkce_challenge, prm_urls, union_scopes
from gwlab.mcp import dpop
from gwlab.mcp.client import AuthError, iss_ok
from gwlab.mcp.server import AsOptions, cimd_url_ok

CALL = lambda tool: {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool}}   # noqa: E731


def test_rfc7636_appendix_b():
    assert pkce_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"


def test_discovery_urls_in_the_spec_order():
    assert as_metadata_urls("https://auth.example.com/tenant1") == [
        "https://auth.example.com/.well-known/oauth-authorization-server/tenant1",
        "https://auth.example.com/.well-known/openid-configuration/tenant1",
        "https://auth.example.com/tenant1/.well-known/openid-configuration"]
    assert as_metadata_urls("https://auth.example.com") == [
        "https://auth.example.com/.well-known/oauth-authorization-server",
        "https://auth.example.com/.well-known/openid-configuration"]
    assert prm_urls("https://example.com/public/mcp") == [
        "https://example.com/.well-known/oauth-protected-resource/public/mcp",
        "https://example.com/.well-known/oauth-protected-resource"]


def test_challenge_parsing_scopes_and_iss():
    scheme, p = parse_www_authenticate('Bearer error="insufficient_scope", scope="mcp:read mcp:write", '
                                       'resource_metadata="https://x/.well-known/oauth-protected-resource/mcp"')
    assert scheme == "Bearer" and p["scope"] == "mcp:read mcp:write" and p["resource_metadata"].endswith("/mcp")
    assert parse_www_authenticate('DPoP error="use_dpop_nonce"')[1]["error"] == "use_dpop_nonce"
    assert union_scopes("mcp:read files:read", "mcp:write mcp:read") == "files:read mcp:read mcp:write"
    assert iss_ok(True, "https://as", "https://as") and not iss_ok(True, None, "https://as")
    assert not iss_ok(False, "https://AS", "https://as") and iss_ok(False, None, "https://as")


@pytest.mark.parametrize("url, loop_ok, ok", [
    ("https://gw.example.com/oauth/client.json", False, True),
    ("http://gw.example.com/oauth/client.json", False, False),
    ("https://gw.example.com", False, False), ("https://gw.example.com/", False, False),
    ("https://gw.example.com/a/../client.json", False, False), ("https://u:p@gw.example.com/c.json", False, False),
    ("https://10.0.0.8/c.json", False, False), ("https://169.254.169.254/c.json", False, False),
    ("http://127.0.0.1:8080/c.json", True, True), ("http://127.0.0.1:8080/c.json", False, False)])
def test_cimd_url_rules(url, loop_ok, ok):
    assert cimd_url_ok(url, loop_ok)[0] is ok


async def _with_servers(opts, fn, prm_in_challenge=True, signer=None):
    srv = McpServers(opts, prm_in_challenge=prm_in_challenge)
    await srv.start()
    app = web.Application()

    async def cimd(req):
        me = f"http://{req.host}"
        return web.json_response({"client_id": me + "/oauth/client.json", "client_name": "gwlab test",
                                  "redirect_uris": [me + "/oauth/callback"]})
    app.router.add_get("/oauth/client.json", cimd)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        async with aiohttp.ClientSession() as http:
            c = McpClient(http, f"http://127.0.0.1:{port}/oauth/client.json", signer=signer)
            return await fn(srv, c)
    finally:
        await srv.stop()
        await runner.cleanup()


def run(opts, fn, **kw):
    return asyncio.run(_with_servers(opts, fn, **kw))


def test_bearer_flow_with_step_up_and_a_token_per_principal():
    async def go(srv, c):
        assert (await c.call("t/alice", srv.resource, CALL("search_notes")))[0] == 200
        assert (await c.call("t/alice", srv.resource, CALL("delete_note")))[0] == 200     # 403 -> step up -> 200
        assert (await c.call("t/bob", srv.resource, CALL("search_notes")))[0] == 200
        return c, srv
    c, srv = run(AsOptions(), go)
    steps = [s[0] for s in c.log]
    assert steps[:5] == ["rs", "prm", "as_metadata_try", "authorized", "token"] and ("rs", 403, "insufficient_scope",
                                                                                       "mcp:read mcp:write") in c.log
    assert {k[0]: sorted(k[2]) for k in c.tokens} == {"t/alice": ["mcp:read", "mcp:write"], "t/bob": ["mcp:read"]}
    assert srv.stats["cimd_fetches"] == 1 and srv.stats["tokens"] == 3


@pytest.mark.parametrize("where", ["oidc", "oidc_suffix"])
def test_metadata_fallback_order(where):
    async def go(srv, c):
        await c.call("p", srv.resource, CALL("search_notes"))
        return [s[1] for s in c.log if s[0] == "as_metadata_try"]
    tried = run(AsOptions(metadata=where), go)
    assert len(tried) == (2 if where == "oidc" else 3) and tried[-1].endswith(
        "/.well-known/openid-configuration/tenant1" if where == "oidc" else "/tenant1/.well-known/openid-configuration")


def test_well_known_prm_when_the_challenge_names_none():
    async def go(srv, c):
        await c.call("p", srv.resource, CALL("search_notes"))
        return [s for s in c.log if s[0] == "prm"][0][1]
    assert run(AsOptions(), go, prm_in_challenge=False).endswith("/.well-known/oauth-protected-resource/mcp")


@pytest.mark.parametrize("opts, msg", [(AsOptions(advertise_pkce=False), "PKCE"), (AsOptions(wrong_issuer=True), "issuer"),
                                       (AsOptions(cimd=False), "Client ID Metadata")])
def test_the_client_refuses_unsafe_servers(opts, msg):
    async def go(srv, c):
        with pytest.raises(AuthError, match=msg):
            await c.call("p", srv.resource, CALL("search_notes"))
    run(opts, go)


def test_refresh_rotates_and_reuse_revokes_the_grant():
    async def go(srv, c):
        await c.call("p", srv.resource, CALL("search_notes"))
        first = c.cached("p", srv.resource)
        await c.refresh("p", srv.resource, first)
        second = c.cached("p", srv.resource)
        assert second.refresh_token != first.refresh_token
        assert (await c.call("p", srv.resource, CALL("search_notes")))[0] == 200
        # an attacker replays the rotated-out refresh token
        async with c.session.post(second.token_endpoint, data={"grant_type": "refresh_token", "refresh_token": first.refresh_token,
                                                               "client_id": c.client_id, "resource": srv.resource}) as r:
            assert r.status == 400 and (await r.json())["error"] == "invalid_grant"
        assert srv.stats["reuse_detected"] == 1
        async with c.session.post(srv.resource, json=CALL("search_notes"),
                                  headers={"Authorization": f"Bearer {second.access_token}"}) as r:
            assert r.status == 401                     # the whole grant is revoked, the active access token included
        with pytest.raises(AuthError):
            await c.refresh("p", srv.resource, second)          # ... and the active refresh token
        status, _ = await c.call("p", srv.resource, CALL("search_notes"))     # the client starts over
        assert status == 200 and ("refresh_failed",) == tuple(x[0] for x in c.log if x[0] == "refresh_failed")[:1]
    run(AsOptions(), go)


@pytest.mark.parametrize("signer_cls", ["hmac", "es256"])
def test_dpop_nonces_at_the_as_and_the_rs(signer_cls):
    if signer_cls == "es256":
        pytest.importorskip("cryptography")
        signer = dpop.ES256Signer()
    else:
        signer = dpop.HmacStandInSigner()

    async def go(srv, c):
        assert (await c.call("p", srv.resource, CALL("search_notes")))[0] == 200
        assert ("as", 400, "use_dpop_nonce") in c.log and ("rs", 401, "use_dpop_nonce", "") in c.log
        assert c.cached("p", srv.resource).token_type == "DPoP"
        srv.rotate_nonces()                                   # the RS moves on: one more 401, then it works again
        assert (await c.call("p", srv.resource, CALL("search_notes")))[0] == 200
        assert sum(1 for x in c.log if x[:3] == ("rs", 401, "use_dpop_nonce")) == 2
        tok = c.cached("p", srv.resource).access_token        # a stolen DPoP token without the key is useless
        async with c.session.post(srv.resource, json=CALL("search_notes"), headers={"Authorization": f"DPoP {tok}"}) as r:
            assert r.status == 401
    run(AsOptions(dpop=True), go, signer=signer)
    assert signer.conformant is (signer_cls == "es256")


def test_proof_checks():
    s = dpop.HmacStandInSigner()
    url = "https://rs.example/mcp"
    p = dpop.make_proof(s, "POST", url + "?x=1", access_token="tok", nonce="n1")
    claims, jkt = dpop.verify_proof(p, method="POST", url=url, access_token="tok", nonce="n1")
    assert claims["htu"] == url and jkt == dpop.jwk_thumbprint(s.jwk())
    seen = set()
    dpop.verify_proof(p, method="POST", url=url, access_token="tok", seen_jti=seen)
    for kw, err in ((dict(seen_jti=seen), "invalid_dpop_proof"), (dict(nonce="n2"), "use_dpop_nonce"),
                    (dict(access_token="other"), "invalid_dpop_proof"), (dict(method="GET"), "invalid_dpop_proof")):
        args = {"method": "POST", "url": url, "access_token": "tok", **kw}
        with pytest.raises(dpop.ProofError) as e:
            dpop.verify_proof(p, **args)
        assert e.value.error == err
    old = dpop.make_proof(s, "POST", url, now=time.time() - 3600)
    with pytest.raises(dpop.ProofError, match="iat"):
        dpop.verify_proof(old, method="POST", url=url)
    assert "NOT DPoP-conformant" in s.label


async def _code(c, srv, resource):
    """An authorization code for `resource` (the fake AS auto-approves the login_hint principal)."""
    import urllib.parse
    from gwlab.mcp.client import make_verifier
    _, asm = c.meta[srv.resource]
    verifier = make_verifier()
    params = {"response_type": "code", "client_id": c.client_id, "redirect_uri": c.redirect_uri, "scope": "mcp:read",
              "state": "s", "code_challenge": pkce_challenge(verifier), "code_challenge_method": "S256",
              "resource": resource, "login_hint": "p"}
    async with c.session.get(asm["authorization_endpoint"], params=params, allow_redirects=False) as r:
        assert r.status == 302
        code = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(r.headers["Location"]).query))["code"]
    return asm, code, verifier


def test_a_token_minted_for_another_resource_is_refused():
    """RFC 8707: the token's audience is the resource it was requested for; this MCP server refuses any other."""
    async def go(srv, c):
        await c.call("p", srv.resource, CALL("search_notes"))                       # discovery, and a token of its own
        other = srv.rs_url + "/other-mcp"
        asm, code, verifier = await _code(c, srv, other)
        form = {"grant_type": "authorization_code", "code": code, "redirect_uri": c.redirect_uri, "client_id": c.client_id,
                "code_verifier": verifier, "resource": other}
        async with c.session.post(asm["token_endpoint"], data=form) as r:
            assert r.status == 200
            tok = (await r.json())["access_token"]
        async with c.session.post(srv.resource, json=CALL("search_notes"), headers={"Authorization": f"Bearer {tok}"}) as r:
            assert r.status == 401 and 'error="invalid_token"' in r.headers["WWW-Authenticate"]
        own = c.cached("p", srv.resource).access_token
        async with c.session.post(srv.resource, json=CALL("search_notes"), headers={"Authorization": f"Bearer {own}"}) as r:
            assert r.status == 200
    run(AsOptions(), go)


def test_the_code_exchange_must_name_the_authorized_resource():
    async def go(srv, c):
        await c.call("p", srv.resource, CALL("search_notes"))
        asm, code, verifier = await _code(c, srv, srv.resource)
        form = {"grant_type": "authorization_code", "code": code, "redirect_uri": c.redirect_uri, "client_id": c.client_id,
                "code_verifier": verifier, "resource": srv.rs_url + "/other-mcp"}
        async with c.session.post(asm["token_endpoint"], data=form) as r:
            assert r.status == 400 and (await r.json())["error"] in ("invalid_target", "invalid_grant")
    run(AsOptions(), go)
