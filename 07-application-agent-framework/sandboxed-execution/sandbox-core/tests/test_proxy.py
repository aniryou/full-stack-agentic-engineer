"""The egress proxy against real local upstreams: allowlist, injection, no redirects, no CONNECT, no leaks."""
import http.client
import io
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sandboxcore import EgressProxy, ProxyPolicy, ProxyServer
from sandboxcore.proxy import REDACTED


class Upstream:
    """A loopback HTTP server that records every request's path and headers.

    ``/redirect`` answers 302 to ``redirect_to``; ``/echo`` reflects the Authorization header in the body.
    """

    def __init__(self, redirect_to: str | None = None):
        self.seen: list[tuple[str, dict]] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.seen.append((self.path, dict(self.headers)))
                if self.path.startswith("/redirect") and redirect_to:
                    self.send_response(302)
                    self.send_header("Location", redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = (f"auth={self.headers.get('Authorization', '<none>')}".encode()
                        if self.path.startswith("/echo") else b"ok")
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def upstreams():
    off = Upstream()                                     # reached as "localhost": NOT on the allowlist
    on = Upstream(redirect_to=f"http://localhost:{off.port}/steal")   # reached as "127.0.0.1": allowed
    yield on, off
    on.close()
    off.close()


def policy():
    return ProxyPolicy(allowlist=("127.0.0.1",), inject={"127.0.0.1": {"Authorization": "Bearer REAL-SECRET"}})


def test_allowed_host_gets_the_injected_credential_and_the_sandbox_never_sees_it(upstreams):
    on, _ = upstreams
    p = EgressProxy(policy())
    status, body, injected = p.fetch("GET", f"http://127.0.0.1:{on.port}/echo")
    assert status == 200 and injected == ["Authorization"]
    assert on.seen[-1][1]["Authorization"] == "Bearer REAL-SECRET"    # the upstream got the credential
    assert b"REAL-SECRET" not in body and REDACTED in body            # the echo was redacted on the way back
    assert "reflected credential redacted" in p.events[-1].note


def test_off_allowlist_host_is_denied_before_any_request_leaves(upstreams):
    _, off = upstreams
    p = EgressProxy(policy())
    status, body, _ = p.fetch("GET", f"http://localhost:{off.port}/steal")
    assert status == 403 and b"denied" in body
    assert off.seen == []                                             # nothing reached it
    assert p.events[-1].decision == "deny" and p.events[-1].host == "localhost"


def test_redirect_to_an_off_allowlist_host_is_not_followed(upstreams):
    on, off = upstreams
    p = EgressProxy(policy())
    status, _, _ = p.fetch("GET", f"http://127.0.0.1:{on.port}/redirect")
    assert status == 302                                              # handed back, not followed
    assert off.seen == []                                             # the credential never went there
    assert "redirect not followed" in p.events[-1].note


def test_caller_credentials_and_hop_by_hop_headers_are_stripped(upstreams):
    on, _ = upstreams
    p = EgressProxy(ProxyPolicy(allowlist=("127.0.0.1",)))            # no injection for this test
    p.fetch("GET", f"http://127.0.0.1:{on.port}/x",
            {"Authorization": "Bearer stolen", "Proxy-Authorization": "Basic sandbox",
             "Cookie": "s=1", "Connection": "close", "Accept": "text/plain"})
    sent = {k.lower(): v for k, v in on.seen[-1][1].items()}
    assert "authorization" not in sent and "proxy-authorization" not in sent and "cookie" not in sent
    assert sent["accept"] == "text/plain"


def test_https_is_refused_rather_than_brokered():
    p = EgressProxy(ProxyPolicy(allowlist=("api.github.com",)))
    assert p.fetch("GET", "https://api.github.com/user")[0] == 403    # no TLS termination here


def test_proxy_server_end_to_end_and_connect_is_405(upstreams):
    on, off = upstreams
    with ProxyServer(policy()) as srv:
        # a sandbox-side client pointed at the proxy (a raw HTTP/1.1 forward-proxy request)
        c = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
        c.request("GET", f"http://127.0.0.1:{on.port}/echo", headers={"Proxy-Authorization": "Basic sandbox"})
        r = c.getresponse()
        assert r.status == 200 and REDACTED in r.read()
        assert "proxy-authorization" not in {k.lower() for k in on.seen[-1][1]}
        c.close()
        c = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
        c.request("CONNECT", f"localhost:{off.port}")
        assert c.getresponse().status == 405                          # no opaque tunnels
        c.close()
        c = http.client.HTTPConnection(srv.host, srv.port, timeout=5)
        c.request("GET", f"http://localhost:{off.port}/steal")
        assert c.getresponse().status == 403
        c.close()
    assert off.seen == []


def test_response_body_is_capped():
    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    p = EgressProxy(ProxyPolicy(allowlist=("h",), max_bytes=1000), opener=lambda req, timeout=5: _Resp(b"x" * 5000))
    status, body, _ = p.fetch("GET", "http://h/big")
    assert status == 200 and len(body) <= 1000 + 40      # capped + a short truncation marker


def test_events_never_record_the_secret_value(upstreams):
    on, _ = upstreams
    p = EgressProxy(policy())
    p.fetch("GET", f"http://127.0.0.1:{on.port}/echo")
    ev = p.events[-1]
    assert ev.injected == ["Authorization"]              # header NAME only
    assert "REAL-SECRET" not in repr(ev)                  # never the value


def test_default_opener_ignores_environment_proxies(monkeypatch, upstreams):
    on, _ = upstreams
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")    # a dead proxy: using it would fail
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    assert EgressProxy(policy()).fetch("GET", f"http://127.0.0.1:{on.port}/x")[0] == 200
    assert urllib.request.getproxies().get("http")               # the env proxy really was set
