"""A tiny allowlisting HTTP egress proxy that injects a credential the sandbox never holds.

The one idea: the sandbox has no network of its own and no secrets. When code legitimately needs an
outside host, it talks to *this* proxy over plain HTTP inside the trust boundary; the proxy checks the
host against an allowlist and, for allowed hosts, adds the credential header on the way out. The secret
lives only in the proxy — the same "gateway path" the identity primer describes (§5), one layer down,
and the header-injection pattern E2B uses host-side (FACTS §8). A blocked host gets 403 and is logged;
nothing about the credential is ever visible to the sandboxed code.

This is a forward proxy for plain HTTP (``GET http://host/path``). HTTPS through a CONNECT tunnel cannot
have headers injected without terminating TLS (FACTS §23), so the honest teaching version brokers HTTP and
refuses CONNECT — the design note explains why and what a production proxy does instead (terminate TLS with
a sandbox-trusted CA, or be the TLS client itself).
"""
from __future__ import annotations

import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable


@dataclass
class ProxyPolicy:
    allowlist: tuple[str, ...] = ()                       # exact hostnames the proxy will reach
    inject: dict[str, dict[str, str]] = field(default_factory=dict)   # host -> headers to add outbound
    max_bytes: int = 1 << 20                              # cap the response body the sandbox can pull back

    def allows(self, host: str) -> bool:
        return host in self.allowlist


@dataclass
class ProxyEvent:
    method: str
    host: str
    url: str
    decision: str            # "allow" | "deny"
    status: int | None
    injected: list[str]      # header names added (never their values)
    bytes_out: int = 0


class EgressProxy:
    """The forward-proxy logic, independent of the HTTP server, so it is unit-testable in isolation."""

    def __init__(self, policy: ProxyPolicy, opener: Callable[..., object] | None = None):
        self.policy = policy
        self.events: list[ProxyEvent] = []
        self._opener = opener or urllib.request.urlopen

    def fetch(self, method: str, url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, list[str]]:
        host = urllib.parse.urlparse(url).hostname or ""
        if not self.policy.allows(host):
            self.events.append(ProxyEvent(method, host, url, "deny", 403, []))
            return 403, b"egress denied: host not on allowlist", []
        out_headers = dict(headers or {})
        injected = []
        for name, value in self.policy.inject.get(host, {}).items():
            out_headers[name] = value             # the sandbox never set this and never sees the value
            injected.append(name)
        req = urllib.request.Request(url, method=method, headers=out_headers)
        try:
            with self._opener(req, timeout=5) as resp:
                body = resp.read(self.policy.max_bytes + 1)
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as e:
            body, status = e.read(self.policy.max_bytes + 1), e.code
        except Exception as e:                    # DNS failure, refused, timeout: report, do not crash
            self.events.append(ProxyEvent(method, host, url, "allow", 502, injected))
            return 502, f"upstream error: {type(e).__name__}".encode(), injected
        truncated = len(body) > self.policy.max_bytes
        body = body[: self.policy.max_bytes]
        self.events.append(ProxyEvent(method, host, url, "allow", status, injected, len(body)))
        if truncated:
            body += b"\n...[truncated by proxy]"
        return status, body, injected


class _Handler(BaseHTTPRequestHandler):
    proxy: EgressProxy = None       # set on the server class

    def log_message(self, *args) -> None:      # keep the test output quiet
        pass

    def do_GET(self) -> None:
        # A forward proxy receives the absolute URL as the request target.
        url = self.path if self.path.startswith("http") else f"http://{self.headers.get('Host','')}{self.path}"
        status, body, _ = self.server.proxy.fetch("GET", url, dict(self.headers))
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_CONNECT(self) -> None:
        # HTTPS tunnels cannot have headers injected without TLS interception (FACTS §23).
        self.send_error(405, "CONNECT not supported: this proxy brokers plain HTTP so it can inject "
                             "credentials the sandbox never holds; terminate TLS at the proxy in production")


class ProxyServer:
    """Run the proxy on a background thread; returns the bound address for the sandbox to point at."""

    def __init__(self, policy: ProxyPolicy, host: str = "127.0.0.1", port: int = 0):
        self.proxy = EgressProxy(policy)
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.proxy = self.proxy
        self.host, self.port = self._httpd.server_address
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __enter__(self) -> "ProxyServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
