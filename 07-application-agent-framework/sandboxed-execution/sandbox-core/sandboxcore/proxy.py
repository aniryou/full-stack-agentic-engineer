"""A tiny allowlisting HTTP egress proxy that injects a credential the sandbox never holds.

The one idea: the sandbox has no network of its own and no secrets. When code legitimately needs an
outside host, it talks to *this* proxy over plain HTTP inside the trust boundary; the proxy checks the
host against an allowlist and, for allowed hosts, adds the credential header on the way out. The secret
lives only in the proxy — the same "gateway path" the identity primer describes (§5), one layer down,
and the header-injection pattern E2B uses host-side (FACTS §8). A blocked host gets 403 and is logged.

Four rules keep the credential in the proxy (each is a test in ``tests/test_proxy.py``):

* **Redirects are not followed.** An allowed host that answers ``302 Location: http://elsewhere/`` would
  otherwise have the request — injected header included — replayed to a host nobody allowed (urllib's
  default redirect handler copies the headers). The 3xx goes back to the caller; a new URL is a new
  request, checked against the allowlist again.
* **The caller's credentials and hop-by-hop headers are dropped** (``Authorization``,
  ``Proxy-Authorization``, ``Cookie``, ``Connection`` …) before injection: the sandbox cannot forward
  anything it found or forged, and only the proxy's own header reaches the upstream.
* **An upstream that reflects the credential is redacted**: the injected values are replaced in the body
  before it goes back to the sandbox.
* **Environment proxies are ignored**: the proxy dials the upstream itself, never via ``HTTP_PROXY``.

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

# Never forwarded upstream: hop-by-hop headers (RFC 9110 §7.6.1) and anything that carries a credential
# the sandbox should not have. Host is set by the proxy from the checked URL.
DROP_HEADERS = frozenset(h.lower() for h in (
    "Connection", "Keep-Alive", "Proxy-Connection", "TE", "Trailer", "Transfer-Encoding", "Upgrade",
    "Proxy-Authorization", "Proxy-Authenticate", "Authorization", "Cookie", "Host", "Content-Length"))
REDACTED = b"[redacted by egress proxy]"


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
    note: str = ""           # e.g. "redirect not followed", "reflected credential redacted"


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Return 3xx to the caller instead of following it (following would carry the injected header)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None          # urllib then raises HTTPError(code), which fetch() passes back as-is


def _default_opener() -> Callable[..., object]:
    # No ProxyHandler from the environment (ProxyHandler({})), and no redirect following.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects).open


class EgressProxy:
    """The forward-proxy logic, independent of the HTTP server, so it is unit-testable in isolation."""

    def __init__(self, policy: ProxyPolicy, opener: Callable[..., object] | None = None):
        self.policy = policy
        self.events: list[ProxyEvent] = []
        self._opener = opener or _default_opener()

    def outbound_headers(self, host: str, headers: dict[str, str] | None) -> tuple[dict[str, str], list[str]]:
        """The caller's headers minus hop-by-hop and credential headers, plus the injected ones."""
        out = {k: v for k, v in (headers or {}).items() if k.lower() not in DROP_HEADERS}
        injected = []
        for name, value in self.policy.inject.get(host, {}).items():
            out[name] = value                     # the sandbox never set this and never sees the value
            injected.append(name)
        return out, injected

    def fetch(self, method: str, url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes, list[str]]:
        parts = urllib.parse.urlparse(url)
        host = parts.hostname or ""
        if parts.scheme != "http" or not self.policy.allows(host):
            self.events.append(ProxyEvent(method, host, url, "deny", 403, []))
            return 403, b"egress denied: host not on allowlist (plain http:// to an allowlisted host only)", []
        out_headers, injected = self.outbound_headers(host, headers)
        req = urllib.request.Request(url, method=method, headers=out_headers)
        note = ""
        try:
            with self._opener(req, timeout=5) as resp:
                body = resp.read(self.policy.max_bytes + 1)
                status = getattr(resp, "status", 200)
        except urllib.error.HTTPError as e:       # includes a 3xx we refused to follow
            body, status = e.read(self.policy.max_bytes + 1), e.code
            if 300 <= e.code < 400:
                note = "redirect not followed: a new URL is a new request, checked again"
        except Exception as e:                    # DNS failure, refused, timeout: report, do not crash
            self.events.append(ProxyEvent(method, host, url, "allow", 502, injected))
            return 502, f"upstream error: {type(e).__name__}".encode(), injected
        for value in self.policy.inject.get(host, {}).values():
            if value and value.encode() in body:  # an upstream that echoes the credential back
                body = body.replace(value.encode(), REDACTED)
                note = (note + "; " if note else "") + "reflected credential redacted"
        truncated = len(body) > self.policy.max_bytes
        body = body[: self.policy.max_bytes]
        self.events.append(ProxyEvent(method, host, url, "allow", status, injected, len(body), note))
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
        self.server.proxy.events.append(ProxyEvent("CONNECT", self.path.rsplit(":", 1)[0], self.path,
                                                   "deny", 405, []))
        self.send_error(405, "CONNECT not supported: this proxy brokers plain HTTP so it can inject "
                             "credentials the sandbox never holds; terminate TLS at the proxy in production")


class ProxyServer:
    """Run the proxy on a background thread; returns the bound address for the sandbox to point at."""

    def __init__(self, policy: ProxyPolicy, host: str = "127.0.0.1", port: int = 0,
                 opener: Callable[..., object] | None = None):
        self.proxy = EgressProxy(policy, opener)
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.daemon_threads = True
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
