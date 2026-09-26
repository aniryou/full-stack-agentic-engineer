"""server.py — an allowlisting egress proxy that holds the credentials so the sandbox never does.

One idea: a sandbox that needs *some* network gets exactly one reachable address — this proxy —
and the proxy decides everything else (PRIMER §4 "Network and secrets"). It is the identity
primer's "gateway path" (06-gateway, identity primer §5: the credential is injected at the edge
and the caller never sees it) moved one layer down, to the boundary of the code sandbox.

Two ways in:

* **Routes (preferred).** ``GET http://egress-proxy:8080/<route>/<path>``: plain HTTP from the
  sandbox to the proxy; the proxy maps ``<route>`` to an upstream the operator configured, strips
  any credential the caller sent, injects the real one (from a file or an env var — a Kubernetes
  Secret mounted *only into the proxy*), and speaks HTTPS upstream if the route says so. The code
  never learns the key or even the upstream's hostname.
* **Forward proxy.** ``HTTP_PROXY=http://egress-proxy:8080``: absolute-form requests to hosts on
  ``forward_allow`` (exact names or ``*.suffix``), resolved and refused if they land on a private,
  loopback or link-local address (DNS rebinding to 169.254.169.254). ``CONNECT`` is refused by
  default: through an HTTPS tunnel the proxy can neither inject a credential nor see the request,
  so it could only allow or deny a hostname.

Every decision is one JSON line on stdout (``kubectl logs deploy/egress-proxy`` is the egress
audit trail) in the shape of the identity lab's ``AuditEvent`` (``event_type: "egress"``).
Responses are capped and any occurrence of an injected secret is redacted: an upstream that
echoes request headers back would otherwise hand the key to the sandbox.

Standard library only and a single file, so it runs from a ConfigMap in ``python:3.12-slim``:

    python3 server.py --config proxy.json --port 8080
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import ipaddress
import json
import os
import socket
import ssl
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
              "transfer-encoding", "upgrade", "proxy-connection"}
CALLER_CREDENTIALS = {"authorization", "cookie", "proxy-authorization", "x-api-key", "x-goog-api-key"}
REDACTED = "[REDACTED]"


# ---- policy: pure functions, tested without sockets ----------------------------------------------------
def normalise_host(host: str) -> str:
    return (host or "").strip().rstrip(".").lower()


def host_allowed(host: str, patterns: list[str]) -> bool:
    """Exact names, or ``*.example.com`` for subdomains only (not the apex, not ``evilexample.com``)."""
    h = normalise_host(host)
    if not h:
        return False
    for p in patterns:
        p = normalise_host(p)
        if p.startswith("*."):
            if h.endswith(p[1:]) and h != p[2:]:
                return True
        elif h == p:
            return True
    return False


def is_public_ip(ip: str) -> bool:
    a = ipaddress.ip_address(ip)
    return not (a.is_private or a.is_loopback or a.is_link_local or a.is_multicast or a.is_reserved
                or a.is_unspecified)


def split_route(path: str) -> tuple[str, str]:
    """``/api-stub/whoami?x=1`` -> ``("api-stub", "/whoami?x=1")``."""
    rest = path.lstrip("/")
    name, _, tail = rest.partition("/")
    return name, "/" + tail


def clean_headers(headers, drop: set[str]) -> dict[str, str]:
    """Remove hop-by-hop headers, anything named in ``Connection``, and ``drop`` (caller creds)."""
    named = {h.strip().lower() for h in (headers.get("Connection") or "").split(",") if h.strip()}
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP | named | drop and k.lower() != "host"}


def redact(data: bytes, secrets: list[str]) -> tuple[bytes, int]:
    n = 0
    for s in secrets:
        if s and s.encode() in data:
            n += data.count(s.encode())
            data = data.replace(s.encode(), REDACTED.encode())
    return data, n


def read_secret(ref: str) -> str:
    """``env:NAME`` or ``file:/path`` (a mounted Secret; re-read per request, so rotation works)."""
    kind, _, where = ref.partition(":")
    if kind == "env":
        return os.environ.get(where, "")
    if kind == "file":
        try:
            with open(where) as f:
                return f.read().strip()
        except OSError:
            return ""
    raise ValueError(f"secret reference must be env:NAME or file:/path, got {ref!r}")


def audit_event(decision: str, reasons: list[str], **extra) -> dict:
    """The identity lab's AuditEvent fields (agentsec.audit.log), event_type "egress"."""
    return {"event_type": "egress", "agent": extra.pop("agent", "sandbox"), "authority": "own",
            "tool": extra.pop("tool", "http"), "decision": decision, "reasons": reasons,
            "latency_ms": extra.pop("latency_ms", None), "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "id": uuid.uuid4().hex, "extra": extra}


DEFAULT_CONFIG = {"routes": {}, "forward_allow": [], "allow_connect": False, "block_private": True,
                  "max_request_bytes": 65536, "max_response_bytes": 1048576, "timeout_s": 10.0}


# ---- the server -----------------------------------------------------------------------------------------
class EgressProxy:
    """Holds the config and the audit trail; ``serve()`` runs a ThreadingHTTPServer in a thread."""

    def __init__(self, config: dict, *, host: str = "127.0.0.1", port: int = 0, emit=None):
        self.config = {**DEFAULT_CONFIG, **config}
        self.events: list[dict] = []
        self._emit = emit
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            server_version = "sandboxlab-egress/0.1"

            def log_message(self, *a):          # the audit line replaces the access log
                pass

            def do_CONNECT(self):
                proxy._connect(self)

            def _any(self):
                proxy._handle(self)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _any

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"
        self._thread: threading.Thread | None = None

    # -- lifecycle
    def serve(self) -> "EgressProxy":
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.serve()

    def __exit__(self, *exc):
        self.close()

    # -- audit
    def record(self, decision: str, reasons: list[str], **extra) -> dict:
        ev = audit_event(decision, reasons, **extra)
        with self._lock:
            self.events.append(ev)
        if self._emit:
            self._emit(ev)
        return ev

    # -- request handling
    def _deny(self, h, status: int, reason: str, **extra):
        self.record("deny", [reason], method=h.command, target=h.path[:200], client=h.client_address[0],
                    status=status, **extra)
        body = json.dumps({"error": "egress_denied", "reason": reason}).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Connection", "close")
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(body)
        h.close_connection = True

    def _connect(self, h):
        host, _, port = h.path.rpartition(":")
        if not self.config["allow_connect"]:
            return self._deny(h, 403, "CONNECT refused: an HTTPS tunnel hides the request, so no credential can be "
                                       "injected; use a route (http://proxy/<route>/...)", host=host)
        return self._deny(h, 501, "CONNECT tunnelling is not implemented in the lab proxy", host=host)

    def _handle(self, h):
        t0 = time.monotonic()
        cfg = self.config
        if h.path == "/healthz":                                  # readiness probe; not audited
            h.send_response(200)
            h.send_header("Content-Length", "2")
            h.end_headers()
            h.wfile.write(b"ok")
            return
        length = int(h.headers.get("Content-Length") or 0)
        if length > cfg["max_request_bytes"]:
            return self._deny(h, 413, f"request body {length} B > {cfg['max_request_bytes']} B")
        body = h.rfile.read(length) if length else None
        secrets: list[str] = []
        if h.path.startswith("/"):                               # route form
            name, tail = split_route(h.path)
            route = cfg["routes"].get(name)
            if route is None:
                return self._deny(h, 403, f"no route {name!r}", route=name)
            if h.command not in route.get("methods", ["GET"]):
                return self._deny(h, 405, f"method {h.command} not allowed on route {name!r}", route=name)
            up = urlsplit(route["upstream"])
            scheme, host, port = up.scheme, up.hostname, up.port or (443 if up.scheme == "https" else 80)
            path = (up.path.rstrip("/") + tail) if up.path else tail
            headers = clean_headers(h.headers, CALLER_CREDENTIALS)
            inj = route.get("inject")
            if inj:
                secret = read_secret(inj["value_from"])
                if not secret:
                    return self._deny(h, 503, f"route {name!r}: credential {inj['value_from']} is empty", route=name)
                headers[inj.get("header", "Authorization")] = inj.get("format", "{}").format(secret)
                secrets.append(secret)
            ctx = {"route": name, "upstream_host": host}
        else:                                                    # absolute-form (forward proxy)
            u = urlsplit(h.path)
            if u.scheme != "http" or not u.hostname:
                return self._deny(h, 400, "absolute-form http:// URL expected")
            host = normalise_host(u.hostname)
            if not host_allowed(host, cfg["forward_allow"]):
                return self._deny(h, 403, f"host {host!r} is not on the egress allowlist", host=host)
            if cfg["block_private"]:
                try:
                    addrs = {ai[4][0] for ai in socket.getaddrinfo(host, u.port or 80, proto=socket.IPPROTO_TCP)}
                except OSError as e:
                    return self._deny(h, 502, f"cannot resolve {host}: {e}", host=host)
                private = sorted(a for a in addrs if not is_public_ip(a))
                if private:
                    return self._deny(h, 403, f"{host} resolves to non-public {private}", host=host)
            scheme, port = "http", u.port or 80
            path = (u.path or "/") + (f"?{u.query}" if u.query else "")
            headers = clean_headers(h.headers, {"proxy-authorization"})
            ctx = {"host": host}
        try:
            conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            kw = {"context": ssl.create_default_context()} if scheme == "https" else {}
            conn = conn_cls(host, port, timeout=cfg["timeout_s"], **kw)
            conn.request(h.command, path, body=body, headers={**headers, "Host": host if port in (80, 443) else f"{host}:{port}"})
            resp = conn.getresponse()
            data = resp.read(cfg["max_response_bytes"] + 1)
            conn.close()
        except (OSError, http.client.HTTPException) as e:
            return self._deny(h, 502, f"upstream error: {type(e).__name__}: {e}", **ctx)
        truncated = len(data) > cfg["max_response_bytes"]
        data = data[: cfg["max_response_bytes"]]
        data, n_red = redact(data, secrets)
        out_headers = {k: v for k, v in resp.getheaders() if k.lower() not in HOP_BY_HOP | {"content-length"}}
        for k in list(out_headers):
            out_headers[k], n = redact(out_headers[k].encode("latin-1", "replace"), secrets)
            out_headers[k] = out_headers[k].decode("latin-1")
            n_red += n
        reasons = ["allowlisted route" if "route" in ctx else "allowlisted host"]
        if n_red:
            reasons.append(f"redacted {n_red} occurrence(s) of the injected credential from the response")
        if truncated:
            reasons.append(f"response truncated to {cfg['max_response_bytes']} B")
        self.record("allow", reasons, method=h.command, client=h.client_address[0], status=resp.status,
                    bytes=len(data), injected=bool(secrets),
                    path_sha256=hashlib.sha256(path.encode()).hexdigest()[:16],
                    latency_ms=round((time.monotonic() - t0) * 1000, 1), **ctx)
        h.send_response(resp.status)
        for k, v in out_headers.items():
            h.send_header(k, v)
        h.send_header("Content-Length", str(len(data)))
        if truncated:
            h.send_header("X-Egress-Truncated", "1")
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="allowlisting egress proxy with credential injection")
    ap.add_argument("--config", required=True, help="JSON: routes, forward_allow, limits")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)

    def emit(ev):
        sys.stdout.write(json.dumps(ev, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    proxy = EgressProxy(cfg, host=a.host, port=a.port, emit=emit)
    sys.stdout.write(json.dumps({"event_type": "proxy.start", "port": proxy.port,
                                 "routes": sorted(proxy.config["routes"]),
                                 "forward_allow": proxy.config["forward_allow"]}) + "\n")
    sys.stdout.flush()
    try:
        proxy.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
