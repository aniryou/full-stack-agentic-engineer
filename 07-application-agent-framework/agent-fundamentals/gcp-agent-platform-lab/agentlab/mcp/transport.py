"""Transports: how JSON-RPC bytes reach an ``McpServer`` (Primer §3.2). Teaching subset.

Streamable HTTP in this revision is one POST per request to a single endpoint.
We answer every request with one JSON object (the request-scoped SSE stream
for progress notifications is out of scope; stdio is out of scope too).

The contract every transport speaks, client side::

    status, headers, body = await transport.request(method, url, headers, body)

and the server-side core that all of them end up calling::

    status, headers, body = await handle(server, method, path, headers, body)

Headers are case-insensitive on the wire; both directions normalise to
lower-case keys at this boundary so the rest of the code never guesses.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

import httpx

from ..auth.oauth import WELL_KNOWN_PRM
from . import protocol as p
from .server import McpServer

HttpTriple = tuple[int, dict[str, str], bytes]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def _json_triple(status: int, headers: Mapping[str, str], body: Any) -> HttpTriple:
    out = {k.lower(): v for k, v in headers.items()}
    out["content-type"] = "application/json"
    return status, out, json.dumps(body).encode()


class McpNotHere(p.McpError):
    def __init__(self, method: str, path: str):
        super().__init__(p.METHOD_NOT_FOUND, f"no MCP endpoint at {method} {path}", http_status=404)


async def handle(server: McpServer, method: str, path: str, headers: Mapping[str, str], body: bytes) -> HttpTriple:
    """Route one HTTP exchange: the public metadata document, or a JSON-RPC POST to the endpoint."""
    if method == "GET" and path.rstrip("/").endswith(WELL_KNOWN_PRM):
        return _json_triple(200, {}, server.protected_resource_metadata())
    if method != "POST" or path.rstrip("/") != server.endpoint_path.rstrip("/"):
        return _json_triple(404, {}, p.error(None, McpNotHere(method, path)))
    try:
        payload = json.loads(body or b"null")
    except ValueError:
        return _json_triple(400, {}, p.error(None, p.McpError(p.PARSE_ERROR, "body is not valid JSON")))
    response = await server.handle_jsonrpc(headers, payload)
    return _json_triple(response.status, response.headers, response.body)


class Transport(Protocol):
    base_url: str

    async def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes) -> HttpTriple: ...


class InProcessTransport:
    """Calls the server (or a gateway) directly — same bytes, same headers, no sockets.

    ``target`` is an ``McpServer`` or anything with ``handle(method, url, headers, body)``
    such as a ``Gateway``; for the latter pass the upstream server's URL as ``base_url``
    so the gateway can resolve the destination.
    """

    def __init__(self, target: Any, base_url: str | None = None):
        self.target = target
        self.base_url = base_url or getattr(target, "resource_url", None)
        if not self.base_url:
            raise ValueError("base_url is required when the target is not an McpServer")

    async def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes) -> HttpTriple:
        if isinstance(self.target, McpServer):
            return await handle(self.target, method, urlparse(url).path, headers, body)
        return await self.target.handle(method, url, headers, body)


class HttpTransport:
    """httpx over the wire. The transport is bound to one origin: absolute URLs (e.g. the metadata URL
    a 401 advertises for the server's *canonical* name) are resolved against it.

    ``connect_origin`` is the private-endpoint case: ``base_url`` stays the canonical URL (the token
    audience, what a registry stores) while the bytes go to a local or private address.
    """

    def __init__(self, base_url: str, *, connect_origin: str | None = None, timeout_s: float = 10.0):
        self.base_url = base_url
        self.timeout_s = timeout_s
        parsed = urlparse(connect_origin or base_url)
        self._origin = f"{parsed.scheme}://{parsed.netloc}"

    async def request(self, method: str, url: str, headers: Mapping[str, str], body: bytes) -> HttpTriple:
        target = self._origin + urlparse(url).path
        async with httpx.AsyncClient(timeout=self.timeout_s, trust_env=False) as client:   # loopback never goes via a proxy
            r = await client.request(method, target, headers=dict(headers), content=body)
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.content


class LocalHttpServer:
    """Serves an ``McpServer`` on 127.0.0.1 from daemon threads, so a notebook kernel stays responsive.

    Two threads: an asyncio loop that runs the server's coroutines (and owns its
    background tasks), and a ``ThreadingHTTPServer`` whose handlers submit work to
    that loop. ``Origin`` is validated when present: a browser page on another
    origin must not be able to drive a local server (DNS-rebinding defence).
    """

    def __init__(self, server: McpServer, host: str = "127.0.0.1", port: int = 0):
        self.server = server
        self.host, self.port = host, port
        self.url: str | None = None
        self._loop = asyncio.new_event_loop()
        self._httpd: ThreadingHTTPServer | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> str:
        mcp, loop = self.server, self._loop

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:   # keep notebook output clean
                pass

            def _serve(self) -> None:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).hostname not in LOCAL_HOSTS:
                    self._reply(403, {}, json.dumps({"error": "origin not allowed"}).encode())
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                future = asyncio.run_coroutine_threadsafe(handle(mcp, self.command, self.path, dict(self.headers.items()), body), loop)
                self._reply(*future.result(timeout=60))

            def _reply(self, status: int, headers: Mapping[str, str], body: bytes) -> None:
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _serve

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://{self.host}:{self._httpd.server_address[1]}{mcp.endpoint_path}"
        self._threads = [threading.Thread(target=loop.run_forever, name="mcp-loop", daemon=True),
                         threading.Thread(target=self._httpd.serve_forever, name="mcp-http", daemon=True)]
        for t in self._threads:
            t.start()
        return self.url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self._loop.call_soon_threadsafe(self._loop.stop)
        for t in self._threads:
            t.join(timeout=5)

    def __enter__(self) -> "LocalHttpServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
