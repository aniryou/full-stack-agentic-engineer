"""client.py — talk to the egress proxy over TCP or a Unix socket, from outside or inside a sandbox.

One idea: the sandboxed code needs exactly one way out, and it should not need to know anything
secret to use it. ``SANDBOX_CLIENT`` is a stdlib snippet you prepend to model-generated code; it
defines ``proxy_get(path)`` using ``$SANDBOX_PROXY_URL`` (``http://egress-proxy:8080`` in a pod,
``unix:/run/egress/proxy.sock`` in a network-less process or container). ``fetch`` is the same
thing for the agent runtime (the ``fetch_url`` tool).
"""
from __future__ import annotations

import http.client
import json
import socket
from urllib.parse import urlsplit


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float = 10.0):
        super().__init__("egress-proxy", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def fetch(proxy_url: str, path: str, *, method: str = "GET", headers: dict | None = None,
          body: bytes | None = None, timeout: float = 10.0) -> tuple[int, bytes]:
    """``GET <proxy>/<route>/<path>`` without any environment proxy settings getting in the way."""
    if proxy_url.startswith("unix:"):
        conn = _UnixHTTPConnection(proxy_url[5:], timeout)
    else:
        u = urlsplit(proxy_url)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    target = path if path.startswith(("/", "http://", "https://")) else "/" + path   # absolute-form stays absolute
    conn.request(method, target, body=body, headers=headers or {})
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, data


def fetch_json(proxy_url: str, path: str, **kw) -> tuple[int, dict]:
    status, data = fetch(proxy_url, path, **kw)
    try:
        return status, json.loads(data or b"{}")
    except ValueError:
        return status, {"raw": data[:200].decode("utf-8", "replace")}


SANDBOX_CLIENT = '''\
import http.client as _hc, json as _json, os as _os, socket as _socket
def proxy_get(path, headers=None):
    """GET <proxy>/<route>/<path> via $SANDBOX_PROXY_URL (http://host:port or unix:/path)."""
    url = _os.environ.get("SANDBOX_PROXY_URL", "")
    if url.startswith("unix:"):
        class _C(_hc.HTTPConnection):
            def connect(self):
                self.sock = _socket.socket(_socket.AF_UNIX); self.sock.settimeout(10); self.sock.connect(url[5:])
        c = _C("egress-proxy")
    else:
        h, _, p = url.split("//", 1)[1].partition(":")
        c = _hc.HTTPConnection(h, int(p or 80), timeout=10)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse(); data = r.read(); c.close()
    try:
        return r.status, _json.loads(data)
    except ValueError:
        return r.status, data.decode("utf-8", "replace")
'''
