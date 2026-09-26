"""stub.py — a stand-in for the third-party API the agent is allowed to call.

One idea: to show that the sandbox never holds the key, you need an upstream that *checks* the
key and one that *misbehaves*. ``/whoami`` answers whether the expected bearer token arrived
(and never echoes it); ``/echo`` returns the request headers verbatim — the reflection channel an
attacker would use to read an injected credential back through the proxy; ``/data`` returns a
small JSON document for the agent demos; ``/page/<name>`` serves text an attacker wrote (the
prompt-injection payloads of ``agent/scenarios.py``). Standard library only, one file (it runs from a
ConfigMap in kind/GKE as the ``api-stub`` Deployment).

    python3 stub.py --port 8081 --token-file /etc/stub/token
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATA = {"city": "Lisbon", "forecast": [{"day": "Mon", "high_c": 24}, {"day": "Tue", "high_c": 26},
                                       {"day": "Wed", "high_c": 23}]}


class StubAPI:
    def __init__(self, token: str | None = None, *, token_file: str | None = None, host: str = "127.0.0.1",
                 port: int = 0, pages: dict[str, str] | None = None):
        self._token, self._token_file = token, token_file
        self.pages = dict(pages or {})           # /page/<name>: text an attacker controls (the injection)
        self.requests: list[dict] = []          # what arrived (header *names* only, never values)
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                stub._handle(self)

            do_POST = do_GET

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"

    def token(self) -> str:
        if self._token_file:
            try:
                with open(self._token_file) as f:
                    return f.read().strip()
            except OSError:
                return ""
        return self._token or ""

    def _handle(self, h):
        n = int(h.headers.get("Content-Length") or 0)
        if n:
            h.rfile.read(n)
        auth = h.headers.get("Authorization") or ""
        ok = bool(self.token()) and hmac.compare_digest(auth, f"Bearer {self.token()}")
        self.requests.append({"path": h.path, "method": h.command, "authorized": ok,
                              "header_names": sorted(h.headers.keys())})
        path = h.path.split("?")[0]
        if path == "/whoami":
            body = {"authorized": ok, "token_sha256_prefix": hashlib.sha256(auth.encode()).hexdigest()[:12] if auth else None}
            status = 200 if ok else 401
        elif path == "/echo":                    # deliberately unsafe: reflects every header back
            body, status = {"headers": dict(h.headers.items())}, 200
        elif path == "/data":
            body, status = (DATA, 200) if ok else ({"error": "unauthorized"}, 401)
        elif path.startswith("/page/") and path[6:] in self.pages:
            body, status = {"url": path, "text": self.pages[path[6:]]}, 200
        else:
            body, status = {"error": "not found"}, 404
        raw = json.dumps(body).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        h.end_headers()
        h.wfile.write(raw)

    def serve(self) -> "StubAPI":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.serve()

    def __exit__(self, *exc):
        self.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--token-file", default=os.environ.get("STUB_TOKEN_FILE"))
    a = ap.parse_args(argv)
    StubAPI(token_file=a.token_file, host=a.host, port=a.port).httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
