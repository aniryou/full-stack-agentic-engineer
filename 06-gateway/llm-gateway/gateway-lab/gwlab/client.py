"""A small synchronous HTTP client (standard library) that times a chat completion the way a user sees it.

`chat()` sends one request and, for a stream, reads it event by event: TTFT is the first chunk that carries
text (the role-only chunk does not count), E2E the last byte; it keeps every parsed chunk and the response
headers the gateway adds (`x-gwlab-target`, `x-gwlab-cache`, `x-ratelimit-*`). `read_chunks=k` closes the
connection after k text chunks — a client that walks away mid-stream. `http.client` is used so it works in a
notebook without an event loop.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.parse
from dataclasses import dataclass, field

from . import sse


@dataclass
class Result:
    status: int
    headers: dict
    json: dict | None = None
    chunks: list = field(default_factory=list)
    done: bool = False
    ttft_s: float | None = None
    e2e_s: float = 0.0
    text: str = ""
    usage: dict | None = None
    error: dict | None = None
    raw: bytes = b""

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    def header(self, name: str, default=None):
        return {k.lower(): v for k, v in self.headers.items()}.get(name.lower(), default)


def _conn(url: str, timeout: float):
    u = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port, timeout=timeout), (u.path or "/") + (("?" + u.query) if u.query else "")


def request(method: str, url: str, body=None, headers: dict | None = None, timeout: float = 60.0):
    conn, path = _conn(url, timeout)
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        parsed = json.loads(raw) if raw else None
    except ValueError:
        parsed = None
    return Result(status=resp.status, headers=dict(resp.getheaders()), json=parsed, raw=raw)


def post_json(url: str, body: dict, token: str | None = None, timeout: float = 60.0) -> Result:
    return request("POST", url, body, {"Authorization": f"Bearer {token}"} if token else None, timeout)


def get_json(url: str, token: str | None = None, timeout: float = 30.0) -> Result:
    return request("GET", url, None, {"Authorization": f"Bearer {token}"} if token else None, timeout)


def get_text(url: str, timeout: float = 30.0) -> str:
    return request("GET", url, timeout=timeout).raw.decode()


def chat(base_url: str, key: str, body: dict, *, stream: bool = False, headers: dict | None = None,
         read_chunks: int | None = None, timeout: float = 120.0) -> Result:
    """POST /v1/chat/completions through a gateway (or straight at a server) and time it."""
    body = {**body, "stream": stream}
    conn, _ = _conn(base_url, timeout)
    hdrs = {"Content-Type": "application/json", "Authorization": f"Bearer {key}", **(headers or {})}
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body).encode(), headers=hdrs)
    resp = conn.getresponse()
    res = Result(status=resp.status, headers=dict(resp.getheaders()))
    if resp.status != 200 or not stream:
        res.raw = resp.read()
        res.e2e_s = time.perf_counter() - t0
        conn.close()
        try:
            res.json = json.loads(res.raw)
        except ValueError:
            res.json = None
        if res.json and res.status == 200:
            msg = res.json["choices"][0]["message"]
            res.text, res.usage = msg.get("content") or "", res.json.get("usage")
        elif res.json:
            res.error = res.json.get("error")
        return res
    parser, acc, text_chunks = sse.SSEParser(), sse.StreamAccumulator(), 0
    try:
        while True:
            raw = resp.read1(65536)
            if not raw:
                break
            res.raw += raw
            for ev in parser.feed(raw):
                if ev.data.strip() == sse.DONE:
                    res.done = True
                    continue
                ch = json.loads(ev.data)
                res.chunks.append(ch)
                if ch.get("error"):
                    res.error = ch["error"]
                acc.add(ch)
                delta = ((ch.get("choices") or [{}])[0].get("delta") or {}) if ch.get("choices") else {}
                if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning"):
                    text_chunks += 1
                    if res.ttft_s is None:
                        res.ttft_s = time.perf_counter() - t0
            if read_chunks is not None and text_chunks >= read_chunks:
                break                                   # walk away mid-stream
    finally:
        res.e2e_s = time.perf_counter() - t0
        conn.close()
    res.text, res.usage = acc.content, acc.usage
    res.json = acc.completion()
    return res
