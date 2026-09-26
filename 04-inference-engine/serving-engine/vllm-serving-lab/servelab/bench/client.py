"""client.py — one streaming request, timed the way ``vllm bench serve`` times it.

One idea: TTFT and ITL are properties of the *stream*, so they are measured at the client from
the arrival time of each server-sent-events chunk that carries a token:

    token chunk = a chunk with ``choices`` (text, content or a finish_reason), except a chat
                  role-only chunk (see below)
    TTFT = t(first token chunk) - t(send)
    ITL  = the gaps between consecutive token chunks
    E2E  = t(last token chunk) - t(send)
    TPOT = (E2E - TTFT) / (output_tokens - 1)      per request; output_tokens from ``usage``

These are the definitions in vLLM's ``vllm/benchmarks/lib/endpoint_request_func.py`` (v0.30.0),
with one deliberate difference. Two consequences worth saying out loud: TTFT includes network,
HTTP, tokenization and *queueing* (the server-side histogram ``vllm:time_to_first_token_seconds``
starts at arrival in the engine); and ITL is per *chunk*, so with speculative decoding or
``--stream-interval`` > 1 one chunk can carry several tokens — then ITL and TPOT differ, and TPOT
is the per-token number.

The difference: a chat stream opens with a role-only chunk (``delta: {"role": "assistant",
"content": ""}``), which vLLM sends in the same engine iteration as, and just before, the first
content chunk. ``vllm bench serve`` treats every chunk with ``choices`` as a stream chunk, so the
role chunk sets its TTFT and the first content chunk adds a ~0 ms entry to that request's ITL
list (its token count comes from ``usage``, so TPOT and E2E are unaffected). Here the role-only
chunk is skipped: TTFT is the same to within microseconds, each chat request's ITL list has one
fewer, near-zero entry, so chat ITL mean and low percentiles read slightly higher (by ~1/n for n
output tokens). For ``/v1/completions`` the definitions are identical. Pass
``ttft_on_content=True`` to also ignore empty-text chunks before the first real token (servers
that send one early).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

import aiohttp

from .workload import Request


@dataclass
class RequestResult:
    ok: bool = False
    start: float = 0.0                  # time.perf_counter() when the request was sent
    ttft: float = math.nan              # seconds
    itl: list = field(default_factory=list)
    latency: float = math.nan           # E2E seconds
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int | None = None    # usage.prompt_tokens_details.cached_tokens, if the server reports it
    chunk_times: list = field(default_factory=list)
    text: str = ""
    error: str = ""
    tag: str = ""
    session: str = ""
    turn: int = 0

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first; NaN for single-token outputs."""
        if not self.ok or self.output_tokens <= 1:
            return math.nan
        return (self.latency - self.ttft) / (self.output_tokens - 1)


class SSEParser:
    """Reassemble server-sent events from arbitrary network chunks; yields ``data:`` payloads."""

    def __init__(self):
        self.buf = b""

    def feed(self, data: bytes) -> list:
        self.buf += data
        out = []
        while b"\n\n" in self.buf:
            event, self.buf = self.buf.split(b"\n\n", 1)
            for line in event.decode("utf-8", "replace").splitlines():
                if line.startswith(":"):      # SSE comment / keep-alive ping
                    continue
                if line.startswith("data:"):
                    out.append(line[5:].strip())
        return out


def build_payload(req: Request, model: str, ignore_eos: bool = True) -> tuple:
    """(path, json body) for a request — the body ``vllm bench serve`` sends, plus temperature 0."""
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.0}
    if ignore_eos:
        body["ignore_eos"] = True    # vLLM extension: generate exactly max_tokens (fixed output lengths)
    if req.messages is not None:
        body.update(messages=req.messages, max_completion_tokens=req.max_tokens)
        return "/v1/chat/completions", body
    body.update(prompt=req.prompt, max_tokens=req.max_tokens)
    return "/v1/completions", body


async def stream_request(session: aiohttp.ClientSession, base_url: str, model: str, req: Request, *,
                         headers: dict | None = None, ignore_eos: bool = True,
                         ttft_on_content: bool = False) -> RequestResult:
    path, body = build_payload(req, model, ignore_eos)
    res = RequestResult(prompt_tokens=req.prompt_tokens, tag=req.tag)
    parser, text = SSEParser(), []
    st = time.perf_counter()
    res.start, last = st, st
    try:
        async with session.post(base_url.rstrip("/") + path, json=body, headers=headers or {}) as resp:
            if resp.status != 200:
                res.error = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return res
            async for chunk in resp.content.iter_any():
                for payload in parser.feed(chunk):
                    if payload == "[DONE]":
                        continue
                    data = json.loads(payload)
                    choices = data.get("choices")
                    if choices:
                        c0 = choices[0]
                        delta = c0.get("delta") or {}
                        piece = c0.get("text") if "text" in c0 else delta.get("content")
                        if "role" in delta and not piece and c0.get("finish_reason") is None:
                            continue                      # role-only chat chunk: not a token
                        if ttft_on_content and not res.chunk_times and not piece:
                            continue
                        now = time.perf_counter()
                        if not res.chunk_times:
                            res.ttft = now - st
                        else:
                            res.itl.append(now - last)
                        res.chunk_times.append(now)
                        last = now
                        text.append(piece or "")
                    elif data.get("usage"):
                        u = data["usage"]
                        res.output_tokens = int(u.get("completion_tokens") or 0)
                        res.prompt_tokens = int(u.get("prompt_tokens") or res.prompt_tokens)
                        details = u.get("prompt_tokens_details") or {}
                        if details.get("cached_tokens") is not None:
                            res.cached_tokens = int(details["cached_tokens"])
        if res.chunk_times:
            res.ok = True
            res.latency = last - st
            res.output_tokens = res.output_tokens or len(res.chunk_times)
        else:
            res.error = "no token chunks received"
    except Exception as e:  # noqa: BLE001 — a failed request is data, not a crash
        res.error = f"{type(e).__name__}: {e}"
    res.text = "".join(text)
    return res


async def discover_model(session: aiohttp.ClientSession, base_url: str, headers: dict | None = None) -> str:
    async with session.get(base_url.rstrip("/") + "/v1/models", headers=headers or {}) as r:
        r.raise_for_status()
        return (await r.json())["data"][0]["id"]


async def is_simulated(session: aiohttp.ClientSession, base_url: str, headers: dict | None = None) -> bool:
    """True when the target is this lab's fake server (so reports can say "simulated")."""
    try:
        async with session.get(base_url.rstrip("/") + "/version", headers=headers or {}) as r:
            return bool(r.status == 200 and (await r.json(content_type=None)).get("simulated"))
    except Exception:  # noqa: BLE001
        return False
