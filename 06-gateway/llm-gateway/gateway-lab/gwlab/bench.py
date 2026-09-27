"""Scripted tenants: open-loop load through the gateway, one Poisson arrival stream per tenant.

The one idea: a gateway's policies only show under contention between tenants — a noisy neighbour, a burst of
long streams, a provider that starts shedding. `run()` drives several tenants at once (each with its own key,
rate, request shape and behaviour, e.g. walking away mid-stream) over real HTTP and records what each request
saw: status, TTFT, E2E, which target served it, cache outcome, token counts (from the usage chunk the client
asked for). Open loop, because a closed loop slows down with the gateway and hides the queue (04 PRIMER §11).
Arrivals are seeded, so two runs against two configurations see the same traffic.
"""
from __future__ import annotations

import asyncio
import json
import random
import statistics
import threading
import time
from dataclasses import dataclass, field

import aiohttp

from . import sse


@dataclass
class TenantScript:
    name: str
    key: str
    rate: float                                  # requests per second (Poisson)
    n: int
    model: str = "chat"
    stream: bool = True
    include_usage: bool = True
    prompt: object = None                        # str, or callable(i) -> str; default: a unique short question
    max_tokens: int | None = None
    temperature: float | None = None
    disconnect_after: int | None = None          # walk away after this many text chunks
    extra: dict = field(default_factory=dict)


@dataclass
class Sample:
    tenant: str
    i: int
    t_send: float
    status: int = 0
    ttft_s: float | None = None
    e2e_s: float = 0.0
    target: str = ""
    cache: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    text_chunks: int = 0
    error: str = ""
    retry_after: str | None = None


@dataclass
class BenchResult:
    samples: list
    duration_s: float
    label: str = ""

    def tenant(self, name: str) -> list:
        return [s for s in self.samples if s.tenant == name]

    def summary(self) -> dict:
        out = {}
        for name in sorted({s.tenant for s in self.samples}):
            ss = self.tenant(name)
            ok = [s for s in ss if s.status == 200 and not s.error]
            ttft = sorted(s.ttft_s for s in ok if s.ttft_s is not None)
            out[name] = {"sent": len(ss), "ok": len(ok), "429": sum(s.status == 429 for s in ss),
                         "5xx": sum(s.status >= 500 for s in ss), "stream_errors": sum(s.status == 200 and bool(s.error) for s in ss),
                         "ttft_p50_ms": _q(ttft, 0.5) * 1e3 if ttft else None, "ttft_p95_ms": _q(ttft, 0.95) * 1e3 if ttft else None,
                         "output_tokens": sum(s.completion_tokens or 0 for s in ok),
                         "targets": dict(sorted(_count(s.target for s in ok).items()))}
        return out

    def table(self) -> str:
        rows = [f"{'tenant':10s} {'sent':>5s} {'ok':>4s} {'429':>4s} {'5xx':>4s} {'TTFT p50':>9s} {'p95':>8s}  served by"]
        for name, r in self.summary().items():
            p50 = f"{r['ttft_p50_ms']:.0f} ms" if r["ttft_p50_ms"] is not None else "-"
            p95 = f"{r['ttft_p95_ms']:.0f} ms" if r["ttft_p95_ms"] is not None else "-"
            rows.append(f"{name:10s} {r['sent']:5d} {r['ok']:4d} {r['429']:4d} {r['5xx']:4d} {p50:>9s} {p95:>8s}  {r['targets']}")
        return "\n".join(rows)


def _q(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def _count(it):
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def arrivals(rate: float, n: int, rng: random.Random) -> list:
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


async def _one(http, url: str, sc: TenantScript, i: int, t0: float) -> Sample:
    p = sc.prompt(i) if callable(sc.prompt) else (sc.prompt or f"Question {i} from {sc.name}: summarise item {i * 7919 % 1000}.")
    body = {"model": sc.model, "messages": [{"role": "user", "content": p}], "stream": sc.stream, **sc.extra}
    if sc.stream and sc.include_usage:
        body["stream_options"] = {"include_usage": True}
    if sc.max_tokens:
        body["max_completion_tokens"] = sc.max_tokens
    if sc.temperature is not None:
        body["temperature"] = sc.temperature
    s = Sample(sc.name, i, time.perf_counter() - t0)
    start = time.perf_counter()
    try:
        async with http.post(url + "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {sc.key}"}) as r:
            s.status, s.target, s.cache = r.status, r.headers.get("x-gwlab-target", ""), r.headers.get("x-gwlab-cache", "")
            s.retry_after = r.headers.get("Retry-After")
            if r.status != 200:
                d = await r.json(content_type=None)
                s.error = ((d or {}).get("error") or {}).get("code", str(r.status))
                return s
            if not sc.stream:
                d = await r.json()
                u = d.get("usage") or {}
                s.prompt_tokens, s.completion_tokens = u.get("prompt_tokens"), u.get("completion_tokens")
                s.ttft_s = time.perf_counter() - start
                return s
            parser = sse.SSEParser()
            async for raw in r.content.iter_any():
                for ev in parser.feed(raw):
                    if ev.data.strip() == sse.DONE:
                        continue
                    ch = json.loads(ev.data)
                    if ch.get("error"):
                        s.error = str(ch["error"].get("code"))
                    if ch.get("usage"):
                        s.prompt_tokens, s.completion_tokens = ch["usage"]["prompt_tokens"], ch["usage"]["completion_tokens"]
                    d = (ch.get("choices") or [{}])[0].get("delta") if ch.get("choices") else None
                    if d and (d.get("content") or d.get("tool_calls")):
                        s.text_chunks += 1
                        if s.ttft_s is None:
                            s.ttft_s = time.perf_counter() - start
                if sc.disconnect_after is not None and s.text_chunks >= sc.disconnect_after:
                    s.error = "client_disconnect"
                    r.close()
                    break
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        s.error = type(e).__name__
    finally:
        s.e2e_s = time.perf_counter() - start
    return s


async def arun(url: str, scripts: list, seed: int = 0) -> BenchResult:
    rng = random.Random(seed)
    plan = sorted((t, sc, i) for sc in scripts for i, t in enumerate(arrivals(sc.rate, sc.n, rng)))
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                                     connector=aiohttp.TCPConnector(limit=0)) as http:
        tasks = []
        for t, sc, i in plan:
            delay = t - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.ensure_future(_one(http, url, sc, i, t0)))
        samples = await asyncio.gather(*tasks)
    return BenchResult(list(samples), time.perf_counter() - t0)


def run(url: str, scripts: list, seed: int = 0, label: str = "") -> BenchResult:
    """Run the bench on a private event loop in a thread (works from a notebook, whose loop is busy)."""
    box = {}

    def target():
        box["r"] = asyncio.run(arun(url, scripts, seed))

    th = threading.Thread(target=target, name="gwlab-bench")
    th.start()
    th.join()
    box["r"].label = label
    return box["r"]


def percentile(xs, q: float) -> float:
    return statistics.quantiles(xs, n=100)[int(q * 100) - 1] if len(xs) > 1 else (xs[0] if xs else float("nan"))
