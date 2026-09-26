"""runner.py — drive a server with a workload: open loop, closed loop, or agent sessions.

One idea: *who decides when the next request is sent* changes what you measure. In an **open
loop** requests arrive at a rate whatever the server is doing, like independent users: past the
knee a queue builds and TTFT explodes, which is the truth about overload. In a **closed loop**
N users each wait for their answer before sending the next: a slow server slows its own
arrivals, so latency looks tame while throughput quietly drops (coordinated omission). Agent
traffic is a mix — sessions arrive open-loop, turns inside a session are closed-loop because
each turn needs the previous reply.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import math
import time
from dataclasses import dataclass, field

import aiohttp

from .client import RequestResult, discover_model, is_simulated, stream_request
from .summary import SLO, Summary, summarize
from .workload import AgentSession, Request, arrival_times


@dataclass
class BenchRun:
    results: list
    duration_s: float
    mode: str
    params: dict
    target: str
    model: str
    simulated: bool
    started_at: float = field(default_factory=time.time)

    @property
    def source(self) -> str:
        return "SIMULATED (servelab fake server)" if self.simulated else f"measured on {self.target}"

    def summary(self, slo: SLO | None = None, label: str = "", tag: str | None = None) -> Summary:
        res = [r for r in self.results if tag is None or r.tag == tag]
        return summarize(res, self.duration_s, slo, label=label or self.mode)

    def report(self, slo: SLO | None = None) -> str:
        return f"[{self.source}] {self.mode} {self.params}\n" + self.summary(slo).text()

    def to_dict(self) -> dict:
        return {"mode": self.mode, "params": self.params, "target": self.target, "model": self.model,
                "simulated": self.simulated, "started_at": self.started_at, "duration_s": self.duration_s,
                "results": [{k: v for k, v in r.__dict__.items() if k not in ("text", "chunk_times")}
                            for r in self.results]}


def run_sync(coro):
    """Run a coroutine from sync code — also inside Jupyter, where an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _session(timeout_s: float = 3600) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                 timeout=aiohttp.ClientTimeout(total=timeout_s))


async def _prepare(http, url, model, headers):
    model = model or await discover_model(http, url, headers)
    return model, await is_simulated(http, url, headers)


def warmup_request(req: Request, i: int) -> Request:
    """A short copy of ``req`` whose *first* token differs, so warm-up cannot leave prefix-cache
    hits behind for the measured requests (a warm-up that replays measured prompts flatters TTFT)."""
    tag = f"warmup{i} "
    if req.messages is not None:
        msgs = [dict(m) for m in req.messages]
        msgs[0]["content"] = tag + str(msgs[0].get("content") or "")
        return Request(messages=msgs, max_tokens=min(req.max_tokens, 16), tag="warmup")
    return Request(prompt=tag + (req.prompt or ""), max_tokens=min(req.max_tokens, 16), tag="warmup")


async def _warmup(http, url, model, requests, n, headers, ignore_eos):
    """First requests pay one-off costs (connection setup, lazy compilation, cold caches)."""
    for i, r in enumerate(requests[:n]):
        await stream_request(http, url, model, warmup_request(r, i), headers=headers, ignore_eos=ignore_eos)


async def open_loop(url: str, requests: list, rate: float = math.inf, burstiness: float = 1.0, seed: int = 0, *,
                    model: str | None = None, headers: dict | None = None, max_concurrency: int | None = None,
                    warmup: int = 0, ignore_eos: bool = True) -> BenchRun:
    """Send ``requests`` at ``rate`` req/s (gamma/Poisson gaps, vLLM semantics); optional cap on
    requests in flight (``max_concurrency``, like ``vllm bench serve --max-concurrency``)."""
    async with _session() as http:
        model, sim = await _prepare(http, url, model, headers)
        await _warmup(http, url, model, requests, warmup, headers, ignore_eos)
        offsets = arrival_times(len(requests), rate, burstiness, seed)
        sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        t0 = time.perf_counter()

        async def one(req: Request, at: float) -> RequestResult:
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if sem is None:
                return await stream_request(http, url, model, req, headers=headers, ignore_eos=ignore_eos)
            async with sem:
                return await stream_request(http, url, model, req, headers=headers, ignore_eos=ignore_eos)

        results = await asyncio.gather(*(one(r, a) for r, a in zip(requests, offsets)))
        duration = time.perf_counter() - t0
    return BenchRun(list(results), duration, "open-loop",
                    {"rate": rate, "burstiness": burstiness, "n": len(requests), "max_concurrency": max_concurrency},
                    url, model, sim)


async def closed_loop(url: str, requests: list, concurrency: int, think_time_s: float = 0.0, *,
                      ramp_s: float = 0.0, model: str | None = None, headers: dict | None = None, warmup: int = 0,
                      ignore_eos: bool = True) -> BenchRun:
    """``concurrency`` users; each sends its next request when its previous answer is complete
    (plus ``think_time_s``). ``ramp_s`` staggers the users' first requests over that many seconds:
    N users starting in the same instant is a burst of N prefills that no real traffic produces."""
    async with _session() as http:
        model, sim = await _prepare(http, url, model, headers)
        await _warmup(http, url, model, requests, warmup, headers, ignore_eos)
        queue = list(reversed(requests))
        results: list = []
        t0 = time.perf_counter()

        async def user(i: int):
            if ramp_s:
                await asyncio.sleep(i * ramp_s / concurrency)
            while queue:
                req = queue.pop()
                results.append(await stream_request(http, url, model, req, headers=headers, ignore_eos=ignore_eos))
                if think_time_s:
                    await asyncio.sleep(think_time_s)

        await asyncio.gather(*(user(i) for i in range(concurrency)))
        duration = time.perf_counter() - t0
    return BenchRun(results, duration, "closed-loop", {"concurrency": concurrency, "think_time_s": think_time_s,
                                                       "ramp_s": ramp_s, "n": len(requests)}, url, model, sim)


async def sessions(url: str, agent_sessions: list, session_rate: float = math.inf, burstiness: float = 1.0,
                   seed: int = 0, *, model: str | None = None, headers: dict | None = None,
                   ignore_eos: bool = True) -> BenchRun:
    """Agent sessions: session starts are open-loop at ``session_rate``; inside a session each turn
    is sent after the previous reply arrived (and the tool 'ran' for ``think_time_s``), with the
    reply appended to the history — so the conversation prefix only ever grows."""
    async with _session() as http:
        model, sim = await _prepare(http, url, model, headers)
        offsets = arrival_times(len(agent_sessions), session_rate, burstiness, seed)
        t0 = time.perf_counter()
        results: list = []

        async def run(s: AgentSession, at: float):
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            history: list = []
            for turn, content in enumerate(s.turns):
                history.append({"role": "user", "content": content})
                msgs = [{"role": "system", "content": s.system(turn)}] + history
                r = await stream_request(http, url, model, Request(messages=msgs, max_tokens=s.max_tokens,
                                                                   tag=f"turn{turn}"),
                                         headers=headers, ignore_eos=ignore_eos)
                r.session, r.turn = s.sid, turn
                results.append(r)
                if not r.ok:
                    return
                history.append({"role": "assistant", "content": r.text})
                if s.think_time_s:
                    await asyncio.sleep(s.think_time_s)

        await asyncio.gather(*(run(s, a) for s, a in zip(agent_sessions, offsets)))
        duration = time.perf_counter() - t0
    return BenchRun(results, duration, "agent-sessions",
                    {"sessions": len(agent_sessions), "session_rate": session_rate, "turns": len(agent_sessions[0].turns)
                     if agent_sessions else 0}, url, model, sim)


# -- sync wrappers (scripts, tests, notebooks) ---------------------------------------------------
def warm_up(url: str, requests: list, n: int = 2, *, model: str | None = None, headers: dict | None = None,
            ignore_eos: bool = True) -> None:
    """Send ``n`` warm-up requests (distinct prompts, see :func:`warmup_request`) and wait for them.
    Do this *before* taking the "before" /metrics scrape so warm-up stays out of the window."""
    async def go():
        async with _session() as http:
            m = model or await discover_model(http, url, headers)
            await _warmup(http, url, m, requests, n, headers, ignore_eos)
    run_sync(go())


def run_open_loop(url: str, requests: list, rate: float = math.inf, **kw) -> BenchRun:
    return run_sync(open_loop(url, requests, rate, **kw))


def run_closed_loop(url: str, requests: list, concurrency: int, **kw) -> BenchRun:
    return run_sync(closed_loop(url, requests, concurrency, **kw))


def run_sessions(url: str, agent_sessions: list, session_rate: float = math.inf, **kw) -> BenchRun:
    return run_sync(sessions(url, agent_sessions, session_rate, **kw))
