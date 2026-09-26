"""A shared-prefix agentic workload and a streaming load generator that measures what users feel.

The one idea: agent traffic is *prefix-heavy*. Every call of an agent re-sends the same long
system prompt + tool schemas, and every turn of a session re-sends the whole history plus one
new tool result. So consecutive requests share almost all of their prompt:

    turn 1:  [system+tools ~2k tok][task]
    turn 2:  [system+tools ~2k tok][task][reply 1][tool result 1]
    turn 3:  [system+tools ~2k tok][task][reply 1][tool result 1][reply 2][tool result 2]

A router that sends turn 3 to the replica that served turn 2 prefills ~200 new tokens; a
round-robin router usually prefills the whole history again. This module generates such
sessions deterministically, drives them closed-loop (next turn after the previous reply +
a short tool "think" time) over streaming HTTP, and records TTFT/E2E, which endpoint served
each request (`x-gateway-destination-endpoint`), and how many prompt tokens the engine
reported as cached (`usage.prompt_tokens_details.cached_tokens`).

Numbers measured against the fake backend are real wall-clock on this machine but the backend
timing is *emulated* — compare policies with them, do not quote them as GPU numbers.
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import threading
import time
from dataclasses import dataclass, field

__all__ = ["AgentType", "Session", "make_agent_types", "agentic_sessions", "Record", "BenchResult",
           "run_sessions", "run_bench", "burst", "percentile", "compare", "ascii_bars"]

_WORDS = ("agent must plan call tools read files write patches run tests check results summarize "
          "findings report status handle errors retry safely respect limits log actions cite sources "
          "query database fetch document parse json validate schema update ticket notify user "
          "compute totals compare versions search index rank candidates select best explain choice").split()


def _filler(rng: random.Random, n_words: int) -> str:
    out = []
    for i in range(n_words):
        out.append(rng.choice(_WORDS))
        if i % 12 == 11:
            out[-1] += "."
    return " ".join(out)


@dataclass
class AgentType:
    name: str
    system: str
    tools: list


@dataclass
class Session:
    sid: str
    agent: AgentType
    task: str
    tool_outputs: list


def make_agent_types(n: int = 4, system_words: int = 1500, seed: int = 0) -> list[AgentType]:
    """n agent 'programs', each with its own long system prompt and tool schemas."""
    out = []
    for k in range(n):
        rng = random.Random(seed * 1000 + k)
        name = f"agent-{k}"
        system = f"You are {name}, an autonomous engineering agent. " + _filler(rng, system_words)
        tools = [{"type": "function", "function": {"name": f"{name}_tool_{j}",
                                                   "description": _filler(rng, 20),
                                                   "parameters": {"type": "object", "properties": {
                                                       "query": {"type": "string"}, "limit": {"type": "integer"}},
                                                       "required": ["query"]}}} for j in range(3)]
        out.append(AgentType(name, system, tools))
    return out


def agentic_sessions(n_sessions: int = 24, turns: int = 4, n_agents: int = 4, hot_share: float | None = None,
                     system_words: int = 1500, tool_words: int = 150, seed: int = 0,
                     agents: list | None = None) -> list[Session]:
    """Sessions spread over agent types (uniformly, or `hot_share` of them on agent 0)."""
    agents = agents or make_agent_types(n_agents, system_words, seed)
    rng = random.Random(seed + 7)
    out = []
    for i in range(n_sessions):
        if hot_share is not None and len(agents) > 1:
            a = agents[0] if rng.random() < hot_share else agents[1 + rng.randrange(len(agents) - 1)]
        else:
            a = agents[i % len(agents)]
        task = f"Session {i}: " + _filler(rng, 30)
        tools = [f"Tool result {t} for session {i}: " + _filler(rng, tool_words) for t in range(turns - 1)]
        out.append(Session(f"s{i:03d}", a, task, tools))
    return out


def session_messages(s: Session, replies: list[str]) -> list[dict]:
    """The chat history for the next turn (turn index = len(replies))."""
    msgs = [{"role": "system", "content": s.agent.system}, {"role": "user", "content": s.task}]
    for t, reply in enumerate(replies):
        msgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": s.tool_outputs[t]})
    return msgs


@dataclass
class Record:
    session: str
    turn: int
    agent: str
    t_start: float
    ttft: float | None = None
    e2e: float | None = None
    status: int = 0
    endpoint: str | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    text: str = ""
    error: str = ""


def percentile(xs, q: float) -> float:
    """Linear-interpolation percentile (q in [0, 100]), like numpy's default."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


@dataclass
class BenchResult:
    label: str
    records: list = field(default_factory=list)
    wall_s: float = 0.0
    endpoints: list = field(default_factory=list)     # all endpoint names (so idle ones count as 0)

    @property
    def ok(self) -> list:
        return [r for r in self.records if r.status == 200 and not r.error]

    def per_endpoint(self) -> dict:
        out: dict = {e: 0 for e in self.endpoints}
        for r in self.ok:
            out[r.endpoint] = out.get(r.endpoint, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

    def summary(self) -> dict:
        ok = self.ok
        pe = self.per_endpoint()
        prompt = sum(r.prompt_tokens for r in ok)
        mean = (sum(pe.values()) / len(pe)) if pe else 0
        return {
            "label": self.label, "requests": len(self.records), "errors": len(self.records) - len(ok),
            "ttft_p50_ms": percentile([r.ttft for r in ok], 50) * 1e3,
            "ttft_p90_ms": percentile([r.ttft for r in ok], 90) * 1e3,
            "ttft_p99_ms": percentile([r.ttft for r in ok], 99) * 1e3,
            "e2e_p50_ms": percentile([r.e2e for r in ok], 50) * 1e3,
            "e2e_p90_ms": percentile([r.e2e for r in ok], 90) * 1e3,
            "hit_rate": (sum(r.cached_tokens for r in ok) / prompt) if prompt else 0.0,
            "per_endpoint": pe,
            "imbalance": (max(pe.values()) / mean) if pe else math.nan,
            "rps": len(ok) / self.wall_s if self.wall_s else 0.0,
            "wall_s": self.wall_s,
        }


def _parse_sse_line(line: bytes):
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return "DONE"
    try:
        return json.loads(payload)
    except ValueError:
        return None


async def _request(http, url: str, body: dict, rec: Record, headers: dict | None = None) -> None:
    t0 = time.perf_counter()
    try:
        async with http.post(url, json=body, headers=headers or {}) as r:
            rec.status = r.status
            rec.endpoint = r.headers.get("x-gateway-destination-endpoint") or r.headers.get("x-inference-pod")
            if r.status != 200:
                rec.error = (await r.text())[:200]
                return
            parts = []
            buf = b""
            async for chunk in r.content.iter_any():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    ev = _parse_sse_line(line)
                    if ev is None or ev == "DONE":
                        continue
                    for ch in ev.get("choices") or []:
                        delta = ch.get("delta") or {}
                        piece = delta.get("content") or ch.get("text") or ""
                        if not piece and delta.get("tool_calls"):      # a model (or simulator) calling a tool
                            piece = json.dumps(delta["tool_calls"], separators=(",", ":"))
                        if piece:
                            if rec.ttft is None:
                                rec.ttft = time.perf_counter() - t0
                            parts.append(piece)
                    u = ev.get("usage")
                    if u:
                        rec.prompt_tokens = u.get("prompt_tokens", 0)
                        rec.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
            rec.text = "".join(parts).strip()
    except Exception as e:                       # record, never crash the whole run
        rec.error = f"{type(e).__name__}: {e}"
    finally:
        rec.e2e = time.perf_counter() - t0


async def run_sessions(url: str, sessions: list, turns: int | None = None, max_tokens: int = 24,
                       think_s: tuple = (0.0, 0.02), stagger_s: float = 0.5, model: str = "lab/llm",
                       headers: dict | None = None, seed: int = 0, label: str = "",
                       endpoints=()) -> BenchResult:
    """Closed-loop sessions: each starts at a random offset in [0, stagger_s) and sends its
    next turn when the previous reply has fully arrived (+ a uniform tool time in think_s)."""
    import aiohttp
    rng = random.Random(seed)
    res = BenchResult(label=label, endpoints=list(endpoints))
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    starts = [rng.uniform(0, stagger_s) for _ in sessions]
    thinks = [[rng.uniform(*think_s) for _ in range(64)] for _ in sessions]

    async def one(i, s: Session, http):
        await asyncio.sleep(starts[i])
        replies: list[str] = []
        n = turns or (len(s.tool_outputs) + 1)
        for t in range(n):
            body = {"model": model, "messages": session_messages(s, replies), "tools": s.agent.tools,
                    "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True}}
            rec = Record(s.sid, t, s.agent.name, time.perf_counter() - t_zero)
            res.records.append(rec)
            await _request(http, endpoint, body, rec, headers)
            if rec.error:
                return
            replies.append(rec.text)
            if t + 1 < n:
                await asyncio.sleep(thinks[i][t % 64])

    t_zero = time.perf_counter()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
        await asyncio.gather(*(one(i, s, http) for i, s in enumerate(sessions)))
    res.wall_s = time.perf_counter() - t_zero
    return res


def _run_coro_in_thread(coro_fn):
    """asyncio.run() in a fresh thread, so this works even where a loop is already running
    (Jupyter). Returns the coroutine's result or re-raises its exception."""
    box = {}

    def target():
        try:
            box["result"] = asyncio.run(coro_fn())
        except BaseException as e:               # noqa: BLE001 - re-raised below
            box["error"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_bench(url: str, sessions: list, **kw) -> BenchResult:
    """Synchronous wrapper around run_sessions (safe to call from a notebook cell)."""
    return _run_coro_in_thread(lambda: run_sessions(url, sessions, **kw))


def burst(url: str, bodies: list, headers: dict | None = None) -> list[Record]:
    """Fire all `bodies` at the same instant (a burst) and return their records in order."""
    async def go():
        import aiohttp
        recs = [Record("burst", i, "", 0.0) for i in range(len(bodies))]
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as http:
            await asyncio.gather(*(_request(http, url.rstrip("/") + "/v1/chat/completions", b, r, headers)
                                   for b, r in zip(bodies, recs)))
        return recs
    return _run_coro_in_thread(go)


def compare(results) -> str:
    """A side-by-side table of BenchResult summaries."""
    rows = [r.summary() if isinstance(r, BenchResult) else r for r in results]
    head = f"{'policy':<24}{'TTFT p50':>10}{'p90':>9}{'p99':>9}{'E2E p50':>10}{'hit rate':>10}{'imbal.':>8}  per-endpoint"
    lines = [head, "-" * len(head)]
    for s in rows:
        lines.append(f"{s['label']:<24}{s['ttft_p50_ms']:>8.1f}ms{s['ttft_p90_ms']:>7.1f}ms{s['ttft_p99_ms']:>7.1f}ms"
                     f"{s['e2e_p50_ms']:>8.1f}ms{s['hit_rate']:>10.1%}{s['imbalance']:>8.2f}  {s['per_endpoint']}"
                     + (f"  errors={s['errors']}" if s["errors"] else ""))
    return "\n".join(lines)


def ascii_bars(values: dict, width: int = 40, fmt: str = "{:.1f}") -> str:
    """Horizontal bars for a {label: number} dict (no plotting library needed)."""
    if not values:
        return ""
    mx = max(values.values()) or 1
    w = max(len(str(k)) for k in values)
    return "\n".join(f"{str(k):<{w}} |{'#' * int(round(width * v / mx)):<{width}}| {fmt.format(v)}"
                     for k, v in values.items())
