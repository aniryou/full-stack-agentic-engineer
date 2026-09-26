"""Workloads: when requests arrive, how big they are, and which KV blocks they could reuse.

The one idea: an LLM request is not a unit of work. Its cost is set by its prompt and output lengths (with the
defaults below a ~4,000-token RAG prompt is ~3.4x the prefill of a 1,200-token chat turn, and ~20x the chat turn's
uncached part once its system prompt is cached) and its *reuse* by which earlier token streams it shares a
prefix with. So a Request carries its lengths plus the chain hashes of its full KV blocks — the
identity vLLM's prefix cache matches on — and nothing else.

    arrivals(rate, duration, rng)    Poisson times; `rate` may be a list of (t_start, req/s) steps (bursts)
    chat(), rag(), agentic()         request streams; agentic() = closed-loop multi-turn agent sessions
    mix(*streams), expand(reqs)      merge streams; list every request including later session turns
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

M64 = (1 << 64) - 1
CHAT, RAG, AGENT, SYS, DOC = 1, 2, 3, 4, 5          # namespaces for segment ids


def mix64(x: int) -> int:
    """splitmix64 finalizer: a fast, deterministic 64-bit hash (Python's str hash is salted per process)."""
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def seg(*parts: int) -> int:
    """A segment id: one distinct piece of content (a system prompt, a document, one tool result)."""
    h = 0
    for p in parts:
        h = mix64(h ^ (p & M64))
    return h


class HashChain:
    """Chain hashes of the FULL blocks of a growing token stream: h_i = H(h_{i-1}, content of block i).

    A stream is described as (segment, n_tokens) pieces. Block i of two streams hashes equal only if every
    token up to the end of block i came from the same segments at the same offsets — exactly when vLLM would
    find the block in its prefix cache. A partial last block has no hash yet (it is not cacheable).
    """

    def __init__(self, block: int = 16):
        self.block, self.hashes, self.tokens = block, [], 0
        self._parent = self._acc = self._fill = 0

    def extend(self, segment: int, n: int) -> "HashChain":
        pos = 0
        while pos < n:
            take = min(n - pos, self.block - self._fill)
            self._acc = mix64(self._acc ^ mix64(segment * 1_000_003 + pos) ^ take)
            pos, self._fill = pos + take, self._fill + take
            if self._fill == self.block:
                self._parent = mix64(self._parent ^ self._acc)
                self.hashes.append(self._parent)
                self._acc = self._fill = 0
        self.tokens += n
        return self

    def fork(self) -> "HashChain":
        c = HashChain(self.block)
        c.hashes, c.tokens = list(self.hashes), self.tokens
        c._parent, c._acc, c._fill = self._parent, self._acc, self._fill
        return c


@dataclass(eq=False)
class Request:
    rid: int
    arrival: float
    prompt: int                      # prompt tokens
    output: int                      # tokens to generate
    hashes: list = field(repr=False)  # chain hashes of full blocks of prompt+output (a session shares one list)
    nblocks: int = 0                 # how many of `hashes` belong to this request: (prompt + output) // block
    block: int = 16
    kind: str = "chat"
    group: int = 0                   # which shared prefix (app / agent / system prompt)
    session: int = -1
    turn: int = 0
    lora: str | None = None
    think: float = 0.0               # delay before this session's next turn arrives (tool or user time)
    priority: int = 0                # InferenceObjective priority: higher is dispatched first under flow control
    next: "Request | None" = field(default=None, repr=False)
    # written by the simulator
    t_dispatch: float | None = None
    t_first: float | None = None
    t_done: float | None = None
    replica: int = -1
    cached: int = 0                  # prompt tokens served from the prefix cache at first admission
    preempted: int = 0
    shed: bool = False               # rejected by flow control (TTL or a full queue): never served

    @property
    def pblocks(self) -> int:
        """Full blocks of the prompt: the most a prefix cache can match for this request."""
        return min(self.nblocks, self.prompt // self.block)

    @property
    def tpot(self) -> float:
        """Mean time per output token after the first (vLLM's request_time_per_output_token)."""
        return (self.t_done - self.t_first) / max(1, self.output - 1)


def arrivals(rate, duration: float, rng: random.Random) -> list[float]:
    """Poisson arrival times on [0, duration). `rate` is req/s, or a list of (t_start, rate) steps."""
    steps = [(0.0, float(rate))] if isinstance(rate, (int, float)) else sorted(rate)
    out = []
    for i, (t0, lam) in enumerate(steps):
        end = min(steps[i + 1][0] if i + 1 < len(steps) else duration, duration)
        t = t0
        while lam > 0:
            t += rng.expovariate(lam)
            if t >= end:
                break
            out.append(t)
    return out


def burst(base: float, peak: float, start: float, end: float) -> list[tuple[float, float]]:
    """A traffic step: `base` req/s, `peak` between start and end, then back to `base`."""
    return [(0.0, base), (start, peak), (end, base)]


def lognormal(rng: random.Random, mean: float, cv: float) -> int:
    """A positive integer length with the given mean and coefficient of variation (heavy right tail)."""
    if cv <= 0:
        return max(1, int(mean))
    s2 = math.log(1 + cv * cv)
    return max(1, int(rng.lognormvariate(math.log(mean) - s2 / 2, math.sqrt(s2))))


def _choose(rng, n, zipf):
    return rng.choices(range(n), [1 / (i + 1) ** zipf for i in range(n)])[0] if n > 1 else 0


def _lora(rng, loras, zipf):
    return loras[_choose(rng, len(loras), zipf)] if loras else None


def chat(rate, duration, *, seed=0, groups=1, zipf=0.0, system=1000, user=200, output=150, cv=0.6,
         block=16, loras=None, lora_zipf=1.0) -> list[Request]:
    """Independent chat turns: one of `groups` shared system prompts (Zipf-popular) + a unique message."""
    rng = random.Random(seed)
    heads = [HashChain(block).extend(seg(SYS, CHAT, g), system) for g in range(groups)]
    out = []
    for i, t in enumerate(arrivals(rate, duration, rng)):
        g = _choose(rng, groups, zipf)
        u, o = lognormal(rng, user, cv), lognormal(rng, output, cv)
        ch = heads[g].fork().extend(seg(CHAT, seed, i, 0), u).extend(seg(CHAT, seed, i, 1), o)
        out.append(Request(i, t, system + u, o, ch.hashes, (system + u + o) // block, block, "chat", g,
                           lora=_lora(rng, loras, lora_zipf)))
    return out


def rag(rate, duration, *, seed=0, system=400, docs=3, doc=1200, corpus=200, zipf=1.0, question=60,
        output=200, cv=0.4, block=16) -> list[Request]:
    """RAG: system prompt + `docs` retrieved chunks from a Zipf-popular corpus (in rank order) + a question.
    Long prompts, short answers — prefill-heavy; requests share a prefix only as far as their leading docs agree."""
    rng = random.Random(seed)
    head = HashChain(block).extend(seg(SYS, RAG, 0), system)
    out = []
    for i, t in enumerate(arrivals(rate, duration, rng)):
        picks = sorted({_choose(rng, corpus, zipf) for _ in range(docs)})
        ch = head.fork()
        for d in picks:
            ch.extend(seg(DOC, d), doc)
        q, o = lognormal(rng, question, cv), lognormal(rng, output, cv)
        ch.extend(seg(RAG, seed, i, 0), q).extend(seg(RAG, seed, i, 1), o)
        p = system + doc * len(picks) + q
        out.append(Request(i, t, p, o, ch.hashes, (p + o) // block, block, "rag", picks[0]))
    return out


def agentic(session_rate, duration, *, seed=0, groups=4, zipf=1.0, system=3000, task=300, turns=(3, 10),
            tool=600, output=120, think=(1.0, 5.0), cv=0.5, block=16, loras=None, lora_zipf=1.0) -> list[Request]:
    """Closed-loop agent sessions. Turn k's prompt = the whole history (system + task + every earlier output and
    tool result) + one new tool result; turn k+1 arrives `think` s after turn k finishes (tool latency).
    Returns the FIRST turn of each session; later turns hang off `.next` and the simulator releases them."""
    rng = random.Random(seed)
    heads = [HashChain(block).extend(seg(SYS, AGENT, g), system) for g in range(groups)]
    firsts, rid = [], 0
    for s, t in enumerate(arrivals(session_rate, duration, rng)):
        g, n = _choose(rng, groups, zipf), rng.randint(*turns)
        lora, ch, ctx, prev = _lora(rng, loras, lora_zipf), heads[g].fork(), system, None
        for k in range(n):
            new = task if k == 0 else lognormal(rng, tool, cv)
            o = lognormal(rng, output, cv)
            ch.extend(seg(AGENT, seed, s, k, 0), new).extend(seg(AGENT, seed, s, k, 1), o)
            r = Request(rid, t if k == 0 else 0.0, ctx + new, o, ch.hashes, (ctx + new + o) // block, block,
                        "agent", g, s, k, lora, rng.uniform(*think) if k < n - 1 else 0.0)
            rid, ctx = rid + 1, ctx + new + o
            if prev is None:
                firsts.append(r)
            else:
                prev.next = r
            prev = r
    return firsts


def expand(requests) -> list[Request]:
    """Every request including later session turns (follows `.next`)."""
    out = []
    for r in requests:
        while r is not None:
            out.append(r)
            r = r.next
    return out


def mix(*streams) -> list[Request]:
    """Merge streams by arrival time and renumber request ids (session turns included)."""
    firsts = sorted((r for s in streams for r in s), key=lambda r: r.arrival)
    for i, r in enumerate(expand(firsts)):
        r.rid = i
    return firsts
