"""workload.py — what you send decides what you measure: lengths, arrivals, sharing.

One idea: a serving benchmark is only as good as its workload, and three properties dominate
the result. **Length distributions** set the work: prompt tokens are prefill (compute-bound),
output tokens are decode steps (bandwidth-bound). The **arrival process** sets the queueing:
open-loop arrivals at a rate (Poisson, or burstier) versus closed-loop users. **Sharing** sets
the prefix-cache hit rate: a system prompt and tool schemas repeated across sessions, a history
that is only ever appended to. Every workload here is explicit and seeded, so two runs differ
only in what you changed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..textgen import synthetic_text


@dataclass
class Request:
    """One request: a completions ``prompt`` or chat ``messages``, and how many tokens to generate."""
    prompt: str | None = None
    messages: list | None = None
    max_tokens: int = 128
    prompt_tokens: int = 0     # intended prompt length under the toy tokenizer
    tag: str = ""              # label to split results later ("short", "long", ...)


@dataclass(frozen=True)
class Lengths:
    """A token-length distribution: ``fixed``, ``uniform``, ``lognormal`` (median, sigma) or ``choice``."""
    kind: str
    a: float = 0
    b: float = 0
    lo: int = 1
    hi: int = 1_000_000
    values: tuple = ()

    @staticmethod
    def fixed(n: int) -> "Lengths":
        return Lengths("fixed", n)

    @staticmethod
    def uniform(lo: int, hi: int) -> "Lengths":
        return Lengths("uniform", lo, hi)

    @staticmethod
    def lognormal(median: float, sigma: float = 0.8, lo: int = 1, hi: int = 1_000_000) -> "Lengths":
        """Long-tailed, like real chat traffic: most requests short, a few very long."""
        return Lengths("lognormal", median, sigma, lo, hi)

    @staticmethod
    def choice(values) -> "Lengths":
        return Lengths("choice", values=tuple(values))

    def sample(self, rng: random.Random) -> int:
        if self.kind == "fixed":
            return int(self.a)
        if self.kind == "uniform":
            return rng.randint(int(self.a), int(self.b))
        if self.kind == "lognormal":
            return max(self.lo, min(self.hi, int(round(rng.lognormvariate(math.log(self.a), self.b)))))
        if self.kind == "choice":
            return int(rng.choice(self.values))
        raise ValueError(self.kind)


def random_requests(n: int, input_len: Lengths = Lengths.fixed(512), output_len: Lengths = Lengths.fixed(128),
                    prefix_tokens: int = 0, seed: int = 0, tag: str = "") -> list:
    """``n`` completions requests with unique random prompts (no accidental prefix sharing), plus an
    optional shared prefix of ``prefix_tokens`` in front of every prompt (vLLM bench's ``--random-prefix-len``)."""
    rng = random.Random(seed)
    prefix = synthetic_text(prefix_tokens, random.Random(seed + 10_007)) if prefix_tokens else ""
    out = []
    for _ in range(n):
        k = input_len.sample(rng)
        body = synthetic_text(k, rng)
        prompt = (prefix + " " + body) if prefix else body
        out.append(Request(prompt=prompt, max_tokens=output_len.sample(rng), prompt_tokens=prefix_tokens + k, tag=tag))
    return out


def mixed_requests(n_short: int, n_long: int, short_in: int = 256, long_in: int = 3000, output: int = 64,
                   seed: int = 0) -> list:
    """Chat-sized requests with a few long (RAG-sized) prompts mixed in, tagged ``short``/``long``:
    the workload that exposes prefill/decode interference (notebook 03)."""
    reqs = (random_requests(n_short, Lengths.fixed(short_in), Lengths.fixed(output), seed=seed, tag="short")
            + random_requests(n_long, Lengths.fixed(long_in), Lengths.fixed(output), seed=seed + 1, tag="long"))
    random.Random(seed).shuffle(reqs)
    return reqs


def arrival_times(n: int, rate: float, burstiness: float = 1.0, seed: int = 0) -> list:
    """Send offsets (seconds) for an open loop, with ``vllm bench serve`` semantics.

    Inter-arrival gaps are Gamma(shape=burstiness, scale=1/(rate·burstiness)): mean 1/rate;
    burstiness 1 is a Poisson process, < 1 is burstier, ``inf`` is evenly spaced. The cumulative
    offsets are then rescaled so the last request goes out at exactly n/rate (vLLM does this to
    make throughput comparable across seeds). ``rate=inf`` sends everything at t=0."""
    if n <= 0:
        return []
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    if math.isinf(burstiness):
        gaps = [1.0 / rate] * n
    else:
        theta = 1.0 / (rate * burstiness)
        gaps = [rng.gammavariate(burstiness, theta) for _ in range(n)]
    t, out = 0.0, []
    for g in gaps:
        t += g
        out.append(t)
    scale = (n / rate) / out[-1] if out[-1] > 0 else 1.0
    return [x * scale for x in out]


# ---------------------------------------------------------------------------------------------
# Agent sessions: shared system prompt + tool schemas, append-only history
# ---------------------------------------------------------------------------------------------
@dataclass
class AgentSession:
    """One agent conversation. Turn 0 is the user's request; later turns are tool results. The
    runner appends the model's actual reply after each turn, as a real agent loop does."""
    sid: str
    system_parts: list              # pieces of the system prompt, in order
    turns: list                     # user / tool-result contents
    max_tokens: int = 48
    think_time_s: float = 0.0       # tool execution time between turns
    timestamp_first: bool = False   # the anti-pattern: a fresh timestamp at the top of every request
    extras: dict = field(default_factory=dict)

    def system(self, turn: int) -> str:
        body = "\n".join(self.system_parts)
        if self.timestamp_first:
            return f"Current time: 2026-09-26T10:{turn:02d}:{random.Random(hash((self.sid, turn))).random():.9f}\n{body}"
        return body


def agent_sessions(n_sessions: int, turns: int = 4, system_tokens: int = 600, tools_tokens: int = 900,
                   n_tools: int = 6, user_tokens: int = 60, tool_result_tokens: int = 250, output_tokens: int = 48,
                   layout: str = "stable", think_time_s: float = 0.0, seed: int = 0) -> list:
    """Sessions of a tool-using agent. ``layout`` decides what the prefix cache can reuse:

    * ``"stable"``: one system prompt and one tool list, shared by every session, then an
      append-only history — every block before the newest message can hit;
    * ``"timestamp_first"``: the same, but a per-request timestamp on the first line — the first
      block differs every time, so *nothing* hits (a surprisingly common template bug);
    * ``"shuffled_tools"``: each session lists the tools in its own order — sessions stop sharing
      each other's prefix, but turns within a session still do."""
    if layout not in ("stable", "timestamp_first", "shuffled_tools"):
        raise ValueError(layout)
    rng = random.Random(seed)
    system = "You are a support agent. " + synthetic_text(max(1, system_tokens - 5), random.Random(seed + 1))
    per_tool = max(3, tools_tokens // n_tools)
    tools = [f"Tool tool_{i}: " + synthetic_text(per_tool - 2, random.Random(seed + 100 + i)) for i in range(n_tools)]
    out = []
    for s in range(n_sessions):
        order = list(tools)
        if layout == "shuffled_tools":
            random.Random(seed + 1000 + s).shuffle(order)
        user = f"Customer {s}: " + synthetic_text(max(1, user_tokens - 2), rng)
        results = [f"Result {t}: " + synthetic_text(max(1, tool_result_tokens - 2), rng) for t in range(1, turns)]
        out.append(AgentSession(sid=f"s{s}", system_parts=[system] + order, turns=[user] + results,
                                max_tokens=output_tokens, think_time_s=think_time_s,
                                timestamp_first=(layout == "timestamp_first")))
    return out
