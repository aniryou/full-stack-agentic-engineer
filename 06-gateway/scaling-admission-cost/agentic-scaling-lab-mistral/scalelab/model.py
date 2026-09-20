"""A fake model with the properties scaling work cares about, on two kinds of backend, plus the real calls.

* ``HostedBackend`` — Mistral's API: a shared per-model pool of tokens per minute (and requests per
  second) that answers **429** when full; latency is whatever the provider gives you.
* ``ServerPool`` — your own vLLM replicas: they never 429. They **queue**, and the time per output
  token grows with the batch they are running (``serving.Replica``). Overload shows up as latency,
  not as errors — unless you cap the queue (``--max-num-queued-reqs`` → 503).
* ``HybridBackend`` — the fleet first, the API when the fleet is saturated (the spill-over pattern).

Prefix caching is modelled on the model: a request whose stable prefix was seen recently reports
cached tokens (billed at 10 % on the API; not re-prefilled on vLLM).

The policy at the bottom decides what the model "says": it issues tool calls for the intent it
detects, then answers. It is not intelligence; it is the shape of a support agent's traffic.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from dataclasses import dataclass, field

from .capacity import cost_per_call
from .clock import CLOCK
from .resilience import RateLimited
from .serving import Replica

INTENT_TOOLS = {  # intent -> ordered tool steps (each step's tools run in parallel)
    "billing": [["get_customer"], ["get_invoice"]],
    "outage": [["get_customer", "check_network"]],
    "plan": [["get_customer"], ["list_plans"]],
    "ticket": [["get_customer"], ["create_ticket"]],
    "generic": [["search_kb"]],
}
KEYWORDS = {"billing": ("bill", "charge", "invoice", "refund"), "outage": ("signal", "outage", "down", "slow"),
            "plan": ("plan", "upgrade", "data"), "ticket": ("complain", "human", "escalat")}


def detect_intent(text: str) -> str:
    t = text.lower()
    return next((k for k, words in KEYWORDS.items() if any(w in t for w in words)), "generic")


class ServerOverloaded(RateLimited):
    """vLLM's 503 when the queue cap is exceeded — handled like a 429 by the gateway."""


# --- hosted: the API's rate limit --------------------------------------------------------------

class SharedPool:
    """One model's limit on the API: ``tpm`` tokens per sliding minute and ``rps`` requests per second."""

    def __init__(self, tpm: int, rps: int | None = None, soft_ratio: float = 0.75, max_inflation: float = 2.0):
        self.tpm, self.rps, self.soft_ratio, self.max_inflation = tpm, rps, soft_ratio, max_inflation
        self.log: deque[tuple[float, int]] = deque()
        self.used = 0
        self.rejected = 0

    def _prune(self) -> None:
        cutoff = CLOCK.now() - 60
        while self.log and self.log[0][0] < cutoff:
            self.used -= self.log.popleft()[1]

    @property
    def utilisation(self) -> float:
        self._prune()
        return self.used / self.tpm

    def admit(self, tokens: int) -> float:
        """Reserve tokens or raise RateLimited; returns a latency inflation factor for busy pools."""
        self._prune()
        now = CLOCK.now()
        if self.used + tokens > self.tpm:
            self.rejected += 1
            wait = self.log[0][0] + 60 - now if self.log else 1.0
            raise RateLimited(retry_after=max(0.5, min(wait, 10.0)))
        if self.rps is not None and sum(1 for t, _ in self.log if t > now - 1.0) >= self.rps:
            self.rejected += 1
            raise RateLimited(retry_after=1.0)
        self.log.append((now, tokens))
        self.used += tokens
        u = self.used / self.tpm
        return 1.0 if u <= self.soft_ratio else 1 + (self.max_inflation - 1) * (u - self.soft_ratio) / (1 - self.soft_ratio)


@dataclass
class HostedBackend:
    """Latency = time-to-first-token (lognormal) + output ÷ tokens-per-second, inflated when the pool is busy."""
    pool: SharedPool | None = None
    ttft_median_s: float = 0.5
    tokens_per_s: float = 150.0
    name: str = "hosted"

    @property
    def saturation(self) -> float:
        return self.pool.utilisation if self.pool else 0.0

    async def serve(self, input_tokens: int, cached_tokens: int, output_tokens: int, rng: random.Random) -> tuple[float, str]:
        # Cached tokens are billed at 10 % but we assume they count fully against the limit (conservative).
        inflation = self.pool.admit(input_tokens + output_tokens) if self.pool else 1.0
        latency = self.ttft_median_s * math.exp(rng.gauss(0, 0.35)) * inflation + output_tokens / (self.tokens_per_s / inflation)
        await CLOCK.sleep(latency)
        return latency, self.name


# --- self-hosted: replicas that queue ------------------------------------------------------------

class ServerPool:
    """``replicas`` copies of one vLLM ``Replica`` behind a round-robin router.

    Every request waits (FIFO) for a slot, then prefills, then decodes in chunks — each chunk at the TPOT
    of the batch running *at that moment*, so more traffic slows everyone down. ``max_queued`` caps the
    waiting queue like ``--max-num-queued-reqs`` (None = unlimited, vLLM's default).
    """

    def __init__(self, replica: Replica, replicas: int, context_tokens: int, shared_prefix_tokens: int = 0,
                 max_queued: int | None = None, target_tpot_s: float = 0.02, name: str = "self-hosted"):
        self.replica, self.replicas, self.context = replica, replicas, context_tokens
        self.max_seqs = replica.max_seqs(context_tokens, shared_prefix_tokens)      # the hard limit: KV memory / max_num_seqs
        self.capacity = replicas * self.max_seqs
        self.target_batch = max(1, min(self.max_seqs, replica.batch_for_tpot(target_tpot_s, context_tokens)))  # the soft one: latency
        self.max_queued, self.name = max_queued, name
        self.running = self.waiting = self.rejected = self.served = 0
        self._waiters: deque[asyncio.Future] = deque()

    @property
    def kv_usage(self) -> float:
        """Share of the hard capacity in use — what ``vllm:kv_cache_usage_perc`` would show."""
        return self.running / self.capacity

    @property
    def saturation(self) -> float:
        """Load against the latency target: 1.0 = every replica at the batch where TPOT meets the target;
        above 1.0 latency is being traded away; a queue only forms at ``capacity``."""
        return (self.running + self.waiting) / (self.replicas * self.target_batch)

    def batch_per_replica(self) -> int:
        return max(1, math.ceil(self.running / self.replicas))

    async def _acquire(self) -> None:
        """Take a slot, or wait FIFO until a finishing request hands one over."""
        if self.running < self.capacity and not self._waiters:
            self.running += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        self.waiting += 1
        try:
            await fut                                   # the slot is ours when this resolves
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self._release()                         # handed over just as we were cancelled: pass it on
            elif fut in self._waiters:
                self._waiters.remove(fut)
            raise
        finally:
            self.waiting -= 1

    def _release(self) -> None:
        self.running -= 1
        while self._waiters:
            fut = self._waiters.popleft()
            if not fut.done():
                self.running += 1                       # the slot goes straight to the next waiter
                fut.set_result(None)
                return

    async def serve(self, input_tokens: int, cached_tokens: int, output_tokens: int, rng: random.Random) -> tuple[float, str]:
        if self.max_queued is not None and self.waiting >= self.max_queued:
            self.rejected += 1
            raise ServerOverloaded(retry_after=1.0)
        t0 = CLOCK.now()
        await self._acquire()
        try:
            await CLOCK.sleep(self.replica.ttft(input_tokens - cached_tokens))
            remaining = output_tokens
            while remaining > 0:                        # decode in ~1-second chunks at the TPOT of the batch running now
                tpot = self.replica.tpot(self.batch_per_replica(), self.context)
                chunk = min(remaining, max(8, math.ceil(1.0 / tpot)))
                await CLOCK.sleep(chunk * tpot)
                remaining -= chunk
        finally:
            self.served += 1
            self._release()
        return CLOCK.now() - t0, self.name


@dataclass
class HybridBackend:
    """The fleet first; the API once the fleet is saturated. Spill-over is the self-hosted equivalent of buying peak."""
    local: ServerPool
    hosted: HostedBackend
    spill_at: float = 0.9

    @property
    def saturation(self) -> float:
        return self.local.saturation

    async def serve(self, input_tokens: int, cached_tokens: int, output_tokens: int, rng: random.Random) -> tuple[float, str]:
        if self.local.saturation >= self.spill_at:
            return await self.hosted.serve(input_tokens, cached_tokens, output_tokens, rng)
        return await self.local.serve(input_tokens, cached_tokens, output_tokens, rng)


# --- the model ----------------------------------------------------------------------------------

@dataclass
class Response:
    text: str
    tool_calls: list[str]
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    served_by: str = "hosted"


@dataclass
class FakeModel:
    model: str = "mistral-small-2603"
    backend: HostedBackend | ServerPool | HybridBackend = field(default_factory=HostedBackend)
    prefix_tokens: int = 3_000       # the stable, cacheable part of every prompt
    cache_ttl_s: float = 300.0
    answer_tokens: int = 300
    tool_step_tokens: int = 60
    rng: random.Random = field(default_factory=lambda: random.Random(7))
    _last_prefix_seen: float = -1e9
    calls: int = 0

    async def generate(self, messages: list[dict], tools: list[str]) -> Response:
        """messages: [{"role": "user"|"assistant"|"tool", "content": str, ...}] — decide, then 'take the time'."""
        self.calls += 1
        input_tokens = self.prefix_tokens + sum(len(m["content"]) // 4 + 8 for m in messages)
        cached = self.prefix_tokens if CLOCK.now() - self._last_prefix_seen < self.cache_ttl_s else 0
        self._last_prefix_seen = CLOCK.now()
        text, calls = self._decide(messages, tools)
        output_tokens = self.answer_tokens if text else self.tool_step_tokens * len(calls)
        latency, where = await self.backend.serve(input_tokens, cached, output_tokens, self.rng)
        cost = cost_per_call(self.model, input_tokens, output_tokens, cached) if where == "hosted" else 0.0   # GPUs are paid for already
        return Response(text, calls, input_tokens, cached, output_tokens, latency, cost, where)

    def _decide(self, messages, tools) -> tuple[str, list[str]]:
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        intent = detect_intent(user)
        done = {m["name"] for m in messages if m["role"] == "tool"}
        for step in INTENT_TOOLS.get(intent, INTENT_TOOLS["generic"]):
            pending = [t for t in step if t in tools and t not in done]
            if pending:
                return "", pending
        return f"Here is what I found about your {intent} question. " + "I have noted this on your account. " * 10, []


# --- the real things, for reference ---------------------------------------------------------------

def mistral_generate(messages: list[dict], system: str, *, model: str = "mistral-small-2603", tools: list[dict] | None = None,
                     prompt_cache_key: str = "telco-support-v1", server: str = "global", priority: bool = False):  # pragma: no cover
    """The same call against Mistral's API (``mistralai`` ≥ 2.10).

    Retries stay OFF in the SDK (its default): ``call_with_retries`` owns backoff, so one place decides how a 429
    is handled. ``prompt_cache_key`` makes the stable prefix cacheable (billed at 10 %); ``service_tier="auto"``
    uses the Priority Tier when the org is entitled to it and falls back to standard.
    """
    import os

    from mistralai.client import Mistral, errors

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], server=server, timeout_ms=30_000)   # "eu" / "us": +10 %, no agents/batch
    try:
        res = client.chat.complete(
            model=model, messages=[{"role": "system", "content": system}, *messages],
            tools=tools, tool_choice="auto" if tools else None, max_tokens=700, temperature=0.2,
            prompt_cache_key=prompt_cache_key, service_tier="auto" if priority else "standard_only")
    except errors.MistralError as e:                                   # 429 arrives here as SDKError
        if e.status_code == 429:
            raise RateLimited(retry_after=float((e.headers or {}).get("retry-after", 1.0))) from e
        raise
    msg, u = res.choices[0].message, res.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return msg.content, [t.function.name for t in (msg.tool_calls or [])], u.prompt_tokens, cached, u.completion_tokens, res.service_tier


def vllm_generate(base_url: str, messages: list[dict], system: str, *, model: str = "mistralai/Ministral-3-14B-Instruct-2512",
                  tools: list[dict] | None = None, timeout_s: float = 30.0):  # pragma: no cover
    """The same call against your own vLLM replica through its OpenAI-compatible endpoint.

    vLLM does not answer 429; with ``--max-num-queued-reqs`` it answers 503 when the queue is full, which the
    gateway treats like a 429. Without the cap it queues — and the only signal is ``vllm:num_requests_waiting``.
    """
    import httpx

    body = {"model": model, "messages": [{"role": "system", "content": system}, *messages], "tools": tools,
            "max_tokens": 700, "temperature": 0.2}
    r = httpx.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout_s)
    if r.status_code == 503:
        raise ServerOverloaded(retry_after=1.0)
    r.raise_for_status()
    data = r.json()
    msg, u = data["choices"][0]["message"], data["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return msg.get("content"), [t["function"]["name"] for t in msg.get("tool_calls") or []], u["prompt_tokens"], cached, u["completion_tokens"]
