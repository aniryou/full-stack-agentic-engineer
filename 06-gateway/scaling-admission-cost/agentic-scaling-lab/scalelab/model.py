"""A fake model with the properties scaling work cares about, on three kinds of backend, plus the real calls.

* ``HostedBackend`` (the default) — a provider's API: a shared per-model pool of tokens per minute
  (``SharedPool``, optionally also requests per second) that raises ``RateLimited`` (a 429) when a sliding
  60-second window is full — this is what turns a load test into a 429 storm. Latency is time-to-first-token
  (lognormal) + output tokens / tokens-per-second, inflated when the pool is busy.
* ``ServerPool`` — your own vLLM replicas (``serving.Replica``): they never 429. They **queue**, and the time
  per output token grows with the batch they are running. Overload shows up as latency, not as errors —
  unless you cap the queue (``--max-num-queued-reqs`` → 503, ``ServerOverloaded``).
* ``HybridBackend`` — the fleet first, the API when the fleet is saturated (the spill-over pattern).

Prefix caching is modelled on the model: a request whose stable prefix was seen recently reports cached
tokens (billed at ~10 % on an API; not re-prefilled on vLLM).

``fake_model(provider)`` builds the model with a provider's traffic shape: ``"gemini"`` (the default, the
Google Cloud anchor scenario) or ``"mistral"`` (the hosted-vs-own-GPUs scenario of ``scalelab.mistral``).
The policy at the bottom decides what the model "says": it issues tool calls for the intent it detects,
then answers. It is not intelligence; it is the shape of a support agent's traffic.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from dataclasses import dataclass, field

from . import capacity
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
    """One model's limit on the API: ``tpm`` tokens per sliding minute and, optionally, ``rps`` requests per second."""

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

def cost_per_call(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """Dollars for one call on the hosted API, from whichever provider's price table lists ``model``."""
    if model in capacity.PRICES:
        return capacity.cost_per_call(model, input_tokens, output_tokens, cached_tokens)
    from . import mistral
    return mistral.cost_per_call(model, input_tokens, output_tokens, cached_tokens)


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
    """``pool``, ``ttft_median_s`` and ``tokens_per_s`` describe the default hosted backend; pass ``backend`` for
    a vLLM fleet (``ServerPool``) or the spill-over (``HybridBackend``)."""
    model: str = "gemini-3.5-flash"
    pool: SharedPool | None = None
    ttft_median_s: float = 0.7
    tokens_per_s: float = 200.0
    prefix_tokens: int = 3_000       # the stable, cacheable part of every prompt
    cache_ttl_s: float = 300.0
    answer_tokens: int = 350
    rng: random.Random = field(default_factory=lambda: random.Random(7))
    _last_prefix_seen: float = -1e9
    calls: int = 0
    backend: HostedBackend | ServerPool | HybridBackend | None = None
    tool_step_tokens: int = 40       # output tokens per proposed tool call
    answer_filler: int = 12          # how long the final answer text is (it becomes context for the next turn)

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = HostedBackend(self.pool, self.ttft_median_s, self.tokens_per_s)
        elif self.pool is None and isinstance(self.backend, HostedBackend):
            self.pool = self.backend.pool

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
        return f"Here is what I found about your {intent} question. " + "I have noted this on your account. " * self.answer_filler, []


# Each provider's traffic shape: the standard model and the cheaper one admission control degrades to.
PROVIDERS = {
    "gemini": {
        "standard": dict(model="gemini-3.5-flash", ttft_median_s=0.7, tokens_per_s=200.0, answer_tokens=350,
                         tool_step_tokens=40, answer_filler=12),
        "lite": dict(model="gemini-3.5-flash-lite", ttft_median_s=0.35, tokens_per_s=350.0, answer_tokens=150,
                     tool_step_tokens=40, answer_filler=12),
    },
    "mistral": {
        "standard": dict(model="mistral-small-2603", ttft_median_s=0.5, tokens_per_s=150.0, answer_tokens=300,
                         tool_step_tokens=60, answer_filler=10),
        "lite": dict(model="ministral-8b-2512", ttft_median_s=0.35, tokens_per_s=250.0, answer_tokens=150,
                     tool_step_tokens=60, answer_filler=10),
    },
}


def fake_model(provider: str = "gemini", tier: str = "standard", **overrides) -> FakeModel:
    """A ``FakeModel`` with ``provider``'s traffic shape (``"gemini"`` or ``"mistral"``); keywords override it."""
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose one of {sorted(PROVIDERS)}")
    return FakeModel(**{**PROVIDERS[provider][tier], **overrides})


# --- the real things, for reference -------------------------------------------------------------

def gemini_generate(project: str, messages: list[dict], system: str, *, model: str = "gemini-3.5-flash",
                    location: str = "global", pt_mode: str | None = None):  # pragma: no cover
    """The same call against Gemini on the Gemini Enterprise Agent Platform (google-genai ≥ 2.20, < 3).

    ``pt_mode``: None = spill-over (Provisioned Throughput first, then pay-as-you-go), "dedicated" = PT only
    (429 when exhausted), "shared" = pay-as-you-go only. Retries stay OFF in the SDK: call_with_retries
    owns backoff, so one place decides how a 429 is handled.
    """
    from google import genai
    from google.genai import types as gt

    client = genai.Client(enterprise=True, project=project, location=location)
    headers = {"X-Vertex-AI-LLM-Request-Type": pt_mode} if pt_mode else None
    contents = [gt.Content(role="user" if m["role"] != "assistant" else "model", parts=[gt.Part(text=m["content"])]) for m in messages]
    resp = client.models.generate_content(
        model=model, contents=contents,
        config=gt.GenerateContentConfig(system_instruction=system, max_output_tokens=700,
                                        thinking_config=gt.ThinkingConfig(thinking_level="LOW"),
                                        http_options=gt.HttpOptions(timeout=30_000, headers=headers)))
    u = resp.usage_metadata
    return resp.text, u.prompt_token_count, u.cached_content_token_count or 0, u.candidates_token_count, str(u.traffic_type)


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
