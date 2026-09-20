"""A fake model with the three properties scaling work cares about, plus the real call.

* latency  = time-to-first-token (lognormal) + output tokens / tokens-per-second, inflated when
             the shared pool is busy;
* a shared tokens-per-minute pool (``SharedPool``) that raises ``RateLimited`` when a sliding
  60-second window is full — this is what turns a load test into a 429 storm;
* prefix caching: a request whose stable prefix was seen recently reports cached tokens.

The policy at the bottom decides what the model "says": it issues tool calls for the intent it
detects, then answers. It is not intelligence; it is the shape of a support agent's traffic.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

from .capacity import cost_per_call
from .clock import CLOCK
from .resilience import RateLimited

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


class SharedPool:
    """The provider's pool for one model family: ``tpm`` tokens per sliding minute."""

    def __init__(self, tpm: int, soft_ratio: float = 0.75, max_inflation: float = 2.0):
        self.tpm, self.soft_ratio, self.max_inflation = tpm, soft_ratio, max_inflation
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
        if self.used + tokens > self.tpm:
            self.rejected += 1
            wait = self.log[0][0] + 60 - CLOCK.now() if self.log else 1.0
            raise RateLimited(retry_after=max(0.5, min(wait, 10.0)))
        self.log.append((CLOCK.now(), tokens))
        self.used += tokens
        u = self.used / self.tpm
        return 1.0 if u <= self.soft_ratio else 1 + (self.max_inflation - 1) * (u - self.soft_ratio) / (1 - self.soft_ratio)


@dataclass
class Response:
    text: str
    tool_calls: list[str]
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


@dataclass
class FakeModel:
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

    async def generate(self, messages: list[dict], tools: list[str]) -> Response:
        """messages: [{"role": "user"|"assistant"|"tool", "content": str, ...}] — decide, then 'take the time'."""
        self.calls += 1
        input_tokens = self.prefix_tokens + sum(len(m["content"]) // 4 + 8 for m in messages)
        cached = self.prefix_tokens if CLOCK.now() - self._last_prefix_seen < self.cache_ttl_s else 0
        self._last_prefix_seen = CLOCK.now()
        text, calls = self._decide(messages, tools)
        output_tokens = self.answer_tokens if text else 40 * len(calls)
        # Cached tokens are billed at 10 % but we assume they count fully against the pool's TPM (conservative).
        inflation = self.pool.admit(input_tokens + output_tokens) if self.pool else 1.0
        latency = self.ttft_median_s * math.exp(self.rng.gauss(0, 0.35)) * inflation + output_tokens / (self.tokens_per_s / inflation)
        await CLOCK.sleep(latency)
        return Response(text, calls, input_tokens, cached, output_tokens, latency,
                        cost_per_call(self.model, input_tokens, output_tokens, cached))

    def _decide(self, messages, tools) -> tuple[str, list[str]]:
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        intent = detect_intent(user)
        done = {m["name"] for m in messages if m["role"] == "tool"}
        for step in INTENT_TOOLS.get(intent, INTENT_TOOLS["generic"]):
            pending = [t for t in step if t in tools and t not in done]
            if pending:
                return "", pending
        return f"Here is what I found about your {intent} question. " + "I have noted this on your account. " * 12, []


# --- the real thing, for reference ----------------------------------------------------------

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
