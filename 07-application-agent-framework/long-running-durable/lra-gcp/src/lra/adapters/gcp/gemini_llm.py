"""Gemini (Vertex AI) :class:`LLM` adapter via the ``google-genai`` SDK.

Two things a long-running agent needs from its model client that a demo does
not:

1. **Usage accounting** — every response reports tokens; we convert to cost
   so the run's :class:`Budget` can fail closed.
2. **Bounded in-call retries** for 429/5xx — a step-level retry re-runs the
   *whole* step (possibly several model calls), so the adapter absorbs
   short blips itself and only surfaces persistent failure.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from ...core.models import LLMResponse, LLMUsage

log = logging.getLogger("lra.gcp.gemini")

# ILLUSTRATIVE prices in USD per 1M tokens (input, output). Pull real numbers
# from cloud.google.com/vertex-ai/generative-ai/pricing for the model you pin.
DEFAULT_PRICING_PER_1M: tuple[float, float] = (0.10, 0.40)


class GeminiLLM:
    RETRYABLE = {429, 500, 502, 503, 504}

    def __init__(
        self,
        *,
        project: str,
        location: str = "global",
        model: str = "gemini-3.1-flash-lite",
        pricing_per_1m: tuple[float, float] = DEFAULT_PRICING_PER_1M,
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        from google import genai  # lazy import

        self._genai = genai
        self.client = client or genai.Client(vertexai=True, project=project, location=location)
        self.model = model
        self.pricing = pricing_per_1m
        self.max_retries = max_retries

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json" if json_mode else None,
        )
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                break
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code in self.RETRYABLE and attempt <= self.max_retries:
                    sleep = min(30.0, (2 ** attempt) + random.random())
                    log.warning("gemini %s (attempt %d); retrying in %.1fs", code, attempt, sleep)
                    time.sleep(sleep)
                    continue
                raise

        meta = getattr(resp, "usage_metadata", None)
        in_tok = int(getattr(meta, "prompt_token_count", 0) or 0)
        out_tok = int(getattr(meta, "candidates_token_count", 0) or 0)
        cost = in_tok / 1e6 * self.pricing[0] + out_tok / 1e6 * self.pricing[1]
        return LLMResponse(
            text=resp.text or "",
            usage=LLMUsage(input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost),
            model=self.model,
        )
