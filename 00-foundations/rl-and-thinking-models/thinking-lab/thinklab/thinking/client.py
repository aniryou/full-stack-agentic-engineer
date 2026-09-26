"""client.py — an OpenAI-compatible client that understands thinking models.

One idea: a thinking model's response has *two* streams — reasoning, then content — and the numbers
that matter are about both: when the first token arrives (TTFT: usually a reasoning token), when the
*answer* starts (time to first content token, TTFC — what a user waiting for the answer feels), how
many tokens were reasoning (``usage.completion_tokens_details.reasoning_tokens``) and whether the
answer arrived at all (``finish_reason == "length"`` with ``content: null`` means ``max_tokens`` ran
out inside the thinking).

The request side is three switches (PRIMER §5 "Thinking models"): ``chat_template_kwargs.enable_thinking``
(Qwen3's hard switch, passed through to the chat template), ``thinking_token_budget`` (vLLM's per-request
cap on reasoning tokens; −1 = unlimited; needs ``--reasoning-parser``) and ``reasoning_effort`` (for
Qwen3 only on/off; gpt-oss maps low/medium/high into its system prompt). Sampling defaults follow the
Qwen3 model card: thinking T=0.6, top-p 0.95, top-k 20; non-thinking T=0.7, top-p 0.8, top-k 20 —
never greedy for a thinking model (endless repetition). ``top_k``/``min_p`` are vLLM extensions.

Standard library only (``urllib``), so it runs anywhere; the load generator in
:mod:`thinklab.workload` is the async counterpart. The response field is read as ``reasoning``
(vLLM 0.30) *or* ``reasoning_content`` (SGLang, DeepSeek API, older vLLM).
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..parsers import REASONING_FIELDS, extract_answer, reasoning_of

SAMPLING = {   # Qwen3 model card (qwen3/docs/source/getting_started/quickstart.md), min_p 0 in both
    True: {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}


@dataclass
class Completion:
    reasoning: str | None = None
    content: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None       # from usage; None when the server has no reasoning parser
    cached_tokens: int | None = None
    ttft: float = math.nan                    # first reasoning-or-content chunk (streaming only)
    ttfc: float = math.nan                    # first *content* chunk: when the answer starts
    latency: float = math.nan
    itl: list = field(default_factory=list)
    error: str = ""
    raw: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def answer(self) -> str | None:
        return extract_answer(self.content)

    @property
    def answer_tokens(self) -> int | None:
        return None if self.reasoning_tokens is None else self.completion_tokens - self.reasoning_tokens


def request_body(messages: list, *, model: str, thinking: bool | None = None, budget: int | None = None,
                 max_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None,
                 top_k: int | None = None, n: int = 1, seed: int | None = None, reasoning_effort: str | None = None,
                 include_reasoning: bool | None = None, stream: bool = False, recommended: bool = True,
                 extra: dict | None = None) -> dict:
    """The JSON body for ``POST /v1/chat/completions`` on vLLM v0.30.0 with a reasoning parser.

    ``thinking=None`` leaves the model's default (Qwen3: on). ``budget`` → ``thinking_token_budget``.
    ``recommended`` fills unset sampling knobs with the model card's values for the mode."""
    body = {"model": model, "messages": messages}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if budget is not None:
        body["thinking_token_budget"] = int(budget)
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if include_reasoning is not None:
        body["include_reasoning"] = bool(include_reasoning)
    defaults = SAMPLING[thinking is not False] if recommended else {}
    for k, v in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)):
        if v is not None:
            body[k] = v
        elif k in defaults:
            body[k] = defaults[k]
    if n != 1:
        body["n"] = n
    if seed is not None:
        body["seed"] = seed
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    body.update(extra or {})
    return body


def parse_response(data: dict) -> list:
    """Non-streaming ``chat.completion`` → one :class:`Completion` per choice (usage split evenly is
    not attempted: per-choice token counts are only exact for n = 1)."""
    u = data.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    out = []
    for ch in data.get("choices", []):
        msg = ch.get("message") or {}
        out.append(Completion(reasoning_of(msg), msg.get("content"), ch.get("finish_reason"),
                              u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                              details.get("reasoning_tokens"), cached, raw=data))
    return out


class ThinkingClient:
    """Blocking client for one OpenAI-compatible server (vLLM, SGLang, the fake server)."""

    def __init__(self, url: str, model: str | None = None, headers: dict | None = None, timeout: float = 900):
        self.url, self.headers, self.timeout = url.rstrip("/"), headers or {}, timeout
        self.model = model or self.models()[0]

    def _open(self, path: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json", **self.headers})
        return urllib.request.urlopen(req, timeout=self.timeout)   # noqa: S310

    def models(self) -> list:
        with self._open("/v1/models") as r:
            return [m["id"] for m in json.loads(r.read())["data"]]

    def chat(self, messages: list, *, stream: bool = False, **kw) -> Completion | list:
        """One request. Non-streaming returns a list when ``n > 1``; streaming measures TTFT/TTFC/ITL."""
        body = request_body(messages, model=self.model, stream=stream, **kw)
        t0 = time.perf_counter()
        try:
            with self._open("/v1/chat/completions", body) as r:
                if not stream:
                    comps = parse_response(json.loads(r.read()))
                    for c in comps:
                        c.latency = time.perf_counter() - t0
                    return comps if kw.get("n", 1) != 1 else comps[0]
                return self._read_stream(r, t0)
        except urllib.error.HTTPError as e:
            return Completion(error=f"HTTP {e.code}: {e.read().decode()[:300]}")
        except (urllib.error.URLError, OSError) as e:
            return Completion(error=f"{type(e).__name__}: {e}")

    def _read_stream(self, r, t0: float) -> Completion:
        c = Completion()
        reasoning, content, last = [], [], None
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if data.get("usage"):
                u = data["usage"]
                c.prompt_tokens, c.completion_tokens = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
                c.reasoning_tokens = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                c.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            for ch in data.get("choices") or []:
                delta = ch.get("delta") or {}
                r_piece = next((delta[f] for f in REASONING_FIELDS if delta.get(f)), None)
                c_piece = delta.get("content") or None
                now = time.perf_counter()
                if r_piece or c_piece:
                    if math.isnan(c.ttft):
                        c.ttft = now - t0
                    elif last is not None:
                        c.itl.append(now - last)
                    last = now
                if r_piece:
                    reasoning.append(r_piece)
                if c_piece:
                    if math.isnan(c.ttfc):
                        c.ttfc = now - t0
                    content.append(c_piece)
                if ch.get("finish_reason"):
                    c.finish_reason = ch["finish_reason"]
        c.latency = time.perf_counter() - t0
        c.reasoning = "".join(reasoning).strip() or None
        c.content = "".join(content) or None
        return c

    def chat_many(self, conversations: list, concurrency: int = 8, **kw) -> list:
        """Send many conversations with up to ``concurrency`` in flight (threads); order preserved."""
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            return list(ex.map(lambda m: self.chat(m, **kw), conversations))
