"""client.py — a small OpenAI-compatible client for a teacher: samples, log-probs, and scoring a given text.

One idea: a served teacher gives a distillation pipeline exactly three things, and each has its own request:

    samples       POST /v1/chat/completions  n, temperature, max_tokens        → SeqKD data (PRIMER §3)
    its top-k     ... with logprobs: true, top_logprobs: k (k ≤ --max-logprobs, 20 by default in vLLM 0.30.0)
                                                                              → an approximate logit-KD target
    scores        POST /v1/completions  prompt = template(question) + text, echo: true, max_tokens: 0,
                  prompt_logprobs: 0                                          → log π_teacher(y_t | y_<t) for every
                                                                                token of a *student's* text: the
                                                                                on-policy reward (PRIMER §4)

A full next-token distribution (151,936 numbers per token for Qwen) is not something an API returns, which is
why logit KD at T1 runs the teacher in-process (``distillab.hf.kd``) and API-served teachers are used for
samples and scores. Reasoning comes back as ``message.reasoning`` (vLLM 0.30.0) or ``reasoning_content``
(SGLang, older vLLM); both are read. Standard library only (``urllib``).
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

QWEN25_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class Completion:
    content: str | None = None
    reasoning: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None
    logprobs: list = field(default_factory=list)        # [(token, logprob)] of the sampled tokens
    top_logprobs: list = field(default_factory=list)    # [[(token, logprob), ...] per position]
    latency: float = math.nan
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def render_chatml(messages: list, *, add_generation_prompt: bool = True, default_system: str | None = QWEN25_SYSTEM) -> str:
    """Qwen's ChatML: ``<|im_start|>role\\ncontent<|im_end|>\\n`` per message, then the assistant header.
    Qwen2.5's template adds its default system prompt when none is given; pass ``default_system=None`` for
    Qwen3, whose template does not (verify against the model's ``tokenizer_config.json``)."""
    msgs = list(messages)
    if default_system and not (msgs and msgs[0].get("role") == "system"):
        msgs = [{"role": "system", "content": default_system}] + msgs
    out = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class Client:
    def __init__(self, url: str, model: str | None = None, headers: dict | None = None, timeout: float = 120.0):
        self.url, self.model, self.headers, self.timeout = url.rstrip("/"), model, headers or {}, timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **self.headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:   # noqa: S310
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"HTTP {e.code} from {path}: {detail}") from None

    def chat(self, messages: list, *, n: int = 1, temperature: float = 1.0, max_tokens: int | None = None,
             top_logprobs: int | None = None, thinking: bool | None = None, seed: int | None = None,
             top_p: float | None = None, extra: dict | None = None) -> list:
        """``n`` samples for one conversation → a list of :class:`Completion` (``usage`` is split evenly)."""
        body = {"model": self.model, "messages": messages, "n": n, "temperature": temperature}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if top_logprobs is not None:
            body.update(logprobs=True, top_logprobs=top_logprobs)
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        if seed is not None:
            body["seed"] = seed
        if top_p is not None:
            body["top_p"] = top_p
        body.update(extra or {})
        t0 = time.perf_counter()
        try:
            r = self._post("/v1/chat/completions", body)
        except (RuntimeError, OSError) as e:
            return [Completion(error=str(e)) for _ in range(n)]
        dt = time.perf_counter() - t0
        u = r.get("usage") or {}
        rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        out = []
        for ch in r["choices"]:
            msg = ch.get("message") or {}
            lp = (ch.get("logprobs") or {}).get("content") or []
            out.append(Completion(
                content=msg.get("content"), reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
                finish_reason=ch.get("finish_reason"), prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=len(lp) if lp else round(u.get("completion_tokens", 0) / max(1, len(r["choices"]))),
                reasoning_tokens=None if rt is None else round(rt / max(1, len(r["choices"]))),
                logprobs=[(e["token"], e["logprob"]) for e in lp],
                top_logprobs=[[(a["token"], a["logprob"]) for a in e.get("top_logprobs") or []] for e in lp],
                latency=dt))
        return out

    def chat_many(self, conversations: list, workers: int = 8, **kw) -> list:
        """``chat`` for many conversations concurrently; returns one list of completions per conversation."""
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda m: self.chat(m, **kw), conversations))

    def score(self, messages: list, completion: str, *, default_system: str | None = QWEN25_SYSTEM) -> list:
        """The teacher's log-probability of each token of ``completion`` as the answer to ``messages``:
        renders the chat template, sends it with the completion appended, ``echo`` and ``max_tokens: 0`` (vLLM
        then samples one token it does not return as prompt) and ``prompt_logprobs: 0``, and keeps the entries
        after the prompt. Returns ``[(token, logprob)]``. The tokens are the *teacher's* — they match a student's
        only when both share a tokenizer (PRIMER §6)."""
        prefix = render_chatml(messages, default_system=default_system)
        r = self._post("/v1/completions", {"model": self.model, "prompt": prefix + completion, "echo": True,
                                           "max_tokens": 0, "prompt_logprobs": 0, "temperature": 0.0})
        entries = r["choices"][0].get("prompt_logprobs") or []
        full = prefix + completion
        decoded = [(info.get("decoded_token") or "", info["logprob"])
                   for e in entries[1:] if e for info in [next(iter(e.values()))]]
        pos = len(full) - sum(len(t) for t, _ in decoded)      # the first token (no entry) covers the rest
        out = []
        for tok, lp in decoded:
            if pos >= len(prefix):                              # a token that straddles the boundary is dropped
                out.append((tok, lp))
            pos += len(tok)
        return out
