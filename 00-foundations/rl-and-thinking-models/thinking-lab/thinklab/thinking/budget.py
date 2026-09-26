"""budget.py — three ways to bound a model's thinking, and what each actually does.

One idea: ``max_tokens`` is **not** a thinking budget. It counts reasoning *and* answer tokens, so a
request that would think for 900 tokens under ``max_tokens=512`` ends inside the thinking with
``content: null`` and ``finish_reason: "length"`` — you paid for 512 tokens and got no answer. Real
budgets force the thinking to *end* and the answer to start:

    strategy      how                                                              where
    truncate      max_tokens only (the trap)                                      any server
    native        thinking_token_budget=B: at B reasoning tokens the sampler      vLLM ≥ the release with
                  forces reasoning_end_str (e.g. "…directly now.</think>")         ReasoningConfig (v0.30.0 has it)
    two_call      Qwen's recipe: call 1 with max_tokens=B; if no answer, append    any server (two round trips,
                  "Considering the limited time … now." + </think> and continue     re-prefills the thinking)

Qwen3's report says the ability to answer from partial thinking "is not explicitly trained but
emerges naturally" from its thinking-mode fusion stage — accuracy vs budget is a curve to measure,
not a given (``sweep``). The second call here uses vLLM's ``continue_final_message`` on the chat API
instead of re-rendering the template client-side and calling ``/v1/completions`` as Qwen's example
does; the tokens sent are the same (verify on your server version).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..fakemodel import EARLY_STOP
from .client import Completion, ThinkingClient
from .evalset import verify


def classify(c: Completion) -> str:
    """answered | cut_in_thinking (max_tokens ran out while reasoning) | cut_in_answer | no_answer | error."""
    if not c.ok:
        return "error"
    if c.finish_reason == "length":
        return "cut_in_answer" if c.content else "cut_in_thinking"
    return "answered" if c.answer is not None else "no_answer"


def truncate(client: ThinkingClient, messages: list, max_tokens: int, **kw) -> Completion:
    return client.chat(messages, max_tokens=max_tokens, **kw)


def native(client: ThinkingClient, messages: list, budget: int, max_tokens: int | None = None, **kw) -> Completion:
    return client.chat(messages, budget=budget, max_tokens=max_tokens, **kw)


def two_call(client: ThinkingClient, messages: list, budget: int, max_tokens: int, **kw) -> Completion:
    """Qwen3's thinking-budget recipe (``docs/source/getting_started/thinking_budget.md``) on a chat API."""
    if max_tokens <= budget:
        raise ValueError("max_tokens must exceed the thinking budget")
    first = client.chat(messages, max_tokens=budget, **kw)
    if not first.ok or first.content is not None:
        return first                                   # finished thinking within the budget
    reasoning = (first.reasoning or "").strip() + "\n\n" + EARLY_STOP
    spent = first.reasoning_tokens if first.reasoning_tokens is not None else first.completion_tokens
    cont = messages + [{"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n"}]
    kw = {k: v for k, v in kw.items() if k != "extra"}
    second = client.chat(cont, max_tokens=max(1, max_tokens - spent),
                         extra={"continue_final_message": True, "add_generation_prompt": False}, **kw)
    return Completion(reasoning, second.content, second.finish_reason, first.prompt_tokens,
                      first.completion_tokens + second.completion_tokens, spent, first.cached_tokens,
                      latency=first.latency + second.latency, error=second.error)


@dataclass
class SweepRow:
    strategy: str
    budget: int | None
    n: int
    accuracy: float
    mean_reasoning_tokens: float
    mean_output_tokens: float
    cut_in_thinking: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def summarize(problems: list, comps: list, strategy: str, budget) -> SweepRow:
    ok = [verify(p, c.content) for p, c in zip(problems, comps)]
    rt = [c.reasoning_tokens or 0 for c in comps]
    return SweepRow(strategy, budget, len(comps), sum(ok) / max(1, len(ok)), sum(rt) / max(1, len(rt)),
                    sum(c.completion_tokens for c in comps) / max(1, len(comps)),
                    sum(classify(c) == "cut_in_thinking" for c in comps) / max(1, len(comps)))


def sweep(client: ThinkingClient, problems: list, budgets: list, strategy: str = "native", max_tokens: int = 4096,
          concurrency: int = 8, seed: int = 0) -> list:
    """Accuracy and tokens per budget. ``budget=None`` means unlimited thinking (the baseline row)."""
    rows = []
    for b in budgets:
        def one(p, b=b):
            m = p.messages()
            if b is None:
                return client.chat(m, max_tokens=max_tokens, seed=seed)
            if strategy == "native":
                return native(client, m, b, max_tokens, seed=seed)
            if strategy == "two_call":
                return two_call(client, m, b, max_tokens, seed=seed)
            return truncate(client, m, b, seed=seed)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(concurrency) as ex:
            comps = list(ex.map(one, problems))
        rows.append(summarize(problems, comps, strategy, b))
    return rows


def expected_cost_tokens(rows: list) -> dict:
    """Output tokens per *correct* answer for each sweep row (∞ when nothing was right)."""
    return {r.budget: (r.mean_output_tokens / r.accuracy if r.accuracy else math.inf) for r in rows}
