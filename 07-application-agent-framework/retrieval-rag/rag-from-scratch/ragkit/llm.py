"""Generation, kept minimal and optional.

The *retrieval* half of RAG is where the lessons are, and it runs fully
offline. Generation needs a model, so this wrapper:

  * uses a real API when a key is in the environment
        ANTHROPIC_API_KEY  -> Anthropic Messages API
        OPENAI_API_KEY     -> OpenAI Chat Completions
  * otherwise falls back to a deterministic *extractive* answer: it returns
    the sentence(s) in the context most similar to the query, with a citation.

The fallback is not clever — that is the point. It lets every notebook run end
to end with no key, while making the grounding/citation mechanics visible.
"""

from __future__ import annotations

import os
import re
from textwrap import shorten

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.replace("\n", " ")) if s.strip()]


def extractive_answer(query: str, contexts: list[str], embedder=None) -> str:
    """Offline fallback: stitch the best-matching context sentences together.

    If an Embedder is passed, rank sentences by cosine similarity to the query;
    otherwise fall back to simple token overlap. Adds a [source] marker so the
    citation flow is identical to the real path.
    """
    sents = []
    for i, ctx in enumerate(contexts):
        for s in _split_sentences(ctx):
            sents.append((i, s))
    if not sents:
        return "I couldn't find anything relevant in the provided context."

    if embedder is not None:
        import numpy as np
        qv = embedder.encode(query)
        sv = embedder.encode([s for _, s in sents])
        scores = sv @ qv
    else:
        q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        scores = [
            len(q_tokens & set(re.findall(r"[a-z0-9]+", s.lower())))
            for _, s in sents
        ]

    order = sorted(range(len(sents)), key=lambda j: float(scores[j]), reverse=True)
    picked = order[:2]
    picked.sort()  # keep original reading order
    answer = " ".join(sents[j][1] for j in picked)
    src = sents[picked[0]][0]
    return f"{answer} [source {src + 1}]"


def complete(prompt: str, system: str = "", *, model: str | None = None,
             max_tokens: int = 512, temperature: float = 0.0,
             contexts: list[str] | None = None, query: str = "",
             embedder=None) -> str:
    """Return a completion. Tries a real API, else extractive fallback.

    Pass `contexts` and `query` so the fallback has something to extract from;
    the real-API path uses `prompt`/`system` and ignores them.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return _anthropic(prompt, system, model, max_tokens, temperature)
    if os.getenv("OPENAI_API_KEY"):
        return _openai(prompt, system, model, max_tokens, temperature)
    return extractive_answer(query or prompt, contexts or [], embedder=embedder)


def available() -> str:
    """Which generation backend is active — handy to print in a notebook."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "extractive-fallback (offline)"


def _anthropic(prompt, system, model, max_tokens, temperature):  # pragma: no cover
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or "claude-3-5-haiku-latest",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "You are a helpful, grounded assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def _openai(prompt, system, model, max_tokens, temperature):  # pragma: no cover
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or "gpt-4o-mini",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system or "You are a helpful, grounded assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def preview(text: str, width: int = 100) -> str:
    return shorten(text.replace("\n", " "), width=width, placeholder=" …")
