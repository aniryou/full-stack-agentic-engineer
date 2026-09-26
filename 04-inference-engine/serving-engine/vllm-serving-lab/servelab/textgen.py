"""textgen.py — a deterministic stand-in tokenizer and synthetic text.

One idea: a serving benchmark controls *token counts*, not characters. The load generator and
the fake server share one toy tokenizer — every whitespace-prefixed word is one token — so a
prompt built from N words is exactly N tokens, identical text always produces identical token
ids (which is what prefix caching keys on), and no vocabulary download is needed.

Real tokenizers differ: random English words are roughly 1.1-1.5 tokens each for Llama/Qwen
tokenizers. When the target is a real vLLM, trust ``usage.prompt_tokens`` in the response (the
benchmark records it), not the counts here.
"""
from __future__ import annotations

import random
import re
import zlib

VOCAB_SIZE = 100_000
# Chat-template special tokens (a real template also turns role markers into special ids).
ROLE_IDS = {"system": VOCAB_SIZE + 1, "user": VOCAB_SIZE + 2, "assistant": VOCAB_SIZE + 3, "tool": VOCAB_SIZE + 4}
END_OF_MESSAGE = VOCAB_SIZE + 5

_PIECE = re.compile(r"\s*\S+")

WORDS = (
    "the of and to in is was for on that with as by at from it an be this which or are have not "
    "has had but were all their one can there been if more when will would who so no out up into "
    "than them some could time these two may then do first any my now such like our over man me "
    "even most made after also did many before must through back years where much your way well "
    "down should because each just those people how too little state good very make world still "
    "own see men work long get here between both life being under never day same another know "
    "while last might us great old year off come since against go came right used take three "
    "cache block token batch prefill decode kernel queue router replica tensor memory latency "
    "request engine sampler schedule budget chunk prefix hash agent tool session system prompt "
    "stream metric bucket window scale model weight layer head value key vector matrix compute "
    "network storage cluster node pod service gateway policy region zone quota spot price hour"
).split()


def token_id(piece: str) -> int:
    """The id of one text piece (a whitespace-prefixed word). Stable across processes."""
    return zlib.crc32(piece.encode("utf-8")) % VOCAB_SIZE


def pieces(text: str) -> list[str]:
    """Split text into token pieces: each piece is optional whitespace + one run of non-space."""
    return _PIECE.findall(text)


def tokenize(text: str) -> list[int]:
    """Text -> token ids. ``len(tokenize(synthetic_text(n)))`` is exactly ``n``."""
    return [token_id(p) for p in pieces(text)]


def count_tokens(text: str) -> int:
    return len(pieces(text))


def chat_tokens(messages: list[dict]) -> list[int]:
    """A chat template: ``<role> content <end>`` per message, then the generation prompt.

    Messages are tokenized one by one, so a conversation that only *appends* messages keeps an
    identical token prefix — the property prefix caching needs (notebook 04)."""
    ids: list[int] = []
    for m in messages:
        ids.append(ROLE_IDS.get(m.get("role", "user"), ROLE_IDS["user"]))
        ids.extend(tokenize(str(m.get("content") or "")))
        ids.append(END_OF_MESSAGE)
    ids.append(ROLE_IDS["assistant"])
    return ids


def synthetic_text(n_tokens: int, rng: random.Random | int | None = None) -> str:
    """``n_tokens`` random words joined by single spaces (exactly ``n_tokens`` toy tokens)."""
    if n_tokens <= 0:
        return ""
    rng = rng if isinstance(rng, random.Random) else random.Random(rng)
    return " ".join(rng.choice(WORDS) for _ in range(n_tokens))


def generated_piece(seed: int) -> tuple[int, str]:
    """What the fake engine 'samples': a word, emitted with a leading space, and its id.

    Because the emitted text re-tokenizes to the same id, a client that appends the model's
    reply to the conversation reproduces the engine's token sequence exactly — so later turns of
    an agent session can hit the prefix cache on the model's own earlier output."""
    text = " " + WORDS[seed % len(WORDS)]
    return token_id(text), text
