"""Pseudo-tokens without a tokenizer: how a router "tokenizes" a request it must route in microseconds.

The one idea: a router does not need the model's real token ids to find shared prefixes — it
needs a *stable* sequence whose prefix structure mirrors the prompt's. The llm-d EPP's default
`token-producer` backend (`estimate`) packs the request bytes into 4-byte pseudo-tokens:

    chat request -> bytes = json(tools) + role + content + role + content + ...
                 -> zero-pad to a multiple of 4 -> one little-endian uint32 per 4 bytes

So 1 pseudo-token = 4 bytes of prompt text (real tokenizers average ~4 characters/token on
English, which is why the estimate is also a usable token *count*). The ids never match the
engine's real token ids — which is fine: the router only compares requests with each other.
"""
from __future__ import annotations

import json
import struct

BYTES_PER_TOKEN = 4

__all__ = ["BYTES_PER_TOKEN", "pack_bytes", "request_bytes", "estimate_tokens"]


def pack_bytes(raw: bytes) -> list[int]:
    """Zero-pad `raw` to a multiple of 4 bytes and read it as little-endian uint32s."""
    if not raw:
        return []
    pad = (-len(raw)) % BYTES_PER_TOKEN
    raw = raw + b"\x00" * pad
    return list(struct.unpack(f"<{len(raw) // BYTES_PER_TOKEN}I", raw))


def _content_text(content) -> str:
    """OpenAI message content is a string or a list of parts; keep the text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "".join(parts)


def request_bytes(body: dict) -> bytes:
    """The byte stream the estimate backend hashes, for chat or completions bodies."""
    if "messages" in body:
        out = b""
        if body.get("tools"):
            out += json.dumps(body["tools"], separators=(",", ":"), sort_keys=True).encode()
        for m in body["messages"]:
            out += str(m.get("role", "")).encode()
            out += _content_text(m.get("content")).encode()
        return out
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):  # a batch of prompts: treat as one stream (lab simplification)
        prompt = "".join(p if isinstance(p, str) else "" for p in prompt)
    return str(prompt).encode()


def estimate_tokens(body: dict) -> list[int]:
    """Pseudo-token ids for an OpenAI chat/completions request body."""
    return pack_bytes(request_bytes(body))
