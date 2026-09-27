"""Token counts at the gateway: an estimate for admission, the provider's `usage` for billing.

The one idea: when a request arrives the gateway does not know how many tokens it will cost. It can
*estimate* the prompt (here: ceil(characters / 4), labelled as an estimate everywhere it appears) and read the
caller's `max_completion_tokens`, but the output length is unknown until the stream ends and is heavy-tailed.
The provider's `usage` object is the authoritative count; the estimate only covers admission and streams that
were cut before the usage chunk arrived. (PRIMER §4 Streaming-aware rate limits, §5 Metering.)
"""
from __future__ import annotations

import json
import math

ESTIMATE_LABEL = "estimate: ceil(chars / 4), not a tokenizer"
PER_MESSAGE_OVERHEAD = 4          # role markers and separators a chat template adds (an estimate)


def estimate_text(text: str | None) -> int:
    """ceil(len(text) / 4), and 0 for empty text. An estimate, never a bill."""
    return 0 if not text else max(1, math.ceil(len(text) / 4))


def message_text(msg: dict) -> str:
    """The text a chat message contributes to the prompt: string content, text parts, tool-call arguments."""
    out = []
    content = msg.get("content")
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        out.append(str(fn.get("name", "")) + str(fn.get("arguments", "")))
    return " ".join(out)


def estimate_prompt(body: dict) -> int:
    """Estimated prompt tokens of a chat-completions body: messages (plus a per-message overhead) and tools."""
    n = 0
    for m in body.get("messages") or []:
        if isinstance(m, dict):
            n += estimate_text(message_text(m)) + PER_MESSAGE_OVERHEAD
    if body.get("tools"):
        n += estimate_text(json.dumps(body["tools"], separators=(",", ":")))
    return n


def requested_output(body: dict) -> int | None:
    """The caller's output cap: `max_completion_tokens`, else the deprecated `max_tokens`, else None."""
    for k in ("max_completion_tokens", "max_tokens"):
        v = body.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return None


def last_user_text(body: dict) -> str:
    """The last user message's text (what a semantic cache embeds)."""
    for m in reversed(body.get("messages") or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return message_text(m)
    return ""


def system_text(body: dict) -> str:
    return " ".join(message_text(m) for m in body.get("messages") or []
                    if isinstance(m, dict) and m.get("role") in ("system", "developer"))
