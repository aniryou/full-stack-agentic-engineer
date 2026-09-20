"""Redaction for logs and tool outputs.

Redact *before* anything reaches a log sink or the model: bearer tokens, JWT-looking strings,
Google API keys, private keys, and a few PII shapes. This is a deterministic first line; on
Google Cloud pair it with Sensitive Data Protection (via Model Armor's SDP filter) for the
long tail.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    (
        "private_key",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    ),
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
]

SENSITIVE_KEYS = {
    "authorization",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "client_secret",
    "cookie",
    "set-cookie",
    "x-api-key",
    "dpop",
}


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact_text(text: str, *, keep_email: bool = False) -> str:
    out = text
    for name, pattern in _PATTERNS:
        if name == "email" and keep_email:
            continue
        if name == "card":

            def _card(m: re.Match[str]) -> str:
                digits = re.sub(r"\D", "", m.group(0))
                return (
                    "[REDACTED:card]"
                    if 13 <= len(digits) <= 19 and _luhn_ok(digits)
                    else m.group(0)
                )

            out = pattern.sub(_card, out)
        else:
            out = pattern.sub(f"[REDACTED:{name}]", out)
    return out


def redact(obj: Any, *, keep_email: bool = False) -> Any:
    """Recursively redact strings and sensitive keys in dicts/lists."""
    if isinstance(obj, str):
        return redact_text(obj, keep_email=keep_email)
    if isinstance(obj, Mapping):
        return {
            k: (
                "[REDACTED]"
                if str(k).lower() in SENSITIVE_KEYS
                else redact(v, keep_email=keep_email)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list | tuple):
        return type(obj)(redact(v, keep_email=keep_email) for v in obj)
    return obj
