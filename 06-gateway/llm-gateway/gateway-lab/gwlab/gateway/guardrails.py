"""Guardrails at the gateway: where a check sits decides what it costs and what it can stop.

The one idea (PRIMER §7): a screen on the input can only add latency *before* the first token; a screen on a
streamed output must choose between latency and leakage:

| output placement | the client sees | added TTFT | leaks before a block |
|---|---|---|---|
| `full` | nothing until the whole answer is checked | ≈ (n − 1)·ITL + check | nothing |
| `window` (W tokens) | each window after its check | ≈ (W − 1)·ITL + check | nothing |
| `parallel` | tokens at once; the stream is cut when a check flags | 0 | up to ~a window + a check of tokens |
| `shadow` | tokens at once; findings are only logged | 0 | everything (measurement only) |

(NeMo Guardrails' streaming output rails default to `stream_first: True`, i.e. `parallel`; a held-back window
needs it off.) Input placements: `inline` (+check to TTFT), `parallel` with the model call started at once and
the first byte held until the check passes (+max(0, check − TTFT)), `shadow`. `added_latency()` computes all of
them for a given check time, TTFT, ITL and output length — the check the notebook runs against the gateway.

Guardrails reduce risk; policy bounds it (the identity primer §0 idea 4, §4.1, §6): a regex or a classifier
lowers how often a bad output gets through, an authorization check decides what a tool call may do at all.
`RegexScreener` is a **stand-in** (labelled) with a simulated check time; the models you would put here —
Llama Prompt Guard 2 (22M/86M, a 512-token window, input side), Llama Guard 3/4, Model Armor — are named in
the primer, not run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SCREENER_LABEL = "regex stand-in, not a classifier; check time simulated"

INPUT_RULES = {
    "prompt_injection": re.compile(r"ignore (all |any )?(previous|prior|above) instructions|reveal (the |your )?system prompt", re.I),
    "exfiltration": re.compile(r"send (the |all )?(api )?keys? to|curl .*\|\s*sh", re.I),
}
OUTPUT_RULES = {
    "secret": re.compile(r"\bsk-[A-Za-z0-9]{16,}\b|-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "card_number": re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b"),
}


@dataclass
class Finding:
    rule: str
    match: str


class RegexScreener:
    """Named regex rules; `check_s` is the time a real classifier call would take (simulated by the caller)."""

    label = SCREENER_LABEL

    def __init__(self, check_s: float = 0.02, input_rules=None, output_rules=None):
        self.check_s = check_s
        self.input_rules = INPUT_RULES if input_rules is None else input_rules
        self.output_rules = OUTPUT_RULES if output_rules is None else output_rules
        self.checks = 0

    def _run(self, rules: dict, text: str) -> list[Finding]:
        self.checks += 1
        return [Finding(name, m.group(0)) for name, rx in rules.items() for m in [rx.search(text)] if m]

    def check_input(self, text: str) -> list[Finding]:
        return self._run(self.input_rules, text)

    def check_output(self, text: str) -> list[Finding]:
        return self._run(self.output_rules, text)


def window_release_times(ttft_s: float, itl_s: float, n_tokens: int, window: int, check_s: float) -> list[float]:
    """When each window of W tokens reaches the client, if windows are checked one after another as they fill.

    Token i (1-based) is generated at ttft + (i − 1)·itl; window k is complete when its last token is; its check
    starts when the window is complete *and* the previous check has finished, and takes check_s.
    """
    out, prev_done = [], 0.0
    for end in range(window, n_tokens + window, window):
        last = min(end, n_tokens)
        ready = ttft_s + (last - 1) * itl_s
        prev_done = max(ready, prev_done) + check_s
        out.append(prev_done)
        if last == n_tokens:
            break
    return out


def added_latency(input_placement: str = "off", output_placement: str = "off", *, check_s: float, ttft_s: float,
                  itl_s: float, n_tokens: int, window: int = 16) -> dict:
    """Seconds a guardrail placement adds to the client's TTFT and end-to-end time (and tokens it can leak).

    Without guardrails the client sees TTFT = ttft_s and E2E = ttft_s + (n − 1)·itl_s.
    """
    base_e2e = ttft_s + (n_tokens - 1) * itl_s
    pre = {"off": 0.0, "shadow": 0.0, "inline": check_s, "parallel": max(0.0, check_s - ttft_s)}[input_placement]
    if output_placement in ("off", "shadow"):
        ttft, e2e, leak = ttft_s, base_e2e, n_tokens if output_placement == "shadow" else 0
    elif output_placement == "parallel":
        ttft, e2e, leak = ttft_s, base_e2e, min(n_tokens, window + int(check_s / itl_s) if itl_s else window)
    elif output_placement in ("window", "full"):
        w = n_tokens if output_placement == "full" else window
        rel = window_release_times(ttft_s, itl_s, n_tokens, w, check_s)
        ttft, e2e, leak = rel[0], rel[-1], 0
    else:
        raise ValueError(output_placement)
    if input_placement == "inline":
        ttft, e2e = ttft + pre, e2e + pre
    elif input_placement == "parallel":
        ttft, e2e = max(ttft, ttft_s + pre), max(e2e, ttft_s + pre)
    return {"ttft_s": ttft, "e2e_s": e2e, "added_ttft_s": ttft - ttft_s, "added_e2e_s": e2e - base_e2e,
            "tokens_leaked_if_flagged": leak}
