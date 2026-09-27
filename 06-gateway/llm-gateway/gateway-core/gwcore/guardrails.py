"""Guardrails and what they cost (PRIMER §7).

The one idea: a guardrail is a check at a *placement* -- the input, a tool call, a tool result, the
streamed output, the final output -- and each placement trades latency, dollars and exposure: inline
checks add their time to TTFT, parallel checks add nothing but let tokens through before the verdict,
held-back windows add a window of generation to TTFT, final checks give up streaming. Checks with a
false-positive rate compound over a conversation. Guardrails reduce risk; deterministic policy bounds it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HOOKS = ("input", "tool_call", "tool_result", "output_stream", "output_final")
STAND_IN = "regex screener (a labelled stand-in for a classifier; not a guardrail model)"
RULES = {
    "prompt_injection": r"ignore (all |any )?(previous|prior|above) instructions|disregard .{0,30}(system prompt|instructions)",
    "secret": r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16})\b",
    "card_number": r"\b(?:\d[ -]?){13,16}\b",
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
}
BLOCKING = {"input": {"prompt_injection", "secret"}, "tool_call": {"secret"}, "tool_result": {"prompt_injection"},
            "output_stream": {"secret", "card_number"}, "output_final": {"secret", "card_number"}}


@dataclass
class Verdict:
    block: bool
    findings: list


class RegexScreener:
    label = STAND_IN

    def __init__(self, rules: dict = RULES, blocking: dict = BLOCKING):
        self.rules = {k: re.compile(v, re.I) for k, v in rules.items()}
        self.blocking = blocking

    def check(self, text: str, hook: str) -> Verdict:
        found = sorted(k for k, rx in self.rules.items() if rx.search(text or ""))
        return Verdict(bool(set(found) & self.blocking[hook]), found)


def added_latency(placement: str, *, t_check: float, ttft: float, itl: float, out_tokens: int, window: int = 200) -> tuple:
    """(seconds added to TTFT, seconds added to end-to-end) for one check at one placement.

    inline_input     check, then call the model:              +t_check, +t_check
    parallel_input   call the model at once, hold its first token until the verdict:  +max(0, t_check - ttft)
    parallel_cancel  stream at once, cancel on a bad verdict:  0, 0 (tokens may reach the user first)
    shadow           check off the path, log only:             0, 0
    held_back        release output in windows of ``window`` tokens, each checked first:
                     +itl*(min(window, out) - 1) + t_check to TTFT, +t_check at the end (if checks keep up)
    final            check the whole answer, then send it:     TTFT becomes end-to-end + t_check
    """
    e2e = ttft + itl * (out_tokens - 1)
    if placement == "inline_input":
        return t_check, t_check
    if placement == "parallel_input":
        extra = max(0.0, t_check - ttft)
        return extra, extra
    if placement in ("parallel_cancel", "shadow"):
        return 0.0, 0.0
    if placement == "held_back":
        return itl * (min(window, out_tokens) - 1) + t_check, t_check
    if placement == "final":
        return e2e - ttft + t_check, t_check
    raise ValueError(placement)


def exposed_tokens(*, t_check: float, ttft: float, itl: float) -> int:
    """Tokens a ``parallel_cancel`` input check lets reach the user before its verdict."""
    return 0 if t_check < ttft else int((t_check - ttft) // itl) + 1


def cost_per_1k_checks(price_per_gpu_hour: float, seconds_per_check: float, concurrency: int = 1) -> float:
    """Dollars per 1,000 checks on a dedicated GPU running ``concurrency`` checks at a time."""
    return price_per_gpu_hour / 3600 * seconds_per_check / concurrency * 1000


def false_block_rate(fpr: float, checks: int) -> float:
    """P(at least one false block) over ``checks`` independent checks: 1 - (1 - fpr)^checks."""
    return 1 - (1 - fpr) ** checks
