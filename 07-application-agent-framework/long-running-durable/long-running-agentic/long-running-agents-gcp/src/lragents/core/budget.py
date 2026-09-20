"""Budget guards — the circuit breaker for agent loops.

An autonomous loop has no natural stopping point. "Stop when done" is decided
by a probabilistic model, so the *deterministic* limits below are what actually
bound cost and blast radius. They are checked in code, before every LLM call,
never delegated to the prompt.
"""

from __future__ import annotations

import time

from .llm import PriceCard
from .models import Run


class BudgetExceeded(Exception):
    pass


def check_budget(run: Run, now: float | None = None) -> None:
    b, u = run.budget, run.usage
    now = now or time.time()
    if u.steps >= b.max_steps:
        raise BudgetExceeded(f"max_steps {b.max_steps} reached")
    if u.tokens >= b.max_tokens:
        raise BudgetExceeded(f"max_tokens {b.max_tokens} reached ({u.tokens})")
    if u.cost_usd >= b.max_cost_usd:
        raise BudgetExceeded(f"max_cost_usd {b.max_cost_usd} reached ({u.cost_usd:.4f})")
    if b.deadline_epoch is not None and now >= b.deadline_epoch:
        raise BudgetExceeded("deadline passed")


def charge(run: Run, tokens_in: int, tokens_out: int, price: PriceCard) -> float:
    cost = price.cost(tokens_in, tokens_out)
    run.usage.steps += 1
    run.usage.tokens += tokens_in + tokens_out
    run.usage.cost_usd += cost
    return cost
