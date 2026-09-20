"""Budgets bound an agent's loop in three units: steps, tokens and wall-clock time.

A loop without all three is a cost incident waiting to happen. The budget is
shared by every nested agent in an invocation, so delegation cannot escape it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str, budget: "Budget"):
        super().__init__(f"budget exceeded: {reason} ({budget.summary()})")
        self.reason = reason
        self.budget = budget


@dataclass
class Budget:
    max_steps: int = 8              # model calls per invocation (across nested agents)
    max_tokens: int = 50_000        # input + output tokens per invocation
    max_seconds: float = 60.0       # wall clock per invocation
    max_depth: int = 3              # nesting depth of delegated agents
    steps_used: int = 0
    tokens_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def consume_step(self) -> None:
        self.steps_used += 1
        if self.steps_used > self.max_steps:
            raise BudgetExceeded(f"max_steps={self.max_steps}", self)

    def consume_tokens(self, n: int) -> None:
        self.tokens_used += n
        if self.tokens_used > self.max_tokens:
            raise BudgetExceeded(f"max_tokens={self.max_tokens}", self)

    def check_time(self) -> None:
        if self.elapsed() > self.max_seconds:
            raise BudgetExceeded(f"max_seconds={self.max_seconds}", self)

    def check_depth(self, depth: int) -> None:
        if depth > self.max_depth:
            raise BudgetExceeded(f"max_depth={self.max_depth}", self)

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())

    def summary(self) -> str:
        return f"steps {self.steps_used}/{self.max_steps}, tokens {self.tokens_used}/{self.max_tokens}, {self.elapsed():.1f}s/{self.max_seconds}s"
