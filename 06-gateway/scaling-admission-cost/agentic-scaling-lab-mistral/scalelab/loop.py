"""The agent turn: a loop with a budget, checkpoints and idempotent tools.

    plan call → tool calls (in parallel) → answer call → … until a final answer or the budget

Every step is checkpointed in the store *before* its result is acted on. If the process dies
and the queue redelivers the turn, ``run_turn`` finds the checkpointed steps and replays them
instead of re-executing — including writes, which are keyed by turn, step and tool so a
redelivered turn gets the ticket it already created back instead of a second one.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field

from .clock import CLOCK
from .model import FakeModel, Response
from .resilience import CircuitBreaker, CircuitOpen, RateLimited, TokenBucket, call_with_retries
from .tools import Tools


@dataclass
class Budget:
    max_steps: int = 6           # model + tool steps
    max_tokens: int = 40_000
    max_cost_usd: float = 0.25
    deadline_s: float = 45.0


class Store:
    """In-memory stand-in for Postgres (checkpoints) and Redis (idempotency markers)."""

    def __init__(self):
        self.steps: dict[str, list[dict]] = {}
        self.idem: dict[str, dict] = {}
        self.writes = 0

    def get_steps(self, turn_id: str) -> list[dict]:
        return list(self.steps.get(turn_id, []))

    def append_step(self, turn_id: str, step: dict) -> None:
        self.steps.setdefault(turn_id, []).append(step)
        self.writes += 1


@dataclass
class TurnResult:
    turn_id: str
    status: str                  # completed | failed
    text: str = ""
    error: str | None = None
    steps: int = 0
    resumed_steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model_attempts: int = 0
    rate_limited: int = 0
    hosted_calls: int = 0        # model calls served by the API (all of them in hosted mode; the spill-over in hybrid)
    tool_names: list[str] = field(default_factory=list)


FALLBACK = "I'm sorry — I couldn't complete that just now. A colleague will follow up."


async def run_turn(turn_id: str, user_message: str, *, model: FakeModel, tools: Tools, store: Store,
                   budget: Budget = Budget(), degrade_level: int = 0, bucket: TokenBucket | None = None,
                   breaker: CircuitBreaker | None = None, customer_id: str = "c1") -> TurnResult:
    t0 = CLOCK.now()
    deadline = t0 + budget.deadline_s
    checkpoints = store.get_steps(turn_id)          # non-empty on a redelivery
    result = TurnResult(turn_id, "completed", resumed_steps=len(checkpoints))
    messages = [{"role": "user", "content": user_message}]
    offered = tools.names(degrade_level)
    idx = 0
    try:
        while True:
            # --- budget: the loop must always terminate, whatever the model does
            if idx >= budget.max_steps:
                raise BudgetExceeded("steps")
            if result.tokens >= budget.max_tokens:
                raise BudgetExceeded("tokens")
            if result.cost_usd >= budget.max_cost_usd:
                raise BudgetExceeded("cost")
            if CLOCK.now() >= deadline - 1:
                raise BudgetExceeded("deadline")

            # --- model step (or replay it)
            if idx < len(checkpoints):
                resp = Response(**checkpoints[idx]["payload"])
            else:
                stats: dict = {}
                est_tokens = model.prefix_tokens * 0.1 + sum(len(m["content"]) // 4 for m in messages) + 400
                try:
                    resp = await call_with_retries(lambda: model.generate(messages, offered), deadline=deadline, bucket=bucket,
                                                   tokens=est_tokens, breaker=breaker, stats=stats)
                finally:  # count attempts and 429s even when the call ultimately fails
                    result.model_attempts += stats.get("attempts", 0)
                    result.rate_limited += stats.get("rate_limited", 0)
                result.tokens += resp.input_tokens + resp.output_tokens
                result.cost_usd += resp.cost_usd
                result.hosted_calls += resp.served_by == "hosted"
                store.append_step(turn_id, {"index": idx, "kind": "model", "payload": asdict(resp)})
            idx += 1
            if not resp.tool_calls:
                result.text = resp.text
                break
            messages.append({"role": "assistant", "content": "calling " + ", ".join(resp.tool_calls)})

            # --- tool step (or replay it)
            if idx < len(checkpoints):
                results = checkpoints[idx]["payload"]["results"]
            else:
                results = await asyncio.gather(*[
                    execute_tool(tools, store, name, {"customer_id": customer_id}, key=f"{turn_id}:{idx}:{i}:{name}")
                    for i, name in enumerate(resp.tool_calls)])
                store.append_step(turn_id, {"index": idx, "kind": "tool", "payload": {"tools": resp.tool_calls, "results": results}})
            idx += 1
            for name, res in zip(resp.tool_calls, results):
                result.tool_names.append(name)
                messages.append({"role": "tool", "name": name, "content": json.dumps(res)})
    except BudgetExceeded as e:
        result.status, result.error, result.text = "failed", f"budget:{e}", FALLBACK
    except (RateLimited, CircuitOpen, TimeoutError) as e:
        result.status, result.error, result.text = "failed", f"overloaded:{type(e).__name__}", FALLBACK
    result.steps = idx
    result.latency_s = CLOCK.now() - t0
    return result


class BudgetExceeded(Exception):
    pass


async def execute_tool(tools: Tools, store: Store, name: str, args: dict, *, key: str) -> dict:
    """Reads just run. Writes check the idempotency marker first, and record it after."""
    if not tools.is_idempotent(name):
        if key in store.idem:
            return {**store.idem[key], "deduplicated": True}
        out = await tools.run(name, args)
        if "error" not in out:
            store.idem[key] = out
        return out
    return await tools.run(name, args)
