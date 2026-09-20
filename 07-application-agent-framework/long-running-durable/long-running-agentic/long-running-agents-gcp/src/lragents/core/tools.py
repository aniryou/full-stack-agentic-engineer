"""Tools and idempotent side effects.

Rule of the road: *at-least-once delivery + idempotent handlers = effectively
once*. Cloud Tasks, Pub/Sub, Cloud Scheduler and Cloud Run retries all deliver
at least once. Nothing on the platform will ever give you exactly-once *actions*,
so the action itself must tolerate a replay. The two mechanisms here:

* ``run_idempotent(store, key, fn)`` – memoise a tool result under a key that is
  stable across retries (``"{run_id}:{step_index}"``). A retry finds the cached
  result and never re-executes the side effect.
* External systems that accept an ``Idempotency-Key`` (Stripe-style). The fake
  ``PaymentGateway`` below shows the contract every mutating tool should demand
  from its downstream API.

``FaultInjector`` lets tests and notebooks kill the process at a named point
("after_side_effect", "before_checkpoint" ...) exactly once, so you can *see*
the recovery path run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .llm import ToolSpec


class ToolError(Exception):
    """Non-retryable tool failure; surfaces to the LLM as an observation."""


class SimulatedCrash(RuntimeError):
    """Raised by FaultInjector to model a process dying mid-step."""


class FaultInjector:
    def __init__(self) -> None:
        self._points: dict[str, int] = {}
        self.fired: list[str] = []

    def crash_once_at(self, point: str) -> None:
        self._points[point] = self._points.get(point, 0) + 1

    def maybe_crash(self, point: str) -> None:
        if self._points.get(point, 0) > 0:
            self._points[point] -= 1
            self.fired.append(point)
            raise SimulatedCrash(f"simulated crash at {point}")


class IdempotencyStore(Protocol):
    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, result: Any) -> None: ...
    def __contains__(self, key: str) -> bool: ...


class InMemoryIdempotencyStore:
    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._d.get(key)

    def put(self, key: str, result: Any) -> None:
        self._d[key] = result

    def __contains__(self, key: str) -> bool:
        return key in self._d


def run_idempotent(store: IdempotencyStore, key: str, fn: Callable[[], Any]) -> tuple[Any, bool]:
    """Execute ``fn`` unless a result for ``key`` already exists.

    Returns (result, replayed). The ordering matters: we execute, *then* record.
    If we crash between the two, the retry executes again — so ``fn`` itself must
    also be safe to replay (pass the key downstream). Defence in depth.
    """
    if key in store:
        return store.get(key), True
    result = fn()
    store.put(key, result)
    return result, False


# --------------------------------------------------------------------------
# Tool registry
# --------------------------------------------------------------------------

ToolFn = Callable[[dict[str, Any], "ToolContext"], Any]


@dataclass
class ToolContext:
    run_id: str
    step_index: int
    idempotency_key: str
    state: dict[str, Any]


@dataclass
class Tool:
    name: str
    description: str
    fn: ToolFn
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    requires_approval: bool = False      # human-in-the-loop gate before execution
    is_async: bool = False               # returns a ticket; completion arrives later
    poll: Callable[[str], dict[str, Any]] | None = None   # for async tools: poll(ticket) -> {"done": bool, "result": ...}

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(self._tools)}")
        return self._tools[name]

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# --------------------------------------------------------------------------
# Fake external systems used by tests and notebooks
# --------------------------------------------------------------------------


@dataclass
class Charge:
    key: str
    amount: float
    ts: float = field(default_factory=time.time)


class PaymentGateway:
    """A downstream API that honours idempotency keys (the contract you want)."""

    def __init__(self) -> None:
        self.charges: list[Charge] = []
        self._by_key: dict[str, Charge] = {}
        self.refunds: list[str] = []

    def charge(self, amount: float, idempotency_key: str) -> dict[str, Any]:
        if idempotency_key in self._by_key:            # replay → same answer, no new charge
            c = self._by_key[idempotency_key]
            return {"charge_id": c.key, "amount": c.amount, "replayed": True}
        c = Charge(key=idempotency_key, amount=amount)
        self.charges.append(c)
        self._by_key[idempotency_key] = c
        return {"charge_id": c.key, "amount": c.amount, "replayed": False}

    def refund(self, charge_id: str) -> dict[str, Any]:
        if charge_id not in self.refunds:              # refund is idempotent by charge id
            self.refunds.append(charge_id)
        return {"refunded": charge_id}


class NaivePaymentGateway:
    """A downstream API WITHOUT idempotency keys — every call charges again.
    Exists to show what goes wrong when the contract is missing."""

    def __init__(self) -> None:
        self.charges: list[float] = []

    def charge(self, amount: float, **_: Any) -> dict[str, Any]:
        self.charges.append(amount)
        return {"charge_id": f"ch_{len(self.charges)}", "amount": amount}


class SlowJobService:
    """An external system whose work takes wall-clock time: submit() returns a
    ticket immediately; status() flips to done after ``duration_s`` on the given clock."""

    def __init__(self, clock: Callable[[], float], duration_s: float = 120.0) -> None:
        self.clock = clock
        self.duration_s = duration_s
        self.jobs: dict[str, dict[str, Any]] = {}

    def submit(self, payload: dict[str, Any], idempotency_key: str) -> str:
        ticket = f"job_{idempotency_key}"
        self.jobs.setdefault(ticket, {"submitted_at": self.clock(), "payload": payload})
        return ticket

    def status(self, ticket: str) -> dict[str, Any]:
        job = self.jobs[ticket]
        done = self.clock() - job["submitted_at"] >= self.duration_s
        return {"done": done, "result": {"summary": f"processed {job['payload']}"} if done else None}
