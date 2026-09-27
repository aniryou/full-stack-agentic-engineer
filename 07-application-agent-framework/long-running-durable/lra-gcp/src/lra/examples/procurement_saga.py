"""Procurement saga: an agent that *acts* across three systems.

    reserve_stock ─▶ charge_payment ─▶ book_shipment ─▶ Done
         │                │                 │
      release           refund           (failure here rolls back both)

The step bodies call fake external systems through ``ctx.effect`` so they are
idempotent; compensations read the effect record to undo precisely.
``input["fail_at"]`` lets tests/notebooks inject a failure at any step.
"""

from __future__ import annotations

from typing import Any

from ..core.models import Budget
from ..core.workflow import Done, Next, StepContext, StepFailed, Workflow
from ..patterns.saga import compensating

procurement = Workflow("procurement", version="1", default_budget=Budget(max_steps=20, max_cost_usd=1.0))


class ExternalSystems:
    """Stand-in for inventory / payments / logistics APIs. Records every call."""

    calls: list[tuple[str, dict[str, Any]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    @classmethod
    def call(cls, name: str, **kw: Any) -> dict[str, Any]:
        cls.calls.append((name, kw))
        return {"ok": True, **kw}


def _fail_if(ctx: StepContext, step: str) -> None:
    if ctx.input.get("fail_at") == step:
        raise StepFailed(f"{step} rejected by external system")
    if ctx.input.get("flaky_at") == step and ctx.attempt == 1:
        raise TimeoutError(f"{step}: upstream timeout")  # retryable, succeeds on attempt 2


@procurement.step(start=True, compensate=compensating("reserve", lambda ctx, rec: ExternalSystems.call("release_stock", reservation_id=rec["reservation_id"])))
def reserve_stock(ctx: StepContext) -> Any:
    _fail_if(ctx, "reserve_stock")
    rec = ctx.effect("reserve", lambda: {"reservation_id": f"res_{ctx.run_id[-6:]}", **ExternalSystems.call("reserve_stock", sku=ctx.input["sku"], qty=ctx.input["qty"])})
    ctx.state["reservation_id"] = rec["reservation_id"]
    return Next("charge_payment")


@procurement.step(compensate=compensating("charge", lambda ctx, rec: ExternalSystems.call("refund", payment_id=rec["payment_id"])), max_attempts=3, backoff_base_s=0.5)
def charge_payment(ctx: StepContext) -> Any:
    _fail_if(ctx, "charge_payment")
    rec = ctx.effect("charge", lambda: {"payment_id": f"pay_{ctx.run_id[-6:]}", **ExternalSystems.call("charge", amount=ctx.input["amount"])})
    ctx.state["payment_id"] = rec["payment_id"]
    return Next("book_shipment")


@procurement.step(compensate=compensating("ship", lambda ctx, rec: ExternalSystems.call("cancel_shipment", shipment_id=rec["shipment_id"])))
def book_shipment(ctx: StepContext) -> Any:
    _fail_if(ctx, "book_shipment")
    rec = ctx.effect("ship", lambda: {"shipment_id": f"shp_{ctx.run_id[-6:]}", **ExternalSystems.call("book_shipment", address=ctx.input.get("address", "?"))})
    return Done({"reservation_id": ctx.state["reservation_id"], "payment_id": ctx.state["payment_id"], "shipment_id": rec["shipment_id"]})


WORKFLOWS = [procurement]
