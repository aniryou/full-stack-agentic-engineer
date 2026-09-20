"""Saga: multi-system side effects with compensation.

When an agent *acts* (reserves stock, charges a card, files a ticket, sends
an e-mail) across systems that share no transaction, the only consistency
you can buy is "eventually, and with undo". The saga pattern makes every
side-effecting step declare its compensation; on failure the engine runs
the compensations of *completed* steps in reverse order.

Rules of thumb that this module encodes:

* Effects are idempotent (``ctx.effect``) so a retried step never double-charges.
* Compensations are idempotent too — they may be retried.
* Compensations use the *effect record* (e.g. the payment id) rather than
  recomputing, so "undo" targets exactly what was done.
* Some effects cannot be undone (an e-mail). Either sequence them last, or
  make the compensation a corrective action (send a retraction).
"""

from __future__ import annotations

from typing import Any, Callable

from ..core.workflow import StepContext


def effect_record(ctx: StepContext, key: str) -> dict[str, Any] | None:
    """Fetch what a previous ``ctx.effect(key, ...)`` produced (for compensations)."""
    return ctx._store.effect_get(f"{ctx.run_id}:{key}")


def compensating(effect_key: str, undo: Callable[[StepContext, dict[str, Any]], None]) -> Callable[[StepContext], None]:
    """Build a compensation that undoes ``effect_key`` using its stored result, idempotently."""

    def _compensate(ctx: StepContext) -> None:
        rec = effect_record(ctx, effect_key)
        if rec is None:
            return  # the effect never happened; nothing to undo
        ctx.effect(f"undo:{effect_key}", lambda: (undo(ctx, rec) or {"undone": effect_key}))

    return _compensate
