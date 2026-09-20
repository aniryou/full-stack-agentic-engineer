"""End-to-end demo on the in-memory adapters: fan-out, reflection, a crash, a 3-day wait, approval, saga.

    python scripts/local_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from lra import Event, RunStatus, SimulatedCrash  # noqa: E402
from lra.examples.procurement_saga import ExternalSystems  # noqa: E402
from conftest import make_harness  # noqa: E402


def banner(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(0, 70 - len(title)))


def main() -> None:
    crashed: list[str] = []

    def chaos(point: str, run) -> None:  # kill the worker once, after committing 'synthesize'
        if point == "after_commit_before_enqueue" and run.current_step == "reflect_critique" and not crashed:
            crashed.append(run.run_id)
            raise SimulatedCrash()

    h = make_harness(chaos=chaos)

    banner("1. start a research run: plan -> 3 child runs -> synthesize")
    run = h.start("research_pipeline", {"goal": "What makes an agent workflow durable?"})
    try:
        h.drain()
    except SimulatedCrash:
        print(f"worker crashed after checkpointing 'synthesize' on {run.run_id} (lease still held)")

    banner("2. reaper recovers the expired lease; reflection loop runs; run parks for review")
    h.clock.advance(seconds=61)
    print("reap:", json.dumps(h.engine.reap()))
    h.drain()
    r = h.run(run.run_id)
    print("status:", r.status.value, "| waiting on:", r.wait.key, "| times out:", r.wait.timeout_at.date())
    print("reflection:", r.state["reflect_loop"]["exit_reason"])
    print("queue length while waiting:", len(h.queue), "(nothing runs, nothing is billed)")

    banner("3. two days pass; the editor approves; publish + notify run; effects are recorded once")
    h.clock.advance(days=2)
    h.engine.resume(Event(run_id=run.run_id, key=r.wait.key, payload={"decision": "approve", "by": "editor@example.com"}))
    h.drain()
    r = h.run(run.run_id)
    print("status:", r.status.value, "| result:", r.result)
    print("steps:", " -> ".join(f"{s.step}[{s.status}]" for s in r.history))
    print(f"budget: {r.budget.steps_used} steps, {r.budget.tokens_used} tokens, ${r.budget.cost_usd:.5f}")

    banner("4. procurement saga: shipment fails after stock + payment succeeded")
    ExternalSystems.reset()
    saga = h.start("procurement", {"sku": "GPU-H100", "qty": 1, "amount": 25_000, "flaky_at": "charge_payment", "fail_at": "book_shipment"})
    h.drain()
    s = h.run(saga.run_id)
    print("status:", s.status.value, "| error:", s.error)
    print("external calls:", [c[0] for c in ExternalSystems.calls], "(charge once despite the retry; undone in reverse)")

    banner("5. every transition was published for dashboards")
    print(sorted({m["payload"]["type"] for m in h.bus.published}))
    assert r.status == RunStatus.SUCCEEDED and s.status == RunStatus.COMPENSATED


if __name__ == "__main__":
    main()
