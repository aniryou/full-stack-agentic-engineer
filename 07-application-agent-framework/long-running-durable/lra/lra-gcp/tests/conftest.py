"""Test harness: one engine over in-memory adapters, driven like Cloud Tasks would."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import pytest

from lra import Engine, Event, Run
from lra.adapters.memory import FakeClock, FakeLLM, InMemoryEventBus, InMemoryStateStore, InMemoryTaskQueue, LocalRunner
from lra.examples import ALL_WORKFLOWS
from lra.examples.scripted import research_routes  # noqa: F401  (re-exported for tests)


@dataclass
class Harness:
    clock: FakeClock
    store: InMemoryStateStore
    queue: InMemoryTaskQueue
    bus: InMemoryEventBus
    llm: FakeLLM
    engine: Engine
    runner: LocalRunner

    def start(self, workflow: str, input: dict[str, Any] | None = None, **kw: Any) -> Run:
        return self.engine.start(workflow, input or {}, **kw)

    def drain(self, **kw: Any) -> int:
        return self.runner.run_until_idle(**kw)

    def run(self, run_id: str) -> Run:
        r = self.store.get(run_id)
        assert r is not None
        return r

    def approve(self, run_id: str, gate: str = "editor", by: str = "reviewer") -> Run | None:
        return self.engine.resume(Event(run_id=run_id, key=f"{gate}:{run_id}", payload={"decision": "approve", "by": by}))

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return self.bus.of_type(event_type)

    def history(self, run_id: str) -> list[tuple[str, str, str]]:
        return [(h.step, h.kind, h.status) for h in self.run(run_id).history]


def make_harness(
    *,
    routes: dict[str, Any] | None = None,
    fail_times: int = 0,
    chaos: Callable[[str, Run], None] | None = None,
    lease_ttl_s: int = 60,
    worker_id: str = "worker-A",
) -> Harness:
    clock = FakeClock()
    store = InMemoryStateStore()
    queue = InMemoryTaskQueue(clock)
    bus = InMemoryEventBus()
    llm = FakeLLM(routes=routes if routes is not None else research_routes(), fail_times=fail_times)
    engine = Engine(
        store=store, queue=queue, bus=bus, llm=llm, clock=clock, workflows=ALL_WORKFLOWS,
        worker_id=worker_id, lease_ttl=timedelta(seconds=lease_ttl_s), chaos=chaos,
    )
    return Harness(clock, store, queue, bus, llm, engine, LocalRunner(engine, queue, clock))


@pytest.fixture
def h() -> Harness:
    return make_harness()
