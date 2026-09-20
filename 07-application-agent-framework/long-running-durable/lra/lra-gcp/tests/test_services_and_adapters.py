"""Cloud Run services (in local mode) and GCP adapter request-building with fake clients."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from lra import LeaseHeldError, StepTask
from lra.core.models import Run


# --------------------------------------------------------------------- services
@pytest.fixture
def clients(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("LRA_BACKEND", "memory")
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "services"))
    monkeypatch.syspath_prepend(str(root))
    for mod in ("common", "services.api.main", "services.worker.main"):
        sys.modules.pop(mod, None)
    common = importlib.import_module("common")
    common.engine.cache_clear()
    common.settings.cache_clear()
    api = importlib.import_module("services.api.main")
    worker = importlib.import_module("services.worker.main")
    return TestClient(api.app), TestClient(worker.app), common.engine()


def _drive(worker, engine) -> list[tuple[str, int, str]]:
    seen = []
    while (t := engine.queue.pop_due()) is not None:
        resp = worker.post("/tasks/step", json=t.model_dump(mode="json"))
        seen.append((t.step, resp.status_code, resp.json()["outcome"]))
        engine.queue.forget(t.dedup_key)
    return seen


def test_api_start_is_idempotent_and_worker_drives_to_completion(clients) -> None:
    api, worker, engine = clients
    body = {"workflow": "procurement", "input": {"sku": "X", "qty": 1, "amount": 5}}
    r1 = api.post("/runs", json=body, headers={"Idempotency-Key": "order-7"})
    r2 = api.post("/runs", json=body, headers={"Idempotency-Key": "order-7"})
    assert r1.status_code == 201 and r1.json()["run_id"] == r2.json()["run_id"] == "order-7"
    assert _drive(worker, engine) == [("reserve_stock", 200, "ok"), ("charge_payment", 200, "ok"), ("book_shipment", 200, "done")]
    got = api.get("/runs/order-7").json()
    assert got["status"] == "SUCCEEDED" and got["result"]["payment_id"].startswith("pay_")
    assert len(got["history"]) == 3


def test_api_events_resume_and_duplicates_are_not_errors(clients) -> None:
    api, worker, engine = clients
    engine.llm.routes.update(
        {
            r"Break the GOAL": json.dumps({"subtasks": [{"title": "A", "instructions": "a"}]}),
            r"rigorous reviewer": json.dumps({"score": 9, "fixes": []}),
        }
    )
    run_id = api.post("/runs", json={"workflow": "research_pipeline", "input": {"goal": "svc"}}).json()["run_id"]
    _drive(worker, engine)
    assert api.get(f"/runs/{run_id}").json()["status"] == "WAITING"
    evt = {"key": f"editor:{run_id}", "payload": {"decision": "approve", "by": "qa"}, "event_id": "evt-1"}
    assert api.post(f"/runs/{run_id}/events", json=evt).json()["applied"] is True
    assert api.post(f"/runs/{run_id}/events", json=evt).json()["applied"] is False
    _drive(worker, engine)
    assert api.get(f"/runs/{run_id}").json()["status"] == "SUCCEEDED"
    assert api.post("/runs/nope/events", json=evt).status_code == 404


def test_worker_maps_lease_held_to_503_for_cloud_tasks_retry(clients) -> None:
    api, worker, engine = clients
    run_id = api.post("/runs", json={"workflow": "procurement", "input": {"sku": "X", "qty": 1, "amount": 5}}).json()["run_id"]
    engine.store.acquire_lease(run_id, "someone-else", timedelta(seconds=60), engine.clock.now())
    task = engine.queue.pop_due()
    resp = worker.post("/tasks/step", json=task.model_dump(mode="json"))
    assert resp.status_code == 503 and resp.json()["outcome"] == "lease-held"
    assert worker.post("/internal/reap").status_code == 200


def test_api_cancel(clients) -> None:
    api, worker, engine = clients
    run_id = api.post("/runs", json={"workflow": "procurement", "input": {"sku": "X", "qty": 1, "amount": 5}}).json()["run_id"]
    assert api.post(f"/runs/{run_id}/cancel").json()["status"] in {"PENDING", "RUNNING"}
    _drive(worker, engine)
    assert api.get(f"/runs/{run_id}").json()["status"] == "CANCELLED"


# ----------------------------------------------------------------- cloud tasks
class _FakeTasksClient:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.names: set[str] = set()

    def queue_path(self, p, l, q):  # noqa: E741
        return f"projects/{p}/locations/{l}/queues/{q}"

    def task_path(self, p, l, q, t):  # noqa: E741
        return f"{self.queue_path(p, l, q)}/tasks/{t}"

    def create_task(self, *, parent, task):
        from google.api_core import exceptions as gexc

        if task["name"] in self.names:
            raise gexc.AlreadyExists("dup")
        self.names.add(task["name"])
        self.created.append(task)


def test_cloud_tasks_request_shape_dedup_and_delay() -> None:
    from lra.adapters.gcp.cloud_tasks_queue import CloudTasksQueue

    q = CloudTasksQueue(
        project="p", location="asia-southeast1", queue="agent-steps",
        target_url="https://worker.run.app/tasks/step", service_account_email="tasks@p.iam.gserviceaccount.com",
        audience="https://worker.run.app", client=_FakeTasksClient(),
    )
    task = StepTask(run_id="run_1", step="charge_payment", attempt=2)
    assert q.enqueue(task) is True
    assert q.enqueue(task) is False, "same name -> AlreadyExists -> dedup"
    req = q.client.created[0]
    assert req["name"].endswith("/tasks/run_1--step--charge_payment--2")
    http = req["http_request"]
    assert http["url"] == "https://worker.run.app/tasks/step"
    assert json.loads(http["body"]) == {"run_id": "run_1", "step": "charge_payment", "attempt": 2, "kind": "step", "not_before": None}
    assert http["oidc_token"] == {"service_account_email": "tasks@p.iam.gserviceaccount.com", "audience": "https://worker.run.app"}
    assert req["dispatch_deadline"].seconds == 1800
    delayed = q.build_request(StepTask(run_id="run_1", step="x", attempt=1), delay=timedelta(minutes=5))
    assert "schedule_time" in delayed


# --------------------------------------------------------------------- pub/sub
def test_pubsub_publish_uses_ordering_key_per_run() -> None:
    from lra.adapters.gcp.pubsub_bus import PubSubEventBus

    class FakePublisher:
        def __init__(self):
            self.calls = []

        def topic_path(self, p, t):
            return f"projects/{p}/topics/{t}"

        def publish(self, topic, data, **kw):
            self.calls.append((topic, data, kw))
            return SimpleNamespace(result=lambda timeout=None: "msg-1")

    bus = PubSubEventBus(project="p", client=FakePublisher())
    assert bus.publish("agent-events", {"type": "run.started"}, {"run_id": "r1", "type": "run.started"}) == "msg-1"
    topic, data, kw = bus.client.calls[0]
    assert topic.endswith("/topics/agent-events") and json.loads(data)["type"] == "run.started"
    assert kw["ordering_key"] == "r1" and kw["run_id"] == "r1"


# ---------------------------------------------------------------------- gemini
def test_gemini_adapter_accounts_usage_and_retries_503() -> None:
    from google.genai import errors

    from lra.adapters.gcp.gemini_llm import GeminiLLM

    class FakeModels:
        def __init__(self):
            self.n = 0

        def generate_content(self, *, model, contents, config):
            self.n += 1
            if self.n == 1:
                raise errors.APIError(503, {"error": {"message": "overloaded"}})
            return SimpleNamespace(text='{"ok": true}', usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=50))

    llm = GeminiLLM(project="p", client=SimpleNamespace(models=FakeModels()), pricing_per_1m=(1.0, 2.0))
    import time

    monkey = time.sleep
    time.sleep = lambda s: None
    try:
        resp = llm.generate("hi", json_mode=True)
    finally:
        time.sleep = monkey
    assert resp.json() == {"ok": True}
    assert resp.usage.input_tokens == 100 and resp.usage.output_tokens == 50
    assert abs(resp.usage.cost_usd - (100 / 1e6 * 1.0 + 50 / 1e6 * 2.0)) < 1e-12
    assert llm.client.models.n == 2


# ------------------------------------------------------------------- firestore
class _FakeSnapshot:
    def __init__(self, doc):
        self._doc = doc

    @property
    def exists(self):
        return self._doc is not None

    def to_dict(self):
        return json.loads(json.dumps(self._doc))

    def get(self, key):
        return self._doc[key]


class _FakeDocRef:
    def __init__(self, coll, key):
        self.coll, self.key = coll, key

    def get(self, transaction=None):
        return _FakeSnapshot(self.coll.docs.get(self.key))

    def create(self, doc):
        from google.api_core import exceptions as gexc

        if self.key in self.coll.docs:
            raise gexc.AlreadyExists("exists")
        self.coll.docs[self.key] = json.loads(json.dumps(doc))


class _FakeCollection:
    def __init__(self):
        self.docs = {}

    def document(self, key):
        return _FakeDocRef(self, key)


class _FakeTx:
    def set(self, ref, doc):
        ref.coll.docs[ref.key] = json.loads(json.dumps(doc))

    def update(self, ref, patch):
        ref.coll.docs[ref.key].update(patch)


class _FakeClient:
    def __init__(self):
        self.colls = {}

    def collection(self, name):
        return self.colls.setdefault(name, _FakeCollection())

    def transaction(self):
        return _FakeTx()


def test_firestore_store_version_check_lease_and_fan_in() -> None:
    from lra.adapters.gcp.firestore_store import FirestoreStateStore
    from lra.core.ports import ConflictError

    store = FirestoreStateStore(client=_FakeClient())
    store._fs = SimpleNamespace(transactional=lambda fn: fn)  # identity decorator: run the closure inline

    run = store.create(Run(run_id="r1", workflow="wf", current_step="a"))
    assert run.version == 1
    with pytest.raises(ConflictError):
        store.create(Run(run_id="r1", workflow="wf"))

    loaded = store.get("r1")
    loaded.state["x"] = 1
    saved = store.save(loaded, expected_version=1)
    assert saved.version == 2 and store.get("r1").state == {"x": 1}
    with pytest.raises(ConflictError):
        store.save(loaded, expected_version=1)  # stale writer loses

    now = saved.updated_at
    leased = store.acquire_lease("r1", "w1", timedelta(seconds=30), now)
    assert leased.lease.owner == "w1"
    with pytest.raises(LeaseHeldError):
        store.acquire_lease("r1", "w2", timedelta(seconds=30), now)
    store.acquire_lease("r1", "w2", timedelta(seconds=30), now + timedelta(seconds=31))  # expired -> takeover
    store.release_lease("r1", "w2")
    assert store.get("r1").lease is None

    assert store.effect_put("r1:charge", {"payment_id": "p1"}) is True
    assert store.effect_put("r1:charge", {"payment_id": "p2"}) is False
    assert store.effect_get("r1:charge") == {"payment_id": "p1"}

    from lra.core.models import FanIn

    parent = store.get("r1")
    parent.fan_in = FanIn(expected=2, then="agg")
    store.save(parent, expected_version=parent.version)
    p = store.record_child_result("r1", "c1", {"ok": 1}, None)
    p = store.record_child_result("r1", "c1", {"ok": 1}, None)  # duplicate ignored
    p = store.record_child_result("r1", "c2", None, "boom")
    assert p.fan_in.completed == 2 and p.fan_in.results == {"c1": {"ok": 1}} and p.fan_in.failures == {"c2": "boom"}
