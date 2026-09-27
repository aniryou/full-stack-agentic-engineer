"""gcp/main.py — the same Engine on Google Cloud, in one Cloud Run service.

    Store  -> Firestore   (one document per run; transaction = compare-and-set on `version`)
    Queue  -> Cloud Tasks (task name = run/step/attempt for dedup; schedule_time for backoff)
    Worker -> this service, POST /tasks, called by Cloud Tasks with an OIDC token
    Reaper -> Cloud Scheduler calling POST /reap every 2 minutes

Nothing in core.py changes: only the two adapters below are new. Deploy with ./deploy.sh.
"""

import json
import os
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request
from google.api_core.exceptions import AlreadyExists
from google.cloud import firestore, tasks_v2
from google.cloud.firestore_v1.base_query import FieldFilter
from google.protobuf import timestamp_pb2

from core import Conflict, Engine
from workflow import STEPS

PROJECT, LOCATION = os.environ["GOOGLE_CLOUD_PROJECT"], os.environ.get("LOCATION", "asia-southeast1")
QUEUE, SERVICE_URL, TASKS_SA = os.environ.get("QUEUE", "agent-steps"), os.environ["SERVICE_URL"], os.environ["TASKS_SA"]


# ------------------------------------------------------------------ Firestore
class FirestoreStore:
    def __init__(self, db):
        self.db, self.runs, self.effects = db, db.collection("runs"), db.collection("effects")

    def get(self, run_id):
        snap = self.runs.document(run_id).get()
        return snap.to_dict() if snap.exists else None

    def save(self, run):
        ref, tx = self.runs.document(run["id"]), self.db.transaction()

        @firestore.transactional
        def commit(tx):
            snap = ref.get(transaction=tx)
            if snap.exists and snap.get("version") != run["version"]:
                raise Conflict(run["id"])
            run["version"] += 1
            tx.set(ref, run)

        commit(tx)
        return run

    def effect_once(self, key, fn):
        ref = self.effects.document(key.replace("/", "_"))
        snap = ref.get()
        if snap.exists:
            return snap.get("value")
        value = fn()
        try:
            ref.create({"value": value})       # fails if a concurrent duplicate got there first
        except AlreadyExists:
            return ref.get().get("value")
        return value

    def active(self):
        return [d.to_dict() for d in self.runs.where(filter=FieldFilter("status", "in", ["RUNNING", "WAITING"])).stream()]


# ---------------------------------------------------------------- Cloud Tasks
class CloudTasksQueue:
    def __init__(self):
        self.client = tasks_v2.CloudTasksClient()
        self.parent = self.client.queue_path(PROJECT, LOCATION, QUEUE)

    def push(self, run_id, step, attempt, now, delay=0):
        task = {
            "name": f"{self.parent}/tasks/{run_id}-{step}-{attempt}",           # dedup by name
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{SERVICE_URL}/tasks",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"run_id": run_id, "step": step, "attempt": attempt}).encode(),
                "oidc_token": {"service_account_email": TASKS_SA},               # Cloud Run verifies it
            },
        }
        if delay:
            ts = timestamp_pb2.Timestamp()
            ts.FromDatetime(datetime.now(timezone.utc) + timedelta(seconds=delay))
            task["schedule_time"] = ts
        try:
            self.client.create_task(parent=self.parent, task=task)
            return True
        except AlreadyExists:
            return False


engine = Engine(FirestoreStore(firestore.Client()), CloudTasksQueue(), STEPS, lease_ttl=120)
app = Flask(__name__)


# ------------------------------------------------------------------ endpoints
@app.post("/runs")                                   # start (idempotent on run_id)
def start():
    body = request.get_json()
    return jsonify(engine.start(body["run_id"], "draft", {"topic": body["topic"]}))


@app.get("/runs/<run_id>")
def get_run(run_id):
    return jsonify(engine.store.get(run_id) or {}), 200


@app.post("/runs/<run_id>/events")                   # approval webhook -> resume
def events(run_id):
    body = request.get_json()
    return jsonify({"applied": engine.resume(run_id, body["key"], body["payload"]) is not None})


@app.post("/tasks")                                  # Cloud Tasks target: one step per request
def tasks():
    body = request.get_json()
    outcome = engine.execute(body["run_id"], body["step"], body["attempt"])
    return jsonify({"outcome": outcome}), (503 if outcome == "busy" else 200)   # 503 -> Cloud Tasks retries


@app.post("/reap")                                   # Cloud Scheduler target
def reap():
    return jsonify(engine.reap())
