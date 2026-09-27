"""consolidate.py — episodes become facts on a schedule, as a durable job: a lease, checkpoints, a crash, a resume.

The one idea (PRIMER §7; lra-gcp primer §3.3, §3.13): consolidation is a batch job over one user's
window of episodes, and a batch job that calls a model must survive being killed halfway. So it runs
as a *durable run*: a deterministic ``run_id`` (``consolidate/acme/u1/2026-W39``, one per user and
week — a retried schedule is the same run), a **lease** row so two workers never run it at once
(a crashed worker's lease expires and the next one takes over), a **checkpoint** row after every step
so a resume skips finished work — the model calls above all — and **idempotent effects** so a step
that crashed after writing but before checkpointing writes once when it re-runs.

The steps: ``load`` (the window's active episodes) → ``extract:i`` (one model call per chunk of
episodes; here the scripted extractor) → ``resolve`` (newer supersedes older, precedence human > user >
tool > inferred, a weaker contradiction is flagged — ``memory.resolve``) → ``apply:j`` (one action per
step) → ``reflect`` (if the window's importance sum crosses a threshold, one insight citing its
evidence — the generative-agents trigger, 150 in their code) → ``mark`` (consolidated episodes get a
TTL: forgetting by decay). ``crash_at="apply:2"`` raises ``SimulatedCrash`` *after* that step's effect
and *before* its checkpoint — the worst place to die. Checkpoints hold the text the run extracted, so a
finished run deletes them (and a forget deletes an in-flight run's): they are a copy like any other.

On Google Cloud this is a Cloud Run job started by Cloud Scheduler (``gcp_commands``; T3, printed).
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

from .extract import day, extract
from .memory import resolve
from .records import MemoryRecord, content_hash, default_deletion_key
from .store.sqlite import SQLiteMemoryStore

DAY = 86400.0


class SimulatedCrash(RuntimeError):
    """Raised by the crash hook: the process 'dies' here."""


class LeaseHeld(RuntimeError):
    """Another worker holds this run's lease and it has not expired."""


def run_id(tenant: str, user: str, window_end: float) -> str:
    """One run per (tenant, user, ISO week of the window's end): a retried or duplicated trigger is the same run."""
    y, w, _ = dt.datetime.fromtimestamp(window_end, dt.timezone.utc).isocalendar()
    return f"consolidate/{tenant}/{user}/{y}-W{w:02d}"


@dataclass
class RunReport:
    run_id: str
    worker: str
    status: str = "running"            # running | done | crashed
    steps_run: list[str] = field(default_factory=list)
    steps_skipped: list[str] = field(default_factory=list)
    model_calls: int = 0
    actions: dict = field(default_factory=dict)
    insights: int = 0
    episodes: int = 0

    def line(self) -> str:
        return (f"{self.run_id} [{self.worker}] {self.status}: ran {len(self.steps_run)} steps, skipped "
                f"{len(self.steps_skipped)}, model calls {self.model_calls}, actions {self.actions}")


class ConsolidationJob:
    def __init__(self, store: SQLiteMemoryStore, *, worker: str = "worker-1", lease_ttl_s: float = 60.0,
                 chunk_size: int = 4, reflect_threshold: float = 150.0, episode_ttl_days: float = 30.0,
                 clock=None, crash_at: str | None = None):
        self.store, self.worker, self.lease_ttl_s = store, worker, lease_ttl_s
        self.chunk_size, self.reflect_threshold, self.episode_ttl_days = chunk_size, reflect_threshold, episode_ttl_days
        self.clock = clock or store.clock or time.time
        self.crash_at = crash_at
        self.jobs = store.job_tables()

    # -- durable-run plumbing ----------------------------------------------------------------------
    def _step(self, rep: RunReport, run: str, step: str, fn):
        """Run ``fn`` unless checkpointed; renew the lease; crash here if asked (after the effect)."""
        done = self.jobs.checkpoint(run, step)
        if done is not None:
            rep.steps_skipped.append(step)
            return done
        if not self.jobs.acquire(run, self.worker, self.clock(), self.lease_ttl_s):   # heartbeat: renew or lose it
            raise LeaseHeld(f"{run}: lease lost to another worker during {step}")
        out = fn()
        rep.steps_run.append(step)
        if self.crash_at == step:
            rep.status = "crashed"
            raise SimulatedCrash(f"{self.worker} died in {step}, after its effect, before its checkpoint")
        self.jobs.save(run, step, out, self.clock())
        return out

    # -- the job -----------------------------------------------------------------------------------
    def run(self, tenant: str, user: str, window_start: float, window_end: float) -> RunReport:
        run = run_id(tenant, user, window_end)
        rep = RunReport(run, self.worker)
        if self.jobs.start(run, self.clock()) == "done":     # the schedule fired twice: nothing to do
            rep.status, rep.steps_skipped = "done", ["(run already done)"]
            return rep
        if not self.jobs.acquire(run, self.worker, self.clock(), self.lease_ttl_s):
            holder, expires, _ = self.jobs.lease(run)
            raise LeaseHeld(f"{run} is held by {holder} until t={expires:.0f}")

        episodes = self._step(rep, run, "load", lambda: [
            r.id for r in self.store.records(tenant, user, statuses=("active",), kind="episodic")
            if window_start <= r.created_at < window_end])
        rep.episodes = len(episodes)
        chunks = [episodes[i:i + self.chunk_size] for i in range(0, len(episodes), self.chunk_size)]
        candidates = []
        for i, chunk in enumerate(chunks):
            def extract_chunk(chunk=chunk):
                rep.model_calls += 1                     # one model call per chunk in a real system
                out = []
                for rid in chunk:
                    ep = self.store.get(rid)
                    for f in extract(ep.text):
                        out.append({"slot": f.slot, "value": f.value, "text": f.text, "confidence": f.confidence,
                                    "importance": f.importance, "valid_from": ep.valid_from, "trust": ep.trust,
                                    "evidence": ep.id})
                return out
            candidates += self._step(rep, run, f"extract:{i}", extract_chunk)

        plan = self._step(rep, run, "resolve", lambda: self._resolve(run, tenant, user, candidates))
        for j, action in enumerate(plan):
            self._step(rep, run, f"apply:{j}", lambda a=action: self._apply(run, tenant, user, a))
            rep.actions[action["action"]] = rep.actions.get(action["action"], 0) + 1

        def reflect():
            importance = sum(self.store.get(r).importance for r in episodes if self.store.get(r))
            changed = [a for a in plan if a["action"] == "UPDATE"]
            if importance < self.reflect_threshold or not changed:
                return {"insight": None, "importance": importance}
            text = "Insight: this week the user's circumstances changed: " + "; ".join(
                f"{a['slot'].replace('_', ' ')} is now {a['value']}" for a in changed) + "."
            evidence = sorted({a["evidence"] for a in changed})
            rec = MemoryRecord(tenant, user, text, kind="semantic", source="consolidation", trust="inferred",
                               provenance=evidence, confidence=0.8, importance=6, created_at=self.clock(),
                               deletion_key=default_deletion_key(tenant, user, "insight:" + run))
            rid, _ = self.store.add(rec, idempotency_key=f"{run}:insight")
            return {"insight": rid, "importance": importance}
        rep.insights = int(self._step(rep, run, "reflect", reflect)["insight"] is not None)

        def mark():
            now = self.clock()
            for rid in episodes:
                ep = self.store.get(rid)
                if ep and ep.ttl_s is None:
                    self.store.set_fields(rid, ttl_s=(now - ep.created_at) + self.episode_ttl_days * DAY)
            return {"marked": len(episodes)}
        self._step(rep, run, "mark", mark)

        rep.status = "done"
        self.jobs.finish(run, rep.__dict__, self.clock())
        self.jobs.clear(run)          # the checkpoints hold extracted text: a copy the deletion checklist would chase
        self.jobs.release(run, self.worker)
        return rep

    def _resolve(self, run: str, tenant: str, user: str, candidates: list[dict]) -> list[dict]:
        """Plan the actions in valid-time order against the partition's active facts (and the plan so far).
        Planned records get deterministic ids, so an UPDATE can name a fact an earlier step of the same
        plan will write, and a re-run writes the same ids."""
        state = self.store.records(tenant, user, statuses=("active",), kind="semantic")
        plan = []
        for c in sorted(candidates, key=lambda c: (c["valid_from"], c["slot"], c["value"])):
            new = MemoryRecord(tenant, user, c["text"], slot=c["slot"], value=c["value"], trust=c["trust"],
                               source="consolidation", valid_from=c["valid_from"], created_at=c["valid_from"],
                               confidence=c["confidence"], importance=c["importance"],
                               id="mem_" + content_hash(run, c["slot"], c["value"], c["evidence"]))
            if new.confidence < 0.7:
                plan.append({**c, "action": "SKIP_LOW_CONFIDENCE", "other": None})
                continue
            action, other = resolve(state, new)
            plan.append({**c, "id": new.id, "action": action, "other": other.id if other else None})
            if action == "UPDATE":
                other.status = "superseded"
                state.append(new)
            elif action == "ADD":
                state.append(new)
        return plan

    def _apply(self, run: str, tenant: str, user: str, a: dict) -> dict:
        if a["action"] in ("NOOP", "SKIP_LOW_CONFIDENCE", "ADD_HISTORY"):
            if a["action"] == "NOOP" and a["other"]:
                self.store.set_fields(a["other"], last_accessed=self.clock())
            return {"id": a["other"]}
        status = "flagged" if a["action"] == "FLAG" else "active"
        rec = MemoryRecord(tenant, user, a["text"], kind="semantic", source="consolidation", trust=a["trust"],
                           slot=a["slot"], value=a["value"], provenance=[a["evidence"]], confidence=a["confidence"],
                           importance=a["importance"], created_at=self.clock(), valid_from=a["valid_from"],
                           status=status, deletion_key=default_deletion_key(tenant, user, a["slot"]), id=a["id"])
        rid, created = self.store.add(rec, idempotency_key=content_hash(run, a["slot"], a["value"], a["evidence"]))
        if a["action"] == "UPDATE" and a["other"]:
            old = self.store.get(a["other"])
            if old and old.status == "active":           # like Graphiti: set expired_at only if not set yet
                self.store.supersede(old.id, at=a["valid_from"], now=self.clock())
        return {"id": rid, "created": created}


def due_partitions(store, window_start: float, window_end: float) -> list[tuple[str, str]]:
    """Every (tenant, user) with active episodes in the window: the job's work list."""
    return store.partitions_with_episodes(window_start, window_end)


def shard(parts: list[tuple[str, str]], index: int, count: int) -> list[tuple[str, str]]:
    """Cloud Run jobs start ``--tasks N`` copies with ``CLOUD_RUN_TASK_INDEX`` / ``CLOUD_RUN_TASK_COUNT``
    (verify): each takes the partitions whose stable hash lands on its index."""
    return [p for p in parts if int(content_hash(*p), 16) % count == index]


def gcp_commands(project: str = "PROJECT_ID", region: str = "us-central1", job: str = "memlab-consolidate",
                 image: str | None = None, schedule: str = "17 3 * * *", tasks: int = 1,
                 service_account: str | None = None, db_secret: str = "memlab-pg-dsn") -> list[str]:
    """The T3 path, printed (no Terraform in this lab): the job, its invoker, the schedule.

    Cloud Scheduler calls the Cloud Run Admin API's ``jobs/<job>:run`` with an **OAuth** token, because
    the target is a ``*.googleapis.com`` API; OIDC is for your own ``*.run.app`` service (the lra-gcp
    reaper). The v2 URI is the one Google's Terraform sample uses (facts sheet §13); ``--task-timeout``
    and ``roles/run.invoker`` sufficing for ``run.jobs.run`` are (verify).
    """
    image = image or f"{region}-docker.pkg.dev/{project}/memlab/memlab:0.1.0"
    sa = service_account or f"memlab-scheduler@{project}.iam.gserviceaccount.com"
    uri = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/jobs/{job}:run"
    return [
        f"gcloud run jobs create {job} --project {project} --region {region} --image {image} "
        f"--tasks {tasks} --max-retries 3 --task-timeout 900s "
        f"--set-secrets MEMLAB_PG_DSN={db_secret}:latest --set-env-vars MEMLAB_WINDOW_DAYS=7 "
        f"--command python --args=-m,memlab,consolidate,--window-days,7",
        f"gcloud iam service-accounts create {sa.split('@')[0]} --project {project}",
        f"gcloud run jobs add-iam-policy-binding {job} --project {project} --region {region} "
        f"--member serviceAccount:{sa} --role roles/run.invoker",
        f"gcloud scheduler jobs create http {job}-nightly --project {project} --location {region} "
        f"--schedule \"{schedule}\" --time-zone UTC --uri {uri} --http-method POST "
        f"--oauth-service-account-email {sa}",
        f"gcloud run jobs execute {job} --project {project} --region {region} --wait",
    ]


def cleanup_commands(project: str = "PROJECT_ID", region: str = "us-central1", job: str = "memlab-consolidate") -> list[str]:
    return [f"gcloud scheduler jobs delete {job}-nightly --project {project} --location {region} --quiet",
            f"gcloud run jobs delete {job} --project {project} --region {region} --quiet"]


def window(end: float, days: float = 7.0) -> tuple[float, float]:
    return end - days * DAY, end


def describe_window(start: float, end: float) -> str:
    return f"{day(start)} .. {day(end)}"
