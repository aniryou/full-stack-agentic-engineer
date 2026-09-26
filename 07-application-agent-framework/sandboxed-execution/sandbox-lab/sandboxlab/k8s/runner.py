"""runner.py — one execution = one Job, or one warm pod via ``kubectl exec``; same result either way.

One idea: on Kubernetes the sandbox's *lifecycle* is the API's — create, schedule, pull, start,
run, report, delete — and each step costs time and can fail differently (PRIMER §5 "Sandboxes on
Kubernetes", §6 "Latency, throughput and cost per action"). ``JobRunner`` pays the whole cold
start per execution and gets a fresh pod every time; ``WarmPoolRunner`` pays only an ``exec``
round trip by keeping started pods idle, and deletes each pod after one use so no state carries
over. Both parse the wrapper's result line from the pod's output into the same ``ExecResult``.

Backends:

* ``KubectlBackend`` drives a real cluster (kind from ``deploy/kind/up.sh``, or GKE).
* ``SimulatedBackend`` needs nothing: it runs the admission chain offline (``admission.admit``),
  runs the code for real in the T0 ``ProcessSandbox``, and *simulates* the Kubernetes lifecycle
  with a small latency model. Every result it returns is labelled simulated.

Idempotency: the Job name is derived from the idempotency key (turn, step, call index, argument
hash — the scaling primer's §5.4 recipe), so a retried request hits ``AlreadyExists`` and gets the
first run's result back instead of running the code twice.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import subprocess
import time
import uuid
from dataclasses import dataclass, field

from ..process import MARKER, Budgets, ExecResult, ProcessSandbox, parse_wrapper_output
from . import admission
from . import policy as P


class AlreadyExists(Exception):
    pass


class AdmissionError(Exception):
    def __init__(self, messages: list[str]):
        super().__init__("; ".join(messages))
        self.messages = messages


def execution_id(idempotency_key: str | None) -> str:
    """A DNS-label-safe id: a hash of the idempotency key (same key, same Job name), else random."""
    if idempotency_key:
        return hashlib.sha256(idempotency_key.encode()).hexdigest()[:16]
    return uuid.uuid4().hex[:16]


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def result_from_pod(logs: str, pod: dict | None, job: dict | None, *, isolation: str, total_s: float,
                    simulated: bool = False) -> ExecResult:
    """Wrapper line if there is one; otherwise the reason Kubernetes gives (OOMKilled, deadline,
    eviction, image pull, a missing runtime handler)."""
    d = parse_wrapper_output(logs)
    notes: list[str] = []
    startup = None
    if pod:
        created = _ts((pod.get("metadata") or {}).get("creationTimestamp"))
        cs = ((pod.get("status") or {}).get("containerStatuses") or [{}])[0]
        state = cs.get("state") or {}
        started = _ts((state.get("terminated") or state.get("running") or {}).get("startedAt"))
        if created and started:
            startup = started - created
            notes.append(f"pod created -> container started: {startup:.0f} s (API timestamps have 1 s resolution)")
    if d is not None:
        r = ExecResult.from_wrapper(d, isolation=isolation, total_s=total_s, notes=notes)
        r.simulated = simulated
        return r
    reason, msg = "sandbox_error", "no result line in the pod's logs"
    jc = {c.get("type"): c for c in ((job or {}).get("status") or {}).get("conditions") or []}
    term = ((((pod or {}).get("status") or {}).get("containerStatuses") or [{}])[0].get("state") or {}).get("terminated") or {}
    waiting = ((((pod or {}).get("status") or {}).get("containerStatuses") or [{}])[0].get("state") or {}).get("waiting") or {}
    pod_reason = ((pod or {}).get("status") or {}).get("reason")
    if term.get("reason") == "OOMKilled":
        reason, msg = "memory_limit", "OOMKilled by the pod's memory cgroup"
    elif (jc.get("Failed") or {}).get("reason") == "DeadlineExceeded":
        reason, msg = "timeout", "Job activeDeadlineSeconds exceeded (start-up included)"
    elif pod_reason == "Evicted":
        reason, msg = "disk_limit", (pod.get("status") or {}).get("message", "evicted")
    elif waiting.get("reason"):
        msg = f"{waiting['reason']}: {waiting.get('message', '')}"
    elif pod_reason:
        msg = f"{pod_reason}: {(pod.get('status') or {}).get('message', '')}"
    return ExecResult(reason, term.get("exitCode"), "", logs[-2000:], isolation=isolation,
                      total_s=round(total_s, 3), notes=notes + [msg], simulated=simulated)


# ---- backends -------------------------------------------------------------------------------------------------
class KubectlBackend:
    """Thin ``kubectl`` wrapper. ``dry_run=True`` records the commands and executes nothing."""

    def __init__(self, context: str | None = None, *, dry_run: bool = False, timeout_s: float = 180):
        self.context, self.dry_run, self.timeout_s = context, dry_run, timeout_s
        self.commands: list[str] = []
        self.simulated = False

    def _argv(self, *args: str) -> list[str]:
        return ["kubectl", *(["--context", self.context] if self.context else []), *args]

    def run(self, *args: str, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
        argv = self._argv(*args)
        self.commands.append(" ".join(argv) + (" < (stdin)" if stdin else ""))
        if self.dry_run:
            return subprocess.CompletedProcess(argv, 0, "{}", "")
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=self.timeout_s)
        if check and p.returncode != 0:
            if "AlreadyExists" in p.stderr or "already exists" in p.stderr:
                raise AlreadyExists(p.stderr.strip())
            if "denied request" in p.stderr or "violates PodSecurity" in p.stderr or "ValidatingAdmissionPolicy" in p.stderr:
                raise AdmissionError([p.stderr.strip()])
            raise RuntimeError(f"{' '.join(argv)}: {p.stderr.strip()}")
        return p

    def get(self, kind: str, name: str | None, ns: str, selector: str | None = None) -> dict:
        args = ["get", kind] + ([name] if name else []) + ["-n", ns, "-o", "json"] + (["-l", selector] if selector else [])
        return json.loads(self.run(*args).stdout or "{}")

    def create(self, obj: dict) -> None:
        self.run("create", "-f", "-", stdin=json.dumps(obj))

    def wait_job(self, name: str, ns: str, timeout_s: float) -> dict:
        end = time.monotonic() + timeout_s
        while True:
            job = self.get("job", name, ns)
            conds = {c["type"] for c in (job.get("status") or {}).get("conditions") or [] if c.get("status") == "True"}
            if self.dry_run or conds & {"Complete", "Failed"} or time.monotonic() > end:
                return job
            time.sleep(0.5)

    def job_pod(self, exec_id: str, ns: str) -> dict | None:
        items = self.get("pods", None, ns, f"{P.EXEC_ID_LABEL}={exec_id}").get("items") or []
        return items[0] if items else None

    def logs(self, pod: str, ns: str) -> str:
        return self.run("logs", pod, "-n", ns, "-c", "run", check=False).stdout or ""

    def delete(self, kind: str, name: str, ns: str) -> None:
        self.run("delete", kind, name, "-n", ns, "--wait=false", "--ignore-not-found", check=False)

    def idle_pods(self, ns: str) -> list[dict]:
        items = self.get("pods", None, ns, f"app=sandbox-warm,{P.STATE_LABEL}=idle").get("items") or []
        ready = lambda p: any(c.get("type") == "Ready" and c.get("status") == "True"   # noqa: E731
                              for c in (p.get("status") or {}).get("conditions") or [])
        return [p for p in items if ready(p) and not (p.get("metadata") or {}).get("deletionTimestamp")]

    def claim(self, pod: dict, ns: str) -> bool:
        """Optimistic concurrency: the label change only succeeds at the resourceVersion we read."""
        md = pod["metadata"]
        p = self.run("label", "pod", md["name"], "-n", ns, f"{P.STATE_LABEL}=claimed", "--overwrite",
                     f"--resource-version={md.get('resourceVersion', '')}", check=False)
        return p.returncode == 0

    def exec(self, pod: str, ns: str, argv: list[str], stdin: str) -> str:
        return self.run("exec", "-i", pod, "-n", ns, "-c", "run", "--", *argv, stdin=stdin, check=False).stdout


@dataclass
class LatencyModel:
    """Inputs of the *simulated* lifecycle (seconds). Illustrative values in the ranges the lab's
    fact sheet cites (a container start is 100-500 ms, gVisor adds to it; verify on your cluster
    with ``python3 -m sandboxlab bench``). They are model parameters, not measurements."""
    api_and_schedule: float = 0.15
    image_pull: float = 4.0                     # first pull on a node; 0 once cached
    sandbox_create: dict = field(default_factory=lambda: {"runc": 0.35, "gvisor": 0.7, "runsc": 0.7})
    container_start: float = 0.25
    status_poll: float = 0.5                    # the runner learns about completion by polling
    exec_roundtrip: float = 0.12
    jitter: float = 0.15                        # +-15 % uniform


class SimulatedBackend:
    """A cluster in a dict. Admission is real (the offline predictor), execution is real (the T0
    process sandbox runs the code), the lifecycle timing is simulated."""

    def __init__(self, policy: P.SandboxPolicy | None = None, *, model: LatencyModel | None = None,
                 executor: ProcessSandbox | None = None, seed: int = 0, cached_image: bool = True):
        self.policy = policy or P.SandboxPolicy.kind()
        self.model = model or LatencyModel()
        self.executor = executor or ProcessSandbox()
        self.rng = random.Random(seed)
        self.cached = cached_image
        self.clock = 0.0                        # simulated seconds since the backend started
        self.jobs: dict[str, dict] = {}
        self.pods: dict[str, dict] = {}
        self.pod_logs: dict[str, str] = {}
        self.ready_at: dict[str, float] = {}
        self.commands: list[str] = []
        self.simulated = True
        rc = P.runtime_class(self.policy)
        self.runtime_classes = {rc["metadata"]["name"]: rc}
        self.node_handlers = {self.policy.runtime_handler}
        for i in range(self.policy.warm_replicas):
            self._new_warm_pod(ready_in=0.0, name=f"sandbox-warm-{i}")

    def _j(self, x: float) -> float:
        return x * (1 + self.rng.uniform(-self.model.jitter, self.model.jitter))

    def cold_start_s(self) -> float:
        m = self.model
        return self._j(m.api_and_schedule + (0 if self.cached else m.image_pull)
                       + m.sandbox_create.get(self.policy.runtime_handler, 0.35) + m.container_start)

    def _new_warm_pod(self, ready_in: float, name: str | None = None) -> None:
        name = name or f"sandbox-warm-{uuid.uuid4().hex[:5]}"
        self.pods[name] = {"metadata": {"name": name, "labels": {"app": "sandbox-warm", P.STATE_LABEL: "idle"}}}
        self.ready_at[name] = self.clock + ready_in

    # -- Job path
    def create(self, obj: dict) -> None:
        self.commands.append(f"kubectl create -f - ({obj['kind']} {obj['metadata']['name']})")
        name = obj["metadata"]["name"]
        if name in self.jobs:
            raise AlreadyExists(f'jobs.batch "{name}" already exists')
        res = admission.admit(obj, runtime_classes=self.runtime_classes,
                              allowed_runtime_classes=self.policy.allowed_runtime_classes,
                              max_deadline_s=self.policy.max_deadline_s, node_handlers=self.node_handlers)
        job_level = [str(v) for v in res.violations if not v.message.startswith("(its Pods)")]
        if job_level:
            raise AdmissionError(job_level)
        self.jobs[name] = {"obj": obj, "admission": res, "created": self.clock}

    def wait_job(self, name: str, ns: str, timeout_s: float) -> dict:
        job = self.jobs[name]
        if "status" in job:
            return job["status"]
        res: admission.AdmissionResult = job["admission"]
        tmpl = job["obj"]["spec"]["template"]["spec"]
        exec_id = job["obj"]["metadata"]["labels"][P.EXEC_ID_LABEL]
        if not res.admitted or res.will_fail:     # Pods rejected by PSA/VAP, or no runtime handler
            why = res.will_fail or "; ".join(str(v) for v in res.violations)
            self.clock += self._j(self.model.api_and_schedule) + self.model.status_poll
            job["status"] = {"status": {"conditions": [{"type": "Failed", "status": "True", "reason": "PodFailed"}]}}
            self.pods[f"run-{exec_id}"] = {"metadata": {"name": f"run-{exec_id}"}, "status": {"reason": "Failed", "message": why}}
            self.pod_logs[f"run-{exec_id}"] = ""
            return job["status"]
        c = tmpl["containers"][0]
        code = next(e["value"] for e in c["env"] if e["name"] == "SANDBOX_CODE")
        budgets = json.loads(c["command"][c["command"].index("--budgets") + 1])
        startup = self.cold_start_s()
        r = self.executor.run(code, Budgets(**{k: v for k, v in budgets.items() if k in Budgets.__dataclass_fields__}))
        self.clock += startup + r.wall_s + self.model.status_poll
        d = {k: getattr(r, k) for k in ("exit_reason", "returncode", "stdout", "stderr", "stdout_bytes", "stderr_bytes",
                                          "stdout_truncated", "stderr_truncated", "wall_s", "cpu_s", "max_rss_kib",
                                          "stragglers_killed", "notes")}
        pod_name = f"run-{exec_id}"
        self.pod_logs[pod_name] = f"{MARKER} {json.dumps(d)}\n"
        base = dt.datetime(2026, 9, 26, 12, 0, 0, tzinfo=dt.timezone.utc)
        created = base + dt.timedelta(seconds=job["created"])
        started = created + dt.timedelta(seconds=startup)
        self.pods[pod_name] = {"metadata": {"name": pod_name, "creationTimestamp": created.isoformat()},
                               "status": {"containerStatuses": [{"state": {"terminated": {
                                   "startedAt": started.isoformat(), "exitCode": 0}}}]}, "_startup": startup}
        job["status"] = {"status": {"conditions": [{"type": "Complete", "status": "True"}]}}
        return job["status"]

    def job_pod(self, exec_id: str, ns: str) -> dict | None:
        return self.pods.get(f"run-{exec_id}")

    def logs(self, pod: str, ns: str) -> str:
        return self.pod_logs.get(pod, "")

    def delete(self, kind: str, name: str, ns: str) -> None:
        self.commands.append(f"kubectl delete {kind} {name} --wait=false")
        if kind == "job":
            self.jobs.pop(name, None)
        if kind == "pod" and name in self.pods and name.startswith("sandbox-warm"):
            del self.pods[name]
            self.ready_at.pop(name, None)
            self._new_warm_pod(ready_in=self.cold_start_s())     # the Deployment replaces it

    # -- warm path
    def idle_pods(self, ns: str) -> list[dict]:
        return [p for n, p in self.pods.items() if n.startswith("sandbox-warm")
                and p["metadata"]["labels"][P.STATE_LABEL] == "idle" and self.ready_at[n] <= self.clock]

    def next_ready_in(self) -> float | None:
        pending = [t - self.clock for n, t in self.ready_at.items() if t > self.clock]
        return min(pending) if pending else None

    def claim(self, pod: dict, ns: str) -> bool:
        pod["metadata"]["labels"][P.STATE_LABEL] = "claimed"
        return True

    def exec(self, pod: str, ns: str, argv: list[str], stdin: str) -> str:
        budgets = json.loads(argv[argv.index("--budgets") + 1])
        r = self.executor.run(stdin, Budgets(**{k: v for k, v in budgets.items() if k in Budgets.__dataclass_fields__}))
        self.clock += self._j(self.model.exec_roundtrip) + r.wall_s
        d = {k: getattr(r, k) for k in ("exit_reason", "returncode", "stdout", "stderr", "stdout_bytes", "stderr_bytes",
                                          "stdout_truncated", "stderr_truncated", "wall_s", "cpu_s", "max_rss_kib",
                                          "stragglers_killed", "notes")}
        return f"{MARKER} {json.dumps(d)}\n"

    def advance(self, seconds: float) -> None:
        """Let simulated time pass (between requests)."""
        self.clock += seconds


# ---- runners ------------------------------------------------------------------------------------------------
class JobRunner:
    """Pod-per-execution: create a Job, wait for it, read the result from its logs.

    Finished Jobs are left for ``ttlSecondsAfterFinished`` to delete (deleting them at once would
    let a retried request create the same Job name again and run the code twice). ``results`` is
    the result store keyed by idempotency key — in production a durable table kept for a day
    (scaling primer §5.4); the idempotency window is the longer of the TTL and that retention."""

    def __init__(self, policy: P.SandboxPolicy, backend, *, delete_after: bool = False,
                 results: dict | None = None):
        self.policy, self.backend, self.delete_after = policy, backend, delete_after
        self.results: dict[str, ExecResult] = results if results is not None else {}
        self.name = f"k8s:job({policy.runtime_class})"

    def run(self, code: str, budgets: Budgets | None = None, *, idempotency_key: str | None = None) -> ExecResult:
        if idempotency_key and idempotency_key in self.results:
            r = self.results[idempotency_key]
            return ExecResult(**{**r.__dict__, "notes": r.notes + ["replayed from the result store; not run again"]})
        p = self.policy
        if budgets:
            p = p.with_(budgets=json.loads(budgets.wrapper_json()))
        eid = execution_id(idempotency_key)
        job = P.run_code_job(p, code, eid, idempotency_key)
        name = job["metadata"]["name"]
        t0 = time.monotonic()
        sim0 = getattr(self.backend, "clock", None)
        replay = False
        try:
            self.backend.create(job)
        except AlreadyExists:
            replay = True                        # same key, Job still there: read its result, do not run again
        except AdmissionError as e:
            return ExecResult("sandbox_error", None, "", "\n".join(e.messages), isolation=self.name,
                              total_s=time.monotonic() - t0, notes=["rejected at admission"],
                              simulated=self.backend.simulated)
        status = self.backend.wait_job(name, P.SANDBOX_NS, p.job_deadline_s + 30)
        pod = self.backend.job_pod(eid, P.SANDBOX_NS)
        logs = self.backend.logs(pod["metadata"]["name"], P.SANDBOX_NS) if pod else ""
        total = (self.backend.clock - sim0) if sim0 is not None else time.monotonic() - t0
        r = result_from_pod(logs, pod, status, isolation=self.name, total_s=total, simulated=self.backend.simulated)
        if replay:
            r.notes.append("replayed: a Job with this idempotency key already existed; the code was not run again")
        if idempotency_key and r.exit_reason != "sandbox_error":
            self.results[idempotency_key] = r
        if self.delete_after and not replay:
            self.backend.delete("job", name, P.SANDBOX_NS)
        return r


class WarmPoolRunner:
    """Claim an idle warm pod, ``kubectl exec`` the wrapper with the code on stdin, delete the pod."""

    def __init__(self, policy: P.SandboxPolicy, backend, *, fallback: JobRunner | None = None):
        self.policy, self.backend, self.fallback = policy, backend, fallback
        self.name = f"k8s:warm({policy.runtime_class})"

    def run(self, code: str, budgets: Budgets | None = None, *, idempotency_key: str | None = None) -> ExecResult:
        p = self.policy
        if budgets:
            p = p.with_(budgets=json.loads(budgets.wrapper_json()))
        t0 = time.monotonic()
        sim0 = getattr(self.backend, "clock", None)
        for pod in self.backend.idle_pods(P.SANDBOX_NS):
            if self.backend.claim(pod, P.SANDBOX_NS):
                break
        else:
            if self.fallback:
                r = self.fallback.run(code, budgets, idempotency_key=idempotency_key)
                r.notes.append("no idle warm pod: fell back to a cold Job")
                return r
            return ExecResult("sandbox_error", None, "", "", isolation=self.name, total_s=0.0,
                              notes=["no idle warm pod"], simulated=self.backend.simulated)
        name = pod["metadata"]["name"]
        eid = execution_id(idempotency_key)
        argv = P.wrapper_command(p, "--stdin")
        argv[argv.index("/work")] = f"/work/{eid}"
        out = self.backend.exec(name, P.SANDBOX_NS, argv, code)
        self.backend.delete("pod", name, P.SANDBOX_NS)          # one use, then replaced
        total = (self.backend.clock - sim0) if sim0 is not None else time.monotonic() - t0
        return result_from_pod(out, None, None, isolation=self.name, total_s=total, simulated=self.backend.simulated)
