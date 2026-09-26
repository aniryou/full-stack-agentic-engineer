"""The runners: a simulated cluster admits, runs (for real, in the process sandbox) and times executions;
idempotent replays never run code twice; Kubernetes failure reasons map to the contract's exit reasons."""
import sys

import pytest

from sandboxlab.k8s import policy as P
from sandboxlab.k8s import runner as R
from sandboxlab.process import Budgets

LINUX = sys.platform.startswith("linux")
POL = P.SandboxPolicy.kind()


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_job_runner_runs_and_replays():
    be = R.SimulatedBackend(POL, seed=0)
    jr = R.JobRunner(POL, be)
    r = jr.run("print(6 * 7)", idempotency_key="turn1:step1:call0:k")
    assert r.exit_reason == "ok" and r.stdout == "42\n" and r.simulated and r.startup_s > 0.3
    fresh = R.JobRunner(POL, be)                                                # a restarted runner: empty result store
    again = fresh.run("print(6 * 7)", idempotency_key="turn1:step1:call0:k")
    assert again.stdout == "42\n" and "not run again" in again.notes[-1]
    assert sum(c.startswith("kubectl create") for c in be.commands) == 2 and len(be.jobs) == 1


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_budgets_travel_into_the_pod():
    be = R.SimulatedBackend(POL, seed=0)
    r = R.JobRunner(POL, be).run("while True: pass", Budgets(cpu_s=1, wall_s=3))
    assert r.exit_reason == "cpu_time"


def test_admission_rejection_is_a_sandbox_error():
    bad = POL.with_(job_deadline_s=10_000)                                      # over the policy's max_deadline_s
    r = R.JobRunner(bad, R.SimulatedBackend(POL)).run("print(1)")
    assert r.exit_reason == "sandbox_error" and "job-deadline" in r.stderr


@pytest.mark.skipif(not LINUX, reason="Linux")
def test_warm_pool_is_fast_until_it_runs_dry():
    be = R.SimulatedBackend(POL, seed=0)
    warm = R.WarmPoolRunner(POL, be, fallback=R.JobRunner(POL, be))
    first = [warm.run(f"print({i})") for i in range(3)]
    assert [r.isolation for r in first] == ["k8s:warm(sandbox-runc)"] * 2 + ["k8s:job(sandbox-runc)"]
    assert first[0].total_s < first[2].total_s and "fell back" in first[2].notes[-1]
    assert any(c.startswith("kubectl delete pod sandbox-warm-0") for c in be.commands)   # one use, then replaced


@pytest.mark.parametrize("pod,job,reason", [
    ({"status": {"containerStatuses": [{"state": {"terminated": {"reason": "OOMKilled", "exitCode": 137}}}]}}, None, "memory"),
    ({"status": {}}, {"status": {"conditions": [{"type": "Failed", "reason": "DeadlineExceeded"}]}}, "wall_timeout"),
    ({"status": {"reason": "Evicted", "message": "Usage of EmptyDir volume work exceeds the limit 64Mi."}}, None, "disk_limit"),
    ({"status": {"containerStatuses": [{"state": {"waiting": {"reason": "ErrImagePull"}}}]}}, None, "sandbox_error"),
])
def test_kubernetes_reasons_map_to_exit_reasons(pod, job, reason):
    assert R.result_from_pod("", pod, job, isolation="k8s", total_s=1.0).exit_reason == reason


def test_kubectl_backend_dry_run_records_commands():
    kb = R.KubectlBackend("kind-sandbox-lab", dry_run=True)
    kb.create(P.run_code_job(POL, "print(1)", "abc"))
    kb.exec("sandbox-warm-x", "sandbox", ["python3", "-V"], "")
    assert kb.commands[0].startswith("kubectl --context kind-sandbox-lab create -f -")
    assert "exec -i sandbox-warm-x -n sandbox -c run -- python3 -V" in kb.commands[1]
