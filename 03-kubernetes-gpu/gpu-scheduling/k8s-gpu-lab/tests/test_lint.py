"""One focused test per lint rule, plus: the lab's own manifests are lint-clean."""
from pathlib import Path

from k8sgpu import lint, machines
from k8sgpu import manifests as m

ROOT = Path(__file__).resolve().parents[1]


def rules(findings, severity=None):
    return {f.rule for f in findings if severity is None or f.severity == severity}


def job_with(container: dict, **spec) -> dict:
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": "j"},
            "spec": {"template": {"spec": {"restartPolicy": "Never", "containers": [container], **spec}}}}


def test_api_server_gpu_rules_are_reproduced_verbatim():
    assert lint.check_gpu_resources({"limits": {m.GPU: 1}}) == []
    assert lint.check_gpu_resources({"requests": {m.GPU: 1}}) == [
        f"limits: Limit must be set for non overcommitable resources ({m.GPU})"]
    assert lint.check_gpu_resources({"requests": {m.GPU: 1}, "limits": {m.GPU: 2}}) == [
        f"requests: 1 must be equal to {m.GPU} limit of 2"]
    assert "must be an integer" in lint.check_gpu_resources({"limits": {m.GPU: "500m"}})[0]


def test_toleration_selector_and_requests_rules():
    c = {"name": "c", "image": "x", "resources": {"limits": {m.GPU: 1}}}
    f = lint.lint(job_with(c))
    assert {"gpu-toleration", "gpu-node-selector", "cpu-memory-requests"} <= rules(f, "warning")
    good = job_with({**c, "resources": {"requests": {"cpu": "1", "memory": "1Gi"}, "limits": {"memory": "1Gi", m.GPU: 1}}},
                    tolerations=[m.GPU_TOLERATION], nodeSelector={m.GKE_ACCELERATOR: "nvidia-l4"})
    assert lint.lint(good) == []


def test_kueue_managed_jobs_only_get_info_for_missing_selector():
    o = job_with({"name": "c", "image": "x", "resources": {"requests": {"cpu": "1", "memory": "1Gi"},
                                                           "limits": {"memory": "1Gi", m.GPU: 1}}},
                 tolerations=[m.GPU_TOLERATION])
    o["metadata"]["labels"] = {m.QUEUE_LABEL: "q"}
    assert [(f.rule, f.severity) for f in lint.lint(o)] == [("gpu-node-selector", "info")]


def test_startup_probe_rule():
    srv = m.GPUContainer(name="s", ports=[8000], liveness_probe=m.http_probe("/health", 8000))
    dep = m.obj("Deployment", "d", "ns", spec={"template": m.pod_template(m.pod_spec([srv], restart_policy=None)),
                                               "strategy": {"type": "Recreate"}})
    assert "startup-probe" in rules(lint.lint(dep), "warning")
    srv.startup_probe = m.http_probe("/health", 8000, period_s=10, failure_threshold=6)     # 60 s budget
    dep["spec"]["template"] = m.pod_template(m.pod_spec([srv], restart_policy=None))
    assert "startup-probe" in rules(lint.lint(dep, expected_load_s=300), "error")
    assert lint.probe_budget_s(srv.startup_probe) == 60


def test_restart_policy_rule_catches_deferred_crd_errors():
    t = m.pod_template(m.pod_spec([m.GPUContainer()]))                       # restartPolicy Never
    lws = m.leader_worker_set("l", "ns", size=2, worker_template=t)
    assert "restart-policy" in rules(lint.lint(lws), "error")                   # StatefulSets need Always
    job = m.job("j", "ns", m.pod_template(m.pod_spec([m.GPUContainer()], restart_policy=None)))
    assert "restart-policy" in rules(lint.lint(job), "error")                   # Jobs cannot be Always


def test_gpu_on_sidecar_rule():
    o = job_with({"name": "c", "image": "x", "resources": {"limits": {m.GPU: 1}}},
                 initContainers=[{"name": "s", "image": "y", "restartPolicy": "Always",
                                  "resources": {"limits": {m.GPU: 1}}}])
    assert "gpu-on-sidecar" in rules(lint.lint(o), "error")


def test_strands_gpus_on_a_4_gpu_node():
    # g2-standard-48: 47.81 allocatable CPU / 4 GPUs = 11.95 per GPU; a 16-CPU pod fits twice -> 2 GPUs idle
    c = m.GPUContainer(gpus=1, cpu="16", memory="40Gi")
    o = m.job("j", "ns", m.pod_template(m.pod_spec([c])))
    f = [x for x in lint.lint(o, machine="g2-standard-48") if x.rule == "strands-gpus"]
    assert f and "2 of 4 GPUs stay idle" in f[0].message


def test_shm_rule_for_multi_gpu_pods():
    o = m.job("j", "ns", m.pod_template(m.pod_spec([m.GPUContainer(gpus=2)])))
    assert "shm-size" in rules(lint.lint(o), "warning")
    o = m.job("j", "ns", m.pod_template(m.pod_spec([m.GPUContainer(gpus=2)], shm_size="1Gi")))
    assert "shm-size" not in rules(lint.lint(o))


def test_rollout_surge_rule():
    t = m.pod_template(m.pod_spec([m.GPUContainer()], restart_policy=None))
    dep = m.obj("Deployment", "d", "ns", spec={"template": t})
    assert "rollout-surge" in rules(lint.lint(dep), "info")
    dep["spec"]["strategy"] = {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1}}
    assert "rollout-surge" not in rules(lint.lint(dep))


def test_machines_allocatable_formula():
    assert round(machines.gke_reserved_cpu(48), 4) == 0.19          # 0.06 + 0.01 + 2x0.005 + 44x0.0025
    assert round(machines.gke_reserved_memory_gib(192), 4) == 10.6977  # 1 + 0.8 + 0.8 + 6.72 + 1.28 + 100 MiB
    a = machines.allocatable("g2-standard-4")
    assert (a.cpu, round(a.memory_gib, 4), a.gpus) == (3.92, 13.3023, 1)
    share = machines.per_gpu_share("g2-standard-48")
    assert (share.cpu, share.memory_gib) == (11.9525, 45.3256)
    assert machines.parse_memory_gib("64Mi") == 0.0625 and machines.parse_cpu("500m") == 0.5


def test_the_labs_own_manifests_are_lint_clean():
    intended = {("Job/no-toleration", "gpu-toleration")}       # the zoo's deliberate mistake
    seen = set()
    for f in sorted((ROOT / "deploy").rglob("*.yaml")):
        for o in m.load_all(f):
            for x in lint.lint(o, machine="g2-standard-4" if f.parent.name == "gke" else None):
                if x.severity != "info":
                    seen.add((f"{o['kind']}/{o['metadata']['name']}", x.rule))
    assert seen == intended


def test_toleration_rule_uses_real_matching():
    c = {"name": "c", "image": "x", "resources": {"requests": {"cpu": "1", "memory": "1Gi"},
                                                  "limits": {"memory": "1Gi", m.GPU: 1}}}
    sel = {m.GKE_ACCELERATOR: "nvidia-l4"}
    wrong_value = job_with(c, nodeSelector=sel,
                           tolerations=[{"key": m.GPU, "operator": "Equal", "value": "true", "effect": "NoSchedule"}])
    f = [x for x in lint.lint(wrong_value) if x.rule == "gpu-toleration"]
    assert f and "Equal needs the same value" in f[0].message        # Pending on nvidia.com/gpu=present
    wrong_effect = job_with(c, nodeSelector=sel, tolerations=[{"key": m.GPU, "operator": "Exists", "effect": "NoExecute"}])
    assert "gpu-toleration" in rules(lint.lint(wrong_effect), "warning")
    for ok in ({"key": m.GPU, "operator": "Equal", "value": "present", "effect": "NoSchedule"},
               {"key": m.GPU, "operator": "Exists"}, {"operator": "Exists"}):
        assert "gpu-toleration" not in rules(lint.lint(job_with(c, nodeSelector=sel, tolerations=[ok]))), ok


def test_shm_larger_than_the_memory_limit_is_flagged():
    small = m.job("j", "ns", m.pod_template(m.pod_spec([m.GPUContainer(gpus=2, memory="64Mi")], shm_size="256Mi")))
    f = [x for x in lint.lint(small) if x.rule == "shm-memory-limit"]
    assert f and f[0].severity == "warning" and "256Mi" in f[0].message
    big = m.job("j", "ns", m.pod_template(m.pod_spec([m.GPUContainer(gpus=2, memory="320Mi")], shm_size="256Mi")))
    assert "shm-memory-limit" not in rules(lint.lint(big))
    no_limit = small["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    no_limit.pop("memory")                                              # no limit: tmpfs is bounded by the node
    assert "shm-memory-limit" not in rules(lint.lint(small))


def test_gcsfuse_sidecar_requests_count_toward_the_bundle():
    # g2-standard-4 allocatable: 3.92 cpu, 13.30 GiB. 3.5 cpu fits; + the 500m sidecar does not.
    c = m.GPUContainer(gpus=1, cpu="3500m", memory="8Gi")
    plain = m.job("j", "ns", m.pod_template(m.pod_spec([c])))
    assert "strands-gpus" not in rules(lint.lint(plain, machine="g2-standard-4"))
    fuse = m.job("j", "ns", m.pod_template(m.pod_spec([c]), annotations={
        "gke-gcsfuse/volumes": "true", "gke-gcsfuse/cpu-request": "500m", "gke-gcsfuse/memory-request": "1Gi"}))
    f = [x for x in lint.lint(fuse, machine="g2-standard-4") if x.rule == "strands-gpus"]
    assert f and f[0].severity == "error" and "cpu 4" in f[0].message
