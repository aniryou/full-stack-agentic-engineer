"""The rendered Kubernetes manifests: valid against 1.34, semantically admissible, and carrying the
load-bearing security fields."""
import re

import pytest

from sandboxcore import SandboxPolicy, render_yaml

yaml = pytest.importorskip("yaml")

KINDS = ["Namespace", "RuntimeClass", "NetworkPolicy", "NetworkPolicy", "Service", "ResourceQuota",
         "LimitRange", "Pod", "Job", "ValidatingAdmissionPolicy", "ValidatingAdmissionPolicyBinding"]


def objs(**kw):
    return SandboxPolicy(name="test", egress_allowlist=("api.github.com",), **kw).render_k8s()


def pod_specs(objects):
    for o in objects:
        if o["kind"] == "Pod":
            yield o["spec"]
        elif o["kind"] == "Job":
            yield o["spec"]["template"]["spec"]


def qty(v: str) -> float:
    """A Kubernetes quantity (the forms these manifests use) in base units."""
    m = re.fullmatch(r"([0-9.]+)(m|Ki|Mi|Gi)?", str(v))
    n, unit = float(m.group(1)), m.group(2)
    return n * {None: 1, "m": 1e-3, "Ki": 2**10, "Mi": 2**20, "Gi": 2**30}[unit]


def test_renders_the_expected_set_in_order():
    assert [o["kind"] for o in objs()] == KINDS
    # on GKE the platform creates RuntimeClass "gvisor" itself (verify), so it is not rendered
    assert "RuntimeClass" not in [o["kind"] for o in objs(render_runtime_class=False)]


def test_runtime_class_names_the_gvisor_handler():
    rc = next(o for o in objs() if o["kind"] == "RuntimeClass")
    assert rc["metadata"]["name"] == "gvisor" and rc["handler"] == "runsc"
    assert all(spec["runtimeClassName"] == "gvisor" for spec in pod_specs(objs()))


def test_namespace_enforces_restricted_pod_security():
    labels = objs()[0]["metadata"]["labels"]
    assert labels["pod-security.kubernetes.io/enforce"] == "restricted"
    assert labels["pod-security.kubernetes.io/enforce-version"] == "v1.34"


def test_job_is_bounded_non_retrying_and_restricted():
    job = next(o for o in objs() if o["kind"] == "Job")
    spec = job["spec"]
    assert spec["backoffLimit"] == 0                    # non-idempotent code must not re-run
    assert spec["ttlSecondsAfterFinished"] == 300
    pod = spec["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["automountServiceAccountToken"] is False  # no cloud creds via the KSA token
    assert pod["securityContext"]["runAsNonRoot"] is True
    ctr = pod["containers"][0]["securityContext"]
    assert ctr["allowPrivilegeEscalation"] is False
    assert ctr["readOnlyRootFilesystem"] is True
    assert ctr["capabilities"]["drop"] == ["ALL"]
    assert ctr["seccompProfile"]["type"] == "RuntimeDefault"


def test_deadlines_leave_room_for_startup_and_the_wall_budget_is_enforced_in_the_pod():
    pol = SandboxPolicy()
    job = pol.job()
    # the Job deadline counts scheduling + scale-up + image pull, so it is NOT just the wall budget
    assert job["spec"]["activeDeadlineSeconds"] == pol.startup_allowance_s + int(pol.max_budgets.wall_s)
    assert job["spec"]["activeDeadlineSeconds"] >= 60
    assert "activeDeadlineSeconds" not in job["spec"]["template"]["spec"]   # no second, tighter clock
    cmd = job["spec"]["template"]["spec"]["containers"][0]["command"]
    assert cmd[:4] == ["timeout", "-s", "KILL", str(int(pol.max_budgets.wall_s))]
    assert pol.pod()["spec"]["activeDeadlineSeconds"] >= pol.startup_allowance_s


def test_every_request_is_at_most_its_limit():
    # kubernetes-validate checks the schema only; the API server also rejects request > limit.
    for spec in pod_specs(objs()):
        for c in spec["containers"]:
            req, lim = c["resources"]["requests"], c["resources"]["limits"]
            for k, v in req.items():
                if k in lim:
                    assert qty(v) <= qty(lim[k]), f"{c['name']}: {k} request {v} > limit {lim[k]}"
            # the workspace emptyDir counts toward the container's ephemeral storage, so it must fit
            work = next(v for v in spec["volumes"] if v["name"] == "work")["emptyDir"]["sizeLimit"]
            assert qty(work) < qty(lim["ephemeral-storage"])
    lr = next(o for o in objs() if o["kind"] == "LimitRange")["spec"]["limits"][0]
    for k, v in lr["defaultRequest"].items():
        assert qty(v) <= qty(lr["default"][k])


def test_egress_opens_only_the_proxy_and_not_dns():
    deny = next(o for o in objs() if o["metadata"]["name"] == "sandbox-default-deny-egress")
    assert deny["spec"]["policyTypes"] == ["Egress"] and "egress" not in deny["spec"]
    allow = next(o for o in objs() if o["metadata"]["name"] == "sandbox-egress-to-proxy")
    rules = allow["spec"]["egress"]
    assert len(rules) == 1
    assert rules[0]["to"][0]["podSelector"]["matchLabels"]["app"] == "egress-proxy"
    assert all(p["port"] != 53 for r in rules for p in r.get("ports", []))   # no DNS-exfiltration channel
    # ...so the pod finds the proxy without DNS: hostAliases -> the Service's pinned ClusterIP
    svc = next(o for o in objs() if o["kind"] == "Service")
    for spec in pod_specs(objs()):
        assert spec["dnsPolicy"] == "None"
        assert spec["hostAliases"] == [{"ip": svc["spec"]["clusterIP"], "hostnames": ["egress-proxy"]}]
        env = {e["name"] for e in spec["containers"][0]["env"]}
        assert "HTTPS_PROXY" not in env        # the proxy refuses CONNECT; don't advertise what fails


def test_emptydir_volumes_are_sized():
    for spec in pod_specs(objs()):
        for v in spec["volumes"]:
            assert "sizeLimit" in v["emptyDir"]             # unbounded emptyDir can fill the node / RAM


def test_admission_policy_denies_non_conforming_pods():
    binding = next(o for o in objs() if o["kind"] == "ValidatingAdmissionPolicyBinding")
    assert binding["spec"]["validationActions"] == ["Deny"]   # Deny + Warn together is invalid
    policy = next(o for o in objs() if o["kind"] == "ValidatingAdmissionPolicy")
    exprs = " ".join(v["expression"] for v in policy["spec"]["validations"])
    assert "runtimeClassName" in exprs and "runAsNonRoot" in exprs
    assert "allowPrivilegeEscalation" in exprs


def test_yaml_round_trips():
    text = render_yaml(SandboxPolicy(name="rt", egress_allowlist=("a.example",)))
    loaded = [d for d in yaml.safe_load_all(text) if d]
    assert [d["kind"] for d in loaded] == KINDS


def test_all_manifests_validate_against_k8s_1_34():
    kv = pytest.importorskip("kubernetes_validate")
    for o in objs():
        kv.validate(o, "1.34.0", strict=True)
