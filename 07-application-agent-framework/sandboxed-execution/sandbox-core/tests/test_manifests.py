"""The rendered Kubernetes manifests: valid against 1.34, and carrying the load-bearing security fields."""
import pytest

from sandboxcore import SandboxPolicy, render_yaml

yaml = pytest.importorskip("yaml")


def objs():
    return SandboxPolicy(name="test", egress_allowlist=("api.github.com",)).render_k8s()


def test_renders_the_expected_set_in_order():
    kinds = [o["kind"] for o in objs()]
    assert kinds == ["Namespace", "NetworkPolicy", "NetworkPolicy", "ResourceQuota",
                     "LimitRange", "Job", "ValidatingAdmissionPolicy", "ValidatingAdmissionPolicyBinding"]


def test_namespace_enforces_restricted_pod_security():
    ns = objs()[0]
    labels = ns["metadata"]["labels"]
    assert labels["pod-security.kubernetes.io/enforce"] == "restricted"
    assert labels["pod-security.kubernetes.io/enforce-version"] == "v1.34"


def test_job_is_bounded_non_retrying_and_restricted():
    job = next(o for o in objs() if o["kind"] == "Job")
    spec = job["spec"]
    assert spec["backoffLimit"] == 0                    # non-idempotent code must not re-run
    assert spec["ttlSecondsAfterFinished"] == 300
    assert spec["activeDeadlineSeconds"] > 0
    pod = spec["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["automountServiceAccountToken"] is False  # no cloud creds via the KSA token
    assert pod["runtimeClassName"] == "gvisor"
    assert pod["securityContext"]["runAsNonRoot"] is True
    ctr = pod["containers"][0]["securityContext"]
    assert ctr["allowPrivilegeEscalation"] is False
    assert ctr["readOnlyRootFilesystem"] is True
    assert ctr["capabilities"]["drop"] == ["ALL"]
    assert ctr["seccompProfile"]["type"] == "RuntimeDefault"


def test_default_deny_egress_then_open_only_the_proxy_and_dns():
    deny = next(o for o in objs() if o["metadata"]["name"] == "sandbox-default-deny-egress")
    assert deny["spec"]["policyTypes"] == ["Egress"] and "egress" not in deny["spec"]
    allow = next(o for o in objs() if o["metadata"]["name"] == "sandbox-egress-to-proxy")
    rules = allow["spec"]["egress"]
    # first rule reaches only the proxy pods; DNS is re-opened explicitly (default-deny also blocks DNS)
    assert rules[0]["to"][0]["podSelector"]["matchLabels"]["app"] == "egress-proxy"
    assert any(p["port"] == 53 for r in rules for p in r.get("ports", []))


def test_emptydir_volumes_are_sized():
    job = next(o for o in objs() if o["kind"] == "Job")
    vols = job["spec"]["template"]["spec"]["volumes"]
    for v in vols:
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
    assert len(loaded) == 8 and loaded[0]["kind"] == "Namespace"


def test_all_manifests_validate_against_k8s_1_34():
    kv = pytest.importorskip("kubernetes_validate")
    for o in objs():
        kv.validate(o, "1.34.0", strict=True)
