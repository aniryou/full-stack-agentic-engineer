"""Kubernetes assets are correct by construction: generated files are current, core kinds validate strictly
for 1.34, CRD objects use only fields the pinned agent-sandbox CRDs define, the admission predictor agrees
with the CEL it renders, and the policy's invariants hold in every rendered pod."""
import copy
import ipaddress
import json
import re
from pathlib import Path

import pytest

from sandboxlab.k8s import admission as A
from sandboxlab.k8s import manifests as m
from sandboxlab.k8s import policy as P
from sandboxlab.k8s import render

ROOT = Path(__file__).resolve().parents[1]
CRDS = json.loads((ROOT / "tests/data/agent_sandbox_crds.json").read_text())["kinds"]
KIND, GKE = P.SandboxPolicy.kind(), P.SandboxPolicy.gke()


def all_objects():
    for target in ("kind", "gke"):
        yield from render.objects(target)


def sandbox_pod_specs():
    for f, o in all_objects():
        for _, t in m.iter_pod_templates(o):
            if "rejected" not in f and (o["metadata"].get("namespace") == P.SANDBOX_NS) and "sandbox" in (t.get("metadata", {}).get("labels") or {}).get(
                    "app.kubernetes.io/name", ""):
                yield f, t["spec"]


def test_generated_files_are_current_and_have_no_orphans(tmp_path):
    assert render.stale() == [], "run: python3 tools/render_manifests.py"
    for rel, text in render.rendered_files().items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    (tmp_path / "deploy/gke/99-old.yaml").write_text("kind: Pod\n")
    assert render.orphans(tmp_path) == ["deploy/gke/99-old.yaml"]


def test_every_object_uses_a_pinned_api_version():
    objs = list(all_objects())
    assert len(objs) > 60
    for f, o in objs:
        assert o["apiVersion"] == m.API_VERSIONS[o["kind"]], f


def test_core_kinds_and_crd_pod_templates_validate_for_1_34():
    kv = pytest.importorskip("kubernetes_validate")
    n = 0
    for f, o in all_objects():
        if o["kind"] in m.CORE_KINDS:
            kv.validate(o, "1.34.0", strict=True)
            n += 1
        else:
            for _, t in m.iter_pod_templates(o):
                meta = {"name": "template", **(t.get("metadata") or {})}
                kv.validate({"apiVersion": "v1", "kind": "Pod", "metadata": meta, "spec": t["spec"]}, "1.34.0", strict=True)
                n += 1
    assert n >= 60


def test_agent_sandbox_objects_match_the_pinned_crds():
    seen = 0
    for f, o in all_objects():
        key = f"{o['apiVersion']}/{o['kind']}"
        if key not in CRDS:
            continue
        seen += 1
        spec, crd = o["spec"], CRDS[key]
        assert set(spec) <= set(crd["fields"]), (f, set(spec) - set(crd["fields"]))
        assert set(crd["required"]) <= set(spec), f
        for k, v in spec.items():
            if crd["fields"][k]["enum"]:
                assert v in crd["fields"][k]["enum"], (f, k, v)
            if isinstance(v, dict) and crd["fields"][k]["properties"]:
                assert set(v) <= set(crd["fields"][k]["properties"]), (f, k)
    assert seen == 3


def test_every_sandbox_pod_has_no_ambient_authority():
    specs = list(sandbox_pod_specs())
    assert len(specs) == 6          # Job + warm pool on kind and on GKE, kind's egress check, the agent-sandbox template
    for f, s in specs:
        assert s["automountServiceAccountToken"] is False and s["enableServiceLinks"] is False, f
        assert s["serviceAccountName"] == "sandbox-exec" and s["dnsPolicy"] == "None", f
        assert s["securityContext"]["runAsUser"] == 65534 and s["securityContext"]["runAsNonRoot"] is True
        assert s["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
        c = s["containers"][0]
        sc = c["securityContext"]
        assert sc["allowPrivilegeEscalation"] is False and sc["readOnlyRootFilesystem"] is True
        assert sc["capabilities"] == {"drop": ["ALL"]} and "appArmorProfile" not in sc
        assert c["resources"]["requests"] == c["resources"]["limits"]
        assert all("sizeLimit" in v["emptyDir"] for v in s["volumes"] if "emptyDir" in v)
        assert not any("secret" in v for v in s["volumes"]) and "envFrom" not in c
        assert A.pss_restricted(s) == [], f


def test_runtime_class_per_target_and_proxy_ip_inside_the_service_range():
    tf = (ROOT / "deploy/gcp/terraform/variables.tf").read_text()
    services = re.search(r'variable "services_cidr".*?default\s*=\s*"([^"]+)"', tf, re.S).group(1)
    kind_cfg = m.load_all(ROOT / "deploy/kind/kind-config.yaml")[0]
    for pol, cidr in ((KIND, kind_cfg["networking"]["serviceSubnet"]), (GKE, services)):
        for ip in (pol.proxy_cluster_ip, pol.stub_cluster_ip, pol.dns_service_ip):
            assert ipaddress.ip_address(ip) in ipaddress.ip_network(cidr)
    assert {s["runtimeClassName"] for f, s in sandbox_pod_specs() if f.startswith("deploy/kind")} == {"sandbox-runc"}
    assert {s["runtimeClassName"] for f, s in sandbox_pod_specs() if f.startswith("deploy/gke")} == {"gvisor"}
    rc = m.load_all(ROOT / "deploy/kind/manifests/05-runtimeclass.yaml")[0]
    assert rc["handler"] == "runc" and rc["scheduling"]["tolerations"][0]["key"] == "sandboxlab/pool"
    assert kind_cfg["nodes"][0]["kubeadmConfigPatches"][0].endswith(f"podPidsLimit: {KIND.pod_pids_limit}\n")


def test_network_policies_default_deny_and_one_way_out():
    pols = {(o["metadata"]["namespace"], o["metadata"]["name"]): o["spec"] for f, o in render.objects("kind")
            if o["kind"] == "NetworkPolicy"}
    for ns in (P.SANDBOX_NS, P.EGRESS_NS, P.UPSTREAM_NS):
        d = pols[(ns, "default-deny")]
        assert d["podSelector"] == {} and d["policyTypes"] == ["Ingress", "Egress"] and "egress" not in d
    out = pols[(P.SANDBOX_NS, "egress-to-proxy-only")]["egress"]
    assert len(out) == 1 and out[0]["ports"] == [{"protocol": "TCP", "port": 8080}]
    peer = out[0]["to"][0]                                                    # ONE entry: namespace AND pod selector
    assert peer["namespaceSelector"]["matchLabels"] == {m.NAME_LABEL: P.EGRESS_NS}
    assert peer["podSelector"]["matchLabels"] == {"app": "egress-proxy"}
    assert all(p["policyTypes"] for p in pols.values())
    dns = [r for r in pols[(P.EGRESS_NS, "proxy-egress")]["egress"] if {"protocol": "UDP", "port": 53} in r["ports"]]
    assert dns and not any(53 in [p["port"] for p in r.get("ports", [])] for r in out)   # DNS for the proxy only


def test_jobs_have_budgets_and_the_name_is_the_idempotency_key():
    job = P.run_code_job(KIND, "print(1)", "abc123", "turn1:step1:call0:x")
    assert job["metadata"]["name"] == "run-abc123" and job["spec"]["backoffLimit"] == 0
    assert job["spec"]["activeDeadlineSeconds"] == 120 and job["spec"]["ttlSecondsAfterFinished"] == 300
    assert job["metadata"]["annotations"][P.IDEMPOTENCY_ANNOTATION] == "turn1:step1:call0:x"
    from sandboxlab.k8s.runner import execution_id
    assert execution_id("k") == execution_id("k") != execution_id("k2") and len(execution_id(None)) == 16


RCS = {"sandbox-runc": P.runtime_class(KIND)}


def test_admission_of_the_examples():
    wl = {name: objs for name, (_, objs) in P.workloads(KIND).items()}
    ok = A.admit(wl["10-run-code-job.yaml"][0], runtime_classes=RCS, allowed_runtime_classes=["sandbox-runc"],
                 node_handlers={"runc"})
    assert ok.admitted and ok.spec["nodeSelector"] == {"sandboxlab/pool": "sandbox"} and ok.will_fail is None
    bad = A.admit(wl["90-rejected-pod.yaml"][0], runtime_classes=RCS, allowed_runtime_classes=["sandbox-runc"])
    assert not bad.admitted and {v.step for v in bad.violations} == {"pss", "vap"}
    job = A.admit(wl["91-rejected-job.yaml"][0], runtime_classes=RCS, allowed_runtime_classes=["sandbox-runc"])
    assert {v.rule for v in job.violations} == {"job-deadline", "job-no-retries", "job-ttl"}   # backoffLimit defaults to 6
    egress = wl["93-egress-must-fail.yaml"][0]
    chk = A.admit(egress, runtime_classes=RCS, allowed_runtime_classes=["sandbox-runc"], node_handlers={"runc"})
    assert chk.admitted and chk.will_fail is None                              # a conforming pod: only the network says no
    code = next(e["value"] for e in egress["spec"]["template"]["spec"]["containers"][0]["env"] if e["name"] == "SANDBOX_CODE")
    assert KIND.stub_cluster_ip in code and KIND.dns_service_ip in code and "CONNECTED" in code
    stub_svc = next(o for f, o in render.objects("kind") if o["kind"] == "Service" and o["metadata"]["name"] == "api-stub")
    assert stub_svc["spec"]["clusterIP"] == KIND.stub_cluster_ip
    rc, pod = P.gvisor_in_kind()[1:]
    g = A.admit(pod, runtime_classes={"gvisor": rc}, allowed_runtime_classes=["gvisor"], node_handlers={"runc"})
    assert g.admitted and "runsc" in g.will_fail                                # admitted, then Failed


def test_runtime_class_merge_conflict_overhead_and_quota():
    spec = {"nodeSelector": {"sandboxlab/pool": "system"}, "containers": [
        {"name": "c", "resources": {"requests": {"cpu": "500m", "memory": "320Mi"}}}]}
    rc = m.runtime_class("kata", "kata-qemu-runtime-rs", node_selector={"sandboxlab/pool": "sandbox"},
                         overhead={"cpu": "250m", "memory": "320Mi"})
    merged, errs = A.apply_runtime_class(spec, rc)
    assert errs and "conflict" in errs[0]
    merged, errs = A.apply_runtime_class({"containers": spec["containers"]}, rc)
    assert not errs and A.effective_requests(merged) == {"cpu": 0.75, "memory": 640 * 2**20}
    quota = m.load_all(ROOT / "deploy/kind/manifests/20-quota.yaml")[0]["spec"]["hard"]
    assert A.quantity(quota["limits.memory"]) == 20 * 320 * 2**20              # 20 sandboxes of 320 Mi
    assert A.quantity(quota["requests.memory"]) // (640 * 2**20) == 10         # the same quota holds 10 kata pods


def test_vap_structure_and_binding():
    objs = P.admission_policies(KIND)
    pod_vap, pod_bind, job_vap, job_bind = objs
    assert pod_vap["spec"]["failurePolicy"] == "Fail"
    assert pod_vap["spec"]["matchConstraints"]["resourceRules"][0]["resources"] == ["pods", "pods/ephemeralcontainers"]
    assert pod_bind["spec"]["validationActions"] == ["Deny"] and pod_bind["spec"]["policyName"] == pod_vap["metadata"]["name"]
    assert job_vap["spec"]["matchConstraints"]["resourceRules"][0]["apiGroups"] == ["batch"]
    with pytest.raises(ValueError):
        m.vap_binding("x", "y", namespaces=["a"], actions=("Deny", "Warn"))


def _cel_eval(obj, rules):
    celpy = pytest.importorskip("celpy")
    env = celpy.Environment()
    act = {"object": celpy.json_to_cel(obj)}
    if obj["kind"] == "Pod":
        act["variables"] = celpy.celtypes.MapType({celpy.celtypes.StringType(v["name"]):
                                                  env.program(env.compile(v["expression"])).evaluate(act)
                                                  for v in A.CEL_VARIABLES})
    res = "pods" if obj["kind"] == "Pod" else "jobs"
    return {r.name: bool(env.program(env.compile(r.cel)).evaluate(act)) for r in rules if r.resource == res}


def test_cel_and_python_agree_on_every_rule():
    good = P.run_code_job(KIND, "print(1)", "x")
    pod = {"kind": "Pod", "metadata": {"name": "p"}, "spec": copy.deepcopy(good["spec"]["template"]["spec"])}
    eph = copy.deepcopy(pod)
    eph["spec"]["ephemeralContainers"] = [{"name": "dbg", "image": "busybox", "securityContext": {"privileged": True}}]
    sec = copy.deepcopy(pod)
    sec["spec"]["containers"][0]["env"].append({"name": "K", "valueFrom": {"secretKeyRef": {"name": "s", "key": "k"}}})
    bad_job = P.rejected_job(KIND)
    bad_job["spec"]["backoffLimit"] = 6                                          # what the API server's defaulting sends
    rules = A.rules(["sandbox-runc"])
    prm = {"runtime_classes": ["sandbox-runc"], "max_deadline_s": 600}
    for obj in (pod, P.rejected_pod(KIND), eph, sec, good, bad_job):
        cel = _cel_eval(obj, rules)
        py = {r.name: r.check(obj, prm) for r in rules if r.name in cel}
        assert cel == py, obj["metadata"]["name"]
    assert not all(_cel_eval(eph, rules).values()) and all(_cel_eval(pod, rules).values())
