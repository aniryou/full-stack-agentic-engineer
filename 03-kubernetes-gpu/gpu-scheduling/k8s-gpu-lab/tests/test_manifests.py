"""Builders produce exactly the fields that matter, with the API versions pinned in FACTS."""
import pytest

from k8sgpu import manifests as m


def test_api_versions_match_the_pinned_releases():
    assert m.API_VERSIONS["JobSet"] == "jobset.x-k8s.io/v1alpha2"
    assert m.API_VERSIONS["LeaderWorkerSet"] == "leaderworkerset.x-k8s.io/v1"
    assert {m.API_VERSIONS[k] for k in ("ClusterQueue", "LocalQueue", "ResourceFlavor", "Topology",
                                        "WorkloadPriorityClass", "AdmissionCheck",
                                        "ProvisioningRequestConfig")} == {"kueue.x-k8s.io/v1beta2"}
    assert m.API_VERSIONS["ResourceClaimTemplate"] == "resource.k8s.io/v1"      # DRA GA in 1.34
    assert m.API_VERSIONS["ComputeClass"] == "cloud.google.com/v1"


def test_gpu_container_sets_integer_limit_equal_to_request():
    c = m.GPUContainer(gpus=2, cpu="4", memory="16Gi").to_dict()
    assert c["resources"]["limits"][m.GPU] == 2 == c["resources"]["requests"][m.GPU]
    assert "cpu" not in c["resources"]["limits"]          # no CPU limit on a GPU feeder
    with pytest.raises(ValueError):
        m.GPUContainer(gpus=0.5).to_dict()


def test_pod_spec_adds_toleration_selector_and_shm():
    spec = m.pod_spec([m.GPUContainer(gpus=4)], shm_size="2Gi")
    assert m.GPU_TOLERATION in spec["tolerations"]
    assert spec["nodeSelector"] == {m.GKE_ACCELERATOR: "nvidia-l4"}
    assert {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}} in spec["volumes"]
    assert spec["containers"][0]["volumeMounts"] == [{"name": "dshm", "mountPath": "/dev/shm"}]


def test_startup_probe_budget_covers_the_load_with_margin():
    p = m.startup_probe_for(300, period_s=10)          # 300 s x 1.5 / 10 s = 45 tries
    assert p["failureThreshold"] == 45 and p["periodSeconds"] == 10


def test_topology_annotations_are_mutually_exclusive():
    assert m.topology_annotations(required=m.TOPOLOGY_HOST, group="g") == {
        m.TAS_REQUIRED: m.TOPOLOGY_HOST, m.TAS_GROUP: "g"}
    with pytest.raises(ValueError):
        m.topology_annotations(required="a", preferred="b")


def test_cluster_queue_uses_v1beta2_field_names():
    cq = m.cluster_queue("q", cohort="c", namespaces=["team-a"],
                         flavors={"f": {"cpu": 8, m.GPU: m.quota(8, borrowing_limit=4)}},
                         admission_checks=[("dws", ["f"])])
    assert cq["spec"]["cohortName"] == "c" and "cohort" not in cq["spec"]
    assert cq["spec"]["admissionChecksStrategy"] == {"admissionChecks": [{"name": "dws", "onFlavors": ["f"]}]}
    rg = cq["spec"]["resourceGroups"][0]
    assert rg["coveredResources"] == ["cpu", m.GPU]
    assert rg["flavors"][0]["resources"][1] == {"name": m.GPU, "nominalQuota": 8, "borrowingLimit": 4}


def test_cluster_queue_validations_mirror_kueue():
    with pytest.raises(ValueError):   # borrowingLimit must be nil without a cohort (CRD CEL rule)
        m.cluster_queue("q", flavors={"f": {m.GPU: m.quota(8, borrowing_limit=4)}})
    with pytest.raises(ValueError):   # every flavor lists the same resources in the same order
        m.cluster_queue("q", flavors={"a": {"cpu": 1, m.GPU: 1}, "b": {m.GPU: 1, "cpu": 1}})
    assert m.cluster_queue("q", flavors={"f": {m.GPU: 1}})["spec"]["namespaceSelector"] == {}   # {} = all


def test_topology_and_tas_flavor_rules():
    with pytest.raises(ValueError):
        m.topology("t", [m.HOSTNAME, m.TOPOLOGY_BLOCK])     # hostname only as the lowest level
    with pytest.raises(ValueError):
        m.resource_flavor("f", topology_name="t")           # TAS flavor needs nodeLabels
    f = m.resource_flavor("f", node_labels={m.GKE_ACCELERATOR: "nvidia-l4"}, topology_name="t",
                          tolerations=[m.GPU_TOLERATION])
    assert f["spec"]["topologyName"] == "t" and f["spec"]["tolerations"] == [m.GPU_TOLERATION]


def test_jobset_and_lws_shapes():
    t = m.pod_template(m.pod_spec([m.GPUContainer(gpus=1)]))
    js = m.jobset("g", "ns", [m.replicated_job("w", t, parallelism=4)], queue="q")
    rj = js["spec"]["replicatedJobs"][0]
    assert rj["template"]["spec"]["completionMode"] == "Indexed" and rj["template"]["spec"]["parallelism"] == 4
    assert js["metadata"]["labels"] == {m.QUEUE_LABEL: "q"}
    lws = m.leader_worker_set("llm", "ns", size=2, worker_template=t, queue="q")
    assert lws["spec"]["leaderWorkerTemplate"]["size"] == 2
    with pytest.raises(ValueError):
        m.leader_worker_set("x" * 40, "ns", size=2, worker_template=t)


def test_iter_pod_templates_finds_every_template():
    t = m.pod_template(m.pod_spec([m.GPUContainer()]))
    js = m.jobset("g", "ns", [m.replicated_job("a", t), m.replicated_job("b", t)])
    lws = m.leader_worker_set("l", "ns", size=2, worker_template=t, leader_template=t)
    assert [p for p, _ in m.iter_pod_templates(js)] == ["spec.replicatedJobs[0].template.spec.template",
                                                        "spec.replicatedJobs[1].template.spec.template"]
    assert len(list(m.iter_pod_templates(lws))) == 2


def test_dra_claim_template_matches_resource_v1():
    rct = m.resource_claim_template("one-l4", "default", cel=['device.attributes["gpu.nvidia.com"].productName == "NVIDIA L4"'])
    req = rct["spec"]["spec"]["devices"]["requests"][0]
    assert req["name"] == "gpu" and req["exactly"]["deviceClassName"] == "gpu.nvidia.com"
    assert req["exactly"]["allocationMode"] == "ExactCount" and req["exactly"]["count"] == 1
    assert "cel" in req["exactly"]["selectors"][0]


def test_compute_class_rungs():
    cc = m.compute_class("c", [m.compute_class_priority(machine_type="g2-standard-4", gpu_type="nvidia-l4", spot=True),
                               m.compute_class_priority(machine_type="g2-standard-4", gpu_type="nvidia-l4",
                                                        flex_start=True, node_recycling_lead_s=3600)])
    assert cc["spec"]["priorities"][0] == {"machineType": "g2-standard-4", "gpu": {"type": "nvidia-l4", "count": 1}, "spot": True}
    assert cc["spec"]["priorities"][1]["flexStart"] == {"enabled": True, "nodeRecycling": {"leadTimeSeconds": 3600}}
    assert cc["spec"]["nodePoolAutoCreation"] == {"enabled": True} and cc["spec"]["whenUnsatisfiable"] == "DoNotScaleUp"


def test_yaml_round_trip_keeps_order_and_header():
    o = m.job("j", "ns", m.pod_template(m.pod_spec([m.GPUContainer()])), queue="q")
    text = m.to_yaml(o, o, header="two jobs")
    assert text.startswith("# two jobs\napiVersion: batch/v1\nkind: Job\n")
    assert m.loads_all(text) == [o, o]


def test_toleration_matching_is_kubernetes_rule():
    # k8s.io/api core/v1 Toleration.ToleratesTaint (v1.34): effect, then key, then Exists / Equal(value)
    taint = m.GPU_TAINT                                                    # nvidia.com/gpu=present:NoSchedule
    assert m.toleration_tolerates({"key": m.GPU, "operator": "Exists", "effect": "NoSchedule"}, taint)
    assert m.toleration_tolerates({"operator": "Exists"}, taint)           # empty key + Exists = everything
    assert m.toleration_tolerates({"key": m.GPU, "value": "present"}, taint)   # operator defaults to Equal
    assert not m.toleration_tolerates({"key": m.GPU, "operator": "Equal", "value": "true"}, taint)
    assert not m.toleration_tolerates({"key": m.GPU, "operator": "Equal"}, taint)   # Equal with no value
    assert not m.toleration_tolerates({"key": m.GPU, "operator": "Exists", "effect": "NoExecute"}, taint)
    assert not m.toleration_tolerates({"key": "other", "operator": "Exists"}, taint)


def test_compute_class_gpu_driver_and_pod_runtime_class():
    rung = m.compute_class_priority(machine_type="g2-standard-4", gpu_type="nvidia-l4", gpu_driver_version="latest")
    assert rung["gpu"] == {"type": "nvidia-l4", "count": 1, "driverVersion": "latest"}
    spec = m.pod_spec([m.GPUContainer()], runtime_class="nvidia", accelerator=None,
                      affinity={"nodeAffinity": {}})
    assert spec["runtimeClassName"] == "nvidia" and spec["affinity"] == {"nodeAffinity": {}}
