"""The Pending-pod explainer: parsing, classification, and every fixture's verdict."""
import json

import pytest

from k8sgpu import pending

MSG = ("0/6 nodes are available: 1 node(s) didn't match Pod's node affinity/selector, "
       "1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }, 4 Insufficient nvidia.com/gpu. "
       "preemption: 0/6 nodes are available: 2 Preemption is not helpful for scheduling, "
       "4 No preemption victims found for incoming pod.")


def test_parse_fit_error_handles_dots_in_resource_names():
    fe = pending.parse_fit_error(MSG)
    assert fe.total_nodes == 6
    assert fe.reasons["Insufficient nvidia.com/gpu"] == 4
    assert fe.reasons["node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }"] == 1
    assert fe.preemption == {"Preemption is not helpful for scheduling": 2,
                             "No preemption victims found for incoming pod": 4}


def test_parse_prefilter_message():
    fe = pending.parse_fit_error("0/3 nodes are available: pod has unbound immediate PersistentVolumeClaims. "
                                 "preemption: 0/3 nodes are available: 3 Preemption is not helpful for scheduling.")
    assert fe.prefilter == "pod has unbound immediate PersistentVolumeClaims" and fe.reasons == {}


def test_classify_reason():
    assert pending.classify_reason("Insufficient nvidia.com/gpu")[0] == "insufficient-gpu"
    assert pending.classify_reason("node(s) had untolerated taint {nvidia.com/gpu: present}")[0] == "gpu-taint"
    assert pending.classify_reason("node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }")[1] is False
    assert pending.classify_reason("cannot allocate all claims")[0] == "dra"
    assert pending.classify_reason("node(s) had volume node affinity conflict")[0] == "volume"


def test_fragmentation_is_enough_in_total_but_not_on_one_node():
    assert pending.is_fragmented([1, 1, 1, 1], 2) is True
    assert pending.is_fragmented([2, 0, 0, 0], 2) is False
    assert pending.is_fragmented({"a": 1}, 2) is False


@pytest.mark.parametrize("name", pending.fixture_names())
def test_every_fixture_gets_its_expected_verdict(name):
    fx = pending.load_fixture(name)
    assert "illustrative" in fx["source"]            # fixtures never pose as measured output
    d = pending.diagnose(fx)
    assert (d.gate, d.category) == (fx["expect"]["gate"], fx["expect"]["category"]), str(d)
    assert d.fixes, "every diagnosis names a fix"


def test_enough_fixtures_to_cover_each_gate():
    gates = {pending.load_fixture(n)["expect"]["gate"] for n in pending.fixture_names()}
    assert gates == {"kueue", "kube-scheduler", "cluster-autoscaler", "kubelet"}


class _FakeKubectl:
    """Answers the four reads diagnose_live makes, from the lws-group-gated fixture."""

    def __init__(self, fx):
        self.fx, self.asked = fx, []

    def run(self, *args, quiet=False, check=True):
        self.asked.append(args)
        if args[:2] == ("get", "pod"):
            return json.dumps(self.fx["pod"])
        if args[:2] == ("get", "workloads.kueue.x-k8s.io"):
            return json.dumps(self.fx["workload"]) if args[2] == self.fx["workload"]["metadata"]["name"] else ""
        return ""

    def get_json(self, what, *rest):
        return {"items": self.fx.get("events", []) if what == "events" else []}


def test_live_diagnosis_finds_the_workload_of_a_gated_lws_pod():
    # LWS / pod-group pods name their Workload with kueue.x-k8s.io/prebuilt-workload-name (an annotation
    # while WorkloadIdentifierAnnotations is on, the v0.19 default; a label otherwise) - not kueue.x-k8s.io/workload
    fx = pending.load_fixture("lws-group-gated")
    ann = fx["pod"]["metadata"]["annotations"]
    assert pending.WORKLOAD_ANNOTATION not in ann and pending.PREBUILT_WORKLOAD in ann
    k = _FakeKubectl(fx)
    d = pending.diagnose_live(fx["pod"]["metadata"]["name"], "team-b", kubectl=k)
    assert (d.gate, d.category) == ("kueue", "waiting-for-quota"), str(d)
    as_label = {"metadata": {"labels": {pending.PREBUILT_WORKLOAD: "wl-1"}}}
    assert pending.workload_name_of(as_label) == "wl-1"


def test_cli_labels_fixture_output(capsys):
    from k8sgpu.__main__ import main
    assert main(["pending", "--fixture", "gke-stockout"]) == 0
    first = capsys.readouterr().out.splitlines()[0]
    assert first.startswith("(fixture: ") and "sample output in the documented format (illustrative)" in first


def test_fixture_sources_are_not_nested():
    for name in pending.fixture_names():
        src = pending.load_fixture(name)["source"]
        assert src.count("in the documented format") == 1 and src.count("(illustrative)") == 1, name
        assert "simulated by k8sgpu.kindsim" in src or "written by hand" in src, name
