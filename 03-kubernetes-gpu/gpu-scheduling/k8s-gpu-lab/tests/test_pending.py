"""The Pending-pod explainer: parsing, classification, and every fixture's verdict."""
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
