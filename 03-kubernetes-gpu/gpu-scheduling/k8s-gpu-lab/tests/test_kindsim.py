"""The predictor agrees with the answer key, and its pieces match Kueue / kube-scheduler rules."""
from collections import Counter

import pytest

from k8sgpu import kindsim, scenarios
from k8sgpu import manifests as m


@pytest.mark.parametrize("sid", sorted(scenarios.EXPECTED))
def test_predictions_match_the_answer_key(sid):
    outs = kindsim.predict_all_steps(sid)
    for step, expected in scenarios.EXPECTED[sid].items():
        assert kindsim.check_expectation(outs[step - 1], expected) == [], (sid, step)


def _leaves(caps: dict[tuple, int]) -> tuple[list[kindsim.Node], dict[str, int]]:
    """Nodes from {(block, subblock, host): free pods}."""
    nodes, cap = [], {}
    for (b, s, h), c in caps.items():
        n = kindsim.Node(h, {m.TOPOLOGY_BLOCK: b, m.TOPOLOGY_SUBBLOCK: s, m.TOPOLOGY_HOST: h, m.HOSTNAME: h}, [], 4)
        nodes.append(n)
        cap[h] = c
    return nodes, cap


LEVELS = m.GKE_TOPOLOGY_LEVELS


def test_best_fit_picks_the_smallest_domain_that_fits():
    nodes, cap = _leaves({("b", "s1", "h1"): 4, ("b", "s1", "h2"): 4, ("b", "s2", "h3"): 3, ("b", "s2", "h4"): 1})
    # subblocks: s1 = 8, s2 = 4 -> a 4-pod gang takes s2 (the tighter fit), leaving s1 whole
    got = kindsim.tas_assign(nodes, LEVELS, cap, 4, "required", m.TOPOLOGY_SUBBLOCK)
    assert got == {"h3": 3, "h4": 1}


def test_required_level_that_cannot_fit_reports_the_best_domain():
    nodes, cap = _leaves({("b", "s1", "h1"): 2, ("b", "s2", "h2"): 1})
    assert kindsim.tas_assign(nodes, LEVELS, cap, 4, "required", m.TOPOLOGY_SUBBLOCK) == (None, 2)


def test_preferred_climbs_a_level_then_spreads():
    nodes, cap = _leaves({("b1", "s1", "h1"): 2, ("b1", "s2", "h2"): 2, ("b2", "s3", "h3"): 1})
    got = kindsim.tas_assign(nodes, LEVELS, cap, 4, "preferred", m.TOPOLOGY_SUBBLOCK)
    assert got == {"h1": 2, "h2": 2}          # no subblock holds 4; block b1 does


def test_unconstrained_uses_least_free_capacity():
    nodes, cap = _leaves({("b", "s", "h1"): 4, ("b", "s", "h2"): 1, ("b", "s", "h3"): 2})
    assert kindsim.tas_assign(nodes, LEVELS, cap, 2, "unconstrained", m.HOSTNAME) == {"h3": 2}
    assert kindsim.tas_assign(nodes, LEVELS, cap, 1, "unconstrained", m.HOSTNAME) == {"h2": 1}


def test_fit_error_message_format():
    msg = kindsim.fit_error(3, Counter({"Insufficient nvidia.com/gpu": 2,
                                        "node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }": 1}),
                            {"cp": True, "a": False, "b": False})
    assert msg == ("0/3 nodes are available: 1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }, "
                   "2 Insufficient nvidia.com/gpu. preemption: 0/3 nodes are available: 1 Preemption is not helpful "
                   "for scheduling, 2 No preemption victims found for incoming pod.")


def test_quota_arithmetic_matches_kueue():
    sim = kindsim.new_sim()
    a = sim.cfg.cqs["team-a-cq"]
    assert sim.max_capacity(a) == 12                  # nominal 8 + min(borrowingLimit 4, team-b's 8)
    assert sim.available(a) == 12                     # nothing in use yet
    sim.apply(m.job("x", "team-a", scenarios.lab_template(4), queue="gpu-queue", parallelism=3))
    assert sim.usage("team-a-cq") == 12 and sim.available(a) == 0
    assert sim.available(sim.cfg.cqs["team-b-cq"]) == 4   # team-b: cohort has 16 - 12 left


def test_topology_file_matches_the_kind_config():
    nodes = kindsim.read_topology()
    workers = [n for n in nodes if "control-plane" not in n.name]
    config = (kindsim.KIND_DIR / "kind-config.yaml").read_text()
    assert config.count("- role: worker") == len(workers) == 5
    assert sum(n.gpus for n in nodes) == 16
    assert all(n.taints == [kindsim.GPU_TAINT] for n in nodes if n.gpus)


def test_kwok_fleet_shape():
    fleet = kindsim.kwok_nodes()
    assert len(fleet) == 32 and sum(n.gpus for n in fleet) == 256
    assert fleet[0].labels[m.TOPOLOGY_SUBBLOCK] == "kwok-b1-s1" and fleet[0].name == "kwok-b1-s1-h1"


def test_nodes_from_kubectl_json():
    items = [{"metadata": {"name": "n1", "labels": {m.HOSTNAME: "n1"}},
              "spec": {"taints": [{"key": m.GPU, "value": "present", "effect": "NoSchedule"}]},
              "status": {"allocatable": {m.GPU: "4"}}}]
    n = kindsim.nodes_from_k8s(items)[0]
    assert (n.name, n.gpus, n.taints[0]["value"]) == ("n1", 4, "present")


def test_quota_message_names_only_the_pod_sets_that_do_not_fit():
    # Kueue v0.19 assignFlavors: pod sets in order, assumed usage carried forward; Message() skips fits.
    sim = kindsim.new_sim()
    tmpl = scenarios.lab_template(4)
    two = m.jobset("two", "team-b", [m.replicated_job("head", tmpl), m.replicated_job("tail", tmpl)], queue="gpu-queue")
    sim.apply(m.job("fill", "team-b", scenarios.lab_template(4), parallelism=2, queue="gpu-queue"))   # 8 of 8
    sim.apply(two)                                                     # 4 borrowable: head fits, tail does not
    msg = sim.workloads["JobSet/team-b/two"].message
    assert msg == ("couldn't assign flavors to pod set tail: insufficient unused quota for nvidia.com/gpu "
                   "in flavor gpu-l4, 4 more needed")
    # grouped pod sets (podset-group-name) are assigned together and share one status: s3 lists both
    s3 = kindsim.predict("s3")["LeaderWorkerSet/team-b/llm-multihost#1"]["message"]
    assert s3.count("4 more needed") == 2 and "pod set leader" in s3 and "pod set worker" in s3


def test_victim_order_puts_evicting_workloads_first():
    sim = kindsim.new_sim()
    for i in range(2):
        sim.apply(m.job(f"low-{i}", "team-a", scenarios.lab_template(4), queue="gpu-queue", priority_class="low"))
    sim.apply(m.job("b", "team-b", scenarios.lab_template(4), parallelism=2, queue="gpu-queue"))
    older = sim.workloads["Job/team-a/low-0"]
    older.evicting = True                                              # evicted but still holding quota
    w = kindsim.Workload("Job/team-a/hi", "team-a", "gpu-queue", 1000, kindsim._podsets(
        m.job("hi", "team-a", scenarios.lab_template(4))), 99)
    victims = sim._find_victims(w, sim.cfg.cqs["team-a-cq"])
    assert victims == [older]                                          # not the newer low-1


def test_max_capacity_check_is_per_pod_set_group():
    # Kueue v0.19 fitsMaxCapacity runs on the group's summed request, with one status for every member.
    ann = {m.TAS_REQUIRED: m.TOPOLOGY_SUBBLOCK, m.TAS_GROUP: "g"}
    for tmpl, expect in ((scenarios.lab_template(4), ["tail: insufficient quota for nvidia.com/gpu in flavor "
                          "gpu-l4, previously considered podsets requests (8) + current podset request (8) > maximum capacity (12)"]),
                         (scenarios.lab_template(4, annotations=ann), [f"{n}: insufficient quota for nvidia.com/gpu in flavor "
                          "gpu-l4, previously considered podsets requests (0) + current podset request (16) > maximum capacity (12)"
                          for n in ("head", "tail")])):
        sim = kindsim.new_sim()
        sim.apply(m.jobset("big", "team-a", [m.replicated_job("head", tmpl, parallelism=2),
                                             m.replicated_job("tail", tmpl, parallelism=2)], queue="gpu-queue"))
        msg = sim.workloads["JobSet/team-a/big"].message
        assert msg == "; ".join(f"couldn't assign flavors to pod set {e}" for e in expect)
