"""How a GPU pod is admitted, filtered, scored and preempted - pinned to upstream behaviour."""
import pytest

from gpusched import (GPU, Cluster, Node, Pod, Scheduler, Taint, Toleration, effective_requests,
                      fit_error, fragmentation, gpu_node, gpu_pod, least_allocated, make_cluster, most_allocated,
                      pick_preemption_node, run_filters, select_victims, stranded_gpus)


def test_toleration_matching():
    t = Taint(GPU, "present", "NoSchedule")
    assert Toleration(GPU, "Exists").tolerates(t)                        # any value, any effect
    assert Toleration(GPU, "Equal", "present", "NoSchedule").tolerates(t)
    assert not Toleration(GPU, "Equal", "absent").tolerates(t)           # value must match
    assert not Toleration(GPU, "Exists", effect="NoExecute").tolerates(t)  # effect must match if set
    assert Toleration(operator="Exists").tolerates(t)                    # empty key + Exists: everything
    assert not Toleration("other", "Exists").tolerates(t)


def test_extended_resources_are_integers_with_requests_equal_limits():
    assert effective_requests({"cpu": 4000}, {GPU: 2})[GPU] == 2         # limit only -> request = limit
    assert effective_requests({"cpu": 1000}, {"cpu": 8000, GPU: 1})["cpu"] == 1000   # cpu may overcommit
    with pytest.raises(ValueError, match="must be an integer"):
        effective_requests({}, {GPU: 0.5})
    with pytest.raises(ValueError, match="must be equal to nvidia.com/gpu limit of 2"):
        effective_requests({GPU: 1}, {GPU: 2})
    with pytest.raises(ValueError, match="Limit must be set for non overcommitable resources"):
        effective_requests({GPU: 1}, {})


def test_gpu_pod_gets_the_extended_resource_toleration():
    pod = gpu_pod("p", 1)
    assert Toleration(GPU, "Exists", effect="NoSchedule") in pod.tolerations
    assert not run_filters(pod, gpu_node("n", 8))
    cpu_only = Pod("web", {"cpu": 500, "memory": 512})
    assert run_filters(cpu_only, gpu_node("n", 8)) == ["node(s) had untolerated taint(s)"]


def test_failed_scheduling_message_is_a_sorted_histogram():
    c = Cluster([gpu_node("g0", 8), gpu_node("g1", 8), Node("cpu0", {"cpu": 32000, "memory": 131072})])
    untolerating = Pod("p", {GPU: 1, "cpu": 1000, "memory": 1024})
    reasons = {n.name: run_filters(untolerating, n) for n in c.nodes.values()}
    assert fit_error(3, reasons) == ("0/3 nodes are available: 1 Insufficient nvidia.com/gpu, "
                                     "2 node(s) had untolerated taint(s).")


def test_scores_pinned_to_the_upstream_formulas():
    node = gpu_node("n", 8)
    node.pods.append(gpu_pod("busy", 6))
    pod = gpu_pod("new", 1)
    gpu_only = ((GPU, 1),)
    assert most_allocated(pod, node, gpu_only) == 87          # (6+1) * 100 // 8
    assert least_allocated(pod, node, gpu_only) == 12         # (8-7) * 100 // 8
    # the default scoring resources (cpu, memory) never look at GPUs:
    a, b = gpu_node("a", 8), gpu_node("b", 8)
    a.pods.append(Pod("x", {GPU: 7, "cpu": 8000, "memory": 65536}))
    b.pods.append(Pod("y", {GPU: 1, "cpu": 8000, "memory": 65536}))
    assert least_allocated(pod, a) == least_allocated(pod, b)
    # an extended resource the pod does not request is skipped, not scored as 0
    cpu_pod = Pod("c", {"cpu": 8000, "memory": 65536})
    assert most_allocated(cpu_pod, b, (("cpu", 1), (GPU, 1))) == most_allocated(cpu_pod, b, (("cpu", 1),))


def test_spreading_strands_gpus_and_packing_does_not():
    spread, packed = make_cluster(hosts=4), make_cluster(hosts=4)
    s1 = Scheduler(spread)
    s1.submit(*[gpu_pod(f"s{i}", 1) for i in range(16)])
    s1.run()
    s2 = Scheduler(packed, strategy="MostAllocated", resources=((GPU, 1),))
    s2.submit(*[gpu_pod(f"p{i}", 1) for i in range(16)])
    s2.run()
    assert [n.free() for n in spread.nodes.values()] == [4, 4, 4, 4]
    assert stranded_gpus(spread, 8) == 16 and fragmentation(spread, 8) == 1.0
    assert stranded_gpus(packed, 8) == 0 and fragmentation(packed, 8) == 0.0
    d = s1.schedule_one(gpu_pod("big", 8))
    assert d.node is None and d.message == (
        "0/4 nodes are available: 4 Insufficient nvidia.com/gpu. "
        "preemption: 0/4 nodes are available: 4 No preemption victims found for incoming pod.")
    assert s2.schedule_one(gpu_pod("big", 8)).node is not None


def test_preemption_reprieves_what_it_can_and_picks_the_cheapest_node():
    c = Cluster([gpu_node("a", 8), gpu_node("b", 8)])
    old, new = gpu_pod("old", 4, priority=0), gpu_pod("new", 4, priority=0)
    c.bind(old, "a")
    c.bind(new, "a")
    c.bind(gpu_pod("mid", 8, priority=5), "b")
    urgent = gpu_pod("urgent", 4, priority=10)
    assert [v.name for v in select_victims(urgent, c.nodes["a"])] == ["new"]   # the older pod is reprieved
    assert [v.name for v in select_victims(urgent, c.nodes["b"])] == ["mid"]
    s = Scheduler(c)
    d = s.schedule_one(urgent)
    assert d.node == "a" and [v.name for v in d.victims] == ["new"]          # lowest-priority victims win


def test_preemption_counts_victims_before_summing_priorities():
    x = [Pod("x0", priority=0, started=1), Pod("x1", priority=-5, started=2)]
    y = [Pod("y0", priority=0, started=3)]
    # plain sums would favour x (-5 < 0); the 2**31 offset makes the single victim on y cheaper
    assert pick_preemption_node({"x": x, "y": y}) == "y"
