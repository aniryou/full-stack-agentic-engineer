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


def test_reprieve_goes_in_order_of_importance():
    c = Cluster([gpu_node("x", 8)])
    for name, g, prio in [("old-low", 4, 0), ("mid", 2, 50), ("new-low", 2, 0)]:
        c.bind(gpu_pod(name, g, priority=prio), "x")
    # all three removed; mid (priority 50) goes back first, then old-low (earlier start) does not fit
    # back, then new-low does: the victim is old-low, not the newer pod of the same priority
    assert [v.name for v in select_victims(gpu_pod("want", 4, priority=100), c.nodes["x"])] == ["old-low"]


def test_preemption_tie_breaks_on_the_latest_start_of_the_top_victims():
    c = Cluster([gpu_node("a", 8), gpu_node("b", 8)])
    c.bind(gpu_pod("started-first", 8), "a")
    c.bind(gpu_pod("started-later", 8), "b")
    d = Scheduler(c).schedule_one(gpu_pod("urgent", 8, priority=10))
    assert d.node == "b" and [v.name for v in d.victims] == ["started-later"]   # the younger pod loses less work


def test_preemption_is_not_helpful_where_a_non_resource_filter_fails():
    c = Cluster([gpu_node("g", 8)])
    c.bind(gpu_pod("low", 8, priority=0), "g")
    no_toleration = Pod("p", {GPU: 8, "cpu": 1000, "memory": 1024}, priority=100)
    d = Scheduler(c).schedule_one(no_toleration)
    assert d.node is None and d.message == (
        "0/1 nodes are available: 1 node(s) had untolerated taint(s). "
        "preemption: 0/1 nodes are available: 1 Preemption is not helpful for scheduling.")


def test_stranded_gpus_are_the_remainder_per_node():
    c = Cluster([gpu_node("a", 8), gpu_node("b", 8)])
    c.bind(gpu_pod("two", 2), "a")                                 # a: 6 free -> one 4-GPU pod + 2 stranded
    assert stranded_gpus(c, 4) == 2 + 0 and fragmentation(c, 4) == pytest.approx(2 / 14)


def test_only_a_gpu_weighted_score_keeps_whole_nodes_free_when_cpu_points_elsewhere():
    """Notebook 02, exercise 2.3: a CPU-heavy non-GPU pod on h0 pulls cpu/memory packing toward h0."""
    def replay(strategy, resources):
        c = make_cluster(hosts=4)
        c.bind(Pod("data-prep", {"cpu": 48000, "memory": 393216},
                   tolerations=[Toleration(GPU, "Exists", effect="NoSchedule")]), "b0-s0-h0")
        c.bind(gpu_pod("embed-0", 4), "b0-s0-h1")
        c.bind(gpu_pod("embed-1", 4), "b0-s0-h2")
        s = Scheduler(c, strategy=strategy, resources=resources)
        s.submit(*[gpu_pod(f"infer-{i}", 1) for i in range(8)])
        s.run()
        return [s.schedule_one(gpu_pod(f"tune-{i}", 8)).node for i in range(2)]
    cpu_mem = (("cpu", 1), ("memory", 1))
    assert replay("LeastAllocated", cpu_mem) == ["b0-s0-h0", None]
    assert replay("MostAllocated", cpu_mem) == ["b0-s0-h3", None]
    assert replay("MostAllocated", cpu_mem + ((GPU, 1),)) == ["b0-s0-h3", None]      # weight 1 is not enough
    assert replay("MostAllocated", cpu_mem + ((GPU, 5),)) == ["b0-s0-h0", "b0-s0-h3"]
