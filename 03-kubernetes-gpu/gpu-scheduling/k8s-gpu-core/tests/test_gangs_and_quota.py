"""Gangs, topology-aware placement (Kueue TAS) and Kueue quota semantics, pinned to upstream examples."""
from gpusched import (GPU, ClusterQueue, Kueue, Quota, Scheduler, Taint, Workload, admit_gangs, best_fit, gpu_pod,
                      interleave, least_free_capacity, make_cluster, place_gang, running)
from gpusched.gang import pods_that_fit


def _gangs():
    return ([gpu_pod(f"a{i}", 2, gang="A") for i in range(4)],
            [gpu_pod(f"b{i}", 2, gang="B") for i in range(4)])


def test_pod_by_pod_scheduling_deadlocks_two_gangs():
    c = make_cluster(hosts=3, gpus=4)                  # 12 GPUs; each job needs 8
    a, b = _gangs()
    s = Scheduler(c)
    s.submit(*interleave(a, b))
    s.run()
    assert c.free() == 0 and not running(a) and not running(b)
    assert sorted(p.name for p in s.pending) == ["a3", "b3"]   # 12 GPUs held, zero jobs running
    assert {n.name[-2:]: [p.name for p in n.pods] for n in c.nodes.values()} == \
        {"h0": ["a0", "b1"], "h1": ["b0", "a2"], "h2": ["a1", "b2"]}  # the layout drawn in PRIMER 4.1


def test_all_or_nothing_admission_runs_one_gang_and_holds_nothing_for_the_other():
    c = make_cluster(hosts=3, gpus=4)
    a, b = _gangs()
    assert admit_gangs(c, [a, b]) == ["A"]
    assert running(a) and not any(p.node for p in b) and c.free() == 4


def test_tas_algorithms_match_the_kep_example():
    caps = {"n1": 3, "n2": 3, "n3": 2, "n4": 1}                 # KEP-2724: a rack of 3, 3, 2, 1 free slots
    assert best_fit(caps, 7) == {"n1": 3, "n2": 3, "n4": 1}      # most room first, tightest last
    assert least_free_capacity(caps, 7) == {"n4": 1, "n3": 2, "n1": 3, "n2": 1}
    assert best_fit(caps, 10) is None


def test_least_free_capacity_first_takes_the_tightest_single_domain_that_holds_everything():
    caps = {"n1": 3, "n2": 3, "n3": 2, "n4": 1}
    assert least_free_capacity(caps, 2) == {"n3": 2}               # not 1 on n4 + 1 on n3
    assert least_free_capacity(caps, 3) == {"n1": 3}
    assert least_free_capacity(caps, 4) == {"n4": 1, "n3": 2, "n1": 1}   # no single node: smallest gaps first
    c = make_cluster(hosts=2)
    c.bind(gpu_pod("x", 4), "b0-s0-h0")                            # h0 has room for one 4-GPU pod, h1 for two
    assert place_gang(c, [gpu_pod("u0", 4), gpu_pod("u1", 4)]) == {"u0": "b0-s0-h1", "u1": "b0-s0-h1"}


def test_unconstrained_ranks_hosts_as_one_flat_list():
    c = make_cluster(blocks=2, hosts=2)                            # b0: h0 h1   b1: h0 h1
    c.bind(gpu_pod("x", 6), "b0-s0-h0")                            # b0-s0-h0 holds one more 2-GPU pod
    c.bind(gpu_pod("y", 2), "b1-s0-h0")                            # b1-s0-h0 holds three
    c.bind(gpu_pod("z", 4), "b0-s0-h1")                            # b0-s0-h1 holds two
    placed = place_gang(c, [gpu_pod(f"u{i}", 2) for i in range(3)])
    assert placed == {"u0": "b1-s0-h0", "u1": "b1-s0-h0", "u2": "b1-s0-h0"}   # the tightest single host
    placed = place_gang(c, [gpu_pod(f"v{i}", 2) for i in range(5)])        # no single host: 1 + 2 + 2
    assert sorted(placed.values()) == ["b0-s0-h0", "b0-s0-h1", "b0-s0-h1", "b1-s0-h0", "b1-s0-h0"]


def test_preferred_descent_pools_the_children_of_every_chosen_domain():
    c = _busy_fleet()
    placement = place_gang(c, [gpu_pod(f"k{i}", 8) for i in range(5)], preferred="subblock")
    per_subblock = {}
    for node in placement.values():
        per_subblock[c.nodes[node].topology[1]] = per_subblock.get(c.nodes[node].topology[1], 0) + 1
    # block b1 is chosen (2 + 4 slots); the sub-block pass says 4 + 1, but Kueue re-runs BestFit over
    # all six hosts of both sub-blocks, which ranks b1-s0's hosts first by name: 2 + 3
    assert per_subblock == {"b1-s0": 2, "b1-s1": 3}


def test_hosts_rejected_by_a_non_resource_filter_have_no_room():
    c = make_cluster(hosts=2)
    c.nodes["b0-s0-h0"].taints.append(Taint("dedicated", "inference", "NoSchedule"))
    c.nodes["b0-s0-h1"].unschedulable = True
    assert [pods_that_fit(gpu_pod("p", 1), n) for n in c.nodes.values()] == [0, 0]
    assert place_gang(c, [gpu_pod("u0", 8)]) is None


def _busy_fleet():                                                # notebook 03 and PRIMER 5.2
    c = make_cluster(blocks=2, subblocks=2, hosts=4)
    busy = {"b0-s0-h0": 8, "b0-s0-h1": 8, "b0-s0-h2": 8, "b0-s1-h0": 8, "b1-s0-h0": 8, "b1-s0-h1": 1}
    for host, gpus in busy.items():
        c.bind(gpu_pod(f"x-{host}", gpus), host)
    return c


def _busy_cluster():
    c = make_cluster(blocks=2, subblocks=2, hosts=4)            # 16 nodes x 8 GPUs
    for name, g in {"b0-s0-h0": 8, "b0-s0-h1": 4, "b0-s1-h0": 2, "b1-s0-h2": 6}.items():
        c.bind(gpu_pod(f"x-{name}", g), name)
    return c


def test_required_topology_picks_the_tightest_domain_or_waits():
    c = _busy_cluster()          # free 8-GPU hosts per sub-block: b0-s0: 2, b0-s1: 3, b1-s0: 3, b1-s1: 4
    three = [gpu_pod(f"w{i}", 8) for i in range(3)]
    placement = place_gang(c, three, required="subblock")
    assert {c.nodes[n].topology[1] for n in placement.values()} == {"b0-s1"}   # 3-slot tie: first by name
    five = [gpu_pod(f"v{i}", 8) for i in range(5)]
    assert place_gang(c, five, required="subblock") is None                    # no sub-block has 5
    spread = place_gang(c, five, preferred="subblock")                         # widens to block b0 (5 slots)
    assert {c.nodes[n].topology[0] for n in spread.values()} == {"b0"}


def _cohort(**team_a):
    a = ClusterQueue("team-a-cq", {"h100": {GPU: Quota(16)}}, cohort="research", **team_a)
    b = ClusterQueue("team-b-cq", {"h100": {GPU: Quota(16)}}, cohort="research")
    return Kueue([a, b], local_queues={"team-a/gpus": "team-a-cq", "team-b/gpus": "team-b-cq"})


def test_available_quota_matches_the_kueue_docs():
    def pair(a_borrow=None, b_lend=None):
        a = ClusterQueue("a", {"f": {"cpu": Quota(9, borrowing_limit=a_borrow)}}, cohort="ab")
        b = ClusterQueue("b", {"f": {"cpu": Quota(12, lending_limit=b_lend)}}, cohort="ab")
        return Kueue([a, b]), a, b
    k, a, b = pair()
    assert k.available(a, "f", "cpu") == 21 and k.available(b, "f", "cpu") == 21     # 9 + 12
    k, a, b = pair(a_borrow=1)
    assert k.available(a, "f", "cpu") == 10 and k.available(b, "f", "cpu") == 21     # 9 + 1
    k, a, b = pair(b_lend=1)
    assert k.available(a, "f", "cpu") == 10                                          # 9 + 1


def test_borrowers_are_reclaimed_newest_first_when_reclaim_is_enabled():
    k = _cohort(reclaim_within_cohort="Any")
    k.submit(*[Workload(f"b{i}", "team-b/gpus", {GPU: 8}) for i in (1, 2, 3)])
    assert [e[4] for e in k.schedule()] == ["", "", "borrowing"]
    k.submit(Workload("a1", "team-a/gpus", {GPU: 16}))
    assert k.schedule() == [("preempted", "b3", "team-b-cq", "a1"), ("admitted", "a1", "team-a-cq", "h100", "")]


def test_reclaim_takes_only_from_queues_that_are_still_borrowing():
    a = ClusterQueue("a", {"f": {GPU: Quota(8)}}, cohort="c", reclaim_within_cohort="Any")
    b = ClusterQueue("b", {"f": {GPU: Quota(8)}}, cohort="c")
    c = ClusterQueue("c", {"f": {GPU: Quota(8)}}, cohort="c")
    k = Kueue([a, b, c])
    k.submit(Workload("b1", "b", {GPU: 8}), Workload("b2", "b", {GPU: 4}))
    k.schedule()                                                   # b borrows 4 of a's idle quota
    k.submit(Workload("c1", "c", {GPU: 8}))
    k.schedule()                                                   # c1 is the newest, but c is within nominal
    k.submit(Workload("a1", "a", {GPU: 8}))
    events = k.schedule()
    assert events == [("preempted", "b2", "b", "a1"), ("admitted", "a1", "a", "f", "")]
    assert "c1" in [w.name for w in k.running]


def test_reclaim_minimises_the_targets_in_reverse():
    a = ClusterQueue("a", {"f": {GPU: Quota(12)}}, cohort="c", reclaim_within_cohort="Any")
    b = ClusterQueue("b", {"f": {GPU: Quota(4)}}, cohort="c")
    k = Kueue([a, b])
    k.submit(Workload("b-big", "b", {GPU: 8}, priority=5))
    k.schedule()
    k.submit(Workload("b-small", "b", {GPU: 2}, priority=0))
    k.schedule()                                                   # b uses 10 of its 4: 6 borrowed
    k.submit(Workload("a1", "a", {GPU: 10}))
    # greedy: b-small (lowest priority) then b-big; in reverse, b-small is given back (a still fits)
    assert k.schedule() == [("preempted", "b-big", "b", "a1"), ("admitted", "a1", "a", "f", "")]


def test_workloads_that_fit_without_borrowing_are_admitted_first():
    x = ClusterQueue("x", {"f": {GPU: Quota(8)}}, cohort="c")
    y = ClusterQueue("y", {"f": {GPU: Quota(8)}}, cohort="c")
    k = Kueue([x, y])
    k.submit(Workload("x-run", "x", {GPU: 8}))
    k.schedule()
    k.submit(Workload("x-old", "x", {GPU: 8}), Workload("y-new", "y", {GPU: 8}))   # x-old must borrow y's 8
    assert [e[:2] for e in k.schedule()] == [("admitted", "y-new")]
    assert [w.name for w in k.pending] == ["x-old"]


def test_nominal_quota_is_not_a_guarantee_without_reclaim():
    k = _cohort()                                                       # reclaimWithinCohort: Never
    k.submit(*[Workload(f"b{i}", "team-b/gpus", {GPU: 8}) for i in (1, 2, 3)])
    k.schedule()
    a1 = Workload("a1", "team-a/gpus", {GPU: 16})
    k.submit(a1)
    assert k.schedule() == []
    assert k.explain(a1) == ("couldn't assign flavors to pod set main: insufficient unused quota "
                             "for nvidia.com/gpu in flavor h100, 8 more needed")


def test_strict_fifo_blocks_behind_the_head_best_effort_does_not():
    for strategy, admitted in (("StrictFIFO", ["small-1"]), ("BestEffortFIFO", ["small-1", "small-2"])):
        k = Kueue([ClusterQueue("cq", {"h100": {GPU: Quota(16)}}, queueing_strategy=strategy)])
        k.submit(Workload("small-1", "cq", {GPU: 8}), Workload("big", "cq", {GPU: 16}),
                 Workload("small-2", "cq", {GPU: 8}))
        assert [e[1] for e in k.schedule()] == admitted


def test_lending_limit_keeps_part_of_the_quota_home():
    serve = ClusterQueue("serve-cq", {"h100": {GPU: Quota(16, lending_limit=8)}}, cohort="c")
    batch = ClusterQueue("batch-cq", {"h100": {GPU: Quota(16)}}, cohort="c")
    k = Kueue([serve, batch])
    k.submit(*[Workload(f"batch-{i}", "batch-cq", {GPU: 8}) for i in range(5)])
    assert len(k.schedule()) == 3                                       # 16 own + 8 lent
    assert k.available(serve, "h100", GPU) == 8                         # 16 - lendingLimit stays home


def test_priority_preemption_inside_a_cluster_queue_takes_the_newest():
    k = Kueue([ClusterQueue("cq", {"h100": {GPU: Quota(16)}}, within_cluster_queue="LowerPriority")])
    k.submit(Workload("batch-1", "cq", {GPU: 8}), Workload("batch-2", "cq", {GPU: 8}))
    k.schedule()
    k.submit(Workload("urgent", "cq", {GPU: 8}, priority=100))
    assert k.schedule()[0] == ("preempted", "batch-2", "cq", "urgent")


def test_borrow_within_cohort_respects_the_priority_threshold():
    def run(threshold):
        a = ClusterQueue("a", {"f": {GPU: Quota(8)}}, cohort="c", reclaim_within_cohort="LowerPriority",
                         borrow_within_cohort="LowerPriority", max_priority_threshold=threshold)
        b = ClusterQueue("b", {"f": {GPU: Quota(8)}}, cohort="c")
        idle = ClusterQueue("idle", {"f": {GPU: Quota(8)}}, cohort="c")
        k = Kueue([a, b, idle])
        k.submit(Workload("b-old", "b", {GPU: 8}, priority=50), Workload("b-new", "b", {GPU: 8}, priority=0))
        k.schedule()                                                    # b-new borrows idle's quota
        k.submit(Workload("a-big", "a", {GPU: 12}, priority=100))     # must borrow 4 itself
        return k.schedule()
    assert run(None)[0] == ("preempted", "b-new", "b", "a-big")
    assert run(0)[0] == ("preempted", "b-new", "b", "a-big")          # priority 0 <= threshold 0
    assert run(-1) == []                                                # nothing at or below the threshold
