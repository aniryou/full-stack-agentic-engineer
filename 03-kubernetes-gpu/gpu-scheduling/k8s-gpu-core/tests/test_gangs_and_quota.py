"""Gangs, topology-aware placement (Kueue TAS) and Kueue quota semantics, pinned to upstream examples."""
from gpusched import (GPU, ClusterQueue, Kueue, Quota, Scheduler, Workload, admit_gangs, best_fit, gpu_pod,
                      interleave, least_free_capacity, make_cluster, place_gang, running)


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
