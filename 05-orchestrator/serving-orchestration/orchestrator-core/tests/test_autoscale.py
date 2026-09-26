"""The HPA algorithm, pinned to the Kubernetes documentation's examples and the controller's arithmetic."""
import pytest

from fleetsim import (HPA, L4_8B, Autoscaler, ColdStart, Fleet, Policy, PowerOfTwo, Replica, Rules, chat,
                      external_metric_replicas, pods_metric_replicas)
from fleetsim.workload import HashChain, Request

NO_WINDOW_DOWN = Rules(0, (Policy("Percent", 100, 15),))


def test_ratio_formula_from_the_docs():
    assert pods_metric_replicas([0.2] * 3, 0.1, 3) == 6          # 200m vs 100m target: double
    assert pods_metric_replicas([0.05] * 4, 0.1, 4) == 2         # 50m vs 100m: halve
    assert pods_metric_replicas([7, 8, 9], 4, 3) == 6            # ceil(3 x 8/4)


def test_ten_percent_tolerance():
    assert pods_metric_replicas([4.4] * 5, 4, 5) == 5            # ratio 1.10: inside the band, no action
    assert pods_metric_replicas([4.5] * 5, 4, 5) == 6            # ratio 1.125: ceil(5.625)
    assert pods_metric_replicas([3.6] * 5, 4, 5) == 5            # ratio 0.90: inside
    assert pods_metric_replicas([4.5] * 5, 4, 5, tol_up=0.2) == 5   # per-HPA tolerance (behavior.scaleUp.tolerance)


def test_pending_pods_count_as_zero_on_a_scale_up():
    # 2 ready pods with 10 queued each (target 2): ceil(ratio 5 x 2 reporting pods) = 10, not 5 x 10 current = 50.
    # What bounds the scale-up is multiplying by the pods that reported, not the Pending ones.
    assert pods_metric_replicas([10, 10], 2, 2) == 10
    assert pods_metric_replicas([10, 10], 2, 10, unready=8) == 10       # (20 + 0 x 8) / 10 / 2 = 1.0: hold
    assert pods_metric_replicas([30, 30], 2, 10, unready=8) == 30       # the queue kept growing: scale further


def test_pending_zeros_only_act_through_the_tolerance_band():
    # desired = ceil(sum / target) with or without the zeros; they matter when the new ratio lands in the band:
    assert pods_metric_replicas([10.5, 10.5], 2, 10, unready=8) == 10   # 21 / 10 / 2 = 1.05: inside, hold at 10
    assert pods_metric_replicas([10.5, 10.5], 2, 10) == 11              # without Pending pods: ceil(21 / 2) = 11
    assert pods_metric_replicas([2.3, 2.3], 2, 4, unready=2) == 4       # 4.6 / 4 / 2 < 1: direction flips, hold


def test_missing_pods_are_left_out_at_exactly_the_target():
    assert pods_metric_replicas([2, 2], 2, 4, missing=2) == 4           # ratio 1.0: upstream fills in neither way


def test_missing_metrics_count_as_target_on_a_scale_down():
    assert pods_metric_replicas([1, 1], 2, 4, missing=2) == 3    # (1 + 1 + 2 + 2) / 4 / 2 = 0.75 -> ceil(3.0)


def test_external_metric_is_how_keda_scales_and_can_start_from_zero():
    assert external_metric_replicas(95, 10, 4) == 10             # ceil(95 / 10)
    assert external_metric_replicas(42, 10, 4) == 4              # 42 / (10 x 4) = 1.05: within tolerance
    assert external_metric_replicas(5, 4, 0) == 2                # from zero replicas: ceil(5 / 4)
    assert external_metric_replicas(0, 4, 3) == 0                # nothing to do: zero
    with pytest.raises(ValueError):
        Autoscaler(HPA(min_replicas=0), metric="waiting", kind="pods")   # minReplicas 0 needs object/external


def test_default_scale_up_policy_is_max_of_4_pods_and_100_percent():
    hpa = HPA(1, 100)
    assert hpa.step(0, 1, 40) == 5                               # max(1 + 4, 1 x 2)
    assert hpa.step(15, 5, 40) == 10                             # the t=0 event is exactly 15 s old: not counted
    assert hpa.step(30, 10, 40) == 20
    assert hpa.step(45, 20, 40) == 40
    assert HPA(1, 100).step(0, 3, 40) == 7                       # max(3 + 4, 6)


def test_scale_down_stabilization_holds_the_max_of_the_last_300_s():
    hpa = HPA(1, 20)
    assert hpa.step(0, 10, 10) == 10
    for t in range(15, 300, 15):
        assert hpa.step(t, 10, 2) == 10                          # the 10 from t=0 is still in the window
    assert hpa.step(300, 10, 2) == 2                             # exactly 300 s old is outside (strict): drop


def test_percent_scale_down_policy_rounds_the_removal_up():
    hpa = HPA(1, 100, down=Rules(0, (Policy("Percent", 10, 60),)))
    assert hpa.step(0, 80, 10) == 72                             # 10% of 80
    assert hpa.step(15, 72, 10) == 72                            # the period still counts those 8: start is 80
    assert hpa.step(60, 72, 10) == 64                            # int(72 x 0.9) = 64: removes ceil(7.2) = 8


def test_select_policy_max_min_disabled():
    two = (Policy("Pods", 4, 60), Policy("Percent", 10, 60))
    assert HPA(1, 100, down=Rules(0, two)).step(0, 30, 1) == 26            # Max: most change (4 > 3)
    assert HPA(1, 100, down=Rules(0, two, "Min")).step(0, 80, 1) == 76     # Min: least change (4 < 8)
    assert HPA(1, 100, down=Rules(0, two, "Disabled")).step(0, 80, 1) == 80


def test_min_max_clamps():
    assert HPA(2, 5, down=NO_WINDOW_DOWN).step(0, 3, 1) == 2
    assert HPA(2, 5).step(0, 3, 50) == 5
    assert HPA(2, 5).step(0, 9, 9) == 5                          # current above max: rescale to max


def _req(rid, prompt):
    return Request(rid, 0.0, prompt, 10, HashChain(16).extend(100 + rid, prompt).hashes, prompt // 16, 16)


def test_backlog_signal_counts_prefill_work_not_requests():
    rag_like, chat_like = Replica(0, L4_8B), Replica(1, L4_8B)
    rag_like.enqueue(_req(0, 6000), 0.0)
    for i in range(6):
        chat_like.enqueue(_req(1 + i, 300), 0.0)
    v = Autoscaler.value
    assert v(rag_like, 0.0, "inflight") == 1 and v(chat_like, 0.0, "inflight") == 6
    assert v(rag_like, 0.0, "backlog_s") == pytest.approx(6000 / 3781.25)          # 1.59 s of prefill
    assert v(chat_like, 0.0, "backlog_s") == pytest.approx(1800 / 3781.25)         # 0.48 s: less work, more requests
    assert Autoscaler.held([_req(9, 3781)], L4_8B, "backlog_s") == pytest.approx(3781 / 3781.25)


def test_several_metrics_take_the_largest_recommendation():
    reps = [Replica(i, L4_8B) for i in range(2)]
    for i in range(6):
        reps[i % 2].enqueue(_req(i, 300), 0.0)                        # 3 in flight on each
    util = {0: 1.0, 1: 1.0}
    one = Autoscaler(HPA(1, 20), "inflight", 1, "external")           # ceil(6 / 1) = 6
    both = Autoscaler(HPA(1, 20), "waiting", 100, "pods", also=[("inflight", 1, "external")])
    assert one.decide(0.0, 2, reps, util, 0, [], L4_8B) == 6
    assert both.decide(0.0, 2, reps, util, 0, [], L4_8B) == 6         # waiting alone would scale down
    fast_up = Rules(0, (Policy("Pods", 20, 15),))                    # no rate limit in the way (default: 2 -> 6)
    big_first = Autoscaler(HPA(1, 20, up=fast_up), "inflight", 0.5, "pods", also=[("inflight", 3, "external")])
    assert big_first.decide(0.0, 2, reps, util, 0, [], L4_8B) == 12   # max(ceil(6 x 2), ceil(6 / 3)), in any order


def test_fleet_warm_start_holds_the_initial_size_for_the_scale_down_window():
    auto = Autoscaler(HPA(1, 8), "inflight", 40, kind="external")
    res = Fleet(L4_8B, 4, PowerOfTwo(seed=1), autoscaler=auto).run(chat(0.1, 400, seed=2), horizon=450)
    early = [row["desired"] for row in res.timeline if row["t"] < 300]
    assert early and all(d == 4 for d in early)                       # as if it had been recommending 4 all along
    assert min(row["desired"] for row in res.timeline if row["t"] >= 300) == 1


def test_cold_start_is_a_sum_of_different_fixes():
    cs = ColdStart.estimate(weight_gb=16, load_gb_s=0.5, image_gb=10, pull_gb_s=0.25, node_s=0, init_s=30)
    assert cs.total == pytest.approx(40 + 32 + 30)
