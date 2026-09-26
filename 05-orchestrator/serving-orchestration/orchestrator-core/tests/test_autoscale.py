"""The HPA algorithm, pinned to the Kubernetes documentation's examples and the controller's arithmetic."""
import pytest

from fleetsim import HPA, Autoscaler, ColdStart, Policy, Rules, external_metric_replicas, pods_metric_replicas

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


def test_not_ready_pods_damp_a_scale_up():
    # 2 ready pods with 10 queued each (target 2) want ceil(5 x 2) = 10 ...
    assert pods_metric_replicas([10, 10], 2, 2) == 10
    # ... but once 8 more are starting they count as 0: (20 + 0 x 8) / 10 / 2 = 1.0 -> within tolerance, hold
    assert pods_metric_replicas([10, 10], 2, 10, unready=8) == 10
    # if the queue keeps growing during the cold start, it can still scale further
    assert pods_metric_replicas([30, 30], 2, 10, unready=8) == 30


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


def test_cold_start_is_a_sum_of_different_fixes():
    cs = ColdStart.estimate(weight_gb=16, load_gb_s=0.5, image_gb=10, pull_gb_s=0.25, node_s=0, init_s=30)
    assert cs.total == pytest.approx(40 + 32 + 30)
