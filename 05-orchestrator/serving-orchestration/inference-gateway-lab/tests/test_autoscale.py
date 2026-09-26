"""The HPA recommender, pinned to kube-controller-manager v1.34 semantics."""
import pytest

from igwlab.autoscale import (DEFAULT_BEHAVIOR, Behavior, FluidPool, HPARecommender, MetricError, PodSample,
                              ScalingPolicy, ScalingRules, Tolerances, external_per_pod_replicas, hpa_manifest,
                              milli, plain_metric_replicas, recommend_from_scrapes, simulate, usage_ratio_replicas)


def pods(*vals, pending=0):
    out = [PodSample(f"p{i}", v) for i, v in enumerate(vals)]
    return out + [PodSample(f"new{i}", None, phase="Pending") for i in range(pending)]


def test_core_formula_ceil_current_times_ratio():
    assert plain_metric_replicas(pods(10, 10), 2, 5) == (4, 10.0)            # ceil(2 * 10/5)
    assert plain_metric_replicas(pods(6, 6, 6), 3, 5)[0] == 4                # ceil(3 * 1.2) = ceil(3.6)
    assert plain_metric_replicas(pods(1, 1, 1, 1), 4, 5)[0] == 1             # ceil(4 * 0.2) = ceil(0.8)


def test_tolerance_band_is_inclusive_10_percent():
    assert plain_metric_replicas(pods(5.5, 5.5), 2, 5)[0] == 2               # ratio 1.1: no change
    assert plain_metric_replicas(pods(4.5, 4.5), 2, 5)[0] == 2               # ratio 0.9: no change
    assert plain_metric_replicas(pods(5.6, 5.6), 2, 5)[0] == 3               # ratio 1.12 -> ceil(2.24)
    assert Tolerances(down=0.1, up=0.5).is_within(1.4)


def test_average_is_integer_milli_units():
    assert milli(0.0015) == 2 and milli(2.5) == 2500                         # MilliValue rounds up
    # (1 + 2 + 2) / 3 = 1.666.. -> 1666 milli (floor) -> ratio 1.666 vs target 1
    assert plain_metric_replicas(pods(1, 2, 2), 3, 1) == (5, 1.666)


def test_missing_and_unready_pods():
    # scale-up: missing pods count as 0 -> (12 + 0)/2 = 6 -> ratio 1.2 -> ceil(1.2*2) = 3
    assert plain_metric_replicas(pods(12, None), 2, 5)[0] == 3
    # scale-down: missing pods count as the target -> (1 + 5)/2 = 3 -> ratio 0.6 -> ceil(1.2) = 2
    assert plain_metric_replicas(pods(1, None), 2, 5)[0] == 2
    # scale-up with a Pending pod: it counts as 0 -> (12+12+0)/3 = 8 -> ceil(1.6*3) = 5
    assert plain_metric_replicas(pods(12, 12, pending=1), 3, 5)[0] == 5
    # direction flip guard: scale-down signal, but filling missing pods at target would flip to >1? no:
    assert plain_metric_replicas(pods(6, None), 2, 5)[0] == 2                 # 1.2 -> fill 0 -> 0.6 flips -> keep
    with pytest.raises(MetricError):
        plain_metric_replicas(pods(None, None), 2, 5)
    with pytest.raises(MetricError):
        plain_metric_replicas([], 2, 5)


def test_object_external_and_scale_from_zero_math():
    assert usage_ratio_replicas(3, 30, 10, ready_pods=3) == 9
    assert usage_ratio_replicas(3, 10.5, 10, ready_pods=3) == 3               # within tolerance
    assert usage_ratio_replicas(0, 25, 10, ready_pods=0) == 3                 # from zero: ceil(usage/target)
    assert external_per_pod_replicas(2, [30, 10], 5) == 8                     # ceil(40/5)
    assert external_per_pod_replicas(4, [10, 10], 5) == 4                     # 20/(5*4) = 1.0


def test_legacy_path_limits_scale_up_to_double_or_four():
    h = HPARecommender(1, 20)                       # behavior unset -> normalizeDesiredReplicas
    assert h.reconcile(0, 1, 50).desired == 4       # max(2*1, 4)
    assert h.reconcile(15, 4, 50).desired == 8      # 2*4
    assert h.reconcile(30, 8, 50).desired == 16
    assert h.reconcile(45, 16, 50).desired == 20    # maxReplicas


def test_legacy_downscale_stabilization_keeps_the_max_of_the_last_300s():
    h = HPARecommender(1, 10)
    assert h.reconcile(0, 6, 6).desired == 6
    assert h.reconcile(15, 6, 2).desired == 6       # the t=0 recommendation of 6 is still in the window
    assert h.reconcile(299, 6, 2).desired == 6
    assert h.reconcile(301, 6, 2).desired == 2      # t=0 sample left the window; recent samples are all 2


def test_default_behavior_policies():
    h = HPARecommender(1, 50, behavior=DEFAULT_BEHAVIOR)
    s = h.reconcile(0, 1, 40)
    assert s.desired == 5 and s.limited == "ScaleUpLimit"         # max(1+4, ceil(1*2)) = 5
    assert h.reconcile(5, 5, 40).desired == 5                     # same 15 s period: start=5-4=1 -> limit 5
    assert h.reconcile(16, 5, 40).desired == 10                   # new period: max(5+4, 10)
    assert h.reconcile(32, 10, 40).desired == 20                  # max(14, 20)


def test_select_policy_min_and_disabled_and_percent_scale_down_truncates():
    up_min = Behavior(scale_up=ScalingRules(0, "Min", (ScalingPolicy("Pods", 4, 60), ScalingPolicy("Percent", 100, 60))))
    assert HPARecommender(1, 50, behavior=up_min).reconcile(0, 1, 40).desired == 2      # min(5, 2)
    no_down = Behavior(scale_down=ScalingRules(0, "Disabled", ()))
    assert HPARecommender(1, 50, behavior=no_down).reconcile(0, 8, 1).desired == 8
    half = Behavior(scale_down=ScalingRules(0, "Max", (ScalingPolicy("Percent", 50, 60),)))
    assert HPARecommender(1, 50, behavior=half).reconcile(0, 5, 1).desired == 2          # int(5*0.5) = 2


def test_behavior_scale_down_window_defaults_to_300s_and_bounds():
    h = HPARecommender(1, 10, behavior=DEFAULT_BEHAVIOR)
    assert h.reconcile(0, 6, 6).desired == 6
    assert h.reconcile(100, 6, 1).desired == 6 and h.history[-1].stabilized == 6
    assert h.reconcile(400, 6, 1).desired == 1
    assert HPARecommender(2, 5).reconcile(0, 7, 3).desired == 5      # above max: clamp first
    assert HPARecommender(2, 5).reconcile(0, 1, 3).desired == 2      # below min
    z = HPARecommender(1, 5).reconcile(0, 0, 3)
    assert z.desired == 0 and "ScalingDisabled" in z.reason          # HPA does not scale from 0 (min >= 1)


def test_manifest_and_simulation_shapes():
    m = hpa_manifest("vllm", "vllm", "vllm:num_requests_waiting", 5, behavior=DEFAULT_BEHAVIOR)
    assert m["apiVersion"] == "autoscaling/v2" and m["spec"]["metrics"][0]["pods"]["target"] == {"type": "AverageValue", "averageValue": "5"}
    assert "stabilizationWindowSeconds" not in m["spec"]["behavior"]["scaleDown"]
    try:
        import kubernetes_validate
        kubernetes_validate.validate(m, "1.34.0", strict=True)
    except ImportError:
        pass

    class Step:
        horizon = 900
        def __call__(self, t):
            return 3.0 if t < 120 else 11.0

    queue_only = simulate(Step(), {"waiting": 5}, pool=FluidPool(ready=2))
    both = simulate(Step(), {"waiting": 5, "running": 6}, pool=FluidPool(ready=2))
    late = lambda rows: [r["ready"] for r in rows if r["t"] >= 780]
    assert min(late(both)) >= 6                                  # running holds ~ lam*S/6 = 7.3 replicas
    assert min(late(queue_only)) < 6                             # queue-only collapsed after draining
    assert all(r["gpu_busy"] == 1.0 for r in both)               # duty cycle says nothing about need


def test_recommendation_from_scraped_vllm_metrics():
    def page(waiting, running):
        return (f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}\n'
                f'vllm:num_requests_running{{engine="0",model_name="m"}} {running}\n')
    scrapes = {"p0": page(9, 8), "p1": page(11, 8), "p2": "# a pod that exports nothing yet\n"}
    best, per = recommend_from_scrapes(scrapes, {"vllm:num_requests_waiting": 5, "vllm:num_requests_running": 6}, 3)
    # waiting: ratio 10/5 = 2; p2 missing counts as 0 on a scale-up: 6.666/5 = 1.333 -> ceil(3.9996) = 4
    # running: ratio 8/6 = 1.333; with p2 at 0 it becomes 5.333/6 = 0.889 -> direction would flip -> keep 3
    assert per["vllm:num_requests_waiting"][0] == 4 and per["vllm:num_requests_running"][0] == 3 and best == 4
