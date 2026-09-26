"""Device plugin -> kubelet accounting, and the autoscaling / obtainability arithmetic."""
import math
import random

import pytest

from gpusched import (GPU, AdmissionError, DevicePlugin, Job, Kubelet, NodePool, expected_runtime_h, gang_survival,
                      least_waste, make_gpus, nodes_needed, provision, simulate, startup_latency)
from gpusched.deviceplugin import UNHEALTHY


def test_capacity_counts_all_devices_allocatable_only_healthy():
    plugin, kubelet = DevicePlugin(make_gpus(8)), Kubelet()
    kubelet.register(plugin)
    assert plugin.register_request()["resource_name"] == GPU
    plugin.set_health("GPU-fake-0003", UNHEALTHY)
    assert kubelet.node_status() == {"capacity": {GPU: 8}, "allocatable": {GPU: 7}}


def test_kubelet_rejects_a_pod_when_devices_ran_out_under_the_scheduler():
    plugin, kubelet = DevicePlugin(make_gpus(8)), Kubelet()
    kubelet.register(plugin)
    kubelet.admit("train", 6)
    plugin.set_health("GPU-fake-0007", UNHEALTHY)          # after the scheduler counted 2 free
    with pytest.raises(AdmissionError, match="Requested: 2, Available: 1"):
        kubelet.admit("infer", 2)
    assert kubelet.admit("small", 1)["envs"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-fake-0006"


def test_time_slicing_multiplies_the_advertised_count_not_the_gpus():
    plugin, kubelet = DevicePlugin(make_gpus(8), replicas=10), Kubelet()
    kubelet.register(plugin)
    assert kubelet.node_status()["allocatable"] == {GPU: 80}
    env = kubelet.admit("two-slices", 2)["envs"]["NVIDIA_VISIBLE_DEVICES"]
    assert env == "GPU-fake-0000,GPU-fake-0001"               # fresh node: spread over the least-loaded GPUs


def test_time_sliced_replicas_can_share_one_physical_gpu_on_a_loaded_node():
    # NVIDIA k8s-device-plugin v0.17 internal/rm/allocate.go distributedAlloc: pick one replica at a
    # time from the GPU with the fewest allocated replicas
    plugin, kubelet = DevicePlugin(make_gpus(8), replicas=10), Kubelet()
    kubelet.register(plugin)
    for i in range(16):                                       # two 1-replica pods per GPU, round robin
        assert kubelet.admit(f"p{i}", 1)["envs"]["NVIDIA_VISIBLE_DEVICES"] == f"GPU-fake-{i % 8:04d}"
    del kubelet.assigned["p3"], kubelet.assigned["p11"]       # both pods on GPU-fake-0003 finish
    env = kubelet.admit("two-slices", 2)["envs"]["NVIDIA_VISIBLE_DEVICES"]
    assert env == "GPU-fake-0003"                             # two "GPUs", one physical device


def test_fail_requests_greater_than_one_rejects_multi_replica_requests_at_admission():
    plugin, kubelet = DevicePlugin(make_gpus(8), replicas=10, fail_requests_greater_than_one=True), Kubelet()
    kubelet.register(plugin)
    # the event text in the NVIDIA k8s-device-plugin README
    with pytest.raises(AdmissionError, match=r"^Allocate failed due to rpc error: code = Unknown desc = request for "
                       r"'nvidia.com/gpu: 2' too large: maximum request size for shared resources is 1, which is unexpected$"):
        kubelet.admit("two-slices", 2)
    assert "two-slices" not in kubelet.assigned
    assert kubelet.admit("one-slice", 1)["envs"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-fake-0000"
    exclusive = DevicePlugin(make_gpus(8), fail_requests_greater_than_one=True)   # no sharing: no limit
    assert exclusive.allocate(["GPU-fake-0000", "GPU-fake-0001"])["envs"]["NVIDIA_VISIBLE_DEVICES"] == \
        "GPU-fake-0000,GPU-fake-0001"


def test_preferred_allocation_keeps_a_container_on_one_nvlink_island():
    plugin, kubelet = DevicePlugin(make_gpus(8, island_size=2)), Kubelet()
    kubelet.register(plugin)
    kubelet.admit("one", 1)                                   # takes GPU 0, island 0 now has 1 free
    assert kubelet.assigned[next(iter(kubelet.assigned))] == ["GPU-fake-0000"]
    assert kubelet.admit("pair", 2)["envs"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-fake-0002,GPU-fake-0003"


def test_nodes_needed_is_first_fit_decreasing():
    assert nodes_needed([4, 4, 2, 2, 1, 1, 1, 1], 8) == 2
    assert nodes_needed([5, 5, 5], 8) == 3                    # 3 GPUs stranded on each node
    assert nodes_needed([1] * 9, 8) == 2
    with pytest.raises(ValueError):
        nodes_needed([9], 8)


def test_least_waste_picks_the_pool_with_fewest_idle_gpus():
    pools = [NodePool("l4x1", 1, max_nodes=20), NodePool("l4x4", 4, max_nodes=5), NodePool("h100x8", 8, max_nodes=2)]
    assert least_waste(pools, [1, 1, 1]).name == "l4x1"
    assert least_waste(pools, [3, 3, 3]).name == "l4x4"       # 3 idle vs 7 idle on two 8-GPU nodes
    assert least_waste(pools, [8, 8, 8]) is None              # would exceed max_nodes=2


def test_spot_gang_math_pinned_and_checked_by_monte_carlo():
    assert gang_survival(4, 10, 0.01) == pytest.approx(math.exp(-0.4))           # 0.670
    expected = expected_runtime_h(10, 4, 0.01)                                   # (e^0.4 - 1) / 0.04
    assert expected == pytest.approx(12.2956, abs=1e-3)
    rng, total = random.Random(0), 0.0
    for _ in range(20000):                                                       # restart-from-scratch runs
        t = 0.0
        while (x := rng.expovariate(0.04)) < 10:
            t += x
        total += t + 10
    assert total / 20000 == pytest.approx(expected, rel=0.03)
    assert expected_runtime_h(10, 16, 0.01) == pytest.approx(24.706, abs=1e-3)      # 4x the gang, 2x the time


def test_restart_overhead_term_checked_by_monte_carlo():
    with_r = expected_runtime_h(10, 4, 0.01, 2.0)                                # (e^0.4 - 1) * (25 + 2)
    assert with_r == pytest.approx((math.exp(0.4) - 1) * 27, abs=1e-9) == pytest.approx(13.2793, abs=1e-3)
    rng, total = random.Random(1), 0.0
    for _ in range(40000):                                                       # each failure costs R as well
        t = 0.0
        while (x := rng.expovariate(0.04)) < 10:
            t += x + 2.0
        total += t + 10
    assert total / 40000 == pytest.approx(with_r, rel=0.02)                      # R = 0 would be 7% low
    assert expected_runtime_h(24, 16, 0.005, 0.25) == pytest.approx(74.217, abs=1e-3)  # the primer's 16-node row


def test_queued_provisioning_does_not_bill_the_partial_gang():
    kw = dict(gpus_per_node=8, boot_s=300, stockout=0.9)
    inc = provision(NodePool("inc", **kw), 8, seed=4)
    q = provision(NodePool("q", queued=True, **kw), 8, seed=4)
    assert inc["gang_start_s"] == q["gang_start_s"]                              # same capacity, same moment
    assert q["waiting_node_h"] == pytest.approx(8 * 300 / 3600)                   # only the boot
    assert inc["waiting_node_h"] > q["waiting_node_h"]


def test_scale_from_zero_then_scale_down_after_unneeded_time():
    r = simulate(NodePool("l4", 1, max_nodes=2, boot_s=300, price_per_node_hr=1.0),
                 [Job("infer", 1, 1.0)], until_s=3 * 3600)
    assert r["jobs"]["infer"] == (300, 3900, 0)                                  # waits for the node to boot
    assert ("idle node removed" in dict(r["log"])[4500])                         # 3900 s + 600 s unneeded
    assert r["node_h"] == pytest.approx(1.25) and r["cost"] == pytest.approx(1.25)


def test_scale_down_waits_for_the_delay_after_the_last_scale_up():
    jobs = lambda: [Job("a", 1, 0.25), Job("b", 2, 0.5, arrive_s=1150)]
    pool = NodePool("l4", 1, max_nodes=4, boot_s=300)
    # b asks for 2 new nodes at 1170 s just before a frees its node; one new node is left over, idle
    # from 1470 s. With a 60 s unneeded time it could go at 1530 s; the delay after add holds it to 1770 s.
    held = simulate(pool, jobs(), until_s=3 * 3600, unneeded_s=60)
    free = simulate(pool, jobs(), until_s=3 * 3600, unneeded_s=60, delay_after_add_s=0)
    first_removal = lambda r: min(t for t, what in r["log"] if what == "idle node removed")
    assert (first_removal(free), first_removal(held)) == (1530, 1170 + 600)


def test_an_ordinary_pool_scales_up_a_gang_it_cannot_finish():
    job = [Job("train", 4, 1.0)]
    ordinary = simulate(NodePool("a3", 8, max_nodes=3, boot_s=300), job, until_s=7200)
    queued = simulate(NodePool("a3q", 8, max_nodes=3, boot_s=300, queued=True), [Job("train", 4, 1.0)], until_s=7200)
    assert ordinary["jobs"]["train"][0] is None and ordinary["node_h"] == pytest.approx(6.0)   # 3 idle nodes
    assert queued["jobs"]["train"][0] is None and queued["node_h"] == 0


def test_startup_latency_is_a_sum_of_stages():
    s = startup_latency(node_s=240, driver_s=60, image_gb=10, pull_GBps=0.5, weights_gb=16, load_GBps=1)
    assert s["image"] == 20 and s["weights"] == 16 and s["total"] == 336


def _loaded(policy):
    """16 one-replica pods round robin over 8 GPUs, then both pods on GPU-fake-0003 finish."""
    plugin, kubelet = DevicePlugin(make_gpus(8), replicas=10, allocation_policy=policy), Kubelet()
    kubelet.register(plugin)
    for i in range(16):
        kubelet.admit(f"p{i}", 1)
    del kubelet.assigned["p3"], kubelet.assigned["p11"]
    return kubelet


def test_allocation_policies_for_time_sliced_replicas():
    # NVIDIA k8s-device-plugin internal/rm/allocate.go allocationComparators (distributed and packed
    # since v0.20.0, spread on main): the same loaded node, one 2-replica request
    envs = {p: _loaded(p).admit("two-slices", 2)["envs"]["NVIDIA_VISIBLE_DEVICES"]
            for p in ("distributed", "packed", "spread")}
    assert envs == {"distributed": "GPU-fake-0003",                 # least loaded GPU, twice
                    # packed put the 16 pods 10 + 6 on GPUs 0 and 1: it tops up GPU 0, then GPU 1
                    "packed": "GPU-fake-0000,GPU-fake-0001",
                    "spread": "GPU-fake-0000,GPU-fake-0003"}        # two distinct physical GPUs
    fresh = Kubelet()
    fresh.register(DevicePlugin(make_gpus(8), replicas=10, allocation_policy="packed"))
    assert fresh.admit("a", 1)["envs"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-fake-0000"
    assert fresh.admit("b", 3)["envs"]["NVIDIA_VISIBLE_DEVICES"] == "GPU-fake-0000"   # packed fills GPU 0 first
    with pytest.raises(ValueError, match="invalid --shared-devices-allocation-policy option: tight"):
        DevicePlugin(make_gpus(8), replicas=10, allocation_policy="tight")


def test_plugin_options_and_prestart():
    plugin = DevicePlugin(make_gpus(8))
    assert plugin.register_request()["options"] == plugin.get_device_plugin_options() == \
        {"pre_start_required": False, "get_preferred_allocation_available": True}
    assert plugin.pre_start_container(["GPU-fake-0000"]) == {}
