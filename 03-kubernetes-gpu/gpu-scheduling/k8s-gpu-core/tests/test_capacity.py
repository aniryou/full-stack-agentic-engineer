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
    assert env == "GPU-fake-0000"                             # two "GPUs", one physical device


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
    assert expected_runtime_h(10, 16, 0.01) > 3 * expected_runtime_h(10, 4, 0.01)  # hazard grows with gang size


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


def test_an_ordinary_pool_scales_up_a_gang_it_cannot_finish():
    job = [Job("train", 4, 1.0)]
    ordinary = simulate(NodePool("a3", 8, max_nodes=3, boot_s=300), job, until_s=7200)
    queued = simulate(NodePool("a3q", 8, max_nodes=3, boot_s=300, queued=True), [Job("train", 4, 1.0)], until_s=7200)
    assert ordinary["jobs"]["train"][0] is None and ordinary["node_h"] == pytest.approx(6.0)   # 3 idle nodes
    assert queued["jobs"]["train"][0] is None and queued["node_h"] == 0


def test_startup_latency_is_a_sum_of_stages():
    s = startup_latency(node_s=240, driver_s=60, image_gb=10, pull_gbps=0.5, weights_gb=16, load_gbps=1)
    assert s["image"] == 20 and s["weights"] == 16 and s["total"] == 336
