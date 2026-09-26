"""Capacity chooser formulas pinned to hand-computed values; GKE plan helpers read the Terraform."""
import math

from k8sgpu import capacity, gke


def test_expected_runtime_renewal_formula():
    # 8 nodes x 0.02 /node-h = 0.16 /h; hourly checkpoints, 15 min restart:
    # 10 segments x (1/0.16 + 0.25) x (e^0.16 - 1) = 10 x 6.5 x 0.173511 = 11.278 h
    assert math.isclose(capacity.expected_runtime_h(10, 0.16, 1.0, 0.25), 11.2782, rel_tol=1e-4)
    # no checkpoints: one 10 h segment: 6.5 x (e^1.6 - 1) = 25.695 h - 2.6x the work
    assert math.isclose(capacity.expected_runtime_h(10, 0.16, None, 0.25), 25.6947, rel_tol=1e-4)
    assert capacity.expected_runtime_h(10, 0.0, None) == 10


def test_gang_rate_and_expected_wait():
    assert capacity.gang_rate(0.02, 8) == 0.16
    assert capacity.expected_wait_h(capacity.OPTIONS["spot"]) == (1 / 0.7 - 1)   # 0.43 retry-hours
    assert capacity.expected_wait_h(capacity.OPTIONS["flex-start"]) == 1.0


def test_checkpointed_batch_prefers_spot_and_serving_avoids_it():
    batch = capacity.Need("train", "a3-highgpu-8g", nodes=8, work_h=10, checkpoint_every_h=1)
    best = capacity.evaluate(batch)[0]
    assert best.option == "spot"
    # 88 $/h x 0.35 x 8 nodes x 11.278 h
    assert math.isclose(best.cost_usd, 88 * 0.35 * 8 * 11.2782, rel_tol=1e-3)
    serving = capacity.Need("serve", "g2-standard-4", nodes=2, work_h=730, serving=True, checkpoint_every_h=None)
    assert capacity.evaluate(serving)[0].option != "spot"


def test_flex_start_leases_and_reservations():
    long_run = capacity.Need("pretrain", "a3-highgpu-8g", nodes=4, work_h=400, checkpoint_every_h=2)
    flex = capacity.evaluate_option(long_run, capacity.OPTIONS["flex-start"])
    assert flex.feasible and any("3 leases" in n for n in flex.notes)          # 400 h / 168 h -> 3 leases
    unbroken = capacity.Need("x", "a3-highgpu-8g", nodes=4, work_h=400, checkpoint_every_h=None)
    assert not capacity.evaluate_option(unbroken, capacity.OPTIONS["flex-start"]).feasible
    idle = capacity.Need("x", "g2-standard-4", nodes=1, work_h=100, reserved_h=730)
    r = capacity.evaluate_option(idle, capacity.OPTIONS["reservation"])
    assert r.cost_usd == round(0.70 * 730, 2)                                   # pays for idle hours


def test_compute_class_fallback_order():
    prios = [{"spot": True}, {"spot": False}, {"flexStart": {"enabled": True}}]
    assert capacity.compute_class_pick(prios, {"spot": True, "on-demand": True}) == (0, "spot")
    assert capacity.compute_class_pick(prios, {"spot": False, "on-demand": False, "flex-start": True}) == (2, "flex-start")
    assert capacity.compute_class_pick(prios, {})[0] is None


def test_queued_provisioning_timeline_and_cold_start():
    ev = capacity.queued_provisioning_timeline(4, wait_h=2, run_h=200)
    assert ev[2][0] == 2 and "all 4 node(s)" in ev[2][1] and ev[-1][0] == 170 and "lease ends" in ev[-1][1]
    parts = capacity.cold_start_s(image_gb=10, pull_gbps=0.2, weights_gb=16, weights_gbps=0.5, engine_init_s=60,
                                  node_provision_s=150, driver_ready_s=90)
    assert parts["image"] == 50 and parts["weights"] == 32 and parts["total"] == 150 + 90 + 50 + 32 + 60
    assert capacity.startup_budget_ok(parts, probe_budget_s=600) and not capacity.startup_budget_ok(parts, 60)


def test_gke_helpers_read_the_terraform():
    v = gke.effective_vars()
    assert v["gpu_machine_type"] == "g2-standard-4" and v["gpu_spot"] is True and v["project_id"]
    names = {(t, n) for _, t, n in gke.tf_resources()}
    assert ("google_container_node_pool", "gpu_spot") in names and ("google_container_cluster", "lab") in names
    assert "l4-spot" in gke.plan_summary(v)
    cmds = gke.gcloud_equivalents({**v, "enable_flex_start_pool": True})
    assert any("--flex-start --enable-queued-provisioning" in c for c in cmds)


def test_checkpoint_cost_and_young_interval():
    # C = 3 min: each 1 h segment needs 1.05 h unbroken -> 10 x 6.5 x (e^0.168 - 1) = 11.891 h
    assert math.isclose(capacity.expected_runtime_h(10, 0.16, 1.0, 0.25, 0.05), 11.8909, rel_tol=1e-4)
    assert capacity.expected_runtime_h(10, 0.0, 1.0, checkpoint_cost_h=0.05) == 10.5   # 10 writes, no failures
    tau = capacity.young_interval_h(0.05, 0.16)                        # sqrt(2 x 0.05 / 0.16) = 0.7906 h
    assert math.isclose(tau, 0.7906, rel_tol=1e-4)
    near = capacity.expected_runtime_h(10, 0.16, tau, 0.25, 0.05)
    assert all(near <= capacity.expected_runtime_h(10, 0.16, t, 0.25, 0.05) for t in (0.4, 0.6, 1.0, 1.5, 3.0))


def test_probability_capacity_arrives_in_time():
    flex = capacity.OPTIONS["flex-start"]                              # queued: exponential wait, mean 1 h
    assert math.isclose(capacity.p_capacity_within(flex, 24), 1 - math.exp(-24), rel_tol=1e-12)
    slow = capacity.CapacityOption("q", 0.5, 1.0, 12.0, 0.0, 168.0, True, False)
    assert math.isclose(capacity.p_capacity_within(slow, 24), 1 - math.exp(-2), rel_tol=1e-12)   # 0.865
    assert math.isclose(24 / math.log(20), 8.0114, rel_tol=1e-4)      # mean wait at which P(24 h) = 0.95
    od = capacity.OPTIONS["on-demand"]                                 # retried hourly, p = 0.9 each
    assert math.isclose(capacity.p_capacity_within(od, 1.5), 1 - 0.1 ** 2, rel_tol=1e-12)
    assert capacity.p_capacity_within(capacity.OPTIONS["reservation"], 0) == 1.0
    assert capacity.p_capacity_within(flex, -1) == 0.0


def test_ties_are_reported_not_hidden():
    serve = capacity.Need("serve", "g2-standard-4", nodes=2, work_h=730, serving=True, checkpoint_every_h=None)
    evs = {e.option: e for e in capacity.evaluate(serve)}
    assert evs["on-demand"].cost_usd == evs["reservation"].cost_usd == 1022.0      # 0.70 x 2 x 730
    assert set(capacity.defensible(serve)) == {"on-demand", "reservation"}
    assert not any("one by one" in n for n in evs["on-demand"].notes)              # servers are not gangs
    pretrain = capacity.Need("pretrain", "a3-highgpu-8g", nodes=8, work_h=72, checkpoint_every_h=None, deadline_h=96)
    assert capacity.defensible(pretrain) == ["flex-start"]                         # P(start in 24 h) ~ 1 at mean 1 h
    slow = dict(capacity.OPTIONS, **{"flex-start": capacity.CapacityOption(
        "flex-start", 0.50, 1.00, 12.0, 0.0, 168.0, True, False)})
    ranked = capacity.evaluate(pretrain, slow)
    flex = next(e for e in ranked if e.option == "flex-start")
    assert flex.p_on_time == round(1 - math.exp(-2), 4) and ranked[0].option == "reservation"
    assert set(capacity.defensible(pretrain, options=slow)) == {"reservation", "on-demand"}
