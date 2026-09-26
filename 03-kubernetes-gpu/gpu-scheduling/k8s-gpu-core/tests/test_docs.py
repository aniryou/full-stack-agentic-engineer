"""The primer's numbers come from this package, and its manifests use the API versions we verified."""
import pathlib
import re

import pytest

from gpusched import (GPU, ClusterQueue, Job, Kueue, NodePool, Quota, Scheduler, best_fit, expected_runtime_h,
                      gang_survival, gpu_node, gpu_pod, interleave, least_allocated, least_free_capacity,
                      make_cluster, most_allocated, place_gang, provision, simulate, startup_latency, stranded_gpus)

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIMER_PATH = ROOT.parent / "PRIMER.md"
if not PRIMER_PATH.exists():          # the core copied out of the repo: skip the doc checks, run the rest
    pytest.skip(f"{PRIMER_PATH} not found (these tests check the primer next to this core)", allow_module_level=True)
PRIMER = PRIMER_PATH.read_text(encoding="utf-8")
SOURCES = [PRIMER] + [p.read_text(encoding="utf-8") for p in sorted((ROOT / "notebooks_src").glob("*.py"))]

# apiVersion/kind pairs checked against upstream types and FACTS.md on 2026-09-26
KNOWN = {
    ("v1", "Pod"), ("batch/v1", "Job"), ("scheduling.k8s.io/v1", "PriorityClass"),
    ("resource.k8s.io/v1", "ResourceClaimTemplate"), ("resource.k8s.io/v1", "ResourceClaim"),
    ("resource.k8s.io/v1", "DeviceClass"), ("kubescheduler.config.k8s.io/v1", "KubeSchedulerConfiguration"),
    ("jobset.x-k8s.io/v1alpha2", "JobSet"), ("leaderworkerset.x-k8s.io/v1", "LeaderWorkerSet"),
    ("autoscaling.x-k8s.io/v1", "ProvisioningRequest"),
} | {("kueue.x-k8s.io/v1beta2", k) for k in ("ClusterQueue", "LocalQueue", "ResourceFlavor", "WorkloadPriorityClass",
                                             "Topology", "AdmissionCheck", "ProvisioningRequestConfig", "Cohort")}


def _manifests(text):
    """(apiVersion, kind) of every manifest written in the docs (YAML blocks, commented or not, and dicts)."""
    pairs = re.findall(r"apiVersion:\s*([\w./-]+)\s*\n\s*#?\s*kind:\s*(\w+)", text)
    pairs += re.findall(r'"apiVersion":\s*"([\w./-]+)",\s*"kind":\s*"(\w+)"', text)
    return pairs


def test_every_manifest_in_the_docs_uses_a_verified_api_version():
    found = [pair for text in SOURCES for pair in _manifests(text)]
    assert len(found) >= 5
    assert set(found) <= KNOWN, set(found) - KNOWN


def test_core_kind_manifests_in_the_primer_validate_against_kubernetes_1_34():
    yaml = pytest.importorskip("yaml")
    kv = pytest.importorskip("kubernetes_validate")
    blocks = [b for b in re.findall(r"```yaml\n(.*?)```", PRIMER, flags=re.S) if "apiVersion: resource.k8s.io" in b]
    assert blocks
    for block in blocks:
        kv.validate(yaml.safe_load(block), "1.34.0", strict=True)


def test_primer_numbers_are_the_simulators_numbers():
    node = gpu_node("n", 8)
    node.pods.append(gpu_pod("busy", 6))
    assert (most_allocated(gpu_pod("p", 1), node, ((GPU, 1),)), least_allocated(gpu_pod("p", 1), node, ((GPU, 1),))) == (87, 12)

    spread = make_cluster(hosts=4)
    s = Scheduler(spread)
    s.submit(*[gpu_pod(f"i{i}", 1) for i in range(16)])
    s.run()
    assert stranded_gpus(spread, 8) == 16

    rack = {"n1": 3, "n2": 3, "n3": 2, "n4": 1}
    assert list(best_fit(rack, 7).values()) == [3, 3, 1] and list(least_free_capacity(rack, 7).values()) == [1, 2, 3, 1]

    deadlock = make_cluster(hosts=3, gpus=4)
    s = Scheduler(deadlock)
    s.submit(*interleave([gpu_pod(f"a{i}", 2) for i in range(4)], [gpu_pod(f"b{i}", 2) for i in range(4)]))
    s.run()
    layout = "   ".join(f"{n.name[-2:]}: " + " ".join(p.name for p in n.pods) for n in deadlock.nodes.values())
    assert layout in PRIMER                                            # 4.1's diagram

    fleet = make_cluster(blocks=2, subblocks=2, hosts=4)               # 5.2's worked placement
    for host, g in {"b0-s0-h0": 8, "b0-s0-h1": 8, "b0-s0-h2": 8, "b0-s1-h0": 8, "b1-s0-h0": 8, "b1-s0-h1": 1}.items():
        fleet.bind(gpu_pod(f"x-{host}", g), host)

    def per_subblock(placement):
        counts = {}
        for node in placement.values():
            sb = fleet.nodes[node].topology[1]
            counts[sb] = counts.get(sb, 0) + 1
        return " + ".join(f"{k} in {sb}" for sb, k in sorted(counts.items()))
    five = place_gang(fleet, [gpu_pod(f"k{i}", 8) for i in range(5)], preferred="subblock")
    three = place_gang(fleet, [gpu_pod(f"u{i}", 8) for i in range(3)])
    assert f"gives {per_subblock(five)} |" in PRIMER and f"so {per_subblock(three)} —" in PRIMER

    def pair(a_borrow=None, b_lend=None):
        a = ClusterQueue("a", {"f": {"cpu": Quota(9, borrowing_limit=a_borrow)}}, cohort="ab")
        b = ClusterQueue("b", {"f": {"cpu": Quota(12, lending_limit=b_lend)}}, cohort="ab")
        return Kueue([a, b]).available(a, "f", "cpu")
    assert (pair(), pair(a_borrow=1), pair(b_lend=1)) == (21, 10, 10)

    run = simulate(NodePool("l4", 1, max_nodes=2, boot_s=300), [Job("j", 1, 1.0)], until_s=3 * 3600)
    assert run["node_h"] == pytest.approx(1.25) and "1.25 node-hours billed for 1 hour of work" in PRIMER

    table = [(n, gang_survival(n, 24, 0.005), expected_runtime_h(24, n, 0.005, 0.25)) for n in (1, 4, 16)]
    for n, survives, hours in table:
        assert f"| {n} | {survives:.1%} | {hours:.1f} h |" in PRIMER

    kw = dict(gpus_per_node=8, boot_s=300, stockout=0.97)
    ordinary = provision(NodePool("a", **kw), 16, tick_s=60, seed=0)
    queued = provision(NodePool("q", queued=True, **kw), 16, tick_s=60, seed=0)
    assert f"{ordinary['gang_start_s'] / 3600:.2f} h" in PRIMER
    assert f"**{ordinary['waiting_node_h']:.1f} node-hours ({ordinary['waiting_gpu_h']:.0f} GPU-hours)**" in PRIMER
    assert f"**{queued['waiting_node_h']:.1f}**" in PRIMER
    runs = [provision(NodePool("a", **kw), 16, tick_s=60, seed=seed) for seed in range(200)]
    mean_node_h = sum(r["waiting_node_h"] for r in runs) / len(runs)
    mean_start_h = sum(r["gang_start_s"] for r in runs) / len(runs) / 3600
    assert f"over 200 seeds the ordinary pool averages **{mean_node_h:.1f} node-hours** and a" in PRIMER
    assert f"{mean_start_h:.2f} h start" in PRIMER

    cold = startup_latency(node_s=150, driver_s=90, image_gb=12, pull_GBps=0.25, weights_gb=16, load_GBps=0.5, warmup_s=60)
    warm = startup_latency(image_gb=12, pull_GBps=2.0, weights_gb=16, load_GBps=4.0, warmup_s=60)
    assert f"**{cold['total']:.0f} s**" in PRIMER and f"**{warm['total']:.0f} s**" in PRIMER
