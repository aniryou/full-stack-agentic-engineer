"""Getting capacity: GPU node pools that scale from zero - slowly, partially, or not at all.

The one idea: a Pending GPU pod is a request for a machine. The cluster autoscaler answers it by
simulating the pod on each pool's template node, adds nodes that take minutes to become
schedulable (VM, driver, device plugin), and removes them after they sit idle. Spot nodes can
vanish mid-job, and a gang needs ALL of its nodes at once - which is why atomic, queued
provisioning exists. Every duration, price and probability here is an input you choose; every
output is simulated.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class NodePool:
    name: str
    gpus_per_node: int = 8
    max_nodes: int = 16
    min_nodes: int = 0
    boot_s: int = 300                 # node created -> GPUs allocatable (simulated; measure yours)
    price_per_node_hr: float = 0.0    # illustrative input; real prices live in COMPUTE.md
    spot: bool = False
    queued: bool = False              # all-or-nothing provisioning (ProvisioningRequest / DWS flex-start)
    stockout: float = 0.0             # chance one node request is not granted in a tick (simulated)


def nodes_needed(pod_gpus: list, gpus_per_node: int) -> int:
    """First-fit decreasing onto empty template nodes: the autoscaler's bin-packing estimate."""
    free = []
    for g in sorted(pod_gpus, reverse=True):
        if g > gpus_per_node:
            raise ValueError(f"a {g}-GPU pod never fits a {gpus_per_node}-GPU node")
        i = next((i for i, f in enumerate(free) if f >= g), None)
        if i is None:
            free.append(gpus_per_node - g)
        else:
            free[i] -= g
    return len(free)


def least_waste(pools: list, pod_gpus: list, in_use: dict | None = None) -> NodePool | None:
    """Expander: the pool whose new nodes leave the fewest idle GPUs. (The real least-waste expander
    ranks idle CPU, then memory; on GPU pools the GPU is the scarce resource, so we rank that.)"""
    options = []
    for p in pools:
        if max(pod_gpus) <= p.gpus_per_node:
            n = nodes_needed(pod_gpus, p.gpus_per_node)
            if (in_use or {}).get(p.name, 0) + n <= p.max_nodes:
                options.append((n * p.gpus_per_node - sum(pod_gpus), n, p.name, p))
    return min(options)[3] if options else None


def provision(pool: NodePool, n: int, tick_s: int = 30, seed: int = 0, max_wait_s: int = 24 * 3600) -> dict | None:
    """When can a gang that needs n new nodes start? Each tick every missing node is granted with
    probability 1 - stockout. An ordinary pool creates - and bills - each node as it is granted; a
    queued pool holds the grants and creates all n together. None if not done by max_wait_s."""
    rng, grants, t = random.Random(seed), [], 0
    while len(grants) < n:
        if t > max_wait_s:
            return None
        grants += [t] * sum(rng.random() >= pool.stockout for _ in range(n - len(grants)))
        t += tick_s
    created = [max(grants)] * n if pool.queued else grants
    start = max(created) + pool.boot_s
    waiting = sum(start - c for c in created) / 3600        # node-hours billed before the gang can start
    return {"created_s": created, "gang_start_s": start, "waiting_node_h": waiting,
            "waiting_gpu_h": waiting * pool.gpus_per_node}


def gang_survival(nodes: int, hours: float, rate_per_node_hr: float) -> float:
    """P(no node of the gang is preempted during the run) = exp(-nodes * rate * hours)."""
    return math.exp(-nodes * rate_per_node_hr * hours)


def expected_runtime_h(work_h: float, nodes: int, rate_per_node_hr: float, restart_h: float = 0.0) -> float:
    """Expected wall-clock hours if any preemption restarts the whole gang from scratch (Poisson
    preemptions at L = nodes * rate, no checkpoints): (exp(L * T) - 1) * (1 / L + R)."""
    lam = nodes * rate_per_node_hr
    return work_h if lam == 0 else (math.exp(lam * work_h) - 1) * (1 / lam + restart_h)


def startup_latency(node_s: float = 0, driver_s: float = 0, image_gb: float = 0, pull_GBps: float = 1,
                    weights_gb: float = 0, load_GBps: float = 1, warmup_s: float = 0) -> dict:
    """Seconds from Pending to Ready as a sum of stages (inputs are yours to measure):
    new node + driver + image pull + weights load + warm-up (graph capture, first requests).
    Sizes are in gigabytes and rates in gigaBYTES per second (GB/s, not Gb/s: divide Gbit/s by 8)."""
    stages = {"node": node_s, "driver": driver_s, "image": image_gb / pull_GBps,
              "weights": weights_gb / load_GBps, "warmup": warmup_s}
    return {**stages, "total": sum(stages.values())}


@dataclass(eq=False)
class Job:
    name: str
    nodes: int                        # whole nodes, one pod per node
    hours: float
    arrive_s: int = 0
    start_s: int | None = None
    end_s: int | None = None
    restarts: int = 0


def simulate(pool: NodePool, jobs: list, until_s: int = 12 * 3600, tick_s: int = 30, unneeded_s: int = 600,
             delay_after_add_s: int = 600, spot_rate_per_node_hr: float = 0.0, seed: int = 0) -> dict:
    """A cluster-autoscaler loop for one pool and FIFO gang jobs (simulated). Each tick: finish jobs;
    reclaim Spot nodes (losing one node restarts its whole gang); start the head job if enough idle
    Ready nodes exist; request the missing nodes (an ordinary pool asks for what fits under
    max_nodes, a queued pool for all or nothing); grant requests; remove nodes idle for unneeded_s
    (--scale-down-unneeded-time) while nothing waits and no scale-up happened in the last
    delay_after_add_s (--scale-down-delay-after-add). Both default to the autoscaler's 10 minutes.
    Not modelled: the utilisation threshold (a node with no job is idle), max-node-provision-time,
    and several pools."""
    rng = random.Random(seed)
    nodes = [{"ready": 0, "job": None, "idle": 0} for _ in range(pool.min_nodes)]
    queue, running, log = sorted(jobs, key=lambda j: j.arrive_s), [], []
    want = granted = node_s = busy_s = 0
    last_add = -delay_after_add_s                       # time of the last scale-up request

    def release(t):
        for n in nodes:
            if n["job"] is not None and n["job"] not in running:
                n["job"], n["idle"] = None, t

    for t in range(0, until_s, tick_s):
        for j in [j for j in running if t >= j.end_s]:
            running.remove(j)
            log.append((t, f"{j.name} finished"))
        release(t)
        if pool.spot and spot_rate_per_node_hr:
            for n in [n for n in nodes if rng.random() < spot_rate_per_node_hr * tick_s / 3600]:
                nodes.remove(n)
                j = n["job"] if n["job"] in running else None
                if j:
                    running.remove(j)
                    j.restarts += 1
                    queue.insert(0, j)
                log.append((t, "Spot node reclaimed" + (f"; {j.name} restarts from scratch" if j else "")))
            release(t)
        waiting = [j for j in queue if j.arrive_s <= t]
        idle = [n for n in nodes if n["job"] is None and n["ready"] <= t]
        if waiting and len(idle) >= waiting[0].nodes:
            j = waiting.pop(0)
            for n in idle[:j.nodes]:
                n["job"] = j
            j.start_s, j.end_s = t, t + int(j.hours * 3600)
            queue.remove(j)
            running.append(j)
            log.append((t, f"{j.name} started on {j.nodes} node(s)"))
        missing = sum(j.nodes for j in waiting) - sum(n["job"] is None for n in nodes) - want
        ask = min(missing, pool.max_nodes - len(nodes) - want)
        if ask > 0 and (ask == missing or not pool.queued):
            want, last_add = want + ask, t
            log.append((t, f"scale-up: {ask} node(s) requested"))
        elif missing > 0 and pool.queued and not any("cannot" in m for _, m in log[-1:]):
            log.append((t, f"queued request cannot be met: {missing} node(s) needed, max_nodes {pool.max_nodes}"))
        granted += sum(rng.random() >= pool.stockout for _ in range(want - granted))
        create = want if granted == want else (0 if pool.queued else granted)
        if create:
            nodes += [{"ready": t + pool.boot_s, "job": None, "idle": t + pool.boot_s} for _ in range(create)]
            want, granted = want - create, granted - create
            log.append((t, f"{create} node(s) created, Ready at {t + pool.boot_s}s"))
        for n in [n for n in nodes if n["job"] is None and n["ready"] <= t and t - n["idle"] >= unneeded_s]:
            if not waiting and len(nodes) > pool.min_nodes and t - last_add >= delay_after_add_s:
                nodes.remove(n)
                log.append((t, "idle node removed"))
        node_s += tick_s * len(nodes)
        busy_s += tick_s * sum(n["job"] is not None for n in nodes)
    return {"jobs": {j.name: (j.start_s, j.end_s, j.restarts) for j in jobs}, "log": log,
            "node_h": node_s / 3600, "busy_node_h": busy_s / 3600,
            "cost": node_s / 3600 * pool.price_per_node_hr}
