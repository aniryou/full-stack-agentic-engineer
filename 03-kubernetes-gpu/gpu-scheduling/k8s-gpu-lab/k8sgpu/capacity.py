"""How to get GPUs: on-demand vs Spot vs DWS flex-start vs reservations — cost against obtainability.

The one idea: the price per GPU-hour is only one term. What you pay for a *result* is
``price x nodes x expected runtime``, and the expected runtime of an interruptible gang job
grows with the interruption rate x nodes x work lost per interruption. With this module's
illustrative defaults (0.02 reclaims per node-hour, 15 min restarts, Spot at 0.35 x list) a
10-hour, 8-node Spot job with hourly checkpoints takes 11.3 h and costs about 40% of on-demand
(0.35 x 11.3 h / 10 h); the same job with no checkpoints takes 25.7 h, 2.6x its 10 h of work.
Meanwhile the scarce shapes (8x H100) may simply not be obtainable on demand, which is what
queued, all-or-nothing provisioning (DWS flex-start) and reservations exist for — and a queue
has a wait you cannot predict exactly, so a hard deadline turns into a *probability* of
starting in time. ``evaluate()`` puts those terms side by side for one workload.

Every price and rate below is an assumption you should replace (``verify``); the formulas are
the point. The hazard rate is illustrative and deliberately higher than the 0.005/node-h the
topic primer uses in §7.2 (both are made up; real Spot reclaim rates vary by zone and shape).
``expected_runtime_h`` is the exact renewal result for Poisson interruptions with periodic
checkpoints; the checkpoint interval that minimises it is layer 01 §7.2's Young/Daly interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---- prices (USD per machine-hour, on-demand, us-central1, Sep 2026 — verify) -------------------
ON_DEMAND_USD_H: dict[str, float] = {
    "g2-standard-4": 0.70,        # 1x L4
    "g2-standard-48": 4.0,        # 4x L4 (verify)
    "n1-standard-8-t4": 0.45,     # N1 + 1x T4
    "a2-highgpu-1g": 3.7,         # 1x A100 40GB
    "a2-ultragpu-1g": 5.0,        # 1x A100 80GB
    "a3-highgpu-8g": 88.0,        # 8x H100 80GB
    "e2-standard-4": 0.134,       # a CPU system node (verify)
}
GKE_CLUSTER_FEE_USD_H = 0.10      # per cluster; the GKE free tier credit covers one zonal cluster (verify)


@dataclass(frozen=True)
class CapacityOption:
    """One way to obtain nodes. Numbers are illustrative defaults (verify for your region/SKU)."""
    name: str
    price_multiplier: float            # x on-demand list price
    obtain_probability: float          # chance a request is satisfied right now
    wait_h: float                      # mean wait for capacity; queued (atomic) options: exponential with this mean
    interruptions_per_node_h: float    # reclaim hazard (Spot)
    max_runtime_h: float | None        # flex-start leases end (up to 7 days)
    atomic: bool                       # all N nodes arrive together (queued provisioning)
    pay_when_idle: bool                # reservations bill whether used or not
    note: str = ""


OPTIONS: dict[str, CapacityOption] = {o.name: o for o in [
    CapacityOption("on-demand", 1.00, 0.90, 0.0, 0.0, None, False, False,
                   "pay-as-you-go; can be unavailable for scarce shapes (stockout)"),
    CapacityOption("spot", 0.35, 0.70, 0.0, 0.02, None, False, False,
                   "60-91% off; reclaimed with ~30 s notice; no availability guarantee"),
    CapacityOption("flex-start", 0.50, 1.00, 1.0, 0.0, 168.0, True, False,
                   "DWS: queued, all-or-nothing, runs up to 7 days; discount is an assumption (verify)"),
    CapacityOption("reservation", 1.00, 1.00, 0.0, 0.0, None, True, True,
                   "capacity held for you (committed-use discounts lower the price); idle hours still billed"),
]}


# ---- the formulas ----------------------------------------------------------------------------------
def expected_runtime_h(work_h: float, rate_per_h: float, checkpoint_every_h: float | None,
                       restart_h: float = 0.25, checkpoint_cost_h: float = 0.0) -> float:
    """Expected wall time to finish ``work_h`` of work under Poisson interruptions.

    Work is split into segments of ``tau`` hours (the checkpoint interval; the whole job if it
    cannot checkpoint). Each segment also writes a checkpoint taking ``checkpoint_cost_h`` (C), so
    it needs ``tau + C`` uninterrupted hours; under failure rate L with restart cost R that takes
    ``(1/L + R)(e^(L (tau + C)) - 1)`` hours on average, and there are ``work_h / tau`` segments.
    L -> 0 gives back ``work_h + (work_h / tau) C``. With C > 0 there is a best interval, close to
    ``young_interval_h(C, L)``.
    """
    tau = work_h if not checkpoint_every_h else min(checkpoint_every_h, work_h)
    cost = checkpoint_cost_h if checkpoint_every_h else 0.0      # no checkpoints, nothing to write
    if rate_per_h <= 0:
        return float(work_h + (work_h / tau) * cost)
    return (work_h / tau) * (1.0 / rate_per_h + restart_h) * math.expm1(rate_per_h * (tau + cost))


def young_interval_h(checkpoint_cost_h: float, rate_per_h: float) -> float:
    """Young's first-order optimum checkpoint interval, ``sqrt(2 C / L)`` = ``sqrt(2 C M)`` with
    M = 1/L the mean time between interruptions (layer 01 §7.2, ``roofline.reliability.young_daly_interval``)."""
    return math.sqrt(2.0 * checkpoint_cost_h / rate_per_h)


def gang_rate(per_node_rate: float, nodes: int) -> float:
    """A gang is interrupted when *any* member is: the rates add up."""
    return per_node_rate * nodes


def expected_wait_h(opt: CapacityOption, retry_every_h: float = 1.0) -> float:
    """Queued options wait ``wait_h``; retried ones also expect ``(1/p - 1)`` failed attempts."""
    retries = (1.0 / opt.obtain_probability - 1.0) * retry_every_h if opt.obtain_probability < 1 else 0.0
    return opt.wait_h + retries


def p_capacity_within(opt: CapacityOption, slack_h: float, retry_every_h: float = 1.0) -> float:
    """Probability the capacity has arrived within ``slack_h`` hours.

    A queued option (DWS flex-start) waits an uncertain time; modelled as exponential with mean
    ``wait_h``: ``1 - e^(-slack / wait_h)``. A retried option (on-demand, Spot) gets one attempt
    now and one every ``retry_every_h``, each succeeding with ``obtain_probability``:
    ``1 - (1 - p)^attempts``. A reservation is there already."""
    if slack_h < 0:
        return 0.0
    if opt.atomic and opt.wait_h > 0:
        return 1.0 - math.exp(-slack_h / opt.wait_h)
    if opt.obtain_probability >= 1.0:
        return 1.0
    attempts = math.floor(slack_h / retry_every_h) + 1
    return 1.0 - (1.0 - opt.obtain_probability) ** attempts


@dataclass(frozen=True)
class Need:
    """What the workload needs from capacity."""
    name: str
    machine: str
    nodes: int
    work_h: float
    checkpoint_every_h: float | None = 1.0
    restart_h: float = 0.25
    deadline_h: float | None = None
    serving: bool = False              # must stay up; interruptions are outages, not delays
    reserved_h: float | None = None    # for reservations: hours you commit to pay for
    checkpoint_cost_h: float = 0.0     # time to write one checkpoint
    on_time_target: float = 0.95       # with a deadline: the chance of starting in time you require


@dataclass
class Evaluation:
    option: str
    feasible: bool
    wait_h: float
    run_h: float
    cost_usd: float
    notes: list[str] = field(default_factory=list)
    p_on_time: float | None = None     # chance capacity arrives early enough to meet the deadline

    @property
    def finish_h(self) -> float:
        return self.wait_h + self.run_h


def evaluate_option(need: Need, opt: CapacityOption, price_usd_h: float | None = None) -> Evaluation:
    price = (price_usd_h if price_usd_h is not None else ON_DEMAND_USD_H[need.machine]) * opt.price_multiplier
    notes: list[str] = []
    rate = gang_rate(opt.interruptions_per_node_h, need.nodes)
    # a server is not a job that "finishes later": interruptions are outages, not extra hours
    run = need.work_h if need.serving else expected_runtime_h(need.work_h, rate, need.checkpoint_every_h,
                                                              need.restart_h, need.checkpoint_cost_h)
    wait = first_wait = expected_wait_h(opt)
    feasible = True
    if opt.max_runtime_h and run > opt.max_runtime_h:
        if need.checkpoint_every_h is None:
            feasible = False
            notes.append(f"needs {run:.0f} h unbroken; leases end after {opt.max_runtime_h:.0f} h")
        else:
            leases = math.ceil(run / opt.max_runtime_h)
            wait *= leases
            notes.append(f"{leases} leases of <= {opt.max_runtime_h:.0f} h, re-queued between them")
    billed_h = run
    if opt.pay_when_idle:
        billed_h = max(run, need.reserved_h or run)
        if need.reserved_h and need.reserved_h > run:
            notes.append(f"pays for {need.reserved_h:.0f} reserved h, uses {run:.0f}")
    if need.serving and opt.interruptions_per_node_h > 0:
        outages = rate * need.work_h
        notes.append(f"~{outages:.1f} node reclaims over {need.work_h:.0f} h: pair with on-demand fallback")
    if need.serving and opt.atomic and opt.max_runtime_h:
        notes.append("leases end: nodes recycle every few days")
    if not opt.atomic and need.nodes > 1 and not need.serving:
        notes.append("nodes arrive one by one: a gang can sit half-provisioned")
    if rate > 0 and need.checkpoint_every_h is None and not need.serving:
        notes.append("no checkpoints: every interruption restarts from zero")
    cost = price * need.nodes * billed_h
    p_on_time = None
    if need.deadline_h is not None:
        # later leases re-queue too; count their expected waits against the slack, then ask how
        # likely the *first* capacity arrives in the time that is left
        slack = need.deadline_h - run - (wait - first_wait)
        p_on_time = round(p_capacity_within(opt, slack), 4)
        if wait + run > need.deadline_h:
            notes.append(f"misses the {need.deadline_h:.0f} h deadline (finishes ~{wait + run:.1f} h)")
        elif p_on_time < need.on_time_target:
            notes.append(f"starts in time with probability {p_on_time:.2f} < {need.on_time_target:.2f} "
                         f"(queue wait is uncertain; slack {slack:.0f} h)")
    return Evaluation(opt.name, feasible, round(wait, 3), round(run, 3), round(cost, 2), notes, p_on_time)


def _constraints(need: Need, e: Evaluation) -> tuple[bool, bool, bool]:
    """(infeasible, late, risky serving): the hard constraints, in the order they rank."""
    late = need.deadline_h is not None and (e.finish_h > need.deadline_h
                                            or (e.p_on_time or 0.0) < need.on_time_target)
    risky_serving = need.serving and OPTIONS.get(e.option, OPTIONS["on-demand"]).interruptions_per_node_h > 0
    return (not e.feasible, late, risky_serving)


def evaluate(need: Need, options: dict[str, CapacityOption] | None = None,
             price_usd_h: float | None = None) -> list[Evaluation]:
    """All options for one need, best first: feasible, on time (with at least ``on_time_target``
    probability), not an interruptible server, then cheapest; equal costs are broken by the
    surer start (``p_on_time``) and then the shorter expected wait."""
    evs = [evaluate_option(need, o, price_usd_h) for o in (options or OPTIONS).values()]
    return sorted(evs, key=lambda e: (*_constraints(need, e), e.cost_usd,
                                      -(e.p_on_time if e.p_on_time is not None else 1.0), e.wait_h))


def defensible(need: Need, rel_tol: float = 0.02, options: dict[str, CapacityOption] | None = None) -> list[str]:
    """Options the model cannot tell apart from the best: they meet the same hard constraints
    and cost within ``rel_tol`` of it. Choosing among them is a judgement (a reservation removes
    stockout risk; on-demand keeps you uncommitted) the model does not make for you."""
    evs = evaluate(need, options)
    best = evs[0]
    return [e.option for e in evs
            if _constraints(need, e) == _constraints(need, best) and e.cost_usd <= best.cost_usd * (1 + rel_tol)]


def format_evaluations(evs: list[Evaluation]) -> str:
    lines = [f"{'option':12} {'feasible':8} {'wait h':>7} {'run h':>7} {'finish h':>8} {'on time':>7} {'cost $':>9}  notes"]
    for e in evs:
        on_time = "-" if e.p_on_time is None else f"{e.p_on_time:.2f}"
        lines.append(f"{e.option:12} {str(e.feasible):8} {e.wait_h:7.2f} {e.run_h:7.2f} {e.finish_h:8.2f} "
                     f"{on_time:>7} {e.cost_usd:9.2f}  {'; '.join(e.notes)}")
    return "\n".join(lines)


# ---- GKE ComputeClass: an ordered fallback ladder ----------------------------------------------------
def rung_kind(priority: dict) -> str:
    """Which capacity type a ComputeClass priority asks for."""
    if priority.get("reservations"):
        return "reservation"
    if (priority.get("flexStart") or {}).get("enabled"):
        return "flex-start"
    return "spot" if priority.get("spot") else "on-demand"


def compute_class_pick(priorities: list[dict], available: dict[str, bool],
                       when_unsatisfiable: str = "DoNotScaleUp") -> tuple[int | None, str]:
    """The rung GKE provisions from: the first whose capacity type is available right now.
    With ``DoNotScaleUp`` and nothing available, the pod stays Pending (and GKE keeps retrying)."""
    for i, p in enumerate(priorities):
        if available.get(rung_kind(p), False):
            return i, rung_kind(p)
    if when_unsatisfiable == "ScaleUpAnyway":
        return None, "general-purpose node without the GPU (useless for a GPU pod)"
    return None, "Pending: no rung available (DoNotScaleUp)"


# ---- DWS flex-start with queued provisioning, as a timeline ------------------------------------------
def queued_provisioning_timeline(nodes: int, wait_h: float, run_h: float,
                                 max_run_h: float = 168.0) -> list[tuple[float, str]]:
    """What Kueue + a ProvisioningRequest (class queued-provisioning.gke.io) do, in order."""
    ev = [(0.0, "Job created (suspended); Kueue reserves quota -> QuotaReserved"),
          (0.0, "Kueue creates one ProvisioningRequest covering every pod (one PodTemplate per pod set); "
                "AdmissionCheck: Pending"),
          (wait_h, f"DWS finds capacity for all {nodes} node(s) at once -> Provisioned=True"),
          (wait_h, "AdmissionCheck Ready -> Workload Admitted -> Job unsuspended, pods bind")]
    if run_h <= max_run_h:
        ev.append((wait_h + run_h, "Job finishes; the nodes scale back down to zero"))
    else:
        ev.append((wait_h + max_run_h, f"lease ends after {max_run_h:.0f} h: nodes are removed mid-run "
                                        "(checkpoint and resubmit)"))
    return ev


# ---- cold start: where the minutes go ------------------------------------------------------------------
def cold_start_s(*, node_provision_s: float = 150.0, driver_ready_s: float = 90.0, image_gb: float = 10.0,
                 pull_gbps: float = 0.15, image_streaming: bool = False, streaming_start_s: float = 15.0,
                 weights_gb: float = 3.0, weights_gbps: float = 0.5, engine_init_s: float = 60.0) -> dict[str, float]:
    """A scale-from-zero serving pod's time to Ready, by stage (all defaults illustrative; the topic
    primer's §8 example uses other illustrative inputs, 12 GB at 0.25 GB/s). For scale: the
    vllm/vllm-openai:v0.30.0 amd64 image is 8.7 GB compressed (registry manifest, 2026-09-26).

    Image streaming (GKE, Artifact Registry images) starts the container after fetching
    metadata instead of the whole image; weights stream at ``weights_gbps`` (GCS FUSE with
    parallel downloads, Hyperdisk ML, or a model streamer)."""
    parts = {"node provision": node_provision_s,
             "GPU driver ready": driver_ready_s,
             "image": streaming_start_s if image_streaming else image_gb / pull_gbps,
             "weights": weights_gb / weights_gbps,
             "engine init": engine_init_s}
    parts["total"] = sum(parts.values())
    return {k: round(v, 1) for k, v in parts.items()}


def startup_budget_ok(parts: dict[str, float], probe_budget_s: float) -> bool:
    """A startup probe runs once the container has started: it must cover weights + engine init."""
    return probe_budget_s >= parts["weights"] + parts["engine init"]
