"""How to get GPUs: on-demand vs Spot vs DWS flex-start vs reservations — cost against obtainability.

The one idea: the price per GPU-hour is only one term. What you pay for a *result* is
``price x nodes x expected runtime``, and the expected runtime of an interruptible gang job
grows with the interruption rate x nodes x work lost per interruption. A 10-hour, 8-node job
on Spot with hourly checkpoints costs about a third of on-demand; the same job with no
checkpoints can take 2.5x as long. Meanwhile the scarce shapes (8x H100) may simply not be
obtainable on demand, which is what queued, all-or-nothing provisioning (DWS flex-start) and
reservations exist for. ``evaluate()`` puts those terms side by side for one workload.

Every price and rate below is an assumption you should replace (``verify``); the formulas are
the point. ``expected_runtime_h`` is the exact renewal result for Poisson interruptions with
periodic checkpoints.
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
    wait_h: float                      # expected wait for capacity when it does come
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
                       restart_h: float = 0.25) -> float:
    """Expected wall time to finish ``work_h`` of work under Poisson interruptions.

    Work is split into segments of ``tau`` hours (the checkpoint interval; the whole job if it
    cannot checkpoint). A segment of length tau under failure rate L with restart cost R takes
    ``(1/L + R)(e^(L tau) - 1)`` hours on average; there are ``work_h / tau`` segments.
    L -> 0 gives back ``work_h``.
    """
    if rate_per_h <= 0:
        return float(work_h)
    tau = work_h if not checkpoint_every_h else min(checkpoint_every_h, work_h)
    return (work_h / tau) * (1.0 / rate_per_h + restart_h) * math.expm1(rate_per_h * tau)


def gang_rate(per_node_rate: float, nodes: int) -> float:
    """A gang is interrupted when *any* member is: the rates add up."""
    return per_node_rate * nodes


def expected_wait_h(opt: CapacityOption, retry_every_h: float = 1.0) -> float:
    """Queued options wait ``wait_h``; retried ones also expect ``(1/p - 1)`` failed attempts."""
    retries = (1.0 / opt.obtain_probability - 1.0) * retry_every_h if opt.obtain_probability < 1 else 0.0
    return opt.wait_h + retries


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


@dataclass
class Evaluation:
    option: str
    feasible: bool
    wait_h: float
    run_h: float
    cost_usd: float
    notes: list[str] = field(default_factory=list)

    @property
    def finish_h(self) -> float:
        return self.wait_h + self.run_h


def evaluate_option(need: Need, opt: CapacityOption, price_usd_h: float | None = None) -> Evaluation:
    price = (price_usd_h if price_usd_h is not None else ON_DEMAND_USD_H[need.machine]) * opt.price_multiplier
    notes: list[str] = []
    rate = gang_rate(opt.interruptions_per_node_h, need.nodes)
    run = expected_runtime_h(need.work_h, rate, need.checkpoint_every_h, need.restart_h)
    wait = expected_wait_h(opt)
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
    if not opt.atomic and need.nodes > 1:
        notes.append("nodes arrive one by one: a gang can sit half-provisioned")
    if rate > 0 and need.checkpoint_every_h is None:
        notes.append("no checkpoints: every interruption restarts from zero")
    cost = price * need.nodes * billed_h
    if need.deadline_h is not None and wait + run > need.deadline_h:
        notes.append(f"misses the {need.deadline_h:.0f} h deadline (finishes ~{wait + run:.1f} h)")
    return Evaluation(opt.name, feasible, round(wait, 3), round(run, 3), round(cost, 2), notes)


def evaluate(need: Need, options: dict[str, CapacityOption] | None = None,
             price_usd_h: float | None = None) -> list[Evaluation]:
    """All options for one need, best first: feasible, meets the deadline, then cheapest."""
    evs = [evaluate_option(need, o, price_usd_h) for o in (options or OPTIONS).values()]

    def key(e: Evaluation):
        late = need.deadline_h is not None and e.finish_h > need.deadline_h
        risky_serving = need.serving and OPTIONS.get(e.option, OPTIONS["on-demand"]).interruptions_per_node_h > 0
        return (not e.feasible, late, risky_serving, e.cost_usd)
    return sorted(evs, key=key)


def format_evaluations(evs: list[Evaluation]) -> str:
    lines = [f"{'option':12} {'feasible':8} {'wait h':>7} {'run h':>7} {'finish h':>8} {'cost $':>9}  notes"]
    for e in evs:
        lines.append(f"{e.option:12} {str(e.feasible):8} {e.wait_h:7.2f} {e.run_h:7.2f} {e.finish_h:8.2f} "
                     f"{e.cost_usd:9.2f}  {'; '.join(e.notes)}")
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
          (0.0, "Kueue creates a ProvisioningRequest for all pods (AdmissionCheck: Pending)"),
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
    """A scale-from-zero serving pod's time to Ready, by stage (all defaults illustrative).

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
