"""DCGM-exporter metrics -> what the GPU is actually doing, and when to page someone.

The one idea: **"GPU utilization" is not utilisation.** ``DCGM_FI_DEV_GPU_UTIL`` (the number
nvidia-smi shows) is the fraction of time *at least one kernel* was running — a single small kernel
occupying one SM out of 58 reads 100 %. The profiling (DCP) fields answer the real questions:

    DCGM_FI_PROF_SM_ACTIVE          fraction of SMs with at least one warp resident   (how wide)
    DCGM_FI_PROF_SM_OCCUPANCY       resident warps / maximum warps                    (how deep)
    DCGM_FI_PROF_PIPE_TENSOR_ACTIVE tensor-core pipe busy                             (matmul-heavy?)
    DCGM_FI_PROF_DRAM_ACTIVE        memory interface busy                             (bandwidth-bound?)

(ratios 0-1, whatever their HELP text says about percent). Health is a separate set of fields —
XID errors, ECC/row remapping, clock-event (throttle) reasons, temperature, PCIe replays — and a
separate decision: drain a node, reset a GPU, or fix an application (primer §8).

This module parses the Prometheus text the exporter serves (``curl localhost:9400/metrics``), groups
it per GPU, derives signals, triages XIDs and evaluates alert rules offline. Each rule carries the
PromQL you would deploy (Google Managed Prometheus ``ClusterRules`` or a ``PrometheusRule``) and a
Python check that encodes the same condition, so the fixture exercises both.

**Prerequisite: the exporter must export the fields.** No exporter configuration serves all of them
by default (upstream files read 2026-09-26; verify for your versions — :data:`EXPORTED_BY`):

* stock dcgm-exporter (``etc/default-counters.csv``) leaves out ``DCGM_FI_DEV_CLOCKS_EVENT_REASONS``
  and has ``DCGM_FI_PROF_SM_ACTIVE``/``SM_OCCUPANCY`` commented out: the thermal and HW-slowdown
  rules cannot fire, nor can the busy-but-underfilled rule, and :func:`bottleneck` answers "unknown";
* Google's GMP DCGM example list (which GKE's managed DCGM package resembles — verify its field list)
  has the profiling fields but no XID, row-remap, clock-event or DRAM fields: none of the health rules
  can fire on it;
* ``deploy/any-gpu/dcgm-counters.csv`` is stock plus the three missing fields: pass it to a
  self-managed exporter (``-f``/``--collectors`` or ``DCGM_EXPORTER_COLLECTORS``; profiling fields need
  ``SYS_ADMIN``). :func:`rules_that_cannot_fire` tells you which rules a given scrape can never trigger.

**Deploying the rules.** Rules only *evaluate*; firing alerts go to an Alertmanager (GMP's managed one,
configured by a Secret in ``gmp-public``; see ``deploy/gke/README.md``), not to Cloud Monitoring
alerting policies. On GMP a namespaced ``Rules`` object only sees metrics from its own namespace, and
the exporter never runs in the workload namespace, so :func:`rules_manifest` emits the cluster-scoped
``ClusterRules``. When a Prometheus scrapes the exporter, its own target labels (``namespace``,
``pod``) win and the exporter's workload labels become ``exported_namespace``/``exported_pod``
(Prometheus ``honor_labels: false``, the default in GMP and in the dcgm-exporter Helm chart — verify).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------- Prometheus text format
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+-?\d+)?\s*$")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


@dataclass
class Sample:
    name: str
    labels: dict
    value: float


def _unescape(v: str) -> str:
    return v.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def parse_prometheus(text: str) -> tuple[list[Sample], dict]:
    """Samples plus ``{metric: {"help": ..., "type": ...}}`` from the text exposition format."""
    samples, meta = [], {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                meta.setdefault(parts[2], {})[parts[1].lower()] = parts[3] if len(parts) > 3 else ""
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        labels = {k: _unescape(v) for k, v in _LABEL.findall(m[2] or "")}
        try:
            value = float(m[3])
        except ValueError:
            continue
        samples.append(Sample(m[1], labels, value))
    return samples, meta


# --------------------------------------------------------------------------- per-GPU snapshots
POD_LABELS = ("pod", "namespace", "container", "exported_pod", "exported_namespace", "exported_container")


@dataclass
class GpuSnapshot:
    host: str
    gpu: str
    instance: str | None  # MIG GPU instance id, when present
    labels: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)  # DCGM field name -> value
    pods: set = field(default_factory=set)

    @property
    def name(self) -> str:
        mig = f"/mig{self.instance}" if self.instance else ""
        return f"{self.host}/gpu{self.gpu}{mig} ({self.labels.get('modelName', '?')})"

    def get(self, *names, default=None):
        """First present field among ``names`` (handles renamed fields)."""
        for n in names:
            if n in self.fields:
                return self.fields[n]
        return default


def snapshots(samples: list[Sample]) -> list[GpuSnapshot]:
    """Group DCGM_* samples by (host, gpu, MIG instance); pod labels are collected, not keyed on."""
    out: dict[tuple, GpuSnapshot] = {}
    for s in samples:
        if not s.name.startswith("DCGM_"):
            continue
        host = s.labels.get("Hostname", s.labels.get("instance", "?"))
        key = (host, s.labels.get("gpu", "?"), s.labels.get("GPU_I_ID"))
        snap = out.setdefault(key, GpuSnapshot(*key))
        snap.labels.update({k: v for k, v in s.labels.items() if k not in POD_LABELS})
        # the exporter's own text says pod/namespace; after a Prometheus scrape they are exported_*
        pod = s.labels.get("exported_pod") or s.labels.get("pod")
        if pod:
            ns = s.labels.get("exported_namespace", s.labels.get("namespace", ""))
            snap.pods.add(f"{ns}/{pod}")
        snap.fields[s.name] = s.value
    return list(out.values())


# --------------------------------------------------------------------------- derived signals
# NVML clock-event ("throttle") reason bits. (verify against nvml.h nvmlClocksEventReason*)
THROTTLE_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
                 0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
                 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}
THROTTLE_FIELDS = ("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", "DCGM_FI_DEV_CLOCK_THROTTLE_REASONS")  # new, old name


def decode_throttle(mask: int) -> list[str]:
    return [name for bit, name in THROTTLE_BITS.items() if int(mask) & bit]


@dataclass
class Signals:
    gpu_util: float | None  # 0-1, "a kernel was running"
    sm_active: float | None
    sm_occupancy: float | None
    tensor_active: float | None
    dram_active: float | None
    fb_used_frac: float | None
    temp_c: float | None
    power_w: float | None
    throttle: list[str]
    xid: int
    remapped_uncorrectable: float
    row_remap_failure: float
    pods: list[str]

    @property
    def util_gap(self) -> float | None:
        """GPU_UTIL minus SM_ACTIVE: how much the headline number overstates the busy fraction of SMs."""
        if self.gpu_util is None or self.sm_active is None:
            return None
        return self.gpu_util - self.sm_active


def derive(snap: GpuSnapshot) -> Signals:
    util = snap.get("DCGM_FI_DEV_GPU_UTIL")
    used, free = snap.get("DCGM_FI_DEV_FB_USED"), snap.get("DCGM_FI_DEV_FB_FREE")
    reserved = snap.get("DCGM_FI_DEV_FB_RESERVED", default=0.0)
    total = (used or 0) + (free or 0) + (reserved or 0)
    return Signals(
        gpu_util=None if util is None else util / 100.0,
        sm_active=snap.get("DCGM_FI_PROF_SM_ACTIVE"),
        sm_occupancy=snap.get("DCGM_FI_PROF_SM_OCCUPANCY"),
        tensor_active=snap.get("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"),
        dram_active=snap.get("DCGM_FI_PROF_DRAM_ACTIVE"),
        fb_used_frac=(used / total) if used is not None and total else None,
        temp_c=snap.get("DCGM_FI_DEV_GPU_TEMP"),
        power_w=snap.get("DCGM_FI_DEV_POWER_USAGE"),
        throttle=decode_throttle(snap.get(*THROTTLE_FIELDS, default=0)),
        xid=int(snap.get("DCGM_FI_DEV_XID_ERRORS", default=0)),
        remapped_uncorrectable=snap.get("DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", default=0.0),
        row_remap_failure=snap.get("DCGM_FI_DEV_ROW_REMAP_FAILURE", default=0.0),
        pods=sorted(snap.pods),
    )


THERMAL = {"sw_thermal_slowdown", "hw_thermal_slowdown"}
HW_SLOW = {"hw_slowdown", "hw_power_brake_slowdown"}


def bottleneck(sig: Signals) -> str:
    """A first-guess reading of the profiling fields. Thresholds are heuristics, not standards."""
    if sig.gpu_util is not None and sig.gpu_util < 0.05:
        return "idle: allocated but unused" if sig.pods else "idle"
    if set(sig.throttle) & (THERMAL | HW_SLOW):
        return "throttled: " + ", ".join(sorted(set(sig.throttle) & (THERMAL | HW_SLOW)))
    if sig.sm_active is None:
        return "unknown: profiling (DCP) fields not exported — only GPU_UTIL, which cannot tell"
    if (sig.tensor_active or 0) >= 0.3:
        return "tensor-core bound (GEMM-heavy: prefill, training)"
    if (sig.dram_active or 0) >= 0.5:
        return "memory-bandwidth bound (typical of LLM decode)"
    if (sig.gpu_util or 0) >= 0.8 and sig.sm_active < 0.3:
        return "busy but under-filled: small kernels/batches, launch overhead or a host-bound loop"
    return "moderate / mixed load"


# --------------------------------------------------------------------------- XID triage
# code: (meaning, owner, action). Owner = who acts: the application team, the node operator (drain /
# reset when convenient), or hardware (drain now; recurring -> replace). Condensed from NVIDIA's XID
# catalogue — verify before automating anything on it.
XIDS = {
    13: ("graphics engine exception", "application", "usually an out-of-bounds access in a kernel: run compute-sanitizer, fix, restart"),
    31: ("GPU memory page fault", "application", "illegal address from a kernel: fix the bug, restart the workload"),
    43: ("GPU stopped processing", "application", "a user process faulted; the GPU is usable once it exits"),
    45: ("preemptive cleanup", "application", "secondary: read the XID that preceded it"),
    61: ("internal micro-controller breakpoint", "node", "reset the GPU; update the driver if it recurs"),
    62: ("internal micro-controller halt", "node", "reset the GPU (reboot the node); update driver/firmware"),
    63: ("row remapping / page retirement recorded", "node", "the remap is pending until a GPU reset: drain and reset"),
    92: ("high single-bit ECC error rate", "node", "schedule diagnostics (dcgmi diag); replace if it persists"),
    94: ("contained ECC error", "node", "only the affected application died: restart it, reset the GPU when drained"),
    119: ("GSP RPC timeout", "node", "reset the GPU; update driver/firmware"),
    120: ("GSP error", "node", "reset the GPU; update driver/firmware"),
    48: ("double-bit ECC error", "hardware", "drain and reset now; recurring -> replace"),
    64: ("row remapper failure", "hardware", "drain; the GPU needs replacement"),
    74: ("NVLink error", "hardware", "drain, check `nvidia-smi nvlink -s`, reset; recurring -> hardware ticket"),
    79: ("GPU has fallen off the bus", "hardware", "drain and reboot; diagnose PCIe, power, thermals; recurring -> replace"),
    95: ("uncontained ECC error", "hardware", "every application on the GPU is affected: drain and reset now"),
}
SEVERITY = {"hardware": "critical", "node": "warning", "application": "info"}  # page / ticket / notify
UNKNOWN_XID = ("not in this table", "node", "look it up in NVIDIA's XID catalogue; collect nvidia-bug-report.sh")


def xids_owned_by(owner: str) -> list[int]:
    return sorted(c for c, (_, o, _) in XIDS.items() if o == owner)


def triage_xid(code: int) -> dict:
    """Owner, severity and action for an XID. Codes missing from the table go to the node operator at
    warning (as ``gpusim.health.triage_xid`` does): never silently to an application team."""
    meaning, owner, action = XIDS.get(int(code), UNKNOWN_XID)
    return {"xid": int(code), "meaning": meaning, "owner": owner, "severity": SEVERITY[owner],
            "action": action, "known": int(code) in XIDS}


# --------------------------------------------------------------------------- alert rules
_FIELD = re.compile(r"\bDCGM_[A-Z0-9_]+")


@dataclass(frozen=True)
class Rule:
    name: str
    expr: str  # PromQL; <pod> and <namespace> stand for the workload labels as your Prometheus stores them
    for_: str
    severity: str
    summary: str
    check: Callable[[Signals], bool]  # the same condition, evaluated on one snapshot

    @property
    def fields(self) -> frozenset:
        """The DCGM fields the rule reads: if the exporter does not export one, the rule can never fire."""
        return frozenset(_FIELD.findall(self.expr))

    def render(self, pod_label: str = "exported_pod", namespace_label: str = "exported_namespace") -> tuple[str, str]:
        sub = lambda t: t.replace("<pod>", pod_label).replace("<namespace>", namespace_label)  # noqa: E731
        return sub(self.expr), sub(self.summary)


def _bit(field_name: str, bit: int, width: int = 1) -> str:
    """PromQL has no bitwise AND: test bits with floor(x / 2^k) % 2^width > 0."""
    return f"floor({field_name} / {bit}) % {2 ** width} > 0"


XID_WINDOW = "15m"
# DCGM_FI_DEV_XID_ERRORS is a gauge holding the *last* XID seen; it does not go back to 0 when the GPU is
# fixed (verify for your DCGM version), so an alert on the value alone would never resolve. Each XID rule
# therefore also requires the value to have changed within the window: it fires when the XID appears and
# resolves XID_WINDOW later (the drain, not the alert, carries the node's state). A repeat of the *same*
# code does not change the gauge: where the exporter offers the counter DCGM_EXP_XID_ERRORS_TOTAL (opt-in
# in default-counters.csv), alert on increase() of it instead (verify its labels).
_XID_RECENT = f"changes(DCGM_FI_DEV_XID_ERRORS[{XID_WINDOW}]) > 0"


def _xid_in(codes: list[int]) -> str:
    return "(" + " or ".join(f"DCGM_FI_DEV_XID_ERRORS == {c}" for c in codes) + ")"


def _xid_rule(codes_expr: str) -> str:
    return f"{codes_expr} and {_XID_RECENT}"  # parenthesised: PromQL's `and` binds tighter than `or`


RULES = [
    Rule("GpuXidHardware", _xid_rule(_xid_in(xids_owned_by("hardware"))), "0m", "critical",
         "Hardware XID {{ $value }} on {{ $labels.Hostname }} gpu {{ $labels.gpu }}: drain now",
         lambda s: s.xid in xids_owned_by("hardware")),
    Rule("GpuXidNode", _xid_rule(_xid_in(xids_owned_by("node"))), "0m", "warning",
         "XID {{ $value }} on {{ $labels.Hostname }}: drain and reset the GPU when convenient",
         lambda s: s.xid in xids_owned_by("node")),
    Rule("GpuXidApplication", _xid_rule(_xid_in(xids_owned_by("application"))), "0m", "info",
         "XID {{ $value }} on {{ $labels.Hostname }}: an application fault; tell its owner",
         lambda s: s.xid in xids_owned_by("application")),
    Rule("GpuXidUnknown", _xid_rule(f"(DCGM_FI_DEV_XID_ERRORS > 0 unless {_xid_in(sorted(XIDS))})"), "0m", "warning",
         "XID {{ $value }} on {{ $labels.Hostname }} is not in the lab's table: look it up, collect nvidia-bug-report.sh",
         lambda s: s.xid > 0 and s.xid not in XIDS),
    Rule("GpuRowRemapFailure", "DCGM_FI_DEV_ROW_REMAP_FAILURE > 0", "0m", "critical",
         "Row remapping failed: the GPU cannot repair its memory, replace it",
         lambda s: s.row_remap_failure > 0),
    Rule("GpuUncorrectableRemappedRows", "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS > 0", "0m", "warning",
         "Rows remapped after uncorrectable errors: reset pending; watch the trend",
         lambda s: s.remapped_uncorrectable > 0),
    Rule("GpuThermalThrottling", _bit("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", 32, 2), "5m", "warning",
         "Thermal slowdown (SW or HW) on {{ $labels.Hostname }} gpu {{ $labels.gpu }}: facilities ticket",
         lambda s: bool(set(s.throttle) & THERMAL)),
    Rule("GpuHardwareSlowdown", f"{_bit('DCGM_FI_DEV_CLOCKS_EVENT_REASONS', 8)} or "
         f"{_bit('DCGM_FI_DEV_CLOCKS_EVENT_REASONS', 128)}", "5m", "warning",
         "HW slowdown / power brake: check power supply and cooling",
         lambda s: bool(set(s.throttle) & HW_SLOW)),
    Rule("GpuBusyButUnderfilled", "DCGM_FI_DEV_GPU_UTIL > 90 and on (Hostname, gpu) DCGM_FI_PROF_SM_ACTIVE < 0.3",
         "15m", "info", "GPU_UTIL high but few SMs busy: batch more, fuse kernels, or capture CUDA Graphs",
         lambda s: (s.gpu_util or 0) > 0.9 and s.sm_active is not None and s.sm_active < 0.3),
    Rule("GpuIdleWhileAllocated", 'DCGM_FI_DEV_GPU_UTIL{<pod>!=""} < 5', "30m", "info",
         "A pod holds a GPU it is not using: {{ $labels.<namespace> }}/{{ $labels.<pod> }}",
         lambda s: bool(s.pods) and s.gpu_util is not None and s.gpu_util < 0.05),
]

# Fields each exporter configuration serves by default (enabled lines of the upstream files, read
# 2026-09-26 — verify for your versions). "lab" is deploy/any-gpu/dcgm-counters.csv (a test keeps it in sync).
_STOCK = frozenset({
    "DCGM_FI_DEV_SM_CLOCK", "DCGM_FI_DEV_MEM_CLOCK", "DCGM_FI_DEV_MEMORY_TEMP", "DCGM_FI_DEV_GPU_TEMP",
    "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
    "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_DEV_ENC_UTIL", "DCGM_FI_DEV_DEC_UTIL",
    "DCGM_FI_DEV_XID_ERRORS", "DCGM_FI_DEV_FB_FREE", "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_FB_RESERVED",
    "DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL", "DCGM_FI_DEV_VGPU_LICENSE_STATUS",
    "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS",
    "DCGM_FI_DEV_ROW_REMAP_FAILURE", "DCGM_FI_DRIVER_VERSION", "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_DRAM_ACTIVE", "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_PCIE_RX_BYTES"})
EXPORTED_BY = {
    "stock dcgm-exporter (etc/default-counters.csv)": _STOCK,
    "GMP DCGM example (prometheus-engine examples/nvidia-dcgm)": frozenset({
        "DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_DEV_MEM_COPY_UTIL", "DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_MEMORY_TEMP",
        "DCGM_FI_DEV_POWER_USAGE", "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_SM_OCCUPANCY",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_PIPE_FP64_ACTIVE", "DCGM_FI_PROF_PIPE_FP32_ACTIVE",
        "DCGM_FI_PROF_PIPE_FP16_ACTIVE", "DCGM_FI_DEV_FB_FREE", "DCGM_FI_DEV_FB_USED", "DCGM_FI_DEV_FB_TOTAL",
        "DCGM_FI_PROF_PCIE_TX_BYTES", "DCGM_FI_PROF_PCIE_RX_BYTES", "DCGM_FI_PROF_NVLINK_TX_BYTES",
        "DCGM_FI_PROF_NVLINK_RX_BYTES"}),
    "lab (deploy/any-gpu/dcgm-counters.csv)": _STOCK | {
        "DCGM_FI_DEV_CLOCKS_EVENT_REASONS", "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_SM_OCCUPANCY"},
}
# What derive()/bottleneck() read, besides the rules' fields.
SIGNAL_FIELDS = frozenset({"DCGM_FI_DEV_GPU_UTIL", "DCGM_FI_PROF_SM_ACTIVE", "DCGM_FI_PROF_SM_OCCUPANCY",
                           "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "DCGM_FI_PROF_DRAM_ACTIVE", "DCGM_FI_DEV_FB_USED",
                           "DCGM_FI_DEV_FB_FREE", "DCGM_FI_DEV_GPU_TEMP", "DCGM_FI_DEV_POWER_USAGE",
                           "DCGM_FI_DEV_CLOCKS_EVENT_REASONS", "DCGM_FI_DEV_XID_ERRORS",
                           "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS", "DCGM_FI_DEV_ROW_REMAP_FAILURE"})


def rules_that_cannot_fire(exported, rules=RULES) -> list[str]:
    """Rules reading a field that ``exported`` (field names) lacks. The older name
    DCGM_FI_DEV_CLOCK_THROTTLE_REASONS counts as the new DCGM_FI_DEV_CLOCKS_EVENT_REASONS."""
    have = set(exported)
    if THROTTLE_FIELDS[1] in have:
        have.add(THROTTLE_FIELDS[0])
    return [r.name for r in rules if not r.fields <= have]


def evaluate(snaps: list[GpuSnapshot], rules=RULES) -> list[tuple[str, str, str]]:
    """(rule, severity, gpu) for every rule whose condition holds in this single snapshot. What one
    snapshot cannot show: a ``for`` duration (the alert would be *pending* until the condition has held
    that long) and the XID rules' "changed within the window" (a live XID is assumed to be recent)."""
    fired = []
    for snap in snaps:
        sig = derive(snap)
        fired += [(r.name, r.severity, snap.name) for r in rules if r.check(sig)]
    return fired


def rules_manifest(rules=RULES, name: str = "gpu-health", kind: str = "gmp", namespace: str = "monitoring",
                   pod_label: str = "exported_pod", namespace_label: str = "exported_namespace") -> str:
    """YAML for Google Managed Prometheus (``monitoring.googleapis.com/v1`` **ClusterRules**:
    cluster-scoped, because a namespaced ``Rules`` only sees its own namespace's metrics) or the
    Prometheus Operator (``monitoring.coreos.com/v1`` PrometheusRule, in ``namespace``).
    ``pod_label``/``namespace_label``: the workload labels as your Prometheus stores them (verify)."""
    import yaml

    api = {"gmp": ("monitoring.googleapis.com/v1", "ClusterRules"),
           "prometheus-operator": ("monitoring.coreos.com/v1", "PrometheusRule")}[kind]
    group_rules = []
    for r in rules:
        expr, summary = r.render(pod_label, namespace_label)
        entry = {"alert": r.name, "expr": expr}
        if r.for_ not in ("0m", "0s", ""):
            entry["for"] = r.for_
        entry["labels"] = {"severity": r.severity}
        entry["annotations"] = {"summary": summary}
        group_rules.append(entry)
    metadata = {"name": name} if kind == "gmp" else {"name": name, "namespace": namespace}
    doc = {"apiVersion": api[0], "kind": api[1], "metadata": metadata,
           "spec": {"groups": [{"name": name, "interval": "60s", "rules": group_rules}]}}
    header = (
        "# Generated by gpurt.dcgm.rules_manifest(): edit RULES in gpurt/dcgm.py, not this file.\n"
        + (f"# {api[1]} for Google Managed Prometheus: cluster-scoped, because a namespaced Rules object only\n"
           "# evaluates metrics from its own namespace and the DCGM exporter runs elsewhere (GKE system or gmp-public).\n"
           if kind == "gmp" else f"# {api[1]} for the Prometheus Operator.\n")
        + "# Firing alerts go to Alertmanager (GMP: the managed one, configured by a Secret in gmp-public;\n"
        "# see deploy/gke/README.md), not to Cloud Monitoring alerting policies.\n"
        "# Field prerequisites: stock dcgm-exporter lacks DCGM_FI_DEV_CLOCKS_EVENT_REASONS and DCGM_FI_PROF_SM_ACTIVE;\n"
        "# Google's GMP DCGM example list (which GKE's managed package resembles, verify) lacks the XID, remap and\n"
        "# clock fields. deploy/any-gpu/dcgm-counters.csv has them all (self-managed exporter);\n"
        "# gpurt.dcgm.rules_that_cannot_fire(fields) lists the rules a given exporter can never trigger.\n"
        f"# VERIFY: the apiVersion; the workload labels ({pod_label}, {namespace_label}: a scraper's own target labels\n"
        "# pod/namespace take precedence); the field names your exporter version exposes\n"
        "# (DCGM_FI_DEV_CLOCKS_EVENT_REASONS was DCGM_FI_DEV_CLOCK_THROTTLE_REASONS in older DCGM).\n")
    return header + yaml.safe_dump(doc, sort_keys=False, width=1000, allow_unicode=True)


def report(text: str) -> str:
    """One paragraph per GPU: signals, reading, XID triage, alerts."""
    samples, _ = parse_prometheus(text)
    lines = []
    exported = {smp.name for smp in samples}
    blind = rules_that_cannot_fire(exported)
    if blind:
        missing = sorted(set().union(*(r.fields for r in RULES if r.name in blind)) - exported)
        lines.append(f"not in this scrape: {', '.join(missing)} -> rules that cannot fire: {', '.join(blind)}")
    for snap in snapshots(samples):
        s = derive(snap)
        pct = lambda v: "n/a" if v is None else f"{v:.0%}"  # noqa: E731
        lines.append(f"{snap.name}  pods={s.pods or '-'}")
        lines.append(f"  GPU_UTIL {pct(s.gpu_util)} | SM_ACTIVE {pct(s.sm_active)} | occupancy {pct(s.sm_occupancy)} | "
                     f"tensor {pct(s.tensor_active)} | DRAM {pct(s.dram_active)} | FB used {pct(s.fb_used_frac)} | "
                     f"{s.temp_c} C | {s.power_w} W | clocks: {', '.join(s.throttle) or 'no events'}")
        lines.append(f"  reading: {bottleneck(s)}")
        if s.xid:
            t = triage_xid(s.xid)
            lines.append(f"  XID {t['xid']} ({t['meaning']}; owner: {t['owner']}): {t['action']}")
        for rule, sev, _ in evaluate([snap]):
            lines.append(f"  alert: {rule} [{sev}]")
    return "\n".join(lines)


if __name__ == "__main__":  # python -m gpurt.dcgm < metrics.txt
    import sys

    print(report(sys.stdin.read()))
