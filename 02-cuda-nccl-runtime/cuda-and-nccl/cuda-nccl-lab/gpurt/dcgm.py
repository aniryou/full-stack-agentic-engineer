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
PromQL you would deploy (Google Managed Prometheus ``Rules`` or a ``PrometheusRule``) and a Python
check that encodes the same condition, so the fixture exercises both.
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
POD_LABELS = ("pod", "namespace", "container")


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
        if s.labels.get("pod"):
            snap.pods.add(f"{s.labels.get('namespace', '')}/{s.labels['pod']}")
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
# code: (meaning, category, action) — condensed from NVIDIA's XID catalogue (verify before automating)
XIDS = {
    13: ("graphics engine exception", "application", "usually an application bug (e.g. out-of-bounds); run compute-sanitizer; many apps failing on one GPU -> suspect hardware"),
    31: ("GPU memory page fault", "application", "illegal address from a kernel; restart the workload, fix the bug"),
    43: ("GPU stopped processing", "application", "a user process faulted; the GPU is usable once it exits"),
    45: ("preemptive cleanup after earlier errors", "informational", "look at the XID that preceded it"),
    48: ("double-bit ECC error", "hardware", "uncorrectable memory error: drain the node, reset the GPU; recurring -> replace"),
    61: ("internal micro-controller breakpoint", "driver/firmware", "reset the GPU; update the driver if it recurs"),
    62: ("internal micro-controller halt", "driver/firmware", "reset the GPU (reboot the node); update driver/firmware"),
    63: ("ECC page retirement / row remap recorded", "hardware", "reset to apply the remap; watch remapped-row counters"),
    64: ("ECC page retirement / row remap failure", "hardware", "drain; the GPU likely needs replacement"),
    74: ("NVLink error", "hardware", "check `nvidia-smi nvlink -s`, reset; recurring -> hardware ticket"),
    79: ("GPU has fallen off the bus", "hardware", "PCIe/power/thermal event: drain the node and reboot; recurring -> hardware ticket"),
    92: ("high single-bit ECC error rate", "hardware", "monitor; schedule replacement if it persists"),
    94: ("contained ECC error", "hardware", "only the affected application stopped; restart it, the GPU continues"),
    95: ("uncontained ECC error", "hardware", "every application on the GPU is affected: drain and reset"),
    119: ("GSP RPC timeout", "driver/firmware", "reset the GPU; update driver/firmware"),
    120: ("GSP error", "driver/firmware", "reset the GPU; update driver/firmware"),
}
HARDWARE_XIDS = sorted(c for c, (_, cat, _) in XIDS.items() if cat == "hardware")


def triage_xid(code: int) -> dict:
    meaning, category, action = XIDS.get(int(code), ("unknown XID", "unknown", "look it up in NVIDIA's XID catalogue"))
    return {"xid": int(code), "meaning": meaning, "category": category, "action": action}


# --------------------------------------------------------------------------- alert rules
@dataclass(frozen=True)
class Rule:
    name: str
    expr: str  # PromQL, as deployed
    for_: str
    severity: str
    summary: str
    check: Callable[[Signals], bool]  # the same condition, evaluated on one snapshot


def _bit(field_name: str, bit: int, width: int = 1) -> str:
    """PromQL has no bitwise AND: test bits with floor(x / 2^k) % 2^width > 0."""
    return f"floor({field_name} / {bit}) % {2 ** width} > 0"


RULES = [
    Rule("GpuXidHardware", " or ".join(f"DCGM_FI_DEV_XID_ERRORS == {c}" for c in HARDWARE_XIDS), "0m", "critical",
         "Hardware-class XID on {{ $labels.Hostname }} gpu {{ $labels.gpu }}: drain and reset",
         lambda s: s.xid in HARDWARE_XIDS),
    Rule("GpuXidOther", "DCGM_FI_DEV_XID_ERRORS > 0 unless (" + " or ".join(
        f"DCGM_FI_DEV_XID_ERRORS == {c}" for c in HARDWARE_XIDS) + ")", "0m", "warning",
         "XID {{ $value }} on {{ $labels.Hostname }}: usually the application — check its logs",
         lambda s: s.xid > 0 and s.xid not in HARDWARE_XIDS),
    Rule("GpuRowRemapFailure", "DCGM_FI_DEV_ROW_REMAP_FAILURE > 0", "0m", "critical",
         "Row remapping failed: the GPU cannot repair its memory — replace it",
         lambda s: s.row_remap_failure > 0),
    Rule("GpuUncorrectableRemappedRows", "DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS > 0", "0m", "warning",
         "Rows remapped after uncorrectable errors: reset pending; watch the trend",
         lambda s: s.remapped_uncorrectable > 0),
    Rule("GpuThermalThrottling", _bit("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", 32, 2), "5m", "warning",
         "Thermal slowdown (SW or HW) on {{ $labels.Hostname }} gpu {{ $labels.gpu }}",
         lambda s: bool(set(s.throttle) & THERMAL)),
    Rule("GpuHardwareSlowdown", f"{_bit('DCGM_FI_DEV_CLOCKS_EVENT_REASONS', 8)} or "
         f"{_bit('DCGM_FI_DEV_CLOCKS_EVENT_REASONS', 128)}", "5m", "warning",
         "HW slowdown / power brake: check power supply and cooling",
         lambda s: bool(set(s.throttle) & HW_SLOW)),
    Rule("GpuBusyButUnderfilled", "DCGM_FI_DEV_GPU_UTIL > 90 and on (Hostname, gpu) DCGM_FI_PROF_SM_ACTIVE < 0.3",
         "15m", "info", "GPU_UTIL high but few SMs busy: batch more, fuse kernels, or capture CUDA Graphs",
         lambda s: (s.gpu_util or 0) > 0.9 and s.sm_active is not None and s.sm_active < 0.3),
    Rule("GpuIdleWhileAllocated", 'DCGM_FI_DEV_GPU_UTIL{pod!=""} < 5', "30m", "info",
         "A pod holds a GPU it is not using: {{ $labels.namespace }}/{{ $labels.pod }}",
         lambda s: bool(s.pods) and s.gpu_util is not None and s.gpu_util < 0.05),
]


def evaluate(snaps: list[GpuSnapshot], rules=RULES) -> list[tuple[str, str, str]]:
    """(rule, severity, gpu) for every rule whose condition holds in this single snapshot. Rules with
    a ``for`` duration would be *pending* until the condition has held that long."""
    fired = []
    for snap in snaps:
        sig = derive(snap)
        fired += [(r.name, r.severity, snap.name) for r in rules if r.check(sig)]
    return fired


def rules_manifest(rules=RULES, name: str = "gpu-health", namespace: str = "gpu-lab", kind: str = "gmp") -> str:
    """YAML for Google Managed Prometheus (``monitoring.googleapis.com/v1`` Rules) or the Prometheus
    Operator (``monitoring.coreos.com/v1`` PrometheusRule). apiVersions: (verify) for your cluster."""
    import yaml

    api = {"gmp": ("monitoring.googleapis.com/v1", "Rules"),
           "prometheus-operator": ("monitoring.coreos.com/v1", "PrometheusRule")}[kind]
    group_rules = []
    for r in rules:
        entry = {"alert": r.name, "expr": r.expr}
        if r.for_ not in ("0m", "0s", ""):
            entry["for"] = r.for_
        entry["labels"] = {"severity": r.severity}
        entry["annotations"] = {"summary": r.summary}
        group_rules.append(entry)
    doc = {"apiVersion": api[0], "kind": api[1], "metadata": {"name": name, "namespace": namespace},
           "spec": {"groups": [{"name": name, "interval": "60s", "rules": group_rules}]}}
    header = ("# Generated by gpurt.dcgm.rules_manifest() — edit RULES in gpurt/dcgm.py, not this file.\n"
              f"# {api[1]} for {'Google Managed Prometheus' if kind == 'gmp' else 'the Prometheus Operator'}. "
              "# VERIFY: apiVersion and the DCGM field names your exporter version exposes.\n")
    return header + yaml.safe_dump(doc, sort_keys=False, width=120)


def report(text: str) -> str:
    """One paragraph per GPU: signals, reading, XID triage, alerts."""
    samples, _ = parse_prometheus(text)
    lines = []
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
            lines.append(f"  XID {t['xid']} ({t['meaning']}, {t['category']}): {t['action']}")
        for rule, sev, _ in evaluate([snap]):
            lines.append(f"  alert: {rule} [{sev}]")
    return "\n".join(lines)


if __name__ == "__main__":  # python -m gpurt.dcgm < metrics.txt
    import sys

    print(report(sys.stdin.read()))
