"""GPU health signals. "GPU utilization" measures the *time* some kernel was running, not how much
of the GPU it used. Judge efficiency from SM active, occupancy, tensor active and DRAM active.
Triage errors (XIDs, ECC, throttling) by who has to act: the app owner, the node operator, or
the hardware vendor.

Field names are dcgm-exporter's (DCGM_FI_*). DCGM_FI_DEV_GPU_UTIL is a percentage (0-100) and
the DCGM_FI_PROF_* fields are ratios (0-1). The XID table condenses NVIDIA's XID catalogue for
the codes a serving fleet actually sees (verify against docs.nvidia.com/deploy/xid-errors).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Xid:
    code: int
    meaning: str
    owner: str      # "app" (fix the code), "node" (operator: drain/reset), "hardware" (RMA path)
    action: str


XIDS = {x.code: x for x in [
    Xid(13, "Graphics engine exception", "app",
        "usually an out-of-bounds access in a kernel: fix the app; if it repeats across apps, run DCGM diagnostics"),
    Xid(31, "GPU memory page fault", "app",
        "illegal address in a kernel (rarely a driver bug): reproduce under compute-sanitizer, restart the app"),
    Xid(43, "GPU stopped processing", "app", "a user app hit a software fault; the GPU is fine: restart the app"),
    Xid(45, "Preemptive cleanup, due to previous errors", "app",
        "secondary (a process was killed, or cleanup after an earlier error): read the XID logged before it"),
    Xid(48, "Double-bit ECC error", "hardware", "uncorrectable memory error: drain, reset the GPU; recurring means RMA"),
    Xid(63, "ECC page retirement or row-remapping recording event", "node",
        "a remap is pending until the GPU is reset: drain and reset at the next window"),
    Xid(64, "ECC page retirement or row-remapper recording failure", "hardware", "remapping failed: drain, RMA"),
    Xid(74, "NVLink error", "hardware", "drain; reset; check NVLink/NVSwitch and fabric manager; recurring means RMA"),
    Xid(79, "GPU has fallen off the bus", "hardware",
        "drain and reboot the node; check PCIe, power, thermals; recurring means RMA"),
    Xid(92, "High single-bit ECC error rate", "node", "correctable but trending: schedule diagnostics and a reset"),
    Xid(94, "Contained ECC error", "node",
        "only the app touching that memory was killed: restart it; reset the GPU once drained"),
    Xid(95, "Uncontained ECC error", "hardware", "every app on the GPU is affected: drain and reset now; recurring means RMA"),
    Xid(119, "GSP RPC timeout", "node", "GPU firmware stopped answering: drain, reset or reboot; update the driver if it repeats"),
]}


def triage_xid(code: int) -> Xid:
    return XIDS.get(code, Xid(code, "not in this table", "node",
                              "look it up in NVIDIA's XID catalogue; collect nvidia-bug-report.sh"))


# NVML clocks-event (formerly "throttle") reasons bitmask, as in DCGM_FI_DEV_CLOCKS_EVENT_REASONS.
THROTTLE = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap",
            0x8: "hw_slowdown", 0x10: "sync_boost", 0x20: "sw_thermal_slowdown",
            0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}


def decode_throttle(mask: int) -> list:
    return [name for bit, name in THROTTLE.items() if int(mask) & bit]


def util_counters(kernels, n_sms: int, window: float) -> dict:
    """kernels = [(start, end, sms_busy)] on one GPU over [0, window). Returns what two counters
    would report, as ratios:
    gpu_util   share of the window with at least one kernel running (nvidia-smi 'GPU-Util',
               DCGM_FI_DEV_GPU_UTIL)
    sm_active  share of SM-time with at least one warp resident (DCGM_FI_PROF_SM_ACTIVE)"""
    edges = sorted({0.0, float(window)} | {min(max(float(t), 0.0), window) for k in kernels for t in k[:2]})
    busy = sm = 0.0
    for a, b in zip(edges, edges[1:]):
        active = sum(s for (t0, t1, s) in kernels if t0 <= (a + b) / 2 < t1)
        if active:
            busy += b - a
            sm += (b - a) * min(active, n_sms) / n_sms
    return {"gpu_util": busy / window, "sm_active": sm / window}


def diagnose(sample: dict) -> list:
    """Derived signals from one DCGM sample: what the raw fields mean together."""
    g = sample.get
    util, sm = g("DCGM_FI_DEV_GPU_UTIL", 0) / 100, g("DCGM_FI_PROF_SM_ACTIVE")
    tensor, dram = g("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE"), g("DCGM_FI_PROF_DRAM_ACTIVE")
    out = []
    if util >= 0.9 and sm is not None and sm < 0.5:
        out.append(f"busy but mostly empty: GPU util {util:.0%} yet SM active {sm:.0%}: too few blocks "
                   "(small batch, launch-bound, serial kernels); GPU util overstates the load")
    if sm is not None and sm > 0.5 and tensor is not None and tensor < 0.05:
        out.append("SMs busy but tensor cores idle: FP32 math, or non-matmul kernels dominate")
    if dram is not None and dram > 0.5 and (tensor or 0) < 0.3:
        out.append(f"memory-bandwidth-bound (DRAM active {dram:.0%}): typical of decode; batch more, "
                   "quantize weights and KV")
    if tensor is not None and tensor > 0.5:
        out.append(f"tensor-core-bound (tensor active {tensor:.0%}): typical of prefill or large batches")
    used, free = g("DCGM_FI_DEV_FB_USED"), g("DCGM_FI_DEV_FB_FREE")
    if used is not None and free is not None and used / (used + free) > 0.95:
        out.append("framebuffer >95% used: expected for engines that pre-allocate the KV cache (vLLM); "
                   "watch the engine's KV-usage metric instead")
    reasons = decode_throttle(g("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", g("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", 0)))
    hw = [r for r in reasons if r.startswith("hw_")]
    if hw:
        out.append(f"hardware slowdown ({', '.join(hw)}): cooling or power delivery problem, not load")
    if "sw_thermal_slowdown" in reasons:
        out.append("thermal slowdown: the GPU is too hot; check airflow and inlet temperature")
    if "sw_power_cap" in reasons:
        out.append("at the power cap: normal under heavy load; clocks drop to stay within TDP")
    if g("DCGM_FI_DEV_XID_ERRORS", 0):
        x = triage_xid(int(g("DCGM_FI_DEV_XID_ERRORS")))
        out.append(f"XID {x.code} ({x.meaning}), owner {x.owner}: {x.action}")
    return out or ["no findings"]


def alerts(sample: dict) -> list:
    """(severity, message) pairs: page = hardware at risk, ticket = operator work, notify = app owner."""
    g, out = sample.get, []
    if g("DCGM_FI_DEV_XID_ERRORS", 0):
        x = triage_xid(int(g("DCGM_FI_DEV_XID_ERRORS")))
        sev = {"hardware": "page", "node": "ticket", "app": "notify"}[x.owner]
        out.append((sev, f"XID {x.code} {x.meaning}: {x.action}"))
    if g("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", 0) > 0:
        out.append(("page", "uncorrectable (double-bit) ECC errors since boot: drain and reset"))
    if g("DCGM_FI_DEV_ROW_REMAP_FAILURE", 0):
        out.append(("page", "row remapping failed: drain, RMA"))
    elif g("DCGM_FI_DEV_ROW_REMAP_PENDING", 0):
        out.append(("ticket", "row remap pending: reset the GPU when drained"))
    reasons = decode_throttle(g("DCGM_FI_DEV_CLOCKS_EVENT_REASONS", g("DCGM_FI_DEV_CLOCK_THROTTLE_REASONS", 0)))
    if any(r.startswith("hw_") or r == "sw_thermal_slowdown" for r in reasons):
        out.append(("ticket", f"clocks reduced by {', '.join(reasons)}: inspect cooling and power"))
    return out
