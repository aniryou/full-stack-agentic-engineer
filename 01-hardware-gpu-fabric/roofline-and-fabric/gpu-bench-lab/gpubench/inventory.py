"""What is in the box: the local CPU, the GPUs ``nvidia-smi`` reports, and which of them look sick.

Before any benchmark, write down the machine: a number without its hardware is useless, and
a surprising number is often explained by the inventory alone — a GPU whose PCIe link trained
at x8 instead of x16 (half the host↔device bandwidth), a power cap below the default (lower
sustained clocks), memory already held by another process, persistence mode off (the first
CUDA call pays driver initialisation).

``parse_csv`` reads ``nvidia-smi --query-gpu=... --format=csv`` (units and ``[N/A]`` included);
``health`` turns rows into findings; ``cpu_info`` describes the CPU from /proc and /sys (Linux)
or sysctl (macOS) so the T0 path has an inventory too.
"""
from __future__ import annotations

import csv
import io
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass

QUERY_FIELDS = ["index", "name", "pci.bus_id", "driver_version", "compute_cap", "memory.total", "memory.used",
                "power.limit", "power.default_limit", "clocks.max.sm", "clocks.max.memory",
                "pcie.link.gen.current", "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
                "temperature.gpu", "ecc.mode.current", "persistence_mode"]
QUERY = ["nvidia-smi", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv"]
_STRING_FIELDS = {"name", "uuid", "pci.bus_id", "driver_version", "compute_cap", "vbios_version",
                  "ecc.mode.current", "persistence_mode", "serial"}
_MISSING = {"[N/A]", "N/A", "[Not Supported]", "[Unknown Error]", "[Insufficient Permissions]", ""}


def _value(field: str, unit: str | None, raw: str):
    raw = raw.strip()
    if raw in _MISSING:
        return None
    if field in _STRING_FIELDS:
        return raw
    if unit and raw.endswith(unit):
        raw = raw[: -len(unit)].strip()
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_csv(text: str) -> list:
    """Rows of ``nvidia-smi --query-gpu=... --format=csv`` as dicts with typed values.

    The header carries units (``memory.total [MiB]``); values repeat them (``81559 MiB``).
    Both are stripped; the unit is kept in ``row['_units']``. ``[N/A]`` becomes None.
    """
    reader = csv.reader(io.StringIO(text.strip()), skipinitialspace=True)
    header = next(reader)
    fields, units = [], {}
    for h in header:
        m = re.match(r"^(.*?)\s*\[(.+)\]$", h.strip())
        name = m.group(1) if m else h.strip()
        fields.append(name)
        if m:
            units[name] = m.group(2)
    rows = []
    for cells in reader:
        if not cells or all(not c.strip() for c in cells):
            continue
        row = {f: _value(f, units.get(f), c) for f, c in zip(fields, cells)}
        row["_units"] = units
        rows.append(row)
    return rows


@dataclass(frozen=True)
class Finding:
    level: str            # "WARN" (costs you performance now) | "INFO" (check it)
    gpu: object
    message: str

    def __str__(self) -> str:
        return f"{self.level:<4} GPU {self.gpu}: {self.message}"


def health(rows) -> list:
    """Rules of thumb that explain surprising benchmark numbers."""
    out = []
    for r in rows:
        g = r.get("index")
        wc, wm = r.get("pcie.link.width.current"), r.get("pcie.link.width.max")
        if isinstance(wc, int) and isinstance(wm, int) and wc < wm:
            out.append(Finding("WARN", g, f"PCIe link trained at x{wc} of x{wm}: host↔device bandwidth is "
                                          f"{wc / wm:.0%} of what it should be (riser, slot, BIOS?)"))
        gc, gm = r.get("pcie.link.gen.current"), r.get("pcie.link.gen.max")
        if isinstance(gc, int) and isinstance(gm, int) and gc < gm:
            out.append(Finding("INFO", g, f"PCIe Gen{gc} now, Gen{gm} max: normal when idle (link power "
                                          f"saving); re-query while a transfer runs"))
        pl, pd = r.get("power.limit"), r.get("power.default_limit")
        if isinstance(pl, (int, float)) and isinstance(pd, (int, float)) and pl < 0.95 * pd:
            out.append(Finding("WARN", g, f"power limit {pl:g} W below the default {pd:g} W: expect lower "
                                          f"sustained clocks and GEMM throughput"))
        mu, mt = r.get("memory.used"), r.get("memory.total")
        if isinstance(mu, (int, float)) and isinstance(mt, (int, float)) and mt and mu > 0.02 * mt:
            out.append(Finding("WARN", g, f"{mu:g} of {mt:g} MiB already in use: another process holds this GPU"))
        temp = r.get("temperature.gpu")
        if isinstance(temp, (int, float)) and temp >= 85:
            out.append(Finding("WARN", g, f"{temp:g} C: at or near thermal throttling"))
        if r.get("persistence_mode") == "Disabled":
            out.append(Finding("INFO", g, "persistence mode off: the first CUDA call pays driver start-up "
                                          "(nvidia-persistenced or nvidia-smi -pm 1)"))
        if r.get("ecc.mode.current") == "Disabled":
            out.append(Finding("INFO", g, "ECC disabled: more usable memory, no protection against bit flips"))
    return out


def query_gpus() -> tuple:
    """Run ``nvidia-smi --query-gpu`` on this machine: ``(rows, raw_text)``, or ``(None, reason)``."""
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found (no NVIDIA driver on this machine)"
    for query in (QUERY, ["nvidia-smi", "--query-gpu=index,name,memory.total,pcie.link.gen.max,"
                                        "pcie.link.width.max", "--format=csv"]):
        try:
            out = subprocess.run(query, capture_output=True, text=True, timeout=30, check=True).stdout
            return parse_csv(out), out
        except (subprocess.SubprocessError, OSError, StopIteration):
            continue          # an older driver may reject a field (e.g. compute_cap): retry smaller
    return None, "nvidia-smi --query-gpu failed"


# -- the CPU ------------------------------------------------------------------------------------------
def _size(s: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([KMG]?)B?\s*", s, flags=re.I)
    if not m:
        return 0
    return int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2).upper()]


def cache_sizes() -> dict:
    """``{"L1d": bytes, "L2": bytes, "L3": bytes}`` for the first CPU (L2 is usually per core,
    L3 shared by the socket). Missing levels are absent."""
    out = {}
    base = "/sys/devices/system/cpu/cpu0/cache"
    if os.path.isdir(base):
        for idx in sorted(os.listdir(base)):
            p = os.path.join(base, idx)
            try:
                level = open(os.path.join(p, "level")).read().strip()
                kind = open(os.path.join(p, "type")).read().strip()
                size = _size(open(os.path.join(p, "size")).read())
            except OSError:
                continue
            if kind == "Instruction":
                continue
            out["L1d" if level == "1" else f"L{level}"] = size
    elif platform.system() == "Darwin":
        for key, name in (("hw.l1dcachesize", "L1d"), ("hw.l2cachesize", "L2"), ("hw.l3cachesize", "L3"),
                          ("hw.perflevel0.l2cachesize", "L2")):
            try:
                v = int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout)
                if v:
                    out[name] = v
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
    return out


def llc_total_bytes() -> int | None:
    """Every last-level cache on the machine added up (Linux).

    ``cache_sizes`` reports cpu0's caches only, but a two-socket server has one L3 per socket and
    an AMD EPYC one per CCD (8 × 32 MB is 256 MB of L3). STREAM's rule — each array at least 4× the
    last-level cache — means 4× the *sum* of every LLC the run's threads can use, so this counts
    each distinct instance once (instances are told apart by ``shared_cpu_list``). None where
    /sys is not available; the caller falls back to ``cache_sizes``.
    """
    root = "/sys/devices/system/cpu"
    if not os.path.isdir(root):
        return None
    instances: dict = {}
    top = 0
    for cpu in os.listdir(root):
        base = os.path.join(root, cpu, "cache")
        if not re.fullmatch(r"cpu\d+", cpu) or not os.path.isdir(base):
            continue
        for idx in os.listdir(base):
            p = os.path.join(base, idx)
            try:
                level = int(open(os.path.join(p, "level")).read())
                if open(os.path.join(p, "type")).read().strip() == "Instruction":
                    continue
                size = _size(open(os.path.join(p, "size")).read())
                shared = open(os.path.join(p, "shared_cpu_list")).read().strip()
            except (OSError, ValueError):
                continue
            instances[(level, shared)] = size
            top = max(top, level)
    total = sum(size for (level, _), size in instances.items() if level == top)
    return total or None


def usable_cpus() -> int:
    """CPUs this process may run on: the affinity mask (a container or ``taskset`` can restrict it),
    not ``os.cpu_count()``, which reports every CPU of the host."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0)) or 1
        except OSError:
            pass
    return os.cpu_count() or 1


def cpu_info() -> dict:
    """Model, usable CPUs, vector ISA, nominal clock, caches and RAM of this machine."""
    from .specs import simd_bits_from_flags

    info = {"system": platform.system(), "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    if hasattr(os, "sched_getaffinity"):
        info["usable_cpus"] = usable_cpus()    # containers may restrict this
    model, flags, mhz = None, set(), None
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", errors="replace"):
            key, _, val = line.partition(":")
            key = key.strip()
            if key in ("model name", "Model", "Processor") and not model:
                model = val.strip()
            elif key in ("flags", "Features") and not flags:
                flags = set(val.split())
            elif key == "cpu MHz" and mhz is None:
                try:
                    mhz = float(val)
                except ValueError:
                    pass
    elif platform.system() == "Darwin":
        try:
            model = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
        if platform.machine() == "arm64":
            flags = {"asimd"}
    info["model"] = model or platform.processor() or "unknown CPU"
    info["simd_bits"] = simd_bits_from_flags(flags)
    info["isa"] = sorted(flags & {"sse4_2", "avx", "avx2", "fma", "avx512f", "avx512_bf16", "avx512_fp16",
                                  "avx512_vnni", "amx_tile", "amx_bf16", "asimd", "sve"})
    m = re.search(r"@\s*([\d.]+)\s*GHz", info["model"])
    info["ghz_nominal"] = float(m.group(1)) if m else (round(mhz / 1000, 2) if mhz else None)
    info["caches"] = cache_sizes()
    info["llc_total_bytes"] = llc_total_bytes()
    try:
        info["memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        info["memory_bytes"] = None
    return info


def load_fixture(name: str) -> str:
    """Sample ``--query-gpu`` output in the documented format (illustrative), bundled with the lab."""
    from importlib import resources
    files = {"hgx-h100-8gpu": "inventory_hgx_h100_8gpu.csv", "colab-t4": "inventory_colab_t4.csv"}
    return resources.files("gpubench").joinpath("fixtures").joinpath(files.get(name, name)).read_text()
