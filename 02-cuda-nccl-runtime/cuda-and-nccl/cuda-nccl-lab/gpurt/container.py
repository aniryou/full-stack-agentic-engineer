"""How this container sees its GPU — and whether the pieces are compatible.

The one idea: a containerised CUDA program is assembled from **two sources**.

* From the **host** (they must match the kernel module, so they cannot ship in the image):
  the character devices ``/dev/nvidia0..N``, ``/dev/nvidiactl``, ``/dev/nvidia-uvm`` and the user-space
  driver — ``libcuda.so`` (the driver API), ``libnvidia-ml.so`` (NVML), ``libnvidia-ptxjitcompiler.so``,
  ``nvidia-smi``. They are **injected when the container starts**: on Docker/containerd/CRI-O by the
  NVIDIA Container Toolkit (its OCI hook, or a CDI spec), each file bind-mounted read-only; on GKE by
  Google's device plugin, which mounts the host driver directory at ``/usr/local/nvidia``.
* From the **image**: the CUDA runtime (``libcudart``), cuBLAS/cuDNN/NCCL and the framework wheels.

Compatibility is a contract between them (primer §1, §6): the driver's CUDA version must be >= the
runtime's — or the same major, under minor-version compatibility, with caveats — and the binary must
carry SASS for the GPU's architecture or PTX the driver can JIT-compile, otherwise:
"no kernel image is available for execution on the device".

Everything below reads files and runs ``nvidia-smi``; the driver API probe (``cuInit`` via ctypes)
runs in a *subprocess* so this process never initialises CUDA.

    python -m gpurt.container          # explain this machine
    python -m gpurt.container --log probe.log   # explain a probe log captured elsewhere (GKE, Docker)
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata as md
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- device nodes
_NODE_KINDS = [
    (re.compile(r"^nvidia\d+$"), "gpu"),
    (re.compile(r"^nvidiactl$"), "control"),
    (re.compile(r"^nvidia-uvm$"), "unified memory"),
    (re.compile(r"^nvidia-uvm-tools$"), "unified memory tools"),
    (re.compile(r"^nvidia-modeset$"), "modeset"),
    (re.compile(r"^nvidia-cap\d+$"), "MIG capability"),
    (re.compile(r"^dxg$"), "WSL2 GPU (dxgkrnl)"),
]


@dataclass
class DeviceNode:
    path: str
    major: int | None
    minor: int | None
    kind: str


def _node_kind(name: str) -> str | None:
    for pattern, kind in _NODE_KINDS:
        if pattern.match(name):
            return kind
    return None


def device_nodes(dev_dir: str = "/dev") -> list[DeviceNode]:
    """NVIDIA character devices this process can see (plus /dev/dxg on WSL2)."""
    paths = glob.glob(os.path.join(dev_dir, "nvidia*")) + glob.glob(os.path.join(dev_dir, "nvidia-caps", "*"))
    paths += glob.glob(os.path.join(dev_dir, "dxg"))
    out = []
    for p in sorted(set(paths)):
        kind = _node_kind(os.path.basename(p))
        if kind is None:
            continue
        try:
            st = os.stat(p)
            major, minor = (os.major(st.st_rdev), os.minor(st.st_rdev)) if stat.S_ISCHR(st.st_mode) else (None, None)
        except OSError:
            major = minor = None
        out.append(DeviceNode(p, major, minor, kind))
    return out


_LS_LINE = re.compile(r"^c\S*\s+\d+\s+\S+\s+\S+\s+(\d+),\s*(\d+)\s+.*?(/dev/\S+)$")


def parse_ls_dev(text: str) -> list[DeviceNode]:
    """Parse ``ls -l /dev/nvidia*`` output (as captured by the probe script)."""
    out = []
    for line in text.splitlines():
        m = _LS_LINE.match(line.strip())
        if m and (kind := _node_kind(os.path.basename(m[3]))):
            out.append(DeviceNode(m[3], int(m[1]), int(m[2]), kind))
    return out


# --------------------------------------------------------------------------- mounts
@dataclass
class Mount:
    mount_id: int
    parent_id: int
    dev: str
    root: str
    mount_point: str
    options: str
    fstype: str
    source: str


def _unescape(s: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)


def parse_mountinfo(text: str) -> list[Mount]:
    """``/proc/self/mountinfo``: ``id parent maj:min root mountpoint opts [optional...] - fstype source superopts``."""
    mounts = []
    for line in text.splitlines():
        f = line.split()
        if not f or f[0].startswith("#") or "-" not in f:
            continue
        sep = f.index("-")
        if sep < 6 or len(f) < sep + 3:
            continue
        mounts.append(Mount(int(f[0]), int(f[1]), f[2], _unescape(f[3]), _unescape(f[4]), f[5],
                            f[sep + 1], _unescape(f[sep + 2])))
    return mounts


_INJECTED = [
    (r"/libcuda\.so[\d.]*$", "driver API (libcuda)"),
    (r"/libnvidia-ml\.so[\d.]*$", "NVML (libnvidia-ml)"),
    (r"/libnvidia-ptxjitcompiler\.so[\d.]*$", "PTX JIT compiler"),
    (r"/libnvidia-nvvm\.so[\d.]*$", "NVVM (JIT)"),
    (r"/libcudadebugger\.so[\d.]*$", "CUDA debugger"),
    (r"/libnvidia-[\w.-]+\.so[\d.]*$", "other driver library"),
    (r"/nvidia-smi$", "nvidia-smi"),
    (r"/nvidia-(debugdump|persistenced|cuda-mps-control|cuda-mps-server)$", "driver tool"),
    (r"^/proc/driver/nvidia", "driver procfs"),
    (r"^/dev/nvidia", "device node (bind-mounted)"),
    (r"/firmware/nvidia/", "GSP firmware"),
    (r"^/usr/local/nvidia$", "host driver directory (GKE device plugin)"),
    (r"nvidia-persistenced/socket$", "persistenced socket"),
]
_INJECTED = [(re.compile(p), k) for p, k in _INJECTED]
_VERSION_IN_PATH = re.compile(r"(?:\.so\.|/nvidia/)(\d{3}\.\d+(?:\.\d+)?)")


@dataclass
class InjectedFile:
    mount_point: str
    kind: str
    source_root: str


def injected_driver_files(mounts: list[Mount]) -> list[InjectedFile]:
    out = []
    for m in mounts:
        for pattern, kind in _INJECTED:
            if pattern.search(m.mount_point):
                out.append(InjectedFile(m.mount_point, kind, m.root))
                break
    return out


def injection_mechanism(files: list[InjectedFile]) -> str:
    kinds = {f.kind for f in files}
    if "host driver directory (GKE device plugin)" in kinds:
        return "GKE device plugin: host driver directory mounted at /usr/local/nvidia"
    if "driver API (libcuda)" in kinds:
        return "NVIDIA Container Toolkit (OCI hook or CDI): driver files bind-mounted one by one"
    return "none found in the mount table"


def driver_version_from_paths(paths) -> str | None:
    for p in paths:
        if m := _VERSION_IN_PATH.search(p):
            return m[1]
    return None


_PROC_VERSION = re.compile(r"Kernel Module(?: for \S+)?\s+(\d+\.\d+(?:\.\d+)?)")


def parse_proc_driver_version(text: str) -> str | None:
    """``/proc/driver/nvidia/version``: 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  550.54.15 ...'."""
    m = _PROC_VERSION.search(text)
    return m[1] if m else None


# --------------------------------------------------------------------------- versions and rules
# Driver branch -> newest CUDA version it supports (the "CUDA Version" in nvidia-smi's header). (verify)
DRIVER_BRANCH_CUDA = {450: "11.0", 455: "11.1", 460: "11.2", 465: "11.3", 470: "11.4", 495: "11.5",
                      510: "11.6", 515: "11.7", 520: "11.8", 525: "12.0", 530: "12.1", 535: "12.2",
                      545: "12.3", 550: "12.4", 555: "12.5", 560: "12.6", 565: "12.7", 570: "12.8",
                      575: "12.9", 580: "13.0"}
# Minimum Linux driver for minor-version compatibility within a CUDA major. (verify)
MIN_DRIVER_MINOR_COMPAT = {11: "450.80.02", 12: "525.60.13", 13: "580.65.06"}
# Compute capability -> architecture and example GPUs. (verify newer entries)
ARCH = {(6, 0): "Pascal (P100)", (7, 0): "Volta (V100)", (7, 5): "Turing (T4, RTX 20xx)",
        (8, 0): "Ampere (A100, A30)", (8, 6): "Ampere (A10, A40, RTX 30xx)", (8, 9): "Ada (L4, L40S, RTX 40xx)",
        (9, 0): "Hopper (H100, H200)", (10, 0): "Blackwell (B200, GB200)", (10, 3): "Blackwell Ultra (B300, GB300)",
        (12, 0): "Blackwell (RTX 50xx, RTX PRO 6000)"}


def vtuple(v: str | None) -> tuple[int, ...] | None:
    if not v:
        return None
    nums = re.findall(r"\d+", str(v))
    return tuple(int(x) for x in nums) if nums else None


def cuda_of_driver(driver_version: str | None) -> str | None:
    """Newest CUDA version a driver supports, from its branch (e.g. 550.54.15 -> 12.4)."""
    t = vtuple(driver_version)
    if not t:
        return None
    branches = [b for b in DRIVER_BRANCH_CUDA if b <= t[0]]
    return DRIVER_BRANCH_CUDA[max(branches)] if branches else None


_ARCH = re.compile(r"^(sm|compute)_(\d+)([af]?)$")


def parse_arch(s: str) -> tuple[str, int, int, str]:
    """'sm_89' -> ('sass', 8, 9, ''); 'compute_90a' -> ('ptx', 9, 0, 'a'); 'sm_120' -> ('sass', 12, 0, '')."""
    m = _ARCH.match(s.strip().lower())
    if not m:
        raise ValueError(f"not an arch string: {s!r}")
    digits = m[2]
    return ("sass" if m[1] == "sm" else "ptx", int(digits[:-1]), int(digits[-1]), m[3])


def sass_runs_on(arch: str, cc: tuple[int, int]) -> bool:
    """Binary compatibility: sm_XY SASS runs on cc X.Z with Z >= Y (same major); 'a' variants only on X.Y."""
    _, major, minor, suffix = parse_arch(arch)
    if suffix == "a":
        return (major, minor) == tuple(cc)
    return major == cc[0] and minor <= cc[1]


def ptx_jits_on(arch: str, cc: tuple[int, int]) -> bool:
    """PTX for compute_XY can be JIT-compiled for any cc >= X.Y (except arch-specific 'a' PTX)."""
    _, major, minor, suffix = parse_arch(arch)
    if suffix == "a":
        return (major, minor) == tuple(cc)
    if suffix == "f":
        return major == cc[0] and minor <= cc[1]
    return (major, minor) <= tuple(cc)


@dataclass
class Verdict:
    level: str  # ok | warn | fail
    topic: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper():4}] {self.topic}: {self.message}"


def compat_verdicts(driver_cuda: str | None, runtime_cuda: str | None, cc: tuple[int, int] | None = None,
                    archs: list[str] | tuple = (), driver_version: str | None = None,
                    forward_compat: bool = False) -> list[Verdict]:
    """The rules engine. ``driver_cuda``: newest CUDA the driver supports (nvidia-smi header);
    ``runtime_cuda``: the CUDA the image/framework was built with; ``archs``: what the binary contains
    (e.g. ``torch.cuda.get_arch_list()``: ['sm_75', 'sm_80', ..., 'compute_90'])."""
    out: list[Verdict] = []
    d, r = vtuple(driver_cuda), vtuple(runtime_cuda)
    if d is None:
        return [Verdict("fail", "driver", "no NVIDIA driver visible to this process — no GPU to run on")]
    newer_runtime = False
    if r is None:
        out.append(Verdict("warn", "runtime", "CUDA runtime version of the image unknown; cannot check the pair"))
    elif r[0] > d[0]:
        if forward_compat:
            out.append(Verdict("warn", "driver/runtime", f"runtime CUDA {runtime_cuda} > driver's {driver_cuda}: works only through "
                               "the CUDA forward-compatibility package (cuda-compat), on datacenter GPUs and supported driver branches"))
        else:
            out.append(Verdict("fail", "driver/runtime", f"runtime CUDA {runtime_cuda} needs a newer driver than one supporting CUDA "
                               f"{driver_cuda}: 'CUDA driver version is insufficient for CUDA runtime version' "
                               "(cudaErrorInsufficientDriver). Upgrade the driver, use an image built for an older CUDA, "
                               "or (datacenter GPUs) the forward-compatibility package"))
    elif r[:2] > d[:2]:
        newer_runtime = True
        msg = (f"runtime CUDA {runtime_cuda} > driver's {driver_cuda}, same major: minor-version compatibility — "
               "prebuilt SASS runs; PTX produced by the newer toolkit cannot be JIT-compiled by this driver "
               "(CUDA_ERROR_UNSUPPORTED_PTX_VERSION), and APIs newer than the driver return cudaErrorCallRequiresNewerDriver")
        floor = MIN_DRIVER_MINOR_COMPAT.get(r[0])
        dv = vtuple(driver_version)
        if floor and dv and dv < vtuple(floor):
            out.append(Verdict("fail", "driver/runtime", msg + f"; and driver {driver_version} is below the {floor} floor for CUDA {r[0]}.x"))
        else:
            out.append(Verdict("warn", "driver/runtime", msg))
    else:
        out.append(Verdict("ok", "driver/runtime", f"driver supports CUDA {driver_cuda} >= runtime {runtime_cuda} "
                           "(newer drivers run older runtimes)"))
    if cc is not None and archs:
        sass = [a for a in archs if a.lower().startswith("sm_")]
        ptx = [a for a in archs if a.lower().startswith("compute_")]
        name = ARCH.get(tuple(cc), "unknown architecture")
        if runner := next((a for a in sass if sass_runs_on(a, cc)), None):
            out.append(Verdict("ok", "architecture", f"{runner} SASS runs on compute capability {cc[0]}.{cc[1]} ({name})"))
        elif jit := next((a for a in ptx if ptx_jits_on(a, cc)), None):
            if newer_runtime:
                out.append(Verdict("fail", "architecture", f"no SASS for {cc[0]}.{cc[1]}; the only route is JIT of {jit} PTX, "
                                   "which this older driver cannot compile (PTX newer than the driver)"))
            else:
                out.append(Verdict("warn", "architecture", f"no SASS for {cc[0]}.{cc[1]} ({name}): the driver JIT-compiles {jit} "
                                   "PTX at first launch — slow start, cached in ~/.nv/ComputeCache"))
        else:
            out.append(Verdict("fail", "architecture", f"'no kernel image is available for execution on the device' "
                               f"(cudaErrorNoKernelImageForDevice): binary has {', '.join(archs)}, GPU is sm_{cc[0]}{cc[1]} ({name}). "
                               "Install a build that includes this architecture"))
    return out


# --------------------------------------------------------------------------- live probes
CU_ERRORS = {3: "CUDA_ERROR_NOT_INITIALIZED", 100: "CUDA_ERROR_NO_DEVICE", 101: "CUDA_ERROR_INVALID_DEVICE",
             304: "CUDA_ERROR_OPERATING_SYSTEM", 803: "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
             804: "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE", 999: "CUDA_ERROR_UNKNOWN"}


def _probe_driver_inprocess() -> dict:
    """Ask libcuda directly (the driver API): version, devices, compute capability. Initialises CUDA!"""
    import ctypes

    lib, err = None, ""
    for name in ("libcuda.so.1", "libcuda.so", "/usr/lib/wsl/lib/libcuda.so.1"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as e:
            err = err or str(e)  # the first (primary) name's error is the informative one
    if lib is None:
        return {"loaded": False, "error": f"libcuda not loadable ({err})", "devices": []}
    rc = lib.cuInit(0)
    if rc != 0:
        return {"loaded": True, "error": f"cuInit returned {rc} {CU_ERRORS.get(rc, '')}".strip(), "devices": []}
    v, n = ctypes.c_int(), ctypes.c_int()
    lib.cuDriverGetVersion(ctypes.byref(v))
    lib.cuDeviceGetCount(ctypes.byref(n))
    devices = []
    for i in range(n.value):
        dev, major, minor, sms = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        lib.cuDeviceGet(ctypes.byref(dev), i)
        name = ctypes.create_string_buffer(256)
        lib.cuDeviceGetName(name, 256, dev)
        lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev)  # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev)  # ..._MINOR
        lib.cuDeviceGetAttribute(ctypes.byref(sms), 16, dev)  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
        mem = ctypes.c_size_t()
        lib.cuDeviceTotalMem_v2(ctypes.byref(mem), dev)
        devices.append({"index": i, "name": name.value.decode(errors="replace"), "cc": [major.value, minor.value],
                        "sms": sms.value, "memory_gib": round(mem.value / 2 ** 30, 1)})
    return {"loaded": True, "error": None, "driver_cuda": f"{v.value // 1000}.{(v.value % 1000) // 10}",
            "devices": devices}


def probe_driver(timeout: float = 60.0) -> dict:
    """Run the libcuda probe in a child process (so *this* process never calls cuInit)."""
    root = str(Path(__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    try:
        out = subprocess.run([sys.executable, "-m", "gpurt.container", "--driver-json"], capture_output=True,
                             text=True, timeout=timeout, env=env)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        return {"loaded": False, "error": f"probe failed: {e}", "devices": []}


def nvidia_smi_header_cuda(text: str) -> tuple[str | None, str | None]:
    """(driver version, CUDA version) from nvidia-smi's banner: 'Driver Version: 550.54.15  CUDA Version: 12.4'."""
    dm = re.search(r"Driver Version:\s*([\d.]+)", text)
    cm = re.search(r"CUDA Version:\s*([\d.]+)", text)
    return (dm[1] if dm else None, cm[1] if cm else None)


def _cuda_from_wheel_tag(version: str) -> str | None:
    m = re.search(r"\+cu(\d+)(\d)$", version)  # '2.8.0+cu128' -> 12.8
    return f"{m[1]}.{m[2]}" if m else None


def image_cuda() -> dict:
    """The image side: CUDA userland versions, found without importing torch."""
    info = {k: os.environ.get(k) for k in ("CUDA_VERSION", "NVIDIA_REQUIRE_CUDA")}
    nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
    if nvcc:
        try:
            out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=20).stdout
            m = re.search(r"release (\d+\.\d+)", out)
            info["nvcc"] = m[1] if m else None
        except (OSError, subprocess.SubprocessError):
            info["nvcc"] = None
    info["cudart_libs"] = sorted({os.path.basename(p) for p in glob.glob("/usr/local/cuda*/lib64/libcudart.so.*")})
    for dist in ("torch", "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime", "numba-cuda", "cupy-cuda12x"):
        try:
            info[dist] = md.version(dist)
        except md.PackageNotFoundError:
            pass
    info["runtime_cuda"], info["runtime_source"] = _runtime_cuda(info)
    return info


def _major_minor(v: str | None) -> str | None:
    return ".".join(v.split(".")[:2]) if v else None


def _runtime_cuda(info: dict) -> tuple[str | None, str | None]:
    """Best guess of the CUDA runtime the workload uses, most specific source first."""
    if tag := _cuda_from_wheel_tag(info.get("torch", "")):
        return tag, "torch wheel tag"
    for dist in ("nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime"):
        if info.get(dist):
            return _major_minor(info[dist]), f"pip {dist}"
    if info.get("nvcc"):
        return info["nvcc"], "nvcc"
    if info.get("CUDA_VERSION"):
        return _major_minor(info["CUDA_VERSION"]), "CUDA_VERSION (image env)"
    return None, None


VISIBLE_ENV = ("NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "CUDA_VISIBLE_DEVICES", "NVIDIA_REQUIRE_CUDA",
               "NVIDIA_DISABLE_REQUIRE", "CUDA_VERSION", "LD_LIBRARY_PATH")


# --------------------------------------------------------------------------- the report
@dataclass
class ContainerReport:
    source: str  # "live" or "log"
    in_container: bool | None = None
    env: dict = field(default_factory=dict)
    nodes: list[DeviceNode] = field(default_factory=list)
    injected: list[InjectedFile] = field(default_factory=list)
    mechanism: str = ""
    driver_version: str | None = None
    driver_cuda: str | None = None
    devices: list[dict] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    verdicts: list[Verdict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _finish(rep: ContainerReport, archs=()) -> ContainerReport:
    rep.mechanism = injection_mechanism(rep.injected)
    rep.driver_cuda = rep.driver_cuda or cuda_of_driver(rep.driver_version)
    cc = tuple(rep.devices[0]["cc"]) if rep.devices and rep.devices[0].get("cc") else None
    rep.verdicts = compat_verdicts(rep.driver_cuda, rep.image.get("runtime_cuda"), cc, archs, rep.driver_version)
    return rep


def probe(dev_dir: str = "/dev", mountinfo: str = "/proc/self/mountinfo", driver: bool = True, archs=()) -> ContainerReport:
    """Inspect *this* process's view: nodes, mounts, env, driver (via a child process), image."""
    from .env import in_container

    rep = ContainerReport(source="live", in_container=in_container())
    rep.env = {k: os.environ[k] for k in VISIBLE_ENV if k in os.environ}
    rep.nodes = device_nodes(dev_dir)
    try:
        rep.injected = injected_driver_files(parse_mountinfo(Path(mountinfo).read_text()))
    except OSError:
        rep.notes.append(f"could not read {mountinfo}")
    try:
        rep.driver_version = parse_proc_driver_version(Path("/proc/driver/nvidia/version").read_text())
    except OSError:
        pass
    rep.driver_version = rep.driver_version or driver_version_from_paths([f.mount_point for f in rep.injected])
    if driver:
        d = probe_driver()
        rep.devices, rep.driver_cuda = d.get("devices", []), d.get("driver_cuda")
        if d.get("error"):
            rep.notes.append(f"driver API: {d['error']}")
    rep.image = image_cuda()
    return _finish(rep, archs)


def parse_probe_log(text: str) -> dict[str, str]:
    """Split the output of ``deploy/any-gpu/probe.sh`` into its ``=== name ===`` sections."""
    sections, name, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^=== (.+?) ===$", line.strip())
        if m:
            if name:
                sections[name] = "\n".join(buf).strip("\n")
            name, buf = m[1], []
        elif name:
            buf.append(line)
    if name:
        sections[name] = "\n".join(buf).strip("\n")
    return sections


def from_probe_log(text: str, archs=()) -> ContainerReport:
    """Build the same report from a probe log captured in another container (GKE pod, Docker, RunPod)."""
    s = parse_probe_log(text)
    rep = ContainerReport(source="log", in_container=True)
    for line in s.get("env", "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k in VISIBLE_ENV:
                rep.env[k] = v
    rep.nodes = parse_ls_dev(s.get("dev", ""))
    rep.injected = injected_driver_files(parse_mountinfo(s.get("mountinfo", "")))
    rep.driver_version = (parse_proc_driver_version(s.get("proc-driver-version", "")) or
                          driver_version_from_paths([f.mount_point for f in rep.injected]))
    smi_driver, smi_cuda = nvidia_smi_header_cuda(s.get("nvidia-smi", ""))
    rep.driver_version = rep.driver_version or smi_driver
    rep.driver_cuda = smi_cuda
    for line in s.get("gpus", "").splitlines():  # index, name, compute_cap, memory.total
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            cc = vtuple(parts[2])
            rep.devices.append({"index": int(parts[0]), "name": parts[1], "cc": list(cc[:2]) if cc else None,
                                "memory": parts[3] if len(parts) > 3 else None})
    rep.image = {"CUDA_VERSION": rep.env.get("CUDA_VERSION"), "runtime_cuda": _major_minor(rep.env.get("CUDA_VERSION")),
                 "runtime_source": "CUDA_VERSION (image env)" if rep.env.get("CUDA_VERSION") else None}
    return _finish(rep, archs)


def explain(rep: ContainerReport) -> str:
    L = [f"How {'this process' if rep.source == 'live' else 'the probed container'} sees NVIDIA GPUs"]
    gpus = [n for n in rep.nodes if n.kind == "gpu"]
    L.append(f"1. Device nodes: {len(gpus)} GPU node(s)" + (": " + ", ".join(
        f"{n.path} ({n.major}:{n.minor})" if n.major is not None else n.path for n in rep.nodes) if rep.nodes else ""))
    if not rep.nodes:
        L.append("   none -> the kernel driver is unreachable from here. Docker needs --gpus all (or CDI "
                 "--device nvidia.com/gpu=all); Kubernetes needs resources.limits nvidia.com/gpu; a VM needs a GPU attached.")
    L.append(f"2. Driver user space: {rep.mechanism}")
    for f in rep.injected[:8]:
        L.append(f"   {f.mount_point}  [{f.kind}]")
    if len(rep.injected) > 8:
        L.append(f"   ... {len(rep.injected) - 8} more")
    L.append(f"3. Driver: {rep.driver_version or 'unknown'}; supports CUDA up to {rep.driver_cuda or 'unknown'}")
    for d in rep.devices:
        line = f"   GPU {d.get('index')}: {d.get('name')}"
        if cc := d.get("cc"):
            line += f" — compute capability {cc[0]}.{cc[1]} ({ARCH.get(tuple(cc), 'unknown')})"
        L.append(line)
    img = rep.image
    shown = {k: img[k] for k in ("CUDA_VERSION", "nvcc", "torch", "numba-cuda", "runtime_cuda", "runtime_source")
             if img.get(k)}
    L.append(f"4. Image userland: {shown if shown else 'no CUDA userland detected'}")
    if rep.env:
        L.append("   env: " + ", ".join(f"{k}={v}" for k, v in rep.env.items() if k != "LD_LIBRARY_PATH"))
    L.append("5. Verdicts:")
    L += [f"   {v}" for v in rep.verdicts]
    L += [f"   note: {n}" for n in rep.notes]
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Explain how this container sees its GPU.")
    p.add_argument("--log", help="explain a probe log (deploy/any-gpu/probe.sh output) instead of this machine")
    p.add_argument("--arch", action="append", default=[], help="an arch the binary contains, e.g. sm_80 (repeat)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--driver-json", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.driver_json:  # child-process mode used by probe_driver()
        print(json.dumps(_probe_driver_inprocess()))
        return 0
    rep = from_probe_log(Path(a.log).read_text(), a.arch) if a.log else probe(archs=a.arch)
    print(json.dumps(rep.as_dict(), indent=2, default=str) if a.json else explain(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
