"""Will this CUDA program run here? There are two independent gates.

1. The host's *driver* must speak the CUDA version your app's *runtime* was built with. Newer
   drivers run older runtimes. A newer runtime runs only by minor-version compatibility (same
   major, driver above that major's floor) or with the forward-compatibility package
   (data-center GPUs only).
2. The binary must carry a *kernel image* this GPU can run: SASS for its compute capability
   (same major, equal or lower minor), or PTX that the driver can JIT-compile.

`check()` walks both gates and names the exact error you would see, or says it cannot judge
when the driver is older than its table. `explain_container()` adds the container failure modes.
The tables are dated Sep 2026 and marked (verify): minimum Linux x86_64 drivers from the CUDA
Toolkit release notes, and the kernel-driver branches each cuda-compat package accepts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# CUDA toolkit (GA) -> minimum Linux x86_64 driver: the release notes' "Toolkit Driver Version"
# table (verify; checked 2026-09-26 against the copy in meson's cuda module, which has 13.2/13.3).
CUDA_MIN_DRIVER = {
    "9.0": "384.81", "9.1": "390.46", "9.2": "396.26",
    "10.0": "410.48", "10.1": "418.39", "10.2": "440.33",
    "11.0": "450.51.05", "11.1": "455.23", "11.2": "460.27.03", "11.3": "465.19.01",
    "11.4": "470.42.01", "11.5": "495.29.05", "11.6": "510.39.01", "11.7": "515.43.04",
    "11.8": "520.61.05",
    "12.0": "525.60.13", "12.1": "530.30.02", "12.2": "535.54.03", "12.3": "545.23.06",
    "12.4": "550.54.14", "12.5": "555.42.02", "12.6": "560.28.03", "12.8": "570.26",
    "12.9": "575.51.03", "13.0": "580.65.06", "13.1": "590.44.01", "13.2": "595.45.04",
    "13.3": "610.43.02", "13.4": "615",
}
# Rows that are a driver *branch* only, inferred rather than read from the release notes: 13.4 from
# NVIDIA's R615 NVML binding (nvidia-ml-py 13.615.71). The exact minimum is some 615.x.
INFERRED = {"13.4"}
# Minor-version compatibility: any runtime of major M runs on a driver >= this floor (release
# notes: 11.x >= 450.80.02, 12.x >= 525, 13.x >= 580; verify).
MINOR_COMPAT_FLOOR = {11: "450.80.02", 12: "525.60.13", 13: "580.65.06"}
# Forward compatibility: the kernel-driver branches a cuda-compat package runs over. Read from the
# NVIDIA_REQUIRE_CUDA constraints of NVIDIA's nvidia/cuda images (brand=tesla,driver>=B,driver<B+1);
# the authoritative list is the CUDA Compatibility guide (verify). Other versions: not tabulated.
COMPAT_BRANCHES = {
    "12.4": (470, 525, 535), "12.8": (470, 535, 550, 560, 565),
    "13.0": (535, 550, 565, 570, 575), "13.1": (535, 550, 570, 575, 580),
}

# GPU -> (compute capability, architecture, data-center class). Verify the Blackwell rows.
GPUS = {
    "V100": ("7.0", "Volta", True), "T4": ("7.5", "Turing", True),
    "A100": ("8.0", "Ampere", True), "A10G": ("8.6", "Ampere", True),
    "RTX 3090": ("8.6", "Ampere", False), "L4": ("8.9", "Ada Lovelace", True),
    "L40S": ("8.9", "Ada Lovelace", True), "RTX 4090": ("8.9", "Ada Lovelace", False),
    "H100": ("9.0", "Hopper", True), "H200": ("9.0", "Hopper", True),
    "B200": ("10.0", "Blackwell", True), "GB200": ("10.0", "Blackwell", True),
    "B300": ("10.3", "Blackwell Ultra", True), "RTX PRO 6000": ("12.0", "Blackwell", True),
    "RTX 5090": ("12.0", "Blackwell", False),
}
# First toolkit that can target each compute capability (verify). CUDA 13 dropped offline
# compilation for Maxwell, Pascal and Volta (sm_50 to sm_72); Turing (7.5) is still supported.
FIRST_CUDA = {"7.0": "9.0", "7.5": "10.0", "8.0": "11.0", "8.6": "11.1", "8.9": "11.8",
              "9.0": "11.8", "10.0": "12.8", "10.3": "12.9", "12.0": "12.8"}

# cudaError_t code -> (name, message you see, what to do)
ERRORS = {
    35: ("cudaErrorInsufficientDriver", "CUDA driver version is insufficient for CUDA runtime version",
         "upgrade the host driver, build against an older CUDA, or (data-center GPU) add cuda-compat"),
    36: ("cudaErrorCallRequiresNewerDriver", "API call is not supported in the installed CUDA driver",
         "a minor-version-compat app called an API newer than the driver: upgrade the driver"),
    100: ("cudaErrorNoDevice", "no CUDA-capable device is detected",
          "inject the GPU (--gpus / CDI device / nvidia.com/gpu request) or install a driver that knows it"),
    209: ("cudaErrorNoKernelImageForDevice", "no kernel image is available for execution on the device",
          "rebuild with SASS for this sm (or a +PTX target), or use a wheel/image built for it"),
    222: ("cudaErrorUnsupportedPtxVersion", "the provided PTX was compiled with an unsupported toolchain.",
           "ship SASS for this GPU, or upgrade the driver to the toolkit's version"),
    802: ("cudaErrorSystemNotReady", "system not yet initialized",
          "NVSwitch systems: start nvidia-fabricmanager (same version as the driver)"),
    803: ("cudaErrorSystemDriverMismatch", "system has unsupported display driver / cuda driver combination",
          "user-mode libcuda must match the kernel module: remove libcuda from the image / fix the compat branch"),
    804: ("cudaErrorCompatNotSupportedOnDevice", "forward compatibility was attempted on non supported HW",
          "forward compatibility is data-center-GPU only: upgrade the host driver instead"),
}
NVML_MISMATCH = ("Failed to initialize NVML: Driver/library version mismatch: the loaded kernel "
                 "module and libnvidia-ml.so differ (driver upgraded without reboot, or NVML baked "
                 "into the image)")


def ver(s: str) -> tuple:
    return tuple(int(x) for x in str(s).split("."))


def driver_cuda(driver: str) -> str | None:
    """The newest CUDA version this driver supports ('CUDA Version' in nvidia-smi's banner)."""
    ok = [c for c, d in CUDA_MIN_DRIVER.items() if ver(driver) >= ver(d)]
    return max(ok, key=ver) if ok else None


def parse_targets(spec: str) -> tuple[list, list]:
    """'8.0 8.6 9.0+PTX' (TORCH_CUDA_ARCH_LIST style) or 'sm_80,sm_90a,compute_90' -> (sass, ptx)."""
    sass, ptx = [], []
    for tok in re.split(r"[\s,;]+", spec.strip()):
        m = re.fullmatch(r"(\d+)\.(\d+)([af]?)(\+PTX)?", tok)
        if m:
            name = f"{m.group(1)}{m.group(2)}{m.group(3)}"
            sass.append("sm_" + name)
            if m.group(4):
                ptx.append("compute_" + name)
        elif tok.startswith("sm_"):
            sass.append(tok)
        elif tok.startswith("compute_"):
            ptx.append(tok)
        elif tok:
            raise ValueError(f"unrecognised target {tok!r}")
    return sass, ptx


def _arch(target: str) -> tuple:
    digits, suffix = re.fullmatch(r"(?:sm|compute)_(\d+)([af]?)", target).groups()
    return (int(digits[:-1]), int(digits[-1])), suffix


def sass_runs_on(target: str, cc: str) -> bool:
    """SASS for sm_XY runs on CC X.Z with Z >= Y: same major, never across majors. 'a' targets
    (sm_90a) use arch-specific features and run only on exactly X.Y; 'f' targets (CUDA 12.9+)
    run within the family, i.e. the same major (verify)."""
    (major, minor), suffix = _arch(target)
    dev = ver(cc)
    if suffix == "a":
        return dev == (major, minor)
    return dev[0] == major and dev[1] >= minor


def ptx_jits_to(target: str, cc: str) -> bool:
    """PTX for compute_XY can be JIT-compiled for any GPU with CC >= X.Y, across majors too.
    Arch-specific 'a' PTX only for exactly X.Y; family-specific 'f' PTX only within the family,
    i.e. the same major with minor >= Y (verify). The driver's JIT must be at least as new as
    the toolkit that produced the PTX."""
    (major, minor), suffix = _arch(target)
    dev = ver(cc)
    if suffix == "a":
        return dev == (major, minor)
    if suffix == "f":
        return dev[0] == major and dev[1] >= minor
    return dev >= (major, minor)


@dataclass
class Verdict:
    ok: bool | None           # None: outside the tables, the engine cannot judge
    code: int | None = None
    reasons: list = field(default_factory=list)

    @property
    def error(self) -> str | None:
        return None if self.code is None else f"{ERRORS[self.code][0]} ({self.code}): {ERRORS[self.code][1]}"

    def __str__(self) -> str:
        head = "OK" if self.ok else "CANNOT JUDGE" if self.ok is None else f"FAILS with {self.error}"
        lines = [head] + [f"  - {r}" for r in self.reasons]
        return "\n".join(lines + ([f"  fix: {ERRORS[self.code][2]}"] if self.code else []))


def check(app_cuda: str, driver: str, gpu: str | None = None, cc: str | None = None,
          targets: str | None = None, compat_cuda: str | None = None,
          compat_branch_ok: bool | None = None, datacenter: bool | None = None) -> Verdict:
    """Will an app built with CUDA `app_cuda`, carrying kernel `targets`, run on `gpu` (or CC
    `cc`) under host `driver`? `compat_cuda` = the cuda-compat package's version, if loaded.
    `compat_branch_ok` overrides the COMPAT_BRANCHES lookup for the driver's branch."""
    if gpu is not None:
        cc, _, dc = GPUS[gpu]
        datacenter = dc if datacenter is None else datacenter
    datacenter = True if datacenter is None else datacenter
    drv = driver_cuda(driver)
    if drv is None:
        oldest = min(CUDA_MIN_DRIVER, key=ver)
        return Verdict(None, None, [f"driver {driver} is older than this table's oldest row (CUDA {oldest} "
                                    f"needs {CUDA_MIN_DRIVER[oldest]}), so neither gate can be judged here"])
    why = [f"driver {driver} supports CUDA up to {drv}"]
    if drv in INFERRED:
        why[0] += f" (inferred from the R{CUDA_MIN_DRIVER[drv]} branch; verify the exact minimum)"

    first = FIRST_CUDA.get(cc)                                    # gate 0: does the driver know the GPU?
    if first and ver(first) > ver(drv):
        why.append(f"CC {cc} is first targeted by CUDA {first}; this driver's branch predates that "
                   "toolkit, and drivers that old generally do not know the GPU, so the kernel module "
                   "does not bind it (nvidia-smi: 'No devices were found'; verify the GPU's minimum driver)")
        return Verdict(False, 100, why)

    effective = drv                                               # gate 1: driver API version
    if compat_cuda and ver(compat_cuda)[:2] <= ver(drv)[:2]:
        why.append(f"the driver already supports CUDA {compat_cuda}: cuda-compat is not needed here")
        compat_cuda = None
    if compat_cuda:
        if not datacenter:
            return Verdict(False, 804, why + ["cuda-compat only works on data-center GPUs"])
        branch = ver(driver)[0]
        listed = COMPAT_BRANCHES.get(".".join(str(x) for x in ver(compat_cuda)[:2]))
        if compat_branch_ok is None and listed is not None:
            compat_branch_ok = branch in listed
        if compat_branch_ok is False:
            return Verdict(False, 803, why + [f"cuda-compat {compat_cuda} does not support kernel-driver "
                                              f"branch R{branch} (supported: {listed or 'see the guide'})"])
        effective = compat_cuda
        why.append(f"cuda-compat {compat_cuda} supplies a newer user-mode libcuda over kernel module {driver}"
                   + ("" if compat_branch_ok else
                      f"; branch R{branch} assumed supported, not in this table (verify)"))
    app, eff = ver(app_cuda)[:2], ver(effective)[:2]
    minor_compat = False
    if app[0] > eff[0]:
        return Verdict(False, 35, why + [f"runtime {app_cuda} is a newer major than the driver's {effective}"])
    if app[0] == eff[0] and app[1] > eff[1]:
        floor = MINOR_COMPAT_FLOOR.get(app[0])
        if not compat_cuda and (floor is None or ver(driver) < ver(floor)):
            return Verdict(False, 35, why + [f"runtime {app_cuda} > {effective} and the driver is below "
                                             f"the CUDA {app[0]}.x floor {floor}"])
        minor_compat = True
        why.append(f"runtime {app_cuda} > driver {effective}, same major: runs by minor-version "
                   f"compatibility (driver >= {floor}); newer APIs may fail with error 36")
    else:
        why.append(f"driver {effective} >= runtime {app_cuda}: backward compatible")

    if targets is None:                                           # gate 2: a kernel image for this GPU
        return Verdict(True, None, why + ["kernel images not checked (no targets given)"])
    sass, ptx = parse_targets(targets)
    native = [s for s in sass if sass_runs_on(s, cc)]
    if native:
        best = max(native, key=lambda s: _arch(s)[0])
        return Verdict(True, None, why + [f"SASS {best} runs natively on CC {cc}"])
    jit = [p for p in ptx if ptx_jits_to(p, cc)]
    if not jit:
        return Verdict(False, 209, why + [f"no SASS for CC {cc} among {sass or 'none'} and no PTX that "
                                          f"can JIT to CC {cc} among {ptx or 'none'}"])
    best = max(jit, key=lambda p: _arch(p)[0])
    if minor_compat:
        return Verdict(False, 222, why + [f"only PTX {best} fits, it was generated by CUDA {app_cuda}, "
                                          f"and this driver's JIT only understands up to {effective}"])
    return Verdict(True, None, why + [f"no matching SASS: the driver JIT-compiles {best} at load "
                                      "(slow first start, cached in ~/.nv/ComputeCache)"])


# The two halves of a GPU container.
HOST_INJECTED = ("/dev/nvidia", "libcuda.so", "libnvidia-", "nvidia-smi", "nvidia-cuda-mps",
                 "nvidia-persistenced", "nvidia-debugdump")


def origin(path: str) -> str:
    """Where a GPU file inside a container comes from: the image, or the host driver install
    (mounted in by the NVIDIA Container Toolkit, so it always matches the kernel module)."""
    if "/compat/" in path:
        return "image (cuda-compat forward-compatibility package)"
    if any(k in path for k in HOST_INJECTED):
        return "host (injected by the NVIDIA Container Toolkit)"
    return "image"


def explain_container(image_cuda: str, driver: str, gpu: str, targets: str | None = None,
                      gpus_injected: bool = True, libcuda_in_image: bool = False,
                      compat_in_image: bool = False) -> Verdict:
    """The image brings the CUDA runtime and libraries. The toolkit injects /dev/nvidia* and the
    host's user-mode driver (libcuda, NVML). Check both, then the two gates of `check()`."""
    if not gpus_injected:
        return Verdict(False, 100, ["no GPU injected: no --gpus / CDI device, NVIDIA_VISIBLE_DEVICES "
                                    "unset or void, or the pod requested no nvidia.com/gpu"])
    if libcuda_in_image:
        return Verdict(False, 803, ["the image ships its own libcuda.so, which shadows the host's; the "
                                    "user-mode driver must match the host kernel module exactly"])
    drv = driver_cuda(driver)
    needs_compat = compat_in_image and drv is not None and ver(image_cuda)[:2] > ver(drv)[:2]
    return check(image_cuda, driver, gpu=gpu, targets=targets,
                 compat_cuda=image_cuda if needs_compat else None)
