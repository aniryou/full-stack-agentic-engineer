# %% [markdown]
# # 05 · How a container sees a GPU: device nodes, injected drivers and the compatibility contract
#
# **Tier:** T0 — inspects *this* machine live (a machine without a GPU is a perfectly good first case) and
# two bundled probe logs (**illustrative**: written in the exact format of `deploy/any-gpu/probe.sh`,
# not captured from real machines). **T1/T3** — run `probe.sh` in your own GPU container, or
# `deploy/gke/run.sh smoke` on GKE, and feed the log to the same cells.
#
# ## The one-minute version
#
# * A containerised CUDA program is assembled from two sources. **From the host**, injected when the
#   container starts: the device nodes (`/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`) and the
#   user-mode driver (`libcuda.so`, `libnvidia-ml.so`, `nvidia-smi`), which must match the host's kernel
#   module exactly. **From the image**: the CUDA runtime, cuBLAS/cuDNN/NCCL and the framework.
# * Docker/containerd use the **NVIDIA Container Toolkit** (an OCI hook, or a CDI spec) to bind-mount the
#   driver files one by one; **GKE's device plugin** mounts the node's driver directory at `/usr/local/nvidia`.
# * Two gates decide whether a kernel runs: **driver vs runtime** (the driver's CUDA version must be ≥ the
#   runtime's, or the same major under minor-version compatibility) and **kernel image vs GPU** (SASS for
#   the GPU's major with minor ≤, or PTX the driver can JIT).
# * The errors, by number: 35 driver too old, 209 no kernel image, 222 PTX too new for the driver,
#   803 a mismatched `libcuda`, 100 no device at all.
#
# Concepts: [the primer](../../PRIMER.md) §1 *The stack from driver to framework* and §6 *How a
# container gets a GPU*.

# %%
from pathlib import Path

from gpurt import container as c
from gpurt import env

print(env.describe())
live = c.probe()  # the driver-API probe runs in a child process: this kernel never calls cuInit
print(c.explain(live))

# %% [markdown]
# On a machine without a GPU every layer is missing — and that is the first lesson: **an image cannot
# bring a GPU with it.** The container runtime (`docker run --gpus all`, a CDI device, a Kubernetes
# `nvidia.com/gpu` request) has to hand in the device nodes and the host's driver files.
#
# ## A container started with `--gpus all`
#
# The Docker sample: the NVIDIA Container Toolkit has bind-mounted each driver file from the host.

# %%
FIX = Path(c.__file__).parent / "fixtures"
docker_log = (FIX / "probe_docker_t4_sample.log").read_text()
print("\n".join(docker_log.splitlines()[:2]))
sections = c.parse_probe_log(docker_log)
print("\nsections:", list(sections))
print("\n".join(sections["mountinfo"].splitlines()[5:9]))

# %% [markdown]
# Each line of `/proc/self/mountinfo` is `id parent major:minor root mount-point options ... - fstype
# source super-options`. The toolkit's entries are read-only bind mounts whose mount point is the path the
# program sees (`/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08`) — the driver version is in the file name.
#
# ## Exercise 5.1 — read a mount line
#
# Write `mount_point(line)` (the fifth whitespace-separated field) and `driver_version(path)`: the
# version in `libcuda.so.<version>` (return `None` if the path is not a versioned libcuda).

# %% exercise
import re  # noqa: E402


def mount_point(line: str) -> str:
    ### BEGIN SOLUTION
    return line.split()[4]
    ### END SOLUTION


def driver_version(path: str) -> str | None:
    ### BEGIN SOLUTION
    m = re.search(r"libcuda\.so\.(\d+\.\d+(?:\.\d+)?)$", path)
    return m[1] if m else None
    ### END SOLUTION

# %% check
libcuda_line = next(l for l in sections["mountinfo"].splitlines() if "libcuda.so" in l)
assert mount_point(libcuda_line) == "/usr/lib/x86_64-linux-gnu/libcuda.so.570.172.08"
assert driver_version(mount_point(libcuda_line)) == "570.172.08"
assert driver_version("/usr/lib/x86_64-linux-gnu/libcuda.so.1") is None
print("✅ the injected libcuda names the host driver: 570.172.08 — which supports CUDA up to",
      c.cuda_of_driver("570.172.08"))

# %%
docker = c.from_probe_log(docker_log)
print(c.explain(docker))

# %% [markdown]
# ## The same GPU, the GKE way
#
# On GKE the device plugin mounts the node's driver directory at `/usr/local/nvidia` (and `nvidia/cuda`
# images already put `/usr/local/nvidia/lib64` on `LD_LIBRARY_PATH`), while the device nodes are added
# to the pod's cgroup.

# %%
gke = c.from_probe_log((FIX / "probe_gke_l4_sample.log").read_text())
print(c.explain(gke))

# %% [markdown]
# ## Exercise 5.2 — gate 1: driver versus runtime
#
# Write `driver_runtime(driver_cuda, runtime_cuda)` returning `"ok"` (the driver supports that CUDA or a
# newer one), `"minor-compat"` (same major, the runtime's minor is newer: runs, with the PTX-JIT and
# new-API caveats) or `"fail"` (a newer major: error 35 unless the forward-compatibility package applies).
# Versions are strings like `"12.8"`.

# %% exercise
def driver_runtime(driver_cuda: str, runtime_cuda: str) -> str:
    ### BEGIN SOLUTION
    d = tuple(int(x) for x in driver_cuda.split("."))
    r = tuple(int(x) for x in runtime_cuda.split("."))
    if r[0] > d[0]:
        return "fail"
    return "minor-compat" if r > d else "ok"
    ### END SOLUTION

# %% check
cases = {("12.8", "12.4"): "ok", ("12.2", "12.4"): "minor-compat", ("11.4", "12.1"): "fail",
         ("13.0", "12.8"): "ok", ("12.8", "13.0"): "fail", ("12.4", "12.4"): "ok"}
for (d, r), want in cases.items():
    assert driver_runtime(d, r) == want, (d, r)
    assert c.compat_verdicts(d, r)[0].level == {"ok": "ok", "minor-compat": "warn", "fail": "fail"}[want]
print("✅ newer drivers run older runtimes; a newer minor runs with caveats; a newer major needs a new driver")

# %% [markdown]
# ## Exercise 5.3 — gate 2: kernel image versus GPU
#
# A binary carries SASS (`sm_XY`, machine code) and/or PTX (`compute_XY`, JIT-able). Write
# `kernel_image(cc, archs)` returning `"sass"` if some SASS runs on compute capability `cc` (same major,
# minor ≤; an `a` suffix runs only on exactly that version), else `"ptx-jit"` if some PTX can be JIT-compiled
# (any version ≤ cc; `a` only exact), else `"no-kernel-image"` (error 209). Use `c.parse_arch` to split
# `"sm_90a"` into `('sass', 9, 0, 'a')`.

# %% exercise
def kernel_image(cc: tuple[int, int], archs: list[str]) -> str:
    ### BEGIN SOLUTION
    def runs(kind, major, minor, suffix):
        if suffix == "a":
            return (major, minor) == tuple(cc)
        if kind == "sass":
            return major == cc[0] and minor <= cc[1]
        return (major, minor) <= tuple(cc)

    parsed = [c.parse_arch(a) for a in archs]
    if any(p[0] == "sass" and runs(*p) for p in parsed):
        return "sass"
    if any(p[0] == "ptx" and runs(*p) for p in parsed):
        return "ptx-jit"
    return "no-kernel-image"
    ### END SOLUTION

# %% check
from itertools import combinations


def reference(cc, archs):  # the lab's compatibility model (gpurt.container, kept in step with gpusim.compat)
    if any(a.startswith("sm_") and c.sass_runs_on(a, cc) for a in archs):
        return "sass"
    if any(a.startswith("compute_") and c.ptx_jits_on(a, cc) for a in archs):
        return "ptx-jit"
    return "no-kernel-image"


assert kernel_image((7, 5), ["sm_80", "sm_86", "sm_90"]) == "no-kernel-image"  # a T4 with an Ampere+ build
assert kernel_image((8, 9), ["sm_80"]) == "sass"  # L4 runs sm_80 SASS: same major, 0 <= 9
assert kernel_image((8, 6), ["sm_80"]) == "sass"  # A10 runs sm_80 SASS
assert kernel_image((8, 0), ["sm_86"]) != "sass"  # but an A100 cannot run sm_86 SASS: its minor is lower
assert kernel_image((9, 0), ["sm_80", "compute_80"]) == "ptx-jit"  # H100: no SASS across majors, PTX JITs up
assert kernel_image((12, 0), ["sm_80", "sm_90", "compute_90"]) == "ptx-jit"  # RTX PRO 6000: JIT from PTX
assert kernel_image((10, 3), ["sm_100", "compute_100"]) == "sass"  # B300 runs sm_100 SASS
assert kernel_image((9, 0), ["sm_90a"]) == "sass"
assert kernel_image((10, 0), ["sm_90a", "compute_90a"]) == "no-kernel-image"  # arch-specific does not carry forward
# ... and against the model over every pair of targets on every GPU generation in c.ARCH
POOL = ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103", "sm_120",
        "compute_75", "compute_80", "compute_86", "compute_90", "compute_90a", "compute_100", "compute_120"]
pairs = [[a] for a in POOL] + [list(p) for p in combinations(POOL, 2)]
wrong = [(cc, archs) for cc in c.ARCH for archs in pairs if kernel_image(cc, archs) != reference(cc, archs)]
assert not wrong, f"{len(wrong)} of {len(c.ARCH) * len(pairs)} cases differ, e.g. cc={wrong[0][0]} archs={wrong[0][1]}: " \
    f"got {kernel_image(*wrong[0])!r}, the model says {reference(*wrong[0])!r}"
print(f"✅ {len(c.ARCH) * len(pairs)} (GPU, build) cases: SASS within a major for an equal or newer minor, "
      "PTX upward, 'a' targets exactly one GPU — anything else is error 209")

# %% [markdown]
# ## Exercise 5.4 — a whole scenario
#
# An image built on `nvidia/cuda:12.8.1-runtime` with a PyTorch wheel whose arch list is
# `["sm_75", "sm_80", "sm_86", "sm_90"]` lands on three nodes. Write `verdicts(driver_version, cc)` that
# returns the pair `(driver_runtime(...), kernel_image(...))` for this image, using `c.cuda_of_driver` to
# turn a driver version into the CUDA version it supports.

# %% exercise
IMAGE_CUDA, IMAGE_ARCHS = "12.8", ["sm_75", "sm_80", "sm_86", "sm_90"]


def verdicts(driver_version: str, cc: tuple[int, int]) -> tuple[str, str]:
    ### BEGIN SOLUTION
    return driver_runtime(c.cuda_of_driver(driver_version), IMAGE_CUDA), kernel_image(cc, IMAGE_ARCHS)
    ### END SOLUTION

# %% check
assert verdicts("570.172.08", (8, 9)) == ("ok", "sass")  # L4, current driver: runs
assert verdicts("550.54.15", (7, 5)) == ("minor-compat", "sass")  # T4 on a 12.4 driver: runs, with caveats
assert verdicts("580.65.06", (12, 0)) == ("ok", "no-kernel-image")  # Blackwell workstation GPU: needs sm_120
print("✅ the same image: fine on an L4, fine-with-caveats on an older driver, error 209 on a GPU it was not built for")

# %% [markdown]
# ## Failure modes, as they show up in a probe
#
# | What went wrong | What the probe shows | Error |
# |---|---|---|
# | container started without `--gpus` / no `nvidia.com/gpu` request | no `/dev/nvidia*`, no driver mounts | 100 (no device) |
# | image copied a `libcuda.so` from a build machine | a `libcuda.so.<other version>` that is not a mount | 803 |
# | image CUDA a newer major than the host driver | driver CUDA < runtime CUDA | 35 |
# | wheel without this GPU's SASS or usable PTX | verdict "no kernel image" | 209 |
# | NCCL slow or hanging in a container | tiny `/dev/shm` (64 MB Docker default) | — (use `--shm-size`, `--ipc=host`) |
#
# `c.stray_libcuda()` looks for the 803 case on a live system: a versioned `libcuda` that was not
# injected. On a GPU box, pass the arch list of the framework you use (`torch.cuda.get_arch_list()`) to
# `c.probe(archs=...)` to get the gate-2 verdict for the real binary.

# %%
if live.devices:
    print("T1: this machine has a GPU; with torch installed, try c.probe(archs=torch.cuda.get_arch_list())")
else:
    print("T0: no GPU here. On a GPU host: `docker run --rm --gpus all -v $PWD/deploy/any-gpu:/w "
          "nvidia/cuda:12.8.1-base-ubuntu24.04 bash /w/probe.sh > probe.log`, then c.from_probe_log(open('probe.log').read())")

# %% [markdown]
# ## In a design review
#
# **Two minutes.** A GPU container has two halves. The host half — device nodes and the user-mode driver,
# which must match the kernel module to the build — is injected when the container starts: the NVIDIA
# Container Toolkit bind-mounts it on Docker and containerd, GKE's device plugin mounts it at
# `/usr/local/nvidia`. The image half is the CUDA userland and the framework. Two gates follow: the
# driver must support the image's CUDA version (backward compatible always, minor-version compatible with
# caveats, a newer major only with the forward-compatibility package on data-center GPUs), and the binary
# must contain SASS for the GPU's architecture or PTX the driver can JIT. When something fails, I read the
# error number — 35, 209, 222, 803, 100 — and it tells me which half and which gate.
#
# **Drill questions**
#
# 1. *`nvidia-smi` works in the container, but `torch.cuda.is_available()` is False. Why?* — `nvidia-smi`
#    needs only NVML; torch needs a CUDA build of the wheel (a CPU-only wheel returns False), a driver that
#    supports the wheel's CUDA version, and no `CUDA_VISIBLE_DEVICES` masking every GPU.
# 2. *Why can't the image ship its own `libcuda.so`?* — The user-mode driver talks to the kernel module
#    through a private interface that changes with every driver build; a mismatch fails with 803. Only the
#    host knows which build it runs, so the runtime injects it.
# 3. *A new GPU generation arrives and last year's image fails with "no kernel image". Fix?* — Rebuild or
#    reinstall the framework for the new compute capability (or ship PTX so the driver can JIT); upgrading
#    the driver alone does not help, because the binary has nothing for that GPU.
