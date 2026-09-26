# %% [markdown]
# # 04 · Compatibility and containers
#
# **Tier:** T0 (CPU; a rules engine over dated version tables, marked *verify*). On a real
# machine, the lab's `05_how_a_container_sees_a_gpu` (T1) reads `/dev/nvidia*`, the injected
# `libcuda` mount and the driver and runtime versions from inside a container.
#
# ## The one-minute version
# Whether a CUDA program runs comes down to **two independent gates**:
#
# 1. **Driver and runtime.** The host driver must support the CUDA version your app or image was
#    built with. Newer drivers run older runtimes. A newer runtime *of the same major* runs on
#    any driver above that major's floor (*minor-version compatibility*), at two costs: the
#    driver cannot JIT-compile that toolkit's PTX, and APIs newer than the driver fail. A newer
#    *major* needs a newer driver, or the forward-compatibility package on a data-center GPU.
# 2. **Kernel image and GPU.** SASS built for `sm_XY` runs only on compute capability `X.Z` with
#    `Z >= Y`. PTX for `compute_XY` can be JIT-compiled for any newer GPU. With neither you get
#    *"no kernel image is available for execution on the device"*.
#
# A container carries the CUDA runtime and libraries. The **host** provides the driver: the
# NVIDIA Container Toolkit injects `/dev/nvidia*`, `libcuda.so` and NVML when the container
# starts. The "CUDA Version" in `nvidia-smi` is the *driver's maximum*, not what is installed.
#
# Primer: §1 *The stack from driver to framework* and §6 *How a container gets a GPU* (`../../PRIMER.md`).

# %%
from gpusim import compat

print("CUDA toolkit -> minimum Linux driver (release notes; dated Sep 2026, verify):")
print("  " + "  ".join(f"{c}:{d}" for c, d in compat.CUDA_MIN_DRIVER.items()))
for drv in ("470.256.02", "535.183.01", "550.127.05", "570.86.10", "580.65.06"):
    print(f"driver {drv:>11} -> nvidia-smi says 'CUDA Version: {compat.driver_cuda(drv)}'")

# %% [markdown]
# `check()` walks both gates and names the error you would get. Three runs of the same
# CUDA 12.4 application on an H100 with a 535 driver, which supports CUDA up to 12.2:

# %%
print(compat.check("12.4", "535.183.01", gpu="H100", targets="8.0 9.0"), "\n")
print(compat.check("12.4", "535.183.01", gpu="H100", targets="8.0+PTX"), "\n")
print(compat.check("13.0", "535.183.01", gpu="H100", targets="9.0"))

# %% [markdown]
# Targets use either spelling. `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0+PTX"` means SASS for
# sm_80, sm_86 and sm_90, plus PTX for compute_90 (the `+PTX`):

# %%
print(compat.parse_targets("8.0 8.6 9.0+PTX"))
print(compat.parse_targets("sm_90a,compute_90a"))
for gpu in ("T4", "A100", "L4", "H100", "B200", "RTX 5090"):
    cc = compat.GPUS[gpu][0]
    runs = [t for t in ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_120") if compat.sass_runs_on(t, cc)]
    print(f"{gpu:>9} (CC {cc:>4}) runs SASS: {runs}")

# %% [markdown]
# ## Exercise 4.1: the binary-compatibility rules
#
# Write `sass_runs(target, cc)` and `ptx_runs(target, cc)` for targets like `"sm_86"`,
# `"sm_90a"`, `"compute_80"` or `"compute_120"`. The digits are major then one minor digit, so
# `sm_100` is 10.0 and `sm_120` is 12.0. `cc` is a string like `"8.9"`.
#
# * SASS `sm_XY`: same major **and** device minor >= Y. With the `a` suffix: exactly X.Y.
# * PTX `compute_XY`: device CC >= X.Y as a (major, minor) pair, across majors too. With `a`:
#   exactly X.Y.
# * `f` (family-specific, CUDA 12.9+; verify): SASS and PTX alike stay inside the family, the same
#   major with device minor >= Y. `sm_100f` runs on 10.0 and 10.3, never on 12.0.

# %% exercise
def _parse(target):
    body = target.split("_")[1]
    suffix = body[-1] if body[-1].isalpha() else ""
    digits = body.rstrip("af")
    return (int(digits[:-1]), int(digits[-1])), suffix

def sass_runs(target, cc):
    ### BEGIN SOLUTION
    (major, minor), suffix = _parse(target)
    dev = tuple(int(x) for x in cc.split("."))
    if suffix == "a":
        return dev == (major, minor)
    return dev[0] == major and dev[1] >= minor
    ### END SOLUTION

def ptx_runs(target, cc):
    ### BEGIN SOLUTION
    (major, minor), suffix = _parse(target)
    dev = tuple(int(x) for x in cc.split("."))
    if suffix == "a":
        return dev == (major, minor)
    if suffix == "f":
        return dev[0] == major and dev[1] >= minor
    return dev >= (major, minor)
    ### END SOLUTION

# %% check
ccs = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "10.3", "12.0"]
for t in ["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_100f", "sm_120"]:
    for cc in ccs:
        assert sass_runs(t, cc) == compat.sass_runs_on(t, cc), (t, cc)
for t in ["compute_75", "compute_80", "compute_90", "compute_90a", "compute_100", "compute_100f", "compute_120"]:
    for cc in ccs:
        assert ptx_runs(t, cc) == compat.ptx_jits_to(t, cc), (t, cc)
print("✅ SASS: within a major, upward only. PTX: upward across majors. 'a': one exact GPU. 'f': one family")

# %% [markdown]
# ## Exercise 4.2: predict the verdict
#
# For each scenario, predict `None` if it runs or the CUDA error **code** it fails with
# (35 insufficient driver, 100 no device, 209 no kernel image, 222 unsupported PTX,
# 803 driver mismatch, 804 forward compatibility on unsupported hardware). Reason it out first.
# Forward compatibility also needs a kernel-driver branch the cuda-compat package supports:
# `compat.COMPAT_BRANCHES` lists them (dated, verify).

# %% exercise
scenarios = {
    "a": dict(app_cuda="12.6", driver="560.35.03", gpu="L4", targets="8.0 8.9"),
    "b": dict(app_cuda="12.6", driver="550.127.05", gpu="L4", targets="8.0 8.9"),
    "c": dict(app_cuda="12.6", driver="550.127.05", gpu="L4", targets="compute_80"),
    "d": dict(app_cuda="13.0", driver="570.86.10", gpu="A100", targets="8.0"),
    "e": dict(app_cuda="12.8", driver="570.86.10", gpu="T4", targets="8.0 9.0+PTX"),
    "f": dict(app_cuda="13.0", driver="535.183.01", gpu="L4", targets="8.9", compat_cuda="13.0"),
    "g": dict(app_cuda="13.0", driver="535.183.01", gpu="RTX 4090", targets="8.9", compat_cuda="13.0"),
    "h": dict(app_cuda="13.0", driver="560.35.03", gpu="H100", targets="9.0", compat_cuda="13.0"),
}
predicted = {k: "?" for k in scenarios}
### BEGIN SOLUTION
predicted.update(a=None,    # driver is new enough, sm_89 SASS
                 b=None,    # 12.6 on a 12.4 driver: minor-version compatibility, and there is SASS
                 c=222,     # same, but PTX only: the 550 JIT cannot read 12.6's PTX
                 d=35,      # a new major on a 12.8 driver, without compat
                 e=209,     # T4 is 7.5: no sm_7x SASS, and compute_90 PTX cannot go down to 7.5
                 f=None,    # forward compatibility on a data-center GPU; R535 is a supported branch
                 g=804,     # forward compatibility on GeForce is refused
                 h=803)     # R560 is not among the 13.0 compat package's branches
### END SOLUTION

# %% check
for k, sc in scenarios.items():
    v = compat.check(**sc)
    assert predicted[k] == v.code, f"scenario {k}: you said {predicted[k]}, the engine says {v.code}\n{v}"
print("✅ all eight verdicts right. Scenario e is the classic: a wheel with no build for your GPU generation")

# %% [markdown]
# ## Exercise 4.3: choose the CUDA version for a fleet's base image
#
# A fleet has three node pools with drivers `535.183.01`, `550.127.05` and `570.86.10`, running
# L4, A100 and H100 GPUs. Your kernels ship SASS for all three GPUs. Compute:
#
# * `native`: the newest CUDA that every pool supports **without** relying on minor-version compatibility
# * `with_minor_compat`: the newest CUDA version in `compat.CUDA_MIN_DRIVER` that runs on every
#   pool **by** minor-version compatibility (same major as every driver, every driver above that
#   major's floor in `compat.MINOR_COMPAT_FLOOR`)

# %% exercise
drivers = ["535.183.01", "550.127.05", "570.86.10"]
### BEGIN SOLUTION
native = min((compat.driver_cuda(d) for d in drivers), key=compat.ver)
majors = {compat.ver(compat.driver_cuda(d))[0] for d in drivers}
assert len(majors) == 1
major = majors.pop()
assert all(compat.ver(d) >= compat.ver(compat.MINOR_COMPAT_FLOOR[major]) for d in drivers)
with_minor_compat = max((c for c in compat.CUDA_MIN_DRIVER if compat.ver(c)[0] == major), key=compat.ver)
### END SOLUTION

# %% check
assert native == "12.2" and with_minor_compat == "12.9"
for d in drivers:
    for gpu, t in (("L4", "8.9"), ("A100", "8.0"), ("H100", "9.0")):
        assert compat.check(with_minor_compat, d, gpu=gpu, targets=t).ok
assert all(compat.check("12.9", d, gpu="H100", targets="8.0+PTX").code == 222 for d in drivers)   # no PTX JIT
print(f"✅ native: CUDA {native}. With minor-version compatibility: CUDA {with_minor_compat}, provided every kernel")
print("   ships SASS: every pool's driver is older than 12.9, so PTX-only kernels hit error 222 on all three.")
print("   CUDA 13 needs a driver upgrade.")

# %% [markdown]
# ## Anatomy of a GPU container
# The image ships the **userland** (CUDA runtime, cuBLAS, cuDNN, NCCL, your framework). The host
# ships the **driver**. At container start the NVIDIA Container Toolkit, either through an OCI
# prestart hook or a CDI spec, adds the device nodes and bind-mounts the host's user-mode driver
# libraries, which must match the host kernel module exactly.

# %%
for path in ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
             "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "/usr/bin/nvidia-smi",
             "/usr/local/cuda/lib64/libcudart.so.12", "/usr/lib/python3/site-packages/nvidia/nccl/lib/libnccl.so.2",
             "/usr/local/cuda/compat/libcuda.so.1"]:
    print(f"{path:<62} <- {compat.origin(path)}")

# %% [markdown]
# ## Exercise 4.4: diagnose five container stories
#
# Predict `None` (it works) or the error code for each. `compat.explain_container()` is the
# referee.
#
# 1. `docker run my-image python -c "import torch; torch.zeros(1).cuda()"`, but the run command
#    has no `--gpus` and no CDI device.
# 2. The Dockerfile copies `libcuda.so` from the build machine into `/usr/lib` "so the image is
#    self-contained". The production host runs a different driver.
# 3. A CUDA 13.0 image that includes `cuda-compat`, on an L4 host with driver 535.
# 4. The same image on a workstation RTX 4090 with driver 535.
# 5. A CUDA 12.8 image whose wheel has SASS for 8.0, 8.6 and 9.0 but no PTX, on a GKE L4 node
#    with driver 535.

# %% exercise
stories = {
    1: dict(image_cuda="12.4", driver="550.127.05", gpu="L4", targets="8.9", gpus_injected=False),
    2: dict(image_cuda="12.4", driver="570.86.10", gpu="L4", targets="8.9", libcuda_in_image=True),
    3: dict(image_cuda="13.0", driver="535.183.01", gpu="L4", targets="8.9", compat_in_image=True),
    4: dict(image_cuda="13.0", driver="535.183.01", gpu="RTX 4090", targets="8.9", compat_in_image=True),
    5: dict(image_cuda="12.8", driver="535.183.01", gpu="L4", targets="8.0 8.6 9.0"),
}
diagnosis = {k: "?" for k in stories}
### BEGIN SOLUTION
diagnosis.update({1: 100,    # no device injected
                  2: 803,    # user-mode driver in the image != host kernel module
                  3: None,   # forward compat on a data-center GPU, over a supported branch (R535)
                  4: 804,    # forward compat refused on GeForce
                  5: None})  # sm_86 SASS runs on 8.9; 12.8 on a 12.2 driver by minor-version compat
### END SOLUTION

# %% check
for k, st in stories.items():
    v = compat.explain_container(**st)
    assert diagnosis[k] == v.code, f"story {k}: you said {diagnosis[k]}, the engine says {v.code}\n{v}"
print("✅ five for five. Rule of thumb: the image brings CUDA, the host brings the driver, and")
print("   nothing in the image may pretend to be the driver")

# %% [markdown]
# ## In a design review
#
# **The two-minute version.** "We pin a driver branch per node pool and build images against a
# CUDA version at or below what that driver supports, or within its major if we have checked that
# every kernel ships SASS for our GPUs. Images carry only userland: CUDA runtime, libraries,
# framework. The NVIDIA Container Toolkit injects the device nodes and the host's libcuda and NVML
# at start, so user-mode and kernel-mode driver always match. We never copy libcuda into an image.
# Our wheels list their targets (`torch.cuda.get_arch_list()`), and we check them against each
# GPU generation before rollout. New architectures (Blackwell, CC 10.x and 12.x) need wheels built
# with CUDA 12.8 or later. Upgrading the driver is the rollout gate, not the image."
#
# **Drill questions**
#
# 1. *`nvidia-smi` says CUDA 12.2 while `nvcc --version` in the container says 12.4. Which is
#    wrong?* Neither. The first is the driver's maximum and the second is the toolkit in the
#    image. It runs by minor-version compatibility, but PTX JIT and APIs newer than 12.2 will
#    not work.
# 2. *PyTorch on a new RTX 50-series card says "no kernel image is available". Why?* The wheel
#    has no sm_120 SASS and no PTX that can JIT to 12.0. Install a build made with CUDA 12.8+
#    that targets sm_120.
# 3. *After a host driver upgrade, `nvidia-smi` says "Driver/library version mismatch". Why?*
#    The kernel module still loaded is the old one, while the user-space NVML is new. Reboot
#    (or reload the module) so they match.
