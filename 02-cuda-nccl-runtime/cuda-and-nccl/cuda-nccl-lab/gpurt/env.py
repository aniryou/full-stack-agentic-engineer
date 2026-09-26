"""Where am I running? One place that answers "is there a GPU, which tier is this, and should
Numba use its CUDA simulator?"

The one idea: Numba picks between the real CUDA target and the CPU simulator **once, when
``numba.cuda`` is first imported**, from the environment variable ``NUMBA_ENABLE_CUDASIM``. So the
decision has to be made before that import — which is what :func:`ensure_numba_mode` does, and why
``gpurt.kernels`` calls it first. Detection here is deliberately cheap and side-effect free: it looks
for device nodes and ``nvidia-smi``; it never calls ``cuInit`` in this process (a process that has
initialised CUDA cannot hand CUDA to children it forks).

Tiers (see the lab README):  T0 = no GPU (simulator, fixtures, CPU backends) · T1 = one GPU ·
T2 = two or more GPUs · T3 = a managed cluster (GKE), detected here only as "in Kubernetes".
"""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager

SIM_ENV = "NUMBA_ENABLE_CUDASIM"


def gpu_device_nodes() -> list[str]:
    """``/dev/nvidia0``, ``/dev/nvidia1``, ... — one per GPU this process may open (Linux)."""
    return sorted(p for p in glob.glob("/dev/nvidia[0-9]*") if p[len("/dev/nvidia"):].isdigit())


def nvidia_smi_gpus(timeout: float = 10.0) -> list[str]:
    """Lines of ``nvidia-smi -L`` (``GPU 0: Tesla T4 (UUID: ...)``), or [] if unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.startswith("GPU ")]


def gpu_count() -> int:
    """How many GPUs are visible, without initialising CUDA in this process.

    Order: device nodes (native Linux, Docker ``--gpus``, Kubernetes device plugins), then
    ``nvidia-smi -L`` (covers WSL2, where GPUs appear as ``/dev/dxg`` instead of ``/dev/nvidia*``).
    ``CUDA_VISIBLE_DEVICES`` narrows what CUDA programs see, so it is applied last.
    """
    n = len(gpu_device_nodes()) or len(nvidia_smi_gpus())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and n:
        ids = [v for v in visible.split(",") if v.strip() not in ("", "-1")]
        n = min(n, len(ids))
    return n


def numba_mode() -> str:
    """'simulator' or 'cuda' if ``numba.cuda`` is already imported (the choice is then fixed), else 'unset'."""
    if "numba.cuda" not in sys.modules:
        return "unset"
    from numba.core import config

    return "simulator" if config.ENABLE_CUDASIM else "cuda"


def ensure_numba_mode(simulator: bool | None = None) -> str:
    """Decide Numba's CUDA mode **before** ``numba.cuda`` is imported, and return it.

    * ``simulator=True/False`` forces the choice;
    * otherwise an explicit ``NUMBA_ENABLE_CUDASIM`` in the environment wins;
    * otherwise: a GPU is visible -> real CUDA target, no GPU -> simulator.

    If ``numba.cuda`` was already imported the mode cannot change any more; the current mode is
    returned (and a forced, conflicting request raises so the surprise is loud, not silent).
    """
    current = numba_mode()
    if current != "unset":
        if simulator is not None and simulator != (current == "simulator"):
            raise RuntimeError(
                f"numba.cuda is already imported in {current!r} mode; restart the kernel/process and set "
                f"{SIM_ENV}={'1' if simulator else '0'} before importing numba.cuda")
        return current
    if simulator is not None:
        os.environ[SIM_ENV] = "1" if simulator else "0"
    elif os.environ.get(SIM_ENV, "").strip() not in ("0", "1"):
        os.environ[SIM_ENV] = "0" if gpu_count() > 0 else "1"
    if "numba" in sys.modules:  # numba imported but not numba.cuda: re-read the environment
        from numba.core import config

        config.reload_config()
    return "simulator" if os.environ[SIM_ENV] == "1" else "cuda"


@contextmanager
def fast_simulator(switch_interval: float = 1e-4):
    """Speed up barrier-heavy kernels in the simulator.

    The simulator runs every CUDA thread of a block as a Python thread and releases them from
    ``cuda.syncthreads()`` by polling. With CPython's default 5 ms GIL switch interval each barrier
    costs milliseconds; 0.1 ms makes tiled kernels ~5x faster. Harmless on a real GPU.
    """
    old = sys.getswitchinterval()
    sys.setswitchinterval(switch_interval)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


def torch_info() -> dict | None:
    """torch version and CUDA view, or None when torch is not installed (import is lazy)."""
    if importlib.util.find_spec("torch") is None:
        return None
    import torch

    cuda_ok = bool(torch.cuda.is_available())
    return {
        "version": torch.__version__,
        "cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_ok,
        "device_count": torch.cuda.device_count() if cuda_ok else 0,
        "nccl": bool(cuda_ok and torch.distributed.is_available() and torch.distributed.is_nccl_available()),
    }


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def in_kubernetes() -> bool:
    return "KUBERNETES_SERVICE_HOST" in os.environ


def in_container() -> bool:
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            text = f.read()
        return any(k in text for k in ("docker", "kubepods", "containerd", "libpod"))
    except OSError:
        return False


def platform_hint() -> str:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return "colab"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if in_kubernetes():
        return "kubernetes"
    if in_container():
        return "container"
    return "local"


def tier(n_gpus: int | None = None) -> str:
    """T0 (no GPU), T1 (one GPU) or T2 (several). T3 is a deployment choice, not something to detect."""
    n = gpu_count() if n_gpus is None else n_gpus
    return "T0" if n == 0 else ("T1" if n == 1 else "T2")


def summary() -> dict:
    """Everything a notebook prints in its first cell. Importing torch here is deliberately avoided."""
    n = gpu_count()
    return {
        "tier": tier(n),
        "gpus": n,
        "gpu_names": [line.split(" (UUID")[0] for line in nvidia_smi_gpus()] if n else [],
        "numba_mode": numba_mode() if numba_mode() != "unset" else (
            "simulator" if os.environ.get(SIM_ENV) == "1" else ("cuda" if n else "simulator (on import)")),
        "torch_installed": has_module("torch"),
        "triton_installed": has_module("triton"),
        "platform": platform_hint(),
    }


def describe() -> str:
    s = summary()
    gpus = ", ".join(s["gpu_names"]) if s["gpu_names"] else "none"
    return (f"tier {s['tier']} | GPUs: {s['gpus']} ({gpus}) | numba: {s['numba_mode']} | "
            f"torch: {'yes' if s['torch_installed'] else 'no'} | platform: {s['platform']}")


if __name__ == "__main__":  # python -m gpurt.env
    print(describe())
