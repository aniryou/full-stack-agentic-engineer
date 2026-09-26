"""The same GPU pod on a box you control: k3s + the real NVIDIA device plugin (T1/T2).

The one idea: kind fakes ``nvidia.com/gpu`` with a node-status patch; on a rented GPU VM the
**device plugin** advertises it, **GPU Feature Discovery** labels the node with what the GPU is
(``nvidia.com/gpu.product``, ``.memory``, ``.count``, ``.replicas``), and time-slicing turns one
physical GPU into several schedulable units (primer §1.2, §1.3, §9). The pod spec barely
changes: it still asks for an integer ``nvidia.com/gpu`` limit. Two k3s specifics: k3s keeps
runc as its default runtime and registers an ``nvidia`` RuntimeClass when it finds the NVIDIA
container runtime, so GPU pods (and the device plugin) ask for that class; and nothing taints
the node, so the toleration is harmless rather than required.

``gpuvm_manifests()`` builds what ``deploy/gpu-vm/`` applies; ``TIME_SLICING_CONFIG`` is the
device plugin's own config format (k8s-device-plugin README, v0.20.1), not a Kubernetes object.
"""
from __future__ import annotations

from . import manifests as m

RUNTIME_CLASS = "nvidia"                     # registered by k3s when it finds nvidia-container-runtime
CUDA_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu24.04"
GFD_PRODUCT = "nvidia.com/gpu.product"       # GPU Feature Discovery label (value e.g. NVIDIA-L4; verify on your node)
TIME_SLICES = 4

# The NVIDIA device plugin's config file (sharing.timeSlicing), passed with helm --set-file.
TIME_SLICING_CONFIG = """\
version: v1
sharing:
  timeSlicing:
    resources:
    - name: nvidia.com/gpu
      replicas: {replicas}
"""

# "any node GFD has labelled as a GPU node": the single-box cluster has one model, so pin the
# label's presence rather than a value (in a mixed fleet, pin the value instead).
_GFD_LABELLED = {"nodeAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": {"nodeSelectorTerms": [
    {"matchExpressions": [{"key": GFD_PRODUCT, "operator": "Exists"}]}]}}}


def time_sliced_allocatable(physical_gpus: int, replicas: int) -> int:
    """``nvidia.com/gpu`` the device plugin advertises with time-slicing: ``gpus x replicas``
    (README: 8 GPUs x 10 replicas -> 80). Each unit is a *reference* to a physical GPU, handed out
    with no isolation; ``failRequestsGreaterThanOne`` rejects a container asking for two."""
    return physical_gpus * max(1, replicas)


def _smi_job(name: str, parallelism: int, sleep_s: int) -> dict:
    c = m.GPUContainer(name="smi", image=CUDA_IMAGE, cpu="250m", memory="256Mi",
                       command=["bash", "-c", f"nvidia-smi -L && sleep {sleep_s}"])
    spec = m.pod_spec([c], accelerator=None, runtime_class=RUNTIME_CLASS, affinity=_GFD_LABELLED,
                      termination_grace_s=None)
    return m.job(name, "default", m.pod_template(spec), parallelism=parallelism,
                 active_deadline_s=900, ttl_after_finished_s=600)


def gpuvm_manifests() -> dict[str, tuple[str, list[dict]]]:
    """File name -> (header, objects) for deploy/gpu-vm/."""
    return {
        "10-smoke.yaml": (
            "One GPU through the real device plugin: the kubelet's device manager asks the plugin to\n"
            "Allocate, the NVIDIA runtime (runtimeClassName: nvidia on k3s) mounts the device and driver\n"
            "libraries, and nvidia-smi -L lists the GPU - the step kind cannot show.", [_smi_job("gpu-smoke", 1, 30)]),
        "20-time-sliced.yaml": (
            f"{TIME_SLICES} one-GPU pods at once. Without time-slicing a 1-GPU VM runs one and leaves the\n"
            f"others Pending (Insufficient nvidia.com/gpu); with TIME_SLICING_REPLICAS={TIME_SLICES} the node\n"
            f"advertises {TIME_SLICES} and all run, each seeing the same GPU UUID - shared, not isolated.",
            [_smi_job("time-sliced", TIME_SLICES, 120)]),
    }
