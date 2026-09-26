"""How a GPU becomes schedulable: the device plugin <-> kubelet contract (deviceplugin v1beta1).

The one idea: Kubernetes itself knows nothing about GPUs. A node-local device plugin registers a
resource name with the kubelet, streams device IDs and their health (ListAndWatch), and when a
container starts tells the kubelet how to expose the devices chosen for it (Allocate). The
kubelet publishes capacity (all devices) and allocatable (healthy devices) in the node status;
the scheduler only ever sees those two integers. Device IDs below are fake.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cluster import GPU

API_VERSION = "v1beta1"
KUBELET_SOCKET = "/var/lib/kubelet/device-plugins/kubelet.sock"
HEALTHY, UNHEALTHY = "Healthy", "Unhealthy"


@dataclass
class Device:
    id: str
    health: str = HEALTHY
    numa: int = 0
    island: int = 0              # NVLink-connected group on this node (simulated)


def make_gpus(n: int = 8, island_size: int | None = None, numa_nodes: int = 2) -> list:
    return [Device(f"GPU-fake-{i:04d}", numa=i * numa_nodes // n, island=i // (island_size or n))
            for i in range(n)]


class AdmissionError(RuntimeError):
    """The pod fails with reason UnexpectedAdmissionError: the kubelet could not allocate devices."""


class DevicePlugin:
    """The node-local plugin. replicas > 1 models time-slicing: each GPU is advertised N times, and
    the replicas are handed out without regard to which physical GPU they belong to.
    fail_requests_greater_than_one mirrors the NVIDIA plugin's `failRequestsGreaterThanOne`."""

    def __init__(self, devices: list, resource: str = GPU, replicas: int = 1,
                 fail_requests_greater_than_one: bool = False):
        self.devices, self.resource, self.replicas = devices, resource, replicas
        self.fail_requests_greater_than_one = fail_requests_greater_than_one

    def register_request(self) -> dict:
        """What the plugin sends to the kubelet's Registration service on KUBELET_SOCKET."""
        return {"version": API_VERSION, "endpoint": "nvidia-gpu.sock", "resource_name": self.resource}

    def list_and_watch(self) -> list:
        """One message of the ListAndWatch stream (re-sent whenever a device changes health)."""
        suffix = (lambda r: f"::{r}") if self.replicas > 1 else (lambda r: "")
        return [{"ID": d.id + suffix(r), "health": d.health} for d in self.devices for r in range(self.replicas)]

    def set_health(self, device_id: str, health: str) -> None:
        next(d for d in self.devices if d.id == device_id).health = health

    def get_preferred_allocation(self, available: list, must_include: list, size: int) -> list:
        """Keep a multi-GPU container on one NVLink island if possible, else one NUMA node."""
        island = {d.id: d.island for d in self.devices}
        numa = {d.id: d.numa for d in self.devices}
        chosen = list(must_include)
        rest = [a for a in available if a not in chosen]
        for group_of in (island, numa):
            groups = {}
            for a in rest:
                groups.setdefault(group_of[a.split("::")[0]], []).append(a)
            fitting = [g for g in groups.values() if len(g) >= size - len(chosen)]
            if fitting:                                  # the tightest group that fits
                return chosen + min(fitting, key=len)[: size - len(chosen)]
        return chosen + rest[: size - len(chosen)]

    def allocate(self, ids: list) -> dict:
        """ContainerAllocateResponse. With the NVIDIA plugin's default `envvar` strategy it is one
        environment variable; the NVIDIA Container Toolkit then injects device nodes and driver
        libraries when the container is created (see layer 02, section 6). Raises ValueError, as the
        plugin's gRPC error, when a shared (time-sliced) request asks for more than one replica and
        fail_requests_greater_than_one is set."""
        if self.replicas > 1 and self.fail_requests_greater_than_one and len(ids) > 1:
            raise ValueError(f"request for '{self.resource}: {len(ids)}' too large: "
                             "maximum request size for shared resources is 1")
        uuids = sorted({i.split("::")[0] for i in ids})
        return {"envs": {"NVIDIA_VISIBLE_DEVICES": ",".join(uuids)}, "mounts": [], "devices": []}


class Kubelet:
    """The kubelet's device manager: plugin stream -> node status, and devices pinned per container."""

    def __init__(self):
        self.plugin: DevicePlugin | None = None
        self.assigned: dict[str, list] = {}

    def register(self, plugin: DevicePlugin) -> None:
        req = plugin.register_request()
        if req["version"] != API_VERSION or "/" not in req["resource_name"]:
            raise ValueError(f"registration rejected: {req}")
        self.plugin = plugin

    def node_status(self) -> dict:
        devs = self.plugin.list_and_watch() if self.plugin else []
        healthy = sum(d["health"] == HEALTHY for d in devs)
        res = self.plugin.resource if self.plugin else GPU
        return {"capacity": {res: len(devs)}, "allocatable": {res: healthy}}

    def admit(self, pod: str, count: int) -> dict:
        """Pick free healthy devices (asking the plugin's preference), then call Allocate."""
        taken = {i for ids in self.assigned.values() for i in ids}
        free = [d["ID"] for d in self.plugin.list_and_watch() if d["health"] == HEALTHY and d["ID"] not in taken]
        if count > len(free):
            raise AdmissionError(f"Allocate failed due to requested number of devices unavailable for "
                                 f"{self.plugin.resource}. Requested: {count}, Available: {len(free)}, which is unexpected")
        ids = self.plugin.get_preferred_allocation(free, [], count)
        try:
            response = self.plugin.allocate(ids)
        except ValueError as e:                          # the plugin's Allocate returned an error
            raise AdmissionError(f"Allocate failed due to rpc error: code = Unknown desc = {e}, "
                                 "which is unexpected") from None
        self.assigned[pod] = ids
        return response
