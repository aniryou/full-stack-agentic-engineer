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


# How the NVIDIA plugin picks time-sliced replicas (`--shared-devices-allocation-policy`, env
# SHARED_DEVICES_ALLOCATION_POLICY): "distributed" (the default) and "packed" since v0.20.0; "spread"
# is on the main branch, not in a release as of 2026-09-26 (verify).
ALLOCATION_POLICIES = ("distributed", "packed", "spread")


class DevicePlugin:
    """The node-local plugin. replicas > 1 models time-slicing: each GPU is advertised N times, and
    `allocation_policy` picks the replicas one at a time, as the NVIDIA plugin does:

    distributed  (default) the GPU with the fewest replicas allocated - so a request for several can
                 still land on one GPU when that GPU is the least loaded
    packed       the GPU with the most replicas allocated: fills GPUs one by one, and puts a
                 multi-replica request on one GPU whenever it has room
    spread       the GPU this request has touched least, then the fewest allocated: a request for
                 several gets distinct physical GPUs while there are enough

    fail_requests_greater_than_one mirrors the NVIDIA plugin's `failRequestsGreaterThanOne`."""

    def __init__(self, devices: list, resource: str = GPU, replicas: int = 1,
                 fail_requests_greater_than_one: bool = False, allocation_policy: str = "distributed"):
        if allocation_policy not in ALLOCATION_POLICIES:
            raise ValueError(f"invalid --shared-devices-allocation-policy option: {allocation_policy}")
        self.devices, self.resource, self.replicas = devices, resource, replicas
        self.fail_requests_greater_than_one = fail_requests_greater_than_one
        self.allocation_policy = allocation_policy

    def get_device_plugin_options(self) -> dict:
        """DevicePluginOptions, as the NVIDIA plugin answers: it implements GetPreferredAllocation and
        needs no PreStartContainer call. The kubelet uses the copy sent with Register."""
        return {"pre_start_required": False, "get_preferred_allocation_available": True}

    def register_request(self) -> dict:
        """What the plugin sends to the kubelet's Registration service on KUBELET_SOCKET."""
        return {"version": API_VERSION, "endpoint": "nvidia-gpu.sock", "resource_name": self.resource,
                "options": self.get_device_plugin_options()}

    def pre_start_container(self, ids: list) -> dict:
        """PreStartContainer: called before each container start only if pre_start_required (a plugin
        that must reset or initialise a device). The NVIDIA plugin returns an empty response."""
        return {}

    def list_and_watch(self) -> list:
        """One message of the ListAndWatch stream (re-sent whenever a device changes health)."""
        suffix = (lambda r: f"::{r}") if self.replicas > 1 else (lambda r: "")
        return [{"ID": d.id + suffix(r), "health": d.health} for d in self.devices for r in range(self.replicas)]

    def set_health(self, device_id: str, health: str) -> None:
        next(d for d in self.devices if d.id == device_id).health = health

    def get_preferred_allocation(self, available: list, must_include: list, size: int) -> list:
        """Keep a multi-GPU container on one NVLink island if possible, else one NUMA node.
        Time-sliced replicas instead go one at a time to the GPU the allocation policy prefers
        (see the class docstring; ties here by listing order, which upstream leaves to its heap)."""
        if self.replicas > 1:
            chosen, rest = list(must_include), [a for a in available if a not in must_include]
            allocated = {d.id: self.replicas for d in self.devices}
            for a in rest:
                allocated[a.split("::")[0]] -= 1
            touched = {d.id: 0 for d in self.devices}          # replicas of each GPU in this request
            for a in chosen:
                touched[a.split("::")[0]] += 1
            key = {"distributed": lambda g: allocated[g], "packed": lambda g: -allocated[g],
                   "spread": lambda g: (touched[g], allocated[g])}[self.allocation_policy]
            while len(chosen) < size and rest:
                pick = min(rest, key=lambda a: key(a.split("::")[0]))
                allocated[pick.split("::")[0]] += 1
                touched[pick.split("::")[0]] += 1
                rest.remove(pick)
                chosen.append(pick)
            return chosen
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
