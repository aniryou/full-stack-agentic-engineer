"""gpusched - how Kubernetes turns GPUs into schedulable integers, and places, queues and scales them.

Pure standard library, deterministic, no cluster needed. Read the modules in this order:

    deviceplugin.py   a GPU becomes an integer: device plugin -> kubelet -> node allocatable
    cluster.py        nodes, pods, taints, tolerations, topology paths; the API's resource rules
    plugins.py        filters (can it run here?), scores (where is best?), preemption victims
    scheduler.py      the one-pod-at-a-time cycle, FailedScheduling messages, fragmentation
    gang.py           all-or-nothing placement and topology-aware domain choice (Kueue TAS)
    quota.py          Kueue-style ClusterQueues, cohorts, borrowing/lending, reclaim preemption
    autoscaler.py     node pools from zero, provisioning delay, Spot, queued provisioning, scale-down
"""
from .autoscaler import (Job, NodePool, expected_runtime_h, gang_survival, least_waste, nodes_needed,
                         provision, simulate, startup_latency)
from .cluster import (GPU, GPU_TAINT, Cluster, Node, Pod, Taint, Toleration, effective_requests,
                      extended_resource_toleration, gpu_node, gpu_pod, make_cluster)
from .deviceplugin import AdmissionError, Device, DevicePlugin, Kubelet, make_gpus
from .gang import admit_gangs, best_fit, bind_gang, interleave, least_free_capacity, place_gang, running
from .plugins import least_allocated, most_allocated, pick_preemption_node, run_filters, select_victims
from .quota import ClusterQueue, Kueue, Quota, Workload
from .scheduler import Decision, Scheduler, fit_error, fragmentation, stranded_gpus
