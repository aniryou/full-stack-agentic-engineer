"""The kind lab's objects, scenarios and expected outcomes — one source of truth.

The one idea: every scenario is a sequence of small, deliberate steps (apply one object, wait
for the control plane to settle, look) whose outcome you can predict *before* running it.
``tools/render_kind_manifests.py`` writes the YAML under ``deploy/kind/`` from the builders
here; ``kindsim.predict`` computes what Kueue and the kube-scheduler should do with those
files; ``kindlab.run`` applies them to a real kind cluster and compares. ``EXPECTED`` is the
hand-written answer key, and a test keeps all three in agreement.

Placement is expressed in GCE-style *host* labels (``host-a1-2``), not node names, so the
expectations do not depend on the kind cluster's name.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import manifests as m

CLUSTER_NAME = "gpu-lab"
KUBE_CONTEXT = f"kind-{CLUSTER_NAME}"
NAMESPACES = ["team-a", "team-b", "zoo"]
LOCAL_QUEUE = "gpu-queue"
FLAVOR = "gpu-l4"
TOPOLOGY = "gke-default"
COHORT = "gpu-lab"

# The printout every lab pod starts with: who am I, where did I land, and is there a device?
POD_SCRIPT = """\
echo "pod=$HOSTNAME node=$NODE_NAME gpus=$GPUS rank=${JOB_COMPLETION_INDEX:-${LWS_WORKER_INDEX:-0}}"
ls /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia*: this GPU exists only as node capacity (simulated)"
sleep 3600
"""


def lab_container(gpus: int, name: str = "worker") -> m.GPUContainer:
    return m.GPUContainer(name=name, gpus=gpus, command=["sh", "-c", POD_SCRIPT],
                          env={"GPUS": str(gpus)}, env_from_field={"NODE_NAME": "spec.nodeName"})


def lab_template(gpus: int, *, annotations: dict | None = None, node_selector: dict | None = None,
                 accelerator: str | None = "nvidia-l4", tolerate_gpu: bool = True,
                 tolerations: list[dict] | None = None, shm: bool | None = None,
                 restart_policy: str | None = "Never") -> dict:
    """Pod template for lab pods. Jobs need restartPolicy Never/OnFailure; anything run by a
    StatefulSet (LeaderWorkerSet groups) needs Always, i.e. ``restart_policy=None``.
    Multi-GPU pods and gang members get a memory-backed /dev/shm, as real ones would."""
    if shm is None:
        shm = gpus > 1
    spec = m.pod_spec([lab_container(gpus)], accelerator=accelerator, tolerate_gpu=tolerate_gpu,
                      node_selector=node_selector, tolerations=tolerations, restart_policy=restart_policy,
                      shm_size="256Mi" if shm else None)
    return m.pod_template(spec, annotations=annotations)


# ---- the cluster-side objects the admin applies once -------------------------------------
def kueue_setup() -> dict[str, tuple[str, list[dict]]]:
    """File name -> (header comment, objects) for deploy/kind/manifests/."""
    namespaces = [m.namespace(n) for n in NAMESPACES]
    topo = m.topology(TOPOLOGY, m.GKE_TOPOLOGY_LEVELS)
    flavor = m.resource_flavor(FLAVOR, node_labels={m.GKE_ACCELERATOR: "nvidia-l4"},
                               tolerations=[dict(m.GPU_TOLERATION)], topology_name=TOPOLOGY)
    preemption = {"withinClusterQueue": "LowerPriority", "reclaimWithinCohort": "Any",
                  "borrowWithinCohort": {"policy": "Never"}}
    cqs, lqs = [], []
    for team in ("team-a", "team-b"):
        cqs.append(m.cluster_queue(
            f"{team}-cq", cohort=COHORT, namespaces=[team], preemption=preemption,
            queueing_strategy="BestEffortFIFO",
            flavors={FLAVOR: {"cpu": 8, "memory": "16Gi", m.GPU: m.quota(8, borrowing_limit=4)}}))
        lqs.append(m.local_queue(LOCAL_QUEUE, team, f"{team}-cq"))
    prios = [m.workload_priority_class("low", 100, "preemptible batch work"),
             m.workload_priority_class("high", 1000, "work that may preempt low in the same queue")]
    return {
        "00-namespaces.yaml": ("Namespaces for the two teams and for the Pending-pod zoo.", namespaces),
        "10-kueue-topology-flavor.yaml": (
            "Topology-Aware Scheduling: the GCE-style levels (block > subblock > host > node) that\n"
            "fake-gpus.sh puts on the kind workers, and one TAS ResourceFlavor for the fake L4 nodes.\n"
            "The flavor's toleration is injected into every admitted pod (TAS honours node taints only\n"
            "because kubernetes.io/hostname is the lowest level).", [topo, flavor]),
        "20-kueue-queues.yaml": (
            "Two teams in one cohort. Each owns 8 of the 16 fake GPUs (nominalQuota), may borrow up to\n"
            "4 more from the other when it is idle (borrowingLimit), reclaims its own quota by\n"
            "preempting a borrower (reclaimWithinCohort: Any), and lets high priority preempt low\n"
            "inside the team (withinClusterQueue: LowerPriority). cpu/memory quotas are generous: GPUs\n"
            "are the resource being taught.", cqs + lqs),
        "30-kueue-priorities.yaml": (
            "Workload priorities (Kueue-only; pod priority is unaffected). Jobs pick one with the\n"
            "kueue.x-k8s.io/priority-class label; no label = priority 0.", prios),
    }


def kwok_setup() -> tuple[str, list[dict]]:
    """Queues for the optional KWOK fleet (32 fake 8x H100 nodes)."""
    flavor = m.resource_flavor("gpu-h100-kwok", node_labels={m.GKE_ACCELERATOR: "nvidia-h100-80gb"},
                               tolerations=[dict(m.GPU_TOLERATION),
                                            {"key": "kwok.x-k8s.io/node", "operator": "Exists", "effect": "NoSchedule"}],
                               topology_name=TOPOLOGY)
    cq = m.cluster_queue("fleet-cq", namespaces=["fleet"], queueing_strategy="BestEffortFIFO",
                         flavors={"gpu-h100-kwok": {"cpu": 2048, "memory": "16Ti", m.GPU: 256}})
    return ("KWOK fleet: a flavor for the fake 8-GPU H100 nodes (they also carry the kwok taint),\n"
            "one ClusterQueue with the whole 256-GPU fleet, and its LocalQueue.",
            [m.namespace("fleet"), flavor, cq, m.local_queue("fleet-queue", "fleet", "fleet-cq")])


# ---- scenarios -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    action: str                 # "apply" | "delete" | "scale"
    file: str = ""              # relative to deploy/kind/workloads/
    target: str = ""            # for scale: "kind/namespace/name"
    replicas: int = 0
    note: str = ""


@dataclass
class Scenario:
    id: str
    slug: str
    title: str
    story: str
    steps: list[Step]
    objects: dict[str, tuple[str, dict]] = field(default_factory=dict)   # file -> (header, obj)
    kwok: bool = False

    @property
    def dir(self) -> str:
        return f"{self.id}-{self.slug}"


def _pinned(host: str) -> dict:
    return {m.TOPOLOGY_HOST: host}


def _s1() -> Scenario:
    no_q = m.job("smoke-no-queue", "team-a", lab_template(1))
    kq = m.job("smoke-kueue", "team-a", lab_template(1), queue=LOCAL_QUEUE)
    return Scenario("s1", "one-fake-gpu", "A fake GPU is still a scheduling unit",
                    "One Job straight to the kube-scheduler, one through Kueue. Both consume a fake "
                    "nvidia.com/gpu; the Kueue one is bin-packed next to the first by TAS.",
                    [Step("apply", "s1-one-fake-gpu/10-smoke-no-queue.yaml",
                          note="default scheduler: any GPU node (equal scores, random tie-break)"),
                     Step("apply", "s1-one-fake-gpu/20-smoke-kueue.yaml",
                          note="Kueue TAS, no annotation = unconstrained: least-free node that fits")],
                    {"10-smoke-no-queue.yaml": ("1 GPU, no queue label: Kueue ignores it; the kube-scheduler places it.", no_q),
                     "20-smoke-kueue.yaml": ("1 GPU through Kueue (queue label). Suspended at creation, unsuspended on admission;\n"
                                             "TAS picks the node with the least free capacity that still fits.", kq)})


def _s2() -> Scenario:
    blocker = m.job("blocker", "team-b", lab_template(2, node_selector=_pinned("host-a1-1")))
    host = m.jobset("gang-host", "team-a", [m.replicated_job(
        "workers", lab_template(1, annotations=m.topology_annotations(required=m.TOPOLOGY_HOST), shm=True),
        parallelism=4)], queue=LOCAL_QUEUE)
    sub = m.jobset("gang-subblock", "team-b", [m.replicated_job(
        "workers", lab_template(1, annotations=m.topology_annotations(required=m.TOPOLOGY_SUBBLOCK), shm=True),
        parallelism=8)], queue=LOCAL_QUEUE)
    waits = m.jobset("gang-waits", "team-a", [m.replicated_job(
        "workers", lab_template(1, annotations=m.topology_annotations(required=m.TOPOLOGY_SUBBLOCK), shm=True),
        parallelism=4)], queue=LOCAL_QUEUE)
    d = "s2-gangs-and-topology/"
    return Scenario("s2", "gangs-and-topology", "Gangs land whole, inside one topology domain",
                    "A non-Kueue pod takes 2 GPUs on host-a1-1. A 4-pod gang that must share a host skips it; "
                    "an 8-pod gang that must share a subblock takes all of subblock-a2; a third gang cannot fit "
                    "any subblock and waits with a TAS message until the blocker goes away.",
                    [Step("apply", d + "10-blocker.yaml", note="plain Job pinned to host-a1-1 (2 of 4 GPUs)"),
                     Step("apply", d + "20-gang-host.yaml", note="4 pods, required host -> best-fit full host"),
                     Step("apply", d + "30-gang-subblock.yaml", note="8 pods, required subblock"),
                     Step("apply", d + "40-gang-waits.yaml", note="4 pods, required subblock: only 2 GPUs left in any subblock"),
                     Step("delete", d + "10-blocker.yaml", note="free host-a1-1: Kueue requeues and admits the waiting gang")],
                    {"10-blocker.yaml": ("Not Kueue-managed: 2 GPUs on host-a1-1 (TAS still counts them as used).", blocker),
                     "20-gang-host.yaml": ("A 4-GPU gang that must share one host (think: one NVLink domain).", host),
                     "30-gang-subblock.yaml": ("An 8-GPU gang that must stay inside one subblock (two hosts).", sub),
                     "40-gang-waits.yaml": ("Another 4-GPU subblock gang: no subblock has 4 free GPUs until the blocker ends.", waits)})


def _s3() -> Scenario:
    ann = m.topology_annotations(required=m.TOPOLOGY_SUBBLOCK, group="llm")
    lws = m.leader_worker_set("llm-multihost", "team-b", size=2, queue=LOCAL_QUEUE,
                              leader_template=lab_template(4, annotations=ann, shm=True, restart_policy=None),
                              worker_template=lab_template(4, annotations=ann, shm=True, restart_policy=None))
    return Scenario("s3", "leaderworkerset", "Multi-host inference replicas are admitted a group at a time",
                    "One LeaderWorkerSet replica = a leader and a worker, 4 GPUs each (think: one model "
                    "sharded across two hosts). The group must share a subblock. Scaling to 2 replicas asks "
                    "for 8 more GPUs; team-b can only borrow 4, so the second group waits whole.",
                    [Step("apply", "s3-leaderworkerset/10-llm-multihost.yaml", note="group 0: leader+worker in one subblock"),
                     Step("scale", target="LeaderWorkerSet/team-b/llm-multihost", replicas=2,
                          note="group 1 needs 8 GPUs; team-b has 0 unused + 4 borrowable")],
                    {"10-llm-multihost.yaml": ("LeaderWorkerSet, 1 replica of size 2, 4 GPUs per pod, leader and workers\n"
                                               "co-located in one subblock (podset-group-name ties them together).", lws)})


def _s4() -> Scenario:
    def one(name: str, ns: str, prio: str | None) -> dict:
        return m.job(name, ns, lab_template(4), queue=LOCAL_QUEUE, priority_class=prio)
    d = "s4-priority-preemption/"
    objs = {"10-b-fill-1.yaml": ("team-b fills its own 8 GPUs, so team-a has nothing to borrow.", one("b-fill-1", "team-b", None)),
            "20-b-fill-2.yaml": ("team-b, second 4-GPU job.", one("b-fill-2", "team-b", None)),
            "30-a-low-1.yaml": ("team-a, low priority, 4 GPUs.", one("a-low-1", "team-a", "low")),
            "40-a-low-2.yaml": ("team-a, low priority, 4 GPUs (admitted last: the preemption victim).", one("a-low-2", "team-a", "low")),
            "50-a-high.yaml": ("team-a, high priority: team-a is full and cannot borrow, so it preempts inside its queue.",
                               one("a-high", "team-a", "high"))}
    return Scenario("s4", "priority-preemption", "High priority preempts low — inside its own queue",
                    "Both teams are at their nominal quota. A high-priority team-a job cannot borrow, so "
                    "Kueue evicts the most recently admitted lower-priority team-a workload.",
                    [Step("apply", d + f) for f in objs], objs)


def _s5() -> Scenario:
    def one(name: str, ns: str) -> dict:
        return m.job(name, ns, lab_template(4), queue=LOCAL_QUEUE)
    d = "s5-cohort-borrowing/"
    objs = {"10-a-job-1.yaml": ("team-a within its nominal quota.", one("a-job-1", "team-a")),
            "20-a-job-2.yaml": ("team-a at its nominal quota (8).", one("a-job-2", "team-a")),
            "30-a-job-3.yaml": ("team-a borrows 4 idle GPUs from team-b (borrowingLimit 4).", one("a-job-3", "team-a")),
            "40-a-job-4.yaml": ("team-a at nominal + borrowingLimit: this one waits.", one("a-job-4", "team-a")),
            "50-b-job-1.yaml": ("team-b uses the cohort's last 4 unused GPUs.", one("b-job-1", "team-b")),
            "60-b-job-2.yaml": ("team-b is still within its nominal quota: Kueue reclaims from the borrower.",
                                one("b-job-2", "team-b"))}
    return Scenario("s5", "cohort-borrowing", "Borrowing is a loan: the lender reclaims it",
                    "team-a borrows team-b's idle GPUs, up to its borrowingLimit. When team-b wants its "
                    "nominal quota back, Kueue preempts the borrower's most recently admitted workload.",
                    [Step("apply", d + f) for f in objs], objs)


def _s6() -> Scenario:
    objs: dict[str, tuple[str, dict]] = {}
    for i, host in enumerate(["host-a1-1", "host-a1-2", "host-a2-1", "host-a2-2"], start=1):
        objs[f"1{i}-filler-{host[5:]}.yaml"] = (f"Takes 3 of the 4 GPUs on {host}: every GPU node keeps exactly 1 free.",
                                                m.job(f"filler-{host[5:]}", "zoo", lab_template(3, node_selector=_pinned(host))))
    objs["20-no-toleration.yaml"] = ("Asks for 1 GPU but does not tolerate the nvidia.com/gpu taint.",
                                     m.job("no-toleration", "zoo", lab_template(1, tolerate_gpu=False)))
    objs["30-wrong-accelerator.yaml"] = ("Selects an accelerator this cluster does not have.",
                                         m.job("wrong-accelerator", "zoo", lab_template(1, accelerator="nvidia-h100-80gb")))
    objs["40-too-big.yaml"] = ("5 GPUs in one pod; no node has more than 4.", m.job("too-big", "zoo", lab_template(5)))
    objs["50-fragmented.yaml"] = ("2 GPUs in one pod: 4 GPUs are free cluster-wide, but only 1 per node.",
                                  m.job("fragmented", "zoo", lab_template(2)))
    objs["60-quota-too-big.yaml"] = ("Through Kueue: 4 pods x 4 GPUs = 16 > team-a's maximum (8 nominal + 4 borrowable).",
                                     m.job("quota-too-big", "team-a", lab_template(4), parallelism=4, queue=LOCAL_QUEUE))
    objs["70-topology-too-tight.yaml"] = ("Through Kueue: 2 x 1 GPU that must share a host; every host has 1 free GPU.",
                                          m.job("topology-too-tight", "team-a",
                                                lab_template(1, annotations=m.topology_annotations(required=m.TOPOLOGY_HOST)),
                                                parallelism=2, queue=LOCAL_QUEUE))
    return Scenario("s6", "pending-zoo", "A zoo of Pending pods, one per failure mode",
                    "Fillers leave exactly one free GPU per node; then six workloads that cannot start, each "
                    "for a different reason. Notebook 03 diagnoses them from kubectl output.",
                    [Step("apply", "s6-pending-zoo/" + f) for f in objs], objs)


def _k1() -> Scenario:
    def gang(name: str, pods: int, level: str) -> dict:
        return m.jobset(name, "fleet", [m.replicated_job(
            "workers", lab_template(8, accelerator="nvidia-h100-80gb", shm=True,
                                    annotations=m.topology_annotations(required=level),
                                    tolerations=[{"key": "kwok.x-k8s.io/node", "operator": "Exists", "effect": "NoSchedule"}]),
            parallelism=pods)], queue="fleet-queue")
    sweep = m.job("sweep", "fleet", lab_template(1, accelerator="nvidia-h100-80gb",
                                                 tolerations=[{"key": "kwok.x-k8s.io/node", "operator": "Exists",
                                                               "effect": "NoSchedule"}]),
                  parallelism=8, queue="fleet-queue")
    objs = {"10-pretrain.yaml": ("16 hosts x 8 H100 = 128 GPUs that must share one block.", gang("pretrain", 16, m.TOPOLOGY_BLOCK)),
            "20-finetune.yaml": ("4 hosts x 8 GPUs that must share one subblock.", gang("finetune", 4, m.TOPOLOGY_SUBBLOCK)),
            "30-sweep.yaml": ("8 single-GPU pods, no topology request: TAS packs them onto one host.", sweep)}
    return Scenario("k1", "kwok-fleet", "Placement at fleet scale with KWOK (optional)",
                    "32 fake 8-GPU nodes in 2 blocks x 4 subblocks x 4 hosts. Fake pods stay Running "
                    "(kwok.sh removes the pod-complete stage) so the placement stays visible.",
                    [Step("apply", "k1-kwok-fleet/" + f) for f in objs], objs, kwok=True)


SCENARIOS: dict[str, Scenario] = {s.id: s for s in (_s1(), _s2(), _s3(), _s4(), _s5(), _s6(), _k1())}


def scenario(key: str) -> Scenario:
    """Look up by id ('s2') or directory name ('s2-gangs-and-topology')."""
    for s in SCENARIOS.values():
        if key in (s.id, s.dir, s.slug):
            return s
    raise KeyError(f"unknown scenario {key!r}; one of {', '.join(SCENARIOS)}")


# ---- the answer key -----------------------------------------------------------------------------
# After the last step of each scenario (and, for s2/s3, after an intermediate step), per object:
#   state:  running | unschedulable | admitted | pending | preempted
#   hosts:  {host-label: pods}, "any" (scheduler's choice), or "same-as:<key>"
#   says:   a substring the Kueue condition / FailedScheduling message must contain
FS_TAINT_CP = "1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: }"
EXPECTED: dict[str, dict[int, dict[str, dict]]] = {
    "s1": {2: {"Job/team-a/smoke-no-queue": {"state": "running", "hosts": "any"},
               "Job/team-a/smoke-kueue": {"state": "admitted", "hosts": "same-as:Job/team-a/smoke-no-queue"}}},
    "s2": {4: {"Job/team-b/blocker": {"state": "running", "hosts": {"host-a1-1": 1}},
               "JobSet/team-a/gang-host": {"state": "admitted", "hosts": {"host-a1-2": 4}},
               "JobSet/team-b/gang-subblock": {"state": "admitted", "hosts": {"host-a2-1": 4, "host-a2-2": 4}},
               "JobSet/team-a/gang-waits": {"state": "pending",
                                            "says": 'topology "gke-default" allows to fit only 2 out of 4 pod(s)'}},
           5: {"JobSet/team-a/gang-host": {"state": "admitted", "hosts": {"host-a1-2": 4}},
               "JobSet/team-b/gang-subblock": {"state": "admitted", "hosts": {"host-a2-1": 4, "host-a2-2": 4}},
               "JobSet/team-a/gang-waits": {"state": "admitted", "hosts": {"host-a1-1": 4}}}},
    "s3": {1: {"LeaderWorkerSet/team-b/llm-multihost#0": {"state": "admitted", "hosts": {"host-a1-1": 1, "host-a1-2": 1}}},
           2: {"LeaderWorkerSet/team-b/llm-multihost#0": {"state": "admitted", "hosts": {"host-a1-1": 1, "host-a1-2": 1}},
               "LeaderWorkerSet/team-b/llm-multihost#1": {"state": "pending",
                                                          "says": "insufficient unused quota for nvidia.com/gpu in flavor gpu-l4, 4 more needed"}}},
    "s4": {5: {"Job/team-b/b-fill-1": {"state": "admitted", "hosts": {"host-a1-1": 1}},
               "Job/team-b/b-fill-2": {"state": "admitted", "hosts": {"host-a1-2": 1}},
               "Job/team-a/a-low-1": {"state": "admitted", "hosts": {"host-a2-1": 1}},
               "Job/team-a/a-low-2": {"state": "preempted", "says": "InClusterQueue"},
               "Job/team-a/a-high": {"state": "admitted", "hosts": {"host-a2-2": 1}}}},
    "s5": {6: {"Job/team-a/a-job-1": {"state": "admitted", "hosts": {"host-a1-1": 1}},
               "Job/team-a/a-job-2": {"state": "admitted", "hosts": {"host-a1-2": 1}},
               "Job/team-a/a-job-3": {"state": "preempted", "says": "InCohortReclamation"},
               "Job/team-a/a-job-4": {"state": "pending", "says": "insufficient unused quota for nvidia.com/gpu in flavor gpu-l4, 4 more needed"},
               "Job/team-b/b-job-1": {"state": "admitted", "hosts": {"host-a2-2": 1}},
               "Job/team-b/b-job-2": {"state": "admitted", "hosts": {"host-a2-1": 1}}}},
    "s6": {10: {"Job/zoo/filler-a1-1": {"state": "running", "hosts": {"host-a1-1": 1}},
                "Job/zoo/no-toleration": {"state": "unschedulable", "says":
                    "0/6 nodes are available: 1 node(s) didn't match Pod's node affinity/selector, " + FS_TAINT_CP +
                    ", 4 node(s) had untolerated taint {nvidia.com/gpu: present}. preemption: 0/6 nodes are available: "
                    "6 Preemption is not helpful for scheduling."},
                "Job/zoo/wrong-accelerator": {"state": "unschedulable", "says":
                    "0/6 nodes are available: " + FS_TAINT_CP + ", 5 node(s) didn't match Pod's node affinity/selector. "
                    "preemption: 0/6 nodes are available: 6 Preemption is not helpful for scheduling."},
                "Job/zoo/too-big": {"state": "unschedulable", "says":
                    "0/6 nodes are available: 1 node(s) didn't match Pod's node affinity/selector, " + FS_TAINT_CP +
                    ", 4 Insufficient nvidia.com/gpu. preemption: 0/6 nodes are available: 6 Preemption is not helpful for scheduling."},
                "Job/zoo/fragmented": {"state": "unschedulable", "says":
                    "0/6 nodes are available: 1 node(s) didn't match Pod's node affinity/selector, " + FS_TAINT_CP +
                    ", 4 Insufficient nvidia.com/gpu. preemption: 0/6 nodes are available: 2 Preemption is not helpful "
                    "for scheduling, 4 No preemption victims found for incoming pod."},
                "Job/team-a/quota-too-big": {"state": "pending", "says":
                    "insufficient quota for nvidia.com/gpu in flavor gpu-l4, previously considered podsets requests (0) "
                    "+ current podset request (16) > maximum capacity (12)"},
                "Job/team-a/topology-too-tight": {"state": "pending",
                                                  "says": 'topology "gke-default" allows to fit only 1 out of 2 pod(s)'}}},
    "k1": {3: {"JobSet/fleet/pretrain": {"state": "admitted", "hosts": {f"kwok-b1-s{s}-h{h}": 1 for s in range(1, 5) for h in range(1, 5)}},
               "JobSet/fleet/finetune": {"state": "admitted", "hosts": {f"kwok-b2-s1-h{h}": 1 for h in range(1, 5)}},
               "Job/fleet/sweep": {"state": "admitted", "hosts": {"kwok-b2-s2-h1": 8}}}},
}
