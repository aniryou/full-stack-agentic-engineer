"""The same ideas on one cloud: GKE node pools, DWS flex-start via Kueue, ComputeClasses, GCS FUSE.

The one idea: on a managed cluster the scheduling objects barely change — a GPU pod still
asks for ``nvidia.com/gpu``, tolerates the GPU taint and selects an accelerator — but *capacity*
becomes something you ask for: a node pool that autoscales from zero, a ComputeClass that
falls back from Spot to on-demand to flex-start, or a ProvisioningRequest that Kueue files
before it admits a gang. ``gke_manifests()`` builds the objects rendered into ``deploy/gke/``;
the helpers read ``deploy/gcp/terraform`` offline so notebook 04 can plan without a project.

GKE-specific names (node-pool labels, the queued-provisioning taint and class, the
provisioning-request node label, ComputeClass fields) are marked (verify).
"""
from __future__ import annotations

import re
from pathlib import Path

from . import capacity
from . import manifests as m

LAB_ROOT = Path(__file__).resolve().parents[1]
TF_DIR = LAB_ROOT / "deploy" / "gcp" / "terraform"

SYSTEM_POOL, SPOT_POOL, FLEX_POOL = "system", "l4-spot", "l4-flex"
CUDA_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu24.04"      # nvidia-smi comes from the host driver mount
VLLM_IMAGE = "vllm/vllm-openai:v0.30.0"                  # (verify the CUDA build against the node driver)
QUEUED_TAINT_KEY = "cloud.google.com/gke-queued"         # (verify) taint on queued-provisioning nodes
PROVREQ_NODE_LABEL = "autoscaling.gke.io/provisioning-request"   # (verify)
COMPUTE_CLASS = "l4-spot-first"


def _pool(name: str) -> dict:
    return {m.GKE_NODEPOOL: name}


def gke_manifests() -> dict[str, tuple[str, list[dict]]]:
    """File name -> (header, objects) for deploy/gke/."""
    # 00: prove the Spot L4 pool scales from zero and the driver works
    smoke = m.job("nvidia-smi", "default", m.pod_template(m.pod_spec(
        [m.GPUContainer(name="smi", image=CUDA_IMAGE, command=["bash", "-c", "nvidia-smi && sleep 30"],
                        cpu="500m", memory="512Mi")],
        accelerator="nvidia-l4", node_selector=_pool(SPOT_POOL), termination_grace_s=None)),
        active_deadline_s=1800, ttl_after_finished_s=600)

    # 10: Kueue on GKE — Spot first, then DWS flex-start behind a ProvisioningRequest admission check
    ml = m.namespace("ml")
    spot = m.resource_flavor(SPOT_POOL, node_labels=_pool(SPOT_POOL), tolerations=[dict(m.GPU_TOLERATION)])
    flex = m.resource_flavor(FLEX_POOL, node_labels=_pool(FLEX_POOL), tolerations=[
        dict(m.GPU_TOLERATION), {"key": QUEUED_TAINT_KEY, "operator": "Exists", "effect": "NoSchedule"}])
    prc = m.provisioning_request_config("dws-config", provisioning_class="queued-provisioning.gke.io",
                                        node_selector_updates={PROVREQ_NODE_LABEL: "ResizeRequestName"})
    ac = m.admission_check("dws-prov", config_name="dws-config")
    cq = m.cluster_queue("gke-gpu", namespaces=["ml"], queueing_strategy="BestEffortFIFO",
                         preemption={"withinClusterQueue": "LowerPriority"},
                         flavors={SPOT_POOL: {"cpu": 6, "memory": "24Gi", m.GPU: 2},
                                  FLEX_POOL: {"cpu": 16, "memory": "64Gi", m.GPU: 4}},
                         admission_checks=[("dws-prov", [FLEX_POOL])])
    lq = m.local_queue("gpu-queue", "ml", "gke-gpu")

    # 20: a 2-node gang that goes through DWS: all-or-nothing, bounded run time
    dws_job = m.job("dws-l4-gang", "ml", m.pod_template(m.pod_spec(
        [m.GPUContainer(name="smi", image=CUDA_IMAGE, command=["bash", "-c", "nvidia-smi -L && sleep 300"],
                        cpu="500m", memory="1Gi")],
        accelerator="nvidia-l4", node_selector=_pool(FLEX_POOL), termination_grace_s=None)),
        parallelism=2, queue="gpu-queue",
        annotations={m.PROVREQ_PREFIX + "maxRunDurationSeconds": "3600"})

    # 30: capacity as code — Spot, then on-demand, then flex-start, for the same L4 shape
    cc = m.compute_class(COMPUTE_CLASS, [
        m.compute_class_priority(machine_type="g2-standard-4", gpu_type="nvidia-l4", gpu_count=1, spot=True),
        m.compute_class_priority(machine_type="g2-standard-4", gpu_type="nvidia-l4", gpu_count=1, spot=False),
        m.compute_class_priority(machine_type="g2-standard-4", gpu_type="nvidia-l4", gpu_count=1,
                                 flex_start=True, node_recycling_lead_s=3600),
    ])

    # 40: a vLLM server that reads weights from GCS through the FUSE CSI driver
    serving = m.namespace("serving")
    ksa = m.obj("ServiceAccount", "model-reader", "serving")
    health = "/health"
    vllm = m.GPUContainer(
        name="vllm", image=VLLM_IMAGE, command=["vllm", "serve"],
        args=["/models/qwen2.5-1.5b-instruct", "--served-model-name", "qwen2.5-1.5b",
              "--max-model-len", "4096", "--gpu-memory-utilization", "0.90", "--port", "8000"],
        gpus=1, cpu="2", memory="10Gi", ports=[8000],
        startup_probe=m.http_probe(health, 8000, period_s=10, failure_threshold=60),
        readiness_probe=m.http_probe(health, 8000, period_s=5, failure_threshold=3),
        liveness_probe=m.http_probe(health, 8000, period_s=15, failure_threshold=3),
        volume_mounts=[{"name": "weights", "mountPath": "/models", "readOnly": True}])
    weights = {"name": "weights", "csi": {"driver": "gcsfuse.csi.storage.gke.io", "readOnly": True,
                                          "volumeAttributes": {
                                              "bucketName": "${WEIGHTS_BUCKET}",
                                              "mountOptions": "implicit-dirs",
                                              "fileCacheCapacity": "-1",
                                              "fileCacheForRangeRead": "true",
                                              "metadataStatCacheCapacity": "-1",
                                              "metadataTypeCacheCapacity": "-1",
                                              "metadataCacheTTLSeconds": "-1"}}}
    pod = m.pod_template(
        m.pod_spec([vllm], accelerator=None, node_selector={m.GKE_COMPUTE_CLASS: COMPUTE_CLASS},
                   restart_policy=None, termination_grace_s=60, service_account="model-reader",
                   volumes=[weights], shm_size="1Gi"),
        labels={"app": "vllm-l4"},
        annotations={"gke-gcsfuse/volumes": "true", "gke-gcsfuse/cpu-request": "500m",
                     "gke-gcsfuse/memory-request": "1Gi", "gke-gcsfuse/ephemeral-storage-request": "20Gi"})
    deploy = m.obj("Deployment", "vllm-l4", "serving", labels={"app": "vllm-l4"}, spec={
        "replicas": 1,
        "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1}},
        "selector": {"matchLabels": {"app": "vllm-l4"}},
        "template": pod})
    svc = m.obj("Service", "vllm-l4", "serving", spec={
        "selector": {"app": "vllm-l4"}, "ports": [{"name": "http", "port": 8000, "targetPort": 8000}]})

    return {
        "00-smoke-l4.yaml": (
            "Smoke test for the Spot L4 pool: the pod forces a scale-up from zero, runs nvidia-smi\n"
            "(the driver GKE installed is mounted into the container) and exits. The Job gives up after\n"
            "30 minutes so a Spot stockout cannot leave it waiting all weekend.", [smoke]),
        "10-kueue-gke.yaml": (
            "Kueue on GKE: two flavors backed by two node pools. l4-spot is plain quota; l4-flex sits\n"
            "behind the dws-prov AdmissionCheck, so Kueue files a ProvisioningRequest\n"
            "(queued-provisioning.gke.io) and admits only when DWS has provisioned every node.\n"
            "Requires Kueue installed (deploy/gke/install-addons.sh) and enable_flex_start_pool = true.",
            [ml, spot, flex, prc, ac, cq, lq]),
        "20-dws-sample-job.yaml": (
            "A 2-pod gang pinned to the flex-start pool, so Kueue picks the l4-flex flavor and the\n"
            "ProvisioningRequest path. maxRunDurationSeconds bounds the lease DWS grants.", [dws_job]),
        "30-computeclass-l4.yaml": (
            "A custom ComputeClass: GKE creates node pools on demand and tries the rungs in order -\n"
            "Spot, then on-demand, then flex-start - for the same L4 shape. Pods opt in with a\n"
            "nodeSelector on cloud.google.com/compute-class. Field names: verify against\n"
            "`kubectl explain computeclass.spec` on your cluster. For latency-critical serving put\n"
            "on-demand (or a reservation) first and Spot last.", [cc]),
        "40-serving-vllm-gcsfuse.yaml": (
            "vLLM on one L4 via the ComputeClass, weights mounted read-only from GCS by the Cloud\n"
            "Storage FUSE CSI driver (the pod's KSA gets objectViewer through Workload Identity; see\n"
            "deploy/gcp/terraform/storage.tf). apply-examples.sh substitutes ${WEIGHTS_BUCKET}.\n"
            "The startup probe allows 10 minutes for the weight load before liveness takes over.",
            [serving, ksa, deploy, svc]),
    }


# ---- reading the Terraform offline ------------------------------------------------------------------
_ASSIGN = re.compile(r'^\s*([a-z_][a-z0-9_]*)\s*=\s*(.+?)\s*(#.*)?$')


def read_tfvars(path: str | Path | None = None) -> dict[str, object]:
    """Scalar ``key = value`` lines of a .tfvars file (strings, numbers, bools; maps/lists skipped)."""
    path = Path(path) if path else TF_DIR / "terraform.tfvars.example"
    out: dict[str, object] = {}
    for line in path.read_text().splitlines():
        mt = _ASSIGN.match(line)
        if not mt or line.lstrip().startswith("#"):
            continue
        key, raw = mt.group(1), mt.group(2).strip()
        if raw.startswith('"') and raw.endswith('"'):
            out[key] = raw[1:-1]
        elif raw in ("true", "false"):
            out[key] = raw == "true"
        elif re.fullmatch(r"-?\d+(\.\d+)?", raw):
            out[key] = float(raw) if "." in raw else int(raw)
    return out


def variable_defaults(tf_dir: str | Path | None = None) -> dict[str, object]:
    """Scalar defaults declared in variables.tf."""
    text = (Path(tf_dir) if tf_dir else TF_DIR).joinpath("variables.tf").read_text()
    out: dict[str, object] = {}
    for block in re.finditer(r'variable\s+"([a-z0-9_]+)"\s*{(.*?)\n}', text, re.S):
        dm = re.search(r'^\s*default\s*=\s*(.+?)\s*$', block.group(2), re.M)
        if not dm:
            continue
        raw = dm.group(1)
        if raw.startswith('"'):
            out[block.group(1)] = raw.strip('"')
        elif raw in ("true", "false"):
            out[block.group(1)] = raw == "true"
        elif re.fullmatch(r"-?\d+(\.\d+)?", raw):
            out[block.group(1)] = float(raw) if "." in raw else int(raw)
    return out


def tf_resources(tf_dir: str | Path | None = None) -> list[tuple[str, str, str]]:
    """(file, resource type, name) for every resource block — an offline `terraform state list`."""
    d = Path(tf_dir) if tf_dir else TF_DIR
    out = []
    for f in sorted(d.glob("*.tf")):
        for mt in re.finditer(r'^resource\s+"([a-z0-9_]+)"\s+"([a-z0-9_]+)"', f.read_text(), re.M):
            out.append((f.name, mt.group(1), mt.group(2)))
    return out


def effective_vars(tfvars: dict | None = None) -> dict[str, object]:
    v = variable_defaults()
    v.update(tfvars if tfvars is not None else read_tfvars())
    return v


def plan_summary(v: dict[str, object]) -> str:
    """What the Terraform would build and what it costs per hour, idle vs one GPU busy."""
    sys_mt, sys_n = str(v["system_machine_type"]), int(v["system_node_count"])
    gpu_mt, gpu_max = str(v["gpu_machine_type"]), int(v["gpu_max_nodes"])
    spot = bool(v["gpu_spot"])
    sys_cost = capacity.ON_DEMAND_USD_H.get(sys_mt, 0.0) * sys_n
    mult = capacity.OPTIONS["spot" if spot else "on-demand"].price_multiplier
    gpu_node = capacity.ON_DEMAND_USD_H.get(gpu_mt, 0.0) * mult
    lines = [f"cluster  {v['cluster_name']} (zonal, {v['zone']}), release channel {v['release_channel']}",
             f"pool     {SYSTEM_POOL:8} {sys_n} x {sys_mt} (on-demand)             ~${sys_cost:.2f}/h always",
             f"pool     {SPOT_POOL:8} 0..{gpu_max} x {gpu_mt} + {v['gpu_count']} x {v['gpu_type']} "
             f"({'Spot' if spot else 'on-demand'}, driver {v['gpu_driver_version']})  ~${gpu_node:.2f}/h per node while busy"]
    if v.get("enable_flex_start_pool"):
        flex = capacity.ON_DEMAND_USD_H.get(str(v["flex_machine_type"]), 0.0) * capacity.OPTIONS["flex-start"].price_multiplier
        lines.append(f"pool     {FLEX_POOL:8} 0..{v['flex_max_nodes']} x {v['flex_machine_type']} "
                     f"(DWS flex-start, queued provisioning)  ~${flex:.2f}/h per node while provisioned")
    idle = sys_cost + capacity.GKE_CLUSTER_FEE_USD_H
    lines.append(f"idle     ~${idle:.2f}/h (system pool + cluster fee before free-tier credit); "
                 f"one busy GPU node adds ~${gpu_node:.2f}/h   (all prices: verify)")
    return "\n".join(lines)


def gcloud_equivalents(v: dict[str, object]) -> list[str]:
    """The imperative version of the Terraform — handy for reading GKE docs side by side."""
    c, z = v["cluster_name"], v["zone"]
    cmds = [
        f"gcloud container clusters create {c} --location={z} --release-channel={str(v['release_channel']).lower()} "
        f"--workload-pool={v.get('project_id', 'PROJECT_ID')}.svc.id.goog --addons=GcsFuseCsiDriver "
        f"--enable-managed-prometheus --image-type=COS_CONTAINERD --enable-image-streaming "
        f"--num-nodes={v['system_node_count']} --machine-type={v['system_machine_type']}",
        f"gcloud container node-pools create {SPOT_POOL} --cluster={c} --location={z} "
        f"--machine-type={v['gpu_machine_type']} "
        f"--accelerator=type={v['gpu_type']},count={v['gpu_count']},gpu-driver-version={str(v['gpu_driver_version']).lower()} "
        f"{'--spot ' if v['gpu_spot'] else ''}--enable-autoscaling --min-nodes=0 --max-nodes={v['gpu_max_nodes']} --num-nodes=0 "
        f"--node-taints=nvidia.com/gpu=present:NoSchedule",
    ]
    if v.get("enable_flex_start_pool"):
        cmds.append(
            f"gcloud container node-pools create {FLEX_POOL} --cluster={c} --location={z} "
            f"--machine-type={v['flex_machine_type']} "
            f"--accelerator=type={v['gpu_type']},count={v['gpu_count']},gpu-driver-version={str(v['gpu_driver_version']).lower()} "
            f"--flex-start --enable-queued-provisioning --enable-autoscaling --num-nodes=0 "
            f"--total-max-nodes={v['flex_max_nodes']} --location-policy=ANY --reservation-affinity=none "
            f"--no-enable-autorepair")
    return cmds
