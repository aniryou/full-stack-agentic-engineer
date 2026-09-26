"""gke.py — the GKE Sandbox deployment, read offline: what the Terraform creates and what it costs.

One idea: on GKE the ladder's top rungs are *node-pool settings*, not code — a node pool with
``sandbox_config { type = "GVISOR" }`` runs every pod that names RuntimeClass ``gvisor`` under
gVisor, on nodes that nothing else lands on (PRIMER §5, §9 "Where to run it"). This module reads
``deploy/gcp/terraform/*.tf`` as text (no Terraform needed) to answer the review questions — is
the pool sandboxed, Spot, scaling from zero, tainted; is there any route to the internet; is
NetworkPolicy enforced — and prints the ``gcloud`` equivalents. Prices are inputs you look up
(``COMPUTE.md``); every default below is marked (verify).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TF_DIR = Path(__file__).resolve().parents[1] / "deploy" / "gcp" / "terraform"


def tf_text(tf_dir: Path = TF_DIR) -> str:
    return "\n".join(p.read_text() for p in sorted(tf_dir.glob("*.tf")))


def variable_default(name: str, tf_dir: Path = TF_DIR) -> str | None:
    m = re.search(rf'variable "{name}"\s*{{(.*?)\n}}', tf_text(tf_dir), re.S)
    if not m:
        return None
    d = re.search(r"^\s*default\s*=\s*(.+)$", m.group(1), re.M)
    return d.group(1).strip().strip('"') if d else None


@dataclass
class Check:
    name: str
    ok: bool
    evidence: str


def review(tf_dir: Path = TF_DIR) -> list[Check]:
    """The design-review checklist for the sandbox cluster, answered from the .tf files."""
    t = tf_text(tf_dir)
    has = lambda pat: re.search(pat, t, re.S) is not None   # noqa: E731
    return [
        Check("sandbox pool runs gVisor", has(r'sandbox_config\s*{\s*type\s*=\s*"GVISOR"'),
              'node_config.sandbox_config.type = "GVISOR" (the provider validates the upper-case value)'),
        Check("COS with containerd image", has(r'image_type\s*=\s*"COS_CONTAINERD"'), "required by GKE Sandbox"),
        Check("sandbox pool on Spot", has(r"spot\s*=\s*var\.sandbox_spot") and variable_default("sandbox_spot", tf_dir) == "true",
              f"sandbox_spot default = {variable_default('sandbox_spot', tf_dir)}"),
        Check("sandbox pool scales from zero", has(r"min_node_count\s*=\s*0"), "autoscaling.min_node_count = 0"),
        Check("sandbox pool dedicated (GKE's taint)", has(r'gke_sandbox_taint\s*=\s*"sandbox.gke.io/runtime=gvisor:NoSchedule"'),
              "GKE taints sandbox nodes itself; RuntimeClass gvisor carries the toleration (verify)"),
        Check("private nodes", has(r"enable_private_nodes\s*=\s*true"), "no external IPs on nodes"),
        Check("no Cloud NAT by default", variable_default("enable_nat", tf_dir) == "false",
              f"enable_nat default = {variable_default('enable_nat', tf_dir)}: no route to the internet"),
        Check("NetworkPolicy enforced", has(r'datapath_provider\s*=\s*"ADVANCED_DATAPATH"'), "Dataplane V2"),
        Check("Workload Identity / GKE metadata server", has(r'mode\s*=\s*"GKE_METADATA"'),
              "pods never see the node's service account"),
        Check("per-pod PID limit", has(r"pod_pids_limit\s*=\s*var\.pod_pids_limit"),
              f"kubelet podPidsLimit = {variable_default('pod_pids_limit', tf_dir)}"),
        Check("Artifact Registry for images", has(r'resource "google_artifact_registry_repository"'),
              "private nodes pull from *.pkg.dev over Private Google Access (verify)"),
        Check("managed Prometheus", has(r"managed_prometheus\s*{\s*enabled\s*=\s*true"), "kill reasons and latency as metrics"),
    ]


GCLOUD_EQUIVALENT = """\
# The Terraform, as gcloud (zonal Standard cluster + a GKE Sandbox pool); names match the tfvars defaults (verify flags).
gcloud container clusters create sandbox-lab --zone us-central1-a --release-channel regular \\
  --enable-dataplane-v2 --enable-private-nodes --master-ipv4-cidr 172.16.0.32/28 --enable-ip-alias \\
  --workload-pool PROJECT_ID.svc.id.goog --enable-managed-prometheus \\
  --num-nodes 1 --machine-type e2-standard-2
gcloud container node-pools create gvisor --cluster sandbox-lab --zone us-central1-a \\
  --sandbox type=gvisor --image-type cos_containerd --machine-type e2-standard-2 --spot \\
  --enable-autoscaling --min-nodes 0 --max-nodes 3 --workload-metadata GKE_METADATA
kubectl get runtimeclass gvisor -o yaml      # created by GKE with the first sandbox pool
"""


def cost_per_execution_usd(held_s: float, node_usd_per_hour: float, pods_per_node: int) -> float:
    """One execution holds one sandbox pod's share of a node for ``held_s`` seconds."""
    return held_s * node_usd_per_hour / 3600.0 / pods_per_node


def pods_per_node(node_vcpu: float, node_mem_gib: float, pod_cpu: float, pod_mem_gib: float,
                  reserved_vcpu: float = 0.5, reserved_mem_gib: float = 1.5, overhead_cpu: float = 0.0,
                  overhead_mem_gib: float = 0.0) -> int:
    """How many sandbox pods fit on a node after system reservations (inputs: verify for your machine type)."""
    by_cpu = (node_vcpu - reserved_vcpu) // (pod_cpu + overhead_cpu)
    by_mem = (node_mem_gib - reserved_mem_gib) // (pod_mem_gib + overhead_mem_gib)
    return int(max(0, min(by_cpu, by_mem)))
