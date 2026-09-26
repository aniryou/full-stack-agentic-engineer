# %% [markdown]
# # 05 · GKE Inference Gateway
#
# **Tier:** T3 — deploying needs a GCP project with billing, L4 quota and ~an hour of an L4 Spot VM.
# This notebook itself runs offline (T0): it inspects the Terraform and manifests in `deploy/`,
# builds the Kubernetes objects in Python, validates them against the upstream CRD schemas, and
# prints the plan. Nothing here calls Google Cloud unless you set `IGW_GKE=1` (read-only `kubectl`).
#
# ## The one-minute version
#
# On GKE, the router of notebooks 01–04 becomes managed infrastructure:
#
# ```
# client ─HTTP─▶ Gateway (gke-l7-regional-external-managed: a regional external Application LB,
#                │          Envoy proxies in the VPC's proxy-only subnet)
#                ├─ HTTPRoute  /  ─▶ backendRef: InferencePool vllm-qwen   (not a Service)
#                │                     ├─ selector app=vllm-qwen, targetPorts [8000]  → vLLM pods on L4 Spot
#                │                     └─ endpointPickerRef vllm-qwen-epp:9002        → llm-d EPP (ext-proc)
#                └─ per request: LB ──ext-proc──▶ EPP: "which pod?" (prefix/queue/KV scores) ──▶ pod
# Managed Prometheus scrapes vLLM /metrics (PodMonitoring) → Custom Metrics Adapter → HPA on the queue
# InferenceObjective premium(100) / standard(0) / batch(-10) ← header x-llm-d-inference-objective
# ```
#
# Terraform owns the cluster (zonal, Gateway API on, Managed Prometheus on), the proxy-only subnet and
# an L4 Spot node pool that autoscales 0 → 2. Helm installs the EPP (and, from the same values, the
# InferencePool and InferenceObjectives). Plain manifests add vLLM, the Gateway/HTTPRoute, the
# PodMonitoring and the HPA. Background: [PRIMER §9 The Kubernetes-native stack, Sep 2026 and
# §10 Where to run it](../../PRIMER.md).

# %%
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

from igwlab import k8s

LAB = Path.cwd().resolve()
while not (LAB / "igwlab").exists():
    LAB = LAB.parent
DEPLOY = LAB / "deploy"

# %% [markdown]
# ## The Terraform, resource by resource

# %%
tf = {p.name: p.read_text() for p in sorted((DEPLOY / "gcp/terraform").glob("*.tf"))}
for name, text in tf.items():
    for kind, rname in re.findall(r'^resource "([a-z0-9_]+)" "([a-z0-9_]+)"', text, re.M):
        print(f"{name:<14} {kind:<32} {rname}")
defaults = dict(re.findall(r'variable "([a-z_]+)" \{[^}]*?default\s+=\s+("[^"]*"|\d+|true|false)', tf["variables.tf"], re.S))
print("\ncheap defaults:", {k: defaults[k] for k in ("zone", "gpu_machine_type", "gpu_type", "gpu_spot", "gpu_max_nodes", "release_channel")})

# %% [markdown]
# Things to notice: the cluster is **zonal** (one control plane zone); the GPU pool has
# `min_node_count = 0`, so no GPU VM exists until a vLLM pod is pending; `spot = true`; GKE installs
# the NVIDIA driver (`gpu_driver_version = "DEFAULT"`); `gateway_api_config` installs the Gateway API
# and GKE's GatewayClasses; the **proxy-only subnet** (`purpose = "REGIONAL_MANAGED_PROXY"`) is where a
# regional Application Load Balancer runs its Envoys — without it the Gateway never gets an address.
#
# ## The Kubernetes objects, built and checked offline
#
# `igwlab.k8s` builds each object and `k8s.check()` validates it against the upstream CRD schema
# snapshots in `igwlab/crds/` (InferencePool v1 from GIE v1.6.2, InferenceObjective v1alpha2 from
# llm-d-router v0.10.0, Gateway/HTTPRoute v1 from Gateway API v1.6.2, PodMonitoring v1 from GMP).

# %%
schemas = k8s.load_schemas()
print("schemas:", [f"{k} ({v})" for v, k in schemas])
for f in ("gke/gateway.yaml", "gke/podmonitoring-vllm.yaml", "gke/rendered/inferencepool.yaml",
          "gke/rendered/inferenceobjectives.yaml"):
    print(f"{f:<42}", {f"{kind}/{name}": errs or "valid" for (kind, name), errs in k8s.check_file(DEPLOY / f, schemas).items()})

# %% [markdown]
# ## Exercise 5.1 — the InferencePool
#
# Build (as a plain dict, without `k8s.inference_pool`) the InferencePool the chart creates for the
# Helm release `vllm-qwen`: it selects pods labelled `app: vllm-qwen`, targets port 8000 over plain
# HTTP (`appProtocol: http`), and points at the EPP Service `vllm-qwen-epp` on port 9002 with
# `failureMode: FailClose` (if the EPP is down, fail the request rather than route blind).

# %% exercise
def my_inference_pool() -> dict:
    ### BEGIN SOLUTION
    return {"apiVersion": "inference.networking.k8s.io/v1", "kind": "InferencePool",
            "metadata": {"name": "vllm-qwen"},
            "spec": {"targetPorts": [{"number": 8000}], "appProtocol": "http",
                     "selector": {"matchLabels": {"app": "vllm-qwen"}},
                     "endpointPickerRef": {"name": "vllm-qwen-epp", "port": {"number": 9002}, "failureMode": "FailClose"}}}
    ### END SOLUTION

# %% check
mine = my_inference_pool()
assert k8s.check(mine, schemas) == [], k8s.check(mine, schemas)
rendered = yaml.safe_load((DEPLOY / "gke/rendered/inferencepool.yaml").read_text())
assert mine == rendered, "compare with deploy/gke/rendered/inferencepool.yaml"
broken = {**mine, "spec": {**mine["spec"], "endpointPickerRef": {"name": "vllm-qwen-epp", "failureMode": "FailClosed"}}}
print("the chart README's spelling would be rejected:", k8s.check(broken, schemas))
print("✅ InferencePool v1 is valid and matches what the chart renders")

# %% [markdown]
# ## Exercise 5.2 — route traffic to the pool
#
# Write the HTTPRoute (dict) named `vllm-qwen` that attaches to the Gateway `inference-gateway` and
# sends every path (`PathPrefix /`) to the InferencePool `vllm-qwen`. The backendRef must name the
# InferencePool's API group and kind; it needs no port (the pool's `targetPorts` decide).

# %% exercise
def my_route() -> dict:
    ### BEGIN SOLUTION
    return {"apiVersion": "gateway.networking.k8s.io/v1", "kind": "HTTPRoute", "metadata": {"name": "vllm-qwen"},
            "spec": {"parentRefs": [{"name": "inference-gateway"}],
                     "rules": [{"matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                                "backendRefs": [{"group": "inference.networking.k8s.io", "kind": "InferencePool",
                                                 "name": "vllm-qwen"}]}]}}
    ### END SOLUTION

# %% check
route = my_route()
assert k8s.check(route, schemas) == []
ref = route["spec"]["rules"][0]["backendRefs"][0]
assert (ref["group"], ref["kind"], ref["name"]) == ("inference.networking.k8s.io", "InferencePool", "vllm-qwen")
on_disk = [d for d in yaml.safe_load_all((DEPLOY / "gke/gateway.yaml").read_text()) if d["kind"] == "HTTPRoute"][0]
assert route == on_disk
print("✅ HTTPRoute -> InferencePool (a Service backendRef would bypass the EPP entirely)")

# %% [markdown]
# ## Exercise 5.3 — from Prometheus series to an HPA metric name
#
# Managed Service for Prometheus stores a scraped series `NAME` of kind `KIND` (`gauge`, `counter`, …)
# in Cloud Monitoring as the metric type `prometheus.googleapis.com/NAME/KIND`. The Custom Metrics
# Stackdriver Adapter exposes metric types to the HPA with `/` replaced by `|` (a metric name cannot
# contain `/`). Write `hpa_metric_name(name, kind)` and check it against `deploy/gke/hpa.yaml`.
# *(The adapter naming is marked VERIFY in the manifest — confirm it against current GKE docs.)*

# %% exercise
def hpa_metric_name(name: str, kind: str) -> str:
    ### BEGIN SOLUTION
    return f"prometheus.googleapis.com/{name}/{kind}".replace("/", "|")
    ### END SOLUTION

# %% check
hpa = yaml.safe_load((DEPLOY / "gke/hpa.yaml").read_text())
assert hpa_metric_name("vllm:num_requests_waiting", "gauge") == hpa["spec"]["metrics"][0]["pods"]["metric"]["name"]
print("✅", hpa_metric_name("vllm:num_requests_waiting", "gauge"))

# %% [markdown]
# ## Exercise 5.4 — what an hour of the lab costs
#
# Write `lab_cost(hours, gpu_nodes, prices)` in dollars for `hours` of: `gpu_nodes` L4 Spot VMs, one
# system VM, the regional load balancer's forwarding rule, and the cluster management fee minus the
# free-tier credit (which covers one zonal cluster, so the fee nets to 0 here). The prices below are
# **assumptions to verify** (us-central1, Sep 2026), not quotes — see [COMPUTE.md](../../../../COMPUTE.md).

# %% exercise
PRICES = {                         # USD per hour — VERIFY before relying on them
    "g2-standard-4": 0.70,         # on-demand (FACTS); Spot is 60-91% cheaper
    "spot_discount": 0.60,         # assume the *smallest* Spot discount
    "e2-standard-4": 0.134,        # system node (verify)
    "lb_forwarding_rule": 0.025,   # regional external ALB, first rule (verify)
    "cluster_fee": 0.10,           # GKE management fee (verify)
    "cluster_fee_credit": 0.10,    # free tier: one zonal/Autopilot cluster per billing account (verify)
}

def lab_cost(hours: float, gpu_nodes: int, prices: dict = PRICES) -> float:
    ### BEGIN SOLUTION
    gpu = gpu_nodes * prices["g2-standard-4"] * (1 - prices["spot_discount"])
    per_hour = gpu + prices["e2-standard-4"] + prices["lb_forwarding_rule"] + max(0.0, prices["cluster_fee"] - prices["cluster_fee_credit"])
    return round(hours * per_hour, 4)
    ### END SOLUTION

# %% check
assert lab_cost(1, 1) == round(0.28 + 0.134 + 0.025, 4) == 0.439
assert lab_cost(2, 2) == round(2 * (0.56 + 0.134 + 0.025), 4)
assert lab_cost(1, 0) == 0.159                           # GPU pool at zero: you still pay for the rest
print(f"✅ one hour with one L4 Spot node ≈ ${lab_cost(1, 1):.2f} (assumed prices); an idle day with the GPU pool at 0 ≈ ${lab_cost(24, 0):.2f}")

# %% [markdown]
# The last number is the reason for `uninstall.sh` **and** `terraform destroy`: scale-to-zero of the
# GPU pool removes the expensive part, but the system node, the load balancer and any disks keep
# billing until the cluster is gone.
#
# ## The plan
#
# `deploy/gke/install.sh` in dry-run mode — the exact order matters: CRDs before the chart that
# creates objects of those kinds; the InferencePool (Helm) before the HTTPRoute that references it;
# the metrics adapter before the HPA that needs its API.

# %%
env = {"DRY_RUN": "1", "PROJECT_ID": "my-project", "PATH": "/usr/bin:/bin"}
out = subprocess.run(["bash", str(DEPLOY / "gke/install.sh")], env=env, capture_output=True, text=True, check=True).stdout
print("\n".join(l for l in out.splitlines() if l.startswith(("==>", "+ "))))

# %% [markdown]
# ## If you have deployed it
#
# With `IGW_GKE=1` and `kubectl` pointed at the cluster, this cell prints read-only status; otherwise
# it prints what to run.

# %%
if os.environ.get("IGW_GKE") == "1" and shutil.which("kubectl"):
    for cmd in (["kubectl", "get", "inferencepools,inferenceobjectives,gateways,httproutes"],
                ["kubectl", "get", "hpa", "vllm-qwen"], ["kubectl", "get", "pods", "-l", "app=vllm-qwen", "-o", "wide"]):
        print("$", " ".join(cmd))
        print(subprocess.run(cmd, capture_output=True, text=True).stdout or "(no output)")
else:
    print("offline: to deploy, run (see deploy/gcp/terraform/README.md and deploy/gke/README.md)")
    print("  cd deploy/gcp/terraform && cp terraform.tfvars.example terraform.tfvars && terraform init && terraform apply")
    print("  PROJECT_ID=<id> ZONE=us-central1-a deploy/gke/install.sh")
    print("then re-run this notebook with IGW_GKE=1 for live status, and tear down with deploy/gke/uninstall.sh + terraform destroy")

# %% [markdown]
# ## In a design review
#
# **Two-minute walkthrough.** "On GKE we use Gateway mode. Terraform creates a zonal Standard cluster
# with the Gateway API and Managed Prometheus enabled, a proxy-only subnet for the regional load
# balancer, and an L4 Spot pool that scales from zero. vLLM runs one replica per L4. The llm-d router
# chart installs the endpoint picker with the same scoring config we tuned locally, plus the
# InferencePool that selects the vLLM pods and our three InferenceObjectives. A Gateway of class
# `gke-l7-regional-external-managed` with an HTTPRoute to the InferencePool makes the load balancer
# ask the EPP for a pod on every request. Managed Prometheus scrapes vLLM, the custom-metrics adapter
# exposes `vllm:num_requests_waiting` to the HPA, and the HPA adds one replica a minute. An idle hour
# costs cents because the GPU pool is at zero; a busy hour is dominated by the Spot L4."
#
# **Drill questions**
#
# 1. *Why a proxy-only subnet?* Regional (and internal) Application Load Balancers run managed Envoy
#    proxies inside your VPC; they need their own subnet with purpose `REGIONAL_MANAGED_PROXY`.
# 2. *What breaks if the HTTPRoute points at a Service instead of the InferencePool?* The load balancer
#    balances by its own policy and never calls the EPP: no prefix affinity, no queue awareness, no
#    objectives or shedding.
# 3. *The EPP pod crashes — what happens to traffic?* It depends on the pool's `failureMode`: `FailClose`
#    fails requests until an EPP is back (safe but unavailable); `FailOpen` routes without scoring.
#    Run 2+ EPP replicas (active-passive with leader election by default in the chart).
