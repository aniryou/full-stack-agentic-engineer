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
# Managed Prometheus scrapes vLLM /metrics (PodMonitoring) → Custom Metrics Adapter → HPA on waiting + running
# InferenceObjective premium(100) / standard(0) / batch(-10) ← header x-llm-d-inference-objective
# ```
#
# Terraform owns the cluster (zonal, Gateway API on, Managed Prometheus on), the proxy-only subnet and
# an L4 Spot node pool that autoscales 0 → 2. Helm installs the EPP (and, from the same values, the
# InferencePool and InferenceObjectives). Plain manifests add vLLM, the Gateway/HTTPRoute, the
# PodMonitoring and the HPA. Background: [PRIMER §9 The Kubernetes-native stack, September 2026 and
# §10 Where to run it](../../PRIMER.md).

# %%
import json
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
# ## Exercise 5.1 — what the API server catches, and what it lets through
#
# A schema check stops some mistakes at `kubectl apply`; others apply cleanly and break the data
# path. Below are six variants of the lab's objects (the shipped ones are in `deploy/gke/`: the
# InferencePool as the chart renders it, the HTTPRoute in `gateway.yaml`; the vLLM pods are
# labelled `app: vllm-qwen`). For each, predict one of:
#
# * `"rejected"` — the API server refuses it (CRD schema or CEL rule);
# * `"no-endpoints"` — it applies, but the pool contains no pod, so every request fails;
# * `"no-epp"` — it applies and traffic flows, but the load balancer never asks the EPP (no prefix
#   affinity, no queue awareness, no objectives or shedding);
# * `"ok"`.

# %%
import copy

pool = yaml.safe_load((DEPLOY / "gke/rendered/inferencepool.yaml").read_text())
route = [d for d in yaml.safe_load_all((DEPLOY / "gke/gateway.yaml").read_text()) if d["kind"] == "HTTPRoute"][0]

def variant(obj, edit):
    o = copy.deepcopy(obj)
    edit(o["spec"])
    return o

described = {
    "A": ("InferencePool, endpointPickerRef.failureMode: FailClosed (the chart README's spelling)",
          variant(pool, lambda s: s["endpointPickerRef"].update(failureMode="FailClosed"))),
    "B": ("InferencePool, 9 targetPorts 8000-8008 (one per data-parallel rank)",
          variant(pool, lambda s: s.update(targetPorts=[{"number": 8000 + i} for i in range(9)]))),
    "C": ("InferencePool, endpointPickerRef without a port",
          variant(pool, lambda s: s["endpointPickerRef"].pop("port"))),
    "D": ("InferencePool, selector app: vllm",
          variant(pool, lambda s: s["selector"].update(matchLabels={"app": "vllm"}))),
    "E": ("HTTPRoute, backendRefs: [{name: vllm-qwen-svc, port: 8000}] (a Service selecting the vLLM pods)",
          variant(route, lambda s: s["rules"][0].update(backendRefs=[{"name": "vllm-qwen-svc", "port": 8000}]))),
    "F": ("HTTPRoute as shipped (backendRef group inference.networking.k8s.io, kind InferencePool)", route),
}
variants = {k: obj for k, (_, obj) in described.items()}
for k, (text, _) in described.items():
    print(k, text)

# %% exercise
verdicts = {}          # {"A": ..., ..., "F": ...}
### BEGIN SOLUTION
verdicts = {"A": "rejected",       # failureMode enum is FailOpen | FailClose
            "B": "rejected",       # targetPorts: 1..8 items
            "C": "rejected",       # CEL: a Service endpointPickerRef needs a port
            "D": "no-endpoints",   # valid, but selects no pod: the EPP has nothing to pick -> errors
            "E": "no-epp",         # group/kind default to core Service: the LB balances on its own
            "F": "ok"}
### END SOLUTION

# %% check
vllm_labels = next(yaml.safe_load_all((DEPLOY / "gke/vllm.yaml").read_text()))["spec"]["template"]["metadata"]["labels"]
for k, obj in variants.items():
    errs = k8s.check(obj, schemas)
    assert (verdicts[k] == "rejected") == bool(errs), (k, verdicts[k], errs)
    if obj["kind"] == "InferencePool" and not errs:
        selects = all(vllm_labels.get(a) == b for a, b in obj["spec"]["selector"]["matchLabels"].items())
        assert (verdicts[k] == "no-endpoints") == (not selects), k
    if obj["kind"] == "HTTPRoute":
        ref = obj["spec"]["rules"][0]["backendRefs"][0]
        assert (verdicts[k] == "no-epp") == (ref.get("kind", "Service") != "InferencePool"), k
print("rejections:", {k: k8s.check(o, schemas) for k, o in variants.items() if k8s.check(o, schemas)})
print("✅ the schema stops A-C at apply time; D and E apply cleanly and fail later — that is what the"
      " route/pool status conditions and a smoke request are for")

# %% [markdown]
# ## Exercise 5.2 — pick the failure mode
#
# `endpointPickerRef.failureMode` decides what the load balancer does when it cannot reach the EPP:
# `FailOpen` forwards the request to some pod of the pool without asking; `FailClose` fails it. Pick
# the mode for each pool and justify it to yourself (the solution states the reasoning):
#
# * `"chat"` — every pod serves the same model; the product prefers a slower answer to an error.
# * `"adapters"` — requests name LoRA adapters, and each adapter is loaded (statically,
#   `--lora-modules`) on only some pods; the EPP's LoRA affinity is what sends a request to a pod
#   that has its adapter. A pod without it answers 404 "model does not exist".

# %% exercise
failure_mode = {}      # {"chat": "FailOpen" | "FailClose", "adapters": ...}
### BEGIN SOLUTION
failure_mode = {"chat": "FailOpen",       # any pod gives a correct answer; the EPP only makes it faster
                "adapters": "FailClose"}  # a blind pick is usually WRONG (404), not slow; a fast 503 is
                                          # retryable and pages someone, a 404 looks like a client bug
### END SOLUTION

# %% check
assert failure_mode == {"chat": "FailOpen", "adapters": "FailClose"}, failure_mode
shipped = pool["spec"]["endpointPickerRef"]["failureMode"]
print(f"✅ FailOpen when routing only optimizes, FailClose when it decides correctness. The lab ships "
      f"{shipped}: one base model, so FailOpen would also be defensible — and either way run 2+ EPP replicas.")

# %% [markdown]
# ## From Prometheus series to an HPA metric name
#
# Managed Service for Prometheus stores a scraped series `NAME` of kind `KIND` in Cloud Monitoring as
# the metric type `prometheus.googleapis.com/NAME/KIND`; the Custom Metrics Stackdriver Adapter
# exposes metric types to the HPA with `/` replaced by `|` (its README says so; the GMP naming is
# marked VERIFY in the manifest). The shipped HPA uses two of them:

# %%
hpa = yaml.safe_load((DEPLOY / "gke/hpa.yaml").read_text())
for m in hpa["spec"]["metrics"]:
    name = m["pods"]["metric"]["name"]
    assert name == f"prometheus.googleapis.com/{name.split('|')[1]}/gauge".replace("/", "|")
    print(f"{name:<58} target {m['pods']['target']['averageValue']} per pod")

# %% [markdown]
# ## Exercise 5.3 — how late is the first extra replica?
#
# A burst starts at t = 0 and pushes `vllm:num_requests_waiting` far above target. Predict the
# **worst-case** time until a new vLLM pod receives traffic. The chain, in order, and where each
# number comes from:
#
# | step | worst case | source |
# |---|---|---|
# | GMP scrapes the pod | one full scrape interval (the burst lands just after a scrape) | `podmonitoring-vllm.yaml` |
# | the sample reaches the HPA through Cloud Monitoring and the adapter | `ADAPTER_LAG_S` | assumption (verify) |
# | the HPA controller's next sync | one full sync period, 15 s | kube-controller-manager default |
# | a new L4 Spot node is provisioned and joins | `NODE_S` | assumption (verify; Spot may not be obtainable at all) |
# | the vLLM image is pulled | `PULL_S` | assumption (verify; image streaming shortens it) |
# | weights download + load, until `/health` answers | `LOAD_S` | assumption (verify) |
# | the readiness probe notices | one full probe period | `vllm.yaml` |
#
# (The scale-up policy — 1 pod per 60 s — does not delay the *first* pod.) Write `worst_case_s()`
# reading the two intervals from the files; it returns seconds.

# %%
ADAPTER_LAG_S = 60      # Cloud Monitoring ingestion + adapter read (assumption, verify)
NODE_S = 120            # L4 Spot VM create + GKE node registration (assumption, verify)
PULL_S = 150            # vllm/vllm-openai is several GB (assumption, verify)
LOAD_S = 60             # Qwen2.5-1.5B: ~3 GB of weights from Hugging Face + load (assumption, verify)
podmon = yaml.safe_load((DEPLOY / "gke/podmonitoring-vllm.yaml").read_text())
vllm_dep = next(yaml.safe_load_all((DEPLOY / "gke/vllm.yaml").read_text()))

# %% exercise
def worst_case_s() -> float:
    ### BEGIN SOLUTION
    scrape = float(podmon["spec"]["endpoints"][0]["interval"].rstrip("s"))
    probe = vllm_dep["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]["periodSeconds"]
    return scrape + ADAPTER_LAG_S + 15 + NODE_S + PULL_S + LOAD_S + probe
    ### END SOLUTION

# %% check
assert worst_case_s() == 15 + 60 + 15 + 120 + 150 + 60 + 5, worst_case_s()
detect = 15 + ADAPTER_LAG_S + 15
print(f"✅ ~{worst_case_s() / 60:.1f} min worst case (assumptions); only {detect} s of it is detection — "
      f"the rest is capacity arriving, so the queue built meanwhile is what minReplicas, a warm node or "
      f"faster cold starts (image streaming, cached weights: layer 03) must absorb")

# %% [markdown]
# ## Exercise 5.4 — what an hour of the lab costs
#
# Write `lab_cost(hours, gpu_nodes, prices)` in dollars for `hours` of: `gpu_nodes` L4 Spot VMs, one
# system VM, the regional load balancer's forwarding rule, and the cluster management fee minus the
# free-tier credit (which covers one zonal cluster, so the fee nets to 0 here). The prices below are
# **assumptions to verify** (us-central1, Sep 2026), not quotes — see [COMPUTE.md](../../../../COMPUTE.md).

# %% exercise
PRICES = {                         # USD per hour — VERIFY before relying on them
    "g2-standard-4": 0.70,         # on-demand, ~Sep 2026 (verify); Spot is 60-91% cheaper
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
print(f"✅ installed, busy or idle: ≈ ${lab_cost(1, 1):.2f}/h with its one L4 Spot node (assumed prices); "
      f"uninstalled but not destroyed, a forgotten day still costs up to ≈ ${lab_cost(24, 0):.2f}")

# %% [markdown]
# Read the three states apart. **Installed and idle**: `lab_cost(1, 1)` ≈ $0.44/h, because the
# HPA's `minReplicas: 1` keeps one vLLM pod and so one L4 Spot node up — the GPU pool never reaches
# 0 while the workloads exist (that needs scale-to-zero: KEDA or the alpha `HPAScaleToZero`).
# **After `uninstall.sh`**: the pool drains to 0 and at most `lab_cost(1, 0)` ≈ $0.16/h remains — the
# system node and disks (~$0.13/h plus disks), and the load balancer's forwarding rule only while a
# Gateway still exists (`uninstall.sh` deletes it; `lab_cost` counts it anyway, as an upper bound).
# **After `terraform destroy`**: $0.
# That last step is the one people forget.
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
    print("then re-run this notebook with IGW_GKE=1 for live status, and tear down with "
          "PROJECT_ID=<id> deploy/gke/uninstall.sh + terraform destroy")

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
# exposes `vllm:num_requests_waiting` and `vllm:num_requests_running` to the HPA — the queue for
# bursts, the occupied batch slots so it does not scale away capacity once the queue drains — and the
# HPA adds at most one replica a minute; the first extra replica still arrives minutes after a burst,
# because a Spot node, the image and the weights come first. With one replica always up, an idle
# hour costs about $0.44 (one L4 Spot node plus the fixed parts); only uninstalling the workloads
# brings the GPU pool to zero, and only `terraform destroy` stops the rest."
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
