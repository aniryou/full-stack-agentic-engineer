# %% [markdown]
# # 04 · GKE: node pools, DWS flex-start and ComputeClasses — capacity is something you ask for
#
# **Tier:** T3 walkthrough (GKE via `deploy/gcp/terraform`; a Spot L4 node costs roughly $0.25/h
# while a pod needs it — verify). Offline everything here is **plan and inspect**: read the
# Terraform, price it, model the capacity choices, and render the manifests; with a GKE
# `kubectl` context the last section looks at the live objects.
#
# ## The one-minute version
# On a laptop the GPUs are simply there. In a cloud they are **obtained**, and there are four
# ways, each trading price against the chance of getting the capacity when you need it:
#
# | how | price | you get | good for |
# |---|---|---|---|
# | on-demand | list | whatever is in stock (scarce shapes stock out) | serving, short jobs |
# | Spot | 60-91 % off | reclaimable with ~30 s notice | checkpointed batch, extra replicas |
# | DWS flex-start (queued) | discounted (verify) | *all N nodes at once* after a queue, ≤ 7 days | gangs of scarce GPUs |
# | reservation / calendar | list or committed | guaranteed, billed when idle | deadlines, steady load |
#
# A node pool that scales from zero, a **ComputeClass** that falls back Spot → on-demand →
# flex-start, and **Kueue + ProvisioningRequest** (all-or-nothing provisioning *before*
# admission) are how GKE turns that table into objects. Startup latency — node boot, driver,
# image, weights — is the other cost of "from zero". Primer §7 *Getting capacity*, §8 *Startup
# latency*, §2 *GPU Operator vs managed drivers*.

# %%
from k8sgpu import capacity, gke
from k8sgpu import manifests as m

v = gke.effective_vars()          # variables.tf defaults + terraform.tfvars.example
print(gke.plan_summary(v))

# %% [markdown]
# ## What the Terraform builds
# One zonal GKE Standard cluster, three pools. The GPU pools carry the
# `nvidia.com/gpu=present:NoSchedule` taint and let GKE install the driver
# (`gpu_driver_installation_config`) — no GPU Operator needed (primer §2). The same thing in
# `gcloud`, which is how most GKE documentation shows it:

# %%
for f, rtype, name in gke.tf_resources():
    print(f"{f:16} {rtype}.{name}")
print()
for cmd in gke.gcloud_equivalents({**v, "enable_flex_start_pool": True}):
    print("$", cmd, "\n")

# %% [markdown]
# ## Spot, in numbers
# A gang is interrupted when **any** of its nodes is reclaimed, so the rates add:
# 8 nodes at 0.02 reclaims per node-hour = 0.16 per hour. Each interruption loses the work
# since the last checkpoint plus a restart. With Poisson interruptions at rate λ, checkpoints
# every τ hours and restart cost R, one τ-hour segment takes on average
# `(1/λ + R)(e^(λτ) - 1)` hours, and a job of W hours has W/τ segments.
#
# ## Exercise 4.1 — expected runtime on Spot
# Implement `runtime_h(work_h, rate_per_h, checkpoint_every_h, restart_h)` (no checkpoints =
# `checkpoint_every_h=None`, i.e. one segment of `work_h`). Then compute `slowdown`: how many
# times longer a 10 h, 8-node job takes on Spot **without** checkpoints than with hourly ones
# (λ = 0.16/h, R = 0.25 h).

# %% exercise
import math

def runtime_h(work_h: float, rate_per_h: float, checkpoint_every_h: float | None, restart_h: float = 0.25) -> float:
    ### BEGIN SOLUTION
    if rate_per_h == 0:
        return work_h
    tau = checkpoint_every_h or work_h
    return (work_h / tau) * (1 / rate_per_h + restart_h) * (math.exp(rate_per_h * tau) - 1)
    ### END SOLUTION

### BEGIN SOLUTION
slowdown = runtime_h(10, 0.16, None) / runtime_h(10, 0.16, 1.0)
### END SOLUTION

# %% check
assert math.isclose(runtime_h(10, 0.16, 1.0), capacity.expected_runtime_h(10, 0.16, 1.0), rel_tol=1e-9)
assert math.isclose(runtime_h(10, 0.16, None), 25.6947, rel_tol=1e-4)
assert math.isclose(slowdown, 2.278, rel_tol=1e-3)
print(f"✅ hourly checkpoints: {runtime_h(10, 0.16, 1.0):.2f} h; none: {runtime_h(10, 0.16, None):.2f} h "
      f"({slowdown:.2f}x). Spot is cheap only if the job can resume.")

# %% [markdown]
# ## Choosing a capacity type
# `capacity.evaluate()` puts the options side by side for one need: expected wait (queues and
# retries), expected runtime (interruptions), cost, and what breaks. All prices and rates are
# assumptions marked *verify* — change them and watch the ranking move.

# %%
needs = {
    "serve-24x7": capacity.Need("serve", "g2-standard-4", nodes=2, work_h=730, serving=True, checkpoint_every_h=None),
    "finetune-10h": capacity.Need("finetune", "a3-highgpu-8g", nodes=1, work_h=10, checkpoint_every_h=1.0),
    "pretrain-3d": capacity.Need("pretrain", "a3-highgpu-8g", nodes=8, work_h=72, checkpoint_every_h=None,
                                 deadline_h=96),
}
for label, need in needs.items():
    print(f"--- {label}\n{capacity.format_evaluations(capacity.evaluate(need))}\n")

# %% [markdown]
# ## Exercise 4.2 — pick one per workload, and say why
# Using the tables above, set `choice` to one of `"on-demand"`, `"spot"`, `"flex-start"`,
# `"reservation"` for each workload. The 3-day pretraining cannot checkpoint (an unusual,
# deliberately hard constraint) and must finish within 96 hours.

# %% exercise
choice = {"serve-24x7": None, "finetune-10h": None, "pretrain-3d": None}
### BEGIN SOLUTION
choice = {
    "serve-24x7": "on-demand",      # interruptions are outages; Spot only for extra replicas behind a fallback
    "finetune-10h": "spot",         # checkpointed, 1 node: the interruption tax is small, the discount large
    "pretrain-3d": "flex-start",    # 8 nodes all-or-nothing, no checkpoints (Spot restarts forever), fits a 7-day lease
}
### END SOLUTION

# %% check
for label, need in needs.items():
    assert choice[label] == capacity.evaluate(need)[0].option, (label, choice[label])
spot = capacity.evaluate_option(needs["pretrain-3d"], capacity.OPTIONS["spot"])
print(f"✅ matches the model; e.g. 8-node no-checkpoint Spot would need ~{spot.run_h:,.0f} h for 72 h of work")

# %% [markdown]
# ## ComputeClass: the fallback ladder as an object
# Instead of one node pool per capacity type, a custom **ComputeClass** lists ways to get a
# node, in order; GKE's node-pool auto-creation tries them top to bottom and (with
# `activeMigration`) moves pods back up the ladder when the preferred capacity returns. Pods
# opt in with `nodeSelector: {cloud.google.com/compute-class: <name>}`. Field names follow
# GKE's CRD (verify with `kubectl explain computeclass.spec`).

# %%
files = gke.gke_manifests()
cc = files["30-computeclass-l4.yaml"][1][0]
print(m.to_yaml(cc))

# %% [markdown]
# ## Exercise 4.3 — which rung provisions?
# Write `pick_rung(priorities, available)`: the index of the first priority whose capacity type
# (`"reservation"` if it has `reservations`, `"flex-start"` if `flexStart.enabled`, else `"spot"`
# or `"on-demand"` from `spot`) is available, or `None` (with `whenUnsatisfiable: DoNotScaleUp`
# the pod then stays Pending).

# %% exercise
def pick_rung(priorities: list[dict], available: dict[str, bool]) -> int | None:
    ### BEGIN SOLUTION
    for i, p in enumerate(priorities):
        kind = ("reservation" if p.get("reservations") else
                "flex-start" if (p.get("flexStart") or {}).get("enabled") else
                "spot" if p.get("spot") else "on-demand")
        if available.get(kind):
            return i
    return None
    ### END SOLUTION

# %% check
ladder = cc["spec"]["priorities"]
assert pick_rung(ladder, {"spot": True, "on-demand": True, "flex-start": True}) == 0
assert pick_rung(ladder, {"spot": False, "on-demand": True}) == 1            # Spot stockout -> on-demand
assert pick_rung(ladder, {"flex-start": True}) == 2                          # only DWS has L4s today
assert pick_rung(ladder, {}) is None
assert pick_rung(ladder, {"on-demand": True}) == capacity.compute_class_pick(ladder, {"on-demand": True})[0]
print("✅ Spot first, on-demand when Spot is gone, flex-start when both are")

# %% [markdown]
# ## DWS flex-start through Kueue: admission waits for *all* the nodes
# `deploy/gke/10-kueue-gke.yaml` gives the ClusterQueue two flavors. `l4-spot` is plain quota.
# `l4-flex` has the `dws-prov` AdmissionCheck: after reserving quota, Kueue files a
# **ProvisioningRequest** (`queued-provisioning.gke.io`) for every pod of the Workload, and admits
# only when DWS has created all the nodes together — no half-started gang holding GPUs hostage.

# %%
print(m.to_yaml(*files["10-kueue-gke.yaml"][1][3:6]))
for t, event in capacity.queued_provisioning_timeline(nodes=2, wait_h=0.5, run_h=1.0):
    print(f"t={t:5.2f} h  {event}")

# %% [markdown]
# ## Startup latency: where the minutes go when you scale from zero
# A Pending serving pod on a pool at zero waits for: a VM (minutes for GPU shapes), the GPU
# driver, the image (vLLM's is ~10 GB — verify), the weights, and engine start-up. Image
# streaming starts the container after fetching metadata; GCS FUSE with parallel downloads (or
# Hyperdisk ML, or a model streamer) moves weights at hundreds of MB/s. Only the last two
# stages happen *after* the container starts — that is what the startup probe must cover.

# %%
for streaming in (False, True):
    print("image streaming" if streaming else "full image pull", capacity.cold_start_s(image_streaming=streaming))

# %% [markdown]
# ## Exercise 4.4 — size the serving pod's startup probe
# The serving example (`deploy/gke/40-serving-vllm-gcsfuse.yaml`) loads a 1.5B model: 3 GB of
# weights over GCS FUSE at 0.5 GB/s, then ~60 s of engine start-up. Its startup probe checks
# `/health` every 10 s. Compute `needed_s` (what the probe must cover) and `threshold`, the
# smallest `failureThreshold` that covers it **twice over** (weights on a cold bucket are
# slower). Is the committed manifest's threshold enough? Set `manifest_ok`.

# %% exercise
parts = capacity.cold_start_s(weights_gb=3, weights_gbps=0.5, engine_init_s=60)
probe = files["40-serving-vllm-gcsfuse.yaml"][1][2]["spec"]["template"]["spec"]["containers"][0]["startupProbe"]
### BEGIN SOLUTION
needed_s = parts["weights"] + parts["engine init"]
threshold = math.ceil(2 * needed_s / 10)
manifest_ok = probe["failureThreshold"] * probe["periodSeconds"] >= 2 * needed_s
### END SOLUTION

# %% check
assert needed_s == 66 and threshold == 14 and manifest_ok is True
assert capacity.startup_budget_ok(parts, probe["failureThreshold"] * probe["periodSeconds"])
print(f"✅ the probe must cover {needed_s:.0f} s; 14 x 10 s would do, the manifest allows "
      f"{probe['failureThreshold'] * probe['periodSeconds']} s")

# %% [markdown]
# ## Live (T3): look at the real objects
# After `terraform apply` and `gcloud container clusters get-credentials`, these read-only
# commands show the pools, the accelerator labels and taints, the ComputeClass and Kueue state.
# Offline they are printed, not run.

# %%
import shutil
import subprocess

commands = [
    ["kubectl", "get", "nodes", "-L", "cloud.google.com/gke-nodepool,cloud.google.com/gke-accelerator,cloud.google.com/gke-spot"],
    ["kubectl", "get", "computeclasses"],
    ["kubectl", "get", "clusterqueues,workloads,provisioningrequests", "-A"],
]
ctx = ""
if shutil.which("kubectl"):
    ctx = subprocess.run(["kubectl", "config", "current-context"], capture_output=True, text=True).stdout.strip()
for cmd in commands:
    print("$", " ".join(cmd))
    if ctx.startswith("gke_"):
        print(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout)
if not ctx.startswith("gke_"):
    print("\n(no GKE context: see deploy/gcp/README.md to create the cluster; `terraform destroy` when done)")

# %% [markdown]
# ## In a design review
# *"We need 16 H100s for three days next week, and a small L4 serving fleet — how do we get the
# capacity?"* — in two minutes: serving runs on on-demand (or a reservation) with Spot only for
# surplus replicas, expressed as a ComputeClass so a stockout falls back instead of paging
# someone. The training gang must start whole, so queue it: Kueue reserves quota, files a
# ProvisioningRequest, and DWS flex-start delivers all nodes at once for up to 7 days; if the
# date is fixed, a reservation (or calendar mode) is the only guarantee. Spot is right for
# checkpointed work, and its real price is the interruption rate times nodes times lost work.
# Budget cold start explicitly: image streaming and a weights path at hundreds of MB/s, and a
# startup probe that covers the load.
#
# **Drill 1.** *Why not run the 8-node training job on Spot to save 65 %?* Without frequent
# checkpoints the gang restarts from zero on every reclaim; at 0.16 interruptions/hour a 72-hour
# job essentially never finishes.
#
# **Drill 2.** *What does Kueue add on top of a flex-start node pool?* All-or-nothing admission:
# the ProvisioningRequest asks for every node of the gang, and the job is admitted only when all
# exist — no partial gang holding GPUs.
#
# **Drill 3.** *A serving pod restarts every few minutes on a fresh node.* The liveness probe
# kills it during the weight load: add or size a startup probe.
