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
# image, weights — is the other cost of "from zero", and time-sharing is how one GPU serves
# several small pods. Primer §7 *Getting capacity*, §8 *Startup latency*, §9 *Sharing GPUs at the
# cluster level*, §2 *GPU Operator vs managed drivers*.

# %%
from k8sgpu import capacity, gke
from k8sgpu import manifests as m

v = gke.effective_vars()          # variables.tf defaults + terraform.tfvars.example
print(gke.plan_summary(v))

# %% [markdown]
# ## What the Terraform builds
# One zonal GKE Standard cluster: a system pool, the Spot L4 pool, and two optional pools (DWS
# flex-start; time-sharing). The GPU pools carry the `nvidia.com/gpu=present:NoSchedule` taint and
# let GKE install the driver (`gpu_driver_installation_config`) — no GPU Operator needed (primer
# §2). The driver is `LATEST`, not `DEFAULT`, for a reason you can check: the serving image,
# `vllm/vllm-openai:v0.30.0`, is a **CUDA 13.0.2** build (its image config says
# `CUDA_VERSION=13.0.2`), and CUDA 13.0 needs an **R580+** driver (>= 580.65.06 on Linux; the
# compatibility rules are layer 02's primer §1.2, `02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md`).
# An older branch fails at start-up with *"CUDA driver version is insufficient"*; the image leaves
# CUDA forward compatibility off (`VLLM_ENABLE_CUDA_COMPATIBILITY=0`). Which branch GKE's `DEFAULT`
# installs depends on the GKE version (verify). The same cluster in `gcloud`, which is how most GKE
# documentation shows it:

# %%
for f, rtype, name in gke.tf_resources():
    print(f"{f:16} {rtype}.{name}")
print()
for cmd in gke.gcloud_equivalents({**v, "enable_flex_start_pool": True, "enable_time_sharing_pool": True}):
    print("$", cmd, "\n")

# %% [markdown]
# ## Spot, in numbers
# A gang is interrupted when **any** of its nodes is reclaimed, so the rates add:
# 8 nodes at 0.02 reclaims per node-hour = 0.16 per hour. (Both this rate and the 0.005 per
# node-hour of primer §7.2 are illustrative; real reclaim rates vary by zone, shape and hour.)
# Each interruption loses the work since the last checkpoint plus a restart. With Poisson
# interruptions at rate λ, checkpoints every τ hours and restart cost R, one τ-hour segment takes
# on average `(1/λ + R)(e^(λτ) - 1)` hours, and a job of W hours has W/τ segments — primer §7.2's
# gang formula with checkpoints added.
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
# Checkpoints are not free, which is why "checkpoint more often" has an optimum. If writing one
# takes C hours, each segment needs τ + C uninterrupted hours: `(1/λ + R)(e^(λ(τ + C)) - 1)`
# (`capacity.expected_runtime_h(..., checkpoint_cost_h=C)`). Frequent checkpoints waste time
# writing; rare ones waste work on every reclaim. The minimum sits near Young's interval
# √(2C/λ) — derived for large jobs in layer 01's
# [primer §7.2 *How often to checkpoint: Young/Daly*](../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)
# (`roofline.reliability.young_daly_interval`). With C = 3 min and λ = 0.16/h:

# %%
C = 0.05
for tau in (0.25, 0.5, capacity.young_interval_h(C, 0.16), 1.0, 2.0, 4.0):
    print(f"checkpoint every {tau:4.2f} h -> {capacity.expected_runtime_h(10, 0.16, tau, 0.25, C):6.2f} h wall time")
print(f"Young's interval sqrt(2C/λ) = {capacity.young_interval_h(C, 0.16):.2f} h")

# %% [markdown]
# ## Choosing a capacity type
# `capacity.evaluate()` puts the options side by side for one need: expected wait (queues and
# retries), expected runtime (interruptions), cost, what breaks — and, with a deadline, the
# **probability the capacity arrives in time** (`on time`). A queue's wait is not a number but a
# distribution; the model treats DWS's as exponential with the option's mean (1 h by default,
# illustrative — for 64 H100s it can be far longer). Ranking: feasible, on time with at least
# 95 % probability, not an interruptible server, then cheapest; equal costs are broken by the
# surer start and then the shorter wait, and `capacity.defensible()` lists every option the model
# cannot tell apart from the best. All prices and rates are assumptions marked *verify* — change
# them and watch the ranking move.

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
# ## Exercise 4.2 — will the queue deliver in time?
# The 3-day pretraining needs 72 h of work and must finish within 96 h: 24 h of slack for the
# capacity to arrive. Write `p_start_within(slack_h, mean_wait_h)`: the probability an
# exponential wait with that mean is at most `slack_h`. Then compute `max_mean_wait`: the largest
# mean queue wait for which the start is still 95 % likely within 24 h.

# %% exercise
def p_start_within(slack_h: float, mean_wait_h: float) -> float:
    ### BEGIN SOLUTION
    return 1.0 - math.exp(-slack_h / mean_wait_h) if slack_h >= 0 else 0.0
    ### END SOLUTION

### BEGIN SOLUTION
max_mean_wait = 24 / math.log(1 / (1 - 0.95))       # 1 - e^(-24/W) = 0.95  ->  W = 24 / ln 20
### END SOLUTION

# %% check
assert math.isclose(p_start_within(24, 1.0), capacity.p_capacity_within(capacity.OPTIONS["flex-start"], 24))
assert math.isclose(p_start_within(24, 12.0), 0.8647, rel_tol=1e-4)
assert math.isclose(max_mean_wait, 8.0114, rel_tol=1e-4) and math.isclose(p_start_within(24, max_mean_wait), 0.95)
print(f"✅ a DWS queue averaging more than {max_mean_wait:.1f} h makes a 24 h-slack deadline a coin you should not flip")

# %% [markdown]
# The same need, if the flex-start queue for 64 H100s averaged 12 h instead of 1 h: the cheapest
# option is no longer good enough, and the answer becomes capacity you hold (a reservation, or
# DWS calendar mode booked ahead — verify its terms).

# %%
import dataclasses
slow_queue = dict(capacity.OPTIONS, **{"flex-start": dataclasses.replace(capacity.OPTIONS["flex-start"], wait_h=12.0)})
print(capacity.format_evaluations(capacity.evaluate(needs["pretrain-3d"], slow_queue)))
print("defensible:", capacity.defensible(needs["pretrain-3d"], options=slow_queue))

# %% [markdown]
# ## Exercise 4.3 — pick one per workload, and say why
# Using the tables above (default assumptions), set `choice` to one of `"on-demand"`, `"spot"`,
# `"flex-start"`, `"reservation"` for each workload, and give the reason in `why`. The 3-day
# pretraining cannot checkpoint (an unusual, deliberately hard constraint) and must finish within
# 96 hours. Where the model ties, either answer passes — the reason is the part that matters.

# %% exercise
choice = {"serve-24x7": None, "finetune-10h": None, "pretrain-3d": None}
why = {"serve-24x7": "", "finetune-10h": "", "pretrain-3d": ""}
### BEGIN SOLUTION
choice = {
    "serve-24x7": "reservation",    # same list price as on-demand here, but the capacity is held: no stockout at 3 a.m.
    "finetune-10h": "spot",         # checkpointed, 1 node: the interruption tax is small, the discount large
    "pretrain-3d": "flex-start",    # 8 nodes all-or-nothing, no checkpoints (Spot restarts forever), fits a 7-day lease
}
why = {
    "serve-24x7": "interruptions are outages, so not Spot; a 24x7 load uses every reserved hour, and a "
                  "committed-use discount (verify) would make it cheaper than on-demand",
    "finetune-10h": "hourly checkpoints keep Spot's interruption tax to ~1.5% of runtime at 65% off",
    "pretrain-3d": "an 8-node gang must start whole: queued all-or-nothing provisioning at half price, and "
                   "at a 1 h mean wait it starts within the 24 h slack with near certainty (above ~8 h: reserve)",
}
### END SOLUTION

# %% check
for label, need in needs.items():
    ok = capacity.defensible(need)
    assert choice[label] in ok, (label, choice[label], "defensible:", ok)
    assert len(why[label].split()) >= 5, f"say why for {label}"
spot = capacity.evaluate_option(needs["pretrain-3d"], capacity.OPTIONS["spot"])
print(f"✅ defensible picks: {choice}. 8-node no-checkpoint Spot would need ~{spot.run_h:,.0f} h for 72 h of work")

# %% [markdown]
# ## ComputeClass: the fallback ladder as an object
# Instead of one node pool per capacity type, a custom **ComputeClass** lists ways to get a
# node, in order; GKE's node-pool auto-creation tries them top to bottom and (with
# `activeMigration`) moves pods back up the ladder when the preferred capacity returns. Pods
# opt in with `nodeSelector: {cloud.google.com/compute-class: <name>}`. Field names follow
# GKE's CRD (verify with `kubectl explain computeclass.spec`). Each rung's `gpu.driverVersion:
# latest` matters here: pools the class *creates* do not inherit the Terraform pools' `LATEST`,
# and without it they would get GKE's default branch — too old, possibly, for the CUDA 13 vLLM
# image that runs on them (field name as in Google's own ComputeClass examples; verify).

# %%
files = gke.gke_manifests()
cc = files["30-computeclass-l4.yaml"][1][0]
print(m.to_yaml(cc))

# %% [markdown]
# ## Exercise 4.4 — which rung provisions?
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
# `deploy/gke/10-kueue-gke.yaml` gives the ClusterQueue two flavors. `l4-spot` is plain quota:
# admission means the quota is yours, and the autoscaler brings nodes one at a time — Kueue's
# `waitForPodsReady` (on by default in v0.19, 30 min) evicts and requeues a gang caught
# half-started (notebook 02). `l4-flex` has the `dws-prov` AdmissionCheck: after reserving quota,
# Kueue files **one ProvisioningRequest** (`queued-provisioning.gke.io`) covering every pod of the
# Workload (one PodTemplate per pod set), and admits only when DWS has created all the nodes
# together — no half-started gang holding GPUs hostage.
#
# Neither flavor sets `topologyName`. GCE publishes the placement labels Kueue TAS reads
# (`cloud.google.com/gce-topology-{block,subblock,host}`) for the accelerator-optimized families:
# Google's own TAS examples use them on A3, A4 and A4X node pools (GoogleCloudPlatform/cluster-toolkit
# `examples/gke-a3-*`, `gke-a4`, `gke-a4x`), none on G2/L4. So the lab's L4 flavors are plain quota,
# and the kind lab's topology labels on fake L4 nodes are there only to teach TAS. Check your nodes
# with `kubectl get nodes -L cloud.google.com/gce-topology-host` (verify); on A3/A4 pools the kind
# lab's TAS flavor carries over with the real labels.

# %%
print(m.to_yaml(*files["10-kueue-gke.yaml"][1][3:6]))
for t, event in capacity.queued_provisioning_timeline(nodes=2, wait_h=0.5, run_h=1.0):
    print(f"t={t:5.2f} h  {event}")

# %% [markdown]
# ## Sharing one GPU between pods (primer §9)
# A 1.5B model on an L4 uses a fraction of the GPU; a dev notebook less. With
# `enable_time_sharing_pool = true` the Terraform adds `l4-shared`, a Spot L4 pool whose nodes
# advertise **each physical GPU as `max_shared_clients_per_gpu` units** of `nvidia.com/gpu`
# (`gpu_sharing_config`, strategy `TIME_SHARING`). Pods still ask for an integer `nvidia.com/gpu: 1`
# and select the pool through GKE's node labels `cloud.google.com/gke-gpu-sharing-strategy` and
# `cloud.google.com/gke-max-shared-clients-per-gpu` (verify). What they share: SM time, by
# context switching. What they do not get: memory limits or fault isolation — one pod can take
# all 24 GB. On a VM you run yourself, the NVIDIA device plugin's time-slicing config does the same
# (`deploy/gpu-vm/`, T1).

# %%
print(m.to_yaml(files["50-time-sharing-l4.yaml"][1][0])[:900], "...")

# %% [markdown]
# ## Exercise 4.5 — what does a time-shared node advertise?
# Compute `alloc_1` (allocatable `nvidia.com/gpu` on one `g2-standard-4` in the `l4-shared` pool with
# 4 clients per GPU) and `alloc_2` (the same on a `g2-standard-24`, which has 2 L4s). Then decide
# `pods_on_one_node`: how many pods of `50-time-sharing-l4.yaml` (each 1 GPU, 500m CPU, 512Mi) run
# at once on one `g2-standard-4`, and `max_per_container`: the most shared GPUs one container may
# request there.

# %% exercise
from k8sgpu import machines
### BEGIN SOLUTION
alloc_1 = machines.CATALOG["g2-standard-4"].gpus * 4
alloc_2 = machines.CATALOG["g2-standard-24"].gpus * 4
a4 = machines.allocatable("g2-standard-4")
pods_on_one_node = min(alloc_1, int(a4.cpu // 0.5), int(a4.memory_gib // 0.5))
max_per_container = 1      # two "GPUs" could be two slices of the same GPU: GKE rejects requests > 1
### END SOLUTION

# %% check
assert alloc_1 == machines.time_shared_allocatable("g2-standard-4", 4) == 4
assert alloc_2 == machines.time_shared_allocatable("g2-standard-24", 4) == 8
assert pods_on_one_node == 4 and max_per_container == 1
print(f"✅ one L4 = {alloc_1} schedulable GPUs; all {pods_on_one_node} pods of the Job share it, none may ask for 2")

# %% [markdown]
# ## Startup latency: where the minutes go when you scale from zero
# A Pending serving pod on a pool at zero waits for: a VM (minutes for GPU shapes), the GPU
# driver, the image (`vllm/vllm-openai:v0.30.0` is 8.7 GB compressed for amd64, read from its
# registry manifest on 2026-09-26), the weights, and engine start-up (what an engine does before
# it serves: serving-engine primer §1, `04-inference-engine/serving-engine/PRIMER.md`). The
# defaults below are illustrative and differ from primer §8's example on purpose. Image
# streaming starts the container after fetching metadata; GCS FUSE with parallel downloads (or
# Hyperdisk ML, or a model streamer) moves weights at hundreds of MB/s. Only the last two
# stages happen *after* the container starts — that is what the startup probe must cover.

# %%
for streaming in (False, True):
    print("image streaming" if streaming else "full image pull", capacity.cold_start_s(image_streaming=streaming))

# %% [markdown]
# ## Exercise 4.6 — size the serving pod's startup probe
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
    ["kubectl", "get", "nodes", "-L", "cloud.google.com/gce-topology-host,cloud.google.com/gke-gpu-sharing-strategy"],
    ["kubectl", "get", "nodes", "-o", "custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\\.com/gpu"],
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
# ProvisioningRequest, and DWS flex-start delivers all nodes at once for up to 7 days — *if* the
# queue for that shape is short relative to the slack. A deadline turns the queue's wait into a
# probability; when it is not good enough (24 h slack needs a mean wait under ~8 h for 95 %),
# a reservation or calendar mode is the only guarantee. Spot is right for checkpointed work, and
# its real price is the interruption rate times nodes times lost work, with the checkpoint
# interval near √(2C/λ). Small models share a GPU through time-sharing, with no isolation. Budget
# cold start explicitly: image streaming, a weights path at hundreds of MB/s, a driver new enough
# for the image's CUDA, and a startup probe that covers the load.
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
# kills it during the weight load: add or size a startup probe. (If instead it exits at once with
# *"CUDA driver version is insufficient"*, the node's driver is older than the image's CUDA:
# CUDA 13 needs R580+.)
