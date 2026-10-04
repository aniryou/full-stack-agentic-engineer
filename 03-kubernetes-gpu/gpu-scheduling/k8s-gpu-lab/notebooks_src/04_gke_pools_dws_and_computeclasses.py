# %% [markdown]
# # 04 · GKE: node pools, DWS flex-start and ComputeClasses — capacity is something you ask for
#
# **Tier:** T3 walkthrough. The cluster is GKE, from `deploy/gcp/terraform`. A Spot L4 node costs
# approximately $0.25/h while a pod needs it (verify). Offline, everything here is **plan and inspect**.
# Read the Terraform, calculate its price, model the capacity choices, and render the manifests. With a
# GKE `kubectl` context, the last section examines the live objects.
#
# ## The one-minute version
# On a laptop, the GPUs are there. In a cloud, you must **get** them. There are four ways, and each
# way balances the price against the chance that you get the capacity when you need it:
#
# | how | price | you get | good for |
# |---|---|---|---|
# | on-demand | list | what is in stock (scarce shapes stock out) | serving, short jobs |
# | Spot | 60-91 % off | the cloud can reclaim it with ~30 s notice | batch jobs with checkpoints, more replicas |
# | DWS flex-start (queued) | discounted (verify) | *all N nodes at once* after a queue, ≤ 7 days | gangs of scarce GPUs |
# | reservation / calendar | list or committed | guaranteed, billed when idle | deadlines, steady load |
#
# GKE turns that table into objects with these three mechanisms:
#
# - a node pool that scales from zero,
# - a **ComputeClass** that falls back from Spot to on-demand, then to flex-start,
# - **Kueue + ProvisioningRequest** (all-or-nothing provisioning *before* admission).
#
# Startup latency is the other cost of "from zero". It contains the node boot, the driver, the image
# and the weights. With time-sharing, one GPU serves several small pods. The primer sections are
# §7 *Getting capacity*, §8 *Startup latency*, §9 *Sharing GPUs at the cluster level* and
# §2 *GPU Operator vs managed drivers*.

# %%
from k8sgpu import capacity, gke
from k8sgpu import manifests as m

v = gke.effective_vars()          # variables.tf defaults + terraform.tfvars.example
print(gke.plan_summary(v))

# %% [markdown]
# ## What the Terraform builds
# The Terraform builds one zonal GKE Standard cluster. It has a system pool, the Spot L4 pool, and two
# optional pools (DWS flex-start, time-sharing). The GPU pools carry the
# `nvidia.com/gpu=present:NoSchedule` taint. They let GKE install the driver
# (`gpu_driver_installation_config`). Thus, no GPU Operator is necessary (primer §2).
#
# The driver is `LATEST`, not `DEFAULT`, for a reason that you can examine. The serving image,
# `vllm/vllm-openai:v0.30.0`, is a **CUDA 13.0.2** build (its image config says
# `CUDA_VERSION=13.0.2`). CUDA 13.0 needs an **R580+** driver (>= 580.65.06 on Linux). The
# compatibility rules are in layer 02's primer §1.2, `02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md`.
# An older branch fails at start-up with *"CUDA driver version is insufficient"*. The image leaves
# CUDA forward compatibility off (`VLLM_ENABLE_CUDA_COMPATIBILITY=0`).
#
# The branch that GKE's `DEFAULT` installs depends on the GKE version (verify). The next cell also shows the same cluster in `gcloud`.
# Most GKE documentation shows a cluster in this form:

# %%
for f, rtype, name in gke.tf_resources():
    print(f"{f:16} {rtype}.{name}")
print()
for cmd in gke.gcloud_equivalents({**v, "enable_flex_start_pool": True, "enable_time_sharing_pool": True}):
    print("$", cmd, "\n")

# %% [markdown]
# ## Spot, in numbers
# A gang has an interruption when the cloud reclaims **any** of its nodes. Thus, the rates add:
# 8 nodes at 0.02 reclaims per node-hour = 0.16 per hour. This rate and the 0.005 per node-hour of
# primer §7.2 are both illustrative. Real reclaim rates change with the zone, the shape and the hour.
# Each interruption costs the work since the last checkpoint, plus a restart.
#
# Let the interruptions be Poisson at rate $\lambda$, with checkpoints every $\tau$ hours and a restart
# cost $R$. Then one $\tau$-hour segment takes $(1/\lambda + R)(e^{\lambda\tau} - 1)$ hours on average.
# A job of $W$ hours has $W/\tau$ segments. This model is the gang formula of primer §7.2, with checkpoints added.
#
# ## Exercise 4.1 — expected runtime on Spot
# Write `runtime_h(work_h, rate_per_h, checkpoint_every_h, restart_h)`. For no checkpoints, use
# `checkpoint_every_h=None`. In that case, the job is one segment of `work_h`. Then calculate `slowdown`. A 10 h,
# 8-node job runs on Spot ($\lambda$ = 0.16/h, $R$ = 0.25 h). `slowdown` is how many times longer
# the job takes **without** checkpoints than with hourly checkpoints.

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
# A checkpoint is not free. Thus, the rule "checkpoint more often" has an optimum. If one checkpoint
# takes $C$ hours to write, each segment needs $\tau + C$ hours with no interruption. The formula
# becomes $(1/\lambda + R)(e^{\lambda(\tau + C)} - 1)$
# (`capacity.expected_runtime_h(..., checkpoint_cost_h=C)`). Frequent checkpoints lose time to the
# writes. Rare checkpoints lose work at each reclaim.
#
# The minimum is near Young's interval, $\sqrt{2C/\lambda}$. Layer 01's
# [primer §7.2 *How often to checkpoint: Young/Daly*](../../../../01-hardware-gpu-fabric/roofline-and-fabric/PRIMER.md)
# derives it for large jobs (`roofline.reliability.young_daly_interval`). The next cell uses
# $C$ = 3 min and $\lambda$ = 0.16/h:

# %%
C = 0.05
for tau in (0.25, 0.5, capacity.young_interval_h(C, 0.16), 1.0, 2.0, 4.0):
    print(f"checkpoint every {tau:4.2f} h -> {capacity.expected_runtime_h(10, 0.16, tau, 0.25, C):6.2f} h wall time")
print(f"Young's interval sqrt(2C/λ) = {capacity.young_interval_h(C, 0.16):.2f} h")

# %% [markdown]
# ## Choosing a capacity type
# `capacity.evaluate()` shows the options next to each other for one need. For each option, it gives:
#
# - the expected wait (queues and retries),
# - the expected runtime (interruptions),
# - the cost,
# - what breaks,
# - with a deadline, the **probability that the capacity arrives in time** (`on time`).
#
# The wait in a queue is not a number but a distribution. The model uses an exponential distribution
# for the DWS wait, with the mean of the option. The default mean is 1 h, an illustrative value. For
# 64 H100s, the mean can be much longer.
#
# The model ranks the options with these criteria, in this order:
#
# 1. feasible,
# 2. on time with at least 95 % probability,
# 3. not an interruptible server,
# 4. the lowest cost.
#
# If two options have equal costs, the more certain start wins, and then the shorter wait.
# `capacity.defensible()` lists every option that the model cannot tell apart from the best option.
# All prices and rates are assumptions, marked *verify*. Change them and look at how the ranking moves.

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
# The 3-day pretraining needs 72 h of work and must finish within 96 h. Thus, the capacity has 24 h of
# slack to arrive. Write `p_start_within(slack_h, mean_wait_h)`. It returns the probability that an
# exponential wait with that mean is at most `slack_h`. Then calculate `max_mean_wait`. It is the
# largest mean queue wait that still gives a start within 24 h with 95 % probability.

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
# Look at the same need if the flex-start queue for 64 H100s has a mean wait of 12 h, not 1 h. Then
# the lowest-cost option is no longer sufficient. The answer becomes capacity that you hold: a
# reservation, or DWS calendar mode that you book in advance (verify its terms).

# %%
import dataclasses
slow_queue = dict(capacity.OPTIONS, **{"flex-start": dataclasses.replace(capacity.OPTIONS["flex-start"], wait_h=12.0)})
print(capacity.format_evaluations(capacity.evaluate(needs["pretrain-3d"], slow_queue)))
print("defensible:", capacity.defensible(needs["pretrain-3d"], options=slow_queue))

# %% [markdown]
# ## Exercise 4.3 — pick one per workload, and say why
# Use the tables of the section *Choosing a capacity type* (default assumptions). For each workload,
# set `choice` to one of `"on-demand"`, `"spot"`, `"flex-start"`, `"reservation"`. Give the reason in
# `why`.
#
# The 3-day pretraining cannot write checkpoints. This is an unusual constraint, and it is hard
# on purpose. The job must also finish within 96 hours. If the model gives a tie, both answers pass.
# The reason is the important part.

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
# A custom **ComputeClass** replaces one node pool per capacity type. It lists the ways to get a node,
# in order. GKE's node-pool auto-creation tries them from top to bottom. With `activeMigration`, it
# also moves pods back up the ladder when the preferred capacity comes back. A pod selects the class
# with `nodeSelector: {cloud.google.com/compute-class: <name>}`. The field names come from GKE's CRD
# (verify with `kubectl explain computeclass.spec`).
#
# On each rung, `gpu.driverVersion: latest` is important. The pools that the class *creates* do not
# get the `LATEST` setting of the Terraform pools. Without `gpu.driverVersion: latest`, these pools get GKE's default branch.
# It is possible that this branch is too old for the CUDA 13 vLLM image that runs on them. The field
# name is as in Google's own ComputeClass examples (verify).

# %%
files = gke.gke_manifests()
cc = files["30-computeclass-l4.yaml"][1][0]
print(m.to_yaml(cc))

# %% [markdown]
# ## Exercise 4.4 — which rung provisions?
# Write `pick_rung(priorities, available)`. It returns the index of the first priority whose capacity
# type is available. The capacity type of a priority is:
#
# - `"reservation"` if it has `reservations`,
# - `"flex-start"` if `flexStart.enabled`,
# - if not, `"spot"` or `"on-demand"`, from `spot`.
#
# If no capacity type is available, it returns `None`. With `whenUnsatisfiable: DoNotScaleUp`, the
# pod then stays Pending.

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
# Admission means that the quota is yours, and the autoscaler adds nodes one at a time. Kueue's
# `waitForPodsReady` (on by default in v0.19, 30 min) evicts a half-started gang and puts it back in
# the queue (notebook 02).
#
# `l4-flex` has the `dws-prov` AdmissionCheck. After Kueue reserves the quota, it creates
# **one ProvisioningRequest** (`queued-provisioning.gke.io`). This request covers every pod of the
# Workload (one PodTemplate per pod set). Kueue admits the Workload only when DWS has created all the
# nodes together. Thus, no half-started gang holds GPUs that it cannot use.
#
# Neither flavor sets `topologyName`. Kueue TAS reads placement labels
# (`cloud.google.com/gce-topology-{block,subblock,host}`). GCE publishes these labels for the
# accelerator-optimized families. Google's own TAS examples use them on A3, A4 and A4X node pools
# (GoogleCloudPlatform/cluster-toolkit `examples/gke-a3-*`, `gke-a4`, `gke-a4x`), and none on G2/L4.
# Thus, the L4 flavors of the lab are plain quota. The topology labels on the fake L4 nodes of the
# kind lab are there only to teach TAS.
#
# Examine your nodes with `kubectl get nodes -L cloud.google.com/gce-topology-host` (verify). On
# A3/A4 pools, the TAS flavor of the kind lab carries over with the real labels.

# %%
print(m.to_yaml(*files["10-kueue-gke.yaml"][1][3:6]))
for t, event in capacity.queued_provisioning_timeline(nodes=2, wait_h=0.5, run_h=1.0):
    print(f"t={t:5.2f} h  {event}")

# %% [markdown]
# ## Sharing one GPU between pods (primer §9)
# A 1.5B model on an L4 uses a fraction of the GPU. A dev notebook uses less. With
# `enable_time_sharing_pool = true`, the Terraform adds `l4-shared`, a Spot L4 pool. Its nodes
# advertise **each physical GPU as `max_shared_clients_per_gpu` units** of `nvidia.com/gpu`
# (`gpu_sharing_config`, strategy `TIME_SHARING`). Pods still ask for an integer `nvidia.com/gpu: 1`.
# They select the pool through GKE's node labels `cloud.google.com/gke-gpu-sharing-strategy` and
# `cloud.google.com/gke-max-shared-clients-per-gpu` (verify).
#
# The pods share the SM time, through context switches. The pods do not get memory limits or fault
# isolation. One pod can take all 24 GB. On a VM that you run yourself, the time-slicing config of the
# NVIDIA device plugin does the same (`deploy/gpu-vm/`, T1).

# %%
print(m.to_yaml(files["50-time-sharing-l4.yaml"][1][0])[:900], "...")

# %% [markdown]
# ## Exercise 4.5 — what does a time-shared node advertise?
# Calculate `alloc_1`. It is the allocatable `nvidia.com/gpu` on one `g2-standard-4` in the
# `l4-shared` pool, with 4 clients per GPU. Then calculate `alloc_2`, the same value on a
# `g2-standard-24`. This machine has 2 L4s.
#
# Then find `pods_on_one_node`. It is the number of pods of `50-time-sharing-l4.yaml` that run at the
# same time on one `g2-standard-4`. Each pod asks for 1 GPU, 500m CPU and 512Mi. Also find
# `max_per_container`. It is the largest number of shared GPUs that one container can request there.

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
# A serving pod that is Pending on a pool at zero waits for these items:
#
# - a VM (minutes for GPU shapes),
# - the GPU driver,
# - the image (`vllm/vllm-openai:v0.30.0` is 8.7 GB compressed for amd64, from its registry manifest
#   on 2026-09-26),
# - the weights,
# - the engine start-up (what an engine does before it serves: serving-engine primer §1,
#   `04-inference-engine/serving-engine/PRIMER.md`).
#
# The defaults in the next cell are illustrative. They are different from the example of primer §8 on
# purpose. Image streaming gets the metadata, then starts the container. GCS FUSE with parallel
# downloads (or Hyperdisk ML, or a model streamer) moves weights at hundreds of MB/s. Only the last two
# stages occur *after* the container starts. The startup probe must cover these two stages.

# %%
for streaming in (False, True):
    print("image streaming" if streaming else "full image pull", capacity.cold_start_s(image_streaming=streaming))

# %% [markdown]
# ## Exercise 4.6 — size the serving pod's startup probe
# The serving example (`deploy/gke/40-serving-vllm-gcsfuse.yaml`) loads a 1.5B model. It loads 3 GB of
# weights over GCS FUSE at 0.5 GB/s. After that, the engine start-up takes ~60 s. The startup probe
# of the serving pod does a check of `/health` every 10 s.
#
# Calculate `needed_s`, the time that the probe must cover. Then
# calculate `threshold`, the smallest `failureThreshold` that covers it **twice over**. The factor of two is
# necessary because weights on a cold bucket are slower. Is the threshold of the committed manifest sufficient? Set `manifest_ok`.

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
# After `terraform apply` and `gcloud container clusters get-credentials`, the read-only commands in
# the next cell show the pools, the accelerator labels and taints, the ComputeClass and the Kueue
# state. Offline, the cell prints the commands and does not run them.

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
# capacity?"*
#
# The answer in two minutes: the serving fleet runs on on-demand capacity (or a
# reservation), with Spot only for surplus replicas. Put this policy in a ComputeClass. Then a
# stockout causes a fallback, not a page to a person.
#
# The training gang must start whole, so put it in a queue. Kueue reserves quota and creates a
# ProvisioningRequest. Then DWS flex-start supplies all nodes at once, for up to 7 days. This works
# *if* the queue for that shape is short relative to the slack.
#
# A deadline changes the wait in the
# queue into a probability. When the probability is not sufficient, a reservation or calendar mode is
# the only guarantee. For 95 %, a 24 h slack needs a mean wait under ~8 h.
#
# Spot is correct for work that writes checkpoints. Its real price is the interruption rate times the
# nodes times the lost work, with the checkpoint interval near $\sqrt{2C/\lambda}$. Small models share
# a GPU through time-sharing, with no isolation. Make an explicit budget for the cold start:
#
# - image streaming,
# - a weights path at hundreds of MB/s,
# - a driver that is sufficiently new for the CUDA of the image,
# - a startup probe that covers the load.
#
# **Drill 1.** *Why not run the 8-node training job on Spot to save 65 %?* Without frequent
# checkpoints, the gang starts again from zero at each reclaim. At 0.16 interruptions/hour, a 72-hour
# job almost never finishes.
#
# **Drill 2.** *What does Kueue add on top of a flex-start node pool?* All-or-nothing admission. The
# ProvisioningRequest asks for every node of the gang. Kueue admits the job only when all the nodes
# exist. Thus, no partial gang holds GPUs.
#
# **Drill 3.** *A serving pod restarts every few minutes on a fresh node.* The liveness probe stops
# the pod during the weight load. Add a startup probe, or set the correct size for it. If the pod
# instead exits at once with *"CUDA driver version is insufficient"*, the cause is the driver of the
# node. This driver is older than the CUDA of the image: CUDA 13 needs R580+.
