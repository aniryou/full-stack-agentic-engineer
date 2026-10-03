# %% [markdown]
# # 06 · GPU sharing and DCGM on GKE: MIG vs time-slicing, and reading the right metrics
#
# **Tier:** T0: the GKE manifests and Terraform pool definitions of the lab, a bundled dcgm-exporter scrape and
# the alert rules. This notebook evaluates all of them offline. The scrape has **illustrative** values in the
# documented format of the exporter, not values from a real cluster. **T1/T2**: any GPU VM that you control
# (Lambda, a GCP VM, your workstation), where you run dcgm-exporter in Docker and read a real scrape here. On
# a rented A100/H100 VM, partition the GPU with MIG or share it with MPS by hand
# (`deploy/any-gpu/README.md` §6). **T3**: the same manifests and rules on the GKE cluster of the lab
# (`deploy/gcp/terraform`, `deploy/gke/run.sh`), with DCGM metrics in Cloud Monitoring.
#
# ## The one-minute version
#
# * **Share a GPU only for many small things** (notebooks, small models, low-QPS endpoints). The continuous
#   batching of an LLM engine already shares the weight reads of the GPU across requests. Thus the engine
#   shares the GPU better than any split at the GPU level.
# * **MIG** divides an A100/H100-class GPU into hardware partitions. Each partition has its own SMs, L2 and
#   memory. MIG gives isolation and a geometry that does not change. **Time-slicing** lets several processes
#   take turns on a whole GPU. It operates on any GPU and gives no isolation. **MPS** runs kernels from several
#   processes at the same time. It gives better utilisation and weak isolation.
# * On GKE, each mode is a **node-pool setting** (`gpu_partition_size`, `gpu_sharing_config`) together with
#   **node labels** that the pods select. The setting changes how many `nvidia.com/gpu` the node advertises.
# * **`DCGM_FI_DEV_GPU_UTIL` is not utilisation**: it is the fraction of the time in which any kernel ran. For
#   performance, read `SM_ACTIVE`, `SM_OCCUPANCY`, `PIPE_TENSOR_ACTIVE` and `DRAM_ACTIVE`. For health, read XIDs,
#   row remapping and clock-event reasons. Route each alert to its owner.
#
# Concepts: [the primer](../../PRIMER.md) §7 *Sharing a GPU*, §8 *Health and observability*,
# §9 *On GCP and elsewhere*.

# %%
from pathlib import Path

import yaml

from gpurt import dcgm

LAB = Path(dcgm.__file__).resolve().parents[1]
manifests = {p.name: yaml.safe_load(p.read_text()) for p in sorted((LAB / "deploy/gke").glob("0[45]-*.yaml"))}
for name, doc in manifests.items():
    spec = doc["spec"]["template"]["spec"]
    print(f"{name}: kind={doc['kind']} nodeSelector={spec['nodeSelector']} "
          f"limits={spec['containers'][0]['resources']['limits']}")

# %% [markdown]
# ## How a sharing mode reaches a pod
#
# `deploy/gcp/terraform/node_pools.tf` defines the pools (`l4`, `l4x2`, `l4-shared`, `a100-mig`). A
# time-sharing pool with `max_shared_clients_per_gpu = 2` shows each physical GPU as **2** `nvidia.com/gpu`.
# On an A100 40GB, a MIG pool with `gpu_partition_size = "1g.5gb"` shows each GPU as **7**. GKE puts the
# applicable labels on the nodes. A pod selects those labels and goes to those nodes. The pod still asks for
# `nvidia.com/gpu: 1`.
#
# | Mode | GKE node-pool setting | Pod selects | Isolation | GPUs |
# |---|---|---|---|---|
# | MIG | `guest_accelerator.gpu_partition_size` | `cloud.google.com/gke-gpu-partition-size` | memory, cache, SMs, faults | A100, H100 class |
# | time-sharing | `gpu_sharing_config` `TIME_SHARING` | `…/gke-gpu-sharing-strategy: time-sharing`, `…/gke-max-shared-clients-per-gpu` | none (shared memory, turns) | any |
# | MPS | `gpu_sharing_config` `MPS` | `…/gke-gpu-sharing-strategy: mps` | limited | any (verify requirements) |
#
# ## Exercise 6.1 — from a node pool to what the node offers
#
# Write `node_offer(pool)` for pools that have the shape of the Terraform `gpu_pools` entries. Return
# `(gpus_advertised_per_node, node_selector)`. The first value is the count of `nvidia.com/gpu` that a node
# advertises. The second value is the set of labels that a pod must select. `MIG_A100_40GB` gives the number of MIG
# instances per A100 40GB GPU for each profile (verify for your GPU).

# %% exercise
MIG_A100_40GB = {"1g.5gb": 7, "2g.10gb": 3, "3g.20gb": 2, "7g.40gb": 1}


def node_offer(pool: dict) -> tuple[int, dict]:
    ### BEGIN SOLUTION
    selector = {"cloud.google.com/gke-accelerator": pool["gpu_type"]}
    per_gpu = 1
    if pool.get("time_sharing_clients"):
        per_gpu = pool["time_sharing_clients"]
        selector["cloud.google.com/gke-gpu-sharing-strategy"] = "time-sharing"
        selector["cloud.google.com/gke-max-shared-clients-per-gpu"] = str(pool["time_sharing_clients"])
    if pool.get("partition_size"):
        per_gpu = MIG_A100_40GB[pool["partition_size"]]
        selector["cloud.google.com/gke-gpu-partition-size"] = pool["partition_size"]
    return pool["gpu_count"] * per_gpu, selector
    ### END SOLUTION

# %% check
l4x2 = {"gpu_type": "nvidia-l4", "gpu_count": 2, "time_sharing_clients": None, "partition_size": None}
shared = {"gpu_type": "nvidia-l4", "gpu_count": 1, "time_sharing_clients": 2, "partition_size": None}
mig = {"gpu_type": "nvidia-tesla-a100", "gpu_count": 1, "time_sharing_clients": None, "partition_size": "1g.5gb"}
assert node_offer(l4x2) == (2, {"cloud.google.com/gke-accelerator": "nvidia-l4"})
assert node_offer(shared) == (2, manifests["04-time-sharing.yaml"]["spec"]["template"]["spec"]["nodeSelector"])
assert node_offer(mig) == (7, manifests["05-mig.yaml"]["spec"]["template"]["spec"]["nodeSelector"])
print("✅ pool settings -> advertised nvidia.com/gpu -> the selectors the lab's manifests already use")

# %% [markdown]
# The scheduler sees only the *count*. Two time-shared pods on one L4 each think that they have a GPU. They
# share its 24 GB and take turns on its SMs. If both pods are busy, each pod gets approximately half the
# throughput. Each pod also gets a latency increase from context switches. If one pod leaks memory, the next
# allocation of the other pod fails.
#
# MIG slices do not have an effect on each other in that way, because each slice has its own memory and SMs.
# But a `1g.5gb` slice has 1/7 of the compute.
#
# ## The utilisation paradox, in numbers
#
# Some kernels have only a few blocks, and the code launches them back to back. Such a kernel keeps "a kernel
# running" 100 % of the time, so `GPU_UTIL` reads 100 %. The block scheduler distributes the blocks across the
# SMs. Thus at most `blocks` SMs have anything to do.
#
# ## Exercise 6.2 — estimate SM_ACTIVE
#
# Write `sm_active(blocks, sms)`. It returns the fraction of SMs that have at least one block when a kernel of
# `blocks` blocks runs continuously. The maximum value is 1.0. Calculate the value for 8 blocks on a 132-SM
# H100 and for 4 blocks on a 58-SM L4.

# %% exercise
def sm_active(blocks: int, sms: int) -> float:
    ### BEGIN SOLUTION
    return min(1.0, blocks / sms)
    ### END SOLUTION

# %% check
assert abs(sm_active(8, 132) - 0.0606) < 1e-3
assert abs(sm_active(4, 58) - 0.069) < 1e-3
assert sm_active(1000, 58) == 1.0
print("✅ GPU_UTIL would read 100 % in both cases while 6-7 % of the SMs work: graph SM_ACTIVE, not GPU_UTIL")

# %% [markdown]
# ## A DCGM scrape, four GPUs, four lessons
#
# `gpurt.dcgm` does these steps:
#
# * It parses the Prometheus text of the exporter.
# * It puts the text into groups, one group for each GPU.
# * It calculates signals from the values.
# * It gives a first interpretation of the values.
# * It sorts the XIDs by **owner** (application / node / hardware).
# * It evaluates the alert rules of the lab.

# %%
text = (Path(dcgm.__file__).parent / "fixtures" / "dcgm_exporter_sample.prom").read_text()
print("\n".join(text.splitlines()[:3]))
print()
print(dcgm.report(text))

# %% [markdown]
# This is how to read the four GPUs:
#
# * **node-a** is an LLM decode server. Its DRAM is busy, and its tensor pipes are idle most of the time. Its
#   frame buffer is 89 % full because the engine pre-allocates its KV cache. Thus memory-used is *not* a health
#   signal. The KV-usage metric of the engine itself is a health signal.
# * **node-b** reads 97 % "utilised" with 9 % of its SMs active. The cause is small kernels, small batches or a
#   launch-bound GPU. Batch more, or capture CUDA Graphs (notebook 02).
# * **node-c** is a GPU that an idle notebook holds after an application XID.
# * **node-d** trains at full load. But it had a thermal slowdown, and it has a remapped row that waits for a
#   reset.
#
# ## Exercise 6.3 — decode clock-event reasons, and why PromQL needs floor and modulo
#
# `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` is the bitmask of NVML. Write `reasons(mask)`. It returns the names from
# `dcgm.THROTTLE_BITS` whose bits are set, from the lowest bit to the highest bit. Then write
# `promql_bit_set(mask, bit)`. It examines one bit with only division, floor and modulo, because PromQL has no
# bitwise AND.

# %% exercise
def reasons(mask: int) -> list[str]:
    ### BEGIN SOLUTION
    return [name for bit, name in sorted(dcgm.THROTTLE_BITS.items()) if mask & bit]
    ### END SOLUTION


def promql_bit_set(mask: int, bit: int) -> bool:
    ### BEGIN SOLUTION
    return (mask // bit) % 2 > 0
    ### END SOLUTION

# %% check
for mask in range(512):
    assert reasons(mask) == dcgm.decode_throttle(mask)
    for bit in dcgm.THROTTLE_BITS:
        assert promql_bit_set(mask, bit) == bool(mask & bit)
thermal_rule = next(r for r in dcgm.RULES if r.name == "GpuThermalThrottling")
print("✅ decoded; the deployed rule tests bits 0x20|0x40 as:", thermal_rule.expr)

# %% [markdown]
# ## Exercise 6.4 — who gets woken up?
#
# Each rule has a severity:
#
# * `critical` sends a page (hardware: drain now).
# * `warning` opens a ticket (node operator).
# * `info` sends a notification (application owner, cost).
#
# An XID that is not in the table of the lab goes to the node operator at `warning` (`GpuXidUnknown`). It
# never goes silently to an application team.
#
# The XID rules have one point that is easy to miss. `DCGM_FI_DEV_XID_ERRORS` is a gauge that holds the
# *last* XID that DCGM saw. The gauge does not go back to 0 after a repair of the GPU (verify for your DCGM
# version). If an alert reads only the value, the alert never resolves. Thus each deployed XID rule also has
# the condition `changes(DCGM_FI_DEV_XID_ERRORS[15m]) > 0`. The rule fires when the XID appears, and it
# resolves 15 minutes later.
#
# Because the alert resolves, the drain keeps the state of the node, and the alert does not. One snapshot
# cannot show a change, so `dcgm.evaluate` treats a present XID as recent.
#
# Write `worst_alert(snapshot)`. It returns the most severe severity of the rules that fire for one GPU, or
# `None`. `dcgm.evaluate([snapshot])` gives `(rule, severity, gpu)` tuples.

# %% exercise
RANK = {"critical": 3, "warning": 2, "info": 1}


def worst_alert(snapshot) -> str | None:
    ### BEGIN SOLUTION
    fired = [sev for _, sev, _ in dcgm.evaluate([snapshot])]
    return max(fired, key=RANK.get) if fired else None
    ### END SOLUTION

# %% check
samples, _ = dcgm.parse_prometheus(text)
by_node = {s.host.split("-")[-1]: s for s in dcgm.snapshots(samples)}
assert {k: worst_alert(v) for k, v in by_node.items()} == {"a": None, "b": "info", "c": "info", "d": "warning"}
print("✅ nobody is paged tonight; node-d gets a ticket (thermal + pending row remap); b and c are notifications")

# %% [markdown]
# ## A rule is only as good as the fields behind it
#
# Each rule reads some DCGM fields (`rule.fields`). No exporter configuration exports all of these fields by
# default. The stock dcgm-exporter does not export the clock-event bitmask. It also has `PROF_SM_ACTIVE` and
# `PROF_SM_OCCUPANCY` commented out. The example list of DCGM fields that Google gives for GMP has the profiling
# fields but no XID, row-remap or clock-event fields. The managed package of GKE is similar to this list (verify).
#
# If the exporter never exports the field of a rule, the rule never fires. Nothing tells you that the rule is
# blind: the dashboard is quiet. `dcgm.EXPORTED_BY` records the upstream lists (read 2026-09-26, verify for your
# versions). `deploy/any-gpu/dcgm-counters.csv` is the stock list plus the three fields that the stock list
# does not have.
#
# ## Exercise 6.5 — which rules can this exporter ever fire?
#
# Write `blind_rules(exported)`. It returns the names of the rules in `dcgm.RULES` that read at least one
# field that is not in the set `exported`. Return the names in rule order. Then look at the result for each
# exporter configuration.

# %% exercise
def blind_rules(exported: set) -> list[str]:
    ### BEGIN SOLUTION
    return [r.name for r in dcgm.RULES if not r.fields <= set(exported)]
    ### END SOLUTION

# %% check
for config, fields in dcgm.EXPORTED_BY.items():
    mine = blind_rules(fields)
    assert mine == dcgm.rules_that_cannot_fire(fields), config
    print(f"{config}:\n    cannot fire: {', '.join(mine) or 'none'}")
stock = dcgm.EXPORTED_BY["stock dcgm-exporter (etc/default-counters.csv)"]
assert blind_rules(stock) == ["GpuThermalThrottling", "GpuHardwareSlowdown", "GpuBusyButUnderfilled"]
print("✅ with the stock counters the utilisation paradox is invisible (no SM_ACTIVE) and throttling never alerts")

# %% [markdown]
# ## On a GPU box you control (T1/T2)
#
# `deploy/any-gpu/README.md` §6 runs dcgm-exporter in Docker with the counters CSV of the lab. It saves a
# scrape to `out/dcgm.prom`. On a rented A100/H100 *VM*, it also gives the manual steps for MIG
# (`nvidia-smi mig`) and MPS (`nvidia-cuda-mps-control`). This notebook reads a scrape that you bring back.
# The values of that scrape are measurements from your GPU.

# %%
scrapes = sorted(LAB.glob("out/*.prom")) + sorted(LAB.glob("deploy/*/out/*.prom"))
for path in scrapes:
    print(f"== {path.relative_to(LAB)} (measured on the machine that produced it)")
    print(dcgm.report(path.read_text()))
if not scrapes:
    print("No scrape in out/ yet (T0). On a GPU box: deploy/any-gpu/README.md §6, then re-run this cell.")

# %% [markdown]
# ## On GKE (T3)
#
# If you set `enable_dcgm = true` in the Terraform, GKE runs the exporter. GKE sends the `DCGM_FI_*` metrics
# to Cloud Monitoring through Managed Prometheus. *Metrics explorer* accepts PromQL.
#
# The next code cell prints the rules. `dcgm.rules_manifest()` generates them, and
# `deploy/gke/06-dcgm-alert-rules.yaml` stores them. The rules are a **`ClusterRules`** object, because a
# namespaced GMP `Rules` object evaluates only metrics from its own namespace. Also, the exporter never runs in the
# namespace of the workloads.
#
# The `namespace`/`pod` target labels of the scraper have priority. Thus the workload labels of the exporter
# arrive as `exported_namespace`/`exported_pod` (verify).
#
# Rules only *evaluate*. Alerts that fire go to the managed Alertmanager of GMP, which needs receivers
# (`deploy/gke/alertmanager/alertmanager.example.yaml`). The alerts do not go to Cloud Monitoring alerting policies.
# Also, the managed exporter has its own field list. With that list, it is possible that the health rules are
# blind (verify). See the section *A rule is only as good as the fields behind it*.
#
# ```bash
# cd deploy/gcp/terraform && terraform apply           # enable_time_sharing_pool = true for 04
# $(terraform output -raw get_credentials)
# cd ../../gke && ./run.sh timeshare                   # two pods, one GPU UUID
# ./run.sh rules                                       # ClusterRules, then the Alertmanager secret it prints
# # Metrics explorer, PromQL:  DCGM_FI_PROF_SM_ACTIVE   vs   DCGM_FI_DEV_GPU_UTIL
# ./run.sh clean && cd ../gcp/terraform && terraform destroy
# ```

# %%
print("\n".join(dcgm.rules_manifest().splitlines()[:34]))

# %% [markdown]
# ## In a design review
#
# **Two minutes.** First, I ask if it is necessary to share the GPU at all. An inference engine with
# continuous batching shares a GPU better than any partition of the GPU. Thus GPU sharing is for many small
# things that come in bursts.
#
# If the tenants need isolation (memory, faults, predictable latency), I use MIG on an A100/H100-class GPU. I
# accept that its slice sizes do not change. If the tenants are cooperative dev workloads, time-slicing on any
# GPU is sufficient. MPS is between the two.
#
# On GKE, each mode is a node-pool setting that multiplies the advertised `nvidia.com/gpu`. The pods select the
# labels of the pool.
#
# For monitoring, I never alert on GPU util. I show SM active and occupancy, and tensor and DRAM activity, on
# graphs to see *how* the GPU is busy. I route health signals by owner. Hardware XIDs and remap failures send a
# page. Node-level events open tickets. Application XIDs and idle allocations send a notification.
#
# **Drill questions**
#
# 1. *Two teams want to share one A100 for notebooks. MIG or time-slicing?* Use MIG if one of the teams needs
#    guaranteed memory or protection from the crashes of the other team. For example, each team gets 3g.20gb.
#    Use time-slicing only if both teams accept contention and a shared 40 GB.
# 2. *GPU util is 100 % but throughput is poor. Next step?* Look at SM_ACTIVE and SM_OCCUPANCY. If few SMs are
#    busy, the cause is small kernels, small batches or a launch-bound workload. In that case, batch, fuse, or
#    use CUDA Graphs. Look at DRAM_ACTIVE: if the workload is bandwidth bound, move fewer bytes. Look at
#    PIPE_TENSOR_ACTIVE to see if the tensor cores are in use at all.
# 3. *XID 79 at 3 am on one node. Who acts, and how?* This is a hardware fault: the GPU "fell off the bus". The
#    automation drains the node (cordon, evict). The on-call engineer sends a page to the infrastructure team.
#    Then the node gets a reboot and a diagnosis (`dcgmi diag`). If the fault occurs again, replace the
#    hardware. The application team only needs the scheduler to schedule its pods again.
# 4. *The thermal-throttling alert has never fired in a year. Good news?* Find out if the exporter exports
#    `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` at all. The stock dcgm-exporter does not export it, so the rule is
#    blind. Every rule needs its fields in the counters CSV (`dcgm.rules_that_cannot_fire`).
