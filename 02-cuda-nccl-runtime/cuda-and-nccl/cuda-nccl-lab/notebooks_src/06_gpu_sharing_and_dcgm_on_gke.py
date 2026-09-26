# %% [markdown]
# # 06 · GPU sharing and DCGM on GKE: MIG vs time-slicing, and reading the right metrics
#
# **Tier:** T0 — the lab's GKE manifests and Terraform pool definitions, a bundled dcgm-exporter scrape
# (**illustrative** values in the exporter's documented format, not a real cluster), and the alert rules,
# all evaluated offline. **T1/T2** — any GPU VM you control (Lambda, a GCP VM, your workstation): run
# dcgm-exporter in Docker and read a real scrape here; on a rented A100/H100 VM, partition it with MIG
# or share it with MPS by hand (`deploy/any-gpu/README.md` §6). **T3** — the same manifests and rules on
# the lab's GKE cluster (`deploy/gcp/terraform`, `deploy/gke/run.sh`), with DCGM metrics in Cloud Monitoring.
#
# ## The one-minute version
#
# * **Share a GPU only for many small things** (notebooks, small models, low-QPS endpoints). An LLM
#   engine's continuous batching already shares the GPU's weight reads across requests — better than any
#   GPU-level split.
# * **MIG** cuts an A100/H100-class GPU into hardware partitions (own SMs, L2, memory): isolation, fixed
#   geometry. **Time-slicing** lets several processes take turns on a whole GPU: any GPU, no isolation.
#   **MPS** runs kernels from several processes concurrently: better utilisation, weak isolation.
# * On GKE each is a **node-pool setting** (`gpu_partition_size`, `gpu_sharing_config`) that changes how
#   many `nvidia.com/gpu` the node advertises, plus **node labels** the pods select.
# * **`DCGM_FI_DEV_GPU_UTIL` is not utilisation**: it is the share of time any kernel ran. Read
#   `SM_ACTIVE`, `SM_OCCUPANCY`, `PIPE_TENSOR_ACTIVE`, `DRAM_ACTIVE` for performance; XIDs, row remapping and
#   clock-event reasons for health — and route each alert to its owner.
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
# time-sharing pool with `max_shared_clients_per_gpu = 2` makes each physical GPU appear as **2**
# `nvidia.com/gpu`; a MIG pool with `gpu_partition_size = "1g.5gb"` on an A100 40GB appears as **7**. GKE
# labels the nodes accordingly, and a pod lands there by selecting those labels while still asking for
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
# Write `node_offer(pool)` for pools shaped like the Terraform `gpu_pools` entries. Return
# `(gpus_advertised_per_node, node_selector)`: the count of `nvidia.com/gpu` a node advertises, and the
# labels a pod must select. MIG instances per A100 40GB GPU for each profile are in `MIG_A100_40GB`
# (verify for your GPU).

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
# Scheduling sees only the *count*: two time-shared pods on one L4 each believe they have a GPU, share its
# 24 GB and take turns on its SMs. If both are busy, each gets roughly half the throughput and a latency
# penalty from context switches; if one leaks memory, the other's next allocation fails. MIG slices do not
# interfere that way — each has its own memory and SMs — but a `1g.5gb` slice has 1/7 of the compute.
#
# ## The utilisation paradox, in numbers
#
# A kernel with a handful of blocks, launched back to back, keeps "a kernel running" 100 % of the time,
# so `GPU_UTIL` reads 100 %. The block scheduler spreads blocks across SMs, so at most `blocks` SMs
# have anything to do.
#
# ## Exercise 6.2 — estimate SM_ACTIVE
#
# Write `sm_active(blocks, sms)`: the fraction of SMs with at least one block when a kernel of `blocks`
# blocks runs continuously (capped at 1.0). Evaluate 8 blocks on a 132-SM H100 and 4 blocks on a 58-SM L4.

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
# `gpurt.dcgm` parses the exporter's Prometheus text, groups it per GPU, derives signals, gives a first
# reading, triages XIDs by **owner** (application / node / hardware) and evaluates the lab's alert rules.

# %%
text = (Path(dcgm.__file__).parent / "fixtures" / "dcgm_exporter_sample.prom").read_text()
print("\n".join(text.splitlines()[:3]))
print()
print(dcgm.report(text))

# %% [markdown]
# Reading them: **node-a** is an LLM decode server — DRAM busy, tensor pipes mostly idle, frame buffer
# 89 % full because the engine pre-allocates its KV cache (so memory-used is *not* a health signal; the
# engine's own KV-usage metric is). **node-b** reads 97 % "utilised" with 9 % of its SMs active: small
# kernels or batches, or launch-bound — batch more or capture CUDA Graphs (notebook 02). **node-c** is a
# GPU held by an idle notebook after an application XID. **node-d** trains at full tilt but hit a thermal
# slowdown and has a remapped row waiting for a reset.
#
# ## Exercise 6.3 — decode clock-event reasons, and why PromQL needs floor and modulo
#
# `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` is NVML's bitmask. Write `reasons(mask)` returning the names from
# `dcgm.THROTTLE_BITS` whose bits are set (ascending bit order), and `promql_bit_set(mask, bit)` that
# tests a bit using only division, floor and modulo — PromQL has no bitwise AND.

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
# Each rule has a severity: `critical` pages (hardware: drain now), `warning` opens a ticket (node
# operator), `info` notifies (application owner, cost). Write `worst_alert(snapshot)`: the most severe
# severity among the rules that fire for one GPU (`dcgm.evaluate([snapshot])` gives `(rule, severity,
# gpu)` tuples), or `None`.

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
# Every rule reads some DCGM fields (`rule.fields`), and no exporter configuration exports all of them
# by default. The stock dcgm-exporter leaves out the clock-event bitmask and has `PROF_SM_ACTIVE` and
# `PROF_SM_OCCUPANCY` commented out; Google's GMP DCGM example list — which GKE's managed package
# resembles (verify) — has the profiling fields but no XID, row-remap or clock-event fields. A rule whose
# field is never exported never fires, and nothing tells you: the dashboard is simply quiet.
# `dcgm.EXPORTED_BY` records the upstream lists (read 2026-09-26, verify for your versions);
# `deploy/any-gpu/dcgm-counters.csv` is stock plus the three missing fields.
#
# ## Exercise 6.5 — which rules can this exporter ever fire?
#
# Write `blind_rules(exported)`: the names of the rules in `dcgm.RULES` that read at least one field not
# in the set `exported`, in rule order. Then look at what it says for each exporter configuration.

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
# `deploy/any-gpu/README.md` §6 runs dcgm-exporter in Docker with the lab's counters CSV and saves a
# scrape to `out/dcgm.prom`; on a rented A100/H100 *VM* it also walks through MIG (`nvidia-smi mig`) and
# MPS (`nvidia-cuda-mps-control`) by hand. A scrape you bring back is read here — measured, on your GPU.

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
# With `enable_dcgm = true` in the Terraform, GKE runs the exporter and ships the `DCGM_FI_*` metrics to
# Cloud Monitoring through Managed Prometheus; *Metrics explorer* accepts PromQL. The rules below
# (generated by `dcgm.rules_manifest()`, stored as `deploy/gke/06-dcgm-alert-rules.yaml`) are a
# **`ClusterRules`** object: a namespaced GMP `Rules` object evaluates only metrics from its own namespace,
# and the exporter never runs in the workloads' namespace. The scraper's own `namespace`/`pod` target
# labels win, so the exporter's workload labels arrive as `exported_namespace`/`exported_pod` (verify).
# Rules only *evaluate*: firing alerts go to GMP's managed Alertmanager, which needs receivers
# (`deploy/gke/alertmanager.example.yaml`) — not to Cloud Monitoring alerting policies. And on the managed
# exporter's field list, the health rules may be blind (above; verify).
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
# **Two minutes.** First I ask whether to share at all: an inference engine with continuous batching
# shares a GPU better than any partitioning, so GPU sharing is for many small, bursty things. If the
# tenants need isolation — memory, faults, predictable latency — I use MIG on an A100/H100-class GPU and
# accept its fixed slice sizes; if they are cooperative dev workloads, time-slicing on any GPU is enough;
# MPS sits between. On GKE each is a node-pool setting that multiplies the advertised `nvidia.com/gpu`,
# and pods select the pool's labels. For monitoring I never alert on GPU util: I graph SM active and
# occupancy, tensor and DRAM activity to see *how* the GPU is busy, and I route health signals by owner —
# hardware XIDs and remap failures page, node-level events open tickets, application XIDs and idle
# allocations notify.
#
# **Drill questions**
#
# 1. *Two teams want to share one A100 for notebooks. MIG or time-slicing?* — MIG if either needs
#    guaranteed memory or protection from the other's crashes (e.g. 3g.20gb each); time-slicing only if
#    both accept contention and a shared 40 GB.
# 2. *GPU util is 100 % but throughput is poor. Next step?* — Look at SM_ACTIVE and SM_OCCUPANCY (few SMs
#    busy → tiny kernels, small batches, launch-bound → batch, fuse, CUDA Graphs), DRAM_ACTIVE (bandwidth
#    bound → fewer bytes), PIPE_TENSOR_ACTIVE (are tensor cores used at all?).
# 3. *XID 79 at 3 am on one node. Who acts, and how?* — Hardware: the GPU fell off the bus. Automation
#    drains the node (cordon, evict), the on-call pages infrastructure, the node is rebooted and diagnosed
#    (`dcgmi diag`); recurring → replace. The application team only needs its pods rescheduled.
# 4. *The thermal-throttling alert has never fired in a year. Good news?* — Check that the exporter exports
#    `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` at all: stock dcgm-exporter does not, so the rule is blind. Every
#    rule needs its fields in the counters CSV (`dcgm.rules_that_cannot_fire`).
