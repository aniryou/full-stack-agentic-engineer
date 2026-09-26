# k8s-gpu-lab — Kubernetes for GPUs, hands-on

After this lab you can write GPU manifests that schedule, explain why a GPU pod is Pending, and watch the real
kube-scheduler and Kueue admit, place, preempt and reclaim GPU work — practised on a real control plane with fake GPUs
on a laptop ($0), predicted offline by a small bundled model, run with the real device plugin on one GPU VM you rent
by the hour, and carried to GKE with Terraform when you want the production shape.

The concepts are in the topic primer, [`../PRIMER.md`](../PRIMER.md); this lab cites its sections by number (§1
*What Kubernetes sees* … §10 *Learning locally*). The minimal, standard-library version of the same ideas is
[`../k8s-gpu-core/`](../k8s-gpu-core/); this lab does not import it.

## Start here

1. Install and run the tests (below) — 118 tests, offline, no cluster.
2. `python3 -m k8sgpu kind predict s2` — the predictor's step-by-step outcome for a gang scenario (simulated).
3. Open [`notebooks/01_manifests_and_the_linter.ipynb`](notebooks/01_manifests_and_the_linter.ipynb); with Docker,
   bring up [`deploy/kind`](deploy/kind/README.md) and continue with notebook 02.

## What you get: tiers

| Tier | What you need | Cost | What runs |
|---|---|---|---|
| **T0** | Python 3.10+ (laptop, Colab CPU) | $0 | builders, linter, Pending explainer, capacity model, and the **predictor** for every kind scenario (prints the `kubectl` commands) |
| **T0 + Docker** | a laptop with Docker, [kind v0.33.0](https://kind.sigs.k8s.io/), kubectl >= 1.27 | $0 | `deploy/kind/up.sh`: a 6-node kind cluster, 16 fake `nvidia.com/gpu`, real kube-scheduler, **Kueue v0.19.6**, **JobSet v0.12.0**, **LeaderWorkerSet v0.11.0**; notebooks 02-03 run the scenarios live and compare with the prediction |
| **T1 / T2** | any GPU VM you control (Lambda, a GCP VM, your own box) with the NVIDIA driver and Container Toolkit | the VM's hourly price | `deploy/gpu-vm/`: k3s + the real NVIDIA device plugin + GPU Feature Discovery labels, optional time-slicing; the device path kind cannot show (a pod gets `/dev/nvidia*`) |
| **T3** | a GCP project with billing and L4 quota | ~$0.23/h idle + ~$0.25/h per busy Spot L4 node (verify) | `deploy/gcp/terraform`: zonal GKE Standard, L4 Spot pool 0→N, optional DWS flex-start and time-sharing pools, GCS FUSE, image streaming, managed Prometheus; `deploy/gke/`: Kueue + DWS, ComputeClass, vLLM with GCS FUSE weights, time-sharing |

Every concept is learnable at T0; T1/T2 adds the real device plugin on any provider, and GKE
(T3) is one production target, never a prerequisite. Where GPUs come from and what they cost
across providers: [`COMPUTE.md`](../../../COMPUTE.md). RunPod and Vast.ai rent
containers, not VMs, so the T1 path needs a VM provider (Lambda, GCP, others).

**What is real and what is simulated on kind** (primer §10): the scheduler, Kueue's quota,
cohorts, preemption and Topology-Aware Scheduling, the JobSet and LWS controllers, taints,
labels and every event you will read are real. The GPUs are an extended resource patched into
node status (`deploy/kind/fake-gpus.sh`): the scheduler counts them and the kubelet admits the
pods, but there is no device plugin, no `/dev/nvidia*`, no CUDA — the pods print as much.

## Run it

```bash
cd 03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                               # 118 tests, ~5 s, offline
python3 -m k8sgpu kind predict s2                  # the predictor's step-by-step outcome (simulated)
python3 -m k8sgpu lint deploy/gke/40-serving-vllm-gcsfuse.yaml --machine g2-standard-4 --load-seconds 120
python3 -m k8sgpu pending --list                   # 17 Pending-pod fixtures (illustrative); --fixture NAME to diagnose one
python3 -m jupyterlab notebooks                    # the exercises
```

With Docker (the primary hands-on path):

```bash
deploy/kind/up.sh                   # ~5 min; DRY_RUN=1 deploy/kind/up.sh prints every command first
python3 -m k8sgpu kind run s1       # apply scenario s1 step by step, observe, compare (s1..s6)
deploy/kind/kwok.sh                 # optional: 32 fake 8-GPU nodes (scenario k1)
deploy/kind/down.sh
```

On a GPU VM (T1/T2): `deploy/gpu-vm/up.sh` (see its README), then the same pod specs get real devices.

## The library

| Module | What it teaches |
|---|---|
| `k8sgpu/manifests.py` | typed builders → YAML for Job, JobSet, LeaderWorkerSet, Kueue (Topology, ResourceFlavor, ClusterQueue, LocalQueue, WorkloadPriorityClass, AdmissionCheck, ProvisioningRequestConfig), DRA `ResourceClaimTemplate` (`resource.k8s.io/v1`), GKE `ComputeClass` |
| `k8sgpu/lint.py` | a GPU pod-spec linter: request == integer limit, GPU taint toleration (Kubernetes' matching rule), accelerator selector, startup probe vs load time, no GPU on sidecars, CPU/memory requests and **stranded GPUs** (GKE's GCS FUSE sidecar counted), `/dev/shm` and its memory limit, restart policy inside CRDs, rollout surge |
| `k8sgpu/machines.py` | what a GPU node gives a pod: GKE allocatable formula (verify), per-GPU share, stranded GPUs (one formula: primer §3.4's `free mod k` when only GPUs bind, the CPU/memory bundle otherwise), time-shared allocatable |
| `k8sgpu/kindsim.py` | the **predictor**: Kueue quota/borrowing/classic preemption, TAS *BestFit* and *LeastFreeCapacity*, the kube-scheduler's filters and its `0/N nodes are available` message |
| `k8sgpu/scenarios.py` | the kind lab's objects, steps and hand-written answer key (renders `deploy/kind/`) |
| `k8sgpu/kindlab.py` | drive a real kind cluster step by step (or dry-run) and compare observation with prediction |
| `k8sgpu/pending.py` | "why is my pod Pending?" across four gates — Kueue, kube-scheduler, cluster autoscaler, kubelet — from `kubectl -o json` |
| `k8sgpu/capacity.py` | on-demand vs Spot vs DWS flex-start vs reservations: expected runtime under interruptions (with checkpoint cost and Young's interval), cost, waits and the probability of starting before a deadline, ties reported as ties; ComputeClass fallback; DWS timeline; cold-start budget |
| `k8sgpu/gke.py` | the GKE manifests (renders `deploy/gke/`) and offline readers for the Terraform (plan, inventory, gcloud equivalents) |
| `k8sgpu/gpuvm.py` | the T1/T2 manifests (renders `deploy/gpu-vm/`): the same GPU pod through the real device plugin on k3s; time-slicing arithmetic |

## Notebooks

Each has worked examples, 4-5 exercises with checks (`✅`), and a *design review* section.
Solutions are in `solutions/`.

1. **`01_manifests_and_the_linter`** (T0) — what makes a pod a GPU pod; the API server's GPU rules; CRDs defer validation; size a startup probe; stranded GPUs; build a lint-clean 16-GPU gang; fix a serving Deployment. Primer §1, §3, §4, §5, §8.
2. **`02_kind_with_fake_gpus_and_kueue`** (T0, live with Docker) — the fake-GPU patch; admit/borrow/wait arithmetic; TAS BestFit; admission vs placement and `waitForPodsReady`; Kueue's victim order; borrow vs preempt; scenarios s1-s6 and k1. Primer §4, §5, §6, §10.
3. **`03_why_is_my_pod_pending`** (T0, live with Docker) — the scheduler's histogram; too-big vs fragmented vs busy; what Kueue is waiting for; pick the fix; the s6 zoo diagnosed live. Primer §3, §6, §7, §8.
4. **`04_gke_pools_dws_and_computeclasses`** (T3; offline = plan/inspect) — the Terraform priced and inventoried; the image's CUDA vs the node driver; Spot's expected runtime and the checkpoint interval; a deadline as a probability; choosing capacity; ComputeClass rungs; DWS through Kueue (and why no TAS on L4); time-sharing one L4; the cold-start budget. Primer §2, §7, §8, §9.

## The kind scenarios

| Id | Shows | Expected outcome (answer key in `k8sgpu/scenarios.py`) |
|---|---|---|
| s1 | a fake GPU is a scheduling unit; Kueue TAS packs | plain Job on any GPU node; the Kueue Job lands on the same node (least free capacity that fits) |
| s2 | gangs + topology | 4-pod `required: host` gang on `host-a1-2`; 8-pod `required: subblock` gang fills `subblock-a2`; a third gang waits (`allows to fit only 2 out of 4 pod(s)`) until the blocker is deleted, then lands on `host-a1-1` |
| s3 | LeaderWorkerSet groups | group 0 (leader + worker, 4 GPUs each) in one subblock; scaling to 2 → group 1 gated, `insufficient unused quota ... 4 more needed` |
| s4 | priority preemption | `a-high` preempts `a-low-2` (the newest low job, `InClusterQueue`) because team-a cannot borrow |
| s5 | cohort borrowing and reclaim | team-a borrows 4 GPUs; team-b reclaims them by preempting `a-job-3` (`InCohortReclamation`) |
| s6 | the Pending zoo | GPU taint, wrong accelerator, too big, fragmented (`FailedScheduling` messages in the kube-scheduler's format, as predicted; `tests/test_upstream_formats.py` rebuilds them from the upstream format strings), Kueue max-quota and topology blockers |
| k1 | KWOK fleet (optional) | 16-host gang takes block `kwok-b1`; 4-host gang takes `kwok-b2-s1`; 8 single-GPU pods packed on `kwok-b2-s2-h1` |

## Deploy targets

* [`deploy/kind/`](deploy/kind/README.md) — the laptop cluster, fake GPUs, Kueue/JobSet/LWS, KWOK; $0.
* [`deploy/gcp/`](deploy/gcp/README.md) — Terraform for a zonal GKE Standard cluster with GPU pools; cost and cleanup.
* [`deploy/gpu-vm/`](deploy/gpu-vm/README.md) — k3s + the NVIDIA device plugin + GFD on one GPU VM, optional time-slicing; the VM's price.
* [`deploy/gke/`](deploy/gke/README.md) — Kueue + DWS flex-start, ComputeClass fallbacks, vLLM with GCS FUSE weights, time-sharing.

`deploy/versions.env` pins every upstream release in one place. The YAML under `deploy/kind/`,
`deploy/gke/` and `deploy/gpu-vm/` is generated from `k8sgpu/scenarios.py`, `k8sgpu/gke.py` and
`k8sgpu/gpuvm.py` (`python3 tools/render_manifests.py`); `python3 -m k8sgpu render --check` and a
test fail if a file drifts or is left behind with no generator.

## Checking it (what CI would run)

```bash
python3 -m pytest -q                                              # builders, linter, predictor, fixtures, fake-cluster runner, assets
python3 tools/build_notebooks.py && python3 tools/run_notebooks.py solutions && python3 tools/run_notebooks.py notebooks --expect-fail
kubernetes-validate --strict -k 1.34.0 $(find deploy -name '*.yaml')   # core kinds (CRDs are skipped with a warning)
python3 tools/check_crds.py                                       # CRD kinds vs the pinned upstream CRDs (network, cached)
bash -n deploy/lib.sh deploy/kind/*.sh deploy/gke/*.sh deploy/gpu-vm/*.sh
cd deploy/gcp/terraform && terraform init -backend=false && terraform validate
```

The scripts are bash-3.2 compatible (macOS) and print every command; `DRY_RUN=1` runs none.

## Verify list (Sep 2026)

Pinned and checked on 2026-09-26: kind v0.33.0 with `kindest/node:v1.34.11` (digest in
`deploy/versions.env`), Kueue v0.19.6 (API `kueue.x-k8s.io/v1beta2`, TAS beta and on by
default, `waitForPodsReady` on by default with a 30 min timeout), JobSet v0.12.0 (`v1alpha2`),
LWS v0.11.0 (`v1`), KWOK v0.8.0, busybox 1.38.0, k3s channel v1.34, NVIDIA device plugin chart
0.20.1, vLLM v0.30.0 (image built on CUDA 13.0.2: needs driver >= 580). Corroborated from
Google-owned repositories, still worth a look on your cluster: the `autoscaling.gke.io/provisioning-request`
node label and the time-sharing node labels (GoogleCloudPlatform/cluster-autoscaler), the
`cloud.google.com/gke-queued=true:NoSchedule` taint and the A3/A4/A4X-only use of the GCE
topology labels (GoogleCloudPlatform/cluster-toolkit), ComputeClass `gpu.driverVersion`
(GoogleCloudPlatform/accelerated-platforms). Re-check before relying on them: whether G2/L4
nodes carry `cloud.google.com/gce-topology-*` labels; the automatic
`nvidia.com/gpu=present:NoSchedule` taint and ExtendedResourceToleration on GKE; the driver
branch GKE's `DEFAULT` installs; ComputeClass field names; flex-start pricing and supported GPU
types; GKE's node-allocatable reservation formula; all prices. MIT licensed.
