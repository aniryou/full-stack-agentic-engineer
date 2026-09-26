# k8s-gpu-lab

**Kubernetes for GPUs, hands-on.** How a GPU becomes schedulable, and how Kubernetes places,
queues, shares and scales it — practised on a real control plane with fake GPUs on a laptop
($0), predicted offline by a small bundled model, and carried to GKE with Terraform when you
want the production shape.

The concepts are in the topic primer, [`../PRIMER.md`](../PRIMER.md); this lab cites its sections
by number (§1 *What Kubernetes sees* … §10 *Learning locally*). The minimal, standard-library
version of the same ideas is [`../k8s-gpu-core/`](../k8s-gpu-core/); this lab does not import it.

## Tiers

| Tier | What you need | Cost | What runs |
|---|---|---|---|
| **T0** | Python 3.10+ (laptop, Colab CPU) | $0 | builders, linter, Pending explainer, capacity model, and the **predictor** for every kind scenario (prints the `kubectl` commands) |
| **T0 + Docker** | a laptop with Docker, [kind v0.33.0](https://kind.sigs.k8s.io/), kubectl >= 1.27 | $0 | `deploy/kind/up.sh`: a 6-node kind cluster, 16 fake `nvidia.com/gpu`, real kube-scheduler, **Kueue v0.19.6**, **JobSet v0.12.0**, **LeaderWorkerSet v0.11.0**; notebooks 02-03 run the scenarios live and compare with the prediction |
| **T3** | a GCP project with billing and L4 quota | ~$0.23/h idle + ~$0.25/h per busy Spot L4 node (verify) | `deploy/gcp/terraform`: zonal GKE Standard, L4 Spot pool 0→N, optional DWS flex-start pool, GCS FUSE, image streaming, managed Prometheus; `deploy/gke/`: Kueue + DWS, ComputeClass, vLLM with GCS FUSE weights |

There is no T1/T2 here: nothing in this layer needs a GPU to be understood. Where GPUs come
from and what they cost across providers: [`../../../COMPUTE.md`](../../../COMPUTE.md).

**What is real and what is simulated on kind** (primer §10): the scheduler, Kueue's quota,
cohorts, preemption and Topology-Aware Scheduling, the JobSet and LWS controllers, taints,
labels and every event you will read are real. The GPUs are an extended resource patched into
node status (`deploy/kind/fake-gpus.sh`): the scheduler counts them and the kubelet admits the
pods, but there is no device plugin, no `/dev/nvidia*`, no CUDA — the pods print as much.

## Quick start

```bash
cd 03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                               # 97 tests, ~5 s, offline
python3 -m k8sgpu kind predict s2                  # what Kueue + the scheduler will do
python3 -m k8sgpu lint deploy/gke/40-serving-vllm-gcsfuse.yaml --machine g2-standard-4 --load-seconds 120
python3 -m k8sgpu pending --list                   # 17 Pending-pod fixtures; --fixture NAME to diagnose one
python3 -m jupyterlab notebooks                    # the exercises
```

With Docker (the primary hands-on path):

```bash
deploy/kind/up.sh                   # ~5 min; DRY_RUN=1 deploy/kind/up.sh prints every command first
python3 -m k8sgpu kind run s1       # apply scenario s1 step by step, observe, compare (s1..s6)
deploy/kind/kwok.sh                 # optional: 32 fake 8-GPU nodes (scenario k1)
deploy/kind/down.sh
```

## The library

| Module | What it teaches |
|---|---|
| `k8sgpu/manifests.py` | typed builders → YAML for Job, JobSet, LeaderWorkerSet, Kueue (Topology, ResourceFlavor, ClusterQueue, LocalQueue, WorkloadPriorityClass, AdmissionCheck, ProvisioningRequestConfig), DRA `ResourceClaimTemplate` (`resource.k8s.io/v1`), GKE `ComputeClass` |
| `k8sgpu/lint.py` | a GPU pod-spec linter: request == integer limit, GPU taint toleration, accelerator selector, startup probe vs load time, no GPU on sidecars, CPU/memory requests and **stranded GPUs**, `/dev/shm`, restart policy inside CRDs, rollout surge |
| `k8sgpu/machines.py` | what a GPU node gives a pod: GKE allocatable formula (verify), per-GPU share, stranded GPUs |
| `k8sgpu/kindsim.py` | the **predictor**: Kueue quota/borrowing/classic preemption, TAS *BestFit* and *LeastFreeCapacity*, the kube-scheduler's filters and its `0/N nodes are available` message |
| `k8sgpu/scenarios.py` | the kind lab's objects, steps and hand-written answer key (renders `deploy/kind/`) |
| `k8sgpu/kindlab.py` | drive a real kind cluster step by step (or dry-run) and compare observation with prediction |
| `k8sgpu/pending.py` | "why is my pod Pending?" across four gates — Kueue, kube-scheduler, cluster autoscaler, kubelet — from `kubectl -o json` |
| `k8sgpu/capacity.py` | on-demand vs Spot vs DWS flex-start vs reservations: expected runtime under interruptions, cost, waits; ComputeClass fallback; DWS timeline; cold-start budget |
| `k8sgpu/gke.py` | the GKE manifests (renders `deploy/gke/`) and offline readers for the Terraform (plan, inventory, gcloud equivalents) |

## Notebooks

Each has worked examples, 4-5 exercises with checks (`✅`), and a *design review* section.
Solutions are in `solutions/`.

1. **`01_manifests_and_the_linter`** (T0) — what makes a pod a GPU pod; the API server's GPU rules; CRDs defer validation; size a startup probe; stranded GPUs; build a lint-clean 16-GPU gang; fix a serving Deployment. Primer §1, §3, §4, §5, §8.
2. **`02_kind_with_fake_gpus_and_kueue`** (T0, live with Docker) — the fake-GPU patch; admit/borrow/wait arithmetic; TAS BestFit; Kueue's victim order; borrow vs preempt; scenarios s1-s6 and k1. Primer §4, §5, §6, §10.
3. **`03_why_is_my_pod_pending`** (T0, live with Docker) — the scheduler's histogram; too-big vs fragmented vs busy; what Kueue is waiting for; pick the fix; the s6 zoo diagnosed live. Primer §3, §6, §7, §8.
4. **`04_gke_pools_dws_and_computeclasses`** (T3; offline = plan/inspect) — the Terraform priced and inventoried; Spot's expected runtime; choosing capacity; ComputeClass rungs; DWS through Kueue; the cold-start budget. Primer §2, §7, §8, §9.

## The kind scenarios

| Id | Shows | Expected outcome (answer key in `k8sgpu/scenarios.py`) |
|---|---|---|
| s1 | a fake GPU is a scheduling unit; Kueue TAS packs | plain Job on any GPU node; the Kueue Job lands on the same node (least free capacity that fits) |
| s2 | gangs + topology | 4-pod `required: host` gang on `host-a1-2`; 8-pod `required: subblock` gang fills `subblock-a2`; a third gang waits (`allows to fit only 2 out of 4 pod(s)`) until the blocker is deleted, then lands on `host-a1-1` |
| s3 | LeaderWorkerSet groups | group 0 (leader + worker, 4 GPUs each) in one subblock; scaling to 2 → group 1 gated, `insufficient unused quota ... 4 more needed` |
| s4 | priority preemption | `a-high` preempts `a-low-2` (the newest low job, `InClusterQueue`) because team-a cannot borrow |
| s5 | cohort borrowing and reclaim | team-a borrows 4 GPUs; team-b reclaims them by preempting `a-job-3` (`InCohortReclamation`) |
| s6 | the Pending zoo | GPU taint, wrong accelerator, too big, fragmented (exact `FailedScheduling` messages), Kueue max-quota and topology blockers |
| k1 | KWOK fleet (optional) | 16-host gang takes block `kwok-b1`; 4-host gang takes `kwok-b2-s1`; 8 single-GPU pods packed on `kwok-b2-s2-h1` |

## Deploy targets

* [`deploy/kind/`](deploy/kind/README.md) — the laptop cluster, fake GPUs, Kueue/JobSet/LWS, KWOK; $0.
* [`deploy/gcp/`](deploy/gcp/README.md) — Terraform for a zonal GKE Standard cluster with GPU pools; cost and cleanup.
* [`deploy/gke/`](deploy/gke/README.md) — Kueue + DWS flex-start, ComputeClass fallbacks, vLLM with GCS FUSE weights.

`deploy/versions.env` pins every upstream release in one place. The YAML under `deploy/kind/`
and `deploy/gke/` is generated from `k8sgpu/scenarios.py` and `k8sgpu/gke.py`
(`python3 tools/render_manifests.py`); a test fails if it drifts.

## Checking it (what CI would run)

```bash
python3 -m pytest -q                                              # builders, linter, predictor, fixtures, fake-cluster runner, assets
python3 tools/build_notebooks.py && python3 tools/run_notebooks.py solutions && python3 tools/run_notebooks.py notebooks --expect-fail
kubernetes-validate --strict -k 1.34.0 $(find deploy -name '*.yaml')   # core kinds (CRDs are skipped with a warning)
python3 tools/check_crds.py                                       # CRD kinds vs the pinned upstream CRDs (network, cached)
bash -n deploy/lib.sh deploy/kind/*.sh deploy/gke/*.sh
cd deploy/gcp/terraform && terraform init -backend=false && terraform validate
```

The scripts are bash-3.2 compatible (macOS) and print every command; `DRY_RUN=1` runs none.

## Verify list (Sep 2026)

Pinned and checked on 2026-09-26: kind v0.33.0 with `kindest/node:v1.34.11` (digest in
`deploy/versions.env`), Kueue v0.19.6 (API `kueue.x-k8s.io/v1beta2`, TAS beta and on by
default), JobSet v0.12.0 (`v1alpha2`), LWS v0.11.0 (`v1`), KWOK v0.8.0, busybox 1.38.0, vLLM
v0.30.0. Re-check before relying on them: GKE's `cloud.google.com/gce-topology-*` labels and
which machine families carry them; the automatic `nvidia.com/gpu=present:NoSchedule` taint and
ExtendedResourceToleration on GKE; ComputeClass field names; the queued-provisioning taint
(`cloud.google.com/gke-queued`) and the `autoscaling.gke.io/provisioning-request` node label;
flex-start pricing and supported GPU types; GKE's node-allocatable reservation formula; all
prices. MIT licensed.
