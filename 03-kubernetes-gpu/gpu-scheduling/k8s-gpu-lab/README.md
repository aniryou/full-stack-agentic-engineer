# k8s-gpu-lab — Kubernetes for GPUs, hands-on

After this lab, you can do these things:

- Write GPU manifests that schedule.
- Explain why a GPU pod is Pending.
- Watch the real kube-scheduler and Kueue admit, place, preempt and reclaim GPU work.

You do the exercises on a real control plane with fake GPUs on a laptop ($0). A small bundled model predicts the
results offline. You run the work with the real device plugin on one GPU VM that you rent by the hour. When you want
the production shape, you take the work to GKE with Terraform.

The concepts are in the topic primer, [`../PRIMER.md`](../PRIMER.md). This lab cites its sections by number (§1
*What Kubernetes sees* … §10 *Learning locally*). The minimal, standard-library version of the same ideas is
[`../k8s-gpu-core/`](../k8s-gpu-core/). This lab does not import it.

## Start here

1. Install the lab and run the tests (see "Run it"). The 128 tests run offline, with no cluster, in about 30 s.
2. Run `python3 -m k8sgpu kind predict s2`. It prints the predictor's step-by-step outcome for a gang scenario
   (simulated).
3. Open [`notebooks/01_manifests_and_the_linter.ipynb`](notebooks/01_manifests_and_the_linter.ipynb). If you have
   Docker, start [`deploy/kind`](deploy/kind/README.md) and then continue with notebook 02.

## What you get: tiers

| Tier | What you need | Cost | What runs |
|---|---|---|---|
| **T0** | Python 3.10+ (laptop, Colab CPU) | $0 | builders, linter, Pending explainer, capacity model, and the **predictor** for every kind scenario (it prints the `kubectl` commands) |
| **T0 + Docker** | a laptop with Docker, [kind v0.33.0](https://kind.sigs.k8s.io/), kubectl >= 1.27 | $0 | `deploy/kind/up.sh`: a 6-node kind cluster, 16 fake `nvidia.com/gpu`, the real kube-scheduler, **Kueue v0.19.6**, **JobSet v0.12.0** and **LeaderWorkerSet v0.11.0**. Notebooks 02-03 run the scenarios live and compare them with the prediction. |
| **T1 / T2** | any GPU VM that you control (Lambda, a GCP VM, your own box), with the NVIDIA driver and Container Toolkit | the VM's hourly price | `deploy/gpu-vm/`: k3s, the real NVIDIA device plugin and GPU Feature Discovery labels, with optional time-slicing. It shows the device path that kind cannot show (a pod gets `/dev/nvidia*`). |
| **T3** | a GCP project with billing and L4 quota | ~$0.23/h idle + ~$0.25/h per busy Spot L4 node (verify) | `deploy/gcp/terraform`: zonal GKE Standard, an L4 Spot pool from 0 to N, optional DWS flex-start and time-sharing pools, GCS FUSE, image streaming and managed Prometheus. `deploy/gke/`: Kueue and DWS, ComputeClass, vLLM with GCS FUSE weights, and time-sharing. |

You can learn every concept at T0. T1/T2 adds the real device plugin on any provider. GKE
(T3) is one production target, never a prerequisite. For where GPUs come from and what they cost
across providers, see [`COMPUTE.md`](../../../COMPUTE.md). RunPod and Vast.ai rent
containers, not VMs. Thus the T1 path needs a VM provider (Lambda, GCP, others).

**What is real and what is simulated on kind** (primer §10). These things are real:

- The scheduler.
- Kueue's quota, cohorts, preemption and Topology-Aware Scheduling.
- The JobSet and LWS controllers.
- Taints, labels and every event that you will read.

The GPUs are an extended resource that `deploy/kind/fake-gpus.sh` patches into the node status. The
scheduler counts them and the kubelet admits the pods. But there is no device plugin, no `/dev/nvidia*`
and no CUDA. The pods print that these three things are not there.

## Run it

```bash
cd 03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab
python3 -m pip install -r requirements.txt && python3 -m pip install -e .
python3 -m pytest -q                               # 128 tests, ~30 s, offline
python3 -m k8sgpu kind predict s2                  # the predictor's step-by-step outcome (simulated)
python3 -m k8sgpu lint deploy/gke/40-serving-vllm-gcsfuse.yaml --machine g2-standard-4 --load-seconds 120
python3 -m k8sgpu pending --list                   # 17 Pending-pod fixtures (illustrative); --fixture NAME to diagnose one
python3 -m jupyterlab notebooks                    # the exercises
```

With Docker (the main hands-on path):

```bash
deploy/kind/up.sh                   # ~5 min; DRY_RUN=1 deploy/kind/up.sh prints every command first
python3 -m k8sgpu kind run s1       # apply scenario s1 step by step, observe, compare (s1..s6)
deploy/kind/kwok.sh                 # optional: 32 fake 8-GPU nodes (scenario k1)
deploy/kind/down.sh
```

On a GPU VM (T1/T2), run `deploy/gpu-vm/up.sh` (see its README). Then the same pod specs get real devices.

## The library

| Module | What it teaches |
|---|---|
| `k8sgpu/manifests.py` | typed builders that make YAML for Job, JobSet, LeaderWorkerSet, Kueue (Topology, ResourceFlavor, ClusterQueue, LocalQueue, WorkloadPriorityClass, AdmissionCheck, ProvisioningRequestConfig), DRA `ResourceClaimTemplate` (`resource.k8s.io/v1`) and GKE `ComputeClass` |
| `k8sgpu/lint.py` | a GPU pod-spec linter. It examines request == integer limit, GPU taint toleration (the match rule of Kubernetes), the accelerator selector, and the startup probe against the load time. It also makes sure that sidecars have no GPU, and examines CPU/memory requests and **stranded GPUs** (it counts GKE's GCS FUSE sidecar). The last items are `/dev/shm` and its memory limit, the restart policy inside CRDs, and the rollout surge. |
| `k8sgpu/machines.py` | what a GPU node gives a pod: the GKE allocatable formula (verify), the per-GPU share, stranded GPUs and time-shared allocatable. Stranded GPUs use one formula. It is primer §3.4's `free mod k` when only GPUs bind, and the CPU/memory bundle otherwise. |
| `k8sgpu/kindsim.py` | the **predictor**: Kueue quota/borrowing/classic preemption, TAS *BestFit* and *LeastFreeCapacity*, and the kube-scheduler's filters and its `0/N nodes are available` message |
| `k8sgpu/scenarios.py` | the objects, steps and hand-written answer key of the kind lab (it renders `deploy/kind/`) |
| `k8sgpu/kindlab.py` | operates a real kind cluster step by step (or as a dry run) and compares the observation with the prediction |
| `k8sgpu/pending.py` | "why is my pod Pending?" across four gates (Kueue, kube-scheduler, cluster autoscaler, kubelet), from `kubectl -o json` |
| `k8sgpu/capacity.py` | a comparison of on-demand, Spot, DWS flex-start and reservations. It calculates the expected runtime under interruptions (with checkpoint cost and Young's interval), the cost and the waits. It also calculates the probability that the job starts before a deadline. It reports ties as ties. It also has the ComputeClass fallback, the DWS timeline and the cold-start budget. |
| `k8sgpu/gke.py` | the GKE manifests (it renders `deploy/gke/`) and offline readers for the Terraform (plan, inventory, gcloud equivalents) |
| `k8sgpu/gpuvm.py` | the T1/T2 manifests (it renders `deploy/gpu-vm/`): the same GPU pod through the real device plugin on k3s, and the time-slicing arithmetic |

## Notebooks

Each notebook has worked examples, 4-5 exercises with checks (`✅`) and a *design review* section.
The solutions are in `solutions/`.

1. **`01_manifests_and_the_linter`** (T0): what makes a pod a GPU pod, the API server's GPU rules, and how CRDs defer validation. Then how to set the size of a startup probe, and stranded GPUs. Then how to build a lint-clean 16-GPU gang, and how to repair a serving Deployment. Primer §1, §3, §4, §5, §8.
2. **`02_kind_with_fake_gpus_and_kueue`** (T0, live with Docker): the fake-GPU patch, the admit/borrow/wait arithmetic and TAS BestFit. Then admission against placement and `waitForPodsReady`, Kueue's victim order, borrow against preempt, and scenarios s1-s6 and k1. Primer §4, §5, §6, §10.
3. **`03_why_is_my_pod_pending`** (T0, live with Docker): the scheduler's histogram, and the difference between too large, fragmented and busy. Then what Kueue waits for, how to select the repair, and the s6 zoo with a live diagnosis. Primer §3, §6, §7, §8.
4. **`04_gke_pools_dws_and_computeclasses`** (T3, and offline it plans and inspects): the price and the inventory of the Terraform. Then the CUDA of the image against the node driver. Then Spot's expected runtime and the checkpoint interval, a deadline as a probability, and how to select capacity. Then ComputeClass rungs, and DWS through Kueue (and why there is no TAS on L4). Last, time-sharing for one L4 and the cold-start budget. Primer §2, §7, §8, §9.

## The kind scenarios

| Id | Shows | Expected outcome (answer key in `k8sgpu/scenarios.py`) |
|---|---|---|
| s1 | a fake GPU is a scheduling unit, and Kueue TAS packs | a plain Job on any GPU node. The Kueue Job lands on the same node (least free capacity that fits). |
| s2 | gangs and topology | a 4-pod `required: host` gang on `host-a1-2`. An 8-pod `required: subblock` gang fills `subblock-a2`. A third gang waits (`allows to fit only 2 out of 4 pod(s)`) until the scenario deletes the blocker. Then the third gang lands on `host-a1-1`. |
| s3 | LeaderWorkerSet groups | group 0 (leader and worker, 4 GPUs each) in one subblock. When you scale to 2, group 1 is gated: `insufficient unused quota ... 4 more needed`. |
| s4 | priority preemption | `a-high` preempts `a-low-2` (the newest low job, `InClusterQueue`) because team-a cannot borrow |
| s5 | cohort borrowing and reclaim | team-a borrows 4 GPUs. team-b reclaims them: it preempts `a-job-3` (`InCohortReclamation`). |
| s6 | the Pending zoo | GPU taint, incorrect accelerator, too large and fragmented. These have `FailedScheduling` messages in the kube-scheduler's format, as predicted. `tests/test_upstream_formats.py` rebuilds them from the upstream format strings. Also Kueue max-quota and topology blockers. |
| k1 | KWOK fleet (optional) | a 16-host gang takes block `kwok-b1`. A 4-host gang takes `kwok-b2-s1`. 8 single-GPU pods land packed on `kwok-b2-s2-h1`. |

## Deploy targets

* [`deploy/kind/`](deploy/kind/README.md): the laptop cluster, fake GPUs, Kueue/JobSet/LWS and KWOK. It costs $0.
* [`deploy/gcp/`](deploy/gcp/README.md): Terraform for a zonal GKE Standard cluster with GPU pools. It also covers the cost and the cleanup.
* [`deploy/gpu-vm/`](deploy/gpu-vm/README.md): k3s, the NVIDIA device plugin and GFD on one GPU VM, with optional time-slicing. It costs the VM's price.
* [`deploy/gke/`](deploy/gke/README.md): Kueue and DWS flex-start, ComputeClass fallbacks, vLLM with GCS FUSE weights, and time-sharing.

`deploy/versions.env` pins every upstream release in one place. `k8sgpu/scenarios.py`,
`k8sgpu/gke.py` and `k8sgpu/gpuvm.py` generate the YAML under `deploy/kind/`, `deploy/gke/` and
`deploy/gpu-vm/` (`python3 tools/render_manifests.py`). `python3 -m k8sgpu render --check` and a
test fail if a file is not the same as the generator output. They also fail if a file stays behind with
no generator.

## Checking it (what CI would run)

```bash
python3 -m pytest -q                                              # builders, linter, predictor, fixtures, fake-cluster runner, assets
python3 tools/build_notebooks.py && python3 tools/run_notebooks.py solutions && python3 tools/run_notebooks.py notebooks --expect-fail
kubernetes-validate --strict -k 1.34.0 $(find deploy -name '*.yaml')   # core kinds (CRDs are skipped with a warning)
python3 tools/check_crds.py                                       # CRD kinds vs the pinned upstream CRDs (network, cached)
bash -n deploy/lib.sh deploy/kind/*.sh deploy/gke/*.sh deploy/gpu-vm/*.sh
cd deploy/gcp/terraform && terraform init -backend=false && terraform validate
```

The scripts are bash-3.2 compatible (macOS). They print every command. With `DRY_RUN=1`, they run no command.

## Verify list (Sep 2026)

The lab pins these releases. The check date is 2026-09-26:

- kind v0.33.0 with `kindest/node:v1.34.11` (digest in `deploy/versions.env`).
- Kueue v0.19.6 (API `kueue.x-k8s.io/v1beta2`, TAS beta and on by default, `waitForPodsReady` on by default with a
  30 min timeout).
- JobSet v0.12.0 (`v1alpha2`).
- LWS v0.11.0 (`v1`).
- KWOK v0.8.0.
- busybox 1.38.0.
- k3s channel v1.34.
- NVIDIA device plugin chart 0.20.1.
- vLLM v0.30.0 (the image uses CUDA 13.0.2, and thus needs driver >= 580).

Google-owned repositories agree with these items. But it is still good to look at them on your cluster:

- The `autoscaling.gke.io/provisioning-request` node label and the time-sharing node labels
  (GoogleCloudPlatform/cluster-autoscaler).
- The `cloud.google.com/gke-queued=true:NoSchedule` taint, and the use of the GCE topology labels on A3/A4/A4X only
  (GoogleCloudPlatform/cluster-toolkit).
- ComputeClass `gpu.driverVersion` (GoogleCloudPlatform/accelerated-platforms).

Examine these items again before you depend on them:

- The `cloud.google.com/gce-topology-*` labels on G2/L4 nodes (present or not).
- The automatic `nvidia.com/gpu=present:NoSchedule` taint and ExtendedResourceToleration on GKE.
- The driver branch that GKE's `DEFAULT` installs.
- ComputeClass field names.
- Flex-start pricing and supported GPU types.
- The formula that GKE uses for the node-allocatable reservation.
- All prices.

MIT licensed.
