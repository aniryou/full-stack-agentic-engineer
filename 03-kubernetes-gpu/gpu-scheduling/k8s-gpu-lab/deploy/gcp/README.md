# deploy/gcp — a GKE cluster shaped for GPU scheduling (Terraform)

**What it does.** `terraform/` creates one **zonal GKE Standard** cluster in its own VPC with
node pools for each way of getting GPU capacity (primer §7), plus an optional shared one (§9):

| Pool | Shape | Capacity type | Scales |
|---|---|---|---|
| `system` | 1 x `e2-standard-4` | on-demand | fixed; runs kube-system, Kueue, JobSet |
| `l4-spot` | `g2-standard-4` + 1 x L4 | **Spot**, GKE-installed driver (`LATEST`) | **0 → 2**, autoscaler adds a node only for a Pending GPU pod |
| `l4-flex` (off by default) | `g2-standard-4` + 1 x L4 | **DWS flex-start, queued provisioning** | 0 → 2, all nodes of a request at once, via Kueue's ProvisioningRequest check |
| `l4-shared` (off by default) | `g2-standard-4` + 1 x L4 | Spot, **GPU time-sharing**: each L4 advertised as `max_shared_clients_per_gpu` (4) `nvidia.com/gpu` | 0 → 1 |

plus: Workload Identity, the Cloud Storage FUSE CSI driver, image streaming (GCFS), managed
Prometheus, a weights bucket readable by exactly one Kubernetes ServiceAccount (`serving/model-reader`,
through a Workload Identity Federation `principal://` binding — no keys). Both GPU pools carry
the `nvidia.com/gpu=present:NoSchedule` taint. The driver is `LATEST` because the lab's vLLM
v0.30.0 image is a CUDA 13.0 build that needs an R580+ driver; `DEFAULT` may be older (verify).

**Cost.** About a dollar for a session of a few hours: ~$0.23/h with the GPU pools at zero, plus ~$0.25/h per busy
Spot L4 node (us-central1, Sep 2026 - verify; the table is under [Cost](#cost-us-central1-sep-2026---verify)).

**Clean up.** `deploy/gke/apply-examples.sh delete`, then `terraform destroy` in `deploy/gcp/terraform` — one pass,
nothing left behind but the enabled APIs (see [Clean up](#clean-up)).

| File | Contents |
|---|---|
| `versions.tf` | Terraform >= 1.9, google provider >= 8.0 (validated with 8.4.0), default labels |
| `variables.tf` | every knob, with the cheapest defaults |
| `apis.tf`, `network.tf` | services; a VPC + subnet with pod/service secondary ranges |
| `cluster.tf` | the zonal cluster: release channel, Workload Identity, GCS FUSE, managed Prometheus, image streaming |
| `node_pools.tf` | the pools: system, l4-spot, and the optional l4-flex and l4-shared |
| `storage.tf` | weights bucket + `roles/storage.objectViewer` for the serving KSA |
| `outputs.tf` | `get_credentials`, pool names, bucket, next steps |

## Before you start

* A project with **billing** (GPUs are not available on a Free Trial account; upgrading keeps
  the credits) and `gcloud auth application-default login`.
* **GPU quota**: `GPUS_ALL_REGIONS` and the regional L4 quotas (on-demand and preemptible) often
  start at 0 — request 1-2 in IAM & Admin > Quotas. L4 requests are usually approved quickly (verify).
* Flex-start and queued provisioning: check the current GKE docs for supported GPU types, regions
  and whether your project needs anything enabled first (verify).

## Run it

```bash
cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars      # set project_id (and zone if us-central1-a lacks L4)
terraform init && terraform plan                   # read the plan: 1 cluster, 2-3 pools, 1 bucket
terraform apply
$(terraform output -raw get_credentials)
cd ../../.. && deploy/gke/apply-examples.sh smoke  # scale l4-spot from zero, run nvidia-smi
```

Then `deploy/gke/README.md` (Kueue + DWS, ComputeClass, vLLM with GCS FUSE weights), or notebook
04 for the offline walkthrough (`python3 -m k8sgpu gke plan` prints the same summary).

## Cost (us-central1, Sep 2026 - verify)

| Item | While idle | While one GPU pod runs |
|---|---|---|
| cluster management fee | ~$0.10/h (the GKE free-tier credit covers one zonal cluster) | same |
| system pool, 1 x e2-standard-4 | ~$0.13/h | same |
| l4-spot, g2-standard-4 Spot (0 nodes idle) | $0 | ~$0.25/h per node (on-demand ~$0.70/h) |
| weights bucket | cents per GB-month | + reads within the region are free |

A session of a few hours costs about a dollar. Nothing scales on its own except the GPU pools,
and they return to zero ~10 minutes after the last GPU pod ends.

## Clean up

```bash
deploy/gke/apply-examples.sh delete     # optional: workloads first, so no node waits on a PDB
cd deploy/gcp/terraform && terraform destroy
```

`deletion_protection = false` and `force_destroy = true` (bucket) make `destroy` complete in one
go; the enabled APIs are left on (`disable_on_destroy = false`).

## Verify list

Items marked `verify` in the `.tf` comments: L4 availability per zone; flex-start + queued
provisioning requirements (min 0 nodes, `NO_RESERVATION`, auto-repair off) and supported GPU
types; whether GKE also applies the GPU taint automatically; the default driver auto-install
version threshold; the free-tier credit; all prices.
