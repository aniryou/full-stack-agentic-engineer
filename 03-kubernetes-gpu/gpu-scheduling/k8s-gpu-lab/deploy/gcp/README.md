# deploy/gcp — a GKE cluster shaped for GPU scheduling (Terraform)

**What it does.** `terraform/` creates one **zonal GKE Standard** cluster in its own VPC. The cluster has
a node pool for each method to get GPU capacity (primer §7). It also has an optional shared node pool (§9):

| Pool | Shape | Capacity type | Scales |
|---|---|---|---|
| `system` | 1 x `e2-standard-4` | on-demand | Does not scale. It runs kube-system, Kueue and JobSet |
| `l4-spot` | `g2-standard-4` + 1 x L4 | **Spot**, GKE-installed driver (`LATEST`) | **0 to 2**. The autoscaler adds a node only for a Pending GPU pod |
| `l4-flex` (off by default) | `g2-standard-4` + 1 x L4 | **DWS flex-start, queued provisioning** | 0 to 2. All nodes of a request come at the same time, through Kueue's ProvisioningRequest check |
| `l4-shared` (off by default) | `g2-standard-4` + 1 x L4 | Spot, **GPU time-sharing**. The node advertises each L4 as `max_shared_clients_per_gpu` (4) `nvidia.com/gpu` | 0 to 1 |

The cluster also has these parts:

- Workload Identity.
- The Cloud Storage FUSE CSI driver.
- Image streaming (GCFS).
- Managed Prometheus.
- A weights bucket that exactly one Kubernetes ServiceAccount (`serving/model-reader`) can read. The access
  goes through a Workload Identity Federation `principal://` binding, with no keys.

Both GPU pools have the `nvidia.com/gpu=present:NoSchedule` taint. The driver is `LATEST` because the
lab's vLLM v0.30.0 image is a CUDA 13.0 build. That build needs an R580+ driver. It is possible that
`DEFAULT` is older (verify).

**Cost.** A session of a few hours costs about a dollar. With the GPU pools at zero, the cluster costs ~$0.23/h.
Each busy Spot L4 node adds ~$0.25/h (us-central1, Sep 2026, verify). The table is in
[Cost](#cost-us-central1-sep-2026---verify).

**Clean up.** Run `deploy/gke/apply-examples.sh delete`. Then run `terraform destroy` in `deploy/gcp/terraform`.
One pass removes everything, except the enabled APIs. They stay on (see [Clean up](#clean-up)).

| File | Contents |
|---|---|
| `versions.tf` | Terraform >= 1.9, google provider >= 8.0 (validated with 8.4.0), default labels |
| `variables.tf` | Every setting, with the lowest-cost defaults |
| `apis.tf`, `network.tf` | The services. A VPC and a subnet with secondary ranges for pods and services |
| `cluster.tf` | The zonal cluster: release channel, Workload Identity, GCS FUSE, managed Prometheus, image streaming |
| `node_pools.tf` | The pools: system, l4-spot, and the optional l4-flex and l4-shared |
| `storage.tf` | The weights bucket and `roles/storage.objectViewer` for the serving KSA |
| `outputs.tf` | `get_credentials`, pool names, bucket, next steps |

## Before you start

* A project with **billing**, and `gcloud auth application-default login`. GPUs are not available on a Free
  Trial account. When you upgrade the account, you keep the credits.
* **GPU quota**: `GPUS_ALL_REGIONS` and the regional L4 quotas (on-demand and preemptible) often start at 0.
  Request 1-2 in IAM & Admin > Quotas. Google usually approves L4 requests fast (verify).
* Flex-start and queued provisioning: read the current GKE docs. Find the supported GPU types and regions, and
  the things that your project must turn on first (verify).

## Run it

```bash
cd deploy/gcp/terraform
cp terraform.tfvars.example terraform.tfvars      # set project_id (and zone if us-central1-a lacks L4)
terraform init && terraform plan                   # read the plan: 1 cluster, 2-3 pools, 1 bucket
terraform apply
$(terraform output -raw get_credentials)
cd ../../.. && deploy/gke/apply-examples.sh smoke  # scale l4-spot from zero, run nvidia-smi
```

Then go to `deploy/gke/README.md` (Kueue and DWS, ComputeClass, vLLM with GCS FUSE weights). Or, for the offline
walkthrough, open notebook 04. `python3 -m k8sgpu gke plan` prints the same summary.

## Cost (us-central1, Sep 2026 - verify)

| Item | While idle | While one GPU pod runs |
|---|---|---|
| cluster management fee | ~$0.10/h (the GKE free-tier credit covers one zonal cluster) | same |
| system pool, 1 x e2-standard-4 | ~$0.13/h | same |
| l4-spot, g2-standard-4 Spot (0 nodes idle) | $0 | ~$0.25/h per node (on-demand ~$0.70/h) |
| weights bucket | cents per GB-month | Same. Reads in the region are free |

A session of a few hours costs about a dollar. Only the GPU pools scale automatically. They go back to zero
~10 minutes after the last GPU pod ends.

## Clean up

```bash
deploy/gke/apply-examples.sh delete     # optional: workloads first, so no node waits on a PDB
cd deploy/gcp/terraform && terraform destroy
```

With `deletion_protection = false` and `force_destroy = true` (bucket), `destroy` completes in one pass. The APIs
stay on (`disable_on_destroy = false`).

## Verify list

The `.tf` comments mark these items `verify`:

- L4 availability per zone.
- The requirements of flex-start and queued provisioning (min 0 nodes, `NO_RESERVATION`, auto-repair off), and
  the supported GPU types.
- If GKE also applies the GPU taint automatically.
- The version threshold of the default driver auto-install.
- The free-tier credit.
- All prices.
