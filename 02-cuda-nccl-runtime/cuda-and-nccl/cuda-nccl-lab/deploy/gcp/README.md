# deploy/gcp — a zonal GKE cluster for the GPU runtime lab (T3, optional)

**What it does:** `terraform/` creates a VPC and a node service account with minimal roles. It also
creates a zonal GKE Standard cluster with DCGM metrics and Google Managed Prometheus. The cluster has a
CPU system pool and GPU pools. The GPU pools autoscale **from zero** with GKE-managed NVIDIA drivers:

| Pool | Default | Machine | GPUs | Purpose |
|---|---|---|---|---|
| `system` | on, 1 node | e2-standard-4 | — | kube-system, collectors |
| `l4` | on, 0 to 2, Spot | g2-standard-4 | 1 x L4 | smoke test, CUDA sample, kernels |
| `l4x2` | on, 0 to 2, Spot | g2-standard-24 | 2 x L4 | nccl-tests (PCIe, no NVLink) |
| `l4-shared` | off | g2-standard-4 | 1 x L4, time-shared | GPU time-sharing |
| `a100-mig` | off | a2-highgpu-1g | 1 x A100 as MIG slices | MIG |

Then run the workloads in [`../gke`](../gke/README.md).

**Cost (idle):** the idle cost is the GKE cluster management fee and one e2-standard-4. The GKE
free-tier credit covers one zonal cluster (verify). One e2-standard-4 costs a few dollars a day
(verify).

**GPU pools cost nothing until a pod requests a GPU.** An L4 Spot node costs a fraction of the
~$0.70/hr on-demand L4 price. Spot discounts are 60–91 % (verify current prices). At the end of a
session, always run `terraform destroy`.

**Prerequisites**

* A project on a *paid* billing account. You cannot use GPUs on a Free Trial account. Your credits
  carry over.
* GPU quota in the region: `GPUS_ALL_REGIONS` >= 2 and the regional L4 quota. For Spot VMs, this is the
  *preemptible* L4 quota (verify the names on the *Quotas* page of *IAM & Admin*). New projects
  frequently start at 0. Request the quota.
* `gcloud auth application-default login`, Terraform >= 1.9, `kubectl` with `gke-gcloud-auth-plugin`.

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars      # set project_id; pick a zone with L4
terraform init && terraform apply
$(terraform output -raw get_credentials)
cd ../../gke && ./run.sh status && ./run.sh smoke

# cleanup
kubectl delete namespace gpu-lab
cd ../gcp/terraform && terraform destroy
```

**Design notes (what each choice teaches):**

* *Zonal, one control plane*: this is the GKE shape with the lowest cost. A regional cluster has three
  times the node count.
* *GKE-managed drivers* (`gpu_driver_installation_config`): GKE installs the NVIDIA driver on COS. Its
  device plugin mounts the driver into pods at `/usr/local/nvidia`. The alternative is
  `INSTALLATION_DISABLED` with the NVIDIA GPU Operator (primer §6, §9).
* *Autoscale from zero + Spot*: this is the default for any workload with bursts. The platform can
  preempt Spot nodes. Thus long jobs must write checkpoints. Layer 03 tells how to get the GPUs:
  reservations, DWS flex-start.
* *DCGM + Managed Prometheus*: they give the profiling fields (`DCGM_FI_PROF_*`). With these fields,
  you can understand what `GPU_UTIL` shows (primer §8, notebook 06).
* *Time-sharing and MIG pools off by default*: these pools are for one experiment. Set a pool on for
  the experiment, and then set it off again.

Some items in the `.tf` files have the mark `# VERIFY:`. Examples are the zones that offer L4/A100 and
the MIG partition sizes that GKE supports. These items are product facts. Examine them again before you
apply the configuration.
