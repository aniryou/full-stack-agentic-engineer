# deploy/gcp — a zonal GKE cluster for the GPU runtime lab (T3, optional)

**What it does:** `terraform/` creates a VPC, a node service account with minimal roles, a zonal GKE
Standard cluster with DCGM metrics and Google Managed Prometheus, a CPU system pool, and GPU pools
that autoscale **from zero** with GKE-managed NVIDIA drivers:

| Pool | Default | Machine | GPUs | Purpose |
|---|---|---|---|---|
| `system` | on, 1 node | e2-standard-4 | — | kube-system, collectors |
| `l4` | on, 0→2, Spot | g2-standard-4 | 1 x L4 | smoke test, CUDA sample, kernels |
| `l4x2` | on, 0→2, Spot | g2-standard-24 | 2 x L4 | nccl-tests (PCIe, no NVLink) |
| `l4-shared` | off | g2-standard-4 | 1 x L4, time-shared | GPU time-sharing |
| `a100-mig` | off | a2-highgpu-1g | 1 x A100 as MIG slices | MIG |

Then run the workloads in [`../gke`](../gke/README.md).

**Cost (idle):** the GKE cluster management fee (one zonal cluster is covered by the GKE free-tier
credit — verify) plus one e2-standard-4 (a few dollars a day — verify). **GPU pools cost nothing
until a pod requests a GPU**; an L4 Spot node is a fraction of the ~$0.70/hr on-demand L4 price
(Spot discounts are 60–91 %, verify current prices). Always `terraform destroy` at the end of a session.

**Prerequisites**

* A project on a *paid* billing account (GPUs are not usable on a Free Trial account; credits carry over).
* GPU quota in the region: `GPUS_ALL_REGIONS` >= 2 and the regional L4 quota — for Spot VMs the
  *preemptible* L4 quota (verify names in *IAM & Admin → Quotas*). New projects often start at 0: request it.
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

* *Zonal, one control plane* — the cheapest GKE shape; regional clusters triple the node count.
* *GKE-managed drivers* (`gpu_driver_installation_config`) — GKE installs the NVIDIA driver on COS and
  its device plugin mounts it into pods at `/usr/local/nvidia`; the alternative is
  `INSTALLATION_DISABLED` plus the NVIDIA GPU Operator (primer §6, §9).
* *Autoscale from zero + Spot* — the default for anything bursty; Spot nodes can be preempted, so
  long jobs need checkpoints (layer 03 covers obtainability: reservations, DWS flex-start).
* *DCGM + Managed Prometheus* — the profiling fields (`DCGM_FI_PROF_*`) that make `GPU_UTIL`
  interpretable (primer §8, notebook 06).
* *Time-sharing and MIG pools off by default* — they exist to be switched on for one experiment and off again.

Things marked `# VERIFY:` in the `.tf` files (zones offering L4/A100, GKE-supported MIG partition
sizes) are product facts to re-check before applying.
