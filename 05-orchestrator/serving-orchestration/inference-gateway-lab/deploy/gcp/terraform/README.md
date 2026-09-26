# deploy/gcp/terraform — a cheap GKE cluster for the Inference Gateway (T3)

| File | Creates |
|---|---|
| `apis.tf` | compute, container, monitoring APIs (never disabled on destroy) |
| `network.tf` | a VPC, a node subnet with Pod/Service secondary ranges, and the **proxy-only subnet** (`REGIONAL_MANAGED_PROXY`) the regional Application Load Balancer behind the Gateway needs |
| `gke.tf` | a **zonal** GKE Standard cluster (Gateway API `CHANNEL_STANDARD`, Managed Service for Prometheus on, Workload Identity), a 1-node `e2-standard-4` system pool, and an **L4 Spot** pool (`g2-standard-4`, 1 × `nvidia-l4`, driver `DEFAULT`, image streaming) autoscaling **0 → 2** |
| `outputs.tf` | the `get-credentials` command and the next step |

```bash
cp terraform.tfvars.example terraform.tfvars      # set project_id (and zone if needed)
terraform init && terraform apply                 # ~10 min
$(terraform output -raw get_credentials)
PROJECT_ID=<id> ZONE=<zone> ../../gke/install.sh  # the workloads, see ../../gke/README.md
```

Before you apply: GPUs need a **paid** billing account (not the free trial) and L4 quota
(`GPUS_ALL_REGIONS` and `NVIDIA_L4_GPUS` in the region) — request it in IAM & Admin → Quotas. The
default release channel is RAPID because the GKE-managed InferencePool v1 CRD needs
GKE ≥ 1.34.0-gke.1626000 (verify which channels carry it when you apply).

**Cost (assumed us-central1 prices, verify):** with the GPU pool at 0 you pay for the system node
(~$0.13/h), the load balancer once the Gateway exists (~$0.025/h) and disks — roughly $0.16/h; each
L4 Spot node adds ~$0.28/h (on-demand ~$0.70/h). The pool is at 0 only while no vLLM pod exists:
with `../../gke` installed, `minReplicas: 1` keeps one L4 node up even when idle (~$0.44/h). The cluster management fee is covered by the free
tier for one zonal cluster (verify). Spot VMs can be preempted at any time.

**Cleanup:** `PROJECT_ID=<id> ../../gke/uninstall.sh` (it also removes the metrics adapter's
project-level IAM binding, which `terraform destroy` does not own) then `terraform destroy`
(deletion protection is off). Check
that no `gke-igw-lab-*` disks or forwarding rules remain in the console.

Validated offline with the google provider 8.4.0 (`terraform fmt -check`, `init`, `validate`).
Items marked `# VERIFY:` are product details to re-check against current GKE documentation.
