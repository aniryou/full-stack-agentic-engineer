# deploy/gcp/terraform — a cheap GKE cluster for the Inference Gateway (T3)

**What it does.** It makes a zonal GKE Standard cluster with a system node and an L4 Spot pool that scales from zero.
The cluster is then ready for [`../../gke`](../../gke/README.md), which installs the Inference Gateway.

**Cost and cleanup, in short.** The cost is ~$0.16/h with the GPU pool at 0, and ~$0.28/h more for each L4 Spot node
(assumed us-central1 prices, verify). For the cleanup, run `PROJECT_ID=<id> ../../gke/uninstall.sh`, then
`terraform destroy`. The details are in the paragraphs after the commands.

| File | Creates |
|---|---|
| `apis.tf` | The compute, container and monitoring APIs (never disabled on destroy) |
| `network.tf` | A VPC, a node subnet with Pod/Service secondary ranges, and the **proxy-only subnet** (`REGIONAL_MANAGED_PROXY`). The regional Application Load Balancer behind the Gateway needs this subnet |
| `gke.tf` | A **zonal** GKE Standard cluster (Gateway API `CHANNEL_STANDARD`, Managed Service for Prometheus on, Workload Identity). A 1-node `e2-standard-4` system pool. An **L4 Spot** pool (`g2-standard-4`, 1 × `nvidia-l4`, driver `DEFAULT`, image streaming) that autoscales from **0 to 2** nodes |
| `outputs.tf` | The `get-credentials` command and the next step |

```bash
cp terraform.tfvars.example terraform.tfvars      # set project_id (and zone if needed)
terraform init && terraform apply                 # ~10 min
$(terraform output -raw get_credentials)
PROJECT_ID=<id> ZONE=<zone> ../../gke/install.sh  # the workloads, see ../../gke/README.md
```

Before you apply, make sure that you have these two things. GPUs need a **paid** billing account (not the free
trial). They also need L4 quota (`GPUS_ALL_REGIONS` and `NVIDIA_L4_GPUS` in the region). Request the quota in IAM &
Admin, Quotas. The default release channel is RAPID, because the GKE-managed InferencePool v1 CRD needs
GKE ≥ 1.34.0-gke.1626000 (verify which channels carry it when you apply).

**Cost (assumed us-central1 prices, verify):** With the GPU pool at 0, you pay for the system node (~$0.13/h), the
load balancer when the Gateway exists (~$0.025/h) and the disks. This is approximately $0.16/h. Each L4 Spot node
adds ~$0.28/h (on-demand ~$0.70/h). The pool is at 0 only while no vLLM pod exists. When `../../gke` is installed,
`minReplicas: 1` keeps one L4 node up, also when it has no work (~$0.44/h). The free tier covers the cluster
management fee for one zonal cluster (verify). The platform can preempt Spot VMs at any time.

**Cleanup:** Run `PROJECT_ID=<id> ../../gke/uninstall.sh`. It also removes the project-level IAM binding of the
metrics adapter, which `terraform destroy` does not own. Then run `terraform destroy` (deletion protection is off).
In the console, make sure that no `gke-igw-lab-*` disks or forwarding rules stay.

The configuration passed offline validation with the google provider 8.4.0 (`terraform fmt -check`, `init`,
`validate`). Items with the mark `# VERIFY:` are product details. Examine them again against the current GKE
documentation.
