# deploy — the same engine, three places to run it

Every target runs the same `vllm serve` flags and is measured by the same `servelab` code; only
the URL changes. Start at T0 (the fake server, no deploy at all), then move up.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m servelab fake` | T0 | the fake vLLM (simulated timing and metrics) | $0 | Ctrl-C |
| [`any-gpu/`](any-gpu/) | T1 | `vllm serve` via docker or pip on any NVIDIA GPU; Colab/Kaggle T4 recipe; RunPod/Vast notes | free (Colab/Kaggle) to ~$0.3-0.7/hr | Ctrl-C; terminate rented machines |
| [`gcp/cloud-run/`](gcp/cloud-run/) | T3 | Cloud Run service with one L4, scale to zero; Terraform or `gcloud run deploy` | per second while an instance exists | `terraform destroy` / `./deploy.sh delete` |
| [`gcp/gke/`](gcp/gke/) | T3 | GKE Deployment on an L4 Spot node pool (0..2) + `PodMonitoring` for `/metrics` | Spot L4 node + small cluster | `./cluster.sh delete` |

Terraform: the Cloud Run service is [`gcp/cloud-run/terraform/`](gcp/cloud-run/terraform/). GKE here
is a `gcloud` script plus manifests; the GKE cluster as Terraform (GPU node pools, driver
installation, Spot, managed Prometheus) is provisioned in layer 03's [`k8s-gpu-lab`](../../../../03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/)
and layer 05's [`inference-gateway-lab`](../../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/), and `gcp/gke/`'s
manifests run on either.

Prices and GPU availability move; the dated table is `COMPUTE.md` at the repo root.
Scaling several replicas behind a router, and autoscaling on queue depth or KV usage, is layer 05.
