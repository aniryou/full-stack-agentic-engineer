# deploy — the same engine, three places to run it

Every target runs `vllm serve` with the same flags, and the same `servelab` code measures it. Only the URL changes.
Start at T0 (the fake server, with no deploy at all). Then move up.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m servelab fake` | T0 | the fake vLLM (simulated times and metrics) | $0 | Ctrl-C |
| [`any-gpu/`](any-gpu/) | T1 | `vllm serve` with docker or pip on any NVIDIA GPU, a Colab/Kaggle T4 recipe, RunPod/Vast notes | free (Colab/Kaggle) to ~$0.3-0.7/hr | Ctrl-C. Terminate rented machines. |
| [`gcp/cloud-run/`](gcp/cloud-run/) | T3 | Cloud Run service with one L4, scale to zero. Terraform or `gcloud run deploy`. | per second while an instance exists | `terraform destroy` / `./deploy.sh delete` |
| [`gcp/gke/`](gcp/gke/) | T3 | GKE Deployment on an L4 Spot node pool (0..2) + `PodMonitoring` for `/metrics` | Spot L4 node + small cluster | `./cluster.sh delete` |

Terraform: the Cloud Run service is in [`gcp/cloud-run/terraform/`](gcp/cloud-run/terraform/). GKE here is a
`gcloud` script and manifests. Layer 03's [`k8s-gpu-lab`](../../../../03-kubernetes-gpu/gpu-scheduling/k8s-gpu-lab/deploy/gcp/terraform/)
and layer 05's [`inference-gateway-lab`](../../../../05-orchestrator/serving-orchestration/inference-gateway-lab/deploy/gcp/terraform/)
have the GKE cluster as Terraform (GPU node pools, driver installation, Spot, managed Prometheus). The manifests of
`gcp/gke/` run on either cluster.

Prices and GPU availability change. The dated table is in [`COMPUTE.md`](../../../../COMPUTE.md). Layer 05 is about
how to scale several replicas behind a router, and how to autoscale on queue depth or KV usage.
