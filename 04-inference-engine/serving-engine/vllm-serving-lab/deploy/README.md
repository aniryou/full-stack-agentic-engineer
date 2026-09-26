# deploy — the same engine, three places to run it

Every target runs the same `vllm serve` flags and is measured by the same `servelab` code; only
the URL changes. Start at T0 (the fake server, no deploy at all), then move up.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m servelab fake` | T0 | the fake vLLM (simulated timing and metrics) | $0 | Ctrl-C |
| [`any-gpu/`](any-gpu/) | T1 | `vllm serve` via docker or pip on any NVIDIA GPU; Colab/Kaggle T4 recipe; RunPod/Vast notes | free (Colab/Kaggle) to ~$0.3-0.7/hr | Ctrl-C; terminate rented machines |
| [`gcp/cloud-run/`](gcp/cloud-run/) | T3 | Cloud Run service with one L4, scale to zero; Terraform or `gcloud run deploy` | per second while an instance exists | `terraform destroy` / `./deploy.sh delete` |
| [`gcp/gke/`](gcp/gke/) | T3 | GKE Deployment on an L4 Spot node pool (0..2) + `PodMonitoring` for `/metrics` | Spot L4 node + small cluster | `./cluster.sh delete` |

Prices and GPU availability move; the dated table is [`COMPUTE.md`](../../../../COMPUTE.md).
Scaling several replicas behind a router, and autoscaling on queue depth or KV usage, is layer 05.
