# deploy — a real thinking model, wherever you have a GPU

Every target serves the same `vllm serve <thinking model> --reasoning-parser ...` and is measured by
the same `thinklab` code; only `THINKLAB_URL` changes. Start at T0 (no deploy at all), then move up.
This lab has no Terraform of its own: the Google Cloud path reuses the 04 serving lab's Cloud Run
Terraform and GKE manifests with a thinking model's settings.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m thinklab fake` | T0 | a fake vLLM serving a simulated thinking model | $0 | Ctrl-C |
| [`any-gpu/`](any-gpu/) | T1 | `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3` via docker or pip; Colab/Kaggle T4 and 24 GB recipes; one GRPO step with vLLM rollouts (`rl_step.sh`) | free (Colab/Kaggle) to ~$0.3-0.7/hr | Ctrl-C; terminate rented machines |
| [`gcp/`](gcp/) | T3 | the 04 lab's Cloud Run service (one L4, scale to zero) or GKE Deployment (L4 Spot), set up for Qwen3-4B with a reasoning parser | per second while an instance exists | `terraform destroy` / `kubectl delete` / the 04 lab's `./cluster.sh delete` |

Prices and GPU availability move; the dated table is [`COMPUTE.md`](../../../../COMPUTE.md).
