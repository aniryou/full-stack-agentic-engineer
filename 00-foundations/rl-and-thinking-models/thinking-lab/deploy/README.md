# deploy — a real thinking model, wherever you have a GPU

Each target serves the same `vllm serve <thinking model> --reasoning-parser ...`. The same `thinklab` code measures
each target. Only `THINKLAB_URL` changes. Start at T0 (no deploy at all), then move up. This lab has no Terraform of
its own. The Google Cloud path uses the 04 serving lab's Cloud Run Terraform and GKE manifests again, with the
settings of a thinking model.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m thinklab fake` | T0 | a fake vLLM that serves a simulated thinking model | $0 | Ctrl-C |
| [`any-gpu/`](any-gpu/) | T1 | `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3` through docker or pip. Colab/Kaggle T4 and 24 GB recipes. One GRPO step with vLLM rollouts (`rl_step.sh`). | free (Colab/Kaggle) to ~$0.3-0.7/hr | Ctrl-C. Terminate rented machines. |
| [`gcp/`](gcp/) | T3 | the 04 lab's Cloud Run service (one L4, scale to zero) or GKE Deployment (L4 Spot), set up for Qwen3-4B with a reasoning parser | per second while an instance exists | `terraform destroy` / `kubectl delete` / the 04 lab's `./cluster.sh delete` |

Prices and GPU availability change. The dated table is in [`COMPUTE.md`](../../../../COMPUTE.md).
