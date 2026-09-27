# deploy — where the lab gateway runs, from a laptop to Google Cloud

| Target | Tier | What runs | Cost | Cleanup |
|---|---|---|---|---|
| [`local/`](local/README.md) | T0 + Docker | the gateway + two fake providers (one fails 30 % of requests) in Docker Compose | free | `./down.sh` |
| [`any-gpu/`](any-gpu/README.md) | T1 | one real `vllm serve` (Qwen2.5-0.5B as `lab/llm`) behind the gateway, a fake fallback | free on Colab/Kaggle; ~$0.3–0.7/h rented (verify) | `./down.sh`, then terminate the machine |
| [`gcp/`](gcp/README.md) | T3 | a README: the 04 lab's Cloud Run or GKE vLLM and the 05 lab's GKE Inference Gateway as upstreams; no new Terraform | the upstreams' GPU hours (verify) | the upstreams' own teardown |

Without Docker or a GPU, `python -m gwlab stack` runs the same gateway and fakes in one process (T0), and the
notebooks start them themselves.
