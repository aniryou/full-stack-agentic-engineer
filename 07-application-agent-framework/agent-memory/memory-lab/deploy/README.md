# deploy — run the memory lab beyond one Python process

This folder has three targets, from free to paid. The notebooks do not need any of them. Every notebook runs at T0.
When one of these targets is reachable, the notebook moves up to it.

| Target | Tier | What it runs | Cost | Cleanup |
|---|---|---|---|---|
| [`local/`](local/) | T0 + Docker | the memory service, a fake OpenAI-compatible model server, and (optional) Postgres + pgvector, all on 127.0.0.1 through Docker Compose | free | `local/down.sh` (deletes the volumes) |
| [`any-gpu/`](any-gpu/) | T1 | A real small chat model with tool calling and a real embedder on vLLM. It uses the `serve.sh` of the 04 serving lab. | free on Colab/Kaggle, ~$0.3–0.7/hr rented (verify) | Stop the processes. Then terminate the instance. |
| [`gcp/`](gcp/) | T3 | Consolidation as a Cloud Run job on Cloud Scheduler. The model on the Cloud Run GPU service of the 04 serving lab. A table of managed memory stores. | pay per use (verify) | the delete commands in its README |

This lab has no Terraform. The GPU and Cloud Run infrastructure already has Terraform in
[`04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/`](../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/).
A scheduled job is four `gcloud` commands, and `python -m memlab gcp-commands` prints them.
For prices and for where to get GPUs, see [`COMPUTE.md`](../../../../COMPUTE.md).
