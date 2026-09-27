# deploy — run the memory lab beyond one Python process

Three targets, from free to paid. Nothing here is needed for the notebooks: every notebook runs at T0 and
upgrades when one of these is reachable.

| Target | Tier | What it runs | Cost | Cleanup |
|---|---|---|---|---|
| [`local/`](local/) | T0 + Docker | the memory service, a fake OpenAI-compatible model server, and (optional) Postgres + pgvector, all on 127.0.0.1 via Docker Compose | free | `local/down.sh` (deletes the volumes) |
| [`any-gpu/`](any-gpu/) | T1 | a real small chat model with tool calling and a real embedder on vLLM, using the 04 serving lab's `serve.sh` | free on Colab/Kaggle, ~$0.3–0.7/hr rented (verify) | stop the processes; terminate the instance |
| [`gcp/`](gcp/) | T3 | consolidation as a Cloud Run job on Cloud Scheduler; the model on the 04 serving lab's Cloud Run GPU service; a table of managed memory stores | pay per use (verify) | the delete commands in its README |

No Terraform in this lab: the GPU and Cloud Run infrastructure already has Terraform in
[`04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/`](../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/gcp/),
and a scheduled job is four `gcloud` commands (`python -m memlab gcp-commands` prints them).
Prices and where to get GPUs: [`COMPUTE.md`](../../../../COMPUTE.md).
