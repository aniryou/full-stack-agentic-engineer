# deploy — produce a quantized checkpoint, serve it on the GPU you have, or on the serving lab's cloud targets

Start at T0 (no deploy at all: `python -m quantlab fake --scheme w4a16` is a fake vLLM that runs at
INT4 speed, labelled simulated), then move up. Every target runs the same `vllm serve` flags that
`python -m quantlab plan --gpu <GPU> --scheme <scheme>` prints, and is measured by the same client.

| Target | Tier | What it does | Cost (verify) | Clean up |
|---|---|---|---|---|
| `python -m quantlab fake` | T0 | the emulator behind an OpenAI-style API | $0 | Ctrl-C |
| [`any-gpu/compress.sh`](any-gpu/) | T1 | llm-compressor in its own virtualenv: FP8_DYNAMIC (no data), W4A16 GPTQ, W8A8, NVFP4 | free (Colab/Kaggle T4) to ~$0.3–0.7/hr for minutes | `rm -rf .venv-llmcompressor` and the checkpoint dir |
| [`any-gpu/serve.sh`](any-gpu/) | T1 | `vllm serve` / `docker run` per scheme after checking it against the GPU | same | Ctrl-C; terminate rented machines |
| [`gcp/`](gcp/) | T3 | the serving lab's Cloud Run service (one L4, scale to zero) or GKE Deployment with a quantized model | per second while an instance exists | `terraform destroy` / `./deploy-quantized.sh delete` |

This lab adds **no Terraform**: the Cloud Run service and the GKE manifests live in
[`../../../serving-engine/vllm-serving-lab/deploy/gcp/`](../../../serving-engine/vllm-serving-lab/deploy/gcp/);
[`gcp/`](gcp/) holds the variables and a wrapper that point them at a quantized model. Prices and GPU
availability move; the dated table is [`COMPUTE.md`](../../../../COMPUTE.md).
