# deploy — the same MoE experiments on a laptop, a GPU box, and GKE

Every target runs the same `vllm serve` flags, and the same `moelab` code reads every target. Only the location of
the server changes. Start at T0. At T0 there is no deploy at all: the notebooks simulate and read bundled samples.
Then move up.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| the notebooks, no server | T0 | tiny MoE on the CPU, simulated step times, illustrative traces and logs | $0 | — |
| [`any-gpu/`](any-gpu/) | T1 / T2 | `vllm serve` with a small MoE on one GPU (offload or INT4 on a T4) or two (TP against EP, `bench_layouts.sh`). Colab/Kaggle recipes. | free (Colab/Kaggle) to ~$0.3–0.7/hr per GPU | `Ctrl-C`. Terminate rented machines. |
| [`gke/`](gke/) | T3 | Qwen1.5-MoE-A2.7B with `--enable-expert-parallel` on layer 02's 2 × L4 `l4x2` pool, with a benchmark from a CPU Job | Spot `g2-standard-24` while the pod runs, and the cluster | `./run.sh clean`, then `terraform destroy` in layer 02 |

There is no Terraform in this lab. GCP is one optional target, and the cluster that it needs already exists as
layer 02's
[`cuda-nccl-lab/deploy/gcp/terraform/`](../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/).
For a single-GPU MoE on Cloud Run or a GKE L4 pool, use layer 04's
[`vllm-serving-lab/deploy/`](../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/) with a
different model name. For example, `allenai/OLMoE-1B-7B-0924-Instruct` fits one L4 in bf16. For prices and for places to get
GPUs, see [`COMPUTE.md`](../../../../COMPUTE.md).
