# deploy — the same MoE experiments on a laptop, a GPU box, and GKE

Every target runs the same `vllm serve` flags and is read by the same `moelab` code; only where the
server lives changes. Start at T0 (no deploy at all: the notebooks simulate and read bundled
samples), then move up.

| Target | Tier | What it runs | Cost (verify) | Clean up |
|---|---|---|---|---|
| the notebooks, no server | T0 | tiny MoE on the CPU, simulated step times, illustrative traces and logs | $0 | — |
| [`any-gpu/`](any-gpu/) | T1 / T2 | `vllm serve` with a small MoE on one GPU (offload or INT4 on a T4) or two (TP vs EP, `bench_layouts.sh`); Colab/Kaggle recipes | free (Colab/Kaggle) to ~$0.3–0.7/hr per GPU | `Ctrl-C`; terminate rented machines |
| [`gke/`](gke/) | T3 | Qwen1.5-MoE-A2.7B with `--enable-expert-parallel` on layer 02's 2 × L4 `l4x2` pool, benchmarked from a CPU Job | Spot `g2-standard-24` while the pod runs + the cluster | `./run.sh clean`, then `terraform destroy` in layer 02 |

There is no Terraform in this lab: GCP is one optional target, and the cluster it needs already
exists as layer 02's
[`cuda-nccl-lab/deploy/gcp/terraform/`](../../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/).
A single-GPU MoE on Cloud Run or a GKE L4 pool is layer 04's
[`vllm-serving-lab/deploy/`](../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/) with the
model name changed (`allenai/OLMoE-1B-7B-0924-Instruct` fits one L4 in bf16). Prices and where to
get GPUs: [`COMPUTE.md`](../../../../COMPUTE.md).
