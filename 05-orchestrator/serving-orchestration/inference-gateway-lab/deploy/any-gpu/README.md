# deploy/any-gpu — the lab router in front of real vLLM (T1/T2)

**Tier:** T1 (one small GPU: Colab or Kaggle T4 for free, a rented 24 GB card for ~$0.3–0.7/h) or
T2 (Kaggle's free 2 × T4, or a rented multi-GPU box). No Kubernetes and no cloud account needed.

This is where the numbers stop being emulated. Two or more real vLLM replicas run on the GPU(s),
the lab router (the same code as notebooks 01–03) routes across them, and notebook 04's T1 cells
compare round-robin with the llm-d default weights, read the engines' own prefix-cache counters,
and feed live `vllm:num_requests_waiting` / `vllm:num_requests_running` scrapes to the HPA
recommender. Every number those cells print is **measured on your GPU**.

**Cost:** Colab/Kaggle free; a rented RTX 4090 ≈ $0.3–0.4/h (verify). **Cleanup:** `./down.sh`,
then *terminate* the rented machine — it bills until you do, not until vLLM stops.

| File | What it does |
|---|---|
| `serve.sh` | starts `REPLICAS` (default 2) × `vllm serve` from pip, one after the other, on ports 8001, 8002, …; replica *i* on GPU *i* mod #GPUs; prints `IGW_BACKENDS` and the router command. `DRY_RUN=1` prints only |
| `docker-compose.yaml` | the same with Docker: 2 × `vllm/vllm-openai:v0.30.0` on one GPU + the lab router on `:9100` |
| `down.sh` | stops what either started |

## The flags, and why the lab needs them

| Flag | Value | Why |
|---|---|---|
| `--served-model-name lab/llm` | | the name the lab's bench sends, so no notebook needs editing |
| `--enable-prompt-tokens-details` | on | vLLM only returns `usage.prompt_tokens_details.cached_tokens` with it; without it the bench reports the hit rate as `n/a` (the notebook then falls back to the `vllm:prefix_cache_*` counters) |
| `--max-num-seqs` | 16 | the batch slots; the `running` HPA target is a fraction of this, so set it rather than inherit vLLM's much larger default (verify the default for your GPU) |
| `--gpu-memory-utilization` | 0.85 ÷ replicas per GPU | several engines can share one GPU only if their fractions add up to less than 1 |
| `--dtype half` | on a T4 | Turing has no bfloat16 (`serve.sh` detects it; in compose add it yourself) |
| `--max-model-len` | 4096 | the lab's agent sessions stay under ~3.5k tokens |

Flag names are from vLLM 0.30.0 (verify for other releases); the single-replica recipe and the
flags in depth are in layer 04's `vllm-serving-lab/deploy/any-gpu`
([`04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md)).

## Colab or Kaggle (free T4)

Runtime → T4 GPU (Kaggle: Accelerator → GPU T4 × 2). In the lab's notebook 04, before its T1 section:

```python
!pip install -q "vllm==0.30.0"          # several minutes; restart the runtime if pip asks (verify)
!REPLICAS=2 bash deploy/any-gpu/serve.sh   # the Colab bootstrap cell has put you in the lab directory
import os                                  # (Kaggle 2 x T4: one replica per GPU; Colab: both on one T4)
os.environ["IGW_BACKENDS"] = "r0=http://127.0.0.1:8001,r1=http://127.0.0.1:8002"
```

Notebook 04's T1 section prints the same commands with this machine's absolute paths.

`serve.sh` returns once both replicas answer `/health` (the first start downloads ~1 GB of weights).
Colab sessions end after idle time and free GPU hours are limited (not guaranteed).

## A rented box (RunPod, Vast.ai, Lambda, a GCP VM)

- **Container hosts (RunPod, Vast.ai):** pick an image with CUDA and Python, `pip install -e .` this lab
  and `vllm==0.30.0`, then `./serve.sh`. Keep the ports private: run the notebook or the bench on
  the same machine, so the network is not part of your TTFT.
- **VMs (Lambda, GCP Compute Engine):** install Docker and the NVIDIA Container Toolkit (layer 02)
  and use `docker compose -f docker-compose.yaml up -d --build`, or use `serve.sh` with pip.
- Prices and availability: [`COMPUTE.md`](../../../../../COMPUTE.md) (all verify).

## What changes from the fake backend

Real prefill and decode speeds, real batch contention and real `/metrics`. Two replicas sharing
one GPU also compete for its compute, which the fake backend never models: a prefill on one
replica slows the other's decode. Treat a shared-GPU run as a routing experiment (hit rate,
per-replica split, relative TTFT), and a one-replica-per-GPU run (Kaggle 2 × T4, T2) as the fair
capacity picture.
