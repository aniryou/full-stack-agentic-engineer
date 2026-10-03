# deploy/any-gpu — the lab router in front of real vLLM (T1/T2)

**Tier:** T1 or T2. T1 is one small GPU: a Colab or Kaggle T4 for free, or a rented 24 GB card for ~$0.3–0.7/h.
T2 is Kaggle's free 2 × T4, or a rented multi-GPU box. You do not need Kubernetes or a cloud account.

Here the numbers are real, not emulated. Two or more real vLLM replicas run on the GPU(s).
The lab router routes across them, with the same code as notebooks 01–03. The T1 cells of notebook 04 do these
things:

- They compare round-robin with the llm-d default weights.
- They read the prefix-cache counters of the engines.
- They send live `vllm:num_requests_waiting` / `vllm:num_requests_running` scrapes to the HPA recommender.

Every number that those cells print is a **measurement on your GPU**.

**Cost:** Colab and Kaggle are free. A rented RTX 4090 costs ≈ $0.3–0.4/h (verify). **Cleanup:** run `./down.sh`.
Then *terminate* the rented machine. The machine bills until you terminate it, not until vLLM stops.

| File | What it does |
|---|---|
| `serve.sh` | Starts `REPLICAS` (default 2) × `vllm serve` from pip, one after the other, on ports 8001, 8002, …. Replica *i* runs on GPU *i* mod #GPUs. It prints `IGW_BACKENDS` and the router command. With `DRY_RUN=1`, it only prints |
| `docker-compose.yaml` | The same with Docker: 2 × `vllm/vllm-openai:v0.30.0` on one GPU, and the lab router on `:9100` |
| `down.sh` | Stops what either of the two started |

## The flags, and why the lab needs them

| Flag | Value | Why |
|---|---|---|
| `--served-model-name lab/llm` | | This is the name that the bench of the lab sends. Thus you do not edit a notebook |
| `--enable-prompt-tokens-details` | on | vLLM returns `usage.prompt_tokens_details.cached_tokens` only with this flag. Without it, the bench shows the hit rate as `n/a`. The notebook then uses the `vllm:prefix_cache_*` counters |
| `--max-num-seqs` | 16 | The batch slots. The `running` HPA target is a fraction of this value. Thus set it, and do not use the much larger default of vLLM (verify the default for your GPU) |
| `--gpu-memory-utilization` | 0.85 ÷ replicas per GPU | Many engines can share one GPU only if the sum of their fractions is less than 1 |
| `--dtype half` | on a T4 | Turing has no bfloat16. `serve.sh` finds this. In compose, add the flag yourself |
| `--max-model-len` | 4096 | The agent sessions of the lab stay under ~3.5k tokens |

The flag names are from vLLM 0.30.0 (verify for other releases). Layer 04's `vllm-serving-lab/deploy/any-gpu` has
the single-replica recipe and the flags in depth
([`04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md`](../../../../../04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/README.md)).

## Colab or Kaggle (free T4)

Select Runtime, then T4 GPU (Kaggle: Accelerator, then GPU T4 × 2). In notebook 04 of the lab, before its T1
section, run this:

```python
!pip install -q "vllm==0.30.0"          # several minutes; restart the runtime if pip asks (verify)
!REPLICAS=2 bash deploy/any-gpu/serve.sh   # the Colab bootstrap cell has put you in the lab directory
import os                                  # (Kaggle 2 x T4: one replica per GPU; Colab: both on one T4)
os.environ["IGW_BACKENDS"] = "r0=http://127.0.0.1:8001,r1=http://127.0.0.1:8002"
```

The T1 section of notebook 04 prints the same commands with the absolute paths of this machine.

`serve.sh` returns when the two replicas answer `/health`. The first start downloads ~1 GB of weights.
Colab ends a session after a time with no activity, and the free GPU hours have a limit (not guaranteed).

## A rented box (RunPod, Vast.ai, Lambda, a GCP VM)

- **Container hosts (RunPod, Vast.ai):** Select an image with CUDA and Python. Run `pip install -e .` for this lab
  and for `vllm==0.30.0`, then run `./serve.sh`. Keep the ports private. Run the notebook or the bench on the same
  machine, so that the network is not a part of your TTFT.
- **VMs (Lambda, GCP Compute Engine):** Install Docker and the NVIDIA Container Toolkit (layer 02). Then use
  `docker compose -f docker-compose.yaml up -d --build`, or use `serve.sh` with pip.
- Prices and availability: [`COMPUTE.md`](../../../../../COMPUTE.md) (all verify).

## What changes from the fake backend

The prefill and decode speeds, the batch contention and the `/metrics` are real. Two replicas that share one GPU
also compete for its compute, and the fake backend never models this. A prefill on one replica makes the decode of
the other replica slower. Thus use a run on a shared GPU as a routing experiment (hit rate, per-replica split,
relative TTFT). Use a run with one replica per GPU (Kaggle 2 × T4, T2) as the fair picture of capacity.
