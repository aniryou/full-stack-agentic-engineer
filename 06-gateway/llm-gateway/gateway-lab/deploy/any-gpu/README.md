# deploy/any-gpu — the gateway in front of one real vLLM (T1)

**Tier:** T1: one small GPU — Colab or Kaggle T4 for free, a rented 24 GB card for about $0.3–0.7/h (verify in
[`COMPUTE.md`](../../../../../COMPUTE.md)). No Kubernetes, no cloud account.
**Cost:** Colab/Kaggle free; a rented RTX 4090 or L4 for the ~2 GPU-hours the T1 cells of notebooks 01–04 take.
**Cleanup:** `./down.sh`, then *terminate* a rented machine — it bills until you do, not until vLLM stops.

This is where the gateway's numbers stop being simulated. `vllm/vllm-openai:v0.30.0` serves
`Qwen/Qwen2.5-0.5B-Instruct` (494 M parameters, ~1 GB in fp16) under the name `lab/llm`; the gateway's `vllm`
config ([`gwlab/configs/vllm.yaml`](../../gwlab/configs/vllm.yaml)) puts it first in the `chat` alias and a fake
OpenAI-dialect provider second, so notebook 02 can stop vLLM mid-run and watch the fallback and the breaker.

| File | What it does |
|---|---|
| `serve.sh` | `vllm serve` from pip on :8000, a fake fallback on :8101, the gateway on :8080; `DRY_RUN=1` prints only |
| `docker-compose.yaml` | the same with Docker (vLLM bound to loopback: callers reach it only through the gateway) |
| `down.sh` | stops what either started |

## The flags, and why the lab needs them

| Flag | Why |
|---|---|
| `--served-model-name lab/llm` | the upstream model name in `configs/vllm.yaml`; clients never see it — they ask for the alias `chat` |
| `--enable-prompt-tokens-details` | without it vLLM's `usage` has no `prompt_tokens_details.cached_tokens`, and notebook 03 cannot measure prefix-cache hits per request (prefix caching itself is on by default) |
| `--max-model-len 4096` | the lab's prompts are short; it keeps the KV cache of a 0.5B model small on a T4 |
| `--dtype half` on a T4 | Turing has no bfloat16 (`serve.sh` detects it; in compose add it yourself) |
| `--api-key` (optional) | set `VLLM_API_KEY`: the gateway holds it, callers never do; vLLM's check covers only `/v1`, `/v2`, `/inference` and `/cohere` (`/health`, `/metrics` and the rest stay open), so keep the port private anyway |

The gateway asks vLLM for `stream_options.include_usage` on every streamed request and strips the usage chunk
from clients that did not ask; it sends each tenant's `cache_salt` (vLLM validates it: at most 128 characters,
no `@ / \`). Reasoning-token counts need `--reasoning-parser` and a thinking model (the 00.5 thinking lab), not
this 0.5B instruct model.

## Colab or Kaggle (free T4)

Runtime -> T4 GPU. In the notebook you are running (its Colab cell has already put you in the lab directory):

```python
!pip install -q "vllm==0.30.0"          # several minutes (verify the version pin still installs on Colab)
!bash deploy/any-gpu/serve.sh           # returns once vLLM, the fake fallback and the gateway answer /health
import os; os.environ["GWLAB_VLLM_URL"] = "http://127.0.0.1:8000"
```

Then re-run the notebook from the top: its T1 cells start their own in-process gateway with the `vllm` config
and measure the real engine; the rest stays as it was.

## A rented box (RunPod, Vast.ai, Lambda, a GCP VM)

- **Container hosts (RunPod, Vast.ai):** an image with CUDA and Python; `pip install -e .` this lab and
  `vllm==0.30.0`, then `./serve.sh`. Keep port 8000 private.
- **VMs (Lambda, GCP Compute Engine):** Docker plus the NVIDIA Container Toolkit (layer 02), then
  `docker compose -f docker-compose.yaml up -d --build`.

## What changes from the fakes

TTFT and inter-token latency are measured on a GPU; `cached_tokens` comes from vLLM's real prefix cache
(16-token blocks, the salt on the first block only); the ledger reconciles against `vllm:prompt_tokens_total`
and `vllm:generation_tokens_total` from vLLM's own `/metrics`. The self-hosted price row is
`GWLAB_GPU_HOUR_USD / (GWLAB_TOKENS_PER_S × 3600 × 0.6)` per token — set both to what you pay and measure.
