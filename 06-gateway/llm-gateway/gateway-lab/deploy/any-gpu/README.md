# deploy/any-gpu — the gateway in front of one real vLLM (T1)

**Tier:** T1: one small GPU. This is a Colab or Kaggle T4 for free, or a rented 24 GB card for about $0.3–0.7/h
(verify in [`COMPUTE.md`](../../../../../COMPUTE.md)). No Kubernetes, no cloud account.

**Cost:** Colab/Kaggle is free. With a rented RTX 4090 or L4, you pay for the ~2 GPU-hours that the T1 cells of
notebooks 01–04 take.

**Cleanup:** Run `./down.sh`. Then *terminate* a rented machine. The machine costs money until you terminate it, not
until vLLM stops.

Here, the numbers of the gateway are no longer simulated. `vllm/vllm-openai:v0.30.0` serves
`Qwen/Qwen2.5-0.5B-Instruct` (494 M parameters, ~1 GB in fp16) under the name `lab/llm`. The `vllm` config of the
gateway ([`gwlab/configs/vllm.yaml`](../../gwlab/configs/vllm.yaml)) puts it first in the `chat` alias, and a fake
OpenAI-dialect provider second. This lets notebook 02 stop vLLM during the run and watch the fallback and the
breaker.

| File | What it does |
|---|---|
| `serve.sh` | `vllm serve` from pip on :8000, a fake fallback on :8101, the gateway on :8080. With `DRY_RUN=1`, it only prints. |
| `docker-compose.yaml` | the same with Docker. vLLM is bound to loopback, so callers reach it only through the gateway. |
| `down.sh` | stops what either of them started |

## The flags, and why the lab needs them

| Flag | Why |
|---|---|
| `--served-model-name lab/llm` | The upstream model name in `configs/vllm.yaml`. Clients never see it. They ask for the alias `chat`. |
| `--enable-prompt-tokens-details` | Without it, the `usage` of vLLM has no `prompt_tokens_details.cached_tokens`. Then notebook 03 cannot measure prefix-cache hits per request. Prefix caching itself is on by default. |
| `--max-model-len 4096` | The prompts of the lab are short. The flag keeps the KV cache of a 0.5B model small on a T4. |
| `--dtype half` on a T4 | Turing has no bfloat16. `serve.sh` detects this. In compose, add the flag yourself. |
| `--api-key` (optional) | Set `VLLM_API_KEY`. The gateway holds it, and callers never do. The check of vLLM covers only `/v1`, `/v2`, `/inference` and `/cohere`. `/health`, `/metrics` and the rest stay open. Thus, keep the port private all the same. |

The gateway asks vLLM for `stream_options.include_usage` on every streamed request. It strips the usage chunk from
clients that did not ask for it. It sends the `cache_salt` of each tenant. vLLM validates the salt: at most 128
characters, no `@ / \`. Counts of reasoning tokens need `--reasoning-parser` and a thinking model (the 00.5 thinking
lab), not this 0.5B instruct model.

## Colab or Kaggle (free T4)

In the Runtime menu, select T4 GPU. Then type these lines in the notebook that you run. Its Colab cell already put
you in the lab directory.

```python
!pip install -q "vllm==0.30.0"          # several minutes (verify the version pin still installs on Colab)
!bash deploy/any-gpu/serve.sh           # returns once vLLM, the fake fallback and the gateway answer /health
import os; os.environ["GWLAB_VLLM_URL"] = "http://127.0.0.1:8000"
```

Then run the notebook again from the top. Its T1 cells start their own in-process gateway with the `vllm` config,
and they measure the real engine. The rest stays as it was.

## A rented box (RunPod, Vast.ai, Lambda, a GCP VM)

- **Container hosts (RunPod, Vast.ai):** Use an image with CUDA and Python. Install this lab with `pip install -e .`,
  and install `vllm==0.30.0`. Then run `./serve.sh`. Keep port 8000 private.
- **VMs (Lambda, GCP Compute Engine):** Install Docker and the NVIDIA Container Toolkit (layer 02). Then run
  `docker compose -f docker-compose.yaml up -d --build`.

## What changes from the fakes

The T1 cells measure TTFT and inter-token latency on a GPU. `cached_tokens` comes from the real prefix cache of vLLM
(16-token blocks, the salt on the first block only). The ledger reconciles against `vllm:prompt_tokens_total` and
`vllm:generation_tokens_total` from vLLM's own `/metrics`. The self-hosted price row is
`GWLAB_GPU_HOUR_USD / (GWLAB_TOKENS_PER_S × 3600 × 0.6)` per token. Set the two variables to what you pay and what
you measure.
