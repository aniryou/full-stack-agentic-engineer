# deploy/any-gpu — a thinking model on whatever NVIDIA GPU you have (T1)

**Tier:** T1: one small GPU (a Colab or Kaggle T4 for free; a rented 24 GB card for ~$0.3-0.7/hr, verify).
T1 replaces the lab's *simulated* numbers with measured ones: point `THINKLAB_URL` at the server
and the notebooks run the same code against the real model.

| Script | What it does |
|---|---|
| [`serve.sh`](serve.sh) | `vllm serve` a thinking model with the right `--reasoning-parser` (picked from the model name), `--reasoning-config` for thinking budgets, `--enable-prompt-tokens-details` for cached-token counts; `--dtype half` below compute capability 8.0; docker if a daemon is reachable, else pip. `DRY_RUN=1` prints the command |
| [`rl_step.sh`](rl_step.sh) | notebook 05's GPU path: vLLM generates 8 × 8 rollouts for `Qwen/Qwen2.5-0.5B-Instruct`, a verifier scores them, transformers takes one GRPO step. `INSTALL=1` installs vLLM and transformers first |

Everything is pinned to **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`), the release the repo's 04 layer
uses; the reasoning flags below were checked against that release's source on 2026-09-26.

## Which model on which card

| GPU | Model | Weights (fp16/bf16) | Parser | Notes |
|---|---|---|---|---|
| T4 16 GB (Colab, Kaggle) | `Qwen/Qwen3-0.6B` | 1.19 GB | `qwen3` | hybrid: thinking on by default, `enable_thinking=false` turns it off; predicted 111,504 KV tokens at `max_model_len` 8192 (≈13 whole 8K traces) |
| T4 16 GB | `Qwen/Qwen3-1.7B` | 3.44 GB | `qwen3` | predicted ≈11 concurrent 8K requests |
| T4 16 GB | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 3.55 GB | `deepseek_r1` | always thinks; no system prompt; temperature 0.6; 2 KV heads = 28 KiB/token |
| T4 16 GB | `Qwen/Qwen3-4B` | 8.04 GB | `qwen3` | fits in fp16 with ≈4.6 concurrent 8K requests (predicted) |
| 24 GB (L4, RTX 4090) | `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Thinking-2507` | 8.04 GB | `qwen3` / `deepseek_r1` | the Thinking-2507 template opens `<think>` in the prompt, so its output only has `</think>` |
| 24 GB | `Qwen/Qwen3-8B` | 16.4 GB | `qwen3` | bf16 fits with little room for KV; does **not** fit a T4 in fp16 |

Capacity numbers are the 04 lab's `servelab.sizing` predictions (verify against the startup log's
"GPU KV cache size" line). vLLM v0.30.0 needs compute capability 7.5 or newer: a T4 works; Kaggle's
**P100 does not** (choose "GPU T4 x2"). On a T4 vLLM picks its Triton attention backend by itself, and
fp16 is the only 16-bit type. Qwen3 is trained in bf16, so check that fp16 outputs are sane (no NaNs,
no garbage) before you trust a measurement (verify).

## Colab or Kaggle (free T4)

```python
!nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv   # want compute_cap >= 7.5
!pip install -q "vllm==0.30.0"      # several minutes
import subprocess, os
subprocess.Popen("vllm serve Qwen/Qwen3-0.6B --dtype half --max-model-len 8192 --gpu-memory-utilization 0.85 "
                 "--reasoning-parser qwen3 --enable-prompt-tokens-details "
                 "--reasoning-config '{\"reasoning_start_str\": \"<think>\", \"reasoning_end_str\": "
                 "\"I have to give the solution based on the reasoning directly now.</think>\"}' "
                 "--port 8000 > vllm.log 2>&1", shell=True)
from thinklab.env import wait_healthy
assert wait_healthy("http://127.0.0.1:8000", timeout_s=900), open("vllm.log").read()[-3000:]
os.environ["THINKLAB_URL"] = "http://127.0.0.1:8000"     # the notebooks now measure the real model
```

Or, from a terminal on any GPU box: `./serve.sh` (Qwen3-0.6B) or `MODEL=Qwen/Qwen3-1.7B ./serve.sh`.
Free Colab sessions are limited and not guaranteed; Kaggle gives about 30 GPU-hours a week (verify,
see [`COMPUTE.md`](../../../../../COMPUTE.md)).

## The flags that matter for thinking models

| Flag / request field | What it does | Pitfall |
|---|---|---|
| `--reasoning-parser qwen3` / `deepseek_r1` | splits the output into `message.reasoning` and `message.content`; fills `usage.completion_tokens_details.reasoning_tokens` | `--enable-reasoning` no longer exists; vLLM names use underscores, SGLang's hyphens |
| `chat_template_kwargs: {"enable_thinking": false}` | Qwen3's hard switch; `--default-chat-template-kwargs` sets the server default | the request-level value wins over the server default |
| `reasoning_effort` | `"none"` turns thinking off; any other value turns it on | for Qwen3 it does not change *how much* it thinks |
| `thinking_token_budget: N` (`-1` = unlimited) | at N reasoning tokens vLLM forces `reasoning_end_str` (set with `--reasoning-config`) and the answer starts | needs the reasoning parser |
| `max_tokens` | caps reasoning **plus** answer | too small ends inside the thinking: `content: null`, `finish_reason: "length"` |
| `include_reasoning: false` | still generates the reasoning, leaves it out of the response | you still pay for the tokens |
| `--enable-prompt-tokens-details` | adds `usage.prompt_tokens_details.cached_tokens` | needed for notebook 04's multi-turn prefix-cache cell |
| sampling | thinking: temperature 0.6, top-p 0.95, top-k 20; non-thinking: 0.7 / 0.8 / 20 | never greedy (endless repetition) |

The response field is `reasoning`; it used to be `reasoning_content`, which SGLang and the DeepSeek
API still use. `thinklab` reads both.

## One GRPO step with vLLM rollouts (notebook 05)

```bash
INSTALL=1 ./rl_step.sh                  # vllm==0.30.0 + transformers, then 8 prompts x 8 rollouts and one step
```

vLLM takes 30% of the card (`gpu_memory_utilization=0.3`). The trainer copy is float32 with plain
SGD (no optimizer state) and gradient checkpointing, so both fit a 15 GB T4 (verify). The step reports
rollout seconds against training seconds, the mean reward, the share of zero-variance groups and the
gap between vLLM's and the trainer's log-probs. It does not push the new weights back into vLLM; TRL's
colocate mode does that every step. For a full TRL run start from
`thinklab.rollout.trl_grpo_config("T4")`: `fp16=True, bf16=False` (a T4 has no bf16; `GRPOConfig`
turns bf16 on otherwise), `use_vllm=True, vllm_mode="colocate"`, `num_generations=8`,
`max_completion_length=256` (TRL 1.14.0 names; its `[vllm]` extra pins `vllm>=0.20.0,<=0.30.0`).

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod / Vast.ai** give you a container: choose the `vllm/vllm-openai:v0.30.0` image, put
  `Qwen/Qwen3-4B --reasoning-parser qwen3 --max-model-len 16384` in the container arguments, expose
  port 8000, and set `--api-key` (the endpoint is public). A 24 GB RTX 4090 costs roughly
  $0.3-0.4/hr (verify in [`COMPUTE.md`](../../../../../COMPUTE.md)); per-second billing makes an
  hour's session cost cents.
* **Lambda** (and GCP Compute Engine) give you a VM: install Docker and the NVIDIA Container Toolkit
  (layer 02), or `pip install vllm`, then run `./serve.sh`.
* Run the notebooks **on the same machine** (`THINKLAB_URL=http://127.0.0.1:8000`) unless you want
  the network in your latencies.

## Cost and cleanup

`Ctrl-C` (or `docker stop`). Colab and Kaggle cost nothing but quota. Rented machines bill until you
*terminate* them, not when vLLM stops, so terminate the pod or instance when the session ends.
Thinking models make long requests: a benchmark with thousands of 8K-token outputs takes much longer
than the same count of chat answers, so budget the session time before you start.
