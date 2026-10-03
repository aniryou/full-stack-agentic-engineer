# deploy/any-gpu — a thinking model on whatever NVIDIA GPU you have (T1)

**Tier:** T1: one small GPU (a Colab or Kaggle T4 for free, or a rented 24 GB card for ~$0.3-0.7/hr, verify).
T1 replaces the lab's *simulated* numbers with measured numbers. Set `THINKLAB_URL` to the address of the server.
Then the notebooks run the same code on the real model.

| Script | What it does |
|---|---|
| [`serve.sh`](serve.sh) | The script runs `vllm serve` for a thinking model with the correct `--reasoning-parser` (selected from the model name). It adds `--reasoning-config` for thinking budgets and `--enable-prompt-tokens-details` for cached-token counts. It adds `--dtype half` below compute capability 8.0. It uses docker if it can reach a daemon, and pip if not. `DRY_RUN=1` prints the command. |
| [`rl_step.sh`](rl_step.sh) | The GPU path of notebook 05. vLLM generates 8 × 8 rollouts for `Qwen/Qwen2.5-0.5B-Instruct`. A verifier scores them. Then transformers takes one GRPO step. `INSTALL=1` installs vLLM and transformers first. |

Everything uses the pinned release **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`). This is the release that the 04
layer of the repo uses. On 2026-09-26, a comparison with the source of that release confirmed the reasoning flags in
this file.

## Which model on which card

| GPU | Model | Weights (fp16/bf16) | Parser | Notes |
|---|---|---|---|---|
| T4 16 GB (Colab, Kaggle) | `Qwen/Qwen3-0.6B` | 1.19 GB | `qwen3` | Hybrid: thinking is on by default, and `enable_thinking=false` turns it off. Predicted 111,504 KV tokens at `max_model_len` 8192 (≈13 whole 8K traces). |
| T4 16 GB | `Qwen/Qwen3-1.7B` | 3.44 GB | `qwen3` | Predicted ≈11 concurrent 8K requests. |
| T4 16 GB | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 3.55 GB | `deepseek_r1` | It always thinks. Do not use a system prompt. Use temperature 0.6. 2 KV heads = 28 KiB/token. |
| T4 16 GB | `Qwen/Qwen3-4B` | 8.04 GB | `qwen3` | It fits in fp16 with ≈4.6 concurrent 8K requests (predicted). |
| 24 GB (L4, RTX 4090) | `Qwen/Qwen3-4B`, `Qwen/Qwen3-4B-Thinking-2507` | 8.04 GB | `qwen3` / `deepseek_r1` | The Thinking-2507 template opens `<think>` in the prompt. Thus its output has only `</think>`. |
| 24 GB | `Qwen/Qwen3-8B` | 16.4 GB | `qwen3` | bf16 fits, with only a small space for KV. It does **not** fit a T4 in fp16. |

The capacity numbers are the predictions of the 04 lab's `servelab.sizing` (verify against the "GPU KV cache size"
line of the startup log). Version v0.30.0 of vLLM needs compute capability 7.5 or newer. A T4 works. Kaggle's **P100 does not**
(select "GPU T4 x2"). On a T4, vLLM selects its Triton attention backend automatically, and fp16 is the only 16-bit
type.

The training of Qwen3 used bf16. Thus, make sure that the fp16 outputs are sane (no NaNs, no meaningless text) before
you trust a measurement (verify).

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

Or, from a terminal on any GPU box, run `./serve.sh` (Qwen3-0.6B) or `MODEL=Qwen/Qwen3-1.7B ./serve.sh`.
Free Colab sessions have limits, and Colab does not guarantee them. Kaggle gives about 30 GPU-hours a week (verify,
see [`COMPUTE.md`](../../../../../COMPUTE.md)).

## The flags that matter for thinking models

| Flag / request field | What it does | Pitfall |
|---|---|---|
| `--reasoning-parser qwen3` / `deepseek_r1` | It divides the output into `message.reasoning` and `message.content`. It fills `usage.completion_tokens_details.reasoning_tokens`. | `--enable-reasoning` no longer exists. vLLM names use underscores, and SGLang names use hyphens. |
| `chat_template_kwargs: {"enable_thinking": false}` | The hard switch of Qwen3. `--default-chat-template-kwargs` sets the server default. | The request-level value has priority over the server default. |
| `reasoning_effort` | `"none"` turns thinking off. Any other value turns it on. | For Qwen3, it does not change *how much* the model thinks. |
| `thinking_token_budget: N` (`-1` = unlimited) | At N reasoning tokens, vLLM forces `reasoning_end_str` (you set it with `--reasoning-config`), and the answer starts. | It needs the reasoning parser. |
| `max_tokens` | It sets the limit for the reasoning **plus** the answer. | A value that is too small stops the output inside the thinking: `content: null`, `finish_reason: "length"`. |
| `include_reasoning: false` | It still generates the reasoning, but it does not put the reasoning in the response. | You still pay for the tokens. |
| `--enable-prompt-tokens-details` | It adds `usage.prompt_tokens_details.cached_tokens`. | Notebook 04's multi-turn prefix-cache cell needs it. |
| sampling | Thinking: temperature 0.6, top-p 0.95, top-k 20. Non-thinking: 0.7 / 0.8 / 20. | Never use greedy decoding (the result is endless repetition). |

The response field is `reasoning`. Its old name is `reasoning_content`, and SGLang and the DeepSeek API still use that
name. `thinklab` reads both.

## One GRPO step with vLLM rollouts (notebook 05)

```bash
INSTALL=1 ./rl_step.sh                  # vllm==0.30.0 + transformers, then 8 prompts x 8 rollouts and one step
```

vLLM takes 30% of the card (`gpu_memory_utilization=0.3`). The trainer copy is float32, with plain SGD (no optimizer
state) and gradient checkpointing. Thus both fit a 15 GB T4 (verify).

The step reports the rollout seconds against the training seconds, and the mean reward. It also reports the share of
zero-variance groups and the gap between the log-probs of vLLM and the log-probs of the trainer. The step does not
push the new weights back into vLLM. TRL's colocate mode does that at each step.

For a full TRL run, start from `thinklab.rollout.trl_grpo_config("T4")`. It sets these values (TRL 1.14.0 names):

- `fp16=True, bf16=False`. A T4 has no bf16, and `GRPOConfig` turns bf16 on if you do not set this value.
- `use_vllm=True, vllm_mode="colocate"`.
- `num_generations=8`.
- `max_completion_length=256`.

The `[vllm]` extra of TRL 1.14.0 pins `vllm>=0.20.0,<=0.30.0`.

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod / Vast.ai** give you a container. Select the `vllm/vllm-openai:v0.30.0` image. Put
  `Qwen/Qwen3-4B --reasoning-parser qwen3 --max-model-len 16384` in the container arguments. Expose port 8000. Set
  `--api-key`, because the endpoint is public. A 24 GB RTX 4090 costs approximately $0.3-0.4/hr (verify in
  [`COMPUTE.md`](../../../../../COMPUTE.md)).

  Per-second billing makes the cost of an hour's session a few cents.
* **Lambda** (and GCP Compute Engine) give you a VM. Install Docker and the NVIDIA Container Toolkit (layer 02), or
  run `pip install vllm`. Then run `./serve.sh`.
* If you do not want the network in your latencies, run the notebooks **on the same machine**
  (`THINKLAB_URL=http://127.0.0.1:8000`).

## Cost and cleanup

Use `Ctrl-C` (or `docker stop`). Colab and Kaggle cost only quota. A rented machine bills until you *terminate* it,
not when vLLM stops. Thus, terminate the pod or instance when the session ends.

Thinking models make long requests. A benchmark with thousands of 8K-token outputs takes much longer than the same
count of chat answers. Thus, set a time budget for the session before you start.
