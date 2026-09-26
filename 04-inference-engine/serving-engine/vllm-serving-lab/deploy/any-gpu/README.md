# deploy/any-gpu — a real vLLM on whatever NVIDIA GPU you have (T1)

**Tier:** T1: one small GPU (Colab or Kaggle T4 for free; a rented 24 GB card for ~$0.3-0.7/hr).
The point of T1 is to replace every *simulated* number from the fake server with a *measured*
one: the notebooks detect a server in `SERVELAB_URL` and run the same code against it.

`serve.sh` starts `vllm serve` with docker (when a daemon is reachable) or from pip, picks
`--dtype half` below compute capability 8.0 (a T4), refuses GPUs below 7.5 (see below), and prints
each step (`DRY_RUN=1` prints only). Everything here is pinned to **vLLM v0.30.0**
(`vllm/vllm-openai:v0.30.0`, on Docker Hub since 2026-09-22); flags and defaults below were checked
against that release's source.

```bash
./serve.sh                                                   # Qwen2.5-0.5B-Instruct on :8000
MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_MODEL_LEN=8192 ./serve.sh
curl -s localhost:8000/v1/models && curl -s localhost:8000/metrics | grep '^vllm:' | head
SERVELAB_URL=http://127.0.0.1:8000 jupyter lab ../../notebooks
```

## The flags, explained

| Flag | This lab's default | Why |
|---|---|---|
| `--dtype half` | below compute capability 8.0 | Turing (T4, 7.5) has no bfloat16; float16 is the same 2 bytes. With `auto` vLLM v0.30.0 falls back to float16 with a warning; an explicit `bfloat16` is refused |
| `--max-model-len 4096` | 4096 | caps the longest request and sets the start-up fit check and the worst-case "Maximum concurrency" line; it does not change the number of KV blocks (notebook 01) |
| `--gpu-memory-utilization 0.92` | 0.92 | fraction of GPU memory vLLM may use (weights + activations + CUDA graphs + KV blocks); 0.92 is v0.30.0's default (older releases used 0.9). Pass the same value to `servelab size` when you compare |
| `--max-num-seqs`, `--max-num-batched-tokens` | vLLM defaults | batch size and per-step token budget (notebook 03); `vllm serve` picks 256 / 2048 below 70 GiB and on A100, 1024 / 8192 on other 70 GiB+ GPUs such as H100, 1024 / 16384 at 160 GiB+ |
| `--tensor-parallel-size 2` | 1 (`TP=2`) | two GPUs, half the weights and KV heads each (notebook 03, exercise 3.6) |
| `--api-key` | unset | **set it** on any machine with a public port; the bench sends `SERVELAB_API_KEY` |
| `--enable-prompt-tokens-details` | unset | adds `usage.prompt_tokens_details.cached_tokens`, so the bench can report per-request prefix-cache hits (notebook 04) |
| `--attention-backend` | unset | vLLM picks one per GPU (below); pin it only to compare backends. It replaces the `VLLM_ATTENTION_BACKEND` environment variable, which v0.30.0 no longer has (nor `VLLM_USE_V1`) |

Prefix caching and chunked prefill are on by default in v0.30.0.

## Which GPUs work, and what runs on a T4

* **Minimum compute capability 7.5.** vLLM v0.30.0's wheels and image are CUDA 13 builds (the PyPI
  wheel depends on `[cu13]` packages; the image is built on CUDA 13.0.3) and compile kernels for
  sm_75 and newer only. A T4 (7.5) works; a V100 (7.0) and Kaggle's **P100 (6.0) do not**.
* **Driver.** CUDA 13 needs an NVIDIA driver from the 580 series or newer (verify against NVIDIA's
  CUDA compatibility table). Check before installing; with an older driver, use an older vLLM
  release built for CUDA 12 instead (verify which on its release notes).
* **Attention on a T4.** vLLM v0.30.0 tries FlashAttention, then FlashInfer, then its Triton
  backend, and keeps the first that supports the GPU: FlashAttention needs sm_80+, FlashInfer is
  held at sm_80+ in this release because it is broken on sm_75, so a T4 gets **`TRITON_ATTN`** by
  itself (the startup log names it). No flag or environment variable is needed; old guides that set
  `VLLM_ATTENTION_BACKEND=XFORMERS` describe backends and variables this release does not have.
* **Precision.** No bfloat16 and no FP8 tensor cores on a T4: `--dtype half`, and AWQ/GPTQ
  checkpoints when memory is tight.

## Colab or Kaggle (free T4)

Runtime -> change runtime type -> T4 GPU. On Kaggle: Settings -> Accelerator -> **GPU T4 x2** —
not "GPU P100", whose compute capability 6.0 vLLM cannot run; one of the two T4s is enough here,
both are the T2 recipe below. Then, in a notebook cell, check the GPU and driver first:

```python
!nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv   # want compute_cap >= 7.5, driver >= 580 (verify)
!pip install -q "vllm==0.30.0"      # several minutes; if pip warns about torch versions, restart the runtime (verify)
import subprocess, os
subprocess.Popen("vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 "
                 "--gpu-memory-utilization 0.85 --port 8000 > vllm.log 2>&1", shell=True)
from servelab.env import wait_healthy
assert wait_healthy("http://127.0.0.1:8000", timeout_s=900), open("vllm.log").read()[-3000:]
os.environ["SERVELAB_URL"] = "http://127.0.0.1:8000"   # every later cell measures the real engine
```

`0.85` leaves headroom for anything else that touches the GPU in the same runtime. Pass the same
value when you compare with a prediction (`servelab size --gpu-memory-utilization 0.85`, or
`GPU_MEM_UTIL=0.85` for notebook 01, which also reads it from the log's
`The current --gpu-memory-utilization=...` line). Read the capacity lines back:
`from servelab.sizing import parse_startup_log; parse_startup_log(open("vllm.log").read())`.
Free Colab sessions last up to ~12 h and disconnect after roughly 90 minutes idle; weekly GPU time
is limited and not guaranteed (verify, see [`COMPUTE.md`](../../../../../COMPUTE.md)). Kaggle gives
about 30 GPU-hours a week (verify).

## Two GPUs: tensor parallelism on Kaggle's T4 x2 (T2)

```python
subprocess.Popen("vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 "
                 "--tensor-parallel-size 2 --port 8000 > vllm-tp2.log 2>&1", shell=True)
```

Or `TP=2 ./serve.sh`. Each T4 then holds half the weights and half the KV heads, and every layer
adds two all-reduces over PCIe (no NVLink on this box). Notebook 03 (exercise 3.6) predicts both
effects — more than twice the KV tokens, well under twice the decode speed — and, with
`SERVELAB_START_VLLM=1` on a two-GPU machine, measures batch-1 ITL at TP=1 and TP=2. A 7B model
in fp16 (15 GB) does not fit one T4 at all; with TP=2 it does. Collective bandwidth and latency
themselves are layer 02's subject.

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod / Vast.ai** give you a *container*: choose the `vllm/vllm-openai` image as the template
  image, put the model and flags in the container arguments, expose port 8000, and set
  `--api-key` (the endpoint is public). An RTX 4090 (24 GB) costs roughly $0.3-0.4/hr (verify in
  [`COMPUTE.md`](../../../../../COMPUTE.md)); per-second billing makes a 30-minute session cost cents.
* **Lambda** (and GCP Compute Engine) give you a *VM*: install Docker + the NVIDIA Container
  Toolkit (layer 02) or `pip install vllm`, then run `./serve.sh`.
* Run the benchmark **from the same machine** (`--url http://127.0.0.1:8000`) unless you want the
  network in your TTFT; the bench's TTFT always includes whatever sits between it and the engine.

## Cleanup

`Ctrl-C` (or `docker stop`). Rented machines bill until you *terminate* them, not when vLLM
stops — terminate the pod/instance when the session ends.
