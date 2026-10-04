# deploy/any-gpu — a real vLLM on whatever NVIDIA GPU you have (T1)

**Tier:** T1: one small GPU (a Colab or Kaggle T4 for free, or a rented 24 GB card for ~$0.3-0.7/hr).
The purpose of T1 is to replace every *simulated* number from the fake server with a *measured*
number. The notebooks detect a server in `SERVELAB_URL` and run the same code against it.

`serve.sh` does these things:

- It starts `vllm serve` with docker (when it can reach a daemon) or from pip.
- It selects `--dtype half` below compute capability 8.0 (a T4).
- It refuses GPUs below 7.5 (see "Which GPUs work, and what runs on a T4").
- It prints each step (`DRY_RUN=1` prints only).

The pin for everything here is **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`, on Docker Hub since 2026-09-22). The
flags and defaults in the next sections come from a check against the source of that release.

```bash
./serve.sh                                                   # Qwen2.5-0.5B-Instruct on :8000
MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_MODEL_LEN=8192 ./serve.sh
curl -s localhost:8000/v1/models && curl -s localhost:8000/metrics | grep '^vllm:' | head
SERVELAB_URL=http://127.0.0.1:8000 jupyter lab ../../notebooks
```

## The flags, explained

| Flag | This lab's default | Why |
|---|---|---|
| `--dtype half` | below compute capability 8.0 | Turing (T4, 7.5) has no bfloat16. float16 also uses 2 bytes for each value. With `auto`, vLLM v0.30.0 uses float16 instead and gives a warning. vLLM refuses an explicit `bfloat16`. |
| `--max-model-len 4096` | 4096 | It sets the limit for the longest request. It sets the start-up fit check and the worst-case "Maximum concurrency" line. It does not change the number of KV blocks (notebook 01). |
| `--gpu-memory-utilization 0.92` | 0.92 | The fraction of GPU memory that vLLM can use (weights + activations + CUDA graphs + KV blocks). 0.92 is the default of v0.30.0 (older releases used 0.9). When you compare, give the same value to `servelab size`. |
| `--max-num-seqs`, `--max-num-batched-tokens` | vLLM defaults | The batch size and the per-step token budget (notebook 03). `vllm serve` selects 256 / 2048 below 70 GiB and on A100. It selects 1024 / 8192 on other 70 GiB+ GPUs, for example H100, and 1024 / 16384 at 160 GiB+. |
| `--tensor-parallel-size 2` | 1 (`TP=2`) | Two GPUs. Each GPU holds half of the weights and half of the KV heads (notebook 03, exercise 3.6). |
| `--api-key` | unset | **Set it** on any machine with a public port. The bench sends `SERVELAB_API_KEY`. |
| `--enable-prompt-tokens-details` | unset | It adds `usage.prompt_tokens_details.cached_tokens`. Thus the bench can report the prefix-cache hits of each request (notebook 04). |
| `--attention-backend` | unset | vLLM selects one for each GPU (see the next section). Set it to one backend only to compare backends. It replaces the `VLLM_ATTENTION_BACKEND` environment variable. v0.30.0 no longer has that variable, and it no longer has `VLLM_USE_V1`. |

Prefix caching and chunked prefill are on by default in v0.30.0.

## Which GPUs work, and what runs on a T4

* **Minimum compute capability 7.5.** The wheels and the image of vLLM v0.30.0 are CUDA 13 builds. The PyPI wheel
  depends on `[cu13]` packages, and the base of the image is CUDA 13.0.3. These builds compile kernels for sm_75 and
  newer only. A T4 (7.5) works. A V100 (7.0) and Kaggle's **P100 (6.0) do not**.
* **Driver.** CUDA 13 needs an NVIDIA driver from the 580 series or newer (verify against NVIDIA's
  CUDA compatibility table). Examine the driver before you install. If the driver is older, use an older vLLM
  release built for CUDA 12 instead (verify which on its release notes).
* **Attention on a T4.** vLLM v0.30.0 tries FlashAttention, then FlashInfer, then its Triton backend. It keeps the
  first backend that supports the GPU. FlashAttention needs sm_80+. In this release, vLLM holds FlashInfer at sm_80+,
  because FlashInfer is broken on sm_75. Thus a T4 gets **`TRITON_ATTN`** automatically (the startup log names it).

  You do not need a flag or an environment variable. Old guides that set `VLLM_ATTENTION_BACKEND=XFORMERS` describe
  backends and variables that this release does not have.
* **Precision.** A T4 has no bfloat16 and no FP8 tensor cores. Use `--dtype half`, and use AWQ/GPTQ checkpoints when
  the free memory is small.

## Colab or Kaggle (free T4)

On Colab, select Runtime, then "change runtime type", then "T4 GPU". On Kaggle, select Settings, then Accelerator,
then **GPU T4 x2**. Do not select "GPU P100", because vLLM cannot run on a GPU with compute capability 6.0. One of the two T4s
is sufficient here. The T2 recipe in "Two GPUs: tensor parallelism on Kaggle's T4 x2 (T2)" uses both. Then, in a
notebook cell, examine the GPU and the driver first:

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

`0.85` keeps free memory for anything else that uses the GPU in the same runtime. When you compare with a prediction,
give the same value. Use `servelab size --gpu-memory-utilization 0.85`, or `GPU_MEM_UTIL=0.85` for notebook 01.
Notebook 01 also reads the value from the `The current --gpu-memory-utilization=...` line of the log. To read the
capacity lines back, use
`from servelab.sizing import parse_startup_log; parse_startup_log(open("vllm.log").read())`.

Free Colab sessions last up to ~12 h. They disconnect after approximately 90 minutes with no activity. The weekly GPU
time is limited and not guaranteed (verify, see [`COMPUTE.md`](../../../../../COMPUTE.md)). Kaggle gives
about 30 GPU-hours a week (verify).

## Two GPUs: tensor parallelism on Kaggle's T4 x2 (T2)

```python
subprocess.Popen("vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 4096 "
                 "--tensor-parallel-size 2 --port 8000 > vllm-tp2.log 2>&1", shell=True)
```

Or run `TP=2 ./serve.sh`. Then each T4 holds half of the weights and half of the KV heads. Every layer adds two
all-reduces over PCIe (this box has no NVLink). Notebook 03 (exercise 3.6) predicts the two effects: more than two
times the KV tokens, and much less than two times the decode speed. With `SERVELAB_START_VLLM=1` on a two-GPU
machine, it also measures batch-1 ITL at TP=1 and TP=2. A 7B model in fp16 (15 GB) does not fit on one T4 at all,
but with TP=2 it fits.

The bandwidth and the latency of the collectives themselves are the subject of layer 02.

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod / Vast.ai** give you a *container*. Select the `vllm/vllm-openai` image as the template image. Put the
  model and the flags in the container arguments. Expose port 8000, and set `--api-key`, because the endpoint is
  public. An RTX 4090 (24 GB) costs approximately $0.3-0.4/hr (verify in
  [`COMPUTE.md`](../../../../../COMPUTE.md)). With per-second billing, a 30-minute session costs a few cents.
* **Lambda** (and GCP Compute Engine) give you a *VM*. Install Docker and the NVIDIA Container Toolkit (layer 02), or
  run `pip install vllm`. Then run `./serve.sh`.
* If you do not want the network in your TTFT, run the benchmark **from the same machine**
  (`--url http://127.0.0.1:8000`). The TTFT of the bench always includes everything between the bench and the engine.

## Cleanup

Use `Ctrl-C` (or `docker stop`). A rented machine costs money until you *terminate* it, not until vLLM stops. When
the session ends, terminate the pod/instance.
