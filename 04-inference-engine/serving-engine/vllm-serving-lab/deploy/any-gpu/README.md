# deploy/any-gpu — a real vLLM on whatever NVIDIA GPU you have (T1)

**Tier:** T1: one small GPU (Colab or Kaggle T4 for free; a rented 24 GB card for ~$0.3-0.7/hr).
The point of T1 is to replace every *simulated* number from the fake server with a *measured*
one: the notebooks detect a server in `SERVELAB_URL` and run the same code against it.

`serve.sh` starts `vllm serve` with docker (when a daemon is reachable) or from pip, picks
`--dtype half` on a T4, and prints each step (`DRY_RUN=1` prints only).

```bash
./serve.sh                                                   # Qwen2.5-0.5B-Instruct on :8000
MODEL=Qwen/Qwen2.5-1.5B-Instruct MAX_MODEL_LEN=8192 ./serve.sh
curl -s localhost:8000/v1/models && curl -s localhost:8000/metrics | grep '^vllm:' | head
SERVELAB_URL=http://127.0.0.1:8000 jupyter lab ../../notebooks
```

## The flags, explained

| Flag | This lab's default | Why |
|---|---|---|
| `--dtype half` | on T4 only | Turing (compute capability 7.5) has no bfloat16; float16 is the same 2 bytes |
| `--max-model-len 4096` | 4096 | caps the longest request; each request's KV reservation is sized by it (notebook 01) |
| `--gpu-memory-utilization 0.90` | 0.90 | fraction of GPU memory vLLM may use; the rest stays free for other processes (vLLM main defaults to 0.92, older releases 0.9 — verify) |
| `--max-num-seqs`, `--max-num-batched-tokens` | vLLM defaults | batch size and per-step token budget (notebook 03) |
| `--api-key` | unset | **set it** on any machine with a public port; the bench sends `SERVELAB_API_KEY` |
| `--enable-prompt-tokens-details` | unset | adds `usage.prompt_tokens_details.cached_tokens`, so the bench can report per-request prefix-cache hits (notebook 04; verify flag name) |

Prefix caching and chunked prefill are on by default in current vLLM.

## Colab or Kaggle (free T4)

Runtime -> change runtime type -> T4 GPU (Kaggle: Settings -> Accelerator -> GPU T4 x2; one GPU is
enough). Then, in a notebook cell:

```python
!pip install -q "vllm>=0.30"        # several minutes; if pip warns about torch versions, restart the runtime (verify)
import subprocess, os
subprocess.Popen("vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype half --max-model-len 4096 "
                 "--gpu-memory-utilization 0.85 --port 8000 > vllm.log 2>&1", shell=True)
from servelab.env import wait_healthy
assert wait_healthy("http://127.0.0.1:8000", timeout_s=900), open("vllm.log").read()[-3000:]
os.environ["SERVELAB_URL"] = "http://127.0.0.1:8000"   # every later cell measures the real engine
```

`0.85` leaves room for the notebook's own process on the same GPU. Read the capacity lines back:
`from servelab.sizing import parse_startup_log; parse_startup_log(open("vllm.log").read())`.
Colab sessions end after a few idle hours and GPU time per week is limited (not guaranteed).

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
