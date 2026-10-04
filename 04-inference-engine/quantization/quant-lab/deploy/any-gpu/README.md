# deploy/any-gpu — make a quantized checkpoint and serve it on whatever NVIDIA GPU you have (T1)

**Tier:** T1: one GPU (a Colab or Kaggle T4 for free, or a rented 24 GB RTX 4090 or L4 for ~$0.3–0.7/hr,
verify). There are two scripts. Both print every step, and both accept `DRY_RUN=1`:

| Script | What it does |
|---|---|
| `compress.sh` | It makes `.venv-llmcompressor` and installs llm-compressor 0.14.0 and this lab. It makes the recipe with `python -m quantlab compress --model ... --print` (the same script that notebook 01 prints), then runs it. The checkpoint goes into the current directory. |
| `serve.sh` | It compares the scheme with the compute capability of the GPU (the rules of `quantlab.serve.plan`). Then it starts `vllm serve` from pip or from the `vllm/vllm-openai:v0.30.0` image. When it uses the image, it mounts a local checkpoint into the container. |

```bash
SCHEME=FP8_DYNAMIC ./compress.sh                              # -> ./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC (no calibration data)
SCHEME=W4A16 ALGO=gptq ./compress.sh                           # -> ./Qwen2.5-1.5B-Instruct-W4A16-gptq (256 calibration samples)
SCHEME=fp8 MODEL=./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC ./serve.sh
SCHEME=w4a16 ./serve.sh                                        # a published AWQ checkpoint (verify the id)
curl -s localhost:8000/v1/models && QUANTLAB_URL=http://127.0.0.1:8000 python -m quantlab bench --url http://127.0.0.1:8000
```

## Which scheme on which GPU

| GPU (compute capability) | Speed lever | What occurs with the others |
|---|---|---|
| **T4** (7.5), free on Colab/Kaggle | `SCHEME=w4a16` (Marlin, faster decode) or `w8a8-int8` (INT8 tensor cores, faster prefill) | `--dtype half` (no bf16). FP8 and NVFP4 load weight-only through Marlin (memory only). **No FP8 KV cache** in any backend. |
| **A100** (8.0) | `w4a16`, `w8a8-int8` | FP8 weight-only (no FP8 tensor cores). FP8 KV through FlashInfer. |
| **L4 / RTX 4090** (8.9) | `fp8` or `fp8-online` (W8A8 on FP8 tensor cores) and `w4a16`. Also `KV_CACHE_DTYPE=fp8`. | FP8 KV changes attention from FlashAttention 2 to FlashInfer. No CUTLASS block-FP8. |
| **H100** (9.0) | `fp8` (+ block FP8, DeepGEMM), `w4a16` through Machete. FP8 KV through FlashAttention 3. | NVFP4 weight-only |
| **B200** (10.0), RTX PRO 6000 (12.0) | `nvfp4` (W4A4 on FP4 tensor cores, CUDA ≥ 12.8), `fp8` | INT8 W8A8 refused. W4A16 runs Marlin (Machete is Hopper-only). |

Every row is `(verify)` against your vLLM version. To make sure of a row, examine the startup log. It names the kernel
(`Using MarlinLinearKernel for CompressedTensorsWNA16`, `Selected CutlassFP8ScaledMMLinearKernel for ...`)
and the attention backend (`Using FLASHINFER attention backend out of potential backends: ...`).
`python -c "from quantlab.serve import parse_startup_log; ..."` reads them.

## Colab or Kaggle (free T4): the INT4 recipe

Select Runtime, then "change runtime type", then T4 GPU. On Kaggle, select **GPU T4 x2**, not P100,
because vLLM v0.30.0 needs sm_75.

```python
!nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
!pip install -q "vllm==0.30.0"          # its own runtime; several minutes (verify the driver it needs)
import subprocess, os
subprocess.Popen("vllm serve Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype half --max-model-len 4096 "
                 "--gpu-memory-utilization 0.85 --port 8000 > vllm.log 2>&1", shell=True)
# wait for "Application startup complete" in vllm.log, then:
os.environ["QUANTLAB_URL"] = "http://127.0.0.1:8000"   # notebooks 02-04 now measure the real engine
```

On a T4, the INT4 checkpoint is the lever, because FP8 is weight-only on that GPU and FP8 KV does not start.
Compare the INT4 checkpoint with `Qwen/Qwen2.5-1.5B-Instruct --dtype half`, with the same client and workload.

## RunPod, Vast.ai, Lambda: FP8 on a 4090 or L4

* **RunPod / Vast.ai** give you a container. Use the `vllm/vllm-openai:v0.30.0` image. Put the model
  and flags in the container arguments (`serve.sh` prints them with `DRY_RUN=1 GPU_CC=8.9`). Expose
  port 8000 and set `API_KEY`, because the endpoint is public. An RTX 4090 costs ~$0.3–0.4/hr (verify).
  `SCHEME=fp8-online` needs no checkpoint of your own.
* **Lambda / a GCP VM** give you a VM. Run both scripts as they are.
* If you do not want the network in TTFT, run the benchmark on the same machine
  (`--url http://127.0.0.1:8000`).

## Cost and cleanup

A compression run on a 0.5–1.5B model takes minutes of GPU time, and GPTQ with 256 samples takes
longest (verify on your GPU). `Ctrl-C` stops `vllm serve`. `rm -rf .venv-llmcompressor recipe_*.py` removes the
build environment. Rented machines bill until you **terminate** them.
