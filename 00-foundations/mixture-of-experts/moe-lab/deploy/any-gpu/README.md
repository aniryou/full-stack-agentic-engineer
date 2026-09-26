# deploy/any-gpu — serve a small MoE with vLLM on the GPUs you have (T1, T2)

**Tier:** T1 with one GPU (a free Colab or Kaggle T4, or a rented 24 GB card for ~$0.3–0.7/hr),
T2 with two (Kaggle's free "GPU T4 x2", or a rented pair for about an hour). The point is to turn
the notebooks' *simulated* numbers into *measured* ones: the notebooks find a server through
`MOELAB_URL` (or start one themselves with `MOELAB_START_VLLM=1`) and run the same code against it.

Everything is pinned to **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`); the MoE flags below were
checked against that release's `vllm/engine/arg_utils.py` and `vllm/config/{parallel,offload}.py`.

| File | What it does |
|---|---|
| [`serve_moe.sh`](serve_moe.sh) | starts `vllm serve` (docker when a daemon answers, else pip) in one of four layouts; `--dtype half` below compute capability 8.0; offloads OLMoE's experts on a 16 GB card; `DRY_RUN=1` prints the command |
| [`bench_layouts.sh`](bench_layouts.sh) | on two GPUs, serves the model as TP, TP+EP and DP+EP in turn and runs `vllm bench serve` at each concurrency; results for notebook 04 |

```bash
./serve_moe.sh                                        # OLMoE-1B-7B, one GPU (6 GiB of experts offloaded on a T4)
ROUTED=1 ./serve_moe.sh                               # + per-token expert ids in every response (notebook 02)
LAYOUT=tp_ep ./serve_moe.sh                           # two GPUs, experts split whole across them
./bench_layouts.sh                                    # two GPUs: all three layouts, vllm bench serve each
MOELAB_URL=http://127.0.0.1:8000 jupyter lab ../../notebooks
```

## The MoE flags, explained

| Flag | Default | What it does |
|---|---|---|
| `--tensor-parallel-size 2` | 1 | attention *and every expert* split in half over 2 GPUs; one all-reduce after attention, one after the MoE |
| `--enable-expert-parallel` (`-ep`) | off | experts are placed whole, E/EP per GPU, EP = TP × DP (there is no EP-size flag); without it the experts are tensor-parallel over TP × DP GPUs |
| `--data-parallel-size 2` | 1 | two attention replicas, each with its own requests and KV cache; with `-ep` the MoE layers are shared and the ranks step in lockstep (an idle rank runs dummy forward passes) |
| `--all2all-backend` | `allgather_reducescatter` | how tokens reach experts when DP > 1. DeepEP (`deepep_low_latency`, `deepep_high_throughput`, `deepep_v2`) needs SM90+ with NVLink/RDMA — not a T4/L4. `pplx` and `naive` are removed in v0.30.0; there is no `VLLM_ALL2ALL_BACKEND` variable |
| `--expert-placement-strategy` | `linear` | `round_robin` spreads experts e mod EP, but vLLM honours it only for models with expert groups (DeepSeek-style) — OLMoE falls back to linear (verify for your version) |
| `--enable-eplb`, `--eplb-config` | off | rebalance (and optionally replicate, `num_redundant_experts`) experts from observed load; defaults `window_size` 1000, `step_interval` 3000 |
| `--cpu-offload-gb N` | 0 | N GiB of weights per GPU live in pinned CPU memory and are read over PCIe in **every** forward pass (UVA), routed or not |
| `--cpu-offload-params experts` | all | offload only parameters whose name has an `experts` segment (`mlp.experts.w2_weight`); `expert` or `w2` would not match |
| `--enable-return-routed-experts` | off | each choice carries `routed_experts`: base64 `.npy`, shape `(tokens − 1, layers, top_k)`; request `"routed_experts_prompt_start": 0` to include the prompt |
| `--dtype half` | auto | Turing (T4, 7.5) has no bfloat16; vLLM refuses `bfloat16` below sm_80 |

**Fused MoE kernel configs.** vLLM looks for a tuned Triton config per
`E=<experts>,N=<expert width per GPU>,device_name=<GPU>[,dtype=...].json`; there is none for T4, L4
or A10 in the tree (main, Sep 2026), so expect *"Using default MoE config. Performance might be
sub-optimal!"* in the log. Generate one with vLLM's `benchmarks/kernels/benchmark_moe.py` and point
`VLLM_TUNED_CONFIG_FOLDER` at it (verify the script's flags for your version).

## Which models fit where (weights only; `python -m moelab fit` does the arithmetic)

| Model (id: verify on the Hub) | 16-bit | 4-bit | One T4 (16 GB) | 24 GB (L4, 4090) | 2 × T4 |
|---|---|---|---|---|---|
| `ibm-granite/granite-3.0-3b-a800m-instruct` | 6.6 GB | 1.9 GB | yes | yes | yes |
| `allenai/OLMoE-1B-7B-0924-Instruct` | 13.8 GB | 4.0 GB | only with `--cpu-offload-gb` | yes | yes, any layout |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 28.6 GB | 8.5 GB (GPTQ-Int4) | INT4 only | INT4, or 2 × L4 in 16-bit | INT4 |
| `Qwen/Qwen3-30B-A3B` | 61.1 GB | 17.1 GB (GPTQ-Int4) | no | INT4, ~3 GiB of KV left | no |
| `openai/gpt-oss-20b` | — | 13.8 GB (MXFP4) | no: MXFP4 needs sm80+ and bf16 | yes | no |

## Colab or Kaggle (free T4)

Runtime → change runtime type → T4 GPU. On Kaggle: Settings → Accelerator → **GPU T4 x2** (not
"GPU P100": compute capability 6.0 is below vLLM's minimum). Check the GPU and the driver first
(vLLM v0.30.0 is a CUDA 13 build; drivers from the 580 series or newer, verify):

```python
!nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
!pip install -q "vllm==0.30.0"      # several minutes; restart the runtime if pip replaces torch (verify)
import subprocess, os
subprocess.Popen("vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype half --max-model-len 4096 "
                 "--cpu-offload-gb 6 --cpu-offload-params experts --enable-return-routed-experts "
                 "--port 8000 > vllm.log 2>&1", shell=True)
from moelab.env import wait_healthy
assert wait_healthy("http://127.0.0.1:8000", timeout_s=1200), open("vllm.log").read()[-3000:]
os.environ["MOELAB_URL"] = "http://127.0.0.1:8000"    # notebooks 02, 03 and 05 now measure this server
```

Offloading pins host memory: Colab's free runtime has roughly 12 GB of RAM (verify), so keep
`--cpu-offload-gb` well below it. On Kaggle's two T4s, `./bench_layouts.sh` (or the cell at the end
of notebook 04) runs the TP versus EP comparison; OLMoE in fp16 needs about 6.5 GiB per GPU there
and needs no offload.

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod / Vast.ai** give a *container*: pick the `vllm/vllm-openai:v0.30.0` image, put the model
  and flags in the container arguments, expose port 8000, and set `--api-key` (the port is public).
  A 24 GB RTX 4090 is ~$0.3–0.4/hr and a 2-GPU pod a little over twice that (verify in
  [`COMPUTE.md`](../../../../COMPUTE.md)).
* **Lambda** and Compute Engine give a *VM*: install the NVIDIA Container Toolkit (layer 02) or
  `pip install "vllm==0.30.0"`, then run the scripts here.
* Benchmark from the same machine (`127.0.0.1`) unless you want the network in your TTFT.

## Cost and cleanup

Colab and Kaggle are free within their weekly GPU quotas (not guaranteed). Rented machines bill
until you **terminate** them, not when vLLM stops: `Ctrl-C` the server, then terminate the pod or
instance. `bench_layouts.sh` stops each server itself; delete `./results` when you are done.
