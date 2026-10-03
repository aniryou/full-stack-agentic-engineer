# deploy/any-gpu — serve a small MoE with vLLM on the GPUs you have (T1, T2)

**Tier:** T1 with one GPU (a free Colab or Kaggle T4, or a rented 24 GB card for ~$0.3–0.7/hr). T2 with two
(Kaggle's free "GPU T4 x2", or a rented pair for about an hour). The purpose is to change the *simulated* numbers
of the notebooks into *measured* ones. The notebooks find a server through `MOELAB_URL`, or start one themselves
with `MOELAB_START_VLLM=1`. Then they run the same code against it.

Everything uses the pinned release **vLLM v0.30.0** (`vllm/vllm-openai:v0.30.0`). The MoE flags in "The MoE flags,
explained" come from a check against the `vllm/engine/arg_utils.py` and `vllm/config/{parallel,offload}.py` of
that release.

| File | What it does |
|---|---|
| [`serve_moe.sh`](serve_moe.sh) | Starts `vllm serve` in one of four layouts, with docker when a daemon answers, and with pip if not. Sets `--dtype half` below compute capability 8.0. Offloads OLMoE's experts on a 16 GB card. With `DRY_RUN=1`, it prints the command. |
| [`bench_layouts.sh`](bench_layouts.sh) | On two GPUs, serves the model as TP, TP+EP and DP+EP, one after the other. Runs `vllm bench serve` at each concurrency. Gives the results for notebook 04. |

```bash
./serve_moe.sh                                        # OLMoE-1B-7B, one GPU (3 GiB of experts offloaded on a T4)
ROUTED=1 ./serve_moe.sh                               # + per-token expert ids in every response (notebook 02)
LAYOUT=tp_ep ./serve_moe.sh                           # two GPUs, experts split whole across them
./bench_layouts.sh                                    # two GPUs: all three layouts, vllm bench serve each
MOELAB_URL=http://127.0.0.1:8000 jupyter lab ../../notebooks
```

## The MoE flags, explained

| Flag | Default | What it does |
|---|---|---|
| `--tensor-parallel-size 2` | 1 | Divides attention *and every expert* in half over 2 GPUs. There is one all-reduce after attention and one after the MoE. |
| `--enable-expert-parallel` (`-ep`) | off | Puts each expert whole on one GPU, E/EP per GPU, EP = TP × DP. There is no EP-size flag. Without it, the experts are tensor-parallel over TP × DP GPUs. |
| `--data-parallel-size 2` | 1 | Two attention replicas, each with its own requests and KV cache. With `-ep`, the replicas share the MoE layers, and the ranks do each step in lockstep. An idle rank runs dummy forward passes. |
| `--all2all-backend` | `allgather_reducescatter` | How tokens get to experts when DP > 1. DeepEP (`deepep_low_latency`, `deepep_high_throughput`, `deepep_v2`) works only on SM90+ with NVLink/RDMA, not on a T4/L4. In v0.30.0, `pplx` and `naive` are removed. There is no `VLLM_ALL2ALL_BACKEND` variable. |
| `--expert-placement-strategy` | `linear` | `round_robin` puts the experts on GPUs by e mod EP. But vLLM applies it only to models with expert groups (DeepSeek-style). OLMoE goes back to linear (verify for your version). |
| `--enable-eplb`, `--eplb-config` | off | Balances the experts again from the observed load, and optionally replicates them (`num_redundant_experts`). The defaults are `window_size` 1000 and `step_interval` 3000. |
| `--cpu-offload-gb N` | 0 | N GiB of weights per GPU stay in pinned CPU memory. The GPU reads them over PCIe in **every** forward pass (UVA), routed or not. |
| `--cpu-offload-params experts` | all | Offloads only the parameters whose name has an `experts` segment (`mlp.experts.w2_weight`). A value of `expert` or `w2` does not match. |
| `--enable-return-routed-experts` | off | Each choice carries `routed_experts`: base64 `.npy`, shape `(tokens − 1, layers, top_k)`. To include the prompt, request `"routed_experts_prompt_start": 0`. |
| `--dtype half` | auto | Turing (T4, 7.5) has no bfloat16. vLLM rejects `bfloat16` below sm_80. |

**Fused MoE kernel configs.** vLLM looks for a tuned Triton config per
`E=<experts>,N=<expert width per GPU>,device_name=<GPU>[,dtype=...].json`. The tree has none for T4, L4 or A10
(main, Sep 2026). Thus, expect *"Using default MoE config. Performance might be
sub-optimal!"* in the log. Generate a config with vLLM's `benchmarks/kernels/benchmark_moe.py`. Then set `VLLM_TUNED_CONFIG_FOLDER` to
it (verify the script's flags for your version).

## Which models fit where (weights only; `python -m moelab fit` does the arithmetic)

| Model (id: verify on the Hub) | 16-bit | 4-bit | One T4 (16 GB) | 24 GB (L4, 4090) | 2 × T4 |
|---|---|---|---|---|---|
| `ibm-granite/granite-3.0-3b-a800m-instruct` | 6.6 GB | 1.9 GB | yes | yes | yes |
| `allenai/OLMoE-1B-7B-0924-Instruct` | 13.8 GB | 4.0 GB | only with `--cpu-offload-gb` | yes | yes, any layout |
| `Qwen/Qwen1.5-MoE-A2.7B-Chat` | 28.6 GB | 8.5 GB (GPTQ-Int4) | INT4 only | INT4, or 2 × L4 in 16-bit | INT4 |
| `Qwen/Qwen3-30B-A3B` | 61.1 GB | 17.1 GB (GPTQ-Int4) | no | INT4, ~3 GiB of KV left | no |
| `openai/gpt-oss-20b` | — | 13.8 GB (MXFP4) | no. For MXFP4, sm80+ and bf16 are necessary. | yes | no |

## Colab or Kaggle (free T4)

**Get the lab onto the machine first.** On Colab, open a notebook through the Colab links in the layer README. Its
first cell clones the repo and installs `moelab`. That cell acts only on Colab. Thus, on **Kaggle** (and on any
rented GPU or VM), make the checkout yourself first. On Kaggle, do these steps:

1. Select *File*, then *Import Notebook*, with the lab's
   [`notebooks/04_expert_parallelism_on_two_gpus.ipynb`](../../notebooks/04_expert_parallelism_on_two_gpus.ipynb)
   (or 02, 03 or 05, and verify the current menu).
2. Select *Settings*, then *Accelerator*, then ***GPU T4 x2***. Do not select "GPU P100". Its compute capability
   6.0 is below vLLM's minimum.
3. Set *Internet* on. For this, a verified phone number is necessary (verify current rules).

Then add this as the first cell:

```python
!git clone --depth 1 https://github.com/aniryou/full-stack-agentic-engineer.git
%cd full-stack-agentic-engineer/00-foundations/mixture-of-experts/moe-lab
!pip install -q -e .
```

Then the bootstrap cell of the notebook finds `moelab/` from the lab directory. Every cell after it runs as it
does locally. On a rented machine, run the same three lines in a shell, with `cd` for `%cd`. Then run
`jupyter lab notebooks/`. On Colab, select *Runtime*, then *Change runtime type*, then *T4 GPU*.

Examine the GPU and the driver. vLLM v0.30.0 is a CUDA 13 build. Use drivers from the 580 series or newer
(verify). Then install vLLM and start a server:

```python
!nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
!pip install -q "vllm==0.30.0"      # several minutes; restart the runtime if pip replaces torch (verify)
import subprocess, os
subprocess.Popen("vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype half --max-model-len 4096 "
                 "--cpu-offload-gb 3 --cpu-offload-params experts --enable-return-routed-experts "
                 "--port 8000 > vllm.log 2>&1", shell=True)
from moelab.env import wait_healthy     # needs the checkout above (Kaggle, rented) or the Colab bootstrap
assert wait_healthy("http://127.0.0.1:8000", timeout_s=1200), open("vllm.log").read()[-3000:]
os.environ["MOELAB_URL"] = "http://127.0.0.1:8000"    # notebooks 02, 03 and 05 now measure this server
```

`--cpu-offload-gb 3` is the smallest offload that leaves room on a 16 GB T4 for four 4K-token sequences
(`python -m moelab fit --model olmoe-1b-7b --gpu T4`). Each offloaded GiB costs ~89 ms per step over PCIe Gen3
(verify).

An offload pins host memory. Colab's free runtime has roughly 12 GB of RAM (verify). Thus, keep
`--cpu-offload-gb` well below it. On Kaggle's two T4s, `!bash deploy/any-gpu/bench_layouts.sh` from the lab
directory runs the TP against EP comparison. The cell at the end of notebook 04 also runs it. On the two T4s, OLMoE
in fp16 uses about 6.5 GiB per GPU, and no offload is necessary.

## Rented GPUs (RunPod, Vast.ai, Lambda)

* **RunPod / Vast.ai** give a *container*. Select the `vllm/vllm-openai:v0.30.0` image. Put the model and flags
  in the container arguments. Expose port 8000. Set `--api-key`, because the port is public. A 24 GB RTX 4090 is
  ~$0.3–0.4/hr, and a 2-GPU pod is more than two times that, but not by much (verify in
  [`COMPUTE.md`](../../../../../COMPUTE.md)).
* **Lambda** and Compute Engine give a *VM*. Install the NVIDIA Container Toolkit (layer 02), or run
  `pip install "vllm==0.30.0"`. Then run the scripts in this folder.
* Run the benchmark from the same machine (`127.0.0.1`), unless you want the network in your TTFT.

## Cost and cleanup

Colab and Kaggle are free within their weekly GPU quotas (there is no guarantee). Rented machines bill until you
**terminate** them, not when vLLM stops. Stop the server with `Ctrl-C`. Then terminate the pod or instance.
`bench_layouts.sh` stops each server itself. When you finish, delete `./results`.
