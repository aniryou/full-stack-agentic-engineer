# moe-lab — watch a mixture-of-experts model route, stream, split and squeeze

After this lab, you can do five things. You do each first on a laptop with simulated or bundled data, labelled as
such. Then you do it on real GPUs with vLLM:

- Train a tiny MoE, and watch its router collapse and recover.
- Record which experts the tokens of a real model select.
- Measure how the decode step of an MoE grows with batch, while the decode step of a dense model stays flat.
- Compare tensor parallelism and expert parallelism on two GPUs.
- Fit a 7B-total MoE on a free 16 GB GPU.

## Start here

1. Run `python3 -m pip install -e ".[dev]" && python3 -m moelab fit --model olmoe-1b-7b --gpu T4`. It takes less
   than a second. It shows why OLMoE-1B-7B has no room for a KV cache on a 16 GB T4 in 16-bit. The model has
   1.3 B active and 6.9 B total. It also shows what `--cpu-offload-gb` or 4-bit experts cost.
2. Open [`notebooks/01_a_tiny_moe_in_torch.ipynb`](notebooks/01_a_tiny_moe_in_torch.ipynb) (T0). A tiny MoE trains
   on your CPU in seconds (torch). Without torch, the notebook reads curves that this code recorded. You see the
   collapse, then two corrections.
3. With any GPU (a free Colab T4 is sufficient), [`deploy/any-gpu/`](deploy/any-gpu/) starts vLLM with a small MoE.
   Set `MOELAB_URL`. Then notebooks 02, 03 and 05 measure that server and do not simulate it.

## What you get

*Tiers: T0 = laptop or Colab CPU, free. T1 = one small GPU (Colab/Kaggle T4 or a rented card). T2 = a multi-GPU
box (Kaggle's free "GPU T4 x2", or a pair rented for an hour). T3 = the Google Cloud deployment, optional.*

Each notebook starts with *the one-minute version*. Then it does worked examples with the library. Then it gives
5–6 exercises: implement the key function, predict a number, or select a setting. After each exercise, a check
prints ✅. The notebook ends with *in a design review*.

The answers are in [`solutions/`](solutions/). The concepts
are in the [`PRIMER.md`](../PRIMER.md) of the topic. The numpy core next to this lab, [`../moe-core/`](../moe-core/),
predicts what this lab measures. The lab never imports it.

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`a_tiny_moe_in_torch`](notebooks/01_a_tiny_moe_in_torch.ipynb) | T0 (torch on CPU, or bundled curves without torch) | The router, then top-k, then the combine step, in the same form as HF v5 models. Total against active. Top-1 collapse: the busiest expert at 6–7× its share, five or six of eight experts dead, a worse loss. Two corrections: the Switch loss (k at balance, HF's normalisation) and the choose-only bias of DeepSeek-V3. What imbalance costs an EP step. Experts specialise by token far more than by domain. | §2, §3, §4 | ~1.5 h |
| 02 | [`watch_the_router`](notebooks/02_watch_the_router.ipynb) | T1 (T0: illustrative traces) | How to capture the routing with forward hooks (three router output layouts) or with vLLM's `--enable-return-routed-experts`. Utilisation, and the balancedness of EPLB against a uniform baseline at the same sample size. Domain divergence by layer. Why the same routing does more damage at EP 16 than at EP 2. | §3.7, §6.3 | ~1.5 h |
| 03 | [`batch_vs_weight_stream`](notebooks/03_batch_vs_weight_stream.ipynb) | T1 (T0: simulated) | Experts touched, E(1−(1−k/E)^T), as a closed form and as a skewed Monte Carlo. Bytes per step and the crossover batch. Layer 01's Mixtral table, reproduced (25.6 GB, 5.34 ms, and 754 against 207). OLMoE against a dense 1.5B: step time against batch. The padding of vLLM's `moe_align_block_size`. How to calibrate the simulation from two measured points. | §5, §6.1 | ~2 h |
| 04 | [`expert_parallelism_on_two_gpus`](notebooks/04_expert_parallelism_on_two_gpus.ipynb) | T2 (T0: simulated) | TP against TP+EP against DP+EP in vLLM: what each GPU holds, and which collectives run (no all-to-all with DP = 1). Latency-bound decode messages. All-to-all bytes compared with layer 02 and DeepEP. When hot experts make a GPU slow (tokens in compute-bound steps, not bytes in decode). How to read `vllm bench serve`. | §6.2–§6.5, §8 | ~2 h |
| 05 | [`moe_on_a_small_gpu`](notebooks/05_moe_on_a_small_gpu.ipynb) | T1 (T0: sizing) | Checkpoint bytes for INT4, MXFP4 and FP8. KV room and sessions per GPU. The correct `--cpu-offload-gb`. UVA offload as a per-step toll against the CPU experts of llama.cpp, and the batch where the two cross. How to read the start-up log of vLLM, with the fused-MoE default-config warning. | §6.6, §6.7, §7, §8 | ~1.5 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the tiny MoE (torch on CPU, optional), every simulator, bundled traces, bench results and logs | all notebooks, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), a 24 GB L4 or RTX 4090 | `vllm serve` with OLMoE-1B-7B, granite-3.0 MoE, Qwen1.5-MoE INT4, Qwen3-30B-A3B INT4 (24 GB), gpt-oss-20b (L4, not T4). Also transformers with hooks. | `MOELAB_URL=...` or `MOELAB_START_VLLM=1`. Also [`deploy/any-gpu/`](deploy/any-gpu/). |
| **T2** | two GPUs: Kaggle "GPU T4 x2" (free, PCIe) or a rented pair | TP, TP+EP and DP+EP, `vllm bench serve` for each | notebook 04, [`deploy/any-gpu/bench_layouts.sh`](deploy/any-gpu/bench_layouts.sh) |
| **T3** | GCP: layer 02's GKE cluster, its 2 × L4 `l4x2` pool | Qwen1.5-MoE-A2.7B with `--enable-expert-parallel` (28.6 GB: more than one L4) | [`deploy/gke/`](deploy/gke/). Manifests only, no new Terraform. |

For prices and for places to get GPUs, see [`COMPUTE.md`](../../../COMPUTE.md).

## Run it

```bash
cd moe-lab
python3 -m pip install -e ".[dev]"             # numpy + notebook/test tooling (torch optional: ".[torch]")
python3 -m pytest -q                           # 126 tests, ~40 s with torch (it trains the toy), offline; torch tests skip without
python3 -m moelab models                       # total / active (two conventions) / KV per token, ten models
python3 -m moelab touched --experts 64 --top-k 8 --batch 1 8 64 256
python3 -m moelab stream --model olmoe-1b-7b --dense qwen2.5-1.5b --gpu L4    # ITL vs batch [simulated]
python3 -m moelab ep --gpu T4 --link pcie-2xT4                                 # TP vs EP on 2 GPUs [simulated]
python3 -m moelab trace                        # the bundled router traces [illustrative]
python3 -m moelab.tinymoe --steps 400 --seeds 0 1 2    # train the tiny MoE three ways (needs torch; ~5 s a run)
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

To measure a real engine instead of the simulation (T1/T2), start one. [`deploy/any-gpu/`](deploy/any-gpu/) has
Colab and Kaggle recipes. Then run `export MOELAB_URL=http://127.0.0.1:8000`. Also set `MOELAB_DENSE_URL` for a
dense model to compare, and `MOELAB_API_KEY` if the server has one.

On a GPU machine with vLLM installed, `MOELAB_START_VLLM=1` lets notebooks 03–05 start and stop `vllm serve`
themselves. `MOELAB_HF_MODEL=allenai/OLMoE-1B-7B-0924-Instruct` makes notebook 02 load the model through
transformers and attach hooks to its routers. `MOELAB_NO_TORCH=1` runs everything on the numpy-only path. A machine
without torch also uses this path.

## The library (`moelab/`, ~2,100 lines)

| Module | Lines | The idea |
|---|---:|---|
| `configs.py` | ~260 | Ten models as literal config fields (Mixtral, Qwen3-30B-A3B, OLMoE, granite-3.0 MoE ×2, Qwen1.5-MoE, gpt-oss ×2, two dense references). From these fields: total, active under three embedding conventions, KV per token, checkpoint bytes for fp16/fp8/INT4/INT4-experts/MXFP4. A table of GPU datasheets. The entry for a served model id. |
| `stream.py` | ~290 | Experts touched (closed form, Zipf Monte Carlo, Gumbel top-k). Bytes and FLOPs of a decode step and the crossover batch (layer 01's `roofline.llm`, re-implemented and reproduced). vLLM's `moe_align_block_size` layout. Simulated ITL and its calibration. A closed-loop streaming client that measures ITL. |
| `hooks.py` | ~300 | vLLM's `routed_experts` wire format. Utilisation / balancedness / hot experts / domain divergence. EP placement and per-rank load. `RouterRecorder` (forward hooks for the HF v5 router layouts). T1 capture through transformers or vLLM (the expert count from the catalogue entry of the served model, k from the data). |
| `ep.py` | ~260 | The three two-GPU layouts and their vLLM flags. All-to-all bytes, alpha-beta collectives, per-GPU weights. A per-GPU per-layer step model (the slowest GPU of each layer). The `vllm bench serve` command and parser. |
| `offload.py` | ~150 | What fits (KV room, sessions). Capability rules (bf16, FP8, MXFP4). The smallest `--cpu-offload-gb`. UVA toll against CPU experts. The start-up log parser. |
| `tinymoe/` | ~470 | The toy task (numpy). `TinyTopKRouter`/`TinyMoE`/`TinyMoETransformer` (torch, HF-shaped). Switch loss, z-loss, the choose-only bias. A trainer that records load and specialisation. Statistics for the bundled curves. |
| `env.py`, `report.py`, `__main__.py` | ~410 | Tier detection and `VLLMServer`. JSON/Markdown reports labelled measured / simulated / illustrative. The CLI. |

[`moelab/fixtures/`](moelab/fixtures/README.md) holds what the notebooks read without a GPU. It holds tiny-MoE
curves *recorded* on a CPU. It also holds router traces, `vllm bench serve` results and start-up logs as *sample
output in the documented format (illustrative)*. `python3 tools/make_fixtures.py` makes these files again.

## Deploy

[`deploy/`](deploy/) has two targets:

- [`any-gpu/`](deploy/any-gpu/) has these parts:
  - `serve_moe.sh`: one GPU with offload, or two in any layout, with docker or pip, and `DRY_RUN=1`.
  - `bench_layouts.sh`: the TP against EP comparison with `vllm bench serve`.
  - Colab, Kaggle and rented-GPU recipes.
  - An explanation of every MoE flag.
- [`gke/`](deploy/gke/) has a vLLM Deployment with `--enable-expert-parallel` on layer 02's `l4x2` pool, a CPU
  benchmark Job, and `run.sh` to change layouts.

Each target has a README with cost and cleanup. There is no Terraform here. The cluster is layer 02's
[`cuda-nccl-lab/deploy/gcp/terraform/`](../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/).

## Regenerating notebooks

The builder generates `notebooks/` (exercises) and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network), ~35 s
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
MOELAB_NO_TORCH=1 python3 tools/run_notebooks.py solutions   # and run without torch too
make check                                              # all of the above + tests + bash -n + kubernetes-validate
```

## Caveats

- **Three labels.** *Measured* numbers come only from a GPU or from a server that you connect the lab to.
  *Simulated* numbers come from the roofline (and an alpha-beta link) with stated efficiencies. `stream.SimParams`
  gives 70% of bandwidth, 50% of FLOP/s and 2.5 ms per step. The `ep.LINKS` values for PCIe pairs are assumptions.
  Notebook 03 shows how to calibrate the stated efficiencies of the simulation. *Illustrative* fixtures have the
  documented shape, and the lab made them from its own models. Thus, agreement with them proves only that the parsers work.
- **The toy is a toy.** The collapse of the tiny MoE and its token-over-domain specialisation are real outputs of
  this code on a toy task. The collapse numbers are max/mean 5.9–6.8 and 5–6 dead experts without load balance,
  and 1.04–1.17 with it, for seeds 0–2. These outputs show the mechanism, not the magnitude in a real model. The
  collapse depends on a direction that every token embedding shares (`TrainConfig.common`). Real hidden states
  also share one such direction. Without it (`--common 0`), the toy only drifts to ~2× on the busiest expert.
- **What the step model leaves out.** The step model leaves out four effects. The first is kernel efficiency at
  small GEMM shapes (TP's half-width experts). The others are the overlap of communication with compute,
  CUDA-graph and scheduler effects, and prefill interference in decode. The measurement is the judge.
- **Not run here.** `bash -n`, `DRY_RUN=1`, schema validation and offline tests examine the GPU, Docker and GKE
  paths. No check runs these paths on real hardware.

## Verify list (facts dated September 2026 that move)

These facts come from a check against source on 2026-09-26. The sources are vLLM **v0.30.0** where an item gives
no other version, transformers `27166ea`, and the deepep, gpt-oss and olmoe repos:

* vLLM flags:
  * `--enable-expert-parallel`/`-ep`, `--data-parallel-size`.
  * `--all2all-backend`: the default is `allgather_reducescatter`. `pplx`/`naive` are removed. There is no
    `VLLM_ALL2ALL_BACKEND`.
  * `--expert-placement-strategy`: `linear` is the default. vLLM applies `round_robin` only to grouped models with
    no redundant experts and with EPLB off. This is true for v0.30.0 and for main `a4eb3f2`.
  * `--enable-eplb` (window 1000, step interval 3000).
  * `--cpu-offload-gb`, `--cpu-offload-params`.
  * `--enable-return-routed-experts` (base64 `.npy`, `(tokens − 1, layers, top_k)`, request field
    `routed_experts_prompt_start`).
* With DP = 1, `--enable-expert-parallel` runs no all-to-all kernels. For `use_all2all_kernels`, DP > 1 is
  necessary, and the MoE output goes through an all-reduce. The sources are `fused_moe/config.py` and `layer.py`
  at v0.30.0, and the reduce of the runner on main.
* The semantics of `moe_align_block_size` (the lab reproduces the docstring example). The default fused-MoE config
  tiles and the list of tuned config files (no T4/L4/A10) come from main `a4eb3f2`.
* The result block and the flags of `vllm bench serve` (`benchmarks/serve.py`, `benchmarks/datasets/datasets.py`,
  v0.30.0). The words of the start-up log, as the check of layer 04's `servelab.sizing` found them.
* `Mxfp4Config.get_min_capability()` = 80 with bf16 activations (gpt-oss does not run on a T4).

These items still need a check, and a check is not possible here:

- the Hub ids of the checkpoints (`...-instruct`, `-GPTQ-Int4`, GGUF files),
- the derived configs marked `verify=True` in `configs.py` (OLMoE expert width, granite-3.0 MoE, gpt-oss-20b's 24
  layers),
- the effective PCIe bandwidths (`GPU.pcie_gbs`),
- the `ep.LINKS` PCIe alpha-beta values,
- the minimum driver of CUDA 13 (580), and the drivers that Colab, Kaggle and GKE's `DEFAULT` supply,
- Colab's RAM for pinned offload,
- if llama-server obeys `ignore_eos`,
- that `vllm bench serve` starts without a GPU in the Job image.

The licence is MIT.
