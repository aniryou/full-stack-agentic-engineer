# moe-lab — watch a mixture-of-experts model route, stream, split and squeeze

After this lab you can train a tiny MoE and watch its router collapse and recover, record which experts a real
model's tokens pick, measure how an MoE's decode step grows with batch where a dense model's stays flat, compare
tensor and expert parallelism on two GPUs, and fit a 7B-total MoE on a free 16 GB GPU — each first on a laptop
(simulated or bundled data, labelled), then on real GPUs with vLLM.

## Start here

1. `python3 -m pip install -e ".[dev]" && python3 -m moelab fit --model olmoe-1b-7b --gpu T4` — under a second:
   why OLMoE-1B-7B (1.3 B active, 6.9 B total) has no room for a KV cache on a 16 GB T4 in 16-bit, and what
   `--cpu-offload-gb` or 4-bit experts would cost.
2. Open [`notebooks/01_a_tiny_moe_in_torch.ipynb`](notebooks/01_a_tiny_moe_in_torch.ipynb) (T0): a tiny MoE trains
   on your CPU in seconds (torch; without torch it reads curves this code recorded) — collapse, then two fixes.
3. With any GPU (a free Colab T4 is enough): [`deploy/any-gpu/`](deploy/any-gpu/) starts vLLM with a small MoE;
   set `MOELAB_URL` and notebooks 02, 03 and 05 measure it instead of simulating.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented card); T2 = a multi-GPU
box (Kaggle's free "GPU T4 x2", or a pair rented for an hour); T3 = the Google Cloud deployment, optional.* Each
notebook opens with *the one-minute version*, works examples against the library, then 5–6 exercises (implement
the key function, predict a number, pick a setting), each followed by a check that prints ✅, and closes with
*in a design review*. Answers are in [`solutions/`](solutions/). Concepts are in the topic's
[`PRIMER.md`](../PRIMER.md); the numpy core next door, [`../moe-core/`](../moe-core/), predicts what this lab
measures (the lab never imports it).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`a_tiny_moe_in_torch`](notebooks/01_a_tiny_moe_in_torch.ipynb) | T0 (torch on CPU; bundled curves without) | router → top-k → combine, written the way HF v5 models are; total vs active; top-1 collapse (the busiest expert at 6–7× its share, five or six of eight experts dead, a worse loss) and two fixes — the Switch loss (k at balance, HF's normalisation) and DeepSeek-V3's choose-only bias; what imbalance costs an EP step; experts specialise by token far more than by domain | §2, §3, §4 | ~1.5 h |
| 02 | [`watch_the_router`](notebooks/02_watch_the_router.ipynb) | T1 (T0: illustrative traces) | capturing routing with forward hooks (three router output layouts) or vLLM's `--enable-return-routed-experts`; utilisation, EPLB's balancedness against a uniform baseline at the same sample size; domain divergence by layer; why the same routing hurts more at EP 16 than at EP 2 | §3.7, §6.3 | ~1.5 h |
| 03 | [`batch_vs_weight_stream`](notebooks/03_batch_vs_weight_stream.ipynb) | T1 (T0: simulated) | experts touched E(1−(1−k/E)^T), closed form and skewed Monte Carlo; bytes per step and the crossover batch — layer 01's Mixtral table (25.6 GB, 5.34 ms; 754 vs 207) reproduced; OLMoE vs a dense 1.5B, step time vs batch; vLLM's `moe_align_block_size` padding; calibrating the simulation from two measured points | §5, §6.1 | ~2 h |
| 04 | [`expert_parallelism_on_two_gpus`](notebooks/04_expert_parallelism_on_two_gpus.ipynb) | T2 (T0: simulated) | TP vs TP+EP vs DP+EP in vLLM: what each GPU holds, which collectives run (no all-to-all with DP = 1), latency-bound decode messages, all-to-all bytes checked against layer 02 and DeepEP; when hot experts slow a GPU (tokens in compute-bound steps, not bytes in decode); reading `vllm bench serve` | §6.2–§6.5, §8 | ~2 h |
| 05 | [`moe_on_a_small_gpu`](notebooks/05_moe_on_a_small_gpu.ipynb) | T1 (T0: sizing) | checkpoint bytes for INT4, MXFP4 and FP8; KV room and sessions per GPU; the right `--cpu-offload-gb`; UVA offload as a per-step toll vs llama.cpp's CPU experts, and the batch where they cross; reading vLLM's start-up log, including the fused-MoE default-config warning | §6.6, §6.7, §7, §8 | ~1.5 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the tiny MoE (torch on CPU, optional), every simulator, bundled traces, bench results and logs | all notebooks, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), a 24 GB L4 or RTX 4090 | `vllm serve` with OLMoE-1B-7B, granite-3.0 MoE, Qwen1.5-MoE INT4, Qwen3-30B-A3B INT4 (24 GB), gpt-oss-20b (L4, not T4); transformers + hooks | `MOELAB_URL=...` or `MOELAB_START_VLLM=1`; [`deploy/any-gpu/`](deploy/any-gpu/) |
| **T2** | two GPUs: Kaggle "GPU T4 x2" (free, PCIe) or a rented pair | TP, TP+EP and DP+EP, `vllm bench serve` each | notebook 04; [`deploy/any-gpu/bench_layouts.sh`](deploy/any-gpu/bench_layouts.sh) |
| **T3** | GCP: layer 02's GKE cluster, its 2 × L4 `l4x2` pool | Qwen1.5-MoE-A2.7B with `--enable-expert-parallel` (28.6 GB: more than one L4) | [`deploy/gke/`](deploy/gke/) — manifests only, no new Terraform |

Prices and where to get GPUs: [`COMPUTE.md`](../../../COMPUTE.md).

## Run it

```bash
cd moe-lab
python3 -m pip install -e ".[dev]"             # numpy + notebook/test tooling (torch optional: ".[torch]")
python3 -m pytest -q                           # 112 tests, ~6 s, offline; torch tests skip without torch
python3 -m moelab models                       # total / active (two conventions) / KV per token, ten models
python3 -m moelab touched --experts 64 --top-k 8 --batch 1 8 64 256
python3 -m moelab stream --model olmoe-1b-7b --dense qwen2.5-1.5b --gpu L4    # ITL vs batch [simulated]
python3 -m moelab ep --gpu T4 --link pcie-2xT4                                 # TP vs EP on 2 GPUs [simulated]
python3 -m moelab trace                        # the bundled router traces [illustrative]
python3 -m moelab.tinymoe --steps 400 --seeds 0 1 2    # train the tiny MoE three ways (needs torch; ~5 s a run)
python3 -m jupyterlab notebooks                # the exercises; answers in solutions/
```

Measure a real engine instead (T1/T2): start one ([`deploy/any-gpu/`](deploy/any-gpu/) has Colab and Kaggle
recipes), then `export MOELAB_URL=http://127.0.0.1:8000` (and `MOELAB_DENSE_URL` for a dense model to compare,
`MOELAB_API_KEY` if the server has one). On a GPU machine with vLLM installed, `MOELAB_START_VLLM=1` lets
notebooks 03–05 start and stop `vllm serve` themselves; `MOELAB_HF_MODEL=allenai/OLMoE-1B-7B-0924-Instruct` makes
notebook 02 load the model through transformers and hook its routers. `MOELAB_NO_TORCH=1` runs everything down the
numpy-only path (what a machine without torch sees).

## The library (`moelab/`, ~2,100 lines)

| Module | Lines | The idea |
|---|---:|---|
| `configs.py` | ~240 | ten models as literal config fields (Mixtral, Qwen3-30B-A3B, OLMoE, granite-3.0 MoE ×2, Qwen1.5-MoE, gpt-oss ×2, two dense references) → total, active under three embedding conventions, KV per token, checkpoint bytes for fp16/fp8/INT4/INT4-experts/MXFP4; GPU datasheet table |
| `stream.py` | ~290 | experts touched (closed form, Zipf Monte Carlo, Gumbel top-k), bytes and FLOPs of a decode step and the crossover batch (layer 01's `roofline.llm`, re-implemented and reproduced), vLLM's `moe_align_block_size` layout, simulated ITL and its calibration, a closed-loop streaming client that measures ITL |
| `hooks.py` | ~275 | vLLM's `routed_experts` wire format, utilisation / balancedness / hot experts / domain divergence, EP placement and per-rank load, `RouterRecorder` (forward hooks for the HF v5 router layouts), T1 capture through transformers or vLLM |
| `ep.py` | ~260 | the three two-GPU layouts and their vLLM flags, all-to-all bytes, alpha-beta collectives, per-GPU weights, a per-GPU per-layer step model (the slowest GPU of each layer), `vllm bench serve` command and parser |
| `offload.py` | ~150 | what fits (KV room, sessions), capability rules (bf16, FP8, MXFP4), the smallest `--cpu-offload-gb`, UVA toll vs CPU experts, the start-up log parser |
| `tinymoe/` | ~470 | the toy task (numpy), `TinyTopKRouter`/`TinyMoE`/`TinyMoETransformer` (torch, HF-shaped), Switch loss, z-loss, the choose-only bias, a trainer that records load and specialisation, statistics for the bundled curves |
| `env.py`, `report.py`, `__main__.py` | ~410 | tier detection and `VLLMServer`; JSON/Markdown reports labelled measured / simulated / illustrative; the CLI |

[`moelab/fixtures/`](moelab/fixtures/README.md) holds what the notebooks read without a GPU: tiny-MoE curves
*recorded* on a CPU, and router traces, `vllm bench serve` results and start-up logs as *sample output in the
documented format (illustrative)*, regenerated by `python3 tools/make_fixtures.py`.

## Deploy

[`deploy/`](deploy/) — [`any-gpu/`](deploy/any-gpu/) (`serve_moe.sh`: one GPU with offload or two in any layout,
docker or pip, `DRY_RUN=1`; `bench_layouts.sh`: the TP vs EP comparison with `vllm bench serve`; Colab, Kaggle and
rented-GPU recipes; every MoE flag explained) and [`gke/`](deploy/gke/) (a vLLM Deployment with
`--enable-expert-parallel` on layer 02's `l4x2` pool, a CPU benchmark Job, `run.sh` to switch layouts). Each has
a README with cost and cleanup. There is no Terraform here: the cluster is layer 02's
[`cuda-nccl-lab/deploy/gcp/terraform/`](../../../02-cuda-nccl-runtime/cuda-and-nccl/cuda-nccl-lab/deploy/gcp/terraform/).

## Regenerating notebooks

`notebooks/` (exercises) and `solutions/` are generated from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network), ~35 s
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
MOELAB_NO_TORCH=1 python3 tools/run_notebooks.py solutions   # and run without torch too
make check                                              # all of the above + tests + bash -n + kubernetes-validate
```

## Caveats

- **Three labels.** *Measured* numbers come only from a GPU or a server you point the lab at. *Simulated* numbers
  are the roofline (and an alpha-beta link) with stated efficiencies (`stream.SimParams`: 70% of bandwidth, 50% of
  FLOP/s, 2.5 ms per step; `ep.LINKS` for PCIe pairs are assumptions) — notebook 03 shows how to calibrate them.
  *Illustrative* fixtures have the documented shape and were generated from the lab's own models, so agreeing with
  them proves only that the parsers work.
- **The toy is a toy.** The tiny MoE's collapse (max/mean 5.9–6.8 and 5–6 dead experts without balancing,
  1.04–1.17 with it, seeds 0–2) and its token-over-domain specialisation are real outputs of this code on a toy
  task; they show the mechanism, not the magnitude in a real model. The collapse relies on a direction shared by
  every token embedding (`TrainConfig.common`), as real hidden states share one; without it (`--common 0`) the toy
  only drifts to ~2× on the busiest expert.
- **What the step model leaves out.** Kernel efficiency at small GEMM shapes (TP's half-width experts), overlap of
  communication with compute, CUDA-graph and scheduler effects, prefill interference in decode — measurement is the
  judge.
- **Not run here.** The GPU, Docker and GKE paths are checked by `bash -n`, `DRY_RUN=1`, schema validation and
  offline tests, not on real hardware.

## Verify list (facts dated September 2026 that move)

Checked against source on 2026-09-26 (vLLM **v0.30.0** unless noted; transformers `27166ea`; deepep, gpt-oss,
olmoe repos):

* vLLM flags: `--enable-expert-parallel`/`-ep`, `--data-parallel-size`, `--all2all-backend` (default
  `allgather_reducescatter`; `pplx`/`naive` removed; no `VLLM_ALL2ALL_BACKEND`), `--expert-placement-strategy`
  (`linear` default; `round_robin` honoured only for grouped models — checked on main `a4eb3f2`), `--enable-eplb`
  (window 1000, step interval 3000), `--cpu-offload-gb`, `--cpu-offload-params`, `--enable-return-routed-experts`
  (base64 `.npy`, `(tokens − 1, layers, top_k)`, request field `routed_experts_prompt_start`).
* With DP = 1, `--enable-expert-parallel` runs no all-to-all kernels (`use_all2all_kernels` requires DP > 1; the
  MoE output is all-reduced) — `fused_moe/config.py` and `layer.py` at v0.30.0, the runner's reduce on main.
* `moe_align_block_size` semantics (docstring example reproduced); the default fused-MoE config tiles and the list
  of tuned config files (no T4/L4/A10) are from main `a4eb3f2`.
* `vllm bench serve` result block and flags (`benchmarks/serve.py`, `benchmarks/datasets/datasets.py`, v0.30.0);
  start-up log wording as layer 04's `servelab.sizing` verified it.
* `Mxfp4Config.get_min_capability()` = 80 with bf16 activations (gpt-oss not on a T4).

Still to verify (not checkable here): Hub ids of the checkpoints (`...-instruct`, `-GPTQ-Int4`, GGUF files),
derived configs marked `verify=True` in `configs.py` (OLMoE expert width, granite-3.0 MoE, gpt-oss-20b's 24
layers), PCIe effective bandwidths (`GPU.pcie_gbs`), the `ep.LINKS` PCIe alpha-beta values, CUDA 13's minimum
driver (580) and the drivers Colab, Kaggle and GKE's `DEFAULT` ship, Colab's RAM for pinned offload, whether
llama-server honours `ignore_eos`, and that `vllm bench serve` starts without a GPU in the Job image.

MIT licensed.
