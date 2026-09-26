# quant-lab — quantize a checkpoint, serve it, and measure both the speed it buys and the accuracy it costs

After this lab you can produce an FP8 or INT4 checkpoint and say, before paying for a GPU, which
kernel vLLM will run it with on that GPU, how much faster decode and prefill get, how many more
sessions fit, and whether the accuracy you lose is real — first on a bundled 300 K-parameter model
on a laptop, then with llm-compressor, `vllm serve` and lm-evaluation-harness on a real GPU.

## Start here

1. `python3 -m pip install -e ".[dev]" && python3 -m quantlab plan --gpu T4 --scheme fp8` — under a
   second: what an FP8 checkpoint does on a free Colab T4 (it loads, but runs weight-only), the kernel to
   look for in vLLM's log, and the command.
2. Open [`notebooks/01_quantize_a_checkpoint.ipynb`](notebooks/01_quantize_a_checkpoint.ipynb) (T0):
   quantize the bundled model with FP8_DYNAMIC and GPTQ, write a compressed-tensors checkpoint, read it back.
3. `python3 -m quantlab eval` — the accuracy cost of each scheme on the bundled model in ~5 seconds;
   notebook 03 explains why round-to-nearest INT4 fails where GPTQ at the same 4.125 bits does not.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T1 = one small GPU (Colab/Kaggle T4 or a rented 24 GB card);
T2 = a multi-GPU box, rented for an hour; T3 = the Google Cloud deployment, optional.* Each notebook
opens with *the one-minute version*, works examples against the library, then 4–5 exercises
(implement the key function, predict a number, pick a setting) each followed by a check that prints
✅, and closes with *in a design review*. Answers are in [`solutions/`](solutions/). Section numbers
refer to the topic's [`PRIMER.md`](../PRIMER.md).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`quantize_a_checkpoint`](notebooks/01_quantize_a_checkpoint.ipynb) | T0 (+T1 llm-compressor) | what a compressed-tensors checkpoint holds (packed INT4 words, FP8 bytes, bf16 scales, the `quantization_config`); predicting its size; why RTN INT4 loses the columns that meet outlier activations and GPTQ/AWQ do not; the group-size divisibility rule; the llm-compressor recipe for a 0.5B model | §3, §4, §9 | ~2 h |
| 02 | [`serve_and_compare_schemes`](notebooks/02_serve_and_compare_schemes.ipynb) | T0 simulated / T1 | the scheme × GPU table (FP8 is weight-only below sm_89, NVFP4 W4A4 only on sm_100+, INT8 W8A8 refused on Blackwell); one GEMM on the roofline (vllm-internals §8.1's table, recomputed); where INT4's edge fades (~120 tokens per step on an L4); TTFT and TPOT per scheme from an emulator and over HTTP; picking a scheme per GPU and workload | §1, §4, §10 | ~2 h |
| 03 | [`measure_the_accuracy_cost`](notebooks/03_measure_the_accuracy_cost.ipynb) | T0 / T1 lm-eval | KL vs argmax agreement vs task accuracy on nine schemes; where the errors land; how many questions a drop needs to be visible; paired flips (McNemar); an accuracy budget and the cheapest scheme that meets it; long generations compound damage; lm-eval commands and results | §8 | ~2 h |
| 04 | [`kv_cache_quantization_in_vllm`](notebooks/04_kv_cache_quantization_in_vllm.ipynb) | T0 / T1 on Ada or newer | `--kv-cache-dtype fp8`: 2,363 → 4,727 blocks for an 8B model on an L4 (the serving lab's sizing, reproduced); which attention backend reads which KV dtype (none on a T4; FlashInfer on L4/A100); where the KV read passes the weight read; the per-tensor scale that ruins FP8 KV and the one that does not matter | §6 | ~1.5 h |
| 05 | [`fp4_and_the_blackwell_path`](notebooks/05_fp4_and_the_blackwell_path.ipynb) | T0 (calculators, verify-marked) | E2M1 and its rounding; MXFP4's power-of-two scale vs NVFP4's two-level scale; checkpoint layouts and bytes; why FP4 *activations* collapse without smoothing; why weight-only FP4 is slower than BF16 at prefill sizes on Blackwell | §2, §5 | ~1.5 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the bundled tiny model, the numpy recipes, the fake server, all calculators | every notebook, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | llm-compressor on a 0.5–1.5B model, `vllm serve` per scheme, lm-eval | [`deploy/any-gpu/`](deploy/any-gpu/), `QUANTLAB_URL=...`, `QUANTLAB_RUN_T1=1` |
| **T2** | Blackwell (B200, RTX PRO 6000), rented | NVFP4 W4A4 | notebook 05's commands (verify) |
| **T3** | GCP | a quantized model on the serving lab's Cloud Run (L4, scale to zero) or GKE | [`deploy/gcp/`](deploy/gcp/) — variables for the serving lab's Terraform, no new infrastructure |

Prices and where to get GPUs: [`COMPUTE.md`](../../../COMPUTE.md).

## Run it

```bash
cd quant-lab
python3 -m pip install -e ".[dev]"            # numpy; dev: pytest, jupyter, pyyaml
python3 -m pytest -q                          # ~70 tests, ~10 s, offline, no GPU
python3 -m quantlab plan --gpu L4             # every scheme on an L4: runs? kernel? flags?
python3 -m quantlab compress --scheme W4A16 --algo gptq --out out/tiny-W4A16   # a compressed-tensors checkpoint
python3 -m quantlab kv --model llama-3.1-8b-instruct --gpu L4                   # blocks and sessions per weight/KV dtype
python3 -m quantlab bench --schemes bf16,fp8,w4a16                              # simulated TTFT/TPOT per scheme
python3 -m quantlab fake --scheme w4a16 --port 8000 &                           # a fake vLLM that runs at INT4 speed
python3 -m quantlab bench --url http://127.0.0.1:8000                           # measure it over HTTP (labelled simulated)
python3 -m jupyterlab notebooks                                                 # the exercises; answers in solutions/
```

On a GPU (T1): `SCHEME=FP8_DYNAMIC deploy/any-gpu/compress.sh` produces a real checkpoint with
llm-compressor in its own environment; `SCHEME=fp8 MODEL=./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC deploy/any-gpu/serve.sh`
serves it after checking the scheme against the GPU; then `export QUANTLAB_URL=http://127.0.0.1:8000`
and the notebooks measure the real server (`QUANTLAB_RUN_T1=1` lets notebooks 01 and 03 run
llm-compressor and lm-eval themselves).

## The library (`quantlab/`, ~3,100 lines)

| Module | The idea |
|---|---|
| `compress.py` | compressed-tensors presets (`FP8_DYNAMIC`, `FP8`, `W8A8`, `W8A16`, `W4A16`, `W4A16_ASYM`, `NVFP4A16`, `NVFP4`); RTN, GPTQ, AWQ and SmoothQuant in numpy; write the checkpoint with compressed-tensors' names, dtypes, shapes and packing; a loader and a validator; the llm-compressor 0.14 and GPTQModel scripts for T1 |
| `serve.py` | which scheme each GPU runs natively and which vLLM kernel it should log (from vLLM v0.30.0 / main source, verify); `vllm serve` / `docker run` commands; checkpoint → vLLM scheme class and minimum capability with vLLM's error text; Cloud Run variables and GKE args for the serving lab's deploy; a startup-log parser |
| `bench.py` | one GEMM on the roofline (reproduces vllm-internals §8.1); a scheme-aware step-time model; a continuous-batching emulator; closed-loop simulation; a streaming HTTP client that measures TTFT/ITL/TPOT like `vllm bench serve` |
| `fakeserver.py` | the emulator behind `/v1/completions` (SSE), `/v1/models`, `/health`, `/version` (`"simulated": true`), `/metrics` with vLLM's names — standard library only |
| `evalharness.py` | `lm_eval` command lines (vllm / hf / local-completions), results and table parsers, unpaired comparison, Wilson intervals, exact McNemar, logit KL; the offline mini-eval |
| `kv.py` | KV bytes per token, blocks and sessions (reproduces `servelab.sizing`), attention backend per GPU and KV dtype, KV quantizer emulation and scale calibration |
| `fp4.py` | E2M1 rounding and packing, NVFP4 (E4M3 per 16 + FP32 global) and MXFP4 (E8M0 per 32) quantizers in compressed-tensors' conventions, checkpoint layouts and bits per weight |
| `report.py` | tables whose every section says where its numbers came from: exact, simulated, measured, sample |
| `numerics.py`, `stio.py` | FP8 E4M3/E5M2 and bf16 bit patterns (checked against torch's casts when torch is present); INT4 packing in compressed-tensors order; safetensors read/write with no dependencies (checked against the `safetensors` package when present) |
| `tinymodel.py`, `data/tiny-adder/` | the bundled model: a 2-layer Llama (RMSNorm, RoPE, GQA, SwiGLU; `LlamaForCausalLM` tensor names; bf16 safetensors) trained by [`tools/train_tiny.py`](tools/train_tiny.py) to 100% on 3-digit addition and 6-digit reversal, with two massive-activation channels planted by a function-preserving rescaling so calibration matters as it does at scale |

This lab never imports the topic's core (`quant-core`) or the serving lab; where a formula already
has a home in the repo the tests pin its numbers: `servelab.sizing`'s block counts (and, when the
serving lab is in the checkout, a live cross-check against its code), vllm-internals §8.1's GEMM table
and §8.2's decode floors, and compressed-tensors' INT4 packing (`0xfcba9810`).

## Deploy

[`deploy/`](deploy/) — [`any-gpu/`](deploy/any-gpu/) (`compress.sh`: llm-compressor in its own
virtualenv; `serve.sh`: `vllm serve` or `docker run` per scheme after checking it against the GPU's
compute capability; Colab/Kaggle T4 INT4 recipe; RunPod/Vast 4090/L4 notes for FP8) and
[`gcp/`](deploy/gcp/) (the serving lab's Cloud Run Terraform and GKE manifests with quantized-model
variables, and a wrapper that drives them). Each has a README with cost and cleanup.

## Regenerating notebooks and the tiny model

```bash
python3 tools/build_notebooks.py                        # notebooks/ and solutions/ from notebooks_src/
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
python3 tools/train_tiny.py                             # re-create data/tiny-adder (needs torch; ~25 min on a busy CPU)
make check                                              # tests + notebooks + bash -n
```

## Caveats

- **Simulated vs measured.** Speeds from `bench` and the fake server are a roofline emulator with
  assumed efficiencies (60% of peak FLOP/s, 80% of bandwidth, 2 ms per step) and are labelled
  SIMULATED everywhere; a real `vllm serve` in `QUANTLAB_URL` gives measurements. The lm-eval results
  and startup logs in `quantlab/data/samples/` are *sample output in the documented format
  (illustrative)*, not measurements.
- **The tiny model is not your model.** It shows mechanisms (outlier channels, GPTQ's compensation,
  FP4 activations collapsing, scale saturation) with real numbers, but the size of an accuracy drop is
  model- and task-specific: gate on your own evals (notebook 03).
- **The scheme table is read from source, not run.** Kernel names, capability floors and backend rules
  come from vLLM v0.30.0 / main (Sep 2026); the startup log is the ground truth on your version.
- **T1 paths are written, not executed here** (no GPU in this environment): llm-compressor, vLLM,
  lm-eval, Docker and the GCP wrapper are checked with `bash -n`, `DRY_RUN=1`, `ast.parse` of the
  generated scripts, and Kubernetes schema validation of the serving lab's Deployment with each scheme's args.

## Verify list (dated 2026-09-26; re-check when you move the pins)

- Versions: vLLM **0.30.0** (`vllm/vllm-openai:v0.30.0`), llm-compressor **0.14.0**, compressed-tensors
  **0.19.0**, lm-eval **0.4.13**, GPTQModel **7.5.0** (PyPI, 2026-09-26).
- `--quantization fp8` on a bf16 checkpoint works in v0.30.0 and raises on main; `fp8_per_tensor` works in
  both. bitsandbytes and GGUF moved to out-of-tree plugins (`vllm-bnb-plugin`, `vllm-gguf-plugin`).
- Capability floors: Marlin sm_75; Machete sm_90 only; CUTLASS FP8 sm_89 (CUDA ≥ 12.4); CUTLASS block-FP8
  sm_90+; CUTLASS NVFP4 sm_100–12x with CUDA ≥ 12.8; `CompressedTensorsW8A8Fp8` 89 (else W8A16 via Marlin);
  INT8 W8A8 not supported on compute capability ≥ 10.0 (vLLM docs).
- FP8 KV: no backend on sm_75; FlashAttention reads FP8 KV only as FA3 (sm_90) / FA4 (sm_10x); FlashInfer
  from sm_80; default `k_scale`/`v_scale` 1.0; the log wording of the attention-backend line.
- Model ids: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`, `Qwen/Qwen2.5-0.5B-Instruct(-AWQ, -GPTQ-Int4)`; the
  `ultrachat_200k` `train_sft` split name for calibration.
- Datasheet numbers in `serve.GPUS` (dense TFLOP/s, bandwidth, driver-reported memory); the RTX 4090's
  FP8/INT8 rates and the B200's usable memory are unverified. The ModelOpt W4A16-vs-BF16 Blackwell
  measurement (announcement dated 2026-09-16).

MIT licensed.
