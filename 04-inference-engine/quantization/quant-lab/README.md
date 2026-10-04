# quant-lab — quantize a checkpoint, serve it, and measure both the speed it buys and the accuracy it costs

After this lab, you can make an FP8 or INT4 checkpoint. Before you pay for a GPU, you can also tell
these things:

- Which kernel vLLM will use to run the checkpoint on that GPU.
- How much faster decode and prefill become.
- How many more sessions fit.
- If the accuracy that you lose is real.

You do this first on a bundled 300 K-parameter model on a laptop. Then you do it with llm-compressor,
`vllm serve` and lm-evaluation-harness on a real GPU.

## Start here

1. Run `python3 -m pip install -e ".[dev]" && python3 -m quantlab plan --gpu T4 --scheme fp8`. It
   takes less than a second. It shows what an FP8 checkpoint does on a free Colab T4: it loads, but it
   runs weight-only. It also shows the kernel to look for in the log of vLLM, and the command.
2. Open [`notebooks/01_quantize_a_checkpoint.ipynb`](notebooks/01_quantize_a_checkpoint.ipynb) (T0).
   Quantize the bundled model with FP8_DYNAMIC and GPTQ. Write a compressed-tensors checkpoint, then read
   it back.
3. Run `python3 -m quantlab eval`. In ~10 seconds, it shows the accuracy cost of each scheme on the
   bundled model. Notebook 03 explains why round-to-nearest INT4 fails where GPTQ at the same 4.125 bits
   does not fail.

## What you get

*Tiers: T0 is a laptop or Colab CPU, free. T1 is one small GPU (a Colab/Kaggle T4 or a rented 24 GB
card). T2 is a multi-GPU box, rented for an hour (nothing in this lab needs one). T3 is the Google Cloud
deployment, optional.*

Each notebook starts with *the one-minute version*. Then it shows worked examples that use the library.
Then it has 4–5 exercises: implement the key function, predict a number, select a setting. A check that
prints ✅ comes after each exercise. The notebook ends with *in a design review*.

The answers are in [`solutions/`](solutions/). The section numbers refer to the topic's [`PRIMER.md`](../PRIMER.md).

| # | Notebook | Tier | You will be able to explain | Primer | Time |
|---|---|---|---|---|---|
| 01 | [`quantize_a_checkpoint`](notebooks/01_quantize_a_checkpoint.ipynb) | T0 (+T1 llm-compressor) | What a compressed-tensors checkpoint holds (packed INT4 words, FP8 bytes, bf16 scales, the `quantization_config`). How to predict its size. Why RTN INT4 loses the columns that meet outlier activations, and GPTQ/AWQ do not. The divisibility rule for the group size. The llm-compressor recipe for a 0.5B model. | §3, §4, §9 | ~2 h |
| 02 | [`serve_and_compare_schemes`](notebooks/02_serve_and_compare_schemes.ipynb) | T0 simulated / T1 | The scheme × GPU table (FP8 is weight-only below sm_89, NVFP4 W4A4 only on sm_100+, INT8 W8A8 refused on Blackwell). One GEMM on the roofline (the table of vllm-internals §8.1, recomputed). Where W4A16 becomes compute-bound (~120 tokens per step on an L4), and where its advantage over BF16 is gone (~460). TTFT and TPOT per scheme from an emulator and over HTTP. How to select a scheme per GPU and workload. | §1, §4, §10 | ~2 h |
| 03 | [`measure_the_accuracy_cost`](notebooks/03_measure_the_accuracy_cost.ipynb) | T0 / T1 lm-eval | A comparison of KL, argmax agreement and task accuracy on nine schemes. Where the errors occur. How many questions a drop needs to be visible. Paired flips (McNemar). An accuracy budget and the lowest-cost scheme that meets it. Over a long generation, the damage compounds. The lm-eval commands and results. | §8 | ~2 h |
| 04 | [`kv_cache_quantization_in_vllm`](notebooks/04_kv_cache_quantization_in_vllm.ipynb) | T0 / T1 on Ada or newer (`QUANTLAB_VLLM_LOG`, `QUANTLAB_URL`) | `--kv-cache-dtype fp8`: from 2,363 to 4,727 blocks for an 8B model on an L4 (the serving lab's block calculation, reproduced). Which attention backend reads which KV dtype (none on a T4, FlashInfer on L4/A100). Where the KV read becomes larger than the weight read. The per-tensor scale that ruins FP8 KV, and the one that does not matter. | §6 | ~1.5 h |
| 05 | [`fp4_and_the_blackwell_path`](notebooks/05_fp4_and_the_blackwell_path.ipynb) | T0 (calculators, with (verify) tags) / T1 on one Blackwell GPU | E2M1 and its rounding. The power-of-two scale of MXFP4 against the two-level scale of NVFP4. Checkpoint layouts and bytes. Why FP4 *activations* collapse without smoothing. Why weight-only FP4 is slower than BF16 at prefill sizes on Blackwell. | §2, §5 | ~1.5 h |

| Tier | Where | What runs | In this lab |
|---|---|---|---|
| **T0** | laptop, Colab CPU, CI | the bundled tiny model, the numpy recipes, the fake server, all calculators | every notebook, all tests |
| **T1** | one GPU: Colab/Kaggle T4 (free), any 24 GB card | llm-compressor on a 0.5–1.5B model, `vllm serve` per scheme, lm-eval | [`deploy/any-gpu/`](deploy/any-gpu/), `QUANTLAB_URL=...`, `QUANTLAB_RUN_T1=1` |
| **T1** (Blackwell) | one rented B200 or RTX PRO 6000. It is one GPU, but not a small or low-cost one. | NVFP4 W4A4 | notebook 05's commands (verify) |
| **T3** | GCP | a quantized model on the serving lab's Cloud Run (L4, scale to zero) or GKE | [`deploy/gcp/`](deploy/gcp/): variables for the serving lab's Terraform, no new infrastructure |

For prices and where to get GPUs, see [`COMPUTE.md`](../../../COMPUTE.md).

## Run it

```bash
cd quant-lab
python3 -m pip install -e ".[dev]"            # numpy; dev: pytest, jupyter, pyyaml
python3 -m pytest -q                          # 94 tests (one needs Terraform), ~35 s, offline, no GPU
python3 -m quantlab plan --gpu L4             # every scheme on an L4: runs? kernel? flags?
python3 -m quantlab compress --scheme W4A16 --algo gptq --out out/tiny-W4A16   # a compressed-tensors checkpoint
python3 -m quantlab kv --model llama-3.1-8b-instruct --gpu L4                   # blocks and sessions per weight/KV dtype
python3 -m quantlab bench --schemes bf16,fp8,w4a16                              # simulated TTFT/TPOT per scheme
python3 -m quantlab fake --scheme w4a16 --port 8000 &                           # a fake vLLM that runs at INT4 speed
python3 -m quantlab bench --url http://127.0.0.1:8000                           # measure it over HTTP (labelled simulated)
python3 -m jupyterlab notebooks                                                 # the exercises; answers in solutions/
```

On a GPU (T1), do these steps:

1. Run `SCHEME=FP8_DYNAMIC deploy/any-gpu/compress.sh`. It makes a real checkpoint with llm-compressor
   in its own environment.
2. Run `SCHEME=fp8 MODEL=./Qwen2.5-1.5B-Instruct-FP8_DYNAMIC deploy/any-gpu/serve.sh`. It compares the
   scheme with the GPU, then it serves the checkpoint.
3. Run `export QUANTLAB_URL=http://127.0.0.1:8000`. Then the notebooks measure the real server.

With `QUANTLAB_RUN_T1=1`, notebooks 01 and 03 run llm-compressor and lm-eval themselves. With
`QUANTLAB_VLLM_LOG=vllm.log`, notebook 04 reads back a real startup log.

## The library (`quantlab/`, ~3,200 lines)

| Module | The idea |
|---|---|
| `compress.py` | Compressed-tensors presets (`FP8_DYNAMIC`, `FP8`, `W8A8`, `W8A16`, `W4A16`, `W4A16_ASYM`, `NVFP4A16`, `NVFP4`). RTN, GPTQ, AWQ and SmoothQuant in numpy. It writes the checkpoint with the names, dtypes, shapes and packing of compressed-tensors. A loader and a validator. The llm-compressor 0.14 and GPTQModel scripts for T1. |
| `serve.py` | Which scheme each GPU runs natively, and which kernel to expect in the vLLM startup log (from vLLM v0.30.0 / main source, verify). `vllm serve` / `docker run` commands. The map from a checkpoint to its vLLM scheme class and minimum capability, with the error text of vLLM. Cloud Run variables and GKE args for the serving lab's deploy. A startup-log parser. |
| `bench.py` | One GEMM on the roofline (it reproduces vllm-internals §8.1). A scheme-aware step-time model. A continuous-batching emulator. Closed-loop simulation. A streaming HTTP client that measures TTFT/ITL/TPOT as `vllm bench serve` does. |
| `fakeserver.py` | The emulator behind `/v1/completions` (SSE), `/v1/models`, `/health`, `/version` (`"simulated": true`), `/metrics` with the names of vLLM. It uses the standard library only. |
| `evalharness.py` | `lm_eval` command lines (vllm / hf / local-completions), results and table parsers, unpaired comparison, Wilson intervals, exact McNemar, logit KL. The offline mini-eval. |
| `kv.py` | KV bytes per token, blocks and sessions (reproduces `servelab.sizing`), attention backend per GPU and KV dtype, KV quantizer emulation and scale calibration. |
| `fp4.py` | E2M1 rounding and packing, NVFP4 (E4M3 per 16 + FP32 global) and MXFP4 (E8M0 per 32) quantizers in compressed-tensors' conventions, checkpoint layouts and bits per weight. |
| `report.py` | Tables whose every section says where its numbers came from: exact, simulated, measured, sample. |
| `numerics.py`, `stio.py` | FP8 E4M3/E5M2 and bf16 bit patterns, compared with the casts of torch when torch is present. INT4 packing in compressed-tensors order. Safetensors read/write with no dependencies, compared with the `safetensors` package when it is present. |
| `tinymodel.py`, `data/tiny-adder/` | The bundled model: a 2-layer Llama (RMSNorm, RoPE, GQA, SwiGLU, `LlamaForCausalLM` tensor names, bf16 safetensors). [`tools/train_tiny.py`](tools/train_tiny.py) trained it to 100% on 3-digit addition and 6-digit reversal. A rescale that does not change the function planted two massive-activation channels in it. The purpose is to make calibration matter as it does at scale. |

### Names in the core and in the lab

The core names a scheme by what its GEMM does. The lab names it by the checkpoint that you serve. The
lab also accepts the spellings of the core (`serve.plan("w8a8-fp8", "H100-SXM")` works).

| Lab (`quantlab.serve`) | Core (`quantcore.cost`) | llm-compressor preset |
|---|---|---|
| `bf16` | `bf16` | none (the original checkpoint) |
| `fp8` | `w8a8-fp8` (on sm_89+). It runs as `w8a16-fp8` below sm_89. | `FP8_DYNAMIC` |
| `fp8-online` | `w8a8-fp8`, per-tensor | none: `--quantization fp8_per_tensor` at load |
| `w4a16` | `w4a16` | `W4A16` (GPTQ) or `W4A16_ASYM` (AWQ) |
| `w8a8-int8` | `w8a8-int8` | `W8A8` (with SmoothQuant) |
| `nvfp4` | `w4a4-nvfp4` (on sm_100+) | `NVFP4` |
| GPUs `H100-80GB`, `RTXPRO6000`, `RTX4090` | `H100-SXM`, `RTX-PRO-6000`, none | |

This lab never imports the topic's core (`quant-core`) or the serving lab. Where a formula already has
a home in the repo, the tests pin its numbers:

- The block counts of `servelab.sizing`. When the serving lab is in the checkout, the tests also do a
  live cross-check against its code.
- The GEMM table of vllm-internals §8.1 and the decode floors of §8.2.
- The INT4 packing of compressed-tensors (`0xfcba9810`).

## Deploy

[`deploy/`](deploy/) has two parts:

- [`any-gpu/`](deploy/any-gpu/): `compress.sh` runs llm-compressor in its own virtualenv. `serve.sh` runs
  `vllm serve` or `docker run` per scheme, after it compares the scheme with the compute capability of
  the GPU. There is also an INT4 recipe for a Colab/Kaggle T4, and RunPod/Vast 4090/L4 notes for FP8.
- [`gcp/`](deploy/gcp/): the serving lab's Cloud Run Terraform and GKE manifests with quantized-model
  variables, and a wrapper that operates them.

Each part has a README with cost and cleanup.

## Regenerating notebooks and the tiny model

```bash
python3 tools/build_notebooks.py                        # notebooks/ and solutions/ from notebooks_src/
python3 tools/run_notebooks.py solutions                # solutions must run clean (T0, no network)
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
python3 tools/train_tiny.py                             # re-create data/tiny-adder (needs torch; ~25 min on a busy CPU)
make check                                              # tests + notebooks + bash -n
```

## Caveats

- **Simulated against measured.** The speeds from `bench` and the fake server come from a roofline emulator.
  It has assumed efficiencies (60% of peak FLOP/s, 80% of bandwidth, 2 ms per step). These speeds have
  the label SIMULATED everywhere. A real `vllm serve` in `QUANTLAB_URL` gives measurements. The lm-eval results
  and startup logs in `quantlab/data/samples/` are *sample output in the documented format
  (illustrative)*, not measurements.
- **The tiny model is not your model.** It shows mechanisms with real numbers: outlier channels, the
  compensation of GPTQ, the collapse of FP4 activations, scale saturation. But the size of an accuracy
  drop is model- and task-specific. Use your own evals as the gate (notebook 03).
- **Do not read a ranking of the good schemes off it.** The answers of the tiny model are saturated
  (top-1 probability above 0.999 at every answer position). Its inputs are also low-rank (a 16-token
  vocabulary, 128-wide layers). Because of these two properties, GPTQ INT4 looks as near to BF16 as FP8 does, and KL differences of
  1e-8 against 1e-9 mean nothing. On real models, expect FP8 and INT8 W8A8 to cost least, INT4 GPTQ/AWQ
  more and INT4 RTN most (PRIMER §10). Its result that RTN g32 is worse than RTN g128 comes from the two
  planted outlier columns (notebook 01), not from group size.
- **The scheme table is read from source, not run.** Kernel names, capability floors and backend rules
  come from vLLM v0.30.0 / main (Sep 2026). On your version, the startup log is the ground truth.
- **T1 paths are written, not executed here** (no GPU in this environment). The checks for
  llm-compressor, vLLM, lm-eval, Docker and the GCP wrapper are `bash -n`, `DRY_RUN=1` and `ast.parse` of
  the generated scripts. Also, a Kubernetes schema validation examines the serving lab's Deployment with
  the args of each scheme.

## Verify list (dated 2026-09-26; re-check when you move the pins)

- Versions: vLLM **0.30.0** (`vllm/vllm-openai:v0.30.0`), llm-compressor **0.14.0**, compressed-tensors
  **0.19.0**, lm-eval **0.4.13**, GPTQModel **7.5.0** (PyPI, 2026-09-26).
- `--quantization fp8` on a bf16 checkpoint works in v0.30.0 and raises an error on main.
  `fp8_per_tensor` works in both. bitsandbytes and GGUF moved to out-of-tree plugins (`vllm-bnb-plugin`, `vllm-gguf-plugin`).
- Capability floors:
  - Marlin sm_75.
  - Machete sm_90 only.
  - CUTLASS FP8 sm_89 (CUDA ≥ 12.4).
  - CUTLASS block-FP8 sm_90+.
  - CUTLASS NVFP4 sm_100–12x with CUDA ≥ 12.8.
  - `CompressedTensorsW8A8Fp8` 89 (else W8A16 through Marlin).
  - No support for INT8 W8A8 on compute capability ≥ 10.0 (vLLM docs).
- FP8 KV:
  - No backend on sm_75.
  - FlashAttention reads FP8 KV only as FA3 (sm_90) / FA4 (sm_10x).
  - FlashInfer from sm_80.
  - The default `k_scale`/`v_scale` is 1.0.
  - The words of the attention-backend line in the log.
- Model ids: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`, `Qwen/Qwen2.5-0.5B-Instruct(-AWQ, -GPTQ-Int4)`. Also the name
  of the `train_sft` split of `ultrachat_200k` for calibration.
- Datasheet numbers in `serve.GPUS` (dense TFLOP/s, bandwidth, driver-reported memory). The FP8/INT8
  rates of the RTX 4090 and the usable memory of the B200 are unverified. The ModelOpt measurement of
  W4A16 against BF16 on Blackwell (announcement dated 2026-09-16).

The licence is MIT.
