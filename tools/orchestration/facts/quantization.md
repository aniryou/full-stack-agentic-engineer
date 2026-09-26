# facts-quantization.md — verified 2026-09-26 for `04-inference-engine/quantization` (quantcore / quantlab)

Source of truth for the two builders and three reviewers. Extends `$SP/FACTS.md`; does not repeat it.
Paths: `$R` = `$SP/ref`. Upstream snapshots: vLLM `main@a4eb3f25` (2026-09-26; `$R/vllm`, plus `$R/vllm-csrc` sparse clone
of the same commit and single files fetched to `$R/raw/`), spot-checked against the tag **v0.30.0** (current PyPI release) and
`5840d95` (the commit vllm-internals pins). compressed-tensors `47f7d42` (2026-09-25), llm-compressor `c6fb66c` (2026-09-25),
lm-eval `d6de816` (2026-09-14), gptqmodel `3b2e435` (2026-09-26), modelopt `23355ed`, gptq `2d65066`, llm-awq `d6e797a`,
smoothquant `c61476d`, kivi `876b4d2`, marlin `1f25790`, gpt-oss, deepseek-v3. llama.cpp files fetched from `master` to `$R/raw/`.
PyPI (2026-09-26): `vllm` 0.30.0, `llmcompressor` 0.14.0 (py>=3.10), `compressed-tensors` 0.19.0, `lm-eval` 0.4.13, `gptqmodel` 7.5.0,
`autoawq` 0.2.9 (deprecated), `nvidia-modelopt` 0.47.0, `vllm-gguf-plugin` 0.0.5, `vllm-bnb-plugin` 0.0.3.
Anything marked **(unverified)** was not found in a source here; builders must write it as `(verify)`.

---

## 0. What the repo already states (stay consistent; link, do not restate)

- **`minengine.quant`** (`04-inference-engine/serving-engine/mini-engine-core/minengine/quant.py`), public API:
  `FP8_E4M3_MAX = 448.0`; `QMAX = {"int8": 127, "int4": 7, "fp8": 448.0}` (**restricted symmetric**: ±127 / ±7, scale = amax/QMAX);
  `fp8_e4m3(x)` (round to nearest E4M3, saturating); `quantize(w, fmt="int8", granularity="channel", group_size=128)` → `(codes, scales, w_hat)`,
  `w` is `(d_in, d_out)` used as `x @ w`, granularity `'tensor'|'channel'|'group'`; `fake_quant(w, **kw)`;
  `quantize_activations(x, fmt="int8", per="token")`; `smoothquant_scales(x_absmax, w_absmax, alpha=0.5)`;
  `error(w, w_hat)` → `{"rel", "max_abs", "sqnr_db"}` with `sqnr_db = 10·log10(Σw²/Σ(w−ŵ)²)`;
  `bits_per_weight(bits, group_size=None, scale_bits=16)` (INT4 g128 = **4.125**); `weight_gb(params, bits, group_size, scale_bits, keep16_params)`;
  `compare_logits(ref, test)` → `{"kl", "top1"}` (mean KL(p_ref‖p_test) in nats, top-1 agreement).
- **Numbers printed by `mini-engine-core/notebooks_src/06_quantization.py`** (re-run here, `rng = np.random.default_rng(0)`,
  weight `standard_normal((256,128))*0.02`) — serving-engine PRIMER §8 quotes them:
  | scheme | bpw | rel err | SQNR dB |  | scheme | bpw | rel err | SQNR dB |
  |---|---|---|---|---|---|---|---|---|
  | int8 tensor | 8 | 0.0103 | 39.8 | | int4 channel | 4 | 0.1280 | 17.9 |
  | int8 channel | 8 | 0.0071 | 43.0 | | int4 group128 | 4.125 | 0.1184 | 18.5 |
  | fp8 tensor | 8 | 0.0263 | 31.6 | | int4 group32 | 4.5 | 0.0979 | 20.2 |
  Also: E4M3 values in [1,2) step 0.125, in [16,32) step 2; largest finite 448, smallest subnormal 2⁻⁹ = 0.001953125; N(0,1) rounding
  error median 2.224 %, max 5.877 %. Outlier channel ×100 (64×32): int8 per-tensor whole-matrix 0.034 but **0.624 on the 31 normal
  channels**; per-channel 0.006/0.006. SmoothQuant (α 0.5, channel 5 ×60): W8A8 output error 0.0250 → 0.0038 (**6.6×**). Tiny-LM logits:
  int8 per-channel KL 0.00001 top-1 99.8 % (greedy identical 13 tokens); fp8 per-tensor KL 0.00017, 97.5 %; int4 g32 KL 0.00162, 96.5 %;
  int4 per-channel KL 0.00201, 90.2 %; FP8 KV KL 0.00002, top-1 99.0 %. Llama-3.1-8B on L4, SIMULATED (`perf`, 80 % BW / 60 % FLOPs,
  2 ms/step): bf16 16.1 GB, 65.1/82.0 ms (b=1/b=32), prefill 1.8k 360 ms, 17 sessions; int8 w-only 9.1 GB 36.0/53.0/360/43;
  int4 g128 5.7 GB 21.9/38.9/360/56; fp8 W8A8 9.1 GB 36.0/53.0/**181**/43; fp8 W8A8+fp8 KV 35.7/44.2/181/**87**; int4+fp8 KV 21.6/30.1/360/**113**.
  Exercise checks: 70B = 141 GB bf16 vs 39.5 GB INT4 g128; bytes say 3.88× but the model says 2.97× (16-bit LM head, KV, overhead).
- **`servelab.sizing`** (`vllm-serving-lab/servelab/sizing.py`): `size(model, gpu_name=None, *, gpu_memory_utilization=0.92,
  max_model_len=None, block_size=16, dtype="auto", quantization=None, kv_cache_dtype="auto", tensor_parallel_size=1,
  max_num_batched_tokens=None, max_num_seqs=None, enforce_eager=False, gpu_memory_bytes=None, overhead_bytes=None,
  kv_budget_bytes=None, typical_len=None) -> SizingReport`; `weight_bytes(m, dtype, quantization, tp)`; `kv_bytes_per_token(m, kv_cache_dtype, dtype, tp)`
  = `2·layers·kv_heads·head_dim·bytes`. `QUANT_BYTES = {"fp8":1.0,"int8":1.0,"w8a8":1.0,"awq":0.5,"gptq":0.5,"int4":0.5}` and
  `INT4_GROUP_OVERHEAD = 2.5/128` bytes (fp16 scale + 4-bit zero point per 128 → **4.156 ≈ 4.16 bpw**). These are servelab's own keys,
  not vLLM flag values. `DTYPE_BYTES` accepts `fp8`, `fp8_e4m3`, `fp8_e5m2`. GPUS (driver GiB, GB/s, bf16/fp8 dense TFLOP/s, cc):
  T4 15.0/320/65/0/"7.5" `bf16=False`; L4 22.49/300/121/242/"8.9"; RTX4090 23.99/1008/165/330/"8.9"; L40S 44.99/864/362/733/"8.9";
  A100-40GB 40/1555/312/0/"8.0"; A100-80GB 80/2039/312/0; H100-80GB 79.65/3352/989/1979/"9.0"; RTXPRO6000 95/1600/500/1000/"12.0" (all verify).
  Re-run here, Llama-3.1-8B-Instruct on L4 (`typical_len=2000`): params total 8,030,261,248 = linear 6,979,321,856 + embedding/LM-head
  1,050,673,152 (+ norms); KV 131,072 B/token (bf16), 65,536 (fp8).
  | quantization / kv | weights B | KV budget B | blocks | sessions @2000 |
  |---|---|---|---|---|
  | None / auto | 16,060,522,496 | 4,957,217,896 | **2,363** | 18.9 |
  | fp8 / auto | 9,081,200,640 | 11,936,539,752 | 5,691 | 45.5 |
  | int4 (= awq, gptq) / auto | 5,727,854,592 | 15,289,885,800 | 7,290 | 58.3 |
  | None / fp8 | 16,060,522,496 | 4,957,217,896 | 4,727 | 37.8 |
  | fp8 / fp8 | 9,081,200,640 | 11,936,539,752 | 11,383 | 91.1 |
  | int4 / fp8 | 5,727,854,592 | 15,289,885,800 | 14,581 | 116.6 |
  Overheads used: activations 447,217,664 + CUDA graphs 536,870,912 + non-torch 214,748,364 B. These match serving-engine §8
  "Sessions (vLLM defaults)" and vllm-internals §4.7 (2,363 → 4,727 with fp8 KV) and §8.2 (+3,328 / +4,927 blocks).
- **Two INT4 bpw conventions in the repo — both correct for their assumptions:** `minengine.quant` 4.125 (symmetric, 16-bit scale per 128)
  and `servelab.sizing` / vllm-internals §8.1 4.16 (asymmetric: + 4-bit zero point). SPEC's "INT4-g128 ≈ 4.16 bpw" is the asymmetric one.
  quantcore's `bits_per_weight` must take a `zero_point_bits` argument and a test must pin both (4.125 and 4.15625).
- **roofline** (`01-hardware-gpu-fabric/roofline-and-fabric/roofline-core/roofline/`): `llm.SCHEMES = {"bf16": w2/kv2/bf16, "fp8": w1/kv1/fp8,
  "w8a16": w1/kv2/bf16, "w4a16": w0.5/kv2/bf16, "w4a16-kv8": w0.5/kv1/bf16}`; 01 PRIMER §3.5 table (Llama-3.1-8B, H100, 2K ctx: FP8 2.00× both
  regimes, W4A16 3.80× at batch 1 but 1.54× at batch 64, crossover batch 75 vs 297; FP8 fits 476 vs 208 sequences) and §8.1 ($/M tokens:
  H100 on-demand bf16 $0.446 → FP8 $0.158 at 100 % util). `specs.DEVICES` dense TFLOP/s (verify): T4 fp16 65, int8 130, int4 260 (no bf16/fp8);
  L4 bf16 121, fp8 242.5, int8 242.5; A100 bf16 312, int8 624 (no fp8); H100/H200 bf16 989.4, fp8 1978.9; B200 bf16 2250, fp8 4500, fp4 9000,
  int8 4500; GB300 fp4 15000 (int8 cut); RTX PRO 6000 bf16 500, fp8 1000, fp4 2000. RTX 4090 is **not** in `roofline.specs` (only `servelab.GPUS`).
- **vllm-internals §8.1** already has the `--quantization` family table, `_POSSIBLE_KERNELS` order and the down_proj roofline table
  (L4, K 14,336 × N 4,096: M=1 BF16 392 µs / W4A16 102 / FP8 196; M=256 423/248/215; M=2048 1,988/1,988/992; W4A16 edge gone above ~120
  tokens on L4, ~85 on H100). **§8.2**: Llama-3.1-8B BF16 16.06 GB, FP8 9.08 GB, INT4 g128 5.73 GB; decode floor L4 50.0/26.8/15.6 ms.
  **§8.3** says "GGUF and bitsandbytes are not in the registry at this commit (verify)" → answered in §2 below (out-of-tree plugins).
  **§6.3**: T4 → Triton attention; A100/L4 → FA2 (FP8 KV only with FA3 on SM90 or FA4 on SM100, so `--kv-cache-dtype fp8` on L4/A100 → FlashInfer).
- **FlashAttention deep dive §9.4** (FP8 error sources: 6.25 % max mantissa error; outlier channel/token tables; `2^8` P offset; calibrated
  `k_scale`/`v_scale`) — cite, do not re-derive. **serving-engine PRIMER §8** knob table rows `kv_cache_dtype fp8` and `quantization (fp8, awq, gptq, …)`.
- **vllm-serving-lab notebook 05** uses `vllm serve meta-llama/Llama-3.1-8B-Instruct --quantization fp8 ...` on a BF16 checkpoint and
  `Qwen/Qwen2.5-1.5B-Instruct-AWQ` — see Pitfall 1.

## 1. Number formats (verified with torch 2.14 `torch.finfo` / bit enumeration here, plus the cited code)

- **FP8 E4M3 (`torch.float8_e4m3fn`)**: 1-4-3, bias 7, max **448**, min normal 2⁻⁶ = 0.015625, min subnormal 2⁻⁹, eps 0.125;
  **no inf**, 2 NaN codes (S.1111.111); 254 finite codes (253 distinct values). Normal dynamic range 448/2⁻⁶ = 28,672 ≈ 2^14.8.
  vLLM doc: "E4M3 … can store values up to +/-448 and `nan`" (`$R/vllm/docs/features/quantization/llm_compressor/fp8.md`).
- **FP8 E5M2 (`torch.float8_e5m2`)**: 1-5-2, max **57,344**, min normal 2⁻¹⁴, min subnormal 2⁻¹⁶, eps 0.25; ±inf and 6 NaN codes.
  Doc: "E5M2 … up to +/-57344, +/- `inf`, and `nan` … lower precision". vLLM `Fp8LinearMethod` weights: "Only support float8_e4m3fn" (`fp8.py`).
- **FP4 E2M1**: grid **{0, 0.5, 1, 1.5, 2, 3, 4, 6}** ± (`kE2M1ToFloat` in `$R/compressed-tensors/src/compressed_tensors/compressors/nvfp4/helpers.py`;
  `FP4_VALUES` in `$R/gpt-oss/gpt_oss/torch/weights.py`). `FP4_E2M1_DATA`: exponent 2, mantissa 1, max 6.0 (`quantization/quant_args.py`).
  Nibble code = index (0.5→1 … 6.0→7) | sign<<3; two values per uint8, **first value in the low nibble** (`pack_fp4_to_uint8`, gpt-oss `idx_lo = blk & 0x0F`).
  vLLM reference rounding (`utils/nvfp4_emulation_utils.py: cast_to_fp4`) uses thresholds 0.25/0.75/1.25/1.75/2.5/3.5/5.0.
- **MXFP4 (OCP MX)**: E2M1 elements, one **E8M0** scale per **32** elements (`OCP_MX_BLOCK_SIZE = 32`, `utils/ocp_mx_utils.py`; vLLM online doc
  "fp4_e2m1 data, e8m0 per-1x32-block scale"). gpt-oss: `tensor.blocks` (uint8, 2 values each) + `tensor.scales` (uint8), dequant
  `ldexp(lut[nibble], scales − 127)` (`$R/gpt-oss/gpt_oss/torch/weights.py`); scaling "among the last dimension"; MoE linear weights only, rest BF16
  (`$R/gpt-oss/README.md` "Precision format"). compressed-tensors scale: `127 + floor(log2(round_to_power_2(amax))) − emax_elem` with
  `_MX_ELEM_OFFSET = {4: 2, 8: 8}` (`quantization/utils/mxfp_utils.py`). **bpw = 4 + 8/32 = 4.25.**
- **NVFP4**: E2M1 elements, one **FP8 E4M3** scale per **16** elements, plus one FP32 per-tensor global scale. compressed-tensors preset `NVFP4`:
  `num_bits=4, type=float, strategy=TENSOR_GROUP, group_size=16, scale_dtype=float8_e4m3fn`; activations same with `dynamic="local"` (`quant_scheme.py`).
  Global scale `generate_gparam`: `global_scale = 448 * 6 / amax_tensor` (FP8 max × FP4 max; a **multiplier**), local scale =
  `global_scale * block_amax / 6` rounded to E4M3 (`quantization/utils/helpers.py: calculate_qparams`, `generate_gparam`); vLLM `ref_nvfp4_quant`
  identical (`scale = global_scale * (vec_max / 6)`, clamp ±448, cast e4m3). ModelOpt checkpoints name the per-tensor scale `weight_scale_2`
  (`modelopt.py: uses_weight_scale_2_pattern`); its direction (amax/2688 vs 2688/amax) **(unverified)**. ModelOpt recipe:
  `num_bits: e2m1, block_sizes: {-1: 16, type: dynamic, scale_bits: e4m3}` (`$R/modelopt/docs/source/guides/10_recipes.rst`).
  **bpw = 4 + 8/16 = 4.5** (ModelOpt: "NVFP4: 4.5 … defaults exceed 4.5 because `lm_head` remains BF16", `announcements/autoquantize.rst`).
  transformers: NVFP4 "requires a Blackwell GPU with compute capability 10.0 or newer"; eligible Linear dims divisible by 16 (`$R/transformers/docs/source/en/quantization/nvfp4.md`).
- **MXFP8**: E4M3 + E8M0 per 32 (8.25 bpw); vLLM online `mxfp8` "Requires SM 100+ … for w8a8, other GPUs use a w8a16 fallback" (`online.md`).
- **INT grids**: compressed-tensors `_calculate_range`: INT q ∈ [−2^(b−1), 2^(b−1)−1] (INT4 −8…7, INT8 −128…127), symmetric scale =
  `amax / (bit_range/2)` = **amax/7.5 (INT4), amax/127.5 (INT8)**; asymmetric `scale=(max−min)/bit_range`, `zp = clamp(bit_min − min/scale)`;
  0 always representable (min ≤ 0 ≤ max). Original GPTQ `quant.py`: unsigned `maxq = 2^b−1`, symmetric `zero=(maxq+1)/2`, `scale=(xmax−xmin)/maxq`
  (same grid as −8…7 for INT4). vLLM stores GPTQ-symmetric as `uint4b8` / `uint8b128` (unsigned with bias 8/128), AWQ as `uint4` + runtime zero point.
  llm-awq `pseudo_quantize_tensor`: asymmetric `scales=(max−min)/(2^b−1)`, `zeros=−round(min/scales)`. minengine uses ±7/±127 (Section 0).
- **Uniform-quantizer error model**: SQNR ≈ 6.02·b + 1.76 dB for a full-scale sinusoid (textbook, **unverified** here); minengine measures
  (43.0 − 17.9)/4 = 6.3 dB per bit (per-channel INT8 vs INT4 on Gaussian weights). Use "~6 dB per bit".
- **bpw arithmetic** (derived): INT4 g128 sym fp16 scale 4.125; asym (+4-bit zp) 4.15625; g64 4.25; g32 4.5 / 4.625 asym; MXFP4 4.25;
  NVFP4 4.5; FP8 block 128×128 with fp32 scales 8.002; Marlin's ideal speedup 16/4.125 = **3.87×** (Marlin README: "optimal 3.87x (note the
  0.125 bits storage overhead of the group scales)").
- **DeepSeek-V3 FP8** (`$R/deepseek-v3/README_WEIGHTS.md`): `"quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method":
  "fp8", "weight_block_size": [128, 128]}`; per-block fp32 `weight_scale_inv`; dequant `(128x128 weight block) * weight_scale_inv`; activations
  "per-token-per-128-channel" online; `inference/model.py: block_size = 128`, `kernel.py: act_quant(x, block_size=128)`.

## 2. vLLM: methods, auto-detection, kernels, minimum capability

- **`QuantizationMethods` literal** (`$R/vllm/vllm/model_executor/layers/quantization/__init__.py`; identical list at v0.30.0): `awq`, `auto_awq`,
  `fp8`, `fbgemm_fp8`, `fp_quant`, `modelopt`, `modelopt_fp4`, `modelopt_mxfp8`, `modelopt_mixed`, `auto_gptq`, `gptq`, `gptq_marlin`,
  `awq_marlin`, `humming`, `compressed-tensors`, `experts_int8`, `quark`, `moe_wna16`, `torchao`, `inc`, `mxfp4`, `gpt_oss_mxfp4`,
  `deepseek_v4_fp8`, `online`, and online shorthands `fp8_per_tensor`, `fp8_per_block`, `fp8_per_channel`, `int8_per_channel_weight_only`,
  `nvfp4_per_token`, `mxfp8`. `DEPRECATED_QUANTIZATION_METHODS = ["fbgemm_fp8", "fp_quant"]` (need `--allow-deprecated-quantization`).
  Mapping: `gptq`/`gptq_marlin`/`auto_gptq` → `AutoGPTQConfig` ("using Marlin kernels"); `awq`/`awq_marlin`/`auto_awq` → `AutoAWQConfig`;
  `modelopt` → `ModelOptFp8Config`, `modelopt_fp4` → `ModelOptNvFp4Config`; `mxfp4` → `Mxfp4Config`; `compressed-tensors` → `CompressedTensorsConfig`.
- **Not in the list: `bitsandbytes`, `gguf`.** Both moved out of tree: `uv pip install vllm-bnb-plugin` (then `--quantization bitsandbytes`) and
  `uv pip install vllm-gguf-plugin` (`vllm serve unsloth/Qwen3-0.6B-GGUF:Q4_K_M --tokenizer Qwen/Qwen3-0.6B`) (`docs/.../bnb.md`, `gguf.md`;
  GGUF "highly experimental and under-optimized").
- **Auto-detection** (`$R/vllm/vllm/config/model.py: _verify_quantization`): reads `quantization_config["quant_method"]` from the HF config;
  probes each method's `override_quantization_method` (order: `auto_gptq, gptq, gptq_marlin, auto_awq, awq, awq_marlin, inc, moe_wna16, modelopt,
  modelopt_fp4, modelopt_mxfp8, mxfp8, modelopt_mixed, mxfp4, gpt_oss_mxfp4, deepseek_v4_fp8, humming`); if the user passes a different
  `--quantization`, raises "Quantization method specified in the model config (X) does not match the quantization method specified in the
  `quantization` argument (Y)". GPTQ override accepts user values `gptq`, `gptq_marlin`, `auto_gptq`, `marlin`. ModelOpt checkpoints are
  detected via `hf_quant_config.json` `quantization.quant_algo` (`FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `FP8_PB_WO`, `NVFP4`, `W4A16_NVFP4`, `MXFP8`;
  `docs/.../modelopt.md`). **Capability gate** (`vllm/config/vllm.py` ~L925): `capability < quant_config.get_min_capability()` →
  "The quantization method … is not supported for the current GPU. Minimum capability: … Current capability: …".
- **`get_min_capability()` per config** (source): `AutoGPTQConfig` 60 (but `AutoGPTQLinearMethod.__init__` calls `verify_marlin_supported`, and
  Marlin returns no types below 75 → effectively **sm75**); `AutoAWQConfig` 75; `Fp8Config` 75; `CompressedTensorsConfig` 70 (per-scheme below);
  `Mxfp4Config` 80; `ModelOptFp8Config` 80, `ModelOptNvFp4Config` 75, MXFP8 80; `OnlineQuantizationConfig` 75; `TorchAOConfig` 75;
  `ExpertsInt8Config` 80; `MoeWNA16Config` 70; `FPQuantConfig` 100; `HummingConfig` 75.
- **compressed-tensors schemes** (`compressed_tensors/schemes/*.py`, min capability): `CompressedTensorsWNA16` 75 (only for `format ==
  "pack-quantized"`, CHANNEL/GROUP, static, no input quant); `CompressedTensorsW8A8Fp8` **89** ("lovelace and up"); `CompressedTensorsW8A16Fp8` 75;
  `CompressedTensorsW8A8Int8` 75; `CompressedTensorsW4A8Fp8` 90; `CompressedTensorsW4A4Fp4` (NVFP4, also `use_a16=True` for NVFP4A16) 75;
  `CompressedTensorsW4A4Mxfp4` 80; `CompressedTensorsW8A8Mxfp8` 75. Dispatch (`compressed_tensors.py: _get_scheme_from_parts`): NVFP4 weights
  require NVFP4 or no input quant; FP8 W8A8 falls back to `CompressedTensorsW8A16Fp8` when `_check_scheme_supported(89)` fails (so an
  FP8_DYNAMIC checkpoint **loads on T4/A100 as weight-only FP8 via Marlin**).
- **Linear kernel priority** (`vllm/model_executor/kernels/linear/__init__.py`, fetched to `$R/raw/`): mixed-precision `_POSSIBLE_KERNELS[CUDA]
  = [CutlassW4A8LinearKernel, MacheteLinearKernel, MarlinLinearKernel, ConchLinearKernel, ExllamaLinearKernel, TritonW4A16LinearKernel,
  HummingLinearKernel]`; FP8 per-tensor/channel `[FlashInferFP8…, CutlassFP8…, B12xTensorFP8…, PerTensorTorchFP8…, ChannelWiseTorchFP8…,
  MarlinFP8…, HummingFP8…]`; FP8 block `[FlashInferFp8DeepGEMMDynamicBlockScaled, DeepGemmFp8BlockScaledMM, CutlassFp8BlockScaledMM,
  B12xFp8BlockScaledMM, MarlinFP8ScaledMM, HummingFP8…, TritonFp8BlockScaledMM, BlockWiseTorchFP8…]`; INT8 `[CutlassInt8…, TritonInt8…, HummingInt8…]`;
  NVFP4 `[FlashInferCuteDsl…, FlashInferCutlass…, FlashInferB12x…, CutlassNvFp4…, FlashInferCuteDslNvFp4W4A16…, MarlinNvFp4…, …, EmulationNvFp4…]`;
  MXFP4 `[FlashInferMxFp4…, MarlinMxFp4…, HummingMxFp4…, B12xMxFp4…, EmulationMxfp4…]`. Log line: `Selected <kernel> for <module>`.
  Override: `--linear-backend` (and `--kernel-config '{"linear_backend_per_quant": {...}}'`); MoE uses `--moe-backend`.
- **Kernel capability rules (code):** Machete `get_min_capability` 90 and `is_device_capability(90)` → **Hopper only** (not Blackwell);
  `MACHETE_PREPACKED_BLOCK_SHAPE = [64, 128]`, group sizes `[-1, 64, 128]` for fp16/bf16. CutlassW4A8: exactly SM90, FP8 e4m3 activations.
  Marlin: `get_min_capability` 75; `query_marlin_supported_quant_types` returns `[]` below 75; GPTQ-style types `uint4b8, uint8b128` (+ `float8_e4m3fn`,
  `float4_e2m1f`), AWQ-style `uint4`; `MARLIN_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]`; `GPTQ_MARLIN_MIN_THREAD_N = 64`,
  `GPTQ_MARLIN_MIN_THREAD_K = 128` (tile misalignment is now zero-padded at weight prep, but **in_features per partition must divide by
  group_size**); logs "Marlin kernel with bf16 on GPUs before SM90 … consider fp16". Exllama min 60 (`uint4b8`, `uint8b128`).
  `MarlinFP8ScaledMMLinearKernel`: "FP8 Marlin kernel for GPUs that lack FP8 hardware support", "requires compute capability 7.5 or higher".
  `cutlass_scaled_mm_supports_fp8` (`$R/vllm-csrc/csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_entry.cu`): SM ≥ 90 needs CUDA ≥ 12.0,
  SM 89 needs **CUDA ≥ 12.4**, else false. `…supports_block_fp8`: SM ≥ 100 CUDA ≥ 12.8, SM 90 CUDA ≥ 12.0, **none on SM89**.
  `cutlass_scaled_mm_supports_fp4`: runtime CUDA ≥ 12.8 and SM 100–119 (`ENABLE_NVFP4_SM100`) or SM 120–129 (`ENABLE_NVFP4_SM120`).
  `is_fp4_marlin_supported`: CUDA and capability ≥ 75. CutlassInt8: any CUDA device.
- **Docs hardware table** (`docs/features/quantization/README.md`; columns Volta/Turing/Ampere/Ada/Hopper): AWQ ❌✅✅✅✅; GPTQ ✅✅✅✅✅;
  Marlin (GPTQ/AWQ/FP8/FP4) ❌✅*✅✅✅ ("*Turing does not support Marlin MXFP4"); llm-compressor INT8 W8A8 ❌✅✅✅✅; FP8 W8A8 ❌❌❌✅✅;
  bitsandbytes all ✅; GGUF all ✅. No Blackwell column. INT8 W8A8 doc: "not supported on compute capability >= 10.0 (e.g., RTX 6000 Blackwell)"
  (`llm_compressor/int8_w8a8.md`). INT4 doc says "compute capability > 8.0" (`llm_compressor/int4.md`) — contradicts the code (Marlin from 75); trust code.
  FP8 doc: "FP8 computation … compute capability >= 8.9 … FP8 models will run on compute capability >= 7.5 (Turing) as weight-only W8A16, utilizing FP8 Marlin";
  "up to a 1.6x improvement in throughput"; block-FP8 GEMM order "FlashInfer/DeepGEMM hybrid (Hopper only), DeepGEMM, CUTLASS, Marlin, Triton,
  Humming, then a PyTorch fallback"; hang workaround `VLLM_USE_DEEP_GEMM=0` or `--linear-backend cutlass`.
- **FP8 checkpoints (`Fp8Config`)**: `ACTIVATION_SCHEMES = ["static", "dynamic"]`; keys `quant_method` (contains "fp8"), `activation_scheme`,
  `ignored_layers` (or `modules_to_not_convert`), `weight_block_size` (2-D; block quant requires `dynamic` activations), `store_dtype`.
  **Online**: at `main@a4eb3f25` and `5840d95`, `Fp8Config.__init__` raises "The `fp8` quantization method no longer supports online quantization.
  Please use `--quantization fp8_per_tensor` instead."; at **v0.30.0** `--quantization fp8` on a BF16 checkpoint still works
  (`Fp8PerTensorOnlineLinearMethod`, `is_checkpoint_fp8_serialized` defaults False). `fp8_per_tensor` exists in both → use it.
- **Online schemes** (`docs/features/quantization/online.md`): `fp8_per_tensor` (e4m3, fp32 per-tensor scales; "On some GPUs (Ada, Hopper) linear
  activations use per-token scaling"); `fp8_per_block` (weights 128×128, activations 1×128); `mxfp8`; `mxfp4`; plus `fp8_per_channel`,
  `int8_per_channel_weight_only`, `nvfp4_per_token` usable as `targets` values. "all Linear modules (except for the final `lm_head`)";
  "latency improvements are limited in this mode" (`llm_compressor/fp8.md`). Fine control: `--quantization-config '{"linear":…, "moe":…,
  "ignore":[…]}'` (dotted form `--quantization-config.moe.activation mxfp8`), or `--quantization online` with `targets`.
- **KV cache** (`vllm/config/cache.py: CacheDType`, same at v0.30.0): `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`,
  `fp8_ds_mla`, `nvfp4_ds_mla`, `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`, `int4_per_token_head`,
  `int8_per_token_head`, `fp8_per_token_head`, `nvfp4`, `nvfp4_4over6`. Docstring: "CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2".
  Scales: `k_scale`/`v_scale` (and `q_scale`, `prob_scale`) from the checkpoint, else **1.0** (`layers/quantization/kv_cache.py`, sentinel −1.0 then
  `copy_(1.0)`); the `*_per_token_head` dtypes compute dynamic scales at runtime. Doc strategies: per-tensor (`q/k/v_scale = [1]`) or per-attention-head
  (`k/v_scale = [num_kv_heads]`, "only with the Flash Attention backend" and llm-compressor calibration); FA3 + FP8 KV also quantizes queries.
  `--kv-cache-dtype-skip-layers sliding_window` or indices (`quantized_kvcache.md`).
  Backend support (`$R/vllm-csrc/vllm/v1/attention/backends/`): `flash_attn.py` `supported_kv_cache_dtypes = [auto, float16, bfloat16, fp8, fp8_e4m3]`
  (+ the FA3/FA4 rule); `flashinfer.py` adds `fp8_e5m2`, `nvfp4`, `nvfp4_4over6` (nvfp4 only on the SM100 family), floor **SM80** ("broken on SM75");
  `triton_attn.py` accepts fp8/e4m3/e5m2 and `int4/int8/fp8_per_token_head` but raises "FP8 KV cache is not supported by the Triton attention backend
  on … native FP8 (fp8e4nv) requires SM89+"; also raises for `bfloat16` KV below SM80. ⇒ **T4: no FP8 KV cache**; A100 FP8 KV → FlashInfer;
  L4 → FlashInfer; H100 → FA3.
- `--dtype` values `auto, half, float16, bfloat16, float, float32`; T4 has no bf16 → `--dtype half`.

## 3. compressed-tensors format (`$R/compressed-tensors`)

- `config.json` `quantization_config` keys (`QuantizationConfig`, `quant_config.py`): `config_groups` (dict name → `QuantizationScheme` or preset
  name → targets), `quant_method` = `"compressed-tensors"`, `format` (default `"fakequant"`), `quantization_status`
  (`initialized|calibration|frozen|compressed`…), `kv_cache_scheme` (QuantizationArgs or null), `global_compression_ratio`, `ignore` (list),
  plus `sparsity_config`, `transform_config`, `version` in the README example. Each group: `targets`, `weights`, `input_activations`,
  `output_activations`, and a per-group `format`.
- `QuantizationArgs` fields: `num_bits` (8), `type` (`"int"`|`"float"`), `symmetric` (true), `group_size`, `strategy`
  (`tensor|channel|group|block|token|tensor_group|attn_head`), `block_structure` ([r, c]), `dynamic` (bool or `"local"` — NVFP4 only),
  `actorder` (`"weight"`; aliases `"static"`→weight, `"dynamic"`→`"group"`, deprecated), `scale_dtype`, `zp_dtype`, `observer`, `observer_kwargs`.
- `CompressionFormat` values: `dense`, `sparse-bitmask`, `sparse-24-bitmask`, `int-quantized`, `float-quantized`, `naive-quantized`,
  `pack-quantized`, `marlin-24`, `mixed-precision`, `nvfp4-pack-quantized`, `mxfp4-pack-quantized`, `mxfp8-quantized` (`config/base.py`).
- **Presets** (`quantization/quant_scheme.py: PRESET_SCHEMES`): `W4A16` = int4, GROUP, group_size **128**, **symmetric**, static, no activations
  (`_int_wnam`); `W4A16_ASYM` same with `symmetric=False`; `W8A16`; `W8A8`/`INT8` = int8 weights CHANNEL sym + int8 activations TOKEN dynamic sym;
  `FP8` = float8 TENSOR weights + static TENSOR activations (`observer="static_minmax"`, needs calibration); **`FP8_DYNAMIC`** = float8 CHANNEL weights +
  dynamic TOKEN activations (no calibration data); `FP8_BLOCK` = weights BLOCK [128,128] + activations GROUP 128 dynamic; `W4AFP8`; `NVFP4A16`;
  `NVFP4`; `MXFP4A16`; `MXFP4` (group 32, `scale_dtype=uint8`); `MXFP8A16`; `MXFP8`; `W2A4 … W7A16`; `UNQUANTIZED`.
- **Tensor names** (per Linear): pack-quantized: `weight_packed` (int32), `weight_scale`, `weight_shape`, `weight_zero_point` (asymmetric only,
  packed along dim 0), `input_global_scale` (TENSOR_GROUP activations); nvfp4-pack-quantized: `weight_packed` (uint8, 2 per byte), `weight_scale`
  (e4m3, `[out, in/16]`), `weight_global_scale` (fp32 `[1]`), `input_global_scale`; naive/float-quantized: `weight` (quantized dtype), `weight_scale`,
  `weight_zero_point` (asym); static activations add `input_scale` (`{base}_scale`, `{base}_zero_point`, `{base}_global_scale` with base
  `input|weight|output`, `lifecycle/initialize.py`); attention/KV: `k_scale`, `v_scale`, `q_scale` (`quant_metadata.py: KVCacheScaleType`).
  Scale shapes: CHANNEL `(out, 1)`; GROUP `(out, ceil(in/group_size))`; BLOCK `(ceil(out/r), ceil(in/c))`; ATTN_HEAD `(heads, 1, 1)`.
- **INT4 packing** (`compressors/pack_quantized/helpers.py: pack_to_int32`, verified by running it): signed values offset by `1 << (b−1)` (−8…7 →
  0…15), packed densely along dim 1 (input features), **element 0 in the lowest bits**: `[-8,-7,0,1,2,3,4,7]` → `0xfcba9810`; a `[4096, 896]`
  INT4 weight → `weight_packed [4096, 112]` int32 (896·4/32). vLLM expects `weight_packed` `{input_dim=0, output_dim=1, packed_dim=0}` after its
  permute (`kernels/linear/mixed_precision/marlin.py`) then `ops.gptq_marlin_repack`.
- Example saved config (README, MXFP4): group_0 `format: "mxfp4-pack-quantized"`, weights `{num_bits: 4, type: "float", strategy: "group",
  group_size: 32, symmetric: true, dynamic: false, scale_dtype: "torch.uint8", observer: "memoryless_minmax"}`, `ignore: ["lm_head"]`,
  `kv_cache_scheme: null`, `quant_method: "compressed-tensors"`, `quantization_status: "compressed"`. Minimal `examples/int4_config.json`:
  `{"config_groups": {"group_1": {"weights": {"num_bits": 4, "type": "int", "group_size": 128, "symmetric": true, "strategy": "group"},
  "targets": ["Linear"]}}}`.

## 4. llm-compressor (`$R/llm-compressor`, 0.14.0)

- Entry point `from llmcompressor import oneshot`; `oneshot(model, recipe=…, dataset=…, splits=…, num_calibration_samples=512 (default),
  max_seq_length=None, batch_size=1, shuffle_calibration_samples=True, save_compressed=True, pipeline="independent",
  moe_calibrate_all_experts=True, output_dir=None, …)` (`src/llmcompressor/entrypoints/oneshot.py`). Save: `model.save_pretrained(dir,
  save_compressed=True)`; generate after calibration with `compressed_tensors.offload.dispatch_model(model)`.
- Modifiers: `llmcompressor.modifiers.quantization.QuantizationModifier` (RTN; `targets`, `scheme`, `ignore`, `config_groups`, `kv_cache_scheme`);
  `llmcompressor.modifiers.gptq.GPTQModifier` (`block_size=128`, `dampening_frac=0.01`, `actorder=Sentinel("static")` i.e. weight order,
  `batched_quantization="auto"`); `llmcompressor.modifiers.transform.awq.AWQModifier` (`mappings`, `duo_scaling=True` (or `"both"`), `n_grid=20`,
  `offload_device`; a transform only — pair it with a `QuantizationModifier`/`GPTQModifier`; `duo_scaling` rejected with TENSOR strategy);
  `llmcompressor.modifiers.transform.smoothquant.SmoothQuantModifier` (`smoothing_strength=0.5` default, `mappings`, `ignore`,
  `algorithm="smoothquant"|"log_equalization"`). Old paths `llmcompressor.modifiers.awq` / `.smoothquant` are deprecated shims (vLLM's int4/int8 docs
  still import `llmcompressor.modifiers.smoothquant`).
- Recipes in the examples (all `targets="Linear"`, `ignore=["lm_head"]`):
  FP8: `QuantizationModifier(scheme="FP8_DYNAMIC")`, `oneshot(model=model, recipe=recipe)` — **no dataset** (`examples/quantization_w8a8_fp8/llama3_example.py`);
  README quick tour uses `scheme="FP8_BLOCK"`, `ignore=["lm_head", "re:.*mlp.gate$"]` for Qwen3-30B-A3B.
  W4A16 GPTQ: `GPTQModifier(scheme="W4A16")`, `dataset="perfectblend", splits="train[:512]", max_seq_length=2048, num_calibration_samples=512`
  (`quantization_w4a16/llama3_example.py`; vLLM doc uses `HuggingFaceH4/ultrachat_200k` `train_sft`, 512 × 2048, chat template).
  AWQ: `[AWQModifier(duo_scaling="both"), QuantizationModifier(scheme="W4A16_ASYM")]`, 256 samples × `max_seq_length=512` (`awq/llama_example.py`).
  W8A8 INT8: `[SmoothQuantModifier(smoothing_strength=0.8), GPTQModifier(scheme="W8A8")]`, 512 × 2048 (`quantization_w8a8_int8/llama3_example.py`).
  NVFP4: `QuantizationModifier(scheme="NVFP4")`, **20** calibration samples × 2048 (only global activation scales need data); "if running inference
  on a machine that is < SM100, vLLM will not run activation quantization, only weight-only" (`quantization_w4a4_fp4/README.md`).
  FP8 KV: `kv_cache_scheme: {num_bits: 8, type: float, strategy: tensor, dynamic: false, symmetric: true}` with 512 × 2048
  (`quantization_kv_cache/llama3_fp8_kv_example.py`; per-head variant `llama3_fp8_head_kv_example.py`). MoE: ignore routers
  (`"re:.*mlp.gate$"`, `"re:.*mlp.shared_expert_gate$"`, Mixtral `"re:.*mlp.gate"`), `with load_context():` to load (`examples/quantizing_moe/`).
  Non-uniform: `examples/quantization_non_uniform/quantization_int4_int8.py`, `quantization_nvfp4_fp8.py`.
- Supported (README): activation quant W8A8 (int8, fp8), W4AFP8, NVFP4/MXFP4/MXFP8; mixed W4A16, W8A16, MXFP8A16, MXFP4A16, NVFP4A16;
  KV/attention FP8, NVFP4; algorithms RTN, GPTQ, AWQ, SmoothQuant, AutoRound, SpinQuant/QuIP rotations, REAP. Group layers whose `in_features %
  group_size != 0` **error at initialize** (`modifiers/quantization/group_size_validation.py`; BLOCK strategy exempt).
- vLLM docs: "Please use separate environments for vLLM and llm-compressor as they might not work together"; recommended eval install
  `pip install vllm "lm-eval[api]>=0.4.12"`. Memory needs for a 0.5B model: fits a T4 easily **(unverified — no number in source)**; RTN FP8_DYNAMIC
  needs no data and can run on CPU **(unverified)**.

## 5. Algorithms as code

- **GPTQ** (`$R/gptq/gptq.py`, ICLR 2023). Hessian accumulation per layer input `X`: `self.H *= nsamples/(nsamples+tmp); nsamples += tmp;
  inp = sqrt(2/nsamples)·inp; H += inp @ inp.T` (⇒ `H = 2/n Σ x xᵀ`). `fasterquant(blocksize=128, percdamp=.01, groupsize=-1, actorder=False,
  static_groups=False)`: dead columns (`diag(H)==0`) → `H[d,d]=1`, `W[:,d]=0`; actorder `perm = argsort(diag(H), descending=True)`;
  `damp = percdamp * mean(diag(H)); H[i,i] += damp; H = cholesky(H); H = cholesky_inverse(H); Hinv = cholesky(H, upper=True)`. Column loop:
  ```
  for i1 in range(0, columns, blocksize):
      W1 = W[:, i1:i2].clone(); Hinv1 = Hinv[i1:i2, i1:i2]
      for i in range(count):
          w = W1[:, i]; d = Hinv1[i, i]
          (if groupsize != -1 and (i1+i) % groupsize == 0: quantizer.find_params(W[:, i1+i : i1+i+groupsize]))
          q = quantize(w.unsqueeze(1), scale, zero, maxq).flatten()
          Losses1[:, i] = (w - q)**2 / d**2
          err1 = (w - q) / d
          W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
          Err1[:, i] = err1
      W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])      # lazy batch update
  ```
  Group params are found **on the already-updated** weights (hence `static_groups`). CLI defaults `--nsamples 128`, `--percdamp .01`,
  `--groupsize -1`, seqlen 2048, calibration c4/wikitext2/ptb. README LLaMa Wiki2 PPL (FP16 / 4-bit RTN / 4-bit GPTQ): 7B 5.68/6.29/6.09;
  13B 5.09/5.53/5.36; 65B 3.53/3.92/3.84; 3-bit RTN 25.54 vs GPTQ 8.07 (7B); 3-bit g128 GPTQ 6.61. llm-compressor's
  `quantize_weight(…, blocksize=128, percdamp=0.01)` is the same loop batched (`modifiers/gptq/gptq_quantize.py`).
- **AWQ** (`$R/llm-awq/awq/quantize/auto_scale.py`): activation statistic `get_act_scale(x) = x.abs().view(-1, C).mean(0)` (mean |x| per input
  channel). Search: `n_grid = 20`, `ratio = k/20` for k = 0…19; `scales = x_max.pow(ratio).clamp(min=1e-4); scales = scales / sqrt(scales.max()*scales.min())`;
  for each linear: `W·diag(s)` then `w_quantize(W·s) / s`; loss = MSE of the block output vs original; keep the best ratio. Folding:
  `ln.weight.div_(scales)` (and bias), `fc.weight.mul_(scales)` (`scale_ln_fcs`); FC→FC: `fc1.weight[-n:].div_(s.view(-1,1))`, `fc2.weight.mul_(s)`.
  llm-compressor "duo" variant: `scales = x_mean.pow(ratio) / (w_mean.pow(1-ratio) + 1e-4)`. Clipping (`auto_clip.py: auto_clip_layer`):
  `n_grid=20, max_shrink=0.5, n_sample_token=512`, candidates `max_val = org_max·(1 − i/20)` for i = 0…9 (1.00 down to 0.55), per group,
  minimise output MSE; **q/k projections skipped** ("due to qk bmm"). Defaults: `run_awq(n_samples=512, seqlen=512, calib_data="pileval")`
  (`mit-han-lab/pile-val-backup`); entry uses `n_samples=128, seqlen=512`; `zero_point` True (asymmetric) by default; `--q_group_size` (128 typical).
- **SmoothQuant** (`$R/smoothquant/smoothquant/smooth.py`): `scales = act_scales.pow(alpha) / weight_scales.pow(1 − alpha)` (clamp ≥1e-5), with
  `act_scales` = per-channel **max** |X| from calibration and `weight_scales` = per-input-channel max |W| over the fused q/k/v (or fc1); fold
  `ln.weight.div_(scales)`, `fc.weight.mul_(scales)`; **default `alpha=0.5`**; calibration `--num-samples 512 --seq-len 512`. Tuned α in README:
  Llama-2-7B/13B 0.85, 70B 0.9, Llama-3-8B/70B 0.85, Mistral-7B/Mixtral 0.8, Falcon-7B 0.6, 40B 0.7 (Llama-3-8B PPL 6.138 → 6.258 W8A8).
  Fake quant `W8A8Linear.from_float(weight_quant="per_channel", act_quant="per_token")`; absmax scale `/(2^(b−1)−1)` = /127.
- **KIVI** (`$R/kivi/README.md`, `models/llama_kivi.py`, `quant/new_pack.py`): keys quantized **per-channel** (tensor transposed to `[B, nh, D, T]`,
  groups of `group_size` tokens along T), values **per-token** (`[B, nh, T, D]`, groups along D); asymmetric min-max
  `scale = (mx − mn)/(2^bit − 1)`, packed `32 // bit` per int32; `k_bits`/`v_bits` 2 or 4; examples `group_size = 32`, `residual_length = 32`
  ("the number of recent fp16 tokens"; must be a multiple of group_size). README: 2.6× less peak memory, up to 4× batch, 2.35–3.47× throughput.
  Derived: 2-bit g32 with fp16 scale+min = 3 bits/element (5.3× vs fp16), 4-bit = 5 bits (3.2×), ignoring the residual window.
- **Rotations**: llm-compressor ships SpinQuant/QuIP transforms (`modifiers/transform/spinquant`, `quip`); compressed-tensors "Transform Support:
  Hadamard, random Hadamard, random matrix". QuaRot not in ref **(unverified details)**. FA deep dive §9.4 already measures a Hadamard rotation.
- **QLoRA/NF4** (`$R/transformers/docs/source/en/quantization/bitsandbytes.md`): `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)`.
- **ModelOpt QAD note** (`$R/modelopt/docs/source/announcements/qwen36-w4a4-qad.rst`, 2026-09-16): on Blackwell, weight-only NVFP4 (W4A16) was
  **slower than BF16 in 10 of 12 shapes** (Marlin dequant-to-BF16 fallback); W4A4 beat BF16 in 9 of 12 (up to 1.30×), checkpoint 67 → 22 GiB (3.1×);
  500 QAD iterations recovered IFBench (−2.6 pp after PTQ). ModelOpt configs: `mtq.FP8_DEFAULT_CFG`, `mtq.NVFP4_DEFAULT_CFG`, `mtq.INT8_SMOOTHQUANT_CFG`;
  export `modelopt.torch.export.export_hf_checkpoint`; serve with `quantization="modelopt"` / `"modelopt_fp4"`.

## 6. Kernels and tools

- **Marlin** (`$R/marlin/README.md`): FP16×INT4, "close to ideal (4x) speedups up to batchsizes of 16-32 tokens"; ideal 3.87× with g128 scales;
  original repo requires CUDA ≥ 11.8 and **compute capability ≥ 8.0** ("Marlin is not yet optimized for Hopper"). vLLM's Marlin runs on **SM75+**
  (Section 2) — cite vLLM's code, not the README, for the floor. Machete = vLLM's Hopper-only CUTLASS mixed-input GEMM.
- **GPTQModel** (`$R/gptqmodel/README.md`, v7.5.0 2026-09-15): methods/formats `METHOD.GPTQ` (`FORMAT.GPTQ`, `GPTQ_V2`, `MARLIN`, `BITBLAS`),
  `METHOD.AWQ` (`GEMM`, `GEMV`, `GEMV_FAST`, `LLM_AWQ`, `MARLIN`, `BITBLAS`), `METHOD.GGUF`, `METHOD.FP8` (`format="float8_e4m3fn"` or `"float8_e5m2"`),
  `METHOD.BITSANDBYTES`, `METHOD.EXL3`, `QQQ`, `PARO`; backends e.g. `BACKEND.GPTQ_MARLIN`, `GPTQ_MACHETE`, `GPTQ_EXLLAMA_V2`, `GPTQ_TRITON`,
  `GPTQ_TORCH`; external `BACKEND.VLLM`/`SGLANG`/`MLX`. Linux NVIDIA "`Turing+` (`sm_75+`)". API (vLLM `gptqmodel.md`):
  `QuantizeConfig(bits=4, group_size=128)`; `GPTQModel.load(model_id, quant_config)`; `model.quantize(calibration_dataset, batch_size=2)`;
  `model.save(quant_path)`; install `pip install -U gptqmodel --no-build-isolation -v`. AutoAWQ: "deprecated … adopted by … llm-compressor"
  (`auto_awq.md`); its `quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}`.
- **llama.cpp GGUF k-quants** (`$R/raw/ggml-common.h`, `$R/raw/quantize-readme.md`, master): `QK_K = 256` super-block, `K_SCALE_SIZE = 12`.
  Block bytes → bpw: `q4_0` 18 B/32 = 4.5; `q8_0` 34 B/32 = 8.5; `q2_K` 84 B/256 = 2.625; `q3_K` 110/256 = 3.4375; `q4_K` 144/256 = 4.5;
  `q5_K` 176/256 = 5.5; `q6_K` 210/256 = 6.5625. Mixes (Llama-3.1-8B, measured bpw): Q2_K 3.1593, Q3_K_M 3.9960, Q4_K_S 4.6672,
  **Q4_K_M 4.8944 (4.58 GiB)**, Q5_K_S 5.5704, Q5_K_M 5.7036, Q6_K 6.5633, Q8_0 8.5008 (7.95 GiB), F16 16.0005 (14.96 GiB). Sizes (Llama 3.1):
  8B 32.1 GB → 4.9 GB Q4_K_M. Tool: `./build/bin/llama-quantize in-bf16.gguf out-Q4_K_M.gguf Q4_K_M`. (GPTQModel's `_GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS`
  says q6_k 6.0 — approximate; use llama.cpp's numbers.)
- **lm-evaluation-harness** (`$R/lm-eval`, 0.4.13): install extras `pip install "lm_eval[hf]"`, `"lm_eval[vllm]"`, `"lm_eval[api]"` (base package has no
  torch/transformers since 2025/12). CLI `lm_eval --model hf --model_args pretrained=<id>,dtype=... --tasks <t> --device cuda:0 --batch_size 8`
  (or `auto`, `auto:4`); `lm_eval --model vllm --model_args pretrained=<id>,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 --tasks …
  --batch_size auto`; server: `--model local-completions --model_args model=<id>,base_url=http://HOST:8000/v1/completions,num_concurrent=1,…`.
  New form `lm-eval run …`; legacy form auto-inserts `run`. `--limit/-L` int count or float fraction, "For testing only"; `--num_fewshot/-f`;
  `--output_path/-o`; `--log_samples/-s`; `--apply_chat_template`; `think_end_token` model arg strips reasoning. Tasks: `gsm8k`
  (`openai/gsm8k` `main`, `generate_until`, **5-shot** default, filters `strict-match` / `flexible-extract`, metric `exact_match`); `mmlu` group =
  `mmlu_stem`, `mmlu_other`, `mmlu_social_sciences`, `mmlu_humanities`; 57 subtasks `mmlu_<subject>` (e.g. `mmlu_abstract_algebra`,
  `mmlu_college_computer_science`), `cais/mmlu`, `multiple_choice`, metric `acc`. vLLM doc example: `lm_eval --model vllm --model_args
  pretrained=$MODEL,add_bos_token=True --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 250` → FP8-Dynamic Llama-3-8B-Instruct
  exact_match 0.768 ± 0.0268 ("Quantized models can be sensitive to the presence of the `bos` token").

## 7. Precision support per GPU for the decision table (derived from Sections 2 and 0)

| GPU (cc) | 16-bit | W4A16 INT4 (GPTQ/AWQ/CT) | FP8 weights | W8A8 INT8 | FP8 W8A8 | NVFP4 / MXFP4 | FP8 KV cache |
|---|---|---|---|---|---|---|---|
| T4 (7.5) | fp16 only (`--dtype half`) | Marlin | W8A16 FP8-Marlin (memory only) | ✅ CUTLASS int8 (130 TOPS) | ❌ | NVFP4 → Marlin W4A16; MXFP4 ❌ (min 80; "Turing does not support Marlin MXFP4") | ❌ (Triton needs SM89, FlashInfer/FA need SM80) |
| A100 (8.0) | bf16/fp16 | Marlin | W8A16 Marlin | ✅ | ❌ | Marlin weight-only | ✅ FlashInfer (FA2 has none) |
| L4, RTX 4090 (8.9) | ✅ | Marlin | ✅ | ✅ | ✅ CUTLASS (CUDA ≥ 12.4); **no CUTLASS block-FP8** | Marlin weight-only | ✅ FlashInfer / Triton |
| H100, H200 (9.0) | ✅ | **Machete** (Hopper only) | ✅ | ✅ | ✅ + block FP8, DeepGEMM | Marlin weight-only | ✅ FA3 (also quantizes Q) |
| B200 (10.0) | ✅ | Marlin (Machete is SM90-only) | ✅ | ❌ ("not supported on cc >= 10.0") | ✅ | ✅ W4A4 CUTLASS/FlashInfer (CUDA ≥ 12.8); `nvfp4` KV | ✅ |

## Model and tool ids to use

- **T0 (no download)**: quantcore's own synthetic `tinymodel`; the lab's bundled tiny model; parameter math from `servelab/data/configs/*.json`
  (bundled: `qwen2.5-0.5b-instruct`, `qwen2.5-1.5b-instruct`, `qwen3-0.6b`, `llama-3.2-1b-instruct`, `llama-3.1-8b-instruct`, …).
- **T1 base model: `Qwen/Qwen2.5-0.5B-Instruct`** (in servelab configs and vLLM docs): 24 layers, hidden 896, 14 heads, **2 KV heads**, head_dim 64,
  intermediate 4,864, vocab 151,936, tied embeddings, qkv bias. Params 494,032,768 = linear 357,826,560 (14,909,440/layer = 2·896² + 2·896·128 +
  3·896·4,864) + embedding 136,134,656 + 71,552 norms/biases. Weights (servelab): BF16 0.988 GB, FP8 0.630 GB, INT4 g128 0.458 GB (0.457 at 4.125 bpw).
  KV 12,288 B/token bf16. Marlin/Machete-friendly: 896 = 7·128, 4,864 = 38·128, qkv out 1,152 = 18·64. sizing: T4 66,023 blocks, L4 103,656 (bf16, 4K).
- **`Qwen/Qwen2.5-1.5B-Instruct`**: 28 layers, hidden 1,536, 12 heads, 2 KV heads, head_dim 128, intermediate 8,960, tied; 1,543,714,304 params
  (linear 1,310,195,712; embedding 233,373,696); BF16 3.087 GB, FP8 1.777 GB, INT4 1.148 GB; KV 28,672 B/token.
- Pre-quantized (ids seen in ref tests/docs; existence on the Hub **unverified** here): `Qwen/Qwen2.5-0.5B-Instruct-AWQ` (llm-compressor e2e test,
  gptqmodel test), `Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4` (gptqmodel test), `RedHatAI/Llama-3.2-1B-FP8` (vLLM `optimization_levels.md`),
  `RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8` (vLLM CPU table), `nm-testing/TinyLlama-1.1B-Chat-v1.0-{W4A16-G128,W8A8-Dynamic-Per-Token,
  FP8-Dynamic}-compressed` (llm-compressor tests), `unsloth/Qwen3-0.6B-GGUF:Q4_K_M` (vLLM gguf.md), `Qwen/Qwen2.5-1.5B-Instruct-AWQ` (repo nb 05).
  RedHatAI FP8/W4A16 variants of Qwen2.5-0.5B **(unverified)** — prefer producing them in notebook 01 with llm-compressor.
- **SmolLM2** (`HuggingFaceTB/SmolLM2-135M-Instruct` appears in vLLM spec-decode docs): config not in ref — hidden 576 / 30 layers / 9 heads / 3 KV
  **(unverified)**; 576 % 128 = 64, so **W4A16 g128 is rejected** by llm-compressor's group validation and vLLM Marlin — avoid for INT4 or use g64/g32.
- Tools: `llmcompressor==0.14.0` (+ `compressed-tensors==0.19.0`), `vllm==0.30.0` (separate env), `lm-eval[vllm]==0.4.13`, `gptqmodel==7.5.0`
  (optional), llama.cpp `llama-quantize` (CPU path). Calibration sets in examples: `perfectblend` `train[:512]`, `HuggingFaceH4/ultrachat_200k`
  `train_sft`, `allenai/c4` (GPTQModel), `mit-han-lab/pile-val-backup` (AWQ).

## Pitfalls

1. **`--quantization fp8` on a BF16 checkpoint**: works (online per-tensor) at v0.30.0, raises at `5840d95`/main. Write `--quantization
   fp8_per_tensor` (valid in both). Pre-quantized FP8 checkpoints need no flag at all (auto-detected). The repo's lab-05 recipe uses the old form.
2. Passing `--quantization` that disagrees with `config.json` raises; omit it for pre-quantized checkpoints.
3. T4: no bf16 (`--dtype half`), no FP8 compute, **no FP8 KV cache in any backend**, no Machete. FP8 checkpoints load as W8A16 via Marlin
   (memory win only). The T4 speed lever is INT4 W4A16 (decode) or INT8 W8A8 (prefill).
4. Weight-only INT4 cuts bytes, not FLOPs: faster decode, no faster (often slower) prefill — vllm-internals §8.1 table; ModelOpt's measured
   NVFP4 W4A16 slower than BF16 on Blackwell in 10/12 shapes.
5. `lm_head`/embeddings stay 16-bit in every recipe (`ignore=["lm_head"]`); MoE routers (`mlp.gate`) must be ignored too. Qwen2.5 ties
   embeddings — the 16-bit share is 27.6 % of the 0.5B model (136.1 M / 494.0 M), so INT4 gives ~2.2×, not 4×, on that model.
6. Group-size divisibility: `in_features % group_size == 0` for GROUP/TENSOR_GROUP (llm-compressor errors at init; vLLM Marlin rejects); Marlin groups
   only `-1, 32, 64, 128`; Machete `-1, 64, 128`.
7. Symmetric INT conventions differ: compressed-tensors/GPTQ use −8…7 with `amax/7.5`; minengine uses ±7 with `amax/7`. Document which one quantcore
   uses; a checkpoint written by quantlab must follow compressed-tensors (offset +8, low-nibble-first int32 packing, `weight_shape` stored).
8. NVFP4 global scale in compressed-tensors is a multiplier (`2688/amax`), local scales are E4M3 of `global·block_amax/6`; don't invert it.
   MXFP4 scales are E8M0 exponents biased 127 (power of two only), 32 elements; NVFP4 is 16 elements with E4M3 scales.
9. FP8 KV with default scales = 1.0 unless the checkpoint carries `k_scale`/`v_scale`; per-head scales need FlashAttention + llm-compressor.
   On L4/A100, `--kv-cache-dtype fp8` switches the attention backend to FlashInfer (throughput changes are not only the KV dtype).
10. vLLM docs contradict code on INT4 floor ("> 8.0") — code says SM75. Original Marlin README says ≥ 8.0 — vLLM's port says 75.
11. INT8 W8A8 is unsupported on Blackwell (cc ≥ 10.0); block-FP8 CUTLASS needs SM90+; CUTLASS FP8 on SM89 needs CUDA ≥ 12.4.
12. `lm_eval` does not add BOS by default: pass `add_bos_token=True` for quantized-model comparisons; `--limit` makes results "for testing only" —
    report the stderr (250 gsm8k samples gave ±0.027).
13. bitsandbytes and GGUF need out-of-tree plugins in vLLM ≥ this snapshot (`vllm-bnb-plugin`, `vllm-gguf-plugin`); QLoRA NF4 is a training recipe.
14. Install llm-compressor and vLLM in **separate environments**; `llmcompressor.modifiers.awq`/`.smoothquant` imports are deprecated.
15. `FP8` preset (static activations) needs calibration data; `FP8_DYNAMIC` needs none; NVFP4 needs a little (20 samples in the example).
16. RTX 4090 figures (165 bf16 / 330 fp8 dense) are FP32-accumulate GeForce rates from `servelab.GPUS` **(unverified)**; not in `roofline.specs`.
