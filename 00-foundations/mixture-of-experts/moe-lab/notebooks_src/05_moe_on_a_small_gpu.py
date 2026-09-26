# %% [markdown]
# # 05 · MoE on a small GPU: 4-bit experts, CPU offload, and what each costs per token
#
# **Tier:** T1 — one 16 GB T4 (free on Colab/Kaggle) or a 24 GB L4/RTX 4090: `vllm serve` with
# `--cpu-offload-gb` or a 4-bit checkpoint; the notebook reads the start-up log and measures the step.
# T0 — the sizing arithmetic (**simulated**) and two start-up logs in vLLM's documented wording
# (**illustrative**: generated from this notebook's own sizing, so they agree with it by construction).
#
# ## The one-minute version
#
# * Memory is sized by the **total**: OLMoE-1B-7B runs 1.3 B parameters per token but needs all 6.9 B
#   (13.8 GB in 16-bit) resident — which leaves no KV cache on a 16 GB T4.
# * **4-bit experts** (GPTQ/AWQ INT4; MXFP4 in gpt-oss) cut the experts' bytes ~3.8x with no PCIe
#   traffic — at an accuracy cost you measure, and only where kernels exist (vLLM's MXFP4 needs
#   compute capability 8.0 and bf16: not a T4).
# * **vLLM UVA offload** (`--cpu-offload-gb N --cpu-offload-params experts`) keeps N GiB in pinned CPU
#   memory and reads it over PCIe **in every forward pass**, whichever experts are routed: a fixed
#   cost per step, so it is amortised by batch.
# * **llama.cpp `--n-cpu-moe`** keeps the experts in CPU RAM; in small-batch decode it *computes them
#   on the CPU*, reading only the touched experts, so it is cheap for one user and bound by CPU FLOP/s
#   as the batch grows. For large prompt batches it copies those weights to the GPU by default
#   (`--op-offload`; `--no-op-offload` keeps the work on the CPU) — read from `common/arg.cpp`, verify.
# * Whatever is left after weights is KV cache — MoE did not shrink it.
#
# Concepts: PRIMER §6 "Running MoE on GPUs" (expert offloading, quantized experts), §7 "Sizing and
# cost" and §8 "In a design review: failure modes" (MoE on one 24 GB GPU; quantizing experts)
# ([`PRIMER.md`](../../PRIMER.md)). Quantization in depth: the
# [quantization primer](../../../../04-inference-engine/quantization/PRIMER.md) — §2 "Number formats"
# (INT4, MXFP4), §5 "Weight-and-activation quantization" (why routers stay 16-bit) and §8 "Measuring the
# accuracy you pay" (calibrating rarely routed experts).

# %%
import os
from pathlib import Path
import numpy as np
from moelab import configs, env, hooks, offload, stream

print(env.describe())
names = {"T4": "T4", "L4": "L4", "4090": "RTX4090", "A100": "A100-80GB", "H100": "H100-80GB"}
GPU = next((configs.gpu(v) for g in env.gpus() for k, v in names.items() if k in g), configs.gpu("T4"))
print(f"sizing for: {GPU.name} ({GPU.memory_gib:g} GiB, compute capability {GPU.compute_capability}, "
      f"bf16 {'yes' if GPU.bf16 else 'no'}, PCIe ~{GPU.pcie_gbs:g} GB/s (verify))")
MOES = ["granite-1b-a400m", "granite-3b-a800m", "olmoe-1b-7b", "qwen1.5-moe-a2.7b", "qwen3-30b-a3b", "gpt-oss-20b"]

# %% [markdown]
# ## Worked example: what fits, at vLLM's defaults
#
# `offload.fit`: KV room = `0.92 × memory − overhead − weights`, with a 1.5 GiB overhead for
# activations and CUDA graphs (an estimate — the start-up log measures it, below). 4K-token
# sequences; "fits" means at least one full-length sequence of KV.

# %%
for key in MOES:
    m = configs.get(key)
    for gname in ("T4", "L4"):
        gg = configs.gpu(gname)
        for scheme in (("mxfp4",) if m.native == "mxfp4" else ("bf16" if gg.bf16 else "fp16", "int4")):
            print(offload.fit(m, gg, scheme).line())

# %% [markdown]
# ## Exercise 5.1 — the bytes of a quantized MoE checkpoint
#
# Write `weights_gb(model, rest_bits, expert_bits)` in GB (10^9 bytes): the routed experts' matmul
# weights (`layers × n_experts × expert_matmul_params()`) at `expert_bits`, the embeddings (and LM
# head, `embedding_params()`) always at 16 bits, everything else (attention, router, shared expert,
# biases) at `rest_bits`. 4-bit formats carry scales: use 4.25 bits (INT4 with a 16-bit scale per 64
# weights, or MXFP4's 8-bit scale per 32).

# %% exercise
def weights_gb(model, rest_bits, expert_bits):
    ### BEGIN SOLUTION
    emb = model.embedding_params()
    exp = model.layers * model.n_experts * model.expert_matmul_params() if model.is_moe else 0
    rest = model.total_params() - emb - exp
    return (rest * rest_bits + exp * expert_bits + emb * 16) / 8 / 1e9
    ### END SOLUTION

# %% check
g = configs.get
assert round(weights_gb(g("gpt-oss-20b"), 16, 4.25), 1) == 13.8           # "within 16GB of memory"
assert round(weights_gb(g("gpt-oss-120b"), 16, 4.25), 1) == 65.2          # "fits a single 80GB GPU"
assert round(weights_gb(g("qwen3-30b-a3b"), 4.25, 4.25), 1) == 17.1       # a GPTQ-Int4 checkpoint, estimated
assert round(weights_gb(g("qwen1.5-moe-a2.7b"), 4.25, 4.25), 1) == 8.5
for key in MOES:
    for scheme, bits in configs.SCHEMES.items():
        assert np.isclose(weights_gb(g(key), *bits), configs.weight_bytes(g(key), scheme) / 1e9)
print(f"✅ gpt-oss-20b MXFP4 {weights_gb(g('gpt-oss-20b'), 16, 4.25):.1f} GB; OLMoE 16-bit "
      f"{weights_gb(g('olmoe-1b-7b'), 16, 16):.1f} GB vs 4-bit experts {weights_gb(g('olmoe-1b-7b'), 16, 4.25):.1f} GB")

# %% [markdown]
# ## Exercise 5.2 — KV room and sessions
#
# Write `kv_room(model, gpu, weights_gib, max_model_len, util=0.92, overhead_gib=1.5)` →
# `(kv_gib, kv_tokens, sessions)`: KV GiB = util × `gpu.memory_gib` − overhead − weights; tokens =
# floor(KV bytes / `model.kv_bytes_per_token()`); sessions = tokens / max_model_len (vLLM's "Maximum
# concurrency"). Negative room means it does not start.

# %% exercise
def kv_room(model, gpu, weights_gib, max_model_len, util=0.92, overhead_gib=1.5):
    ### BEGIN SOLUTION
    kv = util * gpu.memory_gib - overhead_gib - weights_gib
    tokens = int(kv * configs.GiB // model.kv_bytes_per_token()) if kv > 0 else 0
    return kv, tokens, tokens / max_model_len
    ### END SOLUTION

# %% check
for key in MOES:
    for gname in ("T4", "L4"):
        m, gg = g(key), configs.gpu(gname)
        f = offload.fit(m, gg, "int4", max_model_len=4096)
        kv, tok, ses = kv_room(m, gg, f.weights_gib, 4096)
        assert np.isclose(kv, f.kv_gib) and tok == f.kv_tokens and np.isclose(ses, f.sessions)
kv, tok, ses = kv_room(g("olmoe-1b-7b"), configs.gpu("T4"), configs.weight_bytes(g("olmoe-1b-7b"), "fp16") / configs.GiB, 4096)
assert kv < 0 and tok == 0
kv3, tok3, ses3 = kv_room(g("qwen3-30b-a3b"), configs.gpu("L4"), configs.weight_bytes(g("qwen3-30b-a3b"), "int4") / configs.GiB, 8192)
print(f"✅ OLMoE 16-bit on a T4: {kv:.1f} GiB of KV (does not start); Qwen3-30B-A3B INT4 on an L4: "
      f"{kv3:.1f} GiB = {tok3:,} tokens = {ses3:.1f} sessions of 8K")

# %% [markdown]
# ## Exercise 5.3 — how much to offload
#
# vLLM's `--cpu-offload-gb N` moves N GiB of weights per GPU to pinned CPU memory (with
# `--cpu-offload-params experts`, only parameters whose name has an `experts` segment — the routed
# experts). Write `min_offload(model, gpu, kv_tokens, util=0.92, overhead_gib=1.5)`: the smallest N,
# in steps of 0.5 GiB, that leaves room for `kv_tokens` tokens of KV with 16-bit weights.

# %% exercise
def min_offload(model, gpu, kv_tokens, util=0.92, overhead_gib=1.5):
    ### BEGIN SOLUTION
    need = configs.weight_bytes(model, "fp16") / configs.GiB + kv_tokens * model.kv_bytes_per_token() / configs.GiB
    short = need - (util * gpu.memory_gib - overhead_gib)
    return max(0.0, np.ceil(short / 0.5) * 0.5)
    ### END SOLUTION

# %% check
olmoe, T4 = g("olmoe-1b-7b"), configs.gpu("T4")
assert min_offload(olmoe, T4, 16_384) == 3.0
for tokens in (4096, 16_384, 65_536):
    for key in ("olmoe-1b-7b", "qwen1.5-moe-a2.7b"):
        assert min_offload(g(key), T4, tokens) == offload.min_offload_gib(g(key), T4, "fp16", kv_tokens=tokens)
assert min_offload(g("granite-3b-a800m"), T4, 16_384) == 0.0
print(f"✅ OLMoE on a T4 with 16K tokens of KV: --cpu-offload-gb {min_offload(olmoe, T4, 16_384):g}; with 64K: "
      f"{min_offload(olmoe, T4, 65_536):g} (of {configs.expert_bytes(olmoe, 'fp16') / configs.GiB:.1f} GiB of experts)")

# %% [markdown]
# ## Worked example: read the start-up log
#
# vLLM prints what it actually did. `offload.parse_startup_log` pulls the capacity lines (the wording
# of vLLM v0.30.0, as layer 04's `servelab.sizing` verified it) and the fused-MoE config line: on a
# T4 or L4 there is no tuned kernel config, so expect *"Using default MoE config. Performance might be
# sub-optimal!"*. The two logs here are illustrative (made from `offload.fit`, so they match it); on
# a real GPU, `kv_error_gib` is your calibration.

# %%
FIX = Path(hooks.FIXTURES)
for fname, gname, off in (("vllm_startup_olmoe_t4_offload.log", "T4", 3.0), ("vllm_startup_olmoe_l4.log", "L4", 0.0)):
    text = (FIX / fname).read_text()
    log = offload.parse_startup_log(text)
    pred = offload.fit(olmoe, configs.gpu(gname), "fp16", offload_gib=off)
    cmp_ = offload.compare_with_log(pred, log)
    print(f"{fname} [ILLUSTRATIVE]\n   {text.splitlines()[1][2:]}")
    print(f"   loaded {log['model_loading_gib']:.2f} GiB (predicted {cmp_['pred_weights_gib']:.2f}); KV {log['available_kv_gib']:.2f} GiB "
          f"= {log['kv_cache_tokens']:,} tokens, {log['max_concurrency']:.2f}x of {log['max_model_len']:,}; "
          f"default MoE config: {log['moe_default_config']}")

# %% [markdown]
# ## Exercise 5.4 — the price of each way off the GPU
#
# Write `offload_ms(gib, pcie_gbs)`: UVA offload reads every offloaded byte over PCIe in every
# forward pass (GiB = 2^30 bytes, GB/s = 10^9). And `cpu_experts_ms(model, batch, cpu_bw_gbs=40,
# cpu_tflops=0.3)`: llama.cpp-style CPU experts, per step, `max(read, compute)` where read = layers ×
# `stream.experts_touched(E, k, batch)` × bytes of one 4-bit expert (`configs.expert_bytes(model,
# "int4-experts") / (layers × E)`) over the DRAM bandwidth, and compute = 2 × layers × batch × k ×
# `expert_params()` FLOPs over the CPU's FLOP/s. The two CPU numbers describe *your* host (assumed
# here; measure with a STREAM-like copy and a matmul).

# %% exercise
def offload_ms(gib, pcie_gbs):
    ### BEGIN SOLUTION
    return gib * configs.GiB / (pcie_gbs * 1e9) * 1e3
    ### END SOLUTION


def cpu_experts_ms(model, batch, cpu_bw_gbs=40.0, cpu_tflops=0.3):
    ### BEGIN SOLUTION
    one = configs.expert_bytes(model, "int4-experts") / (model.layers * model.n_experts)
    read = model.layers * stream.experts_touched(model.n_experts, model.top_k, batch) * one / (cpu_bw_gbs * 1e9)
    compute = 2 * model.layers * batch * model.top_k * model.expert_params() / (cpu_tflops * 1e12)
    return max(read, compute) * 1e3
    ### END SOLUTION

# %% check
assert round(offload_ms(4, 25)) == 172                                    # ~170 ms per step for 4 GiB at 25 GB/s
for b in (1, 4, 16, 64):
    assert np.isclose(cpu_experts_ms(olmoe, b), offload.cpu_expert_step_s(olmoe, b) * 1e3)
uva = offload_ms(3.0, T4.pcie_gbs)
print(f"   OLMoE on a T4, per step [simulated]: UVA offload of 3 GiB costs {uva:.0f} ms at any batch;")
for b in (1, 4, 16, 64):
    print(f"      batch {b:3d}: CPU experts {cpu_experts_ms(olmoe, b):6.1f} ms per step = {cpu_experts_ms(olmoe, b) / b:5.1f} ms/token; "
          f"UVA {uva / b:5.1f} ms/token")
print("✅ UVA offload is a fixed toll per step (amortised by batch); CPU experts scale with the batch")

# %% [markdown]
# ## Exercise 5.5 — where the two cross
#
# Write `uva_wins_from(model, gpu, gib, max_batch=512)`: the smallest batch at which the UVA toll per
# token (`offload_ms(gib, gpu.pcie_gbs) / batch`) is below the CPU experts' cost per token
# (`cpu_experts_ms(model, batch) / batch`), or None. (Both paths also pay the GPU's own step; it is
# the same order in both and left out.)

# %% exercise
def uva_wins_from(model, gpu, gib, max_batch=512):
    ### BEGIN SOLUTION
    toll = offload_ms(gib, gpu.pcie_gbs)
    for b in range(1, max_batch + 1):
        if toll / b < cpu_experts_ms(model, b) / b:
            return b
    return None
    ### END SOLUTION

# %% check
b_t4 = uva_wins_from(olmoe, T4, 3.0)
assert b_t4 is not None and b_t4 > 1
assert cpu_experts_ms(olmoe, b_t4) > offload_ms(3.0, T4.pcie_gbs) >= cpu_experts_ms(olmoe, b_t4 - 1)
b_l4 = uva_wins_from(olmoe, configs.gpu("L4"), 3.0)
assert b_l4 <= b_t4                                                       # faster PCIe, earlier crossover
print(f"✅ [simulated] one user: CPU experts; from batch {b_t4} on a T4 (Gen3) or {b_l4} on an L4 (Gen4), "
      f"UVA offload is cheaper per token. 4-bit experts on the GPU avoid both tolls where the kernels exist.")

# %% [markdown]
# ## On a real GPU (T1)
#
# Free Colab/Kaggle T4 (16 GB, compute capability 7.5: `--dtype half`, no bf16, no FP8, no MXFP4).
# Checkpoint ids and GGUF files are (verify) on the Hub:
#
# ```bash
# pip install "vllm==0.30.0"
# # 16-bit OLMoE with its experts partly in CPU memory: 3 GiB is Exercise 5.3's answer for 4 x 4K tokens
# # (Colab has ~12 GB of RAM: keep N modest)
# vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype half --max-model-len 4096 \
#     --cpu-offload-gb 3 --cpu-offload-params experts 2>&1 | tee vllm-offload.log
# # a 4-bit MoE that fits outright
# vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --dtype half --max-model-len 4096 2>&1 | tee vllm-int4.log
# # 24 GB (L4 / RTX 4090): Qwen3-30B-A3B in INT4, gpt-oss-20b in MXFP4
# vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 --max-model-len 8192
# vllm serve openai/gpt-oss-20b
# # llama.cpp: experts on the CPU, the rest on the GPU
# llama-server -m <olmoe-1b-7b Q4 GGUF> -ngl 99 --n-cpu-moe 16 --port 8080
# ```
#
# Then `MOELAB_URL=http://127.0.0.1:8000` and run the cell below: it parses the log you saved and
# measures the step at batch 1 and 8. llama-server also serves `/v1/completions`; point `MOELAB_URL`
# at its port to measure it the same way (whether it honours `ignore_eos` is (verify)).

# %%
url = env.server_url()
if url:
    for f in ("vllm-offload.log", "vllm-int4.log"):
        if Path(f).exists():
            print(f, offload.parse_startup_log(Path(f).read_text()))
    name = env.served_model(url, env.auth_headers())
    for b in (1, 8):
        pt = stream.measure_itl(url, name, b, output_tokens=64, headers=env.auth_headers())
        print(f"MEASURED {name} batch {b}: median ITL {pt.median_itl_ms:.1f} ms, {pt.tokens_per_s:.1f} tok/s")
elif env.may_start_vllm() and env.gpu_count() >= 1:
    flags = ["--max-model-len", "4096"] + (["--dtype", "half"] if not GPU.bf16 else [])
    off = min_offload(olmoe, GPU, 16_384)
    if off:
        flags += ["--cpu-offload-gb", f"{off:g}", "--cpu-offload-params", "experts"]
    with env.VLLMServer(olmoe.hf_id, flags, log_path="vllm-offload.log") as u:
        print(offload.parse_startup_log(Path("vllm-offload.log").read_text()))
        for b in (1, 8):
            pt = stream.measure_itl(u, olmoe.hf_id, b, output_tokens=64)
            print(f"MEASURED batch {b}: median ITL {pt.median_itl_ms:.1f} ms")
else:
    print("T0: no MOELAB_URL and no local GPU + vLLM (MOELAB_START_VLLM=1); every number above is simulated or illustrative.")

# %% [markdown]
# ## Worked example: a report for the review
#
# `moelab.report` writes what you found with its provenance on every section, so a table pasted into
# a design document says whether it was measured, simulated or illustrative (`.save()` writes
# Markdown and JSON to `results/`; here it is only printed).

# %%
from moelab.report import Report

rep = Report(f"OLMoE-1B-7B on one {GPU.name}")
rep.add("What fits", env.SIMULATED, "\n".join(offload.fit(olmoe, GPU, s).line() for s in ("fp16", "int4")))
rep.add("Offload toll vs CPU experts, per step", env.SIMULATED,
        "\n".join(f"batch {b:3d}: UVA {offload_ms(3.0, GPU.pcie_gbs):6.1f} ms, CPU experts {cpu_experts_ms(olmoe, b):6.1f} ms"
                  for b in (1, 16, 64)), crossover_batch=uva_wins_from(olmoe, GPU, 3.0))
rep.add("Start-up log", env.MEASURED if url else env.ILLUSTRATIVE,
        (FIX / "vllm_startup_olmoe_t4_offload.log").read_text().splitlines()[5] if not url else "see vllm-offload.log")
print(rep.to_markdown())

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "The MoE's memory bill is its total parameter count, so a 7B-total MoE is a 14 GB
# problem on a 16 GB card even though it computes like a 1.3B model — there is no KV cache left. We
# have three ways out. Four-bit experts cut the expert bytes about 3.8x and keep everything on the GPU;
# that is the default where the kernels exist, after an accuracy check on our evals, keeping the router
# in 16-bit. vLLM's CPU offload keeps some weights in pinned host memory and streams them over PCIe
# every step — a fixed toll per step that only a large batch amortises. llama.cpp's CPU experts read
# only the touched experts and compute them on the CPU — the right shape for a single user, but it
# does not scale with batch. And whatever we choose, what is left is KV cache: MoE does not shrink it."
#
# **Drill 1.** *We offloaded 3 GiB with `--cpu-offload-gb`; routing is sparse, so only the touched
# experts should cross PCIe, right?* — No: UVA offload reads the offloaded tensors in every forward
# pass regardless of routing; at ~12 GB/s (T4, PCIe Gen3, verify) 3 GiB is ~0.27 s per step.
#
# **Drill 2.** *Can we serve gpt-oss-20b on a free T4?* — Not with vLLM: its MXFP4 path needs compute
# capability 8.0 and bf16 activations; a T4 is 7.5 with no bf16. An L4 (24 GB, 8.9) runs it.
#
# **Drill 3.** *Which layers should stay 16-bit when we quantize an MoE?* — The router (gate) and the
# shared-expert gate: routing is sensitive to quantization error (llm-compressor's MoE examples keep
# `mlp.gate` in 16-bit), and they are tiny. The experts are where the bytes are.
