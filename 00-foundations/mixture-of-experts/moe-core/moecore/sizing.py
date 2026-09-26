"""Sizing an MoE deployment: memory by total, prefill by active, decode by what a step streams.

The one idea: three different parameter counts size three different things. HBM must hold the
TOTAL parameters (every expert, hot or not) plus the KV cache; prefill FLOPs follow the ACTIVE
parameters (2 x active x tokens, as `gpu-capacity-planning/capacity.py` computes them); a decode
step's time follows the bytes it STREAMS at your batch -- near active at batch 1, near total at
large batch (touched.py). So pick the GPU count (= the EP degree, with data-parallel attention)
that holds weights plus the batch's KV, check the step time against the ITL target, then divide
the GPU-hours by the tokens. Every number is a roofline bound (simulated), not a measurement.

MODELS: configs as literal data, from each model's config / reference code in the upstream repos
(transformers, deepseek-v3, gpt-oss, llama-models, olmoe) as of 2026-09-26; entries whose fields
could not be read from a source reproduce the published totals and are marked "(verify)".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .ep import Link, decode_on, wide_ep_weights
from .moe import MoEConfig, gqa_params, mla_params
from .touched import Device

AS_OF = "2026-09-26"

C = MoEConfig
MODELS = {
    "mixtral-8x7b": C("Mixtral-8x7B", 32, 4096, 32_000, 32, 8, 128, 8, 2, 14_336,
                      note="transformers configuration_mixtral.py defaults"),
    "qwen3-30b-a3b": C("Qwen3-30B-A3B", 48, 2048, 151_936, 32, 4, 128, 128, 8, 768,
                       note="Qwen3 report Table 2; no shared expert"),
    "qwen3-235b-a22b": C("Qwen3-235B-A22B", 94, 4096, 151_936, 64, 4, 128, 128, 8, 1536,
                         note="d_model and expert size (verify); reproduce 235B/22B"),
    "qwen1.5-moe-a2.7b": C("Qwen1.5-MoE-A2.7B", 24, 2048, 151_936, 16, 16, 128, 60, 4, 1408, shared_ff=5632,
                           n_shared=1, attn=gqa_params(2048, 16, 16, 128, qkv_bias=True), router_extra=2048,
                           note="configuration_qwen2_moe.py defaults; sigmoid-gated shared expert = 4 routed"),
    "deepseek-v3": C("DeepSeek-V3", 61, 7168, 129_280, 128, 1, 128, 256, 8, 2048, shared_ff=2048, n_shared=1,
                     dense_layers=3, dense_ff=18_432, attn=mla_params(7168, 128, 1536, 512, 128, 64, 128),
                     router_extra=256, kv_elems=512 + 64,
                     note="deepseek-v3 config_671B.json; MLA (KV = 576-wide latent); +14B MTP not counted"),
    "gpt-oss-120b": C("gpt-oss-120b", 36, 2880, 201_088, 64, 8, 64, 128, 4, 2880,
                      attn=gqa_params(2880, 64, 8, 64, qkv_bias=True, o_bias=True, sinks=True),
                      expert_extra=2 * 2880 + 2880, router_extra=128,
                      note="gpt-oss torch ModelConfig; experts MXFP4, rest bf16"),
    "gpt-oss-20b": C("gpt-oss-20b", 24, 2880, 201_088, 64, 8, 64, 32, 4, 2880,
                     attn=gqa_params(2880, 64, 8, 64, qkv_bias=True, o_bias=True, sinks=True),
                     expert_extra=2 * 2880 + 2880, router_extra=32, note="24 layers derived (verify)"),
    "llama4-scout": C("Llama 4 Scout (text)", 48, 5120, 202_048, 40, 8, 128, 16, 1, 8192, shared_ff=8192,
                      n_shared=1, note="Llama4TextConfig defaults; card 109B incl. vision"),
    "llama4-maverick": C("Llama 4 Maverick (text)", 48, 5120, 202_048, 40, 8, 128, 128, 1, 8192, shared_ff=8192,
                         n_shared=1, dense_layers=24, dense_ff=16_384,
                         note="MoE every 2nd layer, dense I=16384 (verify); reproduces 400B"),
    "olmoe-1b-7b": C("OLMoE-1B-7B", 16, 2048, 50_304, 16, 16, 128, 64, 8, 1024,
                     note="olmoe OLMoE-1B-7B-0924.yml; expert size derived (verify)"),
    "granite-3.0-1b-a400m": C("granite-3.0-1b-a400m", 24, 1024, 49_155, 16, 8, 64, 32, 8, 512, tied=True,
                              note="config derived from the name (verify)"),
    "granite-3.0-3b-a800m": C("granite-3.0-3b-a800m", 32, 1536, 49_155, 24, 8, 64, 40, 8, 512, tied=True,
                              note="config derived from the name (verify)"),
    "llama-3.1-8b": C("Llama-3.1-8B", 32, 4096, 128_256, 32, 8, 128, expert_ff=14_336, note="dense"),
    "llama-3.1-70b": C("Llama-3.1-70B", 80, 8192, 128_256, 64, 8, 128, expert_ff=28_672, note="dense"),
}


def weight_bytes(cfg: MoEConfig, bits: float = 16, expert_bits: float | None = None) -> float:
    """Bytes to hold the weights: routed-expert matrices at `expert_bits` (MXFP4 = 4.25: four bits
    plus an 8-bit scale per 32 values), everything else at `bits`."""
    routed = cfg.moe_layers * cfg.n_experts * (cfg.expert_params() - cfg.expert_extra)
    eb = bits if expert_bits is None else expert_bits
    return routed * eb / 8 + (cfg.total() - routed) * bits / 8


def kv_gb(cfg: MoEConfig, tokens: int, kv_bytes: float = 2) -> float:
    return tokens * cfg.kv_bytes_per_token(kv_bytes) / 1e9


def sessions(cfg: MoEConfig, device: Device, n_gpus: int, context: int, bits: float = 16,
             expert_bits: float | None = None, kv_bytes: float = 2, reserve: float = 0.10,
             layout: str = "tp") -> int:
    """Sequences of `context` tokens whose KV fits beside the weights (10% headroom, like layer 01).
    "tp": the weights are stored once across the n GPUs. "ep": every rank also holds a full copy of
    the non-expert weights (data-parallel attention), so each rank has less room for KV."""
    per_token = context * cfg.kv_bytes_per_token(kv_bytes)
    if layout == "ep":
        mine = wide_ep_weights(cfg, n_gpus, 1) * weight_bytes(cfg, bits, expert_bits) / cfg.total()
        return n_gpus * max(0, math.floor((device.memory_gb * 1e9 * (1 - reserve) - mine) / per_token))
    free = n_gpus * device.memory_gb * 1e9 * (1 - reserve) - weight_bytes(cfg, bits, expert_bits)
    return max(0, math.floor(free / per_token))


def prefill_flops(active_params: float, tokens: int) -> float:
    """2 x active x tokens -- capacity.prefill_flops(active_b, prompt_tokens), attention ignored."""
    return 2 * active_params * tokens


def decode_floor(total_bytes: float, n_gpus: int, device: Device) -> float:
    """Seconds to stream `total_bytes` once across n GPUs: the large-batch MoE decode floor, when
    every expert is touched every step. Mistral Large 3 FP8 (675 GB) on 8 H200s: 17.6 ms."""
    return total_bytes / (n_gpus * device.bandwidth())


def usd_per_mtok(n_gpus: int, usd_per_gpu_hr: float, tokens_per_s: float) -> float:
    return n_gpus * usd_per_gpu_hr / 3600 / tokens_per_s * 1e6


@dataclass(frozen=True)
class Plan:
    gpus: int
    sessions: int        # what fits at this context
    step_ms: float       # roofline + all-to-all, simulated
    tok_s: float         # batch / step
    usd_per_mtok: float | None


def plan(cfg: MoEConfig, device: Device, batch: int, context: int, itl_ms: float, link: Link | None = None,
         bits: float = 16, kv_bytes: float = 2, usd_per_gpu_hr: float | None = None, layout: str = "ep",
         options=(1, 2, 4, 8, 16, 32, 64), precision: str = "bf16") -> Plan | None:
    """Smallest GPU count that holds the weights plus `batch` sequences of KV and decodes one step
    within `itl_ms` in this layout ("ep" = DP attention + EP; "tp" = everything sharded). None if none does."""
    for n in options:
        fit = sessions(cfg, device, n, context, bits, kv_bytes=kv_bytes, layout=layout)
        if fit < batch:
            continue
        st = decode_on(cfg, device, batch, context, n, layout, link, bits / 8, kv_bytes, precision)
        if st["time"] * 1e3 <= itl_ms:
            tok_s = batch / st["time"]
            usd = usd_per_mtok(n, usd_per_gpu_hr, tok_s) if usd_per_gpu_hr else None
            return Plan(n, fit, st["time"] * 1e3, tok_s, usd)
    return None


def offload_step_s(offload_gb: float, pcie_gbs: float = 25) -> float:
    """vLLM --cpu-offload-gb streams the offloaded weights over PCIe in *every* forward pass,
    touched or not: 4 GiB at ~25 GB/s effective (verify) adds ~172 ms per step."""
    return offload_gb * 2 ** 30 / (pcie_gbs * 1e9)
