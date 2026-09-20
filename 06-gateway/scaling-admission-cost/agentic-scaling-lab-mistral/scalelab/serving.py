"""What one self-hosted replica delivers: memory → concurrency, bandwidth → latency, batch ↔ throughput.

A vLLM replica (one copy of the weights spread over ``tp`` GPUs) has two budgets.

* **Memory.** Weights plus KV cache. KV bytes per token × tokens resident ≤ what is left after the
  weights, so the model's attention layout (GQA vs MLA) decides how many calls can be in flight.
* **Time per decode step.** Every step streams the weights it touches plus the KV of every running
  sequence through HBM. Time-per-output-token (TPOT) therefore grows with the batch (and with
  context length), while aggregate tokens/s grows with the batch until the step is compute-bound.
  A hosted API hides this trade-off; on your own GPUs *the latency you want is the batch you can
  afford*.

Architecture numbers come from each model's config.json / params.json and public GPU specs
(checked 19 Sep 2026). Throughput is an ESTIMATE from first principles with an explicit efficiency
factor — replace it with ``vllm bench serve`` measurements before anyone buys GPUs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# --- hardware --------------------------------------------------------------------------------

@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float
    bandwidth_tb_s: float
    fp8_tflops: float          # dense


GPUS = {
    "h100": GPU("H100 SXM 80 GB", 80, 3.35, 1979),
    "h200": GPU("H200 141 GB", 141, 4.8, 1979),
    "b200": GPU("B200 180 GB", 180, 8.0, 4500),
    "l40s": GPU("L40S 48 GB", 48, 0.864, 733),
}

# USD per GPU-hour, 19 Sep 2026. On-demand unless stated; verify — these move monthly.
GPU_USD_PER_HOUR = {
    "h100": {"aws on-demand": 6.88, "aws capacity block (Tokyo/Sydney/Mumbai)": 4.72, "gcp on-demand": 11.06,
             "gcp 3-year": 4.86, "azure 3-year": 5.40, "neocloud": 3.75},
    "h200": {"aws on-demand": 7.91, "azure singapore on-demand": 13.78, "gcp on-demand": 10.60, "gcp 3-year": 4.65,
             "neocloud": 4.55},
    "b200": {"aws on-demand": 14.24, "neocloud": 6.90},
    "l40s": {"aws on-demand": 1.86, "aws 3-year": 0.80, "neocloud": 1.09},
}
HOURS_PER_MONTH = 730


# --- models ------------------------------------------------------------------------------------

def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bytes_per_elem: int = 2) -> int:
    """Grouped-query attention: K and V for every layer and KV head."""
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


def kv_bytes_per_token_mla(layers: int, kv_lora_rank: int, rope_dim: int, bytes_per_elem: int = 2) -> int:
    """Multi-head latent attention caches one compressed latent (plus the rope part) per layer."""
    return layers * (kv_lora_rank + rope_dim) * bytes_per_elem


@dataclass(frozen=True)
class OpenModel:
    name: str
    hf_repo: str
    params_b: float            # total parameters (billions)
    active_b: float            # active per token (= params_b for dense)
    weights_gb: float          # FP8 checkpoint
    kv_bytes_per_token: int    # BF16 KV cache
    experts: int = 1           # 1 = dense
    active_experts: int = 1
    tp: dict[str, int] | None = None   # GPUs per replica, by GPU key


OPEN_MODELS = {
    "ministral-14b": OpenModel("Ministral 3 14B", "mistralai/Ministral-3-14B-Instruct-2512", 13.9, 13.5, 15.7,
                               kv_bytes_per_token(40, 8, 128), tp={"h100": 1, "h200": 1, "l40s": 1, "b200": 1}),
    "ministral-8b": OpenModel("Ministral 3 8B", "mistralai/Ministral-3-8B-Instruct-2512", 8.8, 8.4, 10.4,
                              kv_bytes_per_token(34, 8, 128), tp={"h100": 1, "h200": 1, "l40s": 1, "b200": 1}),
    "mistral-small-4": OpenModel("Mistral Small 4", "mistralai/Mistral-Small-4-119B-2603", 119, 6.5, 121,
                                 kv_bytes_per_token_mla(36, 256, 64), experts=128, active_experts=4,
                                 tp={"h100": 2, "h200": 1, "b200": 1}),
    "mistral-large-3": OpenModel("Mistral Large 3", "mistralai/Mistral-Large-3-675B-Instruct-2512", 675, 41, 682,
                                 kv_bytes_per_token_mla(61, 512, 64), experts=128, active_experts=4,
                                 tp={"h200": 8, "b200": 8}),
}


# --- one replica -------------------------------------------------------------------------------

@dataclass
class Replica:
    """A vLLM replica: ``tp`` GPUs serving one copy of ``model``.

    ``efficiency`` is the share of peak HBM bandwidth a real decode step achieves; ``mfu`` the share of
    peak FLOPS during prefill; ``step_overhead_s`` the fixed cost of a step (scheduling, kernel launches).
    """
    model: OpenModel
    gpu: GPU
    tp: int
    max_num_seqs: int = 128               # vLLM default
    gpu_memory_utilization: float = 0.92  # vLLM default
    reserve_gb_per_gpu: float = 4.0       # activations, CUDA graphs
    kv_dtype_bytes: int = 2               # 1 with --kv-cache-dtype fp8
    efficiency: float = 0.6
    mfu: float = 0.4
    step_overhead_s: float = 0.002

    @property
    def kv_budget_gb(self) -> float:
        return self.tp * (self.gpu.memory_gb * self.gpu_memory_utilization - self.reserve_gb_per_gpu) - self.model.weights_gb

    @property
    def fits(self) -> bool:
        return self.kv_budget_gb > 0

    def kv_gb_per_token(self) -> float:
        return self.model.kv_bytes_per_token * self.kv_dtype_bytes / 2 / 1e9

    def max_seqs(self, context_tokens: int, shared_prefix_tokens: int = 0) -> int:
        """Sequences resident at once: the KV budget holds the shared prefix once and each sequence's own tokens."""
        if not self.fits:
            return 0
        tokens = self.kv_budget_gb / self.kv_gb_per_token()
        own = max(1, context_tokens - shared_prefix_tokens)
        return int(min(self.max_num_seqs, (tokens - shared_prefix_tokens) / own))

    def weights_read_gb(self, batch: int) -> float:
        """Bytes of weights a decode step streams. Dense: all of them. MoE: the experts the batch touches."""
        m = self.model
        if m.experts == 1:
            return m.weights_gb
        one_token = m.weights_gb * m.active_b / m.params_b               # attention + shared expert + one token's experts
        share = m.active_experts / m.experts
        extra = max(0.0, 1 - (1 - share) ** batch - share)                # further experts hit by the rest of the batch
        return one_token + (m.weights_gb - one_token) * extra

    def step_seconds(self, batch: int, context_tokens: int) -> float:
        """One decode step for ``batch`` sequences of ``context_tokens`` each — this is the TPOT."""
        kv_gb = batch * context_tokens * self.kv_gb_per_token()
        gb_per_s = self.tp * self.gpu.bandwidth_tb_s * 1e3 * self.efficiency
        return (self.weights_read_gb(batch) + kv_gb) / gb_per_s + self.step_overhead_s

    tpot = step_seconds

    def tokens_per_s(self, batch: int, context_tokens: int) -> float:
        return batch / self.step_seconds(batch, context_tokens)

    def ttft(self, new_prompt_tokens: int) -> float:
        """Prefill is compute-bound: 2 FLOPs per active parameter per token, at ``mfu`` of FP8 peak."""
        flops = 2 * self.model.active_b * 1e9 * new_prompt_tokens
        return flops / (self.tp * self.gpu.fp8_tflops * 1e12 * self.mfu)

    def batch_for_tpot(self, target_s: float, context_tokens: int) -> int:
        """The largest batch whose TPOT stays under ``target_s`` (0 if even batch 1 is too slow)."""
        best = 0
        for b in range(1, self.max_num_seqs + 1):
            if self.step_seconds(b, context_tokens) <= target_s:
                best = b
            else:
                break
        return best

    def call_seconds(self, batch: int, new_prompt_tokens: int, context_tokens: int, output_tokens: int) -> float:
        return self.ttft(new_prompt_tokens) + output_tokens * self.tpot(batch, context_tokens)

    def describe(self, context_tokens: int = 5_200, shared_prefix_tokens: int = 2_700) -> dict:
        return {"model": self.model.name, "gpu": f"{self.tp} × {self.gpu.name}", "weights_gb": self.model.weights_gb,
                "kv_kib_per_token": round(self.model.kv_bytes_per_token * self.kv_dtype_bytes / 2 / 1024, 1),
                "kv_budget_gb": round(self.kv_budget_gb, 1),
                "max_seqs": self.max_seqs(context_tokens, shared_prefix_tokens),
                "tpot_ms_at_1": round(self.tpot(1, context_tokens) * 1e3, 1),
                "tpot_ms_at_32": round(self.tpot(32, context_tokens) * 1e3, 1),
                "tok_s_at_32": round(self.tokens_per_s(32, context_tokens)),
                "tok_s_at_max": round(self.tokens_per_s(max(1, self.max_seqs(context_tokens, shared_prefix_tokens)), context_tokens)),
                "ttft_ms_2300_new": round(self.ttft(2_300) * 1e3)}


def replica(model_key: str, gpu_key: str, **kw) -> Replica:
    m, g = OPEN_MODELS[model_key], GPUS[gpu_key]
    if not m.tp or gpu_key not in m.tp:
        raise ValueError(f"{m.name} has no documented configuration on {g.name}")
    return Replica(m, g, m.tp[gpu_key], **kw)


def usd_per_hour(gpu_key: str, price_key: str) -> float:
    return GPU_USD_PER_HOUR[gpu_key][price_key]


if __name__ == "__main__":
    for mk, gk in (("ministral-14b", "h100"), ("ministral-14b", "l40s"), ("mistral-small-4", "h100"), ("mistral-small-4", "h200"),
                   ("mistral-large-3", "h200")):
        print(replica(mk, gk).describe())
