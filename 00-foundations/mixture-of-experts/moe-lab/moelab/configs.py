"""configs.py — the handful of config.json fields that size an MoE model, and the GPUs to run it on.

One idea: three parameter counts answer three different questions. Memory holds the TOTAL (every
expert, hot or cold); a token's FLOPs follow the ACTIVE count (its top-k experts, plus any shared
expert, plus attention); a decode step's bytes follow what the batch TOUCHES (stream.py). Count
them from the config, and name the embedding convention you used — published "active" numbers
differ on it (gpt-oss counts the LM head only; DeepSeek and layer 01's ``roofline.llm`` count both
embedding tables).

The lab is standalone (it never imports this topic's ``moecore`` or layer 01's ``roofline``), so
the counting is re-implemented here in the form ``roofline.llm.ModelConfig`` uses: matmul weights
(+ biases where a model has them) and embeddings, norms ignored (< 0.01%). The tests pin every
entry to the totals in the fact sheet and in the model cards.

Configs are literal data read from the upstream model code/configs on 2026-09-26 (transformers,
gpt-oss, olmoe); entries whose fields are derived rather than read carry ``verify=True``.
"""
from __future__ import annotations

from dataclasses import dataclass

AS_OF = "2026-09-26"
GiB = 1024 ** 3


@dataclass(frozen=True)
class Model:
    """A decoder-only model; ``n_experts == 0`` means dense (``expert_ff`` is then the MLP width)."""
    name: str
    hf_id: str                 # the checkpoint a T1 run would load (verify the id on the Hub)
    layers: int
    d_model: int
    vocab: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_ff: int             # intermediate size of one routed expert (dense: the MLP)
    n_experts: int = 0         # routed experts per MoE layer
    top_k: int = 0             # routed experts per token
    shared_ff: int = 0         # intermediate size of the always-on shared expert (0 = none)
    shared_gate: bool = False  # Qwen2-MoE: sigmoid(x @ w[d, 1]) scales the shared expert
    tied: bool = False         # one embedding table for input and LM head
    qkv_bias: bool = False
    expert_bias: bool = False  # gpt-oss: mlp1 [2I, d] + bias, mlp2 [d, I] + bias
    router_bias: bool = False  # gpt-oss: the gate is a Linear with bias
    attn_extra: int = 0        # per-layer extras (gpt-oss: o-proj bias + attention sinks)
    min_cc: float = 7.5        # lowest compute capability the stated precision runs on in vLLM
    native: str = "bf16"       # checkpoint precision
    verify: bool = False       # some fields derived, not read from a config
    note: str = ""

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    # ---- per layer ------------------------------------------------------------------------
    def attn_params(self) -> int:
        q, kv = self.n_heads * self.head_dim, self.n_kv_heads * self.head_dim
        p = self.d_model * q + 2 * self.d_model * kv + q * self.d_model
        if self.qkv_bias:
            p += q + 2 * kv
        return p + self.attn_extra

    def expert_params(self) -> int:
        """One routed expert (SwiGLU: gate, up, down = 3 matrices); dense: the MLP."""
        p = 3 * self.d_model * self.expert_ff
        if self.expert_bias:
            p += 2 * self.expert_ff + self.d_model
        return p

    def expert_matmul_params(self) -> int:
        """The part of an expert that 4-bit expert formats (MXFP4, INT4 experts) quantize."""
        return 3 * self.d_model * self.expert_ff

    def shared_params(self) -> int:
        if not self.shared_ff:
            return 0
        return 3 * self.d_model * self.shared_ff + (self.d_model if self.shared_gate else 0)

    def router_params(self) -> int:
        if not self.is_moe:
            return 0
        return self.d_model * self.n_experts + (self.n_experts if self.router_bias else 0)

    def layer_params(self) -> int:
        mlp = self.expert_params() * max(1, self.n_experts) + self.shared_params() + self.router_params()
        return self.attn_params() + mlp

    def active_layer_params(self) -> int:
        k = self.top_k if self.is_moe else 1
        return self.attn_params() + k * self.expert_params() + self.shared_params() + self.router_params()

    # ---- whole model ----------------------------------------------------------------------
    def embedding_params(self) -> int:
        return self.vocab * self.d_model * (1 if self.tied else 2)

    def total_params(self) -> int:
        return self.layers * self.layer_params() + self.embedding_params()

    def active_params(self, convention: str = "both") -> int:
        """Parameters one token uses. ``convention``: "both" embedding tables (roofline.llm,
        DeepSeek), "head" = LM head only (OpenAI's gpt-oss numbers), "matmul" = neither."""
        mm = self.layers * self.active_layer_params()
        extra = {"both": self.embedding_params(), "head": self.vocab * self.d_model, "matmul": 0}[convention]
        return mm + extra

    def expert_param_total(self) -> int:
        """All routed experts in all layers — what expert offload and INT4-experts act on."""
        return self.layers * self.n_experts * self.expert_params() if self.is_moe else 0

    def kv_bytes_per_token(self, kv_bytes: float = 2) -> float:
        """2 (K and V) x layers x kv_heads x head_dim x bytes. MoE does not change it."""
        return 2 * self.layers * self.n_kv_heads * self.head_dim * kv_bytes


# ---- the catalogue ---------------------------------------------------------------------------
# Mixtral and Qwen3-30B-A3B match layer 01's roofline.llm.PRESETS field for field (the tests
# reproduce its PRIMER §3.6 table with them). The small models are the T1 candidates.
M = Model
MODELS: dict[str, Model] = {
    "mixtral-8x7b": M("Mixtral-8x7B", "mistralai/Mixtral-8x7B-Instruct-v0.1", 32, 4096, 32_000, 32, 8, 128, 14_336,
                      n_experts=8, top_k=2, min_cc=8.0,
                      note="transformers configuration_mixtral.py defaults; = roofline.llm.PRESETS['mixtral-8x7b']"),
    "qwen3-30b-a3b": M("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", 48, 2048, 151_936, 32, 4, 128, 768,
                       n_experts=128, top_k=8, min_cc=8.0,
                       note="Qwen3 report Table 2; no shared expert; = roofline.llm.PRESETS['qwen3-30b-a3b']"),
    "olmoe-1b-7b": M("OLMoE-1B-7B-0924", "allenai/OLMoE-1B-7B-0924-Instruct", 16, 2048, 50_304, 16, 16, 128, 1024,
                     n_experts=64, top_k=8, verify=True,
                     note="olmoe configs/OLMoE-1B-7B-0924.yml; expert size derived from the published 6.9B/1.3B"),
    "granite-1b-a400m": M("granite-3.0-1b-a400m", "ibm-granite/granite-3.0-1b-a400m-instruct", 24, 1024, 49_155, 16, 8, 64,
                          512, n_experts=32, top_k=8, tied=True, verify=True,
                          note="derived config that reproduces '1b-a400m'; vLLM docs list the -base id"),
    "granite-3b-a800m": M("granite-3.0-3b-a800m", "ibm-granite/granite-3.0-3b-a800m-instruct", 32, 1536, 49_155, 24, 8, 64,
                          512, n_experts=40, top_k=8, tied=True, verify=True,
                          note="derived config that reproduces '3b-a800m'"),
    "qwen1.5-moe-a2.7b": M("Qwen1.5-MoE-A2.7B", "Qwen/Qwen1.5-MoE-A2.7B-Chat", 24, 2048, 151_936, 16, 16, 128, 1408,
                           n_experts=60, top_k=4, shared_ff=5632, shared_gate=True, qkv_bias=True,
                           note="configuration_qwen2_moe.py defaults; the shared expert is the size of 4 routed ones"),
    "gpt-oss-20b": M("gpt-oss-20b", "openai/gpt-oss-20b", 24, 2880, 201_088, 64, 8, 64, 2880,
                     n_experts=32, top_k=4, expert_bias=True, router_bias=True, qkv_bias=True, attn_extra=2880 + 64,
                     min_cc=8.0, native="mxfp4", verify=True,
                     note="gpt-oss torch ModelConfig; 24 layers derived (reproduces 21B/3.6B); MXFP4 experts need sm80+ and bf16"),
    "gpt-oss-120b": M("gpt-oss-120b", "openai/gpt-oss-120b", 36, 2880, 201_088, 64, 8, 64, 2880,
                      n_experts=128, top_k=4, expert_bias=True, router_bias=True, qkv_bias=True, attn_extra=2880 + 64,
                      min_cc=8.0, native="mxfp4", note="gpt-oss torch ModelConfig: 116.8B total, 5.1B active (head only)"),
    # dense references
    "qwen2.5-1.5b": M("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 28, 1536, 151_936, 12, 2, 128, 8960, tied=True,
                      note="= roofline.llm.PRESETS['qwen2.5-1.5b'] (biases ignored there and here)"),
    "llama-3.1-8b": M("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", 32, 4096, 128_256, 32, 8, 128, 14_336,
                      note="= roofline.llm.PRESETS['llama-3.1-8b']"),
}


def get(name: str) -> Model:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}") from None


# ---- precision: what a byte of weights costs ------------------------------------------------
# name -> (bits for attention/router/shared/dense weights, bits for routed-expert matmuls).
# Embeddings and LM head stay 16-bit in every quantized checkpoint considered here (GPTQ, AWQ,
# MXFP4, FP8 recipes keep them), and 4-bit formats carry scales: 4.25 bits per weight
# (INT4 with a 16-bit scale per 64, or MXFP4's 8-bit scale per 32) is the estimate used throughout.
SCHEMES: dict[str, tuple[float, float]] = {
    "fp16": (16, 16),
    "bf16": (16, 16),
    "fp8": (8, 8),
    "int4": (4.25, 4.25),              # GPTQ/AWQ-style: every linear layer
    "int4-experts": (16, 4.25),        # only the routed experts (router and attention stay 16-bit)
    "mxfp4": (16, 4.25),               # gpt-oss: MoE projection weights MXFP4, the rest bf16
}


def weight_bytes(model: Model, scheme: str = "bf16") -> float:
    """Bytes of the checkpoint's weights under ``scheme`` (see ``SCHEMES``)."""
    rest_bits, exp_bits = SCHEMES[scheme]
    emb = model.embedding_params()
    exp_mm = model.layers * model.n_experts * model.expert_matmul_params() if model.is_moe else 0
    rest = model.total_params() - emb - exp_mm
    emb_bits = 16
    return (rest * rest_bits + exp_mm * exp_bits + emb * emb_bits) / 8


def expert_bytes(model: Model, scheme: str = "bf16") -> float:
    """Bytes of the routed experts alone (what ``--cpu-offload-params experts`` can move)."""
    if not model.is_moe:
        return 0.0
    _, exp_bits = SCHEMES[scheme]
    mm = model.layers * model.n_experts * model.expert_matmul_params()
    bias = model.expert_param_total() - mm
    return (mm * exp_bits + bias * 16) / 8


# ---- GPUs ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GPU:
    """Datasheet numbers (dense, no sparsity) as layer 01's roofline.specs and layer 04's
    servelab.sizing use them; ``memory_gib`` is what the driver reports. All (verify)."""
    name: str
    memory_gib: float
    mem_bw_gbs: float          # GB/s
    tflops_16: float           # dense FP16/BF16 tensor TFLOPS
    compute_capability: float
    bf16: bool = True          # False on Turing (T4): vLLM needs --dtype half
    fp8: bool = False          # FP8 tensor cores (Ada, Hopper, Blackwell)
    pcie_gbs: float = 25.0     # effective host<->device bandwidth, pinned memory (verify)
    note: str = ""


GPUS: dict[str, GPU] = {
    "T4": GPU("T4", 15.0, 320, 65, 7.5, bf16=False, pcie_gbs=12.0,
              note="PCIe Gen3 x16; no bf16, no FP8; free on Colab/Kaggle"),
    "L4": GPU("L4", 22.49, 300, 121, 8.9, fp8=True, pcie_gbs=25.0, note="PCIe Gen4 x16; GCP G2, Cloud Run"),
    "RTX4090": GPU("RTX4090", 23.99, 1008, 165, 8.9, fp8=True, pcie_gbs=25.0, note="rented on RunPod/Vast"),
    "A100-80GB": GPU("A100-80GB", 80.0, 2039, 312, 8.0, pcie_gbs=25.0),
    "H100-80GB": GPU("H100-80GB", 79.65, 3352, 989, 9.0, fp8=True, pcie_gbs=50.0),
    "H200": GPU("H200", 141.0 * 1e9 / GiB, 4800, 989.4, 9.0, fp8=True, pcie_gbs=50.0,
                note="roofline.specs 'h200': 989.4 TFLOPS bf16, 4.8 TB/s, 141 GB"),
}


def gpu(name: str) -> GPU:
    try:
        return GPUS[name]
    except KeyError:
        raise KeyError(f"unknown GPU {name!r}; known: {', '.join(GPUS)}") from None


def summary(model: Model) -> str:
    """One line per model: total, active (both conventions), KV per token."""
    t, a, h = model.total_params(), model.active_params("both"), model.active_params("head")
    moe = f"E={model.n_experts} k={model.top_k}" + (f" +shared({model.shared_ff})" if model.shared_ff else "") \
        if model.is_moe else "dense"
    return (f"{model.name:22s} {moe:22s} total {t / 1e9:6.2f} B  active {a / 1e9:5.2f} B (both emb) "
            f"/ {h / 1e9:5.2f} B (head only)  KV {model.kv_bytes_per_token() / 1024:5.0f} KiB/token"
            + ("  (verify)" if model.verify else ""))
