"""memory.py — will this training run fit? Weights, gradients, optimizer states, activations, logits, a teacher.

One idea: training memory is a sum of a few terms, each a count times bytes, and at a 151,936-token vocabulary the
term people forget — the logits — is often the largest (PRIMER §10 "Where to run it"):

    weights + grads + optimizer   16 B/param  mixed-precision AdamW with fp32 master weights (2 + 2 + 4 + 4 + 4);
                                              the HF Trainer on a T4 (fp32 weights, fp16 autocast) is also 16 B
                                   8 B/param  pure bf16 with AdamW states in bf16 (2 + 2 + 4)  — (verify)
                                   2 B/param  a frozen fp16/bf16 model (the teacher, or LoRA's base)
    LoRA adapters                 r · (d_in + d_out) per adapted matrix × layers, at 16 B/param
    activations                   with checkpointing: each layer's input, L · tokens · h · 2 B, plus one layer
                                  recomputed; without: ≈ L · tokens · h · (34 + 5 · heads · S / h) B (Korthikanti
                                  et al.'s count for a GPT layer in 16-bit; verify)
    logits                        tokens · V · 4 B per fp32 copy: the student's (+ its softmax and gradient) and,
                                  for logit KD, the teacher's — 2.49 GB per copy at 4,096 tokens and V = 151,936
                                  unless the loss is chunked

Parameter counts are exact for Qwen2/Qwen3/Llama configs (the q/k/v biases of Qwen2 and the q/k norms of
Qwen3 included; the same count as the 04 lab's ``servelab.sizing.param_count``, checked in
``tests/test_repo_numbers.py``). The verdicts are **predictions** — verify on the card; usable memory is the
driver-reported figure the 04 lab uses (a T4 shows 15.0 GiB). Standard library only.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 2 ** 30
USABLE_GIB = {"T4": 15.0, "L4": 22.49, "RTX4090": 23.99, "A100-40GB": 40.0, "A100-80GB": 80.0, "H100-80GB": 79.65}
BF16_OK = {"T4": False, "L4": True, "RTX4090": True, "A100-40GB": True, "A100-80GB": True, "H100-80GB": True}
BYTES_PER_PARAM = {"full": 16, "pure_bf16": 8, "frozen": 2}
LINEAR = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class Counts:
    total: int
    embedding: int
    non_embedding: int


def _dims(cfg: dict) -> dict:
    heads = int(cfg["num_attention_heads"])
    h = int(cfg["hidden_size"])
    return {"L": int(cfg["num_hidden_layers"]), "h": h, "heads": heads,
            "kv": int(cfg.get("num_key_value_heads") or heads), "hd": int(cfg.get("head_dim") or h // heads),
            "ffn": int(cfg["intermediate_size"]), "V": int(cfg["vocab_size"]),
            "tied": bool(cfg.get("tie_word_embeddings", False)), "type": str(cfg.get("model_type", ""))}


def param_count(cfg: dict) -> Counts:
    """Exact parameters of a dense Llama-style decoder from its config.json."""
    d = _dims(cfg)
    q, kv, o = d["h"] * d["heads"] * d["hd"], 2 * d["h"] * d["kv"] * d["hd"], d["heads"] * d["hd"] * d["h"]
    bias = (d["heads"] + 2 * d["kv"]) * d["hd"] if cfg.get("attention_bias", d["type"] == "qwen2") else 0
    qk_norm = 2 * d["hd"] if d["type"] == "qwen3" else 0
    mlp, norms = 3 * d["h"] * d["ffn"], 2 * d["h"]
    emb = d["V"] * d["h"] * (1 if d["tied"] else 2)
    total = d["L"] * (q + kv + o + bias + qk_norm + mlp + norms) + d["h"] + emb
    return Counts(total, emb, total - emb)


def matrices(cfg: dict) -> dict:
    """(d_in, d_out) of each linear layer PEFT can adapt, per decoder layer."""
    d = _dims(cfg)
    qo, kvw = d["heads"] * d["hd"], d["kv"] * d["hd"]
    return {"q_proj": (d["h"], qo), "k_proj": (d["h"], kvw), "v_proj": (d["h"], kvw), "o_proj": (qo, d["h"]),
            "gate_proj": (d["h"], d["ffn"]), "up_proj": (d["h"], d["ffn"]), "down_proj": (d["ffn"], d["h"])}


def lora_params(cfg: dict, r: int = 16, target="all-linear") -> int:
    """Trainable LoRA parameters: r · (d_in + d_out) per adapted matrix per layer. ``target`` is "all-linear"
    or a list of module names; PEFT 0.21.0's default for qwen2/qwen3/llama is ``["q_proj", "v_proj"]``."""
    names = LINEAR if target == "all-linear" else tuple(target)
    mats = matrices(cfg)
    return _dims(cfg)["L"] * sum(r * sum(mats[n]) for n in names)


def activation_bytes(cfg: dict, batch: int, seq: int, checkpointing: bool = True, bytes_: int = 2) -> float:
    d = _dims(cfg)
    tokens = batch * seq
    per_layer_full = tokens * d["h"] * (34 + 5 * d["heads"] * seq / d["h"]) * bytes_ / 2
    if checkpointing:
        return d["L"] * tokens * d["h"] * bytes_ + per_layer_full
    return d["L"] * per_layer_full


def logits_bytes(cfg: dict, batch: int, seq: int, copies: int = 3, chunk: int | None = None) -> float:
    """fp32 ``[tokens, V]`` tensors alive at once (logits, softmax, gradient ≈ 3 for a plain CE); ``chunk`` bounds
    the rows materialised together, as TRL's chunked losses do."""
    rows = batch * seq if chunk is None else min(chunk, batch * seq)
    return rows * _dims(cfg)["V"] * 4 * copies


def plan(student: dict, *, gpu: str = "T4", regime: str = "full", batch: int = 4, seq: int = 512,
         teacher: dict | None = None, r: int = 16, target="all-linear", checkpointing: bool = True,
         chunk: int | None = None) -> dict:
    """One training configuration's memory (bytes per term, the total and a verdict against ``gpu``).

    ``regime``: "full" (16 B/param), "pure_bf16" (8 B/param; not on a T4) or "lora" (frozen 2 B/param base plus
    adapters at 16 B/param). ``teacher``: a second, frozen model in fp16 plus its logits (logit KD / GKD)."""
    if regime == "pure_bf16" and not BF16_OK.get(gpu, True):
        raise ValueError(f"{gpu} has no bf16: use regime='full' (fp32 weights + fp16 autocast) or 'lora'")
    p = param_count(student).total
    rows = {}
    if regime == "lora":
        rows["student base (frozen, 16-bit)"] = p * BYTES_PER_PARAM["frozen"]
        rows["LoRA adapters + grads + AdamW"] = lora_params(student, r, target) * BYTES_PER_PARAM["full"]
    else:
        rows[f"student weights + grads + AdamW ({BYTES_PER_PARAM[regime]} B/param)"] = p * BYTES_PER_PARAM[regime]
    rows["activations" + (" (checkpointed)" if checkpointing else "")] = activation_bytes(student, batch, seq, checkpointing)
    rows["student logits (fp32 x3)"] = logits_bytes(student, batch, seq, 3, chunk)
    if teacher is not None:
        if _dims(teacher)["V"] != _dims(student)["V"]:
            raise ValueError("teacher and student vocab_size differ: logit KD needs a shared vocabulary")
        rows["teacher weights (frozen, 16-bit)"] = param_count(teacher).total * BYTES_PER_PARAM["frozen"]
        rows["teacher logits (fp32 x2)"] = logits_bytes(teacher, batch, seq, 2, chunk)
    total = sum(rows.values())
    usable = USABLE_GIB[gpu] * GIB
    return {"rows": [{"term": k, "GB": round(v / 1e9, 3)} for k, v in rows.items()], "total GB": round(total / 1e9, 3),
            "usable GB": round(usable / 1e9, 2), "fits": total <= usable, "headroom GB": round((usable - total) / 1e9, 2),
            "label": "PREDICTED (verify on the card)"}
