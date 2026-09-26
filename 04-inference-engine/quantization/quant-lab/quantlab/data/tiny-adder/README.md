# tiny-adder — the bundled checkpoint quant-lab quantizes

A 2-layer Llama-architecture model small enough to ship in the repo (599 KB of bf16 safetensors) and
trained well enough that quantization damage shows up as wrong answers.

| | |
|---|---|
| Architecture | `LlamaForCausalLM` layout: RMSNorm, RoPE (theta 10,000), grouped-query attention (4 heads, 2 KV heads, head_dim 32), SwiGLU MLP, untied LM head |
| Shape | hidden 128, intermediate 256, 2 layers, vocabulary 16 (digits, `+`, `=`, `R`, `<bos>`, `<pad>`), 16 positions |
| Parameters | 299,648 (294,912 in the 14 projections that quantization targets) |
| Tasks | `add`: `457+389=` → `6480` (the 4-digit sum, least-significant digit first); `reverse`: `R381204=` → `402183` |
| Accuracy (bf16) | 100% on 1,000 held-out problems per task (seeds 1000+, excluded from training) |
| Planted outliers | hidden channels 13 and 41: RMSNorm weight ×24 and the matching projection input columns ÷24 — the same function, with massive-activation inputs like real LLMs |
| Produced by | [`tools/train_tiny.py`](../../../tools/train_tiny.py) (torch, CPU, 6,000 AdamW steps, seeded), written with `quantlab.stio` |

The model is a teaching device, not a benchmark: it exhibits the mechanisms (outlier channels,
GPTQ's error compensation, FP4 activations collapsing, KV scale saturation) with real numbers, but the
size of any accuracy drop on it says nothing about a production model. MIT licensed with the lab.
