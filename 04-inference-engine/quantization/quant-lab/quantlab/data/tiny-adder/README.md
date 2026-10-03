# tiny-adder — the bundled checkpoint quant-lab quantizes

This is a 2-layer Llama-architecture model. It is sufficiently small to ship in the repo (599 KB of bf16
safetensors). Its training is sufficiently good that quantization damage shows as incorrect answers.

| | |
|---|---|
| Architecture | `LlamaForCausalLM` layout: RMSNorm, RoPE (theta 10,000), grouped-query attention (4 heads, 2 KV heads, head_dim 32), SwiGLU MLP, untied LM head |
| Shape | hidden 128, intermediate 256, 2 layers, vocabulary 16 (digits, `+`, `=`, `R`, `<bos>`, `<pad>`), 16 positions |
| Parameters | 299,648 (294,912 in the 14 projections that quantization targets) |
| Tasks | `add`: `457+389=` gives `6480` (the 4-digit sum, least-significant digit first). `reverse`: `R381204=` gives `402183`. |
| Accuracy (bf16) | 100% on 1,000 held-out problems per task (seeds 1000+, excluded from training) |
| Planted outliers | Hidden channels 13 and 41: RMSNorm weight ×24, and the projection input columns of the same channels ÷24. The function stays the same, with massive-activation inputs as in real LLMs. |
| Produced by | [`tools/train_tiny.py`](../../../tools/train_tiny.py) (torch, CPU, 6,000 AdamW steps, seeded), written with `quantlab.stio` |

The model is a tool for lessons, not a benchmark. It shows the mechanisms with real numbers: outlier
channels, the error compensation of GPTQ, the collapse of FP4 activations, KV scale saturation. But the
size of any accuracy drop on it says nothing about a production model. It has the MIT licence, with the
lab.
