#!/usr/bin/env python3
"""Train the bundled tiny checkpoint (``quantlab/data/tiny-adder/``). Needs torch (CPU is enough: ~25 min).

    python3 tools/train_tiny.py                 # train, plant the outlier channels, write bf16 safetensors
    python3 tools/train_tiny.py --check         # only evaluate the bundled checkpoint with the numpy model

What it does, in order:
1. trains a 2-layer Llama (the architecture of ``quantlab.tinymodel``, same tensor names) on the
   ``add`` and ``reverse`` tasks, loss on answer tokens only, problems drawn from seeds < 1000 with
   every held-out evaluation problem (seeds 1000-1009) removed;
2. plants two massive-activation channels with a function-preserving rescaling: for each chosen
   hidden channel j and each RMSNorm that feeds projections (``input_layernorm`` -> q/k/v,
   ``post_attention_layernorm`` -> gate/up), multiply the norm weight by ``c`` and divide column j of
   the consuming weights by ``c``. The product is unchanged; the projections' inputs now carry an
   outlier channel ``c`` times larger than the rest, as real LLM activations do;
3. rounds to bfloat16, writes ``config.json`` + ``model.safetensors`` with ``quantlab.stio``, and
   checks the numpy forward pass against torch and the task accuracy.
The run is seeded; the bundled file is the output of this script (torch 2.14, CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab import stio, tinymodel as tm  # noqa: E402

CONFIG = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 128, "intermediate_size": 256,
    "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "vocab_size": 16, "max_position_embeddings": 16, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "hidden_act": "silu",
    "bos_token_id": tm.BOS, "pad_token_id": tm.PAD,
    "_note": "quantlab tiny-adder: a toy checkpoint trained by tools/train_tiny.py (add + reverse tasks)",
}
OUTLIER_CHANNELS = 2        # how many hidden channels get massive activations
OUTLIER_SCALE = 24.0        # c: their RMSNorm weight is multiplied by this, their weight columns divided


def build_torch_model(torch):
    nn = torch.nn
    c = CONFIG
    H, KV, D, E = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["hidden_size"]

    class RMS(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(E))

        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + c["rms_norm_eps"]) * self.weight

    def rope(x):
        T = x.shape[-2]
        inv = 1.0 / c["rope_theta"] ** (torch.arange(0, D, 2, dtype=torch.float64) / D)
        ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None]
        cos, sin = torch.cat([ang, ang], -1).cos().float(), torch.cat([ang, ang], -1).sin().float()
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(E, H * D, bias=False)
            self.k_proj = nn.Linear(E, KV * D, bias=False)
            self.v_proj = nn.Linear(E, KV * D, bias=False)
            self.o_proj = nn.Linear(H * D, E, bias=False)

        def forward(self, x):
            B, T, _ = x.shape
            q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
            k = self.k_proj(x).view(B, T, KV, D).transpose(1, 2)
            v = self.v_proj(x).view(B, T, KV, D).transpose(1, 2)
            q, k = rope(q), rope(k)
            k, v = k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
            a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o_proj(a.transpose(1, 2).reshape(B, T, H * D))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(E, c["intermediate_size"], bias=False)
            self.up_proj = nn.Linear(E, c["intermediate_size"], bias=False)
            self.down_proj = nn.Linear(c["intermediate_size"], E, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm, self.self_attn = RMS(), Attn()
            self.post_attention_layernorm, self.mlp = RMS(), MLP()

        def forward(self, h):
            h = h + self.self_attn(self.input_layernorm(h))
            return h + self.mlp(self.post_attention_layernorm(h))

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(c["vocab_size"], E)
            self.layers = nn.ModuleList([Layer() for _ in range(c["num_hidden_layers"])])
            self.norm = RMS()

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(E, c["vocab_size"], bias=False)

        def forward(self, ids):
            h = self.model.embed_tokens(ids)
            for layer in self.model.layers:
                h = layer(h)
            return self.lm_head(self.model.norm(h))

    return LM()


def held_out() -> set:
    keys = set()
    for task in tm.TASKS:
        for seed in range(1000, 1010):
            p, _ = tm.make_task(task, 2000, seed)
            keys |= {tuple(r) for r in p}
    return keys


def batch(task, n, step, exclude):
    p, a = tm.make_task(task, n * 2, seed=step % 1000)
    keep = [i for i in range(len(p)) if tuple(p[i]) not in exclude][:n]
    return p[keep], a[keep]


def train(steps: int = 6000, bs: int = 128, lr: float = 3e-3):
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(2)
    model = build_torch_model(torch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    exclude = held_out()
    for step in range(steps):
        loss = 0.0
        for task in tm.TASKS:
            p, a = batch(task, bs // 2, step * 7 + (0 if task == "add" else 3), exclude)
            seq = torch.from_numpy(np.concatenate([p, a], 1)).long()
            logits = model(seq[:, :-1])[:, p.shape[1] - 1:]
            loss = loss + torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                            seq[:, p.shape[1]:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"step {step:5d}  loss {loss.item() / 2:.4f}", flush=True)
    return model


def plant_outliers(weights: dict, model_np: tm.TinyLM) -> list:
    """Pick the hidden channels with the largest mean |input| to layer 0's projections and scale them."""
    cap = {}
    for task in tm.TASKS:
        p, a = tm.make_task(task, 256, seed=7)
        model_np.answer_logits(p, a, capture=cap)
    x = np.concatenate(cap["model.layers.0.self_attn.q_proj"])
    chans = sorted(int(j) for j in np.argsort(-np.abs(x).mean(0))[:OUTLIER_CHANNELS])
    for i in range(CONFIG["num_hidden_layers"]):
        p = f"model.layers.{i}."
        for norm, consumers in (("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
                                ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"))):
            for j in chans:
                weights[p + norm + ".weight"][j] *= OUTLIER_SCALE
                for cons in consumers:
                    weights[p + cons + ".weight"][:, j] /= OUTLIER_SCALE
    return chans


def check(path=tm.TINY_DIR):
    m = tm.load(path)
    print(f"{m.num_params():,} parameters; outlier channels {tm.outlier_channels(m)}")
    for task in tm.TASKS:
        print(f"  {task:8s} accuracy {m.accuracy(task, 1000):.3f} (1,000 held-out problems)")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args(argv)
    if args.check:
        check()
        return 0
    import torch
    model = train(args.steps)
    w = {k: v.detach().double().numpy().copy() for k, v in model.state_dict().items()}
    m = tm.TinyLM(CONFIG, w)
    ids = torch.from_numpy(tm.make_task("add", 8, 1000)[0]).long()
    ref = model(ids).detach().double().numpy()
    assert np.allclose(m.forward(ids.numpy()), ref, atol=1e-3), "numpy forward disagrees with torch"
    before = {t: m.accuracy(t, 1000) for t in tm.TASKS}
    chans = plant_outliers(w, m)
    after = tm.TinyLM(CONFIG, w)
    print("accuracy before/after planting outliers:",
          {t: (before[t], after.accuracy(t, 1000)) for t in tm.TASKS}, "channels", chans)
    tm.TINY_DIR.mkdir(parents=True, exist_ok=True)
    (tm.TINY_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")
    stio.save({k: stio.bf16(v) for k, v in w.items()}, tm.TINY_DIR / "model.safetensors",
              metadata={"format": "pt", "producer": "quantlab tools/train_tiny.py"})
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
