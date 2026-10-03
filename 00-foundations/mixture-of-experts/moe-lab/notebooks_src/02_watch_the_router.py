# %% [markdown]
# # 02 · Watch the router: per-token expert choices, hot experts, and what they do to EP ranks
#
# **Tier:** T1: a real MoE on one GPU. A free Colab/Kaggle T4 runs OLMoE-1B-7B in fp16 through transformers,
# or granite-3.0 MoE. Any GPU that runs `vllm serve --enable-return-routed-experts` is also an option. T0:
# bundled traces in the documented format of vLLM (**illustrative**: `tools/make_fixtures.py` generated them,
# and they are not a capture from OLMoE). T0 also uses real traces from the small MoE of notebook 01 when the
# environment has torch. The notebook selects the best source that it can find and labels it.
#
# ## The one-minute version
#
# * It is low-cost to capture the decisions of the router. There are two methods. The first is **forward
#   hooks** on the router modules of a Hugging Face MoE. Transformers v5 fuses the experts into 3D tensors.
#   Thus put the hook on the router, not on the experts. The second is the vLLM flag
#   `--enable-return-routed-experts`. It returns a base64 `.npy` of shape `(tokens − 1, layers, k)` with every
#   response.
# * **Utilisation is uneven**: in each layer, a few experts take a multiple of their fair share. The EPLB of
#   vLLM measures it as *balancedness* = mean load / max load. In a small sample, part of what you see is
#   noise. Compare the measured balancedness with what uniform routing shows at the same sample size.
# * **The hot set moves with the traffic**: code, math, prose and other languages make different experts
#   active. This effect is larger in some layers than in others.
# * Under expert parallelism, each GPU owns a contiguous block of experts. Thus hot experts that share a GPU
#   make that GPU the slowest. The more GPUs there are, the less the imbalance averages out.
#
# Concepts: PRIMER §3 "Routing and load balance" (balance at inference: hot experts and domain skew) and §6
# "Running MoE on GPUs" (EP, EPLB) ([`PRIMER.md`](../../PRIMER.md)).

# %%
import os
import numpy as np
from moelab import env, hooks

print(env.describe())
SOURCE = None
if env.server_url():                                          # T1: a vllm server with routed experts
    url, hdr = env.server_url(), env.auth_headers()
    mid = env.served_model(url, hdr)
    # experts per layer from the catalogue entry the served id names (else MOELAB_EXPERTS); k from the data
    ts = hooks.capture_vllm(url, mid, headers=hdr)
    SOURCE = f"MEASURED: vLLM at {url} ({mid})"
elif os.environ.get("MOELAB_HF_MODEL") and env.has_gpu() and env.has_transformers():   # T1: HF + hooks
    ts = hooks.capture_hf(os.environ["MOELAB_HF_MODEL"])
    SOURCE = f"MEASURED: transformers + forward hooks ({ts.model})"
else:                                                         # T0
    ts = hooks.load_fixture()
    SOURCE = "ILLUSTRATIVE: " + ts.label
print(SOURCE)
print(f"{ts.model}: {ts.n_experts} experts, top-{ts.top_k}; {len(ts.traces)} sequences, domains {ts.domains()}")

# %% [markdown]
# ## Worked example: one response, as vLLM returns it
#
# Start the server with `--enable-return-routed-experts`. Then every chat or completion choice carries
# `routed_experts`. With `"routed_experts_prompt_start": 0` in the request, it also covers the prompt. The
# last sampled token is not yet through the model. Thus there is one row fewer than there are tokens.

# %%
if SOURCE.startswith("ILLUSTRATIVE"):
    import json
    doc = json.loads((hooks.FIXTURES / "router_traces_olmoe.json").read_text())
    first = doc["responses"][0]["response"]
    B64 = first["choices"][0]["routed_experts"]
    usage = first["usage"]
    print("server:", doc["server"])
    print("choice keys:", sorted(first["choices"][0]), "| usage:", usage)
    print("routed_experts (base64 .npy):", B64[:60], "...", f"({len(B64):,} chars)")
else:
    B64 = hooks.encode_routed_experts(ts.traces[0].ids)
    usage = {"total_tokens": ts.traces[0].tokens + 1}

# %% [markdown]
# ## Exercise 2.1 — decode `routed_experts`
#
# Write `decode(b64)`. Change the base64 into bytes. Then do an `np.load` of the `.npy` in these bytes. vLLM
# encodes with `np.save(..., allow_pickle=False)`. Decode in the same way, and never with pickle.

# %% exercise
import base64
import io


def decode(b64):
    ### BEGIN SOLUTION
    return np.load(io.BytesIO(base64.b64decode(b64)), allow_pickle=False)
    ### END SOLUTION

# %% check
ids0 = decode(B64)
assert ids0.shape == (usage["total_tokens"] - 1, ts.traces[0].layers, ts.top_k), ids0.shape
assert (ids0 == ts.traces[0].ids).all()
probe = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
assert (decode(hooks.encode_routed_experts(probe)) == probe).all()
print(f"✅ {ids0.shape} = (tokens - 1, layers, top-k); dtype {ids0.dtype}")

# %% [markdown]
# ## Exercise 2.2 — utilisation per layer
#
# Write `utilisation(ids, n_experts)`. It returns counts `[layers, n_experts]`: for each layer, the number of
# token-slot assignments that went to each expert. The sum of every row is $\text{tokens} \times k$.

# %% exercise
def utilisation(ids, n_experts):
    ### BEGIN SOLUTION
    ids = np.asarray(ids)
    return np.stack([np.bincount(ids[:, l, :].ravel(), minlength=n_experts) for l in range(ids.shape[1])])
    ### END SOLUTION

# %% check
allids = ts.stacked()
U = utilisation(allids, ts.n_experts)
assert U.shape == (allids.shape[1], ts.n_experts) and (U.sum(1) == allids.shape[0] * ts.top_k).all()
assert (U == hooks.utilisation(allids, ts.n_experts)).all()
top = hooks.hot_experts(U, 3)
for l in (0, U.shape[0] // 2, U.shape[0] - 1):
    print(f"   layer {l:2d}: hottest {[(e, f'{s:.1%}') for e, s in top[l]]} (fair share {1 / ts.n_experts:.1%})")
print(f"✅ {allids.shape[0]} tokens x {ts.top_k} slots per layer; the hottest expert of a layer takes "
      f"{U.max(1).mean() / U.sum(1).mean():.1%} on average vs a fair {1 / ts.n_experts:.1%}")

# %% [markdown]
# The next cell shows the utilisation histogram of the last layer, with the busiest experts first. It shows
# the ten hottest experts and a summary of the rest:

# %%
print(hooks.text_histogram(U[-1], top=10))
print(f"... {ts.n_experts - 10} more experts share the remaining {1 - np.sort(U[-1])[-10:].sum() / U[-1].sum():.0%}")

# %% [markdown]
# ## Exercise 2.3 — is that skew real, or a small sample?
#
# The EPLB of vLLM logs **balancedness** = mean tokens per expert / max tokens per expert (1.0 is perfect). In
# a finite sample, even a perfectly uniform router shows a balancedness below 1. Write `balancedness(counts)`
# (for each row). Then write `uniform_baseline(tokens, n_experts, k, trials, seed)`. It returns the mean
# balancedness of `trials` samples of `tokens` tokens with uniform routing ($k$ *distinct* experts per token).
# If the balancedness of a layer is well below the baseline, the layer has a real skew.

# %% exercise
def balancedness(counts):
    ### BEGIN SOLUTION
    c = np.atleast_2d(np.asarray(counts, float))
    return c.mean(axis=1) / c.max(axis=1)
    ### END SOLUTION


def uniform_baseline(tokens, n_experts, k, trials=50, seed=0):
    ### BEGIN SOLUTION
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(trials):
        picks = np.argsort(rng.random((tokens, n_experts)), axis=1)[:, :k]      # k distinct, uniform
        vals.append(balancedness(np.bincount(picks.ravel(), minlength=n_experts))[0])
    return float(np.mean(vals))
    ### END SOLUTION

# %% check
assert np.allclose(balancedness([[2, 2, 2, 2], [4, 2, 1, 1]]), [1.0, 0.5])
assert np.allclose(balancedness(U), hooks.balancedness(U))
base = uniform_baseline(allids.shape[0], ts.n_experts, ts.top_k)
assert 0.5 < base < 1.0 and uniform_baseline(20_000, 64, 8, trials=5) > base      # more tokens, closer to 1
bal = balancedness(U)
print(f"   uniform routing at {allids.shape[0]} tokens: balancedness {base:.2f}; this trace: "
      f"{bal.min():.2f}-{bal.max():.2f} across layers")
if SOURCE.startswith("ILLUSTRATIVE"):
    assert bal.max() < base                    # the sample was generated with skew
print("✅ judge skew against the uniform baseline at the same sample size, not against 1.0")

# %% [markdown]
# ## Exercise 2.4 — does the hot set depend on the domain?
#
# Write `js(p, q)`: the Jensen–Shannon divergence in bits between two count vectors. Normalise each vector,
# then use $m = (p + q)/2$ and
# $\mathrm{JS} = \frac{1}{2}\,\mathrm{KL}(p \,\|\, m) + \frac{1}{2}\,\mathrm{KL}(q \,\|\, m)$. The value 0
# means the same distribution, and 1 means disjoint distributions. Then compare the domains layer by layer.

# %% exercise
def js(p, q):
    ### BEGIN SOLUTION
    p, q = np.asarray(p, float) / np.sum(p), np.asarray(q, float) / np.sum(q)
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    ### END SOLUTION

# %% check
assert np.isclose(js([1, 2, 3], [2, 4, 6]), 0.0) and np.isclose(js([1, 0], [0, 1]), 1.0)
doms = ts.domains()
per_dom = {d: utilisation(ts.stacked(d), ts.n_experts) for d in doms}
L = U.shape[0]
div = np.array([np.mean([js(per_dom[a][l], per_dom[b][l]) for i, a in enumerate(doms) for b in doms[i + 1:]])
                for l in range(L)])
assert np.allclose(div, hooks.domain_divergence(ts))
print("   mean JS divergence between domains, by layer:", " ".join(f"{v:.2f}" for v in div))
print(f"✅ first quarter of layers {div[:L // 4].mean():.3f}, last quarter {div[-(L // 4):].mean():.3f} "
      f"({'illustrative sample' if SOURCE.startswith('ILLUSTRATIVE') else 'your model'})")

# %% [markdown]
# Keep the sample size in mind when you read the result. Noise gives an upward bias to a JS divergence from a
# few hundred tokens per domain. Two samples of the *same* distribution rarely give 0. Real MoEs are different
# in how much routing depends on the domain, and in which layers. This is exactly the type of claim that you
# must measure on your own traffic before you build on it.
#
# ## Exercise 2.5 — from hot experts to a slow GPU
#
# With `--enable-expert-parallel` and the default `--expert-placement-strategy linear` of vLLM, EP rank $r$
# holds experts `[r·E/ep, (r+1)·E/ep)`. The alternative, `round_robin`, puts expert $e$ on rank
# $e \bmod \mathit{ep}$. vLLM uses it only for models with expert groups, like DeepSeek-V3. For other models,
# vLLM uses linear. We examined these two rules on vLLM main, Sep 2026. Make sure that they are true for your
# version.
#
# Write `rank_loads(counts, ep)`. It returns `[layers, ep]`, the assignments per rank under linear placement.
# Then calculate the **imbalance** max/mean for each layer.

# %% exercise
def rank_loads(counts, ep):
    ### BEGIN SOLUTION
    counts = np.atleast_2d(counts)
    return counts.reshape(counts.shape[0], ep, -1).sum(axis=2)
    ### END SOLUTION

# %% check
assert rank_loads([[1, 2, 3, 4]], 2).tolist() == [[3, 7]]
for ep in (2, 4, 8):
    assert (rank_loads(U, ep) == hooks.rank_loads(U, ep, "linear")).all()
imb = {ep: float((lambda r: (r.max(1) / r.mean(1)).mean())(rank_loads(U, ep))) for ep in (2, 4, 8, 16)}
print("   mean over layers of (busiest EP rank / mean rank):", {ep: round(v, 2) for ep, v in imb.items()})
assert imb[16] > imb[2]
print("✅ the same routing costs more as EP grows: fewer experts per GPU average less (EPLB replicates hot experts)")

# %% [markdown]
# EPLB (`--enable-eplb`, and at large scale optionally `--eplb-config '{"num_redundant_experts": 32}'`) places
# the experts again every `step_interval` steps. It uses the load that it observed over `window_size` steps
# (the v0.30.0 defaults are 3000 and 1000). It can also keep extra copies of hot experts. Each copy costs HBM
# on its rank (PRIMER §6.3: ~2.4 GiB per redundant expert per rank for DeepSeek-V3 in FP8).
#
# ## Worked example: the adapters behind `RouterRecorder`
#
# The router outputs are different for each family. `hooks.indices_from_output` knows the three layouts. The
# check in the next cell makes stand-in modules with the real class names and output layouts (T0: no
# downloads). Thus the same code works on the real models at T1.

# %%
if env.has_torch():
    import torch
    from torch import nn

    def make(name, layout):
        class R(nn.Module):
            def __init__(self):
                super().__init__()
                self.top_k, self.lin = 2, nn.Linear(8, 6, bias=False)

            def forward(self, x):
                logits = self.lin(x)
                idx = logits.topk(2, -1).indices
                w = logits.softmax(-1).gather(-1, idx)
                if layout == "hf":
                    return logits, w, idx                                   # OLMoE, Mixtral, Qwen*, gpt-oss, DeepSeek
                if layout == "granite":
                    return idx, w, logits                                   # GraniteMoeTopKRouter
                scores = torch.full_like(logits, float("-inf")).scatter(1, idx, logits.gather(1, idx)).sigmoid()
                return scores, logits                                       # Llama4Router
        R.__name__ = name
        return R()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([make("OlmoeTopKRouter", "hf"), make("GraniteMoeTopKRouter", "granite"),
                                         make("Llama4Router", "llama4")])

        def forward(self, x):
            return [r(x) for r in self.layers]

    torch.manual_seed(0)
    model, xin = Tiny(), torch.randn(5, 8)
    rec = hooks.RouterRecorder(model, top_k=2)
    model(xin)
    got = rec.pop()
    want = model.layers[0].lin(xin).topk(2, -1).indices.numpy()
    print("recorded", rec.names, got.shape)
    assert all((np.sort(got[:, l], 1) == np.sort(want if l == 0 else model.layers[l].lin(xin).topk(2, -1).indices.numpy(), 1)).all()
               for l in range(3))
    rec.remove()
    print("the three router layouts give the same [tokens, layers, k] array")
else:
    print("T0 without torch: RouterRecorder needs torch; the analysis above is numpy only")

# %% [markdown]
# ## On a real GPU (T1)
#
# **transformers + hooks**: a free T4 holds OLMoE-1B-7B in fp16 for short prompts. The granite-3.0 MoE models
# are smaller. The model ids are on the Hub (verify):
#
# ```bash
# pip install "transformers>=5" accelerate
# MOELAB_HF_MODEL=allenai/OLMoE-1B-7B-0924-Instruct jupyter lab notebooks/02_watch_the_router.ipynb
# ```
#
# **vLLM** (see [`deploy/any-gpu/`](../deploy/any-gpu/)):
#
# ```bash
# vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype half --max-model-len 4096 \
#     --cpu-offload-gb 3 --cpu-offload-params experts --enable-return-routed-experts   # T4: offload to fit
# MOELAB_URL=http://127.0.0.1:8000 jupyter lab ...
# ```
#
# The notebook finds the served model in `moelab.configs` (`hooks.expert_layout`) to get its expert count. It
# reads $k$ from the returned arrays. For a model that is not in the catalogue, set `MOELAB_EXPERTS` (and
# `MOELAB_TOPK`) from its `config.json`. If you do not, the capture stops with an error. It does not guess.
#
# Then write your own prompts for each domain (`hooks.capture_vllm(..., prompts={...})`). Use a few hundred
# tokens per domain at least. Use more tokens before you come to a conclusion (the baseline of Exercise 2.3
# tells you how many).

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We record the router. We do not guess it. Forward hooks on the router modules in
# transformers, or `--enable-return-routed-experts` in vLLM, give the experts of each token per layer. In each
# layer, the load is not even: the hottest expert takes a multiple of its fair share. Before we say that this
# is real, we compare it with what uniform routing shows at our sample size.
#
# "The hot set changes with the traffic mix. Thus it is possible that the placement that balances code traffic
# does not balance chat. For serving, this is important because of expert parallelism. GPU $r$ holds a block
# of experts, and the busiest GPU sets the step. The imbalance increases as the number of experts per GPU
# decreases.
#
# "The remedies are placement from the measured load and replicas of hot experts (EPLB), not topic-based
# assignment. Each replica costs HBM."
#
# **Drill 1.** *Balancedness is 0.6 on 500 tokens. Do we need EPLB?* First, calculate the uniform baseline at
# 500 tokens (Exercise 2.3). If it is also ~0.6, you measured noise. Collect more traffic.
#
# **Drill 2.** *Why put the hook on the router and not on the experts in transformers v5?* In each layer, the
# experts are one fused 3D tensor (`gate_up_proj [E, 2I, d]`), not per-expert modules. The output of the
# router already names the experts that each token uses.
#
# **Drill 3.** *EP=2 looked balanced. At EP=16, the step became slower than expected. Why?* Because EP=16 leaves
# 4 of the 64 experts on each GPU, one hot expert is now the main part of the load of its GPU. At EP=2,
# it averaged with 31 other experts.
