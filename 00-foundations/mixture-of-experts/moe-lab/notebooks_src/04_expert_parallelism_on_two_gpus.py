# %% [markdown]
# # 04 · Expert parallelism on two GPUs: what each GPU holds, what crosses the link, who waits
#
# **Tier:** T2 — two GPUs: Kaggle's free "GPU T4 x2" (PCIe, no NVLink), a rented pair, or the
# `l4x2` pool of layer 02's GKE cluster (see [`deploy/gke/`](../deploy/gke/)). T0 — the same
# comparison from a per-GPU roofline and an alpha-beta link (**simulated**), and `vllm bench serve`
# results bundled in the documented format (**illustrative**, generated from that simulation — they
# exercise the parser, they prove nothing about GPUs).
#
# ## The one-minute version
#
# * vLLM has three ways to put an MoE on two GPUs. `--tensor-parallel-size 2` splits **every** expert
#   in half; `--tensor-parallel-size 2 --enable-expert-parallel` gives each GPU **half of the experts**,
#   whole; `--data-parallel-size 2 --enable-expert-parallel` also runs **attention data-parallel** —
#   each GPU its own requests and KV cache. EP size is not a flag: it is TP × DP.
# * What crosses the link: with DP = 1 the tokens are already on both GPUs after attention, so vLLM
#   runs **no all-to-all** — each GPU applies its experts and one all-reduce combines (as in TP). With
#   DP = 2 the default `--all2all-backend allgather_reducescatter` gathers the tokens and scatters the
#   results. The `tokens × k × hidden × bytes` dispatch/combine of layer 02 §5.6 is what dedicated
#   all-to-all kernels (DeepEP: SM90+, NVLink + RDMA) move at scale — not on T4s.
# * Decode messages are small (32 tokens × 2,048 × 2 B = 128 KiB), so on PCIe they are
#   **latency-bound**: the count of collectives per layer matters more than their bytes.
# * Under EP the step ends when the **slowest GPU** finishes. In memory-bound decode a GPU waits for
#   the bytes of its *touched* experts; in compute-bound steps (prefill, huge batches) for the
#   *tokens* its experts got — that is where hot experts hurt, more so as EP grows. TP is balanced by
#   construction but multiplies half-size expert GEMMs.
#
# Concepts: PRIMER §6 "Running MoE on GPUs" (TP vs EP for experts, the hybrid, DP attention + EP) and
# §8 "In a design review: failure modes" (EP across a slow fabric) ([`PRIMER.md`](../../PRIMER.md)).
# Collective costs: [layer 02 PRIMER §5](../../../../02-cuda-nccl-runtime/cuda-and-nccl/PRIMER.md);
# EP in the engine: [layer 04 PRIMER §9](../../../../04-inference-engine/serving-engine/PRIMER.md);
# wide-EP across nodes: [layer 05 PRIMER §8](../../../../05-orchestrator/serving-orchestration/PRIMER.md).

# %%
import subprocess
from pathlib import Path
import numpy as np
from moelab import configs, env, ep, hooks, offload

print(env.describe())
MOE, T4, L4 = configs.get("olmoe-1b-7b"), configs.gpu("T4"), configs.gpu("L4")
LINK = ep.LINKS["pcie-2xT4"]
print(f"{MOE.name}: {MOE.n_experts} experts, top-{MOE.top_k}, hidden {MOE.d_model}, {MOE.layers} layers | "
      f"link {LINK.name}: alpha {LINK.alpha_s * 1e6:.0f} us, {LINK.bw_gbs:g} GB/s ({LINK.note})")
for lay in ep.LAYOUTS:
    tp = 2 if lay != "dp_ep" else 1
    dp = 2 if lay == "dp_ep" else 1
    print(f"  {lay:6s} vllm serve {MOE.hf_id} {' '.join(ep.vllm_flags(lay))}   -> EP size {ep.ep_size(tp, dp, 'ep' in lay)}")

# %% [markdown]
# ## Exercise 4.1 — bytes an all-to-all moves
#
# Write `a2a_bytes(tokens, k, hidden, elem_bytes=2, scale_block=0, scale_bytes=4)`: one direction
# (dispatch or combine), per GPU, when every assignment is remote: tokens × k × (hidden × elem_bytes
# + one scale per `scale_block` channels when the activations are FP8). Check it against layer 02's
# worked number and against DeepEP's published low-latency benchmark (128 tokens, hidden 7,168,
# top-8, FP8 dispatch with 4-byte scales per 128 channels, BF16 combine; EP8 on H800: 77 µs dispatch,
# 114 µs combine — `deepep/docs/legacy.md`).

# %% exercise
def a2a_bytes(tokens, k, hidden, elem_bytes=2, scale_block=0, scale_bytes=4):
    ### BEGIN SOLUTION
    per = hidden * elem_bytes + (hidden / scale_block * scale_bytes if scale_block else 0)
    return tokens * k * per
    ### END SOLUTION

# %% check
assert a2a_bytes(256, 2, 4096) == 4 * 2**20                                          # layer 02 §5.6: 4 MiB
disp, comb = a2a_bytes(128, 8, 7168, 1, 128), a2a_bytes(128, 8, 7168, 2)
assert disp == 7_569_408 and comb == 14_680_064
assert round(disp / 77e-6 / 1e9, 1) == 98.3                                           # DeepEP reports 98 GB/s
for t in (1, 32, 256):
    assert a2a_bytes(t, MOE.top_k, MOE.d_model) == ep.a2a_bytes(t, MOE.top_k, MOE.d_model)
print(f"✅ DeepEP EP8 dispatch {disp:,} B / 77 us = {disp / 77e-6 / 1e9:.1f} GB/s, combine {comb / 114e-6 / 1e9:.1f} GB/s; "
      f"OLMoE decode at 32 tokens/GPU would dispatch {a2a_bytes(32, 8, 2048) / 2**20:.1f} MiB per layer")

# %% [markdown]
# ## Exercise 4.2 — latency-bound or bandwidth-bound?
#
# The alpha-beta model of layer 02: a collective over p ranks and an S-byte buffer costs
# `a · alpha + c · S / B` with (ring all-reduce) a = 2(p − 1), c = 2(p − 1)/p; (ring all-gather,
# ring reduce-scatter, pairwise all-to-all) a = p − 1, c = (p − 1)/p; (direct all-to-all) a = 1,
# c = (p − 1)/p. Write `ab_time(op, size, p, alpha, bw_gbs, algo="ring")`.

# %% exercise
def ab_time(op, size, p, alpha, bw_gbs, algo="ring"):
    ### BEGIN SOLUTION
    f = (p - 1) / p
    a, c = {("all_reduce", "ring"): (2 * (p - 1), 2 * f), ("all_gather", "ring"): (p - 1, f),
            ("reduce_scatter", "ring"): (p - 1, f), ("all_to_all", "pairwise"): (p - 1, f),
            ("all_to_all", "direct"): (1, f)}[(op, algo)]
    return a * alpha + c * size / (bw_gbs * 1e9)
    ### END SOLUTION

# %% check
four = 4 * 2**20
assert round(ab_time("all_to_all", four, 8, 2e-6, 450, "pairwise") * 1e6) == 22       # layer 02 §5.6: 22 us
assert round(ab_time("all_to_all", four, 8, 2e-6, 450, "direct") * 1e6) == 10        # and 10 us
for op, algo in (("all_reduce", "ring"), ("all_gather", "ring"), ("all_to_all", "direct")):
    assert np.isclose(ab_time(op, 12345, 2, 20e-6, 8, algo), ep.collective_time(op, 12345, 2, LINK, algo))
S = 32 * MOE.d_model * 2
t = ab_time("all_reduce", S, 2, LINK.alpha_s, LINK.bw_gbs)
print(f"✅ decode, 32 tokens: one all-reduce of {S // 1024} KiB on 2 x T4 = {t * 1e6:.0f} us with the assumed link, "
      f"of which {2 * LINK.alpha_s / t:.0%} is latency (alpha) — count collectives, not bytes")

# %% [markdown]
# ## Worked example: what each GPU holds and what it sends, per layout
#
# Per MoE layer: TP and TP+EP all-reduce after attention and after the MoE (2 collectives); DP+EP
# all-gathers the tokens before the experts and reduce-scatters after (2 collectives, no attention
# collective, but each GPU streams the full attention weights for its own requests).

# %%
print(f"{'layout':7s} {'weights/GPU':>12s}   collectives per layer (decode, 32 tokens)     comm per step [simulated]")
for lay in ep.LAYOUTS:
    w = ep.per_gpu_weight_bytes(MOE, lay) / configs.GiB
    per_layer = ep.moe_layer_comm(lay, 32, MOE.d_model, LINK)
    what = "2 x all-reduce" if lay != "dp_ep" else "all-gather + reduce-scatter"
    print(f"{lay:7s} {w:9.2f} GiB   {what:30s} {per_layer * 1e6:7.0f} us/layer   {MOE.layers * per_layer * 1e3:5.2f} ms")

# %% [markdown]
# ## Exercise 4.3 — does it fit two T4s, and with how much KV?
#
# Write `per_gpu_gib(model, layout, wb=2)`: TP and TP+EP shard attention, shared expert, the experts
# and the vocabulary (embedding and LM head) over the 2 GPUs and replicate the router; DP+EP
# replicates everything but the experts, which it splits. Use `model.attn_params()`,
# `shared_params()`, `router_params()` (per layer), `expert_param_total()`, `embedding_params()`.
# Then check which models fit two T4s with a 4K-token KV cache left over.

# %% exercise
def per_gpu_gib(model, layout, wb=2):
    ### BEGIN SOLUTION
    L = model.layers
    attn, shared, router = L * model.attn_params(), L * model.shared_params(), L * model.router_params()
    experts, emb = model.expert_param_total(), model.embedding_params()
    if layout in ("tp", "tp_ep"):
        per = (attn + shared + emb + experts) / 2 + router
    else:
        per = attn + shared + emb + router + experts / 2
    return per * wb / configs.GiB
    ### END SOLUTION

# %% check
budget = offload.DEFAULT_UTIL * T4.memory_gib - offload.DEFAULT_OVERHEAD_GIB          # per GPU, as in notebook 05
rows = {}
for key in ("granite-3b-a800m", "olmoe-1b-7b", "qwen1.5-moe-a2.7b"):
    m = configs.get(key)
    for lay in ep.LAYOUTS:
        assert np.isclose(per_gpu_gib(m, lay), ep.per_gpu_weight_bytes(m, lay) / configs.GiB)
    rows[key] = {lay: budget - per_gpu_gib(m, lay) for lay in ep.LAYOUTS}
    print(f"   {m.name:22s} KV room per T4: " + "  ".join(f"{lay} {v:5.2f} GiB" for lay, v in rows[key].items()))
kv4k = 4096 * configs.get("olmoe-1b-7b").kv_bytes_per_token() / configs.GiB
assert all(v > kv4k for v in rows["olmoe-1b-7b"].values())             # OLMoE fp16 fits 2 x T4 in every layout
assert all(v < 0 for v in rows["qwen1.5-moe-a2.7b"].values())          # Qwen1.5-MoE fp16 needs 24 GB cards (2 x L4)
print(f"✅ OLMoE fp16: ~{per_gpu_gib(MOE, 'tp'):.1f} GiB per T4 under TP, leaving room for "
      f"{rows['olmoe-1b-7b']['tp'] / kv4k:.0f} x 4K tokens; Qwen1.5-MoE (26.7 GiB) needs the l4x2 pool or INT4")

# %% [markdown]
# ## Worked example: the three layouts, simulated
#
# `ep.ep_decode_step` prices each GPU's step on the roofline with `SimParams` efficiencies (the same
# assumptions as notebook 03), takes the slowest GPU, and adds the link time. Uniform routing first,
# then the routing of the illustrative traces from notebook 02, where some experts are hot.

# %%
rows = ep.compare_layouts(MOE, T4, LINK, batches=(1, 8, 32, 128), context=512)
print(ep.format_rows(rows))
ts = hooks.load_fixture()
skewed = ts.stacked()                                      # [tokens, layers, k] from the illustrative traces
print("\nwith the illustrative (skewed) routing of notebook 02:")
for b in (8, 32, 128):
    st = ep.ep_decode_step(MOE, T4, "tp_ep", b, 512, LINK, ids=skewed)
    un = ep.ep_decode_step(MOE, T4, "tp_ep", b, 512, LINK)
    print(f"   batch {b:4d}: tp_ep step {st.time * 1e3:6.2f} ms (uniform {un.time * 1e3:6.2f} ms), "
          f"GPU imbalance {st.imbalance:.2f} (uniform {un.imbalance:.2f})")

# %% [markdown]
# Two GPUs over PCIe, a small MoE: from batch 8 up the three layouts land within ~10% of each other
# in this model, because each GPU mostly waits on its own HBM. Batch 1 separates them: TP splits each
# of the k touched experts over both GPUs, while under EP the k experts of a layer rarely split evenly
# (the per-layer imbalance of 1.25), and DP+EP also streams the full attention weights on both GPUs
# for one request. What this roofline leaves out may matter more: TP halves each expert's GEMM
# (N = I/2 per GPU), and small GEMMs run below the roofline — one reason EP wins at scale. Note what
# the skewed routing did: decode got slightly *faster* (fewer distinct experts touched, fewer bytes —
# notebook 03's Exercise 3.2) and the GPUs stayed nearly balanced. A measurement decides;
# Exercise 4.4 shows where skew does hurt.
#
# ## Exercise 4.4 — when do hot experts slow a GPU?
#
# A GPU's share of a layer depends on its experts in two ways: the **bytes** of its experts that are
# touched (what a memory-bound decode step waits for) and the **tokens** routed to them (the FLOPs a
# compute-bound step — prefill, or a very large decode batch — waits for). Every layer ends in a
# collective, so the GPUs meet once per layer and the step pays the busiest GPU *of each layer*.
# For routing `ids [tokens, layers, k]` and linear placement (GPU r holds experts
# `[r·E/ep, (r+1)·E/ep)`), write `ep_loads(ids, n_experts, ep)` → `(touched, tokens)`, two
# `[layers, ep]` arrays, and `imbalance(v)` = Σ over layers of the busiest GPU / Σ of the mean GPU.

# %% exercise
def ep_loads(ids, n_experts, ep_size):
    ### BEGIN SOLUTION
    ids = np.asarray(ids)
    home = np.arange(n_experts) // (n_experts // ep_size)
    touched, tokens = [], []
    for l in range(ids.shape[1]):
        counts = np.bincount(ids[:, l, :].ravel(), minlength=n_experts)
        touched.append(np.bincount(home, weights=counts > 0, minlength=ep_size))
        tokens.append(np.bincount(home, weights=counts, minlength=ep_size))
    return np.array(touched), np.array(tokens)
    ### END SOLUTION


def imbalance(v):
    ### BEGIN SOLUTION
    v = np.asarray(v, float)
    return float(v.max(axis=1).sum() / v.mean(axis=1).sum())
    ### END SOLUTION

# %% check
toy = np.array([[[0, 1]], [[0, 2]]])                                   # 2 tokens, 1 layer, top-2, 4 experts
t_, a_ = ep_loads(toy, 4, 2)
assert t_.tolist() == [[2, 1]] and a_.tolist() == [[3, 1]]             # experts 0, 1 on GPU 0; 2 on GPU 1
assert np.isclose(imbalance([[3, 1], [2, 2]]), (3 + 2) / (2 + 2))
assert np.allclose(ep_loads(skewed, 64, 8)[1], hooks.rank_loads(hooks.utilisation(skewed, 64), 8))
prefill = np.concatenate([skewed] * 4)[:2048]                           # a 2,048-token prefill chunk
print("   busiest GPU / mean GPU, per layer     bytes (touched experts)      tokens (FLOPs)")
res = {}
for name, ids_u, ids_s in (("decode, 32 tokens", ep.routes(MOE, 32), skewed[:32]),
                           ("prefill, 2,048 tokens", ep.routes(MOE, 2048), prefill)):
    for epn in (2, 8):
        (tu, au), (tsk, ask) = ep_loads(ids_u, 64, epn), ep_loads(ids_s, 64, epn)
        res[(name, epn)] = (imbalance(tu), imbalance(tsk), imbalance(au), imbalance(ask))
        print(f"   {name:22s} EP={epn}      uniform {imbalance(tu):4.2f} skewed {imbalance(tsk):4.2f}    "
              f"uniform {imbalance(au):4.2f} skewed {imbalance(ask):4.2f}")
u_b, s_b, u_t, s_t = res[("prefill, 2,048 tokens", 8)]
assert s_b < 1.02 and s_t > u_t + 0.05                     # every expert touched; tokens pile onto a hot GPU
assert res[("prefill, 2,048 tokens", 8)][3] > res[("prefill, 2,048 tokens", 2)][3]
sk = ep_loads(prefill, 64, 8)[1] * 2 * MOE.expert_params() / (T4.tflops_16 * 1e12 * 0.5)
print(f"✅ a compute-bound 2,048-token chunk at EP=8 spends {sk.max(1).sum() * 1e3:.1f} ms in experts where a "
      f"balanced one would spend {sk.mean(1).sum() * 1e3:.1f} ms [simulated]; decode bytes stay balanced")

# %% [markdown]
# So EPLB and redundant experts pay where tokens per expert are large — prefill and big decode
# batches at wide EP — and matter little for a two-GPU decode server. At EP=2 even the tokens
# average out over 32 experts per GPU.
#
# %% [markdown]
# ## Exercise 4.5 — read what `vllm bench serve` prints
#
# The benchmark ends with a `Serving Benchmark Result` block of `"{label:<40} {value}"` lines.
# Write `parse(text)` → `{key: number}` with keys like `mean_itl_ms` and
# `output_token_throughput_tok_s` (lower-case the label, drop the trailing colon, turn every run of
# non-alphanumerics into `_`). Then tabulate the bundled runs (TP, TP+EP, DP+EP at concurrency 1
# and 16).

# %% exercise
import re


def parse(text):
    ### BEGIN SOLUTION
    out, inside = {}, False
    for line in text.splitlines():
        if "Serving Benchmark Result" in line:
            inside = True
            continue
        if inside and line.startswith("=" * 50):
            break
        m = re.match(r"^(.+?):\s+(-?[\d.]+)\s*$", line) if inside else None
        if m:
            out[re.sub(r"[^a-z0-9]+", "_", m.group(1).lower()).strip("_")] = float(m.group(2))
    return out
    ### END SOLUTION

# %% check
FIX = Path(hooks.FIXTURES)
res = {(lay, c): parse((FIX / f"bench_serve_olmoe_2xT4_{lay}_c{c}.txt").read_text()) for lay in ep.LAYOUTS for c in (1, 16)}
one = res[("tp_ep", 16)]
assert {"mean_itl_ms", "median_ttft_ms", "p99_tpot_ms", "output_token_throughput_tok_s", "successful_requests"} <= set(one)
assert all(np.isclose(v, ep.parse_bench_serve((FIX / "bench_serve_olmoe_2xT4_tp_ep_c16.txt").read_text())[k]) for k, v in one.items())
print("   ILLUSTRATIVE (generated from this notebook's simulation; replace with your T2 run)")
print("   layout  concurrency  mean ITL (ms)  output tok/s")
for (lay, c), r in res.items():
    print(f"   {lay:6s}  {c:11d}  {r['mean_itl_ms']:13.2f}  {r['output_token_throughput_tok_s']:12.1f}")
print("✅ parsed; with a real run this table is the measurement that decides between the layouts")

# %% [markdown]
# ## On two GPUs (T2): run the three layouts and benchmark each
#
# Kaggle: Settings → Accelerator → **GPU T4 x2** (not P100). Kaggle has no Colab bootstrap, so make the
# first cell of an imported copy of this notebook the checkout from
# [`deploy/any-gpu/`](../deploy/any-gpu/README.md#colab-or-kaggle-free-t4)
# (`git clone`, `%cd .../moe-lab`, `pip install -e .`). One server at a time (each takes both
# GPUs); OLMoE in fp16 fits two T4s in every layout (Exercise 4.3). The same flags work on the
# `l4x2` GKE pool ([`deploy/gke/`](../deploy/gke/)) and on a rented pair.
#
# ```bash
# pip install "vllm==0.30.0"
# vllm serve allenai/OLMoE-1B-7B-0924-Instruct --dtype half --max-model-len 4096 --tensor-parallel-size 2 &
# vllm bench serve --base-url http://127.0.0.1:8000 --model allenai/OLMoE-1B-7B-0924-Instruct \
#     --dataset-name random --random-input-len 256 --random-output-len 128 --num-prompts 128 \
#     --max-concurrency 16 --ignore-eos | tee tp_c16.txt
# # then --tensor-parallel-size 2 --enable-expert-parallel, and --data-parallel-size 2 --enable-expert-parallel
# ```
#
# Or let the cell below do it (`MOELAB_START_VLLM=1` on a machine with two GPUs); it saves each
# result next to the notebook and parses it with your `parse`. With DP=2, vLLM keeps the two ranks in
# lockstep (an idle rank runs dummy forward passes), and DeepEP-style backends need SM90 — the
# default `allgather_reducescatter` is what runs here.

# %%
if env.gpu_count() >= 2 and env.may_start_vllm():
    flags = ["--max-model-len", "4096"] + (["--dtype", "half"] if (env.compute_capability() or 0) < 8.0 else [])
    for lay in ep.LAYOUTS:
        with env.VLLMServer(MOE.hf_id, flags + ep.vllm_flags(lay), log_path=f"vllm-{lay}.log") as url:
            for c in (1, 16):
                out = subprocess.run(ep.bench_command(MOE.hf_id, c, base_url=url), capture_output=True, text=True)
                Path(f"bench_{lay}_c{c}.txt").write_text(out.stdout)
                r = parse(out.stdout)
                print(f"MEASURED {lay:6s} c={c:2d}: mean ITL {r.get('mean_itl_ms', float('nan')):.2f} ms, "
                      f"{r.get('output_token_throughput_tok_s', float('nan')):.1f} tok/s")
else:
    print(f"T{2 if env.gpu_count() >= 2 else 0}: not starting vLLM (needs 2 GPUs, vLLM and MOELAB_START_VLLM=1). Commands:")
    for lay in ep.LAYOUTS:
        print("  vllm serve", MOE.hf_id, "--dtype half --max-model-len 4096", " ".join(ep.vllm_flags(lay)))
    print("  " + " ".join(ep.bench_command(MOE.hf_id, 16)))

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "On two GPUs vLLM gives us three layouts. TP splits every expert in half; EP gives
# each GPU half the experts whole; DP+EP also makes attention data-parallel. With DP = 1 there is no
# all-to-all at all — tokens are already on both GPUs, and an all-reduce combines the experts'
# outputs — so on a PCIe pair the choice is less about bytes than about the number of latency-bound
# collectives per layer and about balance: under EP the slowest GPU sets the step — in decode the one
# with the most touched experts, in prefill the one whose experts got the most tokens, which is where
# hot experts hurt and where it worsens as experts per GPU shrink. For a small MoE on two T4s the layouts are close in
# our model; we would run all three with `vllm bench serve` and keep the fastest at our concurrency.
# EP earns its keep at scale — many GPUs, DP attention, DeepEP over NVLink and RDMA — where it lets
# each expert see the tokens of the whole cluster."
#
# **Drill 1.** *Where are the all-to-alls when we run `--tensor-parallel-size 2 --enable-expert-parallel`?*
# — There are none: with DP = 1 every GPU already has every token; each applies its local experts and
# an all-reduce sums the outputs. All-to-all kernels are used when DP > 1 (vLLM `use_all2all_kernels`).
#
# **Drill 2.** *Should we run EP across two nodes over 100 Gb/s Ethernet?* — Each layer adds collectives
# whose latency and bytes now cross the slow link, twice per layer, per step; keep EP (and TP) inside
# the NVLink domain and replicate across nodes, unless the model does not fit one node (layer 02 §5.6).
#
# **Drill 3.** *EP=2 decode is slower than TP=2 for a single user. Is EP broken?* — At batch 1 only k
# experts per layer are touched and they may sit mostly on one GPU; TP splits each of them over both.
# EP pays off with batch (every expert busy) and balance, not at batch 1.
