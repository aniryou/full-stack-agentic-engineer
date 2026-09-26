# moe-core — build a mixture-of-experts layer in numpy and predict what it costs to serve

After this you can route tokens through an MoE layer by hand, watch a router collapse and fix it, predict how many
experts a decode batch reads and when it turns compute-bound, price the expert-parallel all-to-alls, and size a
deployment — with `moecore`, a standard-library + numpy package small enough to read in a sitting (~1,100 lines).

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) "The one-minute version", then §1–§2 (why sparsity, the MoE layer).
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 75 tests in about 30 s, including "the
   sparse forward equals every expert on every token" and "layer 01's MoE table, reproduced to the byte".
3. Open [`notebooks/01_the_moe_layer.ipynb`](notebooks/01_the_moe_layer.ipynb) and route six tokens.

The fastest win, from this directory:

```python
from moecore import touched
from moecore.sizing import MODELS
mix, h200 = MODELS["mixtral-8x7b"], touched.DEVICES["h200"]
print(mix.total(), mix.active())                                   # 46702526464 12879659008
for b in (1, 16, 64):
    s = touched.decode_step(mix, h200, b, 1024)
    print(b, round(touched.experts_touched(8, 2, b), 2), f"{s.bytes / 1e9:.1f} GB {s.time * 1e3:.2f} ms")
# 1 2.0 25.6 GB 5.34 ms   |   16 7.92 94.4 GB 19.66 ms   |   64 8.0 101.7 GB 21.20 ms   (simulated roofline)
print(touched.decode_crossover_batch(mix, h200))                   # 754: decode turns compute-bound
```

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU and no network.* Each notebook opens with
"The one-minute version", works examples against the code, sets exercises with a check cell that prints ✅, and ends
with "In a design review". Finished versions are in [`solutions/`](solutions/). About 7 hours with the primer.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_the_moe_layer`](notebooks/01_the_moe_layer.ipynb) | route tokens through the real routers (Mixtral, OLMoE and Qwen3's `norm_topk_prob` settings, DeepSeek-V3, gpt-oss, Llama 4) and say why their weights differ; write the sparse forward pass; count total and active parameters from a config and name the "active" convention a published number uses; argue for fine-grained experts | §1, §2 | ~1.5 h | T0 |
| [`02_routing_and_load_balance`](notebooks/02_routing_and_load_balance.ipynb) | compute the Switch loss in both normalisations, the z-loss, capacity and dropping, dropless padding; balance a router with DeepSeek's bias; watch a tiny MoE collapse and recover (hand-written gradients) | §3, §4 | ~1.5 h | T0 |
| [`03_which_experts_a_batch_touches`](notebooks/03_which_experts_a_batch_touches.ipynb) | derive E(1 − (1 − k/E)^T); reproduce layer 01's MoE table; predict the decode crossover from weights streamed ÷ weights multiplied (about total ÷ active); explain rows per expert (B·k/E) and why MoE wants big batches; show what skew does (simulated) | §5 | ~1.5 h | T0 |
| [`04_expert_parallelism_and_all_to_all`](notebooks/04_expert_parallelism_and_all_to_all.ipynb) | price dispatch and combine on NVLink, PCIe and InfiniBand; explain DeepEP's published numbers; model each rank as its own roofline and see why skew slows prefill GEMMs but decode exchanges; rebalance; compare TP, vLLM's default exchange and all-to-all kernels; choose a wide-EP degree and price a two-node fabric | §6 | ~1.5 h | T0 |
| [`05_sizing_and_cost`](notebooks/05_sizing_and_cost.ipynb) | size memory by total, prefill by active, decode by bytes streamed; budget one small GPU the way vLLM does; plan GPUs and EP degree against an ITL target; cost per million tokens against a dense model and against a dense model of the active size; price CPU offload on a small GPU | §7, §8 | ~1 h | T0 |

## Run it

```bash
cd moe-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 75 tests, ~30 s
python3 -m jupyterlab notebooks                # do the exercises
```

## The whole library

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`moecore/moe.py`](moecore/moe.py) | ~270 | the layer: `Expert` (SwiGLU), `route()` with the families' routers (`ROUTERS`), group-limited routing, `MoELayer.forward` (sort by expert → one GEMM per expert → weighted scatter-add) and `forward_dense` (the reference); `MoEConfig` counts total and active parameters (three embedding conventions), KV bytes per token and attention FLOPs per cached position (MHA/GQA or absorbed MLA) |
| [`moecore/routing.py`](moecore/routing.py) | ~120 | load balance: the Switch aux loss (hf and Megatron normalisations) and its gradient, the sequence-level loss, z-loss, capacity and dropping, vLLM-style dropless padding (`align_block_size`), DeepSeek's sign-step bias, expert choice, utilisation stats |
| [`moecore/train.py`](moecore/train.py) | ~140 | a tiny MoE on a clustered regression task with hand-written gradients (`loss_and_grads`, checked by finite differences) and Adam: collapse without balancing, balance with the aux loss or the bias |
| [`moecore/touched.py`](moecore/touched.py) | ~150 | experts touched (closed form and Monte Carlo with Zipf skew), a device catalogue, the decode-step roofline (`decode_step`, same model as `roofline.llm.decode`), the crossover batch, the KV share |
| [`moecore/ep.py`](moecore/ep.py) | ~230 | expert parallelism: dispatch bytes (FP8 scales included), α-β all-to-all (one link or two levels), placement, the exchange matrix, each rank as its own roofline and the slowest rank, EPLB-style rebalancing, TP vs vLLM's `allgather_reducescatter` vs all-to-all kernels, wide-EP memory, a decode step on n GPUs in the EP or TP layout |
| [`moecore/sizing.py`](moecore/sizing.py) | ~180 | `MODELS` (Mixtral, Qwen3, Qwen1.5-MoE, DeepSeek-V3, gpt-oss, Llama 4, OLMoE, granite, and dense Llama 3.1 for comparison; dated, verify-marked), weight bytes with quantized experts, a dense model of the active size, sessions that fit (round fleet budget) and KV room as vLLM budgets one GPU (`kv_room_gib`, `min_offload_gib`, the lab's numbers), prefill FLOPs, the all-experts floor, `plan()`, cost per million tokens, CPU-offload penalty |

## What the tests prove

`tests/` has one focused test per concept (67, plus 8 notebook-tooling checks; offline, ~30 s in all). The ones that carry the claims:

- **Sparse == dense.** For all five router families, the grouped forward equals every expert on every token
  (weighted by a dense gate) to 10⁻¹²; with E = k = 1 the layer is the dense MLP; unchosen experts run on no rows
  (`test_moe.py`).
- **Repo numbers, reproduced.** Layer 01's PRIMER §3.6 table (Mixtral on an H200: 25,631,531,008 bytes and 5.34 ms at
  batch 1 … 21.20 ms at 64; Qwen3's experts per layer) and its crossovers 207 / 754 / 2,055, plus a check that layer
  01's primer still prints those rows (`test_touched.py`); layer 02's 4 MiB, 22 µs pairwise and 10 µs direct
  all-to-all (`test_ep.py`); the capacity primer's Mistral Large 3 17.6 ms floor and 2 × active × tokens prefill,
  and the lab's `moelab.offload.fit` numbers for OLMoE on a T4 and Qwen3-30B-A3B on an L4 (`test_sizing.py`).
- **The slowest rank, in both regimes.** Each EP rank is its own roofline: a skewed decode batch keeps every rank's
  expert time equal (weight reads) and costs only at the busiest port, while a prefill-sized batch makes the
  busiest rank's rows the layer time (`test_ep.py`).
- **Configs to published totals.** Mixtral 46,702,526,464 / 12,879,659,008 and Qwen3-30B-A3B 30,531,911,680 /
  3,352,821,760 exactly; DeepSeek-V3 671.03B / 37.55B, gpt-oss-120b 116.83B / 5.13B (LM head only), Llama 4
  Maverick 400.71B / 17.18B and six more to 0.01B (`test_moe.py`).
- **Balance.** The Switch loss is k (hf) or 1 (Megatron) when uniform; its gradient matches finite differences;
  the bias rule is a sign step; vLLM's `moe_align_block_size` docstring example is reproduced (`test_routing.py`).
  The trainer's gradients match finite differences; without balancing at least one expert dies on every seed tested,
  with the aux loss or the bias all four carry 15–40% and the loss is lower (`test_train.py`).
- **Formulas pinned to hand-computed values:** DeepEP's published 98 GB/s from the byte formula, EPLB's 2.38 GiB per
  redundant DeepSeek-V3 expert, MXFP4 gpt-oss at 65.2 / 13.8 GB, sessions that fit, plans, cost and offload.
- **The primer says what the code computes.** `test_primer_numbers.py` recomputes every computed number in
  [`../PRIMER.md`](../PRIMER.md) and fails if the text drifts.

## Caveats: what is modelled and what is simplified

Every time, bandwidth and cost this package prints is a **model** — a roofline bound or an α-β estimate with layer
01's illustrative link numbers — and labelled simulated; real kernels land below the roofline, and real all-to-alls
add synchronisation the model ignores. Faithful to the sources: the routers' arithmetic (transformers, DeepSeek-V3's
`Gate`, gpt-oss and Llama 4 reference code), the Switch-loss normalisations, Megatron's bias rule, vLLM's padding and
EPLB memory formula, the decode-step model of layer 01. Simplified: uniform or Zipf routing instead of measured
traces (the lab records real ones); perfectly balanced EP in `decode_on` (the slowest rank is studied separately
in `layer_time`); no overlap of communication with compute; absorbed-MLA attention FLOPs from the latent widths;
PCIe peer-to-peer α-β values assumed (the lab's); prices are placeholders. The trainer is a
toy (four linear experts, full-batch Adam): the direction of its results is robust across seeds, the exact numbers
are not.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants (stable cell ids)
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs this
package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../moe-lab/`](../moe-lab/README.md) to train a tiny MoE in torch, watch a real router with hooks, measure decode step
time against batch in vLLM, run `--enable-expert-parallel` on two GPUs (Kaggle's free 2×T4), and fit an MoE onto a
16–24 GB GPU with offload or 4-bit experts. For the engine around the layer, see
[`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/README.md); for wide-EP in a fleet,
[`05-orchestrator`](../../../05-orchestrator/README.md). Where each tier runs and what it costs:
[`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
