# moe-core — build a mixture-of-experts layer in numpy and predict what it costs to serve

After this, you can do these things with `moecore`:

- Route tokens through an MoE layer by hand.
- See a router collapse, and repair it.
- Predict how many experts a decode batch reads, and when the batch becomes compute-bound.
- Give the cost of the expert-parallel all-to-alls.
- Size a deployment.

`moecore` is a standard-library + numpy package (~1,100 lines). It is sufficiently small that you can read it in one
session.

## Start here

1. Read "The one-minute version" in [`../PRIMER.md`](../PRIMER.md). Then read §1–§2 (why sparsity, the MoE layer).
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 75 tests run in about 30 s. One of
   them is "the sparse forward equals every expert on every token". Another is "layer 01's MoE table, reproduced to
   the byte".
3. Open [`notebooks/01_the_moe_layer.ipynb`](notebooks/01_the_moe_layer.ipynb) and route six tokens.

For the fastest result, run this from this directory:

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

*Tier T0 is a laptop or a Colab CPU, at no cost. Everything here runs with no GPU and no network.*

Each notebook starts with "The one-minute version". Then it has worked examples against the code, and exercises.
Each exercise has a check cell that prints ✅. The notebook ends with "In a design review". The finished versions
are in [`solutions/`](solutions/). The notebooks take about 7 hours with the primer.

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_the_moe_layer`](notebooks/01_the_moe_layer.ipynb) | Route tokens through the real routers (Mixtral, OLMoE and Qwen3's `norm_topk_prob` settings, DeepSeek-V3, gpt-oss, Llama 4). Tell why their weights are different. Write the sparse forward pass. Count total and active parameters from a config. Name the "active" convention that a published number uses. Give the argument for fine-grained experts. | §1, §2 | ~1.5 h | T0 |
| [`02_routing_and_load_balance`](notebooks/02_routing_and_load_balance.ipynb) | Calculate the Switch loss in both normalisations, the z-loss, capacity and token drops, and dropless padding. Give a router a balanced load with DeepSeek's bias. See a small MoE collapse and then recover (hand-written gradients). | §3, §4 | ~1.5 h | T0 |
| [`03_which_experts_a_batch_touches`](notebooks/03_which_experts_a_batch_touches.ipynb) | Derive E(1 − (1 − k/E)^T). Reproduce the MoE table of layer 01. Predict the decode crossover from weights streamed ÷ weights multiplied (about total ÷ active). Explain rows per expert (B·k/E), and why MoE wants large batches. Show what skew does (simulated). | §5 | ~1.5 h | T0 |
| [`04_expert_parallelism_and_all_to_all`](notebooks/04_expert_parallelism_and_all_to_all.ipynb) | Calculate the cost of dispatch and combine on NVLink, PCIe and InfiniBand. Explain the published numbers of DeepEP. Treat each rank as its own roofline. Then see why skew slows the GEMMs of prefill but the exchanges of decode. Rebalance. Compare TP, vLLM's default exchange and all-to-all kernels. Select a wide-EP degree, and calculate the cost of a two-node fabric. | §6 | ~1.5 h | T0 |
| [`05_sizing_and_cost`](notebooks/05_sizing_and_cost.ipynb) | Size memory by total, prefill by active and decode by bytes streamed. Make a budget for one small GPU the way vLLM does. Plan GPUs and EP degree against an ITL target. Calculate the cost per million tokens against a dense model, and against a dense model of the active size. Calculate the cost of CPU offload on a small GPU. | §7, §8 | ~1 h | T0 |

## Run it

```bash
cd moe-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 75 tests, ~30 s
python3 -m jupyterlab notebooks                # do the exercises
```

## The whole library

Read the modules in this order. Each module starts with a docstring that states the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`moecore/moe.py`](moecore/moe.py) | ~270 | The layer: `Expert` (SwiGLU), `route()` with the routers of the families (`ROUTERS`), group-limited routing, `MoELayer.forward` and `forward_dense` (the reference). `MoELayer.forward` sorts by expert, then does one GEMM per expert, then does a weighted scatter-add. `MoEConfig` counts total and active parameters (three embedding conventions) and KV bytes per token. It also counts attention FLOPs per cached position (MHA/GQA or absorbed MLA). |
| [`moecore/routing.py`](moecore/routing.py) | ~120 | Load balance: the Switch aux loss (hf and Megatron normalisations) and its gradient, the sequence-level loss and z-loss. It also has capacity and token drops, vLLM-style dropless padding (`align_block_size`), DeepSeek's sign-step bias, expert choice and utilisation stats. |
| [`moecore/train.py`](moecore/train.py) | ~140 | A small MoE on a clustered regression task, with Adam and hand-written gradients (`loss_and_grads`, with a finite-difference check). It shows collapse without load balance, and balance with the aux loss or the bias. |
| [`moecore/touched.py`](moecore/touched.py) | ~150 | Experts touched (closed form and Monte Carlo with Zipf skew) and a device catalogue. It also has the decode-step roofline (`decode_step`, the same model as `roofline.llm.decode`), the crossover batch and the KV share. |
| [`moecore/ep.py`](moecore/ep.py) | ~230 | Expert parallelism: dispatch bytes (FP8 scales included), α-β all-to-all (one link or two levels), placement and the exchange matrix. It also has each rank as its own roofline, the slowest rank and an EPLB-style rebalance. It compares TP, vLLM's `allgather_reducescatter` and all-to-all kernels. It has wide-EP memory, and a decode step on n GPUs in the EP or TP layout. |
| [`moecore/sizing.py`](moecore/sizing.py) | ~180 | `MODELS` has Mixtral, Qwen3, Qwen1.5-MoE, DeepSeek-V3, gpt-oss, Llama 4, OLMoE, granite, and dense Llama 3.1 for comparison. The entries have dates and a verify mark. The module also has weight bytes with quantized experts, and a dense model of the active size. It has sessions that fit (round fleet budget), and KV room in the budget that vLLM makes for one GPU (`kv_room_gib`, `min_offload_gib`, the lab's numbers). It also has prefill FLOPs, the all-experts floor, `plan()`, cost per million tokens and the CPU-offload penalty. |

## What the tests prove

`tests/` has one focused test per concept. There are 67, plus 8 checks of the notebook tools. All of them run
offline, in ~30 s. These tests carry the claims:

- **Sparse == dense.** For all five router families, the grouped forward equals every expert on every token
  (weighted by a dense gate) to 10⁻¹². With E = k = 1, the layer is the dense MLP. Experts that the router does not
  select run on no rows (`test_moe.py`).
- **Repo numbers, reproduced.** The tests reproduce the table in PRIMER §3.6 of layer 01 and the crossovers
  207 / 754 / 2,055 of that primer. The table gives Mixtral on an H200 (25,631,531,008 bytes and 5.34 ms at batch 1 …
  21.20 ms at 64) and Qwen3's experts per layer. A check makes sure that layer 01's primer still prints those rows
  (`test_touched.py`). The tests also reproduce layer 02's 4 MiB, 22 µs pairwise and 10 µs direct all-to-all
  (`test_ep.py`). In `test_sizing.py`, the tests reproduce the 17.6 ms floor of Mistral Large 3 and the prefill of
  2 × active × tokens from the capacity primer. The same file also reproduces the lab's `moelab.offload.fit` numbers
  for OLMoE on a T4 and Qwen3-30B-A3B on an L4.
- **The slowest rank, in both regimes.** Each EP rank is its own roofline. A skewed decode batch keeps the expert
  time of every rank equal (weight reads), and costs only at the busiest port. But a prefill-sized batch makes the
  rows of the busiest rank the layer time (`test_ep.py`).
- **Configs to published totals.** The tests match Mixtral 46,702,526,464 / 12,879,659,008 and Qwen3-30B-A3B
  30,531,911,680 / 3,352,821,760 exactly. They match these to 0.01B: DeepSeek-V3 671.03B / 37.55B, gpt-oss-120b
  116.83B / 5.13B (LM head only), Llama 4 Maverick 400.71B / 17.18B and six more (`test_moe.py`).
- **Balance.** The Switch loss is k (hf) or 1 (Megatron) when the routing is uniform. Its gradient matches finite
  differences. The bias rule is a sign step. The tests reproduce the docstring example of vLLM's
  `moe_align_block_size` (`test_routing.py`). The gradients of the trainer match finite differences, and without load
  balance at least one expert becomes dead on every seed that the tests tried. With the aux loss or the bias, all
  four experts carry 15–40% and the loss is lower (`test_train.py`).
- **Formulas pinned to hand-computed values:** The tests pin DeepEP's published 98 GB/s from the byte formula, and
  EPLB's 2.38 GiB per redundant DeepSeek-V3 expert. They also pin MXFP4 gpt-oss at 65.2 / 13.8 GB, sessions that
  fit, plans, cost and offload.
- **The primer says what the code computes.** `test_primer_numbers.py` recomputes every computed number in
  [`../PRIMER.md`](../PRIMER.md). It fails if the text and the computed numbers do not agree.

## Caveats: what is modelled and what is simplified

Every time, bandwidth and cost that this package prints is a **model**, and each has the label "simulated". The model is
a roofline bound, or an α-β estimate with the illustrative link numbers of layer 01. Real kernels stay below the
roofline. Real all-to-alls add synchronisation that the model ignores.

These items agree with the sources:

- The arithmetic of the routers (transformers, DeepSeek-V3's `Gate`, gpt-oss and Llama 4 reference code).
- The Switch-loss normalisations.
- Megatron's bias rule.
- vLLM's padding and EPLB memory formula.
- The decode-step model of layer 01.

The model simplifies these items:

- Uniform or Zipf routing instead of measured traces. The lab records real traces.
- Perfectly balanced EP in `decode_on`. A separate function, `layer_time`, studies the slowest rank.
- No overlap of communication with compute.
- Absorbed-MLA attention FLOPs from the latent widths.
- The α-β values for PCIe peer-to-peer, which the model assumes (the values of the lab).
- Prices, which are placeholders.

The trainer is a toy (four linear experts, full-batch Adam). The direction of its results stays the same across
seeds, but the exact numbers do not.

## Regenerating notebooks

The builder generates `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with
`### BEGIN SOLUTION` blocks). Edit the sources. Then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants (stable cell ids)
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three commands and the tests. On Colab, the first cell of each notebook clones the repo and
installs this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../moe-lab/`](../moe-lab/README.md). There, you can do these things:

- Train a small MoE in torch.
- Monitor a real router with hooks.
- Measure decode step time against batch in vLLM.
- Run `--enable-expert-parallel` on two GPUs (Kaggle's free 2×T4).
- Fit an MoE onto a 16–24 GB GPU with offload or 4-bit experts.

For the engine around the layer, see
[`04-inference-engine/serving-engine`](../../../04-inference-engine/serving-engine/README.md). For wide-EP in a fleet,
see [`05-orchestrator`](../../../05-orchestrator/README.md). For where each tier runs and what it costs, see
[`COMPUTE.md`](../../../COMPUTE.md). The licence is MIT.
