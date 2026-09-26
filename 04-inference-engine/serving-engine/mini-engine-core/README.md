# mini-engine-core — build an inference engine's step loop, scheduler and KV cache in numpy

After this you can explain every decision an engine like vLLM makes in one step — who runs, how many tokens, which
KV blocks, which request is preempted, which cached prefix is reused — because you will have filled in the code
that makes it, in `minengine`, a numpy "nano-vLLM" small enough to read in a sitting (~960 lines).

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (anatomy of an engine, continuous batching).
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 67 tests in about 15 s, including
   "paged attention equals dense attention to 1e-10".
3. Open [`notebooks/01_the_step_loop_and_continuous_batching.ipynb`](notebooks/01_the_step_loop_and_continuous_batching.ipynb)
   and take one engine step apart.

## What you get

*Tier T0 = laptop or Colab CPU, free: everything here runs with no GPU and no network.* Each notebook opens with
"The one-minute version", works examples against the code, sets exercises with a check cell that prints ✅, and ends
with "In a design review". Finished versions are in [`solutions/`](solutions/). About 11 hours in all with the
primer (the repo's curriculum, modules 04.1–04.6).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_the_step_loop_and_continuous_batching`](notebooks/01_the_step_loop_and_continuous_batching.ipynb) | take one step apart (the flat batch, slot mapping); compare static and continuous batching; count peak KV blocks per request and concurrency from KV blocks; split one MLP across two "GPUs" (tensor parallelism and its all-reduces) | §1, §2, §9 | ~2 h | T0 |
| [`02_chunked_prefill_and_the_token_budget`](notebooks/02_chunked_prefill_and_the_token_budget.ipynb) | find the knee of the step-time curve; show prefill/decode interference and what chunked prefill does; sweep the token budget at moderate load and at saturation, with goodput (simulated); choose a budget for an ITL SLO; size KV to avoid preemption; show what the whole-prompt admission check buys | §3, §4, §11 | ~2 h | T0 |
| [`03_prefix_caching`](notebooks/03_prefix_caching.ipynb) | name blocks by a chained hash and say why the parent is in the name; share a system prompt with refcounts, within one step; evict LRU; predict a hit rate; lay out an agent prompt for 80%+ hits; compare a radix tree with block hashing | §5 | ~2 h | T0 |
| [`04_sampling_and_structured_output`](notebooks/04_sampling_and_structured_output.ipynb) | implement temperature, top-k/p, min-p, penalties, seeds and raw logprobs; force valid JSON with an FSM mask (syntax, not sense); compile a JSON-schema automaton into per-state masks over multi-character tokens | §6 | ~1.5 h | T0 |
| [`05_speculative_decoding`](notebooks/05_speculative_decoding.ipynb) | prove the rejection rule exact; measure α and tokens per pass on a draft/target pair (and why the formula over-predicts deep k); use prompt lookup for copy-heavy outputs; say when speculation stops paying (simulated) | §7 | ~2 h | T0 |
| [`06_quantization`](notebooks/06_quantization.ipynb) | quantize to INT8/INT4/FP8 at each scale granularity; handle outliers with SmoothQuant; measure model-level damage; say what each scheme and FP8 KV buys for decode, prefill and concurrency on an L4 (simulated) | §8 | ~1.5 h | T0 |

Multi-LoRA (primer §10) has no notebook: its numbers come from `perf.lora_params()` and the adapter-salted block
names, pinned in `tests/test_perf.py` and `tests/test_kv.py`.

## Run it

```bash
cd mini-engine-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 67 tests, ~15 s
python3 -m jupyterlab notebooks                # do the exercises
```

The library itself needs only numpy:

```python
from minengine import Engine, SamplingParams

eng = Engine(num_blocks=32, block_size=4, max_num_batched_tokens=16)   # a tiny model is built for you
outs = eng.generate(["The engine ", "When memory runs out"], SamplingParams(max_tokens=10, temperature=0))
print(outs[0].text, outs[0].finish_reason)
print(eng.trace())          # one line per step: who ran, how many tokens, which kind, KV blocks in use
# step   1 |  16 tok | r0 prefill 11 [0:11]->'t' | r1 chunk 5 [0:5] | kv 5/32
# step   2 |  16 tok | r0 decode 1 [11:12]->'o' | r1 prefill 15 [5:20]->'o' | kv 8/32
```

## The whole library

Read the modules in this order; each opens with a docstring stating the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`minengine/model.py`](minengine/model.py) | ~230 | a tiny Llama-style decoder (byte vocabulary, RMSNorm, RoPE, GQA, SwiGLU) whose attention reads K/V **through block tables** from a flat batch of many requests' tokens; `forward_dense` is the textbook reference |
| [`minengine/kv.py`](minengine/kv.py) | ~200 | the KV cache manager: block pool, refcounts, prefix cache keyed by `hash(parent, tokens, extra)`, LRU free queue with tail-first freeing, hit accounting in tokens (re-admissions after preemption kept apart), an invariant checker |
| [`minengine/scheduler.py`](minengine/scheduler.py) | ~210 | continuous batching: one token budget per step, running requests first, chunked prefill, FCFS admission with a whole-prompt check and watermark, preemption by recompute, publishing full blocks at scheduling time, stop conditions |
| [`minengine/sampler.py`](minengine/sampler.py) | ~130 | penalties → greedy/temperature → min-p → top-k → top-p → a seeded draw; raw logprobs; `ChoiceFSM`, a structured-output token mask |
| [`minengine/engine.py`](minengine/engine.py) | ~170 | the loop: schedule → build the flat batch (tokens, positions, slot mapping, block tables) → one forward pass → sample → update; `add_request`, `step`, `generate`, traces |
| [`minengine/spec.py`](minengine/spec.py) | ~120 | speculative decoding: exact accept/recover/bonus rejection sampling, α, expected tokens per pass, speed-up and best k, n-gram prompt lookup |
| [`minengine/quant.py`](minengine/quant.py) | ~80 | INT8 per-tensor/per-channel, INT4 group-wise, FP8-E4M3 emulation, SmoothQuant scales, error metrics, bits per weight |
| [`minengine/perf.py`](minengine/perf.py) | ~260 | `max(bytes/BW, FLOPs/peak) + overhead` per step for real GPUs and models (quantized weights keep a 16-bit embedding and LM head), KV capacity, the speculation cost model, and `simulate()`: this package's scheduler on a virtual clock under Poisson load — every output labelled SIMULATED |

## What the tests prove

`tests/` has one focused test per concept (67, offline, ~15 s). The ones that carry the correctness claims:

- **Paged == dense.** `forward` over scattered block tables, random chunk sizes and several sequences per batch
  equals `forward_dense` to 1e-10; the engine's greedy tokens equal `generate_dense` — including under
  preemption and with prefix-cache hits — and seeded sampled outputs do not change under preemption either; a
  burst of three requests sharing a prefix hits within one step (`[0, 64, 64]`) and every logprob still equals
  the dense reference to 1e-9 (`test_model.py`, `test_engine.py`, `test_scheduler.py`).
- **The prefix cache never serves a block computed under a different prefix.** An oracle records the full token
  prefix behind every block and checks every hit over hundreds of random prompts; the same harness shows that
  a block name *without* the parent does serve wrong K/V (`test_kv.py`).
- **Speculative decoding is exact.** A chi-square test over all 64 three-token outcomes matches the target's
  joint distribution, and the same test rejects a plausible wrong implementation (resampling from `p` instead
  of the residual) — the test has power (`test_spec.py`).
- **Scheduler invariants.** Token budget and sequence limit hold every step; a block is published only if a
  request in that step's batch is scheduled to fill it; refcounts, free queue and cache map stay consistent after
  every step; no block leaks after preemption, finish or abort; a victim already in the batch gives its tokens
  back; the whole-prompt check and the watermark each cut preemptions on a tight pool (`test_scheduler.py`).
- **The sampler's order.** Three settings pairs whose kept sets differ under any other order pin temperature →
  min-p → top-k → top-p, as in vLLM's `Sampler`; a reordered pipeline fails (`test_sampler.py`).
- **Formulas pinned to hand-computed values:** KV bytes per token, the decode step as weight read, prefill FLOPs,
  the knee, KV blocks, quantized weight bytes with a 16-bit embedding and LM head (5.70 GB for Llama-3.1-8B INT4),
  LM-head FLOPs per verified position, the speculation cost model against `E / (k c + 1)`, goodput, expected
  tokens `(1 − α^(k+1)) / (1 − α)`, bits per weight, the FP8 grid, hit accounting with preempted re-lookups kept
  apart (`test_perf.py`, `test_spec.py`, `test_quant.py`, `test_kv.py`).

## Caveats: what is faithful to vLLM, and what is simplified

Faithful (checked against vLLM's V1 source, Sep 2026 — see the primer's Verify list): unified scheduling on
`num_computed_tokens`; running before waiting; no admissions in a step that preempted; FCFS victim = the newest
running request; `max_cache_hit_length = num_tokens − 1`; chained block hashes with extra keys; blocks published
at scheduling time, so requests admitted in the same step share a prefix (the core publishes a running request's
blocks once the running pass is final, so a request preempted later in that pass never names blocks it will not
compute); tail-first freeing into an LRU free queue with lazy eviction; hits counted in tokens, with preempted
re-lookups kept out of the exported counters; the sampler's order; raw logprobs; the accepted/recovered/bonus
rejection rule; `RequestStatus`-style names. Simplified: one KV cache group (no hybrid or sliding-window models);
no async scheduling, CUDA graphs, tensor parallelism or LoRA (TP appears as a numpy exercise in notebook 01);
speculation runs outside the engine loop (`spec.py`) rather than as lookahead slots in the scheduler (it takes any
draft distribution q — sampled, or one-hot for greedy drafting, which is vLLM's default); the model is float64
numpy.

Every latency and throughput `minengine.perf` prints is **simulated** (a roofline model driving this package's real
scheduler on a virtual clock) and labelled so; its KV sizing uses round inputs (0.9 of the GPU's datasheet memory
minus a flat 1 GB), which [`../PRIMER.md`](../PRIMER.md) §4 compares with vLLM v0.30.0's defaults.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three plus the tests. On Colab, each notebook's first cell clones the repo and installs
this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../vllm-serving-lab/`](../vllm-serving-lab/) to size a real model before serving it, drive vLLM with an
open-loop load generator, read its `/metrics`, sweep the same knobs against an SLO, and deploy it on Cloud Run
GPU or GKE; read the real scheduler and block pool in [`../../vllm-internals/`](../../vllm-internals/README.md). For
many replicas — routing, autoscaling, disaggregation — continue to
[`05-orchestrator`](../../../05-orchestrator/README.md). Where each tier runs and what it costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
