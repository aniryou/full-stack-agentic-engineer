# mini-engine-core — build an inference engine's step loop, scheduler and KV cache in numpy

After this, you can explain every decision that an engine like vLLM makes in one step:

- Which request runs.
- How many tokens the engine schedules for it.
- Which KV blocks it gets.
- Which request the engine preempts.
- Which cached prefix the engine uses again.

You can explain them because you fill in the code that makes these decisions. This code is in `minengine`, a numpy
"nano-vLLM" (~1,400 lines). It is small, so you can read it in two sessions.

## Start here

1. Read [`../PRIMER.md`](../PRIMER.md) §1–§2 (anatomy of an engine, continuous batching).
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 75 tests run in about 50 s. They
   include "paged attention equals dense attention to 1e-10".
3. Open [`notebooks/01_the_step_loop_and_continuous_batching.ipynb`](notebooks/01_the_step_loop_and_continuous_batching.ipynb).
   Then take one engine step apart.

## What you get

*Tier T0 is a laptop or a Colab CPU, at no cost. Everything here runs with no GPU and no network.*

Each notebook starts with "The one-minute version". Then it does worked examples against the code. It sets
exercises, each with a check cell that prints ✅. It ends with "In a design review". The finished versions are in
[`solutions/`](solutions/). The notebooks take about 11 hours in all with the primer (the curriculum of the
repository, modules 04.1–04.6).

| Notebook | You will be able to… | Primer | Time | Tier |
|---|---|---|---|---|
| [`01_the_step_loop_and_continuous_batching`](notebooks/01_the_step_loop_and_continuous_batching.ipynb) | Take one step apart (the flat batch, slot mapping). Compare static and continuous batching. Count peak KV blocks per request, and calculate concurrency from KV blocks. Divide one MLP across two "GPUs" (tensor parallelism and its all-reduces). | §1, §2, §9 | ~2 h | T0 |
| [`02_chunked_prefill_and_the_token_budget`](notebooks/02_chunked_prefill_and_the_token_budget.ipynb) | Find the knee of the step-time curve. Show prefill/decode interference and what chunked prefill does. Do a sweep of the token budget at moderate load and at saturation, with goodput (simulated). Select a budget for an ITL SLO. Size KV to prevent preemption. Show what the whole-prompt admission check gives. | §3, §4, §11 | ~2 h | T0 |
| [`03_prefix_caching`](notebooks/03_prefix_caching.ipynb) | Name blocks by a chained hash, and say why the parent is in the name. Share a system prompt with refcounts, within one step. Evict by LRU. Predict a hit rate. Lay out an agent prompt for 80%+ hits. Compare a radix tree with block hashing. | §5 | ~2 h | T0 |
| [`04_sampling_and_structured_output`](notebooks/04_sampling_and_structured_output.ipynb) | Write the code for temperature, top-k/p, min-p, penalties, seeds and raw logprobs. Force valid JSON with an FSM mask (syntax, not sense). Compile a JSON-schema automaton into per-state masks over multi-character tokens. | §6 | ~1.5 h | T0 |
| [`05_speculative_decoding`](notebooks/05_speculative_decoding.ipynb) | Prove that the rejection rule is exact. Measure α and tokens per pass on a draft/target pair, and explain why the formula over-predicts deep k. Use prompt lookup for copy-heavy outputs. Say when speculation costs more time than it saves (simulated). | §7 | ~2 h | T0 |
| [`06_quantization`](notebooks/06_quantization.ipynb) | Quantize to INT8/INT4/FP8 at each scale granularity. Handle outliers with SmoothQuant. Measure model-level damage. Say what each scheme and FP8 KV give for decode, prefill and concurrency on an L4 (simulated). | §8 | ~1.5 h | T0 |

Multi-LoRA (primer §10) has no notebook. Its numbers come from `perf.lora_params()` and the adapter-salted block
names. `tests/test_perf.py` and `tests/test_kv.py` pin them.

## Run it

```bash
cd mini-engine-core
python3 -m pip install -r requirements.txt    # numpy + what the notebooks and tests need
python3 -m pytest -q                           # 75 tests, ~50 s
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

Read the modules in this order. Each module starts with a docstring that states the one idea it teaches.

| File | Lines | What it teaches |
|------|------:|-----------------|
| [`minengine/model.py`](minengine/model.py) | ~230 | A small Llama-style decoder (byte vocabulary, RMSNorm, RoPE, GQA, SwiGLU). Its attention reads K/V **through block tables** from a flat batch of tokens from many requests. `forward_dense` is the textbook reference. |
| [`minengine/kv.py`](minengine/kv.py) | ~200 | The KV cache manager. It has a block pool, refcounts, and a prefix cache with the key `hash(parent, tokens, extra)`. It frees blocks tail first into its LRU free queue. It counts hits in tokens and keeps re-admissions after preemption apart. It also has an invariant checker. |
| [`minengine/scheduler.py`](minengine/scheduler.py) | ~210 | Continuous batching. It has one token budget per step, running requests first, chunked prefill, and FCFS admission with a whole-prompt check and watermark. It also has preemption by recompute, publication of full blocks at scheduling time, and stop conditions. |
| [`minengine/sampler.py`](minengine/sampler.py) | ~130 | The sampler order: penalties, then greedy/temperature, then min-p, then top-k, then top-p, then a seeded draw. Also raw logprobs and `ChoiceFSM`, a structured-output token mask. |
| [`minengine/engine.py`](minengine/engine.py) | ~170 | The loop. It schedules, then builds the flat batch (tokens, positions, slot mapping, block tables). Then it does one forward pass, samples and updates. Also `add_request`, `step`, `generate` and traces. |
| [`minengine/spec.py`](minengine/spec.py) | ~120 | Speculative decoding: exact accept/recover/bonus rejection sampling, α, expected tokens per pass, speed-up and best k, n-gram prompt lookup. |
| [`minengine/quant.py`](minengine/quant.py) | ~80 | INT8 per-tensor/per-channel, INT4 group-wise, FP8-E4M3 emulation, SmoothQuant scales, error metrics, bits per weight. |
| [`minengine/perf.py`](minengine/perf.py) | ~260 | `max(bytes/BW, FLOPs/peak) + overhead` per step for real GPUs and models (quantized weights keep a 16-bit embedding and LM head). Also KV capacity, the speculation cost model and `simulate()`. `simulate()` runs the scheduler of this package on a virtual clock under Poisson load. Every output has the label SIMULATED. |

## What the tests prove

`tests/` has one focused test per concept (67, plus 8 checks of the notebook tools). They run offline, in ~50 s in all. The tests in the list that follows carry the correctness claims:

- **Paged == dense.** `forward` over scattered block tables, random chunk sizes and several sequences per batch
  equals `forward_dense` to 1e-10. The greedy tokens of the engine equal `generate_dense`, also under preemption
  and with prefix-cache hits. Seeded sampled outputs also do not change under preemption. A burst of three requests
  with a shared prefix hits within one step (`[0, 64, 64]`). In that burst, every logprob still equals the dense
  reference to 1e-9 (`test_model.py`, `test_engine.py`, `test_scheduler.py`).
- **The prefix cache never serves a block computed under a different prefix.** An oracle records the full token
  prefix behind every block. It examines every hit over hundreds of random prompts. The same harness shows that a
  block name *without* the parent does serve incorrect K/V (`test_kv.py`).
- **Speculative decoding is exact.** A chi-square test over all 64 three-token outcomes matches the joint
  distribution of the target. The same test rejects plausible incorrect code that samples again from `p` instead of
  from the residual. Thus the test has power (`test_spec.py`).
- **Scheduler invariants.** The token budget and the sequence limit hold at every step. The scheduler publishes a
  block only if it schedules a request in the batch of that step to fill the block. Refcounts, the free queue and
  the cache map stay consistent after every step. No block leaks after preemption, finish or abort. A victim that is
  already in the batch gives its tokens back. The whole-prompt check and the watermark each decrease preemptions on
  a tight pool (`test_scheduler.py`).
- **The sampler's order.** Three settings pairs pin the order temperature, min-p, top-k, top-p, as in the `Sampler`
  of vLLM. Under any other order, the kept sets of these pairs are different. A pipeline in a different order fails
  (`test_sampler.py`).
- **Formulas pinned to hand-computed values:**
  - KV bytes per token.
  - The decode step as weight read.
  - Prefill FLOPs.
  - The knee.
  - KV blocks.
  - Quantized weight bytes with a 16-bit embedding and LM head (5.70 GB for Llama-3.1-8B INT4).
  - LM-head FLOPs per verified position.
  - The speculation cost model against `E / (k c + 1)`.
  - Goodput.
  - Expected tokens `(1 − α^(k+1)) / (1 − α)`.
  - Bits per weight.
  - The FP8 grid.
  - The method that counts hits, with preempted re-lookups kept apart.

  These tests are in `test_perf.py`, `test_spec.py`, `test_quant.py` and `test_kv.py`.

## Caveats: what is faithful to vLLM, and what is simplified

A check against the V1 source of vLLM (Sep 2026, see the Verify list of the primer) found these parts faithful:

- Unified scheduling on `num_computed_tokens`.
- Running requests before waiting requests.
- No admissions in a step that preempted.
- The FCFS victim is the newest running request.
- `max_cache_hit_length = num_tokens − 1`.
- Chained block hashes with extra keys.
- Blocks published at scheduling time. Thus requests that the engine admits in the same step share a prefix. The
  core publishes the blocks of a running request when the running pass is final. Thus a request that the engine
  preempts later in that pass never names blocks that it will not compute.
- Blocks freed tail first into an LRU free queue, with lazy eviction.
- Hits counted in tokens. The exported counters do not include preempted re-lookups.
- The order of the sampler.
- Raw logprobs.
- The accepted/recovered/bonus rejection rule.
- `RequestStatus`-style names.

These parts are simplified:

- One KV cache group (no hybrid or sliding-window models).
- No async scheduling, CUDA graphs, tensor parallelism or LoRA. TP appears as a numpy exercise in notebook 01.
- Speculation runs outside the engine loop (`spec.py`), not as lookahead slots in the scheduler. It takes any draft
  distribution q: sampled, or one-hot for a greedy draft, which is the default of vLLM.
- The model is float64 numpy.

Every latency and throughput that `minengine.perf` prints is **simulated**, and has a label that says so. A roofline
model drives the real scheduler of this package on a virtual clock. The KV sizing of `minengine.perf` uses round
inputs (0.9 of the datasheet memory of the GPU minus a flat 1 GB). [`../PRIMER.md`](../PRIMER.md) §4 compares these
inputs with the defaults of vLLM v0.30.0.

## Regenerating notebooks

The builder makes `notebooks/` and `solutions/` from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources. Then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

`make check` runs all three and also the tests. On Colab, the first cell of each notebook clones the repo and
installs this package (see [`../../../COLAB.md`](../../../COLAB.md)).

## When you outgrow this

Go to [`../vllm-serving-lab/`](../vllm-serving-lab/). There, you size a real model before you serve it and drive
vLLM with an open-loop load generator. You read its `/metrics`, do sweeps of the same knobs against an SLO, and
deploy it on Cloud Run GPU or GKE. Read the real scheduler and block pool in [`../../vllm-internals/`](../../vllm-internals/README.md).

For many replicas (routing, autoscaling, disaggregation), continue to
[`05-orchestrator`](../../../05-orchestrator/README.md). Where each tier runs and what it costs: [`COMPUTE.md`](../../../COMPUTE.md). MIT licensed.
