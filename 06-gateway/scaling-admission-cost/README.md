# scaling-admission-cost — size an agent service and keep it up under load

After this topic you can go from conversations per day to tokens per minute, turns in flight and dollars per
conversation, say what breaks first, find the break-even for provisioned throughput or your own GPUs, and design the
rate limits, retries, circuit breakers and admission control that keep the service up when the model answers 429.

## Start here

1. Read the [scaling primer](agentic-scaling-lab/docs/01-scaling-primer.md) §1–3: what is different about scaling
   agents, the dimensions of scale, the arithmetic worked (about 40 min).
2. `cd agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m scalelab.capacity` — under a second:
   the capacity plan for 100,000 conversations a day; then `python3 -m pytest -q` (14 tests, ~4 s).
3. Open [`01_scaling_math`](agentic-scaling-lab/notebooks/01_scaling_math.ipynb) and work notebooks 01–04 in order.

## What you get

*Tiers: T0 = laptop or Colab CPU, free (no GPU, no key, no cloud account).* Times are rough and come from modules
06.1–06.5 in [`CURRICULUM.md`](../../CURRICULUM.md).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`agentic-scaling-lab/`](agentic-scaling-lab/README.md) | the core concepts in ~650 lines of Python, one module per idea (capacity, resilience, admission, the turn loop, a fake model with a shared tokens-per-minute pool, a load simulator on a virtual clock): capacity math and provisioned-throughput break-even; budgets, checkpoints and idempotent writes in a turn; token bucket, full-jitter backoff, circuit breaker and degrade-before-shed admission; settings derived from a load test. A [primer](agentic-scaling-lab/docs/01-scaling-primer.md), a Cloud Run + Gemini [reference architecture](agentic-scaling-lab/docs/02-reference-architecture.md), 4 notebooks with solutions, 14 tests | ~6 h | T0 (a Gemini key is optional) |
| [`agentic-scaling-lab-mistral/`](agentic-scaling-lab-mistral/README.md) | the same concepts on Mistral models, plus the second way to pay for tokens: what one vLLM replica delivers (`scalelab/serving.py`), the fleet for peak, the break-even GPU price, and a spill-over rule from your fleet to the API; a Kubernetes reference architecture; 4 notebooks, 19 tests | ~2 h after the GCP lab (its hosted-vs-own-GPUs parts) | T0 (a `MISTRAL_API_KEY` is optional) |

## Run it

The two labs both install a package named `scalelab`, so install one at a time, each in its own venv (or skip the
install and run from the lab folder with `PYTHONPATH=. python3 -m pytest -q`).

```bash
cd agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 14 tests, ~4 s
python3 -m scalelab.capacity                                                                # the capacity plan
# in a second venv:
cd ../agentic-scaling-lab-mistral && python3 -m pip install -e ".[dev]" && python3 -m pytest -q   # 19 tests, ~2 s
python3 -m scalelab.serving                                                                 # one vLLM replica per model and GPU
```

Open the notebooks in JupyterLab (`python3 -m pip install jupyterlab`) or from the Colab links in the
[layer README](../README.md#run-in-colab).

## How it fits

Builds on [`00-foundations/gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/README.md) (tokens,
TTFT and TPOT, fleet sizing); the Mistral variant's replica model is the fleet view of layer 04's
[`serving-engine`](../../04-inference-engine/serving-engine/README.md), whose step-time model is the reference. It
sits one layer above the orchestrator ([`05-orchestrator`](../../05-orchestrator/README.md)): admission decides
whether a request runs, the router then decides where, and layer 03's Kueue quotas are the same "shape demand to
capacity" idea for GPU jobs. It leads to layer 07's agents, whose turns this topic bounds, and pairs with
[`identity-security/`](../identity-security/README.md). In the [curriculum's spiral](../../CURRICULUM.md#31-why-this-order)
it is step 23, after the orchestrator.

## Caveats

- Latencies, 429s and costs come from simulations that run 50× faster than real time on a virtual clock; only the
  arithmetic is exact. Prices, quotas, rate limits and model ids are dated snapshots (5 and 19 September 2026) in
  each primer's verify list (verify).
- The notebooks use their own check helper: a check prints `PASS`, `FAIL` or `---- not attempted yet` rather than
  ✅, so an untouched practice notebook prints "not attempted" everywhere by design.
- The Mistral variant's serving model is simplified (a decode step is bytes ÷ (bandwidth × 0.6) + 2 ms, no compute
  term): read it as the fleet view, not a kernel model.
