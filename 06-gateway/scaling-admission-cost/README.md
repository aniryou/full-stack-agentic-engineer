# scaling-admission-cost — size an agent service and keep it up under load

After this topic, you can do these things:

- Go from conversations per day to tokens per minute, turns in flight and dollars per conversation.
- Say what breaks first.
- Find the break-even for provisioned throughput or for your own GPUs.
- Design the rate limits, retries, circuit breakers and admission control that keep the service up when the model
  answers 429.

## Start here

1. Read §1–3 of the [scaling primer](agentic-scaling-lab/docs/01-scaling-primer.md) (about 40 min). These sections
   tell what is different when agents scale. They also give the dimensions of scale and the arithmetic, worked.
2. Run `cd agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m scalelab.capacity`. In less
   than a second, it shows the capacity plan for 100,000 conversations a day. Then run `python3 -m pytest -q`
   (37 tests, ~3 s).
3. Open [`01_scaling_math`](agentic-scaling-lab/notebooks/01_scaling_math.ipynb). Do notebooks 01–04 in sequence.
   Then do `05_hosted_or_own_gpus` with §3.5–3.6 of the [Mistral primer](agentic-scaling-lab/docs/mistral/01-scaling-primer.md).
   They are for the decision between a hosted API and your own GPUs.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost (no GPU, no key, no cloud account).* The times are approximate.
They come from modules 06.1–06.5 in [`CURRICULUM.md`](../../CURRICULUM.md).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`agentic-scaling-lab/`](agentic-scaling-lab/README.md) notebooks 01–04 | This path teaches the core concepts, with one module for each idea. The modules are capacity, resilience, admission, the turn loop, a fake model and a load simulator. The fake model has a shared tokens-per-minute pool. The load simulator runs on a virtual clock. You do the capacity math and find the provisioned-throughput break-even. You use budgets, checkpoints and idempotent writes in a turn. You use a token bucket, full-jitter backoff, a circuit breaker and degrade-before-shed admission. You get the settings from a load test. The path also has a [primer](agentic-scaling-lab/docs/01-scaling-primer.md) and a Cloud Run + Gemini [reference architecture](agentic-scaling-lab/docs/02-reference-architecture.md). | ~6 h | T0 (a Gemini key is optional) |
| The provider path of the same lab: notebook 05, `scalelab/mistral.py`, `scalelab/serving.py` | This path teaches the second way to pay for tokens, worked on Mistral models. You find what one vLLM replica delivers, the fleet for peak and the break-even GPU price. You see a fleet under load: no 429s, but every user gets slower answers. You write a spill-over rule from your fleet to the API. The path also has a [Mistral primer](agentic-scaling-lab/docs/mistral/01-scaling-primer.md) and the self-hosted model layer on Kubernetes (§11 of the reference architecture). | ~2 h after notebooks 01–04 | T0 (a `MISTRAL_API_KEY` is optional) |

The backend of the lab is a switch. In code, use `make_setup(mode, provider=...)`. For the notebooks, use
`SCALELAB_BACKEND` (`hosted` | `local` | `hybrid`) and `SCALELAB_PROVIDER` (`gemini` | `mistral`). If you do not set
them, the backend is the hosted Gemini pool.

## Run it

```bash
cd agentic-scaling-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q          # 37 tests, ~3 s
python3 -m scalelab.capacity                                                                # the capacity plan (Gemini)
python3 -m scalelab.mistral                                                                 # Mistral's API or a fleet, and the break-even
python3 -m scalelab.serving                                                                 # one vLLM replica per model and GPU
```

Open the notebooks in JupyterLab (`python3 -m pip install jupyterlab`). You can also open them from the Colab links in
the [layer README](../README.md#run-in-colab).

## How it fits

This topic builds on [`00-foundations/gpu-capacity-planning`](../../00-foundations/gpu-capacity-planning/README.md):
tokens, TTFT and TPOT, and fleet sizing. The replica model of the lab (`scalelab/serving.py`) is the fleet view of
layer 04's [`serving-engine`](../../04-inference-engine/serving-engine/README.md). The step-time model of that
topic is the reference.

The topic is one layer above the orchestrator ([`05-orchestrator`](../../05-orchestrator/README.md)). Admission
decides if a request runs. Then the router decides where it runs. Layer 03's Kueue quotas are the same "shape demand
to capacity" idea for GPU jobs.

The topic leads to layer 07's agents. This topic sets the limits of their turns. Its companion topic is
[`identity-security/`](../identity-security/README.md). In the [curriculum's spiral](../../CURRICULUM.md#31-why-this-order),
it is step 24, after the orchestrator.

## Caveats

- The latencies, 429s and costs come from simulations. The simulations run on a virtual clock, 50× faster than real
  time. Only the arithmetic is exact. The prices, quotas, rate limits and model ids are dated snapshots (5 and 19
  September 2026). They are in the verify list of each primer (verify).
- The notebooks use their own check helper. A check prints `PASS`, `FAIL` or `---- not attempted yet`, not ✅.
  Thus a practice notebook that you did not change prints "not attempted" at every check. This is intentional.
- The replica model in `scalelab/serving.py` is simplified. A decode step is bytes ÷ (bandwidth × 0.6) + 2 ms, with
  no compute term. Read it as the fleet view, not as a kernel model.
