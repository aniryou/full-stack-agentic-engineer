# agentic-scaling-lab — size, protect and price an agent service, on a hosted API or your own GPUs

After this lab, you can do these things:

- Take "100,000 conversations a day" to tokens per minute, turns in flight, a capacity request and dollars, in two
  minutes.
- Show the overload feedback loop and the mechanisms that break it: smoothing, jittered retries, breakers, degrade
  levels and shedding.
- Say at what GPU price a self-hosted vLLM fleet costs less than the API.

## Start here

1. Read Parts 1–3 of [`docs/01-scaling-primer.md`](docs/01-scaling-primer.md) (about 45 min). They tell why agents
   scale differently, and they give the arithmetic.
2. Run `python -m pip install -e ".[dev]" && python -m scalelab.capacity`. It shows the capacity plan in a second.
   Then run `python -m pytest -q` (37 tests, ~3 s).
3. Open [`notebooks/01_scaling_math.ipynb`](notebooks/01_scaling_math.ipynb). Do notebooks 01–04. Then do
   `05_hosted_or_own_gpus` for the fleet.

## What you get

*T0 is a laptop or a Colab CPU, at no cost: no GPU, no key, no cloud account. The lab deploys nothing. The
simulations run 50× faster than real time.* The times are approximate. The lab is modules 06.1–06.5 in
[`CURRICULUM.md`](../../../CURRICULUM.md).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| `notebooks/01_scaling_math` | Go from conversations/day to tokens/min, in-flight turns, Provisioned Throughput units and dollars. | 60 min | T0 |
| `notebooks/02_turn_loop_and_durability` | Make a crash in the middle of a turn produce one ticket, not two. Know what to ack and what to nack. | 60 min | T0 |
| `notebooks/03_rate_limits_and_admission` | Explain a 429. Smooth the traffic, add jitter to retries, break circuits, and degrade before you shed load. | 60 min | T0 |
| `notebooks/04_load_to_settings` | Show the overload feedback loop. Calculate the Cloud Run concurrency, the number of instances and the in-flight cap from a load test. | 60 min | T0 |
| `notebooks/05_hosted_or_own_gpus` | Calculate the size of one vLLM replica and of a fleet. Find the break-even GPU price. See overload on a fleet: no 429s, but every user gets slower answers. Write the spill-over rule. Calculate the vLLM and Kubernetes settings. | 2 h | T0 |
| [`docs/01-scaling-primer.md`](docs/01-scaling-primer.md) | This is the primer. It covers the dimensions of scale, the arithmetic, the mechanisms, the failure catalogue, the growth path, and how to walk the design in a review. | 1–2 h | — |
| [`docs/mistral/01-scaling-primer.md`](docs/mistral/01-scaling-primer.md) | This is the same method on Mistral's API and on your own GPUs. It covers the two ways to pay for tokens, the replica, the fleet, the break-even and sovereignty. | 1–2 h | — |
| [`docs/02-reference-architecture.md`](docs/02-reference-architecture.md) | This is the Cloud Run + Gemini reference architecture. Its §11 is the self-hosted model layer on Kubernetes. | 45 min | — |
| [`docs/03-capacity-plan.md`](docs/03-capacity-plan.md) | These are the numbers for both providers. The code generates them. | 10 min | — |
| [`docs/04-gcp-mapping.md`](docs/04-gcp-mapping.md) | This document connects each concept to its Google Cloud service, and each service to the setting that matters. Then it gives the vLLM and Kubernetes settings for a fleet, and notes for each cloud. | 20 min | — |
| [`docs/scaling-agentic-solutions-on-google-cloud.md`](docs/scaling-agentic-solutions-on-google-cloud.md) | This is the long-form companion, worked on a fictional insurer. Its Part 3 is `scalelab.capacity`, and a test pins it. | 2 h | — |

**The provider path.** Everything runs on the Google Cloud anchor (a hosted Gemini pool) if you do not switch it. The
Mistral provider and the self-hosted backends are `scalelab/mistral.py`, `scalelab/serving.py` and the `ServerPool` /
`HybridBackend` backends. To select them, use `make_setup(mode, provider=...)`. You can also use the environment
variables `SCALELAB_BACKEND` (`hosted` | `local` | `hybrid`) and `SCALELAB_PROVIDER` (`gemini` | `mistral`).

The whole provider path runs without a key. The optional `mistral` extra (`pip install -e ".[mistral]"`) adds only the clients for the real
calls. These calls are at the bottom of `scalelab/mistral.py` and `scalelab/model.py`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m scalelab.capacity                 # the Gemini capacity plan
python -m scalelab.mistral                  # Mistral's API or a fleet: the plan, the fleet, the break-even
python -m scalelab.serving                  # one vLLM replica per model and GPU (estimates)
python -m pytest -q                         # 37 tests, ~3 s; the load simulations run in virtual time
jupyter lab notebooks/                      # start with 01_scaling_math.ipynb
SCALELAB_BACKEND=local jupyter lab notebooks/   # notebook 04's load test and exercise (d) on a vLLM fleet
```

This command generates `docs/03-capacity-plan.md`, and a test compares the file with the output of the command:
`{ python -m scalelab.capacity; echo; python -m scalelab.mistral --section; } > docs/03-capacity-plan.md`.

## The code (`scalelab/`, about 1,800 lines with docstrings)

| Module | The idea |
|---|---|
| `capacity.py` | It goes from conversations to turns, from turns to calls, and from calls to tokens/min. It also has Little's law, Provisioned Throughput units and the break-even, the cost per conversation, and what breaks first (Gemini). |
| `mistral.py` | It does the same on Mistral's API: the rate-limit request and the cost per conversation. It also has the fleet sized for peak, the break-even GPU price and the two-replica floor. |
| `serving.py` | It describes one vLLM replica with first-principles estimates. KV bytes per token (GQA against MLA) give the resident sequences. HBM bandwidth gives the time per token. It shows the relation between batch and throughput. It calculates TTFT from prefill FLOPs. |
| `model.py` | It is a fake model with prefix caching, on three backends. The backends are a hosted API, a fleet and the hybrid. The hosted API is a shared pool that answers 429. The fleet puts requests in a queue and becomes slower. The module also has `fake_model(provider)` and the real Gemini and vLLM calls. |
| `resilience.py` | A token bucket, backoff with full jitter, a circuit breaker, and `call_with_retries` with a deadline as its limit. |
| `admission.py` | It sets degrade levels 0–3 from in-flight turns, the pushback ratio, the queue age and (on the Mistral scenario) the backend saturation. It also has an in-flight cap, shedding with `Retry-After`, and hysteresis. |
| `loop.py` | The turn: budgets on four dimensions, a checkpoint for each step, idempotent writes and a resume after a crash. |
| `tools.py` | Simulated CRM / billing systems with QPS ceilings and structured errors. |
| `sim.py` | A load generator: many users against any backend. It gives the latency, the shedding, the spill-over share and the cost for each regime. It also has the backend switch. |
| `clock.py` | The virtual clock, and a virtual-time event loop for deterministic tests. |

The notebooks use their own check helper (`nbutil.py`, next to `scalelab/`). An exercise is a `todo()` placeholder, not
`# YOUR CODE HERE`. A check prints `PASS`, `FAIL` or `---- not attempted yet`, not a ✅. The practice notebooks
(`notebooks/`) run from start to end with no change. Their solutions (`solutions/`, the same file names) run, and every
check passes.

## The anchor numbers (verify before quoting — prices, limits and GPU rates move)

**Gemini, 5 September 2026.** 100 k conversations/day give 15 model calls/s on average, 46 at peak and 153 in an
incident. These calls send 4.6 / 13.75 / 45.8 M input tokens per minute, against a baseline of 10 M TPM for the Flash
tier. The same load gives 42 / 125 / 417 turns in flight. The cost is $0.068 per conversation with routing and caching ($0.14 without). The
base load needs 69 GSUs of Provisioned Throughput, with the break-even at 75 % utilisation on a 1-year term.

In the Gemini case, the first thing that breaks is the token budget. The billing mainframe is next. Cloud Run is about 0.1 % of the bill.

**Mistral and a fleet, 19 September 2026.** The same demand is 4.8 / 14.3 / 47.7 M total tokens per minute. This
demand needs a rate-limit request of 60 RPS and 19 M TPM. On the planning mix, the cost is $0.0131 per conversation (90 %
Small 4 + 10 % Medium 3.5, cached). On this mix, the total cost is $39.7 k/month.

The other path is Ministral 3 14B on H100s: batch 24 at 20 ms per token (estimate), 4 / 11 / 34 replicas, and
$55 k/month on on-demand GPUs. The fleet costs less than the API when an H100-hour costs less than about $4.95. Thus
residency decides the path, and the arithmetic puts a price on the decision.

## How it fits

This lab comes after layer 05's [serving orchestration](../../../05-orchestrator/serving-orchestration/README.md)
(routing and replica autoscaling) and layer 04's [serving engine](../../../04-inference-engine/serving-engine/README.md).
The `minengine.perf` model of the serving engine is the reference step-time model for the simpler estimate in `serving.py`.
Beside this lab is [identity and security](../../identity-security/README.md). That topic says what an agent has
permission to do. This lab says how much it has permission to run. After this lab comes the durable turn in layer 07.

## Caveats

- Every latency and throughput number comes from a simulation (the load tests) or from a first-principles estimate
  (a replica). Before anyone buys GPUs, replace these numbers with a load test against staging and with
  `vllm bench serve` on the target GPUs.
- The prices, limits and model ids are dated (5 and 19 September 2026). The docs mark them `(verify)`.
- Licence: MIT.
