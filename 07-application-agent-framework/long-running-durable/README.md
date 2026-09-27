# long-running-durable — agents that survive crashes, duplicate deliveries and multi-day waits

After this topic you can take an agent that has to wait (for a tool, a person, the world or the clock) and design
it as a durable state machine: state the invariants that keep it correct, show where each crash window is closed,
and map it onto a queue, a store and stateless compute on any cloud, with Google Cloud worked end to end.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) §0–§4 (45 min): why long-running is a different problem, the run as a state
   machine, the five invariants, and the pattern catalogue.
2. `cd lra-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 13 tests in about a second,
   standard library; then `python3 workflow.py kill; python3 workflow.py resume` to watch one process die after
   publishing and a second one finish the run without publishing again.
3. Open [`lra-core/notebooks/01_durable_loop.ipynb`](lra-core/notebooks/01_durable_loop.ipynb), then move on to
   [`lra-gcp`](lra-gcp/README.md), the full engine.

## What you get

*Tiers: T0 = laptop or Colab CPU, free; T3 = the Google Cloud deployment, optional. No GPU anywhere, so the repo's
T1/T2 do not apply; model keys are optional (every notebook runs a scripted model).* Each notebook has a blank in
`notebooks/` that stops at its first exercise and a worked answer under the same name in `solutions/`.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | explain the five invariants, eight patterns and three reference architectures; quote the Google Cloud limits that decide a design; estimate wake-ups, writes and tokens; answer the design drills in §11 | 2 h | T0 |
| [`lra-core`](lra-core/README.md) | build the durable loop from scratch in ~250 lines of standard library (store, named-task queue, lease, checkpoint-before-enqueue, effect records, reaper) and break it at every crash window, including a real two-process crash; 3 notebooks | 2 h | T0 (T3 optional: one Cloud Run service) |
| [`lra-gcp`](lra-gcp/README.md) | run the full engine (pydantic only offline): fan-out/fan-in, human-in-the-loop, sagas, reflection, budgets, versioning, scheduled ticks, long-running tools, a model-chosen tool loop; Firestore, Cloud Tasks, Pub/Sub, Cloud Run, Cloud Workflows and Terraform; optional ADK 2 and Mistral paths; 6 notebooks | 6 h (8 h with the optional paths) | T0, T3 |

The topic folds three earlier lineages into one: from the Google Cloud edition (a standard-library core and an ADK-based lab) came this primer, its drills, the ADK 2 ticket-queue workflow, the scheduled and long-running-tool patterns and the two-process crash; from the Mistral edition, the Mistral adapter, the Mistral Workflows version and their tests; and the engine, its tests and its notebooks are the `lra` lineage's.

## Run it

```bash
cd lra-core
python3 -m pip install -r requirements.txt   # pytest and Jupyter's runner; core.py itself is standard library
python3 -m pytest -q                         # 13 tests, ~1 s
python3 tools/run_notebooks.py solutions     # the worked answers run clean

cd ../lra-gcp
python3 -m pip install -e ".[dev,services]"  # pydantic, FastAPI for the services, Jupyter's runner
python3 -m pytest -q                         # 73 tests: 66 pass, 7 skip without the optional extras, ~15 s
python3 scripts/local_demo.py                # fan-out -> crash -> reaper -> 3-day wait -> approval -> saga rollback
```

## How it fits

Read the agent loop first ([`agent-core`](../agent-fundamentals/agent-core/README.md), module 07.1): this topic makes
that loop survive time. The platform lab's notebook 03
([`gcp-agent-platform-lab`](../agent-fundamentals/gcp-agent-platform-lab/README.md)) covers state and checkpoints
inside one session and points here for anything that waits. Next door, layer 06
([`06-gateway`](../../06-gateway/README.md)) bounds the agent's turns and authority, and the curriculum's module 07.3
([`CURRICULUM.md`](../../CURRICULUM.md)) sets the path and the drills.

## Caveats

- At T0 the store, queue and clock are in-memory twins with the same semantics, and the model is scripted, so costs
  and latencies in the notebooks are simulated. The Google Cloud adapters are unit-tested against fake clients and the
  Terraform is validated, not applied here.
- The ADK 2 path (`pip install -e ".[adk]"`, about 220 MB) and the Mistral Workflows path (Python 3.12–3.14,
  `mistralai-workflows`) are optional and off by default; their tests and notebook 04 skip without them (measured
  2026-09-26, verify).
- Product facts (Cloud Run, Cloud Tasks, Workflows and Pub/Sub limits, ADK 2, Gemini and Mistral model names) are
  dated September 2026 and marked (verify); the primer ends with a list to re-check.
