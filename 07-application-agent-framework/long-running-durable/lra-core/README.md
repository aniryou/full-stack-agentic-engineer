# lra-core — the durable agent loop in one standard-library file

After this you can explain, and show in ~250 lines, how an agent workflow survives worker crashes, duplicate
deliveries and multi-day waits: three invariants, a lease and a reaper, and nothing else.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) (10 min): the concept in two pages; the topic primer is [`../PRIMER.md`](../PRIMER.md).
2. `python3 workflow.py` — start a run, park it at review, let "two days" pass, approve, publish. Then
   `python3 workflow.py kill` followed by `python3 workflow.py resume`: the first process publishes and dies before its
   checkpoint (exit 137); the second finds the run on disk and finishes it without publishing again.
3. Open [`notebooks/01_durable_loop.ipynb`](notebooks/01_durable_loop.ipynb) and fill in the blanks.

## What you get

*T0 = a laptop or Colab CPU, free: standard library only, no key. T3 = a Google Cloud project, optional, billed per
use.* Time: about 2 h (rough); module 07.3 in [`CURRICULUM.md`](../../../CURRICULUM.md), before [`lra-gcp`](../lra-gcp/README.md).

| Path | What it teaches | Time | Tier |
|---|---|---|---|
| `core.py` | `Store` (and `FileStore`, the same store on one JSON file), `Queue` (named tasks with a due time), `Ctx`, `Engine` (guard → lease → step → checkpoint → enqueue), the reaper; ~250 lines | 30 min | T0 |
| `workflow.py` | draft → review (wait for a human) → publish (an idempotent effect) → notify; the `kill` / `resume` pair | 10 min | T0 |
| `test_core.py` | 13 tests: one per claim in `core.py`'s docstring, the reaper's crash windows, and a real crash across two processes | 15 min | T0 |
| [`notebooks/`](notebooks/) | three notebooks: the durable loop, waiting and resuming, retries and the reaper (with the two-process crash); each blank stops at its first exercise; worked answers under the same names in [`solutions/`](solutions/) | 1 h | T0 |
| `gcp/` | the same `Engine` on Firestore + Cloud Tasks + Cloud Run (+ Cloud Scheduler for `/reap`), ~125 lines, and `deploy.sh` | 20 min | T3 |

## Run it

```bash
python3 -m pip install -r requirements.txt   # pytest and Jupyter's runner; core.py and workflow.py need nothing
python3 -m pytest -q                         # 13 tests, ~1 s
python3 tools/run_notebooks.py solutions                  # the worked answers run clean
python3 tools/run_notebooks.py notebooks --expect-fail    # each blank stops at its first exercise
jupyter lab notebooks/                       # fill in the blanks; the asserts tell you when you are right
```

Read the code in this order: `core.py` top to bottom (the docstring states the three invariants; `Engine.execute` is
the whole idea), then `workflow.py` (note `("wait", key, then)` and `ctx.effect(...)`), then `test_core.py` (each test
kills the worker at a different moment and checks exactly one recovery), then `gcp/main.py` (only the two adapters
are new; the engine is untouched).

Deploy the one-service version to Google Cloud (optional, T3):

```bash
export PROJECT=your-project LOCATION=asia-southeast1
cd gcp && ./deploy.sh              # ~2 minutes; prints curl commands to start and approve a run
```

## Caveats

- The store and queue are in-memory twins of Firestore and Cloud Tasks (`FileStore` persists to one JSON file, which
  is enough for two processes on one machine, not for concurrency); the clock is controllable, so "two days" pass in
  one line.
- The step-up is [`lra-gcp`](../lra-gcp/README.md): fan-out into child runs, saga compensation, budgets and deadlines,
  workflow versioning, scheduled ticks, long-running tools, Terraform, Cloud Workflows, and optional ADK 2 and Mistral
  paths. Everything there is this loop plus those features; the three invariants do not change.
