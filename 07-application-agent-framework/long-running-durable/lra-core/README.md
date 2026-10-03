# lra-core — the durable agent loop in one standard-library file

After this lab, you can explain how an agent workflow stays correct through worker crashes, duplicate deliveries and
multi-day waits. You can also show the mechanism in ~250 lines. The design needs three invariants, a lease and a reaper, and
nothing else.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) (10 min). It gives the concept in two pages. The topic primer is
   [`../PRIMER.md`](../PRIMER.md).
2. Run `python3 workflow.py`. The script starts a run, and the run waits at review. Then the script lets "two days"
   pass, approves the run and publishes. Then run `python3 workflow.py kill`. After that, run
   `python3 workflow.py resume`. The first process publishes and crashes before its checkpoint (exit 137). The second
   process finds the run on disk. It completes the run and does not publish again.
3. Open [`notebooks/01_durable_loop.ipynb`](notebooks/01_durable_loop.ipynb). Fill in the blanks.

## What you get

*T0 is a laptop or a Colab CPU, at no cost. At T0, the lab needs only the standard library and no key. T3 is a Google Cloud
project. It is optional, and you pay for each use.* Time: approximately 2 h (a rough value). This lab is in module 07.3 of
[`CURRICULUM.md`](../../../CURRICULUM.md), before [`lra-gcp`](../lra-gcp/README.md).

| Path | What it teaches | Time | Tier |
|---|---|---|---|
| `core.py` | `Store` (and `FileStore`, the same store on one JSON file), `Queue` (named tasks with a due time), `Ctx`, `Engine` and the reaper. `Engine` does these operations in this sequence: the guard, the lease, the step, the checkpoint and the enqueue. The file has ~250 lines. | 30 min | T0 |
| `workflow.py` | The workflow has four steps in sequence: draft, review, publish and notify. Review waits for a person. Publish is an idempotent effect. The file also has the `kill` / `resume` pair. | 10 min | T0 |
| `test_core.py` | 13 tests. There is one test for each claim in the docstring of `core.py`. Other tests examine the crash windows of the reaper and a real crash across two processes. | 15 min | T0 |
| [`notebooks/`](notebooks/) | There are three notebooks: the durable loop, the wait and the resume, and retries and the reaper (with the two-process crash). Each blank stops at its first exercise. The worked answers have the same names in [`solutions/`](solutions/). | 1 h | T0 |
| `gcp/` | The same `Engine` on Firestore, Cloud Tasks and Cloud Run, with Cloud Scheduler for `/reap`. The code has ~125 lines. The folder also has `deploy.sh`. | 20 min | T3 |

## Run it

```bash
python3 -m pip install -r requirements.txt   # pytest and Jupyter's runner; core.py and workflow.py need nothing
python3 -m pytest -q                         # 13 tests, ~1 s
python3 tools/run_notebooks.py solutions                  # the worked answers run clean
python3 tools/run_notebooks.py notebooks --expect-fail    # each blank stops at its first exercise
jupyter lab notebooks/                       # fill in the blanks; the asserts tell you when you are right
```

Read the code in this sequence:

1. `core.py`, from top to bottom. The docstring gives the three invariants. `Engine.execute` is the whole idea.
2. `workflow.py`. Look at `("wait", key, then)` and `ctx.effect(...)`.
3. `test_core.py`. Each test makes the worker crash at a different time and examines exactly one recovery.
4. `gcp/main.py`. Only the two adapters are new. The engine has no changes.

Deploy the one-service version to Google Cloud (optional, T3):

```bash
export PROJECT=your-project LOCATION=asia-southeast1
cd gcp && ./deploy.sh              # ~2 minutes; prints curl commands to start and approve a run
```

## Caveats

- The store and the queue are in-memory twins of Firestore and Cloud Tasks. `FileStore` keeps its data in one JSON
  file. One JSON file is sufficient for two processes on one machine, but not for concurrency. You can control the
  clock. Thus "two days" pass in one line.
- The next step is [`lra-gcp`](../lra-gcp/README.md). It adds fan-out into child runs, saga compensation, budgets and
  deadlines, workflow versions, scheduled ticks and long-running tools. It also adds Terraform, Cloud Workflows, and
  optional ADK 2 and Mistral paths. Everything there is this loop plus those features. The three invariants do not
  change.
