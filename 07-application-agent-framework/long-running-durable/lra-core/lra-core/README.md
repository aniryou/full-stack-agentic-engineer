# lra-core — long-running agents, the core concept

~200 lines that show how an agent workflow survives crashes, duplicate deliveries and multi-day waits,
plus the same loop deployed to Google Cloud in one service. The full-featured version (fan-out,
sagas, budgets, Terraform, ADK) is the step-up; this is the part to understand first.

```
core.py          the loop: Store, Queue, Ctx, Engine (guard -> lease -> step -> checkpoint -> enqueue), reaper   ~220 lines
workflow.py      draft -> review (wait for a human) -> publish (idempotent effect) -> notify                     ~60 lines
test_core.py     twelve tests: one per claim in core.py's docstring, plus the crash windows
gcp/main.py      same Engine on Firestore + Cloud Tasks + Cloud Run (+ Scheduler for /reap)                     ~125 lines
gcp/deploy.sh    gcloud commands: enable APIs, Firestore, queue, deploy, scheduler
notebooks/       three fill-in-the-blank notebooks (solutions/ has the answers, executed)
PRIMER.md        the concept in two pages
```

**Time and tier:** ~2 h (rough; ~10 h with `lra-gcp`); module 07.3 in [`CURRICULUM.md`](../../../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: standard library only, no key. A Google Cloud project adds the one-service deploy in `gcp/` (T3, optional, billed per use).

## Run it

```bash
python workflow.py                 # start a run, park it at review, "two days" pass, approve, publish
python -m pytest -q                # 12 tests: stale guard, retries, crash windows, orphans, resume, timeout, lease
jupyter lab notebooks/             # fill in the blanks; asserts tell you when you're right
```

No dependencies for the core (stdlib only). `pytest` and `jupyter` for tests/notebooks.

## Read it in this order

1. `core.py` top to bottom — the docstring states the three invariants; `Engine.execute` is the whole idea.
2. `workflow.py` — what a step looks like; note `("wait", key, then)` and `ctx.effect(...)`.
3. `test_core.py` — each test kills the worker at a different moment and checks exactly one recovery.
4. `gcp/main.py` — only the two adapters are new; the engine is untouched.

## Deploy to GCP

```bash
export PROJECT=your-project LOCATION=asia-southeast1
cd gcp && ./deploy.sh              # ~2 minutes; prints curl commands to start and approve a run
```

## The step-up

When you need fan-out into child runs, saga compensation, budgets/deadlines, workflow versioning,
Terraform, Cloud Workflows and the ADK/Agent Engine path, use the full `lra-gcp` repo. Everything
there is this loop plus those features; the three invariants don't change.
