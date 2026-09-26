# Long-Running Agents on Google Cloud (`lra`)

A primer on long-running agentic workflows and their design patterns, with a working implementation on GCP:
a durable-execution engine (Firestore + Cloud Tasks + Cloud Run + Pub/Sub + Gemini), a Cloud Workflows version of
the same flow, and an ADK 2 `Workflow` for Vertex AI Agent Engine. Everything runs locally on in-memory adapters
with the same semantics, so the crash/resume/timeout behaviour is testable in seconds.

**Start here:** [`docs/primer.md`](docs/primer.md) → `notebooks/practice/00_core_idea.ipynb` → the rest (answers in `notebooks/worked/`).

**Time and tier:** ~8 h after `lra-core` (rough); module 07.3 in [`CURRICULUM.md`](../../../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: in-memory adapters with the same semantics, no key, no cloud project. A Google Cloud project adds the optional T3 deploy (Terraform in `infra/terraform/`), billed per use; the `adk` extra adds the ADK 2 workflow.

## Quick start (no GCP needed)

```bash
pip install -e ".[dev,services]"
make test          # 34 pass, 5 skip: crash windows, duplicate delivery, leases, HITL, fan-out, saga, budgets, services
make demo          # fan-out -> crash -> reaper -> 3-day wait -> approval -> saga rollback, narrated
make notebooks     # executes the worked notebooks headlessly
```

The four GCP adapter tests (Cloud Tasks, Pub/Sub, Gemini, Firestore) drive fake clients but import the real Google
libraries, so they skip without the `gcp` extra; `pip install -e ".[dev,services,gcp]"` runs them too (38 pass; the
ADK test still skips until the `adk` extra below is installed). No credentials are needed for either.

Optional managed path (`pip install -e ".[adk]"`): `make adk-demo` runs the ADK 2 workflow, pauses at the review gate,
and resumes it from a **new process**.

## What's in the box

| Path | What |
|---|---|
| `docs/primer.md` | The primer: failure model, the core idea in 40 lines, 13 patterns with GCP mappings and trade-offs, reference architecture, engine vs Workflows vs Agent Engine, scale/cost, security, testing |
| `docs/code-evaluation-drills.md` | Six find-the-bug snippets (each a real defect found while building this) + design-round prompts |
| `docs/runbook.md` | Deploy/rollback, dashboards, common incidents, manual operations |
| `src/lra/core/` | Engine (`engine.py`), domain model (`models.py`), workflow DSL (`workflow.py`), ports (`ports.py`) |
| `src/lra/adapters/memory/` | In-memory store/queue/bus, scripted LLM, controllable clock, `LocalRunner` (drives tasks like Cloud Tasks, including redelivery after a crash) |
| `src/lra/adapters/gcp/` | `FirestoreStateStore`, `CloudTasksQueue`, `PubSubEventBus`, `GeminiLLM`, `build_gcp_engine` |
| `src/lra/patterns/` | `hitl`, `reflection`, `orchestrator_worker`, `saga` |
| `src/lra/examples/` | `research_pipeline` (uses every pattern), `procurement_saga`, scripted model routes |
| `services/` | Cloud Run `api` (start/inspect/events/cancel) and `worker` (Cloud Tasks target, reaper, Pub/Sub push), one Dockerfile |
| `workflows/research_approval.yaml` | Same flow in Cloud Workflows: `parallel`, `retry`, `create_callback_endpoint`/`await_callback`, try/except compensation |
| `examples/adk_agent_engine/` | ADK 2 `Workflow` (parallel workers, routing loop, interrupt/resume) + Agent Engine deploy wrapper |
| `infra/terraform/` | Firestore, Cloud Tasks queue, Pub/Sub (+DLQ, optional BigQuery sink), Cloud Run ×2, Scheduler reaper, Workflows, IAM |
| `notebooks/worked/`, `notebooks/practice/` | Four pairs: core idea from scratch, durable execution, HITL, fan-out/saga/reflection/budgets. Practice = same notebook with `____` blanks; the worked one is the solution |
| `tests/` | Chaos-hook tests for each crash window, HITL, fan-out/saga/budget/versioning, services, adapters with fake clients, ADK flow |

## The engine in one picture

```
client ─▶ Cloud Run api ─▶ Firestore agent_runs (checkpoints, leases, version)
                │ enqueue (run, step, attempt)              ▲
                ▼                                            │ CAS
          Cloud Tasks ──OIDC──▶ Cloud Run worker ──▶ Gemini / tools
                ▲                     │ progress ─▶ Pub/Sub agent-events ─▶ BigQuery / UI / DLQ
        Cloud Scheduler ─ reap ───────┘
```

Three invariants (primer §2): **checkpoint before enqueue**; **`(run, step, attempt)` identifies work**; **effects are recorded
before the checkpoint, keyed by intent**. The tests in `tests/test_durability.py` kill the worker at each window and assert
exactly one recovery and zero repeated side effects.

## Writing a workflow

```python
from lra import Workflow, Next, Done, Wait, Budget
from lra.patterns.hitl import request_approval, approval_decision

wf = Workflow("expense", version="1", default_budget=Budget(max_steps=20, max_cost_usd=1.0))

@wf.step(start=True)
def draft(ctx):
    ctx.state["summary"] = ctx.llm("Summarise: " + ctx.input["text"]).text     # budgeted model call
    return Next("gate")

@wf.step()
def gate(ctx):
    return request_approval(ctx, then="pay", gate="manager", timeout=timedelta(days=3))   # sleeps for free

@wf.step(compensate=lambda ctx: ctx.effect("refund", refund_fn))
def pay(ctx):
    approval_decision(ctx, gate="manager")                                          # raises on reject/timeout
    rec = ctx.effect("pay", lambda: payments.charge(...))                           # at most once, ever
    return Done({"tx": rec["id"]})
```

Register it in `lra.examples.ALL_WORKFLOWS` (or pass it to `Engine(workflows=[...])`), then `POST /runs {"workflow": "expense", ...}`.

## Deploying to GCP

```bash
export PROJECT_ID=... REGION=asia-southeast1
./scripts/deploy.sh                       # Cloud Build image + terraform apply (see infra/terraform/variables.tf)
gcloud workflows run research-approval --data='{"goal":"...","subtopics":["a","b"]}'
```

Set `LRA_BACKEND=gcp` and the variables in `.env.example`; both services read them. Pin `LRA_GEMINI_MODEL` to a model
enabled in your project and set real prices in `GeminiLLM` before trusting `cost_usd`.

## Status and caveats

- Engine, adapters, services, patterns, examples, notebooks: tested locally (39 tests, 4 executed notebooks).
- GCP adapters are unit-tested against fake clients; Terraform is written but not applied here — review names, quotas and
  org policies before `terraform apply`.
- ADK/Agent Engine code was verified against `google-adk 2.8` and `vertexai 2.1`; the Agent Engine deploy surface moves
  between releases, so confirm `AdkApp`/`agent_engines.create` against current docs.
