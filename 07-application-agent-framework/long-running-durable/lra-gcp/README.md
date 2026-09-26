# lra-gcp — a durable-execution engine for long-running agents, run locally and on Google Cloud

After this lab you can run, break and repair a long-running agent end to end: child runs that fan out and fan in, a
human gate that sleeps for days, sagas that undo themselves, budgets that fail closed, scheduled ticks, slow tools,
a loop in which the model chooses the next tool, and the same engine on Firestore, Cloud Tasks, Pub/Sub and Cloud Run.

## Start here

1. Read the topic primer [`../PRIMER.md`](../PRIMER.md) §2–§4 (40 min), then this lab's design notes
   [`docs/primer.md`](docs/primer.md) §2–§3 (the engine's three invariants and its pattern catalogue).
2. `python3 -m pip install -e ".[dev,services]" && python3 scripts/local_demo.py` — fan-out → crash → reaper →
   3-day wait → approval → saga rollback, narrated, in a few seconds.
3. Open [`notebooks/00_core_idea.ipynb`](notebooks/00_core_idea.ipynb) and work through 00 → 05; the worked answers
   are in [`solutions/`](solutions/) under the same names.

## What you get

*T0 = a laptop or Colab CPU, free: in-memory adapters with the same semantics, a scripted model, no key, no cloud
project. T3 = the Google Cloud deployment (Terraform in `infra/terraform/`), optional, billed per use.* Time: about
8 h after [`lra-core`](../lra-core/README.md) (rough); module 07.3 in [`CURRICULUM.md`](../../../CURRICULUM.md).
Each blank in `notebooks/` stops at its first exercise until you fill it in.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| `00_core_idea` | build the store, the named-task queue and the worker from scratch, then break them with a crash and a duplicate delivery | 45 min | T0 |
| `01_durable_execution` | drive the engine one task at a time: explicit retries, chaos hooks at each crash window, leases between two workers | 1 h | T0 |
| `02_human_in_the_loop` | suspend for days, resume idempotently by key, reject, time out through the reaper, auto-approve, cancel | 45 min | T0 |
| `03_fanout_saga_reflection` | fan out into child runs, treat partial failure as data, write the fan-in counter, compensate a saga in reverse, bound a reflection loop, fail closed on budget and deadline | 1 h | T0 |
| `04_adk_workflow` | run the same shapes on ADK 2's `Workflow` (interrupts, resume, routing, the staleness guard, the new-invocation mistake) and build the graph yourself | 45 min | T0 with the `adk` extra |
| `05_tool_loop_and_mistral` | journal a model's decision before acting on it, approve exactly what runs, then swap in Mistral (function calling, the `LLM` port) and see the loop on Mistral Workflows | 1 h | T0 |
| [`docs/`](docs/) | [`primer.md`](docs/primer.md) (the engine's design notes: 13 patterns with GCP mappings, reference architecture, scale and cost, security, testing), [`code-evaluation-drills.md`](docs/code-evaluation-drills.md), [`runbook.md`](docs/runbook.md), [`gcp-cheatsheet.md`](docs/gcp-cheatsheet.md) (delivery semantics, limits, CLI and SDK snippets, IAM), [`mistral.md`](docs/mistral.md) | 2 h | T0 |
| `src/lra/` | `core/` (engine, models, workflow DSL, ports), `adapters/memory`, `adapters/gcp` (Firestore, Cloud Tasks, Pub/Sub, Gemini), `adapters/mistral` (optional), `patterns/` (hitl, reflection, orchestrator_worker, saga, scheduled, async_tool), `examples/` (research pipeline, procurement saga, tool agent) | — | T0 |
| `services/`, `workflows/`, `infra/terraform/` | Cloud Run `api` and `worker`, the same flow in Cloud Workflows (`parallel`, callbacks, compensation), Terraform for all of it | 1 h to read; deploy ~15 min | T3 |
| `examples/adk_agent_engine/`, `examples/adk_ticket_queue/` | ADK 2 `Workflow` graphs, a model-backed agent with a long-running tool, a Cloud Run `/wake` endpoint, an Agent Engine deploy | 1 h | T0 with the `adk` extra, T3 |

## Run it

```bash
python3 -m pip install -e ".[dev,services]"
python3 -m pytest -q          # 70 tests: 63 pass, 7 skip (Google Cloud clients, ADK 2, Mistral Workflows), ~15 s
python3 scripts/local_demo.py
make notebooks                # the solutions run clean; each blank stops at its first exercise
```

The optional extras add paths, never requirements: `pip install -e ".[dev,services,gcp]"` runs the four Google Cloud
adapter tests against fake clients (no credentials); `".[adk]"` (ADK 2 and the Google Cloud clients, about 220 MB,
measured 2026-09-26, verify) runs the two ADK test files, notebook 04 and `make adk-demo`; `".[mistral]"` adds the
`mistralai` SDK and, on Python 3.12–3.14, Mistral Workflows (see [`docs/mistral.md`](docs/mistral.md)), whose four
tests start a local Temporal dev server downloaded from `temporal.download` on first use. With every extra (`gcp`, `adk`,
`mistral`) on Python 3.12 and that download reachable, 77 tests run.

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

## Caveats

- Engine, adapters, services, patterns, examples and notebooks are tested locally; the Google Cloud adapters are
  unit-tested against fake clients, and the Terraform is written and validated but not applied here: review names,
  quotas and org policies before `terraform apply`.
- The ADK and Agent Engine code was verified against `google-adk` 2.10 offline (the ticket-queue graph and its
  grader) and 2.8 for the Agent Engine example; the Agent Engine deploy surface moves between releases, so confirm
  `AdkApp` and `agent_engines.create` against current docs (verify). `examples/adk_ticket_queue/main.py` and
  `agent.py` are written against the ADK docs and not run live here.
- Costs in the notebooks are simulated (illustrative per-token prices in `FakeLLM`, `GeminiLLM` and `MistralLLM`); pin
  real prices before trusting `cost_usd`.
