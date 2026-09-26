# Long-running agentic workflows on Google Cloud

A primer on designing agents that run for minutes to weeks — waiting on tools, humans, the world and the clock — plus a runnable implementation of every pattern on Google Cloud (Cloud Run, Cloud Tasks, Pub/Sub, Cloud Scheduler, Cloud Workflows, Firestore, Gemini on Vertex AI, ADK 2, Agent Runtime).

Long-running agents on Google Cloud, as a primer, a reference implementation and design drills. Everything runs **offline** in ~2 seconds; the GCP backends are one environment variable away.

## Start here (5 minutes)

```bash
pip install -e ".[dev]"          # or: pip install -r <(python -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
make test                        # 30 tests: crash-after-side-effect, duplicate delivery, lease expiry, HITL, saga, fan-in, ADK resume …
jupyter lab notebooks/           # 4 worked notebooks (executed, with outputs) + 4 practice notebooks (fill in the blanks)
```

Read in this order:

1. [`../../00_primer.md`](../../00_primer.md) (the layer's shared long-running primer) — the concepts: five invariants, the run state machine, eight patterns, GCP building blocks with the limits that decide designs, three reference architectures, estimation, observability, how to explain the design.
2. `notebooks/01_durable_loop_worked.ipynb` → `02` → `03` → `04` — each pattern executed step by step with a crash injected at the worst moment.
3. `notebooks/*_practice.ipynb` — rebuild the core of each pattern yourself; `lragents.practice_checks` tells you what's wrong. Solutions in `notebooks/solutions/`.
4. `docs/02_design_drills.md` — six system-design prompts with answer sketches, eight "find the bug" snippets, rapid-fire.
5. `docs/01_gcp_cheatsheet.md` — limits, delivery semantics, CLI/SDK snippets, IAM sketch.

## What's in the box

```
src/lragents/
  core/          Run model (journal + state + version + lease + budget), stores (in-memory, Firestore),
                 LLM adapters (ScriptedLLM for tests, GeminiLLM on Vertex), tools + idempotency + fault injection,
                 transport (in-memory dispatcher, Cloud Tasks, Pub/Sub), FakeClock
  patterns/      P1 durable loop (+ async tools: poll/callback)   P2 fan-out/fan-in   P3 human-in-the-loop gate
                 P4 saga / compensations   P5 scheduled agent (lease + due time)   P6 reflection loop
  adk/           P8 ADK 2 Workflow graph with RequestInput interrupts (offline), model-backed agent with
                 LongRunningFunctionTool + before_tool_callback + App(resumability, compaction),
                 Cloud Run entry (ADK FastAPI app + /wake), Agent Runtime deploy script
  service/       FastAPI service for Cloud Run: /runs, /approve, Cloud Tasks + Pub/Sub + Scheduler + callback handlers,
                 OIDC verification in gcp mode; LRAGENTS_BACKEND=memory|gcp
  practice_checks.py   the graders used by the practice notebooks and tests/test_solutions.py
tests/           30 offline tests (pytest)
notebooks/       4 worked (executed) + 4 practice + solutions/; regenerate with tools/build_notebooks.py
docs/            GCP cheat sheet, design drills (the primer is ../../00_primer.md, shared by the layer)
infra/           terraform (Firestore, Cloud Tasks queue, Pub/Sub + DLQ + push subs, Scheduler, Workflows, SAs/IAM),
                 workflows/ (HITL callback wait, parallel fan-out/fan-in), deploy.sh (Cloud Run source deploys)
```

## The pattern → GCP map

| Pattern | Local objects | GCP objects |
|---|---|---|
| Durable loop | `DurableAgentLoop`, `InMemoryRunStore`, `InMemoryDispatcher` | Cloud Run handler ← Cloud Tasks named tasks (OIDC) · Firestore run doc + `idempotency_keys` (TTL) · Gemini |
| Async tool | `is_async` tool, `poll`, `resume_with_event` | delayed Cloud Task (`schedule_time`, back-off) or webhook → `/internal/callbacks` |
| Human gate | `approve`, `expire_stale_approvals`, `approval_link` | `/runs/{id}/approve` behind IAP · Scheduler tick for TTL · or Cloud Workflows `await_callback` |
| Fan-out/fan-in | `FanOutFanIn` (transactional counter) | Pub/Sub topic → workers → Firestore transaction → named aggregate task · or Workflows `parallel` |
| Saga | `SagaRunner`, `SagaStep` | Cloud Tasks per step; compensations journaled with keys; `on_stuck` → ticket |
| Scheduled | `ScheduledAgent` (lease + `next_due`) | Cloud Scheduler → Pub/Sub → push `/internal/scheduler/tick` |
| Reflection | `ReflectionLoop` (threshold, max_iters) | same loop machinery; critic on a separate model config |
| ADK workflow | `Workflow` + `@node(rerun_on_resume)` + `RequestInput` | Cloud Run + Cloud SQL sessions (`postgresql+asyncpg://…?host=/cloudsql/…`) + GCS artifacts; Scheduler → Pub/Sub → `/wake`; or Agent Runtime |

## Running the services

```bash
make run-local                                   # library service, in-memory backend, http://localhost:8080/docs
curl -s -XPOST localhost:8080/runs -H 'content-type: application/json' -d '{"goal":"buy ABC"}'
make run-adk-web                                 # ADK dev UI for the model-backed agent (needs Vertex ADC creds)
python -m lragents.adk.nightly_workflow          # the ADK Workflow demo, no model needed
```

Deploy (after `terraform apply` in `infra/terraform`): `make deploy` — see `infra/deploy.sh` for the two Cloud Run services and the Workflows. Environment variables are listed in `.env.example`.

## What was verified, and what wasn't

* **Verified locally:** all 30 tests; all four worked notebooks executed end to end (outputs are committed); practice notebooks fail only at their TODO cells with explicit messages; the ADK 2 Workflow demo (ADK 2.8.0) parks/resumes across four wake-ups with exactly one queue join and one purchase, routes to `abandon` on a sold-out staleness check, and reproduces the "new invocation instead of resume ⇒ second ticket" mistake.
* **Written against the docs but not run live** (needs a GCP project): `FirestoreRunStore`, `CloudTasksDispatcher`, `PubSubDispatcher`, `GeminiLLM`, the OIDC verifier, `adk/main.py` on Cloud Run, `deploy_agent_runtime.py`, the Terraform and the two Workflows YAMLs (parsed, not deployed).
* **Moving targets:** ADK 2 resumability/compaction configs are pre-GA; Gemini model ids and the Agent Runtime naming change often. The primer ends with a "verify before relying on it" list.
