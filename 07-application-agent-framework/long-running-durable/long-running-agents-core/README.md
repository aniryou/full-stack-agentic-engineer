# Long-running agents — the core idea

One file, stdlib only, ~170 lines of code: `durable.py`.

> **A long-running agent is not a long-running process.**
> It wakes up, does *one* step, checkpoints, and goes back to sleep. The store is the only memory. A queue delivers wake-ups (at-least-once). Waiting for a human, a slow job or the world means **parking** the run — nothing runs, nothing costs — until something calls `resume()`.

Five rules make it safe (marked `## (n)` in the code):

| # | Rule | Where in `durable.py` |
|---|---|---|
| 1 | **Durable state** — every step ends in `save()`; after a crash only the store is real | `Store.save` |
| 2 | **Intent → act** — journal the call *with an idempotency key*, save, *then* act. A retry repeats the same call with the same key and never re-asks the model (the model is non-deterministic) | `Agent._decide` (checkpoint 1), `Agent._execute` (checkpoint 2) |
| 3 | **Lease** — one worker advances a run at a time; leases *expire*, so a dead worker can't wedge a run | `Store.acquire_lease` |
| 4 | **Budget** — a hard step limit in code; prompts can't enforce limits | `Agent._decide` |
| 5 | **Park, don't wait** — a wait is a status + a token, not a sleeping process | `Agent._park`, `Agent.resume` |

**Time and tier:** 1–2 h (rough); module 07.3, an alternative core in [`CURRICULUM.md`](../../../CURRICULUM.md). T0 = a laptop or Colab CPU, free: standard library only, no key.

## Run it

```bash
python demo.py            # happy path · crash after the charge + retry from another worker · human gate · slow tool
python demo.py kill       # charges the card, then dies (exit 137) before the checkpoint
python demo.py resume     # a *different process* finds the run in runs.json and finishes it — one charge
python tests/test_core.py # 10 tests, no dependencies (or: python -m pytest tests -q)
jupyter lab notebooks/    # 01_worked.ipynb (executed) · 02_practice.ipynb (re-implement the 4 methods; graded by the tests)
```

## The state machine

```
RUNNING ──step──▶ RUNNING ──final──▶ DONE
   │                 │
   │  tool needs     │  tool returns Wait(token)        budget hit ──▶ FAILED
   │  approval       ▼
   └───────────▶ WAITING ──resume(token, payload)──▶ RUNNING
```

A run's journal is append-only: `decision → intent(key, done=False) → … → intent(done=True, result)`. An intent that is journaled but not done is exactly "work that may or may not have happened" — the retry re-executes it under the same key, and the downstream API de-duplicates.

## Where it goes on GCP

| Here | There |
|---|---|
| `Store` (JSON file) | Firestore, one document per run; `acquire_lease` is a transaction (compare-and-set) |
| `Queue` (named, at-least-once, retry on raise) | Cloud Tasks: named tasks, OIDC-authenticated HTTP target, `schedule_time` for delayed polls |
| `agent.handle(msg)` | `POST /internal/tasks/step` on Cloud Run — return 2xx only after a durable commit; 429 on `LeaseHeld` |
| `resume()` | an HTTP endpoint hit by a person (behind IAP), a webhook, or Cloud Scheduler → Pub/Sub |
| `FakeModel` | Gemini on Vertex AI, given the journal as context |
| `PaymentAPI(key)` | any API that honours an `Idempotency-Key` header |
| `FakeClock.advance(61)` | the lease TTL passing |

## The step-up

`long-running-agents-gcp` is the same five rules with the machinery: Firestore/Cloud Tasks/Pub/Sub backends, fan-out/fan-in, sagas, scheduled agents, reflection loops, ADK 2 `Workflow` with interrupts, a Cloud Run service, Terraform, the primer and the design drills. Read this repo first; open that one when you want to see how each rule survives contact with real services.
