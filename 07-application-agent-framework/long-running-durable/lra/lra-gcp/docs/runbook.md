# Runbook — operating long-running agent runs

## Deploy / rollback

```bash
export PROJECT_ID=... REGION=asia-southeast1
./scripts/deploy.sh                      # Cloud Build image + terraform apply
gcloud run services update-traffic lra-worker --to-revisions=PREV=100 --region $REGION   # rollback
```

Before removing a workflow version from the image: `agent_runs where workflow_version == X and status in (PENDING, RUNNING, WAITING, COMPENSATING)` must be empty, or those runs will fail with "no longer deployed".

## Dashboards / alerts (from the `agent-events` stream)

| Signal | Query | Page when |
|---|---|---|
| Stuck runs | `status == WAITING and wait.timeout_at < now` | > 0 for 10 min (reaper not running?) |
| Dying workers | `reap.leases_recovered` per hour | rising trend |
| Retry storm | `step.retry` per step per 10 min | > 5 % of steps |
| Money at risk | `run.compensation_failed` on `agent-alerts` | any |
| Cost | `run.succeeded` cost_usd p95 | > budget × 0.8 |

## Common incidents

**Runs pile up in RUNNING with expired leases.** Worker is crashing or timing out. Check Cloud Run logs for the run ids; raise `--timeout`, lower `--concurrency`, or split the step. The reaper will re-drive once fixed.

**A run is WAITING but the reviewer says they approved.** `GET /runs/{id}` → check `wait.key`; the approval was probably POSTed with a different key (wrong gate) → `applied:false` in the API log. Re-POST with the right key; duplicates are harmless.

**Run FAILED with "compensation of X failed".** Manual undo required. The effect record in `agent_effects/{run}:X` has the external id. After fixing, set `status=COMPENSATED` via the admin path (or re-enqueue the compensation) and record the manual action in the run's history.

**Budget-failed runs after a model price change.** Update `pricing_per_1m` in `GeminiLLM`; historical `cost_usd` is not recomputed.

**Cloud Tasks "AlreadyExists" spam in logs.** Expected during recovery (reaper re-enqueue vs in-flight retry). Only investigate if a run is not progressing.

## Manual operations

```bash
# resume a run by hand (e.g. approval came by e-mail)
curl -X POST $API/runs/$RUN/events -H 'Content-Type: application/json' \
  -d '{"key":"editor:'$RUN'","payload":{"decision":"approve","by":"ops"}}'

# force the reaper
curl -X POST $WORKER/internal/reap -H "Authorization: Bearer $(gcloud auth print-identity-token --audiences=$WORKER)"
```
