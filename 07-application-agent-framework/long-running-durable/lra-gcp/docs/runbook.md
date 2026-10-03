# Runbook — operating long-running agent runs

## Deploy / rollback

```bash
export PROJECT_ID=... REGION=asia-southeast1
./scripts/deploy.sh                      # Cloud Build image + terraform apply
gcloud run services update-traffic lra-worker --to-revisions=PREV=100 --region $REGION   # rollback
```

Before you remove a workflow version from the image, make sure that this query gives no result:
`agent_runs where workflow_version == X and status in (PENDING, RUNNING, WAITING, COMPENSATING)`. If it gives a
result, those runs will fail with "no longer deployed".

## Dashboards / alerts (from the `agent-events` stream)

| Signal | Query | Page when |
|---|---|---|
| Stuck runs | `status == WAITING and wait.timeout_at < now` | > 0 for 10 min. Possible cause: the reaper does not run. |
| Workers that crash | `reap.leases_recovered` per hour | The trend increases. |
| Retry storm | `step.retry` per step per 10 min | > 5 % of steps. |
| Money at risk | `run.compensation_failed` on `agent-alerts` | Any event. |
| Cost | `run.succeeded` cost_usd p95 | > budget × 0.8. |

## Common incidents

**Runs collect in RUNNING with expired leases.** The worker crashes or stops at its timeout. Examine the Cloud Run
logs for the run ids. Increase `--timeout`, decrease `--concurrency`, or divide the step. When you repair the worker,
the reaper re-drives the runs.

**A run is WAITING, but the reviewer says that they approved it.** Examine `wait.key` in the response to `GET /runs/{id}`. It is
probable that the approval came in a POST with a different key (an incorrect gate). If so, the API log shows
`applied:false`. Send the POST again with the correct key. A duplicate POST causes no damage.

**A run goes to FAILED with "compensation of X failed".** Reverse the effect by hand. The effect record in
`agent_effects/{run}:X` has the external id. After you repair the problem, set `status=COMPENSATED` through the admin
path, or enqueue the compensation again. Then record the manual action in the history of the run.

**Runs fail on their budget after the price of a model changes.** Change `pricing_per_1m` in `GeminiLLM`. The old
`cost_usd` values stay as they are, because the engine does not calculate them again.

**The logs show many Cloud Tasks "AlreadyExists" messages.** These messages are normal during a recovery. They
occur when the reaper enqueues a task again while a retry of the same task is in flight. If a run does not continue, examine the messages. If all the runs continue, no action is necessary.

## Manual operations

```bash
# resume a run by hand (e.g. approval came by e-mail)
curl -X POST $API/runs/$RUN/events -H 'Content-Type: application/json' \
  -d '{"key":"editor:'$RUN'","payload":{"decision":"approve","by":"ops"}}'

# force the reaper
curl -X POST $WORKER/internal/reap -H "Authorization: Bearer $(gcloud auth print-identity-token --audiences=$WORKER)"
```
