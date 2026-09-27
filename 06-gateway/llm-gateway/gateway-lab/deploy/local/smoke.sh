#!/usr/bin/env bash
# Issue a virtual key, send ten streamed requests through the gateway and show which provider served each:
# acme fails 30 % of requests before the first byte, so some are served by bolt (the fallback) -- and a few
# consecutive failures open acme's breaker. Then the gateway's own counters.
#   GATEWAY_URL (default http://localhost:8080)  GWLAB_ADMIN_TOKEN (default dev-admin-token)  DRY_RUN=1 prints only.
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
ADMIN="${GWLAB_ADMIN_TOKEN:-dev-admin-token}"
DRY_RUN="${DRY_RUN:-0}"

step() { echo; echo "==> $*"; }

step "issue a virtual key for tenant team-a (the secret is shown once; the gateway keeps its SHA-256)"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "+ curl -sS -X POST ${GATEWAY_URL}/admin/keys -H 'Authorization: Bearer <admin>' -d '{\"tenant\":\"team-a\"}'"
  KEY="gwk_example"
else
  KEY="$(curl -sS -X POST "${GATEWAY_URL}/admin/keys" -H "Authorization: Bearer ${ADMIN}" \
         -H 'Content-Type: application/json' -d '{"tenant":"team-a"}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])')"
  echo "key: ${KEY:0:8}..."
fi

BODY='{"model":"chat","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"Say hello"}]}'
for i in $(seq 1 10); do
  step "request ${i}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ curl -sS -D - -o /dev/null -H 'Authorization: Bearer <key>' -d '<body>' ${GATEWAY_URL}/v1/chat/completions"
  else
    curl -sS -D - -o /dev/null -H "Authorization: Bearer ${KEY}" -H 'Content-Type: application/json' -d "${BODY}" \
      "${GATEWAY_URL}/v1/chat/completions" | grep -i -E '^(HTTP|x-gwlab-target|x-gwlab-attempts)' || true
  fi
done

step "the gateway's counters"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "+ curl -sS ${GATEWAY_URL}/metrics | grep -E '^gwlab_(attempts|requests|cost_usd)_total'"
else
  curl -sS "${GATEWAY_URL}/metrics" | grep -E '^gwlab_(attempts|requests|cost_usd)_total' || true
fi
