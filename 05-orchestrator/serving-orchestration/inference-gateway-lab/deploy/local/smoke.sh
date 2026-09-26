#!/usr/bin/env bash
# Send the same long-prefix request three times through the router and show where each went.
# With a prefix-aware policy the 2nd and 3rd land on the replica that served the 1st.
#   ROUTER_URL (default http://localhost:9000)   DRY_RUN=1 prints the commands.
set -euo pipefail

ROUTER_URL="${ROUTER_URL:-http://localhost:9000}"
DRY_RUN="${DRY_RUN:-0}"
SYSTEM="$(printf 'You are a careful agent that plans, calls tools and verifies results. %.0s' $(seq 1 40))"
BODY="{\"model\":\"lab/llm\",\"max_tokens\":8,\"messages\":[{\"role\":\"system\",\"content\":\"${SYSTEM}\"},{\"role\":\"user\",\"content\":\"status?\"}]}"

step() { echo; echo "==> $*"; }

for i in 1 2 3; do
  step "request ${i}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "+ curl -sS -D - -o /dev/null -H 'Content-Type: application/json' -d '<body>' ${ROUTER_URL}/v1/chat/completions"
  else
    curl -sS -D - -o /dev/null -H 'Content-Type: application/json' -d "${BODY}" \
      "${ROUTER_URL}/v1/chat/completions" | grep -i -E '^(HTTP|x-gateway-destination-endpoint|x-inference-pod)'
  fi
done

step "router metrics (requests per endpoint)"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "+ curl -sS ${ROUTER_URL}/metrics | grep igw_request_total"
else
  curl -sS "${ROUTER_URL}/metrics" | grep '^igw_request_total' || true
fi
