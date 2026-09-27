#!/usr/bin/env bash
# Start the local stack: the memory service (:8080), a fake OpenAI-compatible server (:8000) and, with
# --pgvector, Postgres + pgvector (:5432). Prints each step; DRY_RUN=1 prints the commands only.
# Without a reachable Docker daemon it prints the commands and exits 0 (the notebooks then use their T0 paths).
#
#   ./up.sh                # memory service + fake model server
#   ./up.sh --pgvector     # ... plus pgvector/pgvector:0.8.6-pg17
#   DRY_RUN=1 ./up.sh --pgvector
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"
PROFILE=()
if [[ "${1:-}" == "--pgvector" ]]; then PROFILE=(--profile pgvector); fi

run() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi
}

echo ">> 1/4 is Docker here?"
if [[ "${DRY_RUN}" != "1" ]] && ! docker info >/dev/null 2>&1; then
  echo "   no reachable Docker daemon: printing the commands instead (the notebooks run their T0 paths)"
  DRY_RUN=1
fi

echo ">> 2/4 build the image and start the stack"
run docker compose -f "${HERE}/compose.yaml" ${PROFILE[@]+"${PROFILE[@]}"} up -d --build

echo ">> 3/4 wait until the memory service answers /healthz"
if [[ "${DRY_RUN}" != "1" ]]; then
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8080/healthz >/dev/null 2>&1; then break; fi
    sleep 2
  done
  curl -fsS http://127.0.0.1:8080/healthz && echo
else
  echo "+ curl -fsS http://127.0.0.1:8080/healthz"
fi

echo ">> 4/4 point the lab at it"
echo "   export MEMLAB_TOKEN_KEY=local-dev-only-change-me     # the compose default; mint tokens with: python -m memlab token --tenant acme --user u1"
echo "   export MEMLAB_LLM_URL=http://127.0.0.1:8000          # the fake server (simulated); a real vLLM makes it T1"
if [[ ${#PROFILE[@]} -gt 0 ]]; then
  echo "   pip install \"psycopg[binary]>=3.1\""
  echo "   export MEMLAB_PG_DSN=postgresql://memlab:memlab-local-only@127.0.0.1:5432/memlab"
fi
echo "   stop and delete the data with: ${HERE}/down.sh"
