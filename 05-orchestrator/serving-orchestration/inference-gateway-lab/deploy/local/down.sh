#!/usr/bin/env bash
# Stop and remove the local stack (both profiles) and the image it built. DRY_RUN=1 prints the commands.
# `down --rmi local` would keep igwlab:local (it only removes images WITHOUT a custom tag, and every lab
# service sets `image: igwlab:local`), so the image is removed explicitly. Pulled images
# (llm-d-inference-sim, Prometheus) are kept; remove them with `docker image rm` if you want the space back.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${DRY_RUN:-0}"
run() { echo "+ $*"; if [[ "${DRY_RUN}" != "1" ]]; then "$@"; fi; }
echo "==> removing containers and networks"
run docker compose -f "${HERE}/docker-compose.yaml" --profile sim --profile fake down
echo "==> removing the locally built image igwlab:local"
run docker image rm igwlab:local || echo "(igwlab:local not present)"
