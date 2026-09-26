#!/usr/bin/env bash
# Run one piece of Python in a hardened container, through the lab's wrapper (budgets + one result line).
# The same flags as sandboxlab/docker.py (DockerSandbox.hardened); a test keeps the two in step.
#   echo 'print(1)' | deploy/docker/run-hardened.sh
#   deploy/docker/run-hardened.sh my_script.py
#   RUNTIME=runsc deploy/docker/run-hardened.sh my_script.py     # gVisor (after install-gvisor.sh)
#   DRY_RUN=1 deploy/docker/run-hardened.sh my_script.py          # print the command only
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
# shellcheck source=../versions.env
. "$HERE/../versions.env"

RUNTIME="${RUNTIME:-}"            # empty = runc; runsc = gVisor
NETWORK="${NETWORK:-none}"        # none, or an --internal network (see run-with-proxy.sh for the socket variant)
BUDGETS="${BUDGETS:-{\"cpu_s\":2.0,\"wall_s\":5.0,\"memory_mib\":256,\"pids\":32,\"file_mib\":8,\"nofile\":64,\"output_bytes\":65536,\"output_kill_bytes\":1048576\}}"
need docker "Install Docker Engine or Docker Desktop."

args=(docker run --rm -i --label app=sandboxlab)
[[ -n "$RUNTIME" ]] && args+=(--runtime "$RUNTIME")
args+=(--network "$NETWORK" --read-only --cap-drop ALL
  --security-opt no-new-privileges --security-opt "seccomp=$HERE/seccomp-sandbox.json"
  --pids-limit 64 --memory 320m --memory-swap 320m --cpus 1 --user 65534:65534
  --tmpfs /work:rw,nosuid,nodev,noexec,size=16m,mode=1777 --tmpfs /tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777
  --ulimit nofile=256:256 --ulimit core=0 --init --hostname sandbox --workdir /work)
[[ -n "${SANDBOX_PROXY_SOCKET:-}" ]] && args+=(--volume "$SANDBOX_PROXY_SOCKET:/run/egress/proxy.sock"
  --env SANDBOX_PROXY_URL=unix:/run/egress/proxy.sock)
args+=(--volume "$LAB/sandboxlab/wrapper.py:/opt/sandbox/wrapper.py:ro" "$SANDBOX_IMAGE"
  python3 -I /opt/sandbox/wrapper.py --stdin --workspace /work --budgets "$BUDGETS" --pass-env SANDBOX_PROXY_URL)
[[ "$RUNTIME" == "runsc" ]] && args+=(--nproc)

log "hardened run (${RUNTIME:-runc}, network $NETWORK)"
if [[ $# -ge 1 ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then run "${args[@]}"; else run "${args[@]}" <"$1"; fi
else
  run "${args[@]}"
fi
