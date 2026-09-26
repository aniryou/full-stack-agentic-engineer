#!/usr/bin/env bash
# The egress pattern on one machine: the proxy listens on a Unix socket on the host, the container has
# --network none and the socket bind-mounted, so the proxy is its only way out and the credential stays
# in a host file only the proxy reads. Also starts the stand-in upstream (sandboxlab/proxy/stub.py).
#   deploy/docker/run-with-proxy.sh examples/fetch_weather.py
#   DRY_RUN=1 deploy/docker/run-with-proxy.sh examples/fetch_weather.py
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB="$(cd "$HERE/../.." && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"

STATE="${STATE_DIR:-/tmp/sandboxlab-egress}"
SOCK="$STATE/proxy.sock"
need python3 "Python 3.10+ runs the proxy and the stub (standard library only)."

log "1/4 state directory $STATE (the token file is readable by you only)"
run mkdir -p "$STATE"
run chmod 711 "$STATE"
random_token "$STATE/token"

log "2/4 stand-in upstream on 127.0.0.1:8081 (checks the bearer token)"
run python3 "$LAB/sandboxlab/proxy/stub.py" --host 127.0.0.1 --port 8081 --token-file "$STATE/token" &
STUB_PID=$!

log "3/4 egress proxy on $SOCK (route api-stub -> 127.0.0.1:8081, credential injected from the token file)"
cat_config() {
  cat <<JSON
{"routes": {"api-stub": {"upstream": "http://127.0.0.1:8081", "methods": ["GET"],
  "inject": {"header": "Authorization", "value_from": "file:$STATE/token", "format": "Bearer {}"}}},
 "forward_allow": [], "allow_connect": false}
JSON
}
if [[ "$DRY_RUN" != "1" ]]; then cat_config >"$STATE/proxy.json"; fi
run python3 "$LAB/sandboxlab/proxy/server.py" --config "$STATE/proxy.json" --unix "$SOCK" &
PROXY_PID=$!
trap 'kill ${STUB_PID:-} ${PROXY_PID:-} 2>/dev/null || true' EXIT
[[ "$DRY_RUN" == "1" ]] || sleep 1

log "4/4 the hardened container with only the proxy socket"
SANDBOX_PROXY_SOCKET="$SOCK" "$HERE/run-hardened.sh" "$@"
