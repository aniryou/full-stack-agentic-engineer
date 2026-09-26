#!/usr/bin/env bash
# Remove k3s (and with it the device plugin and every pod) from the GPU VM. The VM itself keeps
# billing until you stop or delete it in your provider's console - do that too.
#   deploy/gpu-vm/down.sh
#   DRY_RUN=1 deploy/gpu-vm/down.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"

SUDO=""
[[ "$(id -u)" == "0" ]] || SUDO="sudo"
log "uninstall k3s (the script the k3s installer left behind)"
run $SUDO /usr/local/bin/k3s-uninstall.sh
echo "now stop or delete the VM in your provider's console: an idle GPU VM still bills by the hour"
