#!/usr/bin/env bash
# Install gVisor (runsc) from its apt repository and register it with Docker as the runtime "runsc".
# Steps from gVisor's install guide (g3doc/user_guide/install.md, 2026-09); Debian/Ubuntu, x86_64 or arm64,
# Linux 5.6+. Needs sudo. Not for Colab or Docker Desktop's VM (no systemd/dockerd you control).
#   DRY_RUN=1 deploy/docker/install-gvisor.sh     # read it first
#   deploy/docker/install-gvisor.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib.sh
. "$HERE/../lib.sh"
need sudo "Run as a user with sudo."
need curl "sudo apt-get install -y curl"

log "1/4 apt prerequisites"
run sudo apt-get update
run sudo apt-get install -y apt-transport-https ca-certificates curl gnupg
log "2/4 gVisor signing key and repository (release channel)"
run sh -c 'curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor --yes -o /usr/share/keyrings/gvisor-archive-keyring.gpg'
run sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list >/dev/null'
log "3/4 runsc (the package configures Docker when Docker is installed)"
run sudo apt-get update
run sudo apt-get install -y runsc
log "4/4 register with Docker and check"
run sudo runsc install
run sudo systemctl restart docker
run docker run --rm --runtime=runsc hello-world
run runsc --version
