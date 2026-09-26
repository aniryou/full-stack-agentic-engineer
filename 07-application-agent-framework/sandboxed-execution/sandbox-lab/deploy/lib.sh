#!/usr/bin/env bash
# Shared helpers for this lab's deploy scripts. Source it, do not run it:
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
# Every command goes through `run`, which prints it first; with DRY_RUN=1 nothing is executed,
# so `DRY_RUN=1 deploy/kind/up.sh` shows the whole procedure without Docker or a cluster.
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

log()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Print arguments as a copy-pasteable shell command.
quote_cmd() {
  local a out=""
  for a in "$@"; do
    if [[ "$a" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then
      out+="$a "
    else
      out+="'${a//\'/\'\\\'\'}' "
    fi
  done
  printf '%s' "${out% }"
}

# run CMD ARGS... : print, then execute unless DRY_RUN=1.
run() {
  printf '+ %s\n' "$(quote_cmd "$@")"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

# retry ATTEMPTS SLEEP CMD ARGS... : for webhooks that answer a few seconds after their Deployment is Available.
retry() {
  local attempts="$1" pause="$2" i
  shift 2
  for ((i = 1; i <= attempts; i++)); do
    if run "$@"; then
      return 0
    fi
    [[ "$DRY_RUN" == "1" ]] && return 0
    warn "attempt $i/$attempts failed; retrying in ${pause}s"
    sleep "$pause"
  done
  die "gave up after $attempts attempts: $(quote_cmd "$@")"
}

# need CMD [HINT] : fail early (or just warn in DRY_RUN) when a tool is missing.
need() {
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    warn "$1 not found (ignored because DRY_RUN=1)"
    return 0
  fi
  die "$1 not found. ${2:-}"
}

# random_token FILE : write a random 48-hex-character token to FILE (mode 600). Never on a command line.
random_token() {
  local f="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ (write a random token to %s)\n' "$f"
    return 0
  fi
  umask 077
  python3 -c 'import secrets; print(secrets.token_hex(24))' >"$f"
}
