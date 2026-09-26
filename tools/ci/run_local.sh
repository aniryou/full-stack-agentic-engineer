#!/usr/bin/env bash
# Run what CI runs (.github/workflows/tests.yml), on a laptop, each job in a fresh venv.
#
#   tools/ci/run_local.sh roofline-core lra-gcp   # the tests job for these labs (ids from tools/ci/labs.json)
#   tools/ci/run_local.sh --all-labs              # the tests job for every lab (about 37 venvs; slow)
#   tools/ci/run_local.sh --solutions agent-core  # the manual solutions job for these labs (lra-gcp's rewrites the
#                                                 # outputs of its worked notebooks: git restore them afterwards)
#   tools/ci/run_local.sh --notebooks             # every builder + the Colab injector are no-ops (needs a clean tree)
#   tools/ci/run_local.sh --docs                  # site tests, mkdocs.yml up to date, strict build, relative links
#   tools/ci/run_local.sh --colab-index           # tools/gen_colab_index.py is a no-op
#   tools/ci/run_local.sh --list                  # the lab ids
#
# Needs python3.11 (and python3.12 for the *-py312 entries) on PATH. Venvs go under $CI_LOCAL_DIR
# (default: a temporary directory) and are removed after each job unless KEEP_VENVS=1.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
WORK="${CI_LOCAL_DIR:-$(mktemp -d)}"
mkdir -p "$WORK"
declare -a SUMMARY=()
FAILED=0

lab_field() {  # lab_field <id> <field>
  python3 -c 'import json,sys; print(next(l for l in json.load(open("tools/ci/labs.json"))["labs"] if l["id"]==sys.argv[1]).get(sys.argv[2],""))' "$1" "$2"
}

new_venv() {  # new_venv <name> <python-version>; prints the venv path
  local venv="$WORK/venv-$1" py="python$2"
  command -v "$py" > /dev/null || { echo "$py not found on PATH" >&2; return 1; }
  rm -rf "$venv"
  "$py" -m venv "$venv"
  "$venv/bin/python" -m pip install -q -U pip wheel
  echo "$venv"
}

record() {  # record <job> <rc> <seconds> <log>
  local status="ok"
  [ "$2" -eq 0 ] || { status="FAIL (log: $4)"; FAILED=1; }
  SUMMARY+=("$(printf '%-40s %5ss  %s' "$1" "$3" "$status")")
}

run_job() {  # run_job <name> <python-version> <install-command> <run-command>
  local name="$1" log="$WORK/$1.log" start rc venv
  start=$(date +%s)
  echo "== $name"
  if venv=$(new_venv "$name" "$2"); then
    # shellcheck disable=SC1091
    ( . "$venv/bin/activate" && export PIP_DISABLE_PIP_VERSION_CHECK=1 && eval "$3" && eval "$4" ) > "$log" 2>&1 && rc=0 || rc=$?
    [ "${KEEP_VENVS:-0}" = 1 ] || rm -rf "$venv"
  else
    rc=1
  fi
  tail -n 3 "$log" 2> /dev/null | sed 's/^/   /' || true
  record "$name" "$rc" "$(( $(date +%s) - start ))" "$log"
}

lab() {  # lab <id> <phase: test|solutions>
  local id="$1" phase="$2"
  [ -n "$(lab_field "$id" dir)" ] || { echo "unknown lab id: $id (see --list)" >&2; exit 2; }
  run_job "$phase-$id" "$(lab_field "$id" python)" \
    "python tools/ci/ci.py run '$id' install" "python tools/ci/ci.py run '$id' $phase"
}

phase="test"
[ $# -gt 0 ] || { sed -n '2,14p' "$0"; exit 2; }
for arg in "$@"; do
  case "$arg" in
    --list) python3 -c 'import json; [print(l["id"]) for l in json.load(open("tools/ci/labs.json"))["labs"]]'; exit 0 ;;
    --solutions) phase=solutions ;;
    --all-labs)
      while IFS= read -r id; do lab "$id" "$phase"; done < <(python3 -c 'import json; [print(l["id"]) for l in json.load(open("tools/ci/labs.json"))["labs"]]') ;;
    --notebooks) run_job notebooks 3.11 "python -m pip install -q -r tools/ci/embeddings-lab-build.txt" "bash tools/ci/notebooks.sh" ;;
    --docs) run_job docs 3.11 "python -m pip install -q -r requirements-site.txt" "bash tools/ci/docs.sh" ;;
    --colab-index) run_job colab-index 3.11 "true" "bash tools/ci/colab_index.sh" ;;
    -*) echo "unknown option: $arg" >&2; exit 2 ;;
    *) lab "$arg" "$phase" ;;
  esac
done

echo
echo "summary (logs in $WORK):"
printf '%s\n' "${SUMMARY[@]}"
exit "$FAILED"
