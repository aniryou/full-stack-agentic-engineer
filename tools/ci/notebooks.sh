#!/usr/bin/env bash
# The notebooks job: every notebook builder, then the Colab injector over the notebooks it owns.
# Both must be no-ops against the committed tree. Needs: pip install -r tools/ci/embeddings-lab-build.txt
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

dirty() {  # tracked changes or new untracked files
  [ -n "$(git status --porcelain --untracked-files=all)" ]
}
if dirty; then
  echo "the working tree is not clean; commit or stash first (this check diffs against HEAD)" >&2
  git status --short >&2
  exit 2
fi

n=0
while IFS= read -r builder; do
  n=$((n + 1))
  echo "== [$n] $builder"
  (cd "$(dirname "$builder")" && python "$(basename "$builder")" > /dev/null)
done < <(python tools/ci/ci.py builders)
echo "$n builders ran"
if dirty; then
  echo "::error::a notebook builder changed the tree: rebuild with the builder and commit the result" >&2
  git status --short >&2
  git diff --stat >&2
  git diff --text | cut -c1-300 | head -n 200 >&2
  exit 1
fi

python tools/ci/ci.py bootstrap-targets > "${TMPDIR:-/tmp}/bootstrap-targets.txt"
xargs -d '\n' python tools/inject_colab_bootstrap.py < "${TMPDIR:-/tmp}/bootstrap-targets.txt"
if dirty; then
  echo "::error::tools/inject_colab_bootstrap.py changed notebooks: re-run it on them and commit" >&2
  git status --short >&2
  exit 1
fi
echo "notebooks: every rebuild and the injector are no-ops"
