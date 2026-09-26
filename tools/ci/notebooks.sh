#!/usr/bin/env bash
# The notebooks job: every notebook builder must be a no-op against the committed tree, and the Colab
# injector must have nothing to change in the setup cell of the notebooks it owns. Needs: pip install -r tools/ci/embeddings-lab-build.txt
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

# The Colab injector over the notebooks it owns (first cell tagged colab-bootstrap) must change nothing a
# reader runs; percent-source labs write their own setup cell, and a notebook with neither fails here.
python tools/ci/ci.py bootstrap-targets > /dev/null
python tools/ci/ci.py bootstrap-check
echo "notebooks: every rebuild is a no-op and every Colab setup cell is current"
