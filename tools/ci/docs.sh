#!/usr/bin/env bash
# The docs job: site generator tests, site nav regeneration (mkdocs.yml must not change), a strict
# site build, and a relative-link check over every tracked Markdown file (after the site pages are
# generated, since site/guide links to them). Needs: pip install -r requirements-site.txt
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

python -m pytest -q tools/site/tests tools/ci/tests
python tools/site/build_site_content.py
if ! git diff --exit-code --stat mkdocs.yml; then
  echo "::error::mkdocs.yml is stale: run python3 tools/site/build_site_content.py and commit mkdocs.yml" >&2
  exit 1
fi
python -m mkdocs build --strict --quiet --site-dir "$(mktemp -d)"
git ls-files -z '*.md' | xargs -0 python tools/orchestration/mdlinks.py
