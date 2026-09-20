# raw/ — the inbox

Drop new learning content here, then ask Claude to **sort raw** (see [`../CLAUDE.md`](../CLAUDE.md)).
Everything in this folder is gitignored **except this README**, so nothing here is committed to the
mono-repo until it has been moved into a layer.

## What a good drop looks like

One self-contained **lab folder** per topic, kebab-case, structured so it slots into a layer and
runs in Colab with no extra work:

- `README.md` — what it teaches, which stack layer it targets, how to run it.
- **Its own dependencies** — a `pyproject.toml` (preferred, so `pip install -e .` works) or a
  `requirements.txt`. Stdlib-only labs need neither. Labs stay independent; there is no repo-wide env.
- A package dir (`mylab/` or `src/mylab/`) for importable code; keep notebooks thin.
- `notebooks/` — exercises. Best authored as percent-format sources (`notebooks_src/*.py`) plus a
  `tools/build_notebooks.py` that emits blank `notebooks/` + filled `solutions/`
  (see `07-application-agent-framework/agent-fundamentals/google-fde-prep-lab`). Hand-written
  notebooks are fine too.
- `tests/` — one check per concept. `docs/` — a primer and cheatsheets.

## Help Claude place it

Name the folder for its topic and, if you can, its layer — e.g. `kv-cache-lab` rather than `files7`.
Layer signal keywords are in `../CLAUDE.md`. Opaque names (`files3/`, `files6/`) still work but will
prompt a clarifying question.

## What "sort raw" does

1. Reads each item; maps it to its primary layer (01-hardware … 07-application) using the keywords in `../CLAUDE.md`.
2. Proposes the mapping and confirms anything ambiguous.
3. Moves it to `<layer>/<topic>/`, makes its notebooks Colab-ready
   (`python3 tools/inject_colab_bootstrap.py <lab>`), regenerates `COLAB.md`, commits, and pushes.
4. Leaves this folder empty again.
