#!/usr/bin/env python3
"""Percent-format notebook sources: every Markdown heading sits in a markdown cell (standard library only).

A percent source (``notebooks_src/NN_name.py`` in the percent-source labs, ``src/*.py`` in embeddings-lab) is a
sequence of cells, each opened by a marker line: ``# %% [markdown]`` is prose, one ``# `` comment per line, and
``# %%``, ``# %% exercise`` and ``# %% check`` are code. A heading such as ``# ## Exercise 2.5 — ...`` that follows
a code, exercise or check cell without a ``# %% [markdown]`` line of its own is Python to every builder, which
writes it, and the prose under it, as comments at the end of that code cell: a learner then reads an exercise
statement, a worked example's introduction or the "In a design review" section at the bottom of a check cell.
The builders drop the lines before the first marker, so a heading there is reported too.

    python3 tools/ci/nb_sources.py            # every tracked percent source with its cell counts, then every finding
    python3 tools/ci/nb_sources.py --check    # findings only; exit 1 if there is any (run by `ci.py check`)
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MARKER = re.compile(r"^# %%(.*)$")
HEADING = re.compile(r"^# #{1,6} \S")     # a Markdown heading as a markdown cell's line: "# ## Title"
MARKDOWN = "markdown"
BEFORE_FIRST = "before the first cell marker"


SOURCE_GLOBS = ("*/notebooks_src/*.py", "*/embeddings-lab/src/*.py")   # what the builders read


def tracked_sources() -> list[str]:
    """Every tracked source a builder reads: notebooks_src/*.py in the percent-source labs and src/*.py in
    embeddings-lab (SOURCE_GLOBS). Not "any .py with a `# %%` line": a test fixture or a script with cell markers
    is not a notebook source."""
    out = subprocess.run(["git", "ls-files", "--", *SOURCE_GLOBS], cwd=REPO, check=True, capture_output=True, text=True)
    return sorted(p for p in out.stdout.splitlines() if p)


def cell_kind(marker_rest: str) -> str:
    """The kind every builder gives a cell from what follows its ``# %%``: 'markdown' for ``[markdown]``, else the
    first word (``exercise``, ``check``) or 'code' when there is none."""
    rest = marker_rest.strip()
    if rest.startswith("[markdown]"):
        return MARKDOWN
    return rest.split()[0] if rest else "code"


def cells(text: str) -> list[tuple[str, int, list[str]]]:
    """[(kind, line number of the marker, the cell's lines)], split as the builders split a source. Lines before the
    first marker, if any, come first under the kind BEFORE_FIRST with line number 0."""
    out: list[tuple[str, int, list[str]]] = []
    kind, start, buf = BEFORE_FIRST, 0, []
    for n, line in enumerate(text.splitlines(), 1):
        m = MARKER.match(line)
        if m:
            if kind != BEFORE_FIRST or buf:
                out.append((kind, start, buf))
            kind, start, buf = cell_kind(m.group(1)), n, []
        else:
            buf.append(line)
    if kind != BEFORE_FIRST or buf:
        out.append((kind, start, buf))
    return out


def misplaced_headings(text: str) -> list[tuple[int, str, str]]:
    """[(line number, cell kind, the line)] for every heading line outside a markdown cell."""
    found = []
    for kind, start, lines in cells(text):
        if kind == MARKDOWN:
            continue
        found += [(start + offset, kind, line) for offset, line in enumerate(lines, 1) if HEADING.match(line)]
    return found


def findings(paths: list[str], root: Path = REPO) -> list[tuple[str, int, str]]:
    """[(path, line number, what is wrong)] over the given sources (repo-relative, or absolute)."""
    out = []
    for p in paths:
        for n, kind, line in misplaced_headings((root / p).read_text(encoding="utf-8")):
            where = kind if kind == BEFORE_FIRST else f"in a {kind} cell"
            out.append((p, n, f"a heading {where} (add a `# %% [markdown]` line before it): {line.rstrip()}"))
    return out


def main(argv: list[str]) -> int:
    paths = tracked_sources()
    if "--check" not in argv:
        width = max((len(p) for p in paths), default=0)
        for p in paths:
            counts = Counter(kind for kind, _, _ in cells((REPO / p).read_text(encoding="utf-8")))
            print(f"{p:<{width}}  " + "  ".join(f"{k} {counts[k]}" for k in sorted(counts)))
        print()
    bad = findings(paths)
    for p, n, why in bad:
        print(f"notebook source: {p}:{n}: {why}")
    print(f"{len(paths)} percent-format sources, {len(bad)} headings outside a markdown cell")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
