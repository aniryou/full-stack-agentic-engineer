"""Executes every solution notebook end to end. Slow; run with `pytest -m slow` or `make check`."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOLUTIONS = sorted((ROOT / "solutions").glob("*.ipynb"))


@pytest.mark.slow
@pytest.mark.parametrize("path", SOLUTIONS, ids=[p.stem for p in SOLUTIONS])
def test_solution_notebook_runs(path: Path):
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "run_notebooks.py"), str(path.relative_to(ROOT))],
                          capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
