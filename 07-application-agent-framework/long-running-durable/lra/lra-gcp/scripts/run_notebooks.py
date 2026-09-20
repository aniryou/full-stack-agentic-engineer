"""Execute the worked notebooks headlessly (CI check). Practice notebooks are not executed: they contain blanks by design."""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failures = 0
    for path in sorted((ROOT / "notebooks" / "worked").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
        try:
            client.execute()
            nbformat.write(nb, path)  # keep outputs so the notebooks read well on GitHub
            print(f"ok   {path.name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {path.name}: {str(e)[:400]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
