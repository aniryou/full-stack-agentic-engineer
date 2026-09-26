"""tools/orchestration/mdlinks.py: broken relative links are reported; link syntax shown as code is not a link."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MDLINKS = Path(__file__).resolve().parents[3] / "tools" / "orchestration" / "mdlinks.py"


def check(tmp_path, text):
    (tmp_path / "here.md").write_text("x")
    f = tmp_path / "doc.md"
    f.write_text(text)
    return subprocess.run([sys.executable, str(MDLINKS), str(f)], capture_output=True, text=True)


def test_broken_link_is_reported(tmp_path):
    r = check(tmp_path, "see [a](missing.md) and [b](here.md)\n")
    assert r.returncode == 1 and "missing.md" in r.stdout and "here.md" not in r.stdout


def test_code_link_text_still_checked(tmp_path):
    r = check(tmp_path, "open [`missing.md`](missing.md)\n")
    assert r.returncode == 1 and "missing.md" in r.stdout


def test_link_syntax_in_inline_code_is_not_a_link(tmp_path):
    r = check(tmp_path, "Name files as links (`[PRIMER.md](PRIMER.md)`), and [ok](here.md).\n")
    assert r.returncode == 0, r.stdout


def test_fenced_blocks_are_skipped(tmp_path):
    r = check(tmp_path, "```\n[x](nowhere.md)\n```\n")
    assert r.returncode == 0, r.stdout
