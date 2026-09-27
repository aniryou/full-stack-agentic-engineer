"""The command line runs offline and says what it did."""
import os
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def run(*args, env=None):
    e = {**os.environ, "PYTHONPATH": str(LAB), **(env or {})}
    return subprocess.run([sys.executable, "-m", "memlab", *args], capture_output=True, text=True, timeout=120,
                          cwd=LAB, env=e)


def test_env_and_demo(tmp_path):
    r = run("env")
    assert r.returncode == 0 and '"fts5": true' in r.stdout
    r = run("demo", "--db", str(tmp_path / "d.db"))
    assert r.returncode == 0 and "'Flores': 0" in r.stdout and "records          1" in r.stdout
    r = run("residue", "--db", str(tmp_path / "d.db"), "Flores")
    assert r.returncode == 0 and '"Flores": 0' in r.stdout


def test_token_harness_and_gcp_commands():
    r = run("token", "--tenant", "acme", "--user", "u1", env={"MEMLAB_TOKEN_KEY": "k" * 16})
    assert r.returncode == 0 and r.stdout.count(".") == 1
    r = run("harness", "--users", "2", "--modes", "none,implicit")
    assert r.returncode == 0 and "implicit" in r.stdout and "95% Wilson" in r.stdout
    r = run("gcp-commands", "--project", "p")
    assert r.returncode == 0 and "--oauth-service-account-email" in r.stdout and "gcloud run jobs delete" in r.stdout


def test_consolidate_command_on_an_empty_window(tmp_path):
    r = run("consolidate", "--db", str(tmp_path / "e.db"), "--window-days", "7")
    assert r.returncode == 0 and "0 partitions" in r.stdout
