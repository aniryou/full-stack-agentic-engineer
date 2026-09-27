"""The command line runs offline."""
import os
import subprocess
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]


def run(*args, env=None):
    r = subprocess.run([sys.executable, "-m", "distillab", *args], cwd=LAB, capture_output=True, text=True, timeout=120,
                       env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_env_memory_cost_and_spec_config():
    assert "torch" in run("env")
    assert "does NOT fit" in run("memory", "--teacher", "qwen2.5-1.5b-instruct")
    out = run("cost")                                              # the teacher on two H100s by default
    assert "0.8896" in out and "2xH100" in out and "break-even" in out
    assert "5.352" in run("cost", "--teacher-gpus", "1")
    assert "--speculative-config" in run("spec-config")


def test_tinylm_without_torch_prints_the_recorded_run():
    out = run("tinylm", env={"DISTILLAB_NO_TORCH": "1"})
    assert "recorded run (illustrative)" in out and "seqkd" in out


def test_teacher_data_against_the_fake(tmp_path):
    out = run("teacher-data", "--problems", "20", "-n", "2", "--out", str(tmp_path / "t"))
    assert "SIMULATED" in out and (tmp_path / "t_pc.jsonl").exists() and (tmp_path / "t_msgs.jsonl").exists()
    verified = len((tmp_path / "t_pc.jsonl").read_text().splitlines())
    out = run("teacher-data", "--problems", "20", "-n", "2", "--keep", "all", "--out", str(tmp_path / "a"))
    everything = len((tmp_path / "a_pc.jsonl").read_text().splitlines())
    assert "right or wrong" in out and everything > verified       # a draft's data keeps the target's mistakes
