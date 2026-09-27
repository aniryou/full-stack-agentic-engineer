"""Agreement and task-accuracy statistics."""
import numpy as np
import pytest

from distillab import agreement as A


def test_kl_agreement_and_topk_overlap():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(100, 20)) * 2
    assert A.kl(ref, ref) == pytest.approx(0) and A.argmax_agreement(ref, ref) == 1 and A.topk_overlap(ref, ref, 5) == 1
    noisy = ref + rng.normal(size=ref.shape)
    assert 0 < A.kl(ref, noisy) and A.argmax_agreement(ref, noisy) < 1
    assert A.topk_overlap(ref, noisy, 5) >= A.argmax_agreement(ref, noisy) - 0.2
    # KL of two known distributions (the five-token example of the fact sheet at T = 1)
    z, v = np.array([[4, 3, 1, 0, -1.0]]), np.array([[3, 3.5, 0, 0.5, -1.0]])
    assert A.kl(z, v) == pytest.approx(0.25650, abs=1e-5) and A.kl(v, z) == pytest.approx(0.271424, abs=1e-5)


def test_paired_flips_and_gap_table():
    t = [True] * 80 + [False] * 20
    s = [True] * 70 + [False] * 10 + [True] * 5 + [False] * 15
    r = A.paired(t, s)
    assert (r["lost"], r["gained"]) == (10, 5) and r["mcnemar_z"] == pytest.approx(-5 / np.sqrt(15))
    rows = A.gap_table([{"difficulty": d, "teacher": True, "student": d < 3} for d in (1, 2, 3, 4) for _ in range(10)])
    assert [r["gap"] for r in rows] == [0, 0, 1.0, 1.0] and rows[0]["student 95% CI"].startswith("0.72")


def test_lm_eval_command_for_thinking_models():
    plain = A.lm_eval_command("Qwen/Qwen2.5-0.5B-Instruct")
    assert plain[:3] == ["lm_eval", "--model", "vllm"] and "--limit" in plain and "--apply_chat_template" not in plain
    think = A.lm_eval_command("Qwen/Qwen3-0.6B", thinking=True)
    assert "--apply_chat_template" in think and any("enable_thinking=True" in x for x in think)


def test_run_lm_eval_without_the_harness_prints_the_command(capsys, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert A.run_lm_eval("Qwen/Qwen2.5-0.5B-Instruct", limit=50) is None
    out = capsys.readouterr().out
    assert "lm_eval --model vllm" in out and "--output_path" in out and "--limit 50" in out
