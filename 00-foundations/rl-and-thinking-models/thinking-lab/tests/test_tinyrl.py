"""The tiny-RL task and verifier and the recorded run: pure Python, no torch (the torch tests are in
test_tinyrl_torch.py)."""
import random

from thinklab.tinyrl.curves import load_recorded, show
from thinklab.tinyrl.task import EOS, END_THINK, PAD, THINK, DigitSum, parse, reward


def test_task_and_verifier():
    t = DigitSum(6, 5)
    p = t.sample(random.Random(0))
    assert p.running_sums()[-1] == p.answer == sum(p.digits) % 5
    for j in range(7):
        c = t.demo(p, j)
        assert reward(p, c + [PAD, PAD]) == 1.0 and parse(c)["scratch"] == j and len(c) == j + 4
    assert reward(p, [THINK, END_THINK, (p.answer + 1) % 5, EOS]) == 0.0
    assert reward(p, [THINK, 1, 2, EOS]) == 0.0 and reward(p, [END_THINK, p.answer, EOS]) == 0.0
    assert t.max_completion == 10 and t.seq_len == 17


def test_recorded_run_is_labelled_and_shows_the_effect():
    run = load_recorded()
    assert run["source"].startswith("recorded run (illustrative)")
    assert run["after"]["accuracy"] > run["before"]["accuracy"] + 0.2
    assert run["rl"][-1]["length"] > run["rl"][0]["length"] + 2
    assert "GRPO" in show(run)


def test_no_torch_switch(monkeypatch):
    from thinklab import env
    monkeypatch.setenv("THINKLAB_NO_TORCH", "1")
    assert env.has_torch() is False and env.torch_device() == "none"
