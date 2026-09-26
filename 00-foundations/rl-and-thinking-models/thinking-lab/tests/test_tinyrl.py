"""The tiny-RL task and verifier (pure Python), the recorded run, and — when torch is installed — the model,
sampler/scorer consistency, the GRPO loss against the pure-Python formulas, and a short real run."""
import math
import random

import pytest

from thinklab.rollout import aggregate, clipped_token_loss
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


torch = pytest.importorskip("torch")
from thinklab.tinyrl import train as T  # noqa: E402
from thinklab.tinyrl.model import TinyGPT, sample, token_logprobs  # noqa: E402


def test_sampler_logprobs_equal_scorer_logprobs():
    torch.manual_seed(0)
    task = DigitSum(4, 5)
    m = TinyGPT(15, 32, 1, 2, task.seq_len)
    P = torch.tensor([task.sample(random.Random(i)).prompt for i in range(6)])
    comps, lp, mask = sample(m, P, task.max_completion, 1.0, torch.Generator().manual_seed(0))
    lp2 = token_logprobs(m, P, comps)
    assert torch.allclose(lp * mask, lp2 * mask, atol=1e-5)
    for row, m in zip(comps.tolist(), mask.tolist()):           # live up to and including the first <eos>, PAD after
        n = int(sum(m))
        assert m == [1.0] * n + [0.0] * (len(m) - n) and all(t == PAD for t in row[n:])
        assert EOS not in row[: n - 1] and (n == len(row) or row[n - 1] == EOS)


def test_grpo_loss_matches_pure_python():
    cfg = T.TinyRLConfig(epsilon=0.2, epsilon_high=0.28)
    logp = torch.tensor([[-0.5, -1.0, -0.2], [-0.3, -0.1, 0.0]])
    old = torch.tensor([[-0.7, -1.0, -0.2], [-0.2, -0.4, 0.0]])
    adv = torch.tensor([1.0, -0.5])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    per = [[clipped_token_loss(a, b, A, 0.2, 0.28) for a, b in zip(r[:int(m.sum())], o[:int(m.sum())])]
           for r, o, A, m in zip(logp.tolist(), old.tolist(), adv.tolist(), mask)]
    for kind in ("grpo", "dr_grpo", "dapo"):
        cfg.loss_type = kind
        loss, st = T.grpo_loss(logp, old, logp.clone(), adv, mask, cfg, 3)
        assert loss.item() == pytest.approx(aggregate(per, kind, 3), abs=1e-6)
    assert st["kl"] == pytest.approx(0.0, abs=1e-7) and 0 <= st["clip_frac"] <= 1
    adv_t = T.group_advantages(torch.tensor([1.0, 0.0, 0.0, 1.0]), 1)
    assert adv_t.tolist() == pytest.approx([0.865875, -0.865875, -0.865875, 0.865875], abs=1e-5)


def test_a_short_real_run():
    cfg = T.TinyRLConfig(k=4, sft_steps=40, rl_steps=3, prompts_per_step=4, num_generations=4, eval_prompts=32, log_every=1)
    run = T.run(cfg, log=lambda *a: None)
    assert run["source"] == "measured" and len(run["rl"]) == 4 and run["sft"][-1]["loss"] < run["sft"][0]["loss"]
    assert all(0 <= r["reward"] <= 1 and 4 <= r["length"] <= 8 for r in run["rl"])
    model, task = T.warm_start(T.TinyRLConfig(k=4, sft_steps=5), log=lambda *a: None)
    ro = T.make_rollouts(model, task, prompts=2, generations=3)
    assert len(ro) == 6 and all(len(r["sampler_logps"]) == len(r["trainer_logps"]) == r["length"] for r in ro)
    assert max(abs(a - b) for r in ro for a, b in zip(r["sampler_logps"], r["trainer_logps"])) < 0.2
