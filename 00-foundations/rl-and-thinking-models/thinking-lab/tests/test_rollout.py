"""RL rollout bookkeeping pinned to TRL's formulas and the fact sheet's hand values."""
import math

import pytest

from thinklab import rollout as R


def test_group_advantages_trl_exact():
    assert all(abs(a - e) < 1e-6 for a, e in zip(R.group_advantages([1, 0, 0, 1]), [0.865875, -0.865875, -0.865875, 0.865875]))
    assert all(abs(a - e) < 1e-4 for a, e in zip(R.group_advantages([1, 0, 0, 0]), [1.4997, -0.4999, -0.4999, -0.4999]))
    assert R.group_advantages([1, 0], scale="none") == [0.5, -0.5]
    assert R.group_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]


def test_k3_and_soft_overlong():
    assert abs(R.k3(0, 0.1) - 0.0051709) < 1e-7 and abs(R.k3(0, -0.1) - 0.0048374) < 1e-7
    assert abs(R.k3(0, 0.5) - 0.1487213) < 1e-7
    assert [R.soft_overlong(x, 100, 20) for x in (80, 90, 100, 101)] == [0.0, -0.5, -1.0, -1.0]


def test_is_ratio_modes():
    s, t = [-1.0, -1.0], [-0.9, -1.2]
    assert R.is_ratios(s, t, "token_truncate") == pytest.approx([math.exp(0.1), math.exp(-0.2)])
    assert R.is_ratios([-3.0], [0.0], "token_truncate") == [3.0]
    assert R.is_ratios([-3.0], [0.0], "token_mask") == [0.0]
    assert R.is_ratios(s, t, "sequence_truncate") == pytest.approx([math.exp(-0.1)] * 2)
    assert R.is_ratios([-1, -1], [0, 0], "sequence_mask") == [0.0, 0.0]          # e^2 > 3


def test_aggregations_and_length_bias():
    batch = [[-1.0] * 2, [-1.0] * 8]
    assert R.aggregate(batch, "grpo") == -1.0 and R.aggregate(batch, "dapo") == -1.0
    assert R.aggregate(batch, "dr_grpo", 10) == -0.5
    assert R.clipped_token_loss(math.log(2), 0.0, 1.0) == -1.2       # ratio 2 clipped at 1.2
    assert R.clipped_token_loss(math.log(2), 0.0, -1.0) == 2.0       # negative advantage: min keeps the unclipped


def test_groups_dynamic_sampling_and_rollout_phase():
    rs = [R.Rollout("a", 3, 1.0), R.Rollout("a", 3, 0.0), R.Rollout("b", 3, 1.0), R.Rollout("b", 3, 1.0)]
    g = R.group_by_prompt(rs)
    assert R.frac_zero_std(g) == 0.5 and list(R.dynamic_sampling(g)) == ["a"]
    ph = R.rollout_phase([100, 200, 1000], lambda b: 0.01, overlap_training_s=5.0)
    assert ph["phase_s"] == pytest.approx(10.0) and ph["idle_share"] == pytest.approx(1 - 1300 / 3000)
    assert ph["overlapped_step_s"] == pytest.approx(10.0) and ph["sync_step_s"] == pytest.approx(15.0)
    assert R.weight_sync_bytes(494e6) == 988e6 and R.weight_sync_bytes(494e6, changed_fraction=0.02) == pytest.approx(19.76e6)


def test_trl_config_for_a_t4():
    c = R.trl_grpo_config("T4")
    assert c["fp16"] is True and c["bf16"] is False and c["use_vllm"] and c["vllm_mode"] == "colocate"
    assert c["per_device_train_batch_size"] * c["gradient_accumulation_steps"] % c["num_generations"] == 0
    assert "bf16" not in R.trl_grpo_config("L4")
