"""The distillation losses in torch, pinned to hand-computed values (the fact sheet's five-token example) and to
TRL's GKD convention. Skipped without torch."""
import math

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from distillab.losses import generalized_jsd, kd_loss, soft_kl_rows, token_rewards  # noqa: E402

Z = torch.tensor([[4.0, 3.0, 1.0, 0.0, -1.0]], dtype=torch.float64)      # teacher logits (float64: the high-T
# values are differences of nearly equal numbers)
V = torch.tensor([[3.0, 3.5, 0.0, 0.5, -1.0]], dtype=torch.float64)      # student logits


def test_soft_targets_and_kl_at_several_temperatures():
    assert soft_kl_rows(Z, V, 1.0).item() == pytest.approx(0.25650, abs=1e-5)
    assert soft_kl_rows(Z, V, 2.0).item() == pytest.approx(0.06639, abs=1e-5)
    for T, t2kl in ((2, 0.26558), (4, 0.25542), (10, 0.24196), (100, 0.23129), (1000, 0.23013)):
        assert T * T * soft_kl_rows(Z, V, T).item() == pytest.approx(t2kl, abs=2e-5)
    # the high-temperature limit is logit matching on centred logits: (1/2N) ||c(v) - c(z)||^2 = 0.23 exactly
    cz, cv = Z - Z.mean(), V - V.mean()
    assert ((cv - cz) ** 2).sum().item() / (2 * 5) == pytest.approx(0.23)


def test_kd_gradient_is_T_times_q_minus_p():
    v = V.clone().requires_grad_(True)
    loss = kd_loss(v, Z, alpha=1.0, temperature=2.0)
    loss.backward()
    p, q = F.softmax(Z / 2, -1), F.softmax(V / 2, -1)
    assert torch.allclose(v.grad, 2.0 * (q - p), atol=1e-6)            # T^2 x (q - p)/T
    assert torch.allclose((q - p) / 2, torch.tensor([[-0.073543, 0.071047, -0.016410, 0.015853, 0.003053]], dtype=torch.float64), atol=1e-5)


def test_alpha_mixes_the_hard_label_loss_and_chunking_changes_nothing():
    g = torch.Generator().manual_seed(0)
    s, t = torch.randn(3, 7, 11, generator=g), torch.randn(3, 7, 11, generator=g)
    y = torch.randint(0, 11, (3, 7), generator=g)
    mask = torch.rand(3, 7, generator=g) > 0.3
    hard = F.cross_entropy(s[mask], y[mask])
    assert kd_loss(s, t, y, mask, 2.0, 0.0).item() == pytest.approx(hard.item(), abs=1e-6)
    full = kd_loss(s, t, y, mask, 2.0, 0.7)
    assert kd_loss(s, t, y, mask, 2.0, 0.7, chunk=4).item() == pytest.approx(full.item(), abs=1e-6)
    soft = kd_loss(s, t, None, mask, 2.0, 1.0)
    assert full.item() == pytest.approx(0.7 * soft.item() + 0.3 * hard.item(), abs=1e-6)
    with pytest.raises(ValueError):
        kd_loss(s, t, None, mask, 2.0, 0.5)


def test_generalized_jsd_follows_trls_convention():
    p, q = F.softmax(Z, -1), F.softmax(V, -1)
    fwd = (p * (p.log() - q.log())).sum().item()
    rev = (q * (q.log() - p.log())).sum().item()
    assert generalized_jsd(V, Z, 0.0).item() == pytest.approx(fwd) == pytest.approx(0.256501, abs=1e-6)
    assert generalized_jsd(V, Z, 1.0).item() == pytest.approx(rev) == pytest.approx(0.271424, abs=1e-6)
    for beta, want in ((0.01, 0.002539), (0.1, 0.023026), (0.5, 0.064431), (0.9, 0.024067), (0.99, 0.002683)):
        assert generalized_jsd(V, Z, beta).item() == pytest.approx(want, abs=2e-6)
    assert generalized_jsd(V, Z, 0.5).item() <= math.log(2)


def test_token_rewards_sum_to_minus_the_log_ratio():
    lp_s = torch.tensor([-0.1, -2.0, -0.5])
    lp_t = torch.tensor([-0.2, -0.1, -0.5])
    r = token_rewards(lp_s, lp_t)
    assert r.tolist() == pytest.approx([-0.1, 1.9, 0.0]) and r.sum().item() == pytest.approx(-(lp_s.sum() - lp_t.sum()).item())
