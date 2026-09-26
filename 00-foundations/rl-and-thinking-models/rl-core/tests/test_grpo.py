"""GRPO's pieces pinned to hand-computed values (TRL's formulas), the length bias of per-sequence averaging,
and the DAPO fixes."""
import numpy as np

from rlcore import Policy, ThinkTask, grpo


def test_group_advantages_match_trl():
    """TRL: (r − mean) / (std + 1e-4) with Bessel's correction."""
    a = grpo.group_advantages([[1, 0, 0, 1], [1, 0, 0, 0]])
    assert np.allclose(a[0], [0.865875, -0.865875, -0.865875, 0.865875], atol=1e-6)
    assert np.allclose(a[1], [1.4997, -0.4999, -0.4999, -0.4999], atol=1e-4)
    assert np.allclose(grpo.group_advantages([[1, 1, 1, 1]]), 0)            # all right: no signal
    assert np.allclose(grpo.group_advantages([[1, 0, 0, 1]], "none"), [[0.5, -0.5, -0.5, 0.5]])


def test_std_scaling_upweights_nearly_solved_prompts():
    """Dr. GRPO's difficulty bias: the one miss on an easy prompt gets −2.47, a miss on a 50/50 prompt −0.94."""
    easy, even = grpo.group_advantages([[1] * 7 + [0], [1] * 4 + [0] * 4])
    assert round(easy[-1], 2) == -2.47 and round(easy[0], 3) == 0.353 and round(even[-1], 3) == -0.935


def test_k3_hand_values_and_properties():
    assert round(float(grpo.k3(0.0, 0.1)), 7) == 0.0051709
    assert round(float(grpo.k3(0.0, -0.1)), 7) == 0.0048374
    assert round(float(grpo.k3(0.0, 0.5)), 7) == 0.1487213
    rng = np.random.default_rng(0)                    # π, π_ref two categoricals; samples from π
    p, q = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])
    x = rng.choice(3, 200000, p=p)
    kl = float(np.sum(p * np.log(p / q)))
    s1, s3 = grpo.k1(np.log(p[x]), np.log(q[x])), grpo.k3(np.log(p[x]), np.log(q[x]))
    assert abs(s1.mean() - kl) < 0.005 and abs(s3.mean() - kl) < 0.005      # both unbiased
    assert s3.min() >= 0 > s1.min() and s3.var() < s1.var()                 # k3: never negative, quieter


def test_clipped_surrogate_stops_the_gradient_past_the_clip():
    obj, d = grpo.clipped_surrogate([0.9, 1.1, 1.3, 1.3, 0.7], [1, 1, 1, -1, -1], 0.2)
    assert np.allclose(obj, [0.9, 1.1, 1.2, -1.3, -0.8])
    assert np.allclose(d, [1, 1, 0, -1, 0])            # A>0 past 1.2: clipped; A<0 below 0.8: clipped
    _, d_hi = grpo.clipped_surrogate([1.25], [1.0], 0.2, 0.28)
    assert d_hi[0] == 1.0                              # clip-higher leaves room to raise a rare token further


def test_token_weights_by_loss_type():
    lengths = [10, 50]
    g, t, dr = (grpo.token_weights(lengths, k, max_len=100) for k in ("grpo", "dapo", "dr_grpo"))
    assert np.isclose(g[0][0], 1 / 20) and np.isclose(g[1][0], 1 / 100)       # a short answer's tokens: 5×
    assert np.isclose(t[0][0], 1 / 60) and np.isclose(t[1][0], 1 / 60)
    assert np.isclose(dr[0][0], 1 / 200) and all(np.isclose(sum(w.sum() for w in g), 1) for _ in [0])


def test_soft_overlong_penalty_dapo_eq13():
    vals = [grpo.soft_overlong_penalty(n, 100, 20) for n in (80, 90, 100, 101)]
    assert vals == [0.0, -0.5, -1.0, -1.0]


def test_per_sequence_averaging_biases_the_update_toward_truncation():
    """Average many one-group updates at a fixed policy: token-level and constant normalisers point along
    the true gradient of accuracy and cut truncation; the per-sequence mean does not."""
    task = ThinkTask(e0=0.8, q=0.15, max_think=16)
    pol = Policy.for_task(task)
    pol.theta[:, 1] = np.log(0.1 / 0.9)                                     # answer w.p. 0.1 per step
    eps, true = 1e-5, np.zeros_like(pol.theta)
    for i in range(16):
        for j in range(2):
            up, dn = pol.copy(), pol.copy()
            up.theta[i, j] += eps
            dn.theta[i, j] -= eps
            true[i, j] = (task.expected(up.stop_probs(task))["accuracy"] - task.expected(dn.stop_probs(task))["accuracy"]) / (2 * eps)
    out = {}
    for lt in ("grpo", "dapo"):
        g = grpo.expected_update(pol, task, np.random.default_rng(0), lt, n_groups=600)
        step = pol.copy()
        step.step(g, 1e-3 / np.linalg.norm(g))
        out[lt] = ((g * true).sum() / np.linalg.norm(g) / np.linalg.norm(true),
                   task.expected(step.stop_probs(task))["truncated"] - task.expected(pol.stop_probs(task))["truncated"])
    assert out["dapo"][0] > 0.95 and out["grpo"][0] < 0.8
    assert out["dapo"][1] < 0 < out["grpo"][1]


def test_mu_one_never_clips_and_zero_std_groups_are_counted():
    task = ThinkTask(prompts=[(0.0, 0.1), (0.8, 0.15)], max_think=8)     # prompt 0: always right
    pol = Policy.for_task(task)
    s = grpo.grpo_step(pol, task, np.random.default_rng(0), grpo.GRPOConfig(num_generations=4), prompts=(0, 1))
    assert s["clipped"] == 0.0 and s["frac_reward_zero_std"] >= 0.5
    s = grpo.grpo_step(pol, task, np.random.default_rng(0), grpo.GRPOConfig(num_generations=4, dynamic_sampling=True),
                       prompts=(0, 1))
    assert s["groups_generated"] > 2                                        # it paid extra rollouts to refill
    cfg = grpo.GRPOConfig(num_generations=8, num_iterations=4, lr=50.0)
    s = grpo.grpo_step(Policy.for_task(task), task, np.random.default_rng(1), cfg, prompts=(1, 1))
    assert s["clipped"] > 0                                                 # μ > 1: the ratio moves, the clip binds


def test_grpo_gradient_matches_finite_differences():
    """grpo_step ascends Σ w·[min(ρA, clip(ρ)·A) − β·k3] with a hand-derived gradient (surrogate_grad).
    Check it against central differences of that objective, computed independently, with π_old and π_ref
    away from π so the clip binds on some tokens and β > 0 — for every loss_type's token weights."""
    task = ThinkTask(e0=0.8, q=0.15, max_think=8)
    rng = np.random.default_rng(3)
    shape = (task.n_states, task.n_actions)
    pol = Policy.for_task(task, rng.normal(0, 0.5, shape))
    old = Policy.for_task(task, pol.theta + rng.normal(0, 0.4, shape))
    ref = Policy.for_task(task, pol.theta + rng.normal(0, 0.4, shape))
    trajs = old.sample(task, rng, 8)
    batch = list(zip(trajs, rng.normal(0, 1, len(trajs))))           # any advantages: the gradient is linear in A
    old_lp, ref_lp = [old.token_logprobs(t) for t in trajs], [ref.token_logprobs(t) for t in trajs]
    for lt in ("grpo", "dapo", "dr_grpo"):
        cfg = grpo.GRPOConfig(beta=0.3, epsilon=0.1, epsilon_high=0.15, loss_type=lt)
        weights = grpo.token_weights([t.length for t in trajs], lt, task.max_think)

        def objective(theta):
            p = Policy.for_task(task, theta)
            total = 0.0
            for (t, a), w, o, r in zip(batch, weights, old_lp, ref_lp):
                lp = p.token_logprobs(t)
                obj, _ = grpo.clipped_surrogate(np.exp(lp - o), a, cfg.epsilon, cfg.epsilon_high)
                total += float((w * (obj - cfg.beta * grpo.k3(lp, r))).sum())
            return total

        g, clipped = grpo.surrogate_grad(pol, batch, weights, old_lp, ref_lp, cfg)
        fd, eps = np.zeros_like(pol.theta), 1e-6
        for idx in np.ndindex(*shape):
            up, dn = pol.theta.copy(), pol.theta.copy()
            up[idx] += eps
            dn[idx] -= eps
            fd[idx] = (objective(up) - objective(dn)) / (2 * eps)
        assert clipped > 0 and np.abs(g).max() > 1e-3                 # the clip binds somewhere; not a zero test
        assert np.allclose(g, fd, atol=1e-8, rtol=1e-6), (lt, np.abs(g - fd).max())
