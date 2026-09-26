"""Every computed number the topic's PRIMER.md quotes is recomputed here and must appear verbatim.

If a formula, a seed or an experiment changes, the primer fails this test until it is updated. Inputs and
cited facts (the R1, DAPO and TRL numbers, prices) are quoted, not computed, and are not checked here.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from rlcore import Policy, SeqTask, ThinkTask, grpo, pg, pref, ttc
from rlcore import workload as w


def _norm(text: str) -> str:
    return " ".join(text.split())


PRIMER = _norm((Path(__file__).resolve().parents[2] / "PRIMER.md").read_text(encoding="utf-8"))


def present(*fragments):
    missing = [f for f in fragments if _norm(f) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


@pytest.fixture(scope="module")
def world():
    task = SeqTask("brackets", 8)
    seqs = task.all_sequences()
    ref = Policy.for_task(task)
    for _ in range(2):
        pg.sft_step(ref, [task.as_trajectory(s) for s in seqs if task.verify(s)], lr=1.0)
    ok = np.array([task.verify(s) for s in seqs])
    return task, seqs, ref, ref.sequence_probs(task, seqs), ok


def test_s1_rl_loop_and_rollout_cost(world):
    task, _, ref, _, _ = world
    pol = ref.copy()
    pg.train_reinforce(pol, task, np.random.default_rng(0), steps=150, batch=16, lr=0.5)
    present(f"right {pg.expected(ref, task, task.verify):.1%} of the time",
            f"take it to {pg.expected(pol, task, task.verify):.1%}")
    m7 = w.Model("7.6B policy", 7.6, 28, 4, 128)
    lengths = np.minimum(4000 * np.exp(0.7 * np.random.default_rng(0).standard_normal(8192)), 20480)
    r = w.rl_step_time(m7, w.GPUS["H100"], 64, 500, lengths)
    present(f"generation {r['generate_s']:.0f} s, training {r['train_s']:.0f} s, so rollouts are "
            f"{r['rollout_fraction']:.0%} of the step", f"only {r['batch_occupancy']:.0%} occupied on average")


def test_s2_policy_gradient_numbers(world):
    task, seqs, ref, ref_p, ok = world
    present(f"8·log(1/2) = {8 * math.log(0.5):.3f}".replace("-", "−"),
            f"14/256 = {task.random_success_rate():.1%}")
    think = ThinkTask(e0=0.8, q=0.15, max_think=16)
    p0 = Policy.for_task(think)
    v = {(b, o): pg.grad_variance(p0, think, np.random.default_rng(1), b, batch=8, trials=150, offset=o)
         for b in ("none", "mean") for o in (0.0, 5.0)}
    present(f"{v[('none', 0.0)]:.3f} with no baseline, {v[('mean', 0.0)]:.3f} with the batch mean",
            f"no-baseline variance becomes {v[('none', 5.0)]:.3f} while the baselined one stays {v[('mean', 5.0)]:.3f}")
    rows = [pg.kl_optimal(ref_p, ok, b) for b in (10, 1, 0.3, 0.1)]
    present("| E[R] = P(balanced) | " + " | ".join(f"{er:.3f}" for _, er, _ in rows) + " |",
            "| KL(π* ‖ π_ref), nats | " + " | ".join(f"{kl:.2f}" for _, _, kl in rows) + " |")
    present(f"passes it {pg.expected(ref, task, task.buggy_verify):.1%} of the time and is right "
            f"{pg.expected(ref, task, task.verify):.1%}")
    buggy = SeqTask("brackets", 8, verifier="buggy")
    for beta in (0.0, 0.3):
        pol = ref.copy()
        pg.train_reinforce(pol, buggy, np.random.default_rng(0), steps=200, batch=16, lr=0.5, ref=ref, beta=beta)
        first = sum(p for p, s in zip(pol.sequence_probs(task, seqs), seqs) if s[0] == 1)
        present(f"| β = {beta:g} | {pg.expected(pol, task, task.buggy_verify):.3f} | {pg.expected(pol, task, task.verify):.3f} "
                f"| {first:.2f} | {pg.kl_seq(pol, ref, task):.2f} |")
    rb = np.array([task.buggy_verify(s) for s in seqs])
    present(f"{(ref_p @ ok) / (ref_p @ rb):.1%} as β → 0")
    present(f"L* = {ThinkTask(e0=0.8, q=0.1, max_think=64).optimal_length(cost=0.01):.2f}")
    lengths = {}
    for c in (0.0, 0.02):
        t = ThinkTask(e0=0.8, q=0.15, max_think=16, cost=c)
        p = Policy.for_task(t)
        lengths[c] = (t.expected(p.stop_probs(t))["length"], t)
        pg.train_reinforce(p, t, np.random.default_rng(0), steps=300, batch=16, lr=2.0)
        lengths[c] = (lengths[c][0], t.expected(p.stop_probs(t))["length"], t.optimal_length())
    present(f"a mean of {lengths[0.0][0]:.1f} thinking token",
            f"grow thinking to {lengths[0.0][1]:.1f} tokens with no cost and to {lengths[0.02][1]:.1f} with c = 0.02 "
            f"(L* = {lengths[0.02][2]:.1f}")


def test_s3_preference_numbers(world):
    task, seqs, ref, ref_p, ok = world
    rng = np.random.default_rng(0)
    xa, xb = rng.standard_normal((4000, 2)), rng.standard_normal((4000, 2))
    wt = np.array([1.5, -0.5])
    a_wins = rng.random(4000) < pref.bt_prob(xa @ wt, xb @ wt)
    wfit = pref.fit_bradley_terry(np.where(a_wins[:, None], xa, xb), np.where(a_wins[:, None], xb, xa), l2=0.0, steps=1500)
    present(f"recovers ({wfit[0]:.2f}, {wfit[1]:.2f})".replace("-", "−"), f"σ(2 − 1) = {float(pref.bt_prob(2, 1)):.4f}")
    g = [pref.gae([0, 0, 1], [0.5, 0.6, 0.8], 1.0, lam) for lam in (0.0, 0.5, 1.0)]
    present("λ = 0 → (" + ", ".join(f"{x:g}" for x in g[0]) + ")", "λ = 0.5 → (" + ", ".join(f"{x:g}" for x in g[1]) + ")",
            "λ = 1 → (" + ", ".join(f"{x:g}" for x in g[2]) + ")")
    present(f"= {float(pref.dpo_loss(1, -1, 0.1)):.6f}; zero margin gives ln 2 = {math.log(2):.6f}",
            f"1/(2β) (5 at β = 0.1")
    R = 3.0 * ok
    pistar, _, _ = pg.kl_optimal(ref_p, R, 1.0)
    rng = np.random.default_rng(0)
    ia, ib = rng.choice(256, 4096, p=ref_p), rng.choice(256, 4096, p=ref_p)
    wins = rng.random(4096) < pref.bt_prob(R[ia], R[ib])
    pairs = [(seqs[i], seqs[j]) if x else (seqs[j], seqs[i]) for i, j, x in zip(ia, ib, wins)]
    w_rm = pref.fit_bradley_terry([[task.verify(c)] for c, _ in pairs], [[task.verify(r)] for _, r in pairs],
                                  l2=0.0, steps=3000, lr=3.0)
    rlhf = ref.copy()
    pg.train_reinforce(rlhf, SeqTask("brackets", 8, reward_fn=lambda s: w_rm[0] * task.verify(s)),
                       np.random.default_rng(1), steps=400, batch=16, lr=0.3, ref=ref, beta=1.0)
    data, dpo = pref.encode_pairs(task, pairs), ref.copy()
    for _ in range(151):
        m = pref.dpo_step(dpo, ref, data, beta=1.0, lr=5.0)
    p = dpo.sequence_probs(task, seqs)
    ir = np.array([pref.implicit_reward(dpo, ref, task, s, 1.0) for s in seqs])
    present(f"({np.mean(ok[ia] == ok[ib]):.0%} are ties", f"| reference | {ref_p @ ok:.3f} |",
            f"| {pistar @ ok:.3f} |", f"r̂ = {w_rm[0]:.2f}·balanced, then REINFORCE with β = 1 | {pg.expected(rlhf, task, task.verify):.3f} |",
            f"KL(π_DPO ‖ π*) = {np.sum(p * np.log(p / pistar)):.4f} | {p @ ok:.3f} |",
            f"by {ir[ok > 0].mean() - ir[ok == 0].mean():.2f} (true gap 3)",
            f"rewards/chosen {m['rewards/chosen']:.3f}, rewards/rejected {m['rewards/rejected']:.3f}, margins "
            f"+{m['rewards/margins']:.3f}, accuracies {m['rewards/accuracies']:.3f}".replace("-", "−"))
    cat = pref.response_catalogue()
    rc = np.exp(-cat["padding"] / 150)
    rc /= rc.sum()
    ch, rj = pref.length_biased_prefs(cat, 3000, length_weight=0.6, ref_probs=rc)
    X = np.stack([cat["quality"], cat["length"] / 100], axis=1)
    wl = pref.fit_bradley_terry(X[ch], X[rj], steps=2000)
    present(f"r̂ = {wl[0]:.2f}·quality + {wl[1]:.2f}·length/100")
    rows = [pg.kl_optimal(rc, X @ wl, b) for b in (10, 1, 0.5, 0.3, 0.1)]
    fmt = lambda x: f"{x:.3f}".replace("-", "−")
    present("| KL, nats | 0 | " + " | ".join(f"{kl:.2f}" for _, _, kl in rows) + " |",
            f"| reward-model score | {rc @ (X @ wl):.2f} | " + " | ".join(f"{er:.2f}" for _, er, _ in rows) + " |",
            f"| true quality | {fmt(rc @ cat['quality'])} | " + " | ".join(fmt(pi @ cat["quality"]) for pi, _, _ in rows) + " |",
            f"| mean length (tokens) | {rc @ cat['length']:.0f} | " + " | ".join(f"{pi @ cat['length']:.0f}" for pi, _, _ in rows) + " |")
    betas = [10, 5, 3, 2, 1.5, 1, 0.7, 0.5, 0.3, 0.1]
    best = max(betas, key=lambda b: pg.kl_optimal(rc, X @ wl, b)[0] @ cat["quality"])
    pig = pg.kl_optimal(rc, cat["quality"], 0.1)[0]
    present(f"best at β = {best}, a KL of {pg.kl_optimal(rc, X @ wl, best)[2]:.2f} nats",
            f"reaches {pig @ cat['quality']:.3f} at {pig @ cat['length']:.0f} tokens")


def test_s4_grpo_numbers(world):
    task, seqs, ref, _, ok = world
    a = list(grpo.group_advantages([[1, 0, 0, 1], [1, 0, 0, 0]])) + list(grpo.group_advantages([[1] * 7 + [0], [1] * 4 + [0] * 4]))
    present(f"[1, 0, 0, 1] → ±{a[0][0]:.6f}", f"[1, 0, 0, 0] → [{a[1][0]:.4f}, −{-a[1][1]:.4f}",
            f"gets −{-a[2][-1]:.2f}, a miss on a 50/50 prompt −{-a[3][-1]:.2f}")
    present(*(f"{d} → {float(grpo.k3(0.0, v)):.7f}" for d, v in (("= 0.1", 0.1), ("−0.1", -0.1), ("0.5", 0.5))))
    p_, q_ = np.array([0.5, 0.3, 0.2]), np.array([0.3, 0.3, 0.4])
    x = np.random.default_rng(0).choice(3, 100000, p=p_)
    k1, k3 = grpo.k1(np.log(p_[x]), np.log(q_[x])), grpo.k3(np.log(p_[x]), np.log(q_[x]))
    present(f"with KL {np.sum(p_ * np.log(p_ / q_)):.4f}", f"k1 mean {k1.mean():.4f} with std {k1.std():.3f} (minimum "
            f"{k1.min():.3f})".replace("-", "−"), f"k3 mean {k3.mean():.4f} with std {k3.std():.3f}")
    present(f"at most {0.01 * 1.2:.4f} with ε = 0.2 ({0.01 * 1.28:.4f} with ε_high = 0.28)", f"can reach {0.5 * 1.2:.4f}")
    pol = ref.copy()
    grpo.train_grpo(pol, task, np.random.default_rng(0), grpo.GRPOConfig(num_generations=8, num_iterations=4, lr=20.0),
                    steps=100, prompts=(0, 0))

    def eff(p):
        pr = p.sequence_probs(task, seqs)
        pv = pr[ok > 0] / pr[ok > 0].sum()
        return pr @ ok, np.exp(-(pv * np.log(pv)).sum())
    (r0, e0), (r1, e1) = eff(ref), eff(pol)
    present(f"P(correct) from {r0:.3f} to {r1:.3f}", f"falls from {e0:.1f} to {e1:.1f}")
    runs = {}
    for c in (0.0, 0.05):
        p_c = ref.copy()
        pg.train_reinforce(p_c, task, np.random.default_rng(0), steps=300, batch=16, lr=0.5, entropy_coef=c)
        runs[c] = eff(p_c)
    present(f"keep {runs[0.05][1]:.1f} effective correct answers with c = 0.05 against {runs[0.0][1]:.1f} without, at "
            f"P(correct) {runs[0.05][0]:.3f} against {runs[0.0][0]:.3f}")
    wts = {lt: grpo.token_weights([10, 50], lt, 100) for lt in ("grpo", "dapo", "dr_grpo")}
    present(*(f"| {v[0][0]:.4f} / {v[1][0]:.4f} |" for v in wts.values()))
    think = ThinkTask(e0=0.8, q=0.15, max_think=16)
    tp = Policy.for_task(think)
    tp.theta[:, 1] = math.log(0.1 / 0.9)
    present(f"{think.expected(tp.stop_probs(think))['truncated']:.1%} of completions truncated at 16 tokens")
    exact = np.zeros_like(tp.theta)
    for i in range(16):
        for j in range(2):
            up, dn = tp.copy(), tp.copy()
            up.theta[i, j] += 1e-5
            dn.theta[i, j] -= 1e-5
            exact[i, j] = (think.expected(up.stop_probs(think))["accuracy"] - think.expected(dn.stop_probs(think))["accuracy"]) / 2e-5
    cos = {}
    for lt in ("grpo", "dapo", "dr_grpo"):
        g = grpo.expected_update(tp, think, np.random.default_rng(0), lt, n_groups=600)
        cos[lt] = (g * exact).sum() / np.linalg.norm(g) / np.linalg.norm(exact)
    present(f"cosine {cos['grpo']:.2f} for `\"grpo\"`, {cos['dapo']:.2f} for `\"dapo\"`, {cos['dr_grpo']:.2f} for `\"dr_grpo\"`")
    assert [grpo.soft_overlong_penalty(n, 100, 20) for n in (80, 90, 100, 101)] == [0.0, -0.5, -1.0, -1.0]
    present("lengths 80, 90, 100, 101 score 0, −0.5, −1.0, −1")
    mixed = ThinkTask(prompts=[(0.02, 0.1)] * 3 + [(0.8, 0.15)] * 3 + [(0.995, 0.01)] * 3, max_think=16)
    stats = {}
    for ds in (False, True):
        h = grpo.train_grpo(Policy.for_task(mixed), mixed, np.random.default_rng(0),
                            grpo.GRPOConfig(num_generations=8, dynamic_sampling=ds, lr=20.0), steps=60, batch_prompts=4,
                            log_every=1)
        stats[ds] = (np.mean([x["frac_reward_zero_std"] for x in h]), np.mean([x["groups_generated"] for x in h]))
    present(f"{stats[False][0]:.0%} of generated groups are silent", f"at {stats[True][1]:.1f} groups generated per step "
            f"instead of {stats[False][1]:.1f}")


def test_s5_thinking_numbers():
    think = ThinkTask(e0=0.8, q=0.15, max_think=16)
    base = Policy.for_task(think)
    rng, pol, lens = np.random.default_rng(0), base.copy(), []
    for _ in range(3):
        kept = [s for s in pol.sample(think, rng, 512) if s.info["correct"]]
        for _ in range(20):
            pg.sft_step(pol, kept, lr=1.0)
        lens.append(think.expected(pol.stop_probs(think))["length"])
    b = think.expected(base.stop_probs(think))
    present(f"from {b['length']:.1f} to {lens[0]:.2f}, {lens[1]:.2f} and {lens[2]:.2f} tokens and accuracy from "
            f"{b['accuracy']:.3f} to {think.expected(pol.stop_probs(think))['accuracy']:.3f}")
    teacher = base.copy()
    pg.train_reinforce(teacher, think, np.random.default_rng(0), steps=300, batch=16, lr=2.0)
    student, traces = base.copy(), teacher.sample(think, np.random.default_rng(1), 1000)
    for _ in range(60):
        pg.sft_step(student, traces, lr=2.0)
    t, s = think.expected(teacher.stop_probs(think)), think.expected(student.stop_probs(think))
    present(f"reaches accuracy {s['accuracy']:.3f} against the teacher's {t['accuracy']:.3f}, with the same mean thinking "
            f"length, {t['length']:.1f} tokens")
    assert round(s["length"], 1) == round(t["length"], 1)
    present(f"costs ${w.api_cost(5000, 350, 2700):.6f}; the same call with 3,500 output tokens costs "
            f"${w.api_cost(5000, 3500, 2700):.6f}, {w.api_cost(5000, 3500, 2700) / w.api_cost(5000, 350, 2700):.2f}×")


def test_s6_test_time_compute_numbers():
    qs = ttc.question_set()
    present(f"(mean {qs['a'].mean():.2f}), q lognormal (median crack length {1 / np.median(qs['q']):,.0f} tokens)")
    accs = [ttc.accuracy(qs, L) for L in (0, 500, 1000, 2000, 4000, 8000, 16000)]
    Ls = (0, 500, 1000, 2000, 4000, 8000, 16000)
    present("| accuracy | " + " | ".join(f"{a:.3f}" for a in accs[1:]) + " |",
            "| gain per 1K tokens | " + " | ".join(f"+{(accs[i] - accs[i - 1]) / (Ls[i] - Ls[i - 1]) * 1000:.3f}" for i in range(1, 7)) + " |")
    a, q = qs["a"], qs["q"]
    waste = 1 - (a * (1 - (1 - q) ** 4000) / q + (1 - a) * 4000).mean() / 4000
    present(f"{waste:.0%} of a fixed 4,000-token think")
    rng = np.random.default_rng(0)
    bon = {}
    for noise in (0.0, 0.5, 1.0):
        vals = [ttc.best_of_n_accuracy(0.3, n, noise, rng) for n in (1, 4, 16, 64)]
        bon[noise] = vals[2]
    present(f"at n = 16, {bon[0.0]:.3f} with a verifier, {bon[0.5]:.3f} with noise 0.5, {bon[1.0]:.3f} with noise 1.0")
    for wrong in ([0.42, 0.18], [0.3, 0.3], [0.15] * 4):
        present("| " + " | ".join(f"{ttc.majority_accuracy(0.4, wrong, n):.3f}" for n in (1, 15, 31)) + " |")
    present(f"pass@5 = {ttc.pass_at_k(10, 3, 5):.6f}", f"pass@8 = {ttc.pass_at_k(64, 16, 8):.6f}",
            f"its expectation is {ttc.expected_estimate(ttc.plugin_pass_at_k, 10, 5, 0.1):.4f} against a truth of "
            f"{1 - 0.9 ** 5:.4f}", f"pass@4 = {ttc.pass_at_k(16, 4, 4):.6f} but pass^4 = {ttc.pass_hat_k(16, 4, 4):.6f}")
    fmt = lambda r: f"({r[0]}, {r[1]})"
    only_slow = ttc.question_set(viable=None)
    present(" / ".join(fmt(ttc.allocate(only_slow, b)[1]) for b in (1000, 4000, 16000)),
            " / ".join(fmt(ttc.allocate(qs, b)[1]) for b in (1000, 4000, 16000)),
            " / ".join(fmt(ttc.allocate(qs, b, method="vote")[1]) for b in (1000, 4000, 16000)))
    big = ttc.question_set(median_q=1 / 1000, viable=(4.0, 1.0), seed=1)
    small = ttc.question_set(median_q=1 / 2500, viable=(1.5, 1.0), seed=1)
    ns = (1, 2, 4, 8, 16, 32)
    best = lambda qs_, b, m: ttc.allocate(qs_, b, method=m, ns=ns)[1][2]
    present(f"({best(small, 5000, 'verifier'):.3f} vs {best(big, 1000, 'verifier'):.3f} at 1K large-model tokens)",
            f"({best(small, 5000, 'vote'):.3f} vs {best(big, 1000, 'vote'):.3f} at 1K, {best(small, 20000, 'vote'):.3f} vs "
            f"{best(big, 4000, 'vote'):.3f} at 4K)")


def test_s7_serving_numbers():
    present(f"mean {w.lognormal_mean(1500, 1.0):,.0f}, p90 {w.lognormal_quantile(1500, 1.0, 0.9):,.0f} and p99 "
            f"{w.lognormal_quantile(1500, 1.0, 0.99):,.0f}")
    kv = w.kv_per_token_kb(w.QWEN3_0_6B, "fp16")
    present(f"{kv:.0f} KiB of KV per token in fp16 ({kv * 1024:,.0f} B", f"holds {8192 * kv * 1024 / 1e9:.2f} GB against "
            f"{w.weight_gb(w.QWEN3_0_6B, 'fp16'):.2f} GB of weights")
    k = lambda L: w.kv_token_steps(1500, L)
    present(f"L = 300 → {k(300):,}; L = 3,000 → {k(3000):,} = {k(3000) / k(300):.1f}×; L = 1,500 → {k(1500) / k(300):.1f}×; "
            f"L = 8,000 → {k(8000) / k(300):.1f}×", f"({k(1000) / k(300):.1f}× at 1,000")
    H, S, rps = w.GPUS["H100"], w.MISTRAL_SMALL, 10_000 * 0.10 * 0.5 / 60
    plans = [w.plan(S, H, rps, 1500, 300), w.plan(S, H, rps, 1500, 3000), w.plan(S, H, rps, 1500, 3000, tpot_ms=20)]
    present("| request lifetime | " + " | ".join(f"{p['duration_s']:.2f} s" for p in plans) + " |",
            "| live requests (Little's law) | " + " | ".join(f"{p['concurrency']:,.1f}" for p in plans) + " |",
            "| average context; KV per session (FP8) | " + " | ".join(
                f"{p['avg_ctx']:,}; {w.kv_per_session_gb(S, p['avg_ctx'], 'fp8'):.3f} GB" for p in plans) + " |",
            "| sessions per GPU by HBM | " + " | ".join(f"{p['sessions_per_gpu']:.1f}" for p in plans) + " |",
            "| batch per GPU within the ITL SLO | " + " | ".join(f"{p['itl_batch']}" for p in plans) + " |",
            "| GPUs by memory / ITL / decode / prefill | " + " | ".join(" / ".join(
                f"{p['gpus'][c]:.2f}" for c in ("memory", "itl_slots", "decode", "prefill")) for p in plans) + " |")
    assert [p["gpus_needed"] for p in plans] == [1, 5, 3] and [p["binding"] for p in plans] == ["prefill", "memory", "itl_slots"]
    present(f"({plans[1]['gpus']['memory']:.2f} vs {plans[0]['gpus']['memory']:.3f})",
            f"{1000 * w.kv_per_session_gb(S, 3000, 'fp8'):.0f} GB of KV on an 80 GB card",
            f"arrives {w.request_duration_s(24, 1500, 2700, H):.2f} s after the request")
    t1 = w.turn_prefills(1000, [(100, 800, 200)] * 4)
    t2 = w.turn_prefills(1000, [(100, 800, 200)] * 4, keep_thinking=True)
    present("| thinking dropped: prompt / prefilled | " + " | ".join(f"{p:,} / {p - h:,}" for p, h in t1) + " |",
            "| thinking kept (one tool loop): prompt / prefilled | " + " | ".join(f"{p:,} / {p - h:,}" for p, h in t2) + " |")
    cells = {c: (w.budget_outcome(1500, 1.0, 300, 0.7, max_tokens=c), w.budget_outcome(1500, 1.0, 300, 0.7, budget=c - 300))
             for c in (2048, 4096, 8192, 16384)}
    present("| `max_tokens`: accuracy / truncated | " + " | ".join(f"{a['accuracy']:.3f} / {a['truncated']:.3f}" for a, _ in cells.values()) + " |",
            "| thinking budget (limit − 300): accuracy | " + " | ".join(f"{b['accuracy']:.3f}" for _, b in cells.values()) + " |",
            "| mean output tokens (either) | " + " | ".join(f"{b['tokens']:,.0f}" for _, b in cells.values()) + " |",
            f"({cells[4096][0]['truncated']:.1%} at a 4K cap")
    need = 1500 + w.lognormal_quantile(1500, 1.0, 0.99) + 300
    present(f"{int(math.ceil(need / 1024) * 1024):,} (a multiple of 1,024)")
    off, on = w.api_cost(5000, 350, 2700), w.api_cost(5000, 3500, 2700)
    acc = {"easy": {"off": 0.95, "on": 0.97}, "hard": {"off": 0.30, "on": 0.85}}
    share = {"easy": 0.7, "hard": 0.3}
    for pol in ({"easy": "on", "hard": "on"}, {"easy": "off", "hard": "on"}, {"easy": "off", "hard": "off"}):
        cost = sum(share[c] * (on if pol[c] == "on" else off) for c in share)
        a = sum(share[c] * acc[c][pol[c]] for c in share)
        present(f"{a:.3f} at ${w.cost_per_correct(cost, a):.6f}")
    present(f"a GPU holds {w.max_batch_for_itl(S, H, 3000, 0.020, 'fp8')} sessions")
