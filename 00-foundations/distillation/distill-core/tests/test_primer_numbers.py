"""Every computed number the topic's PRIMER.md quotes is recomputed here and must appear verbatim.

If a formula, a seed or an experiment changes, the primer fails this test until it is updated. Cited facts (TRL
and vLLM defaults, the R1, Qwen3, Minitron and EAGLE results, prices, the T4 memory predictions) are quoted and
marked (verify) in the primer; they are not checked here.

Two kinds of computed number. A closed-form one — a formula on fixed inputs, an enumeration, the roofline — is
formatted from the recomputation and must match to the digit (`present()`). A trained or sampled one — a seeded
Adam run of a tiny model, or samples drawn from one — is one CPU's run: numpy's OpenBLAS picks its GEMM kernel per
microarchitecture, the kernels round differently in the last bit, and hundreds of Adam steps grow that into a
slightly different, equally valid model. Such a number is pinned with `near()` (tests/pins.py): the primer quotes
the reference run's value verbatim, and the recomputation must land within a tolerance of it — at least twice the
spread measured across OpenBLAS kernels, numpy SIMD levels and last-bit perturbations (tools/host_sensitivity.py).
The claim behind each such number (soft beats hard, on-policy removes the bias, …) is asserted on the host that
runs the test, whatever its digits.
"""
from pathlib import Path

import numpy as np
import pytest

from distillcore import (ModLang, ThinkToy, TinyLM, cost as K, divergences as D, draft as Dr, eval as E,
                         losses as L, onpolicy as op, reasoning as R, seqkd, train)
from distillcore.tinylm import fit_language
from tests.pins import near


def _norm(text: str) -> str:
    return " ".join(text.split())


PRIMER = _norm((Path(__file__).resolve().parents[2] / "PRIMER.md").read_text(encoding="utf-8"))


def present(*fragments):
    missing = [f for f in fragments if _norm(f) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


def neg(s: str) -> str:
    return s.replace("-", "−")


Z = np.array([[4.0, 3.0, 1.0, 0.0, -1.0]])
V = np.array([[3.0, 3.5, 0.0, 0.5, -1.0]])
H100 = K.GPUS["h100"]


def test_s1_the_roofline_and_capacity_view():
    rows = {}
    for key, label in (("qwen2.5-32b", "32B (teacher)"), ("qwen2.5-1.5b", "1.5B (student)")):
        m = K.SHAPES[key]
        w, kv = K.weight_gb(m.params()), m.kv_bytes_per_token()
        rows[key] = (m, w, kv)
        present(f"| {label} | {m.params() / 1e9:.2f} B | {w:.2f} GB | {kv:,.0f} B | {2 * m.params() / 1e9:.2f} GFLOP | "
                f"{K.decode_step(m, H100, 1, 1024) * 1e3:.3f} ms | {K.sessions_per_gpu(80, w, kv / 1e3, 2048):.2f} |")
    t, s = rows["qwen2.5-32b"][0], rows["qwen2.5-1.5b"][0]
    present(f"The weights are {t.params() / s.params():.1f}× smaller and the KV per token "
            f"{t.kv_bytes_per_token() / s.kv_bytes_per_token():.1f}× smaller")
    w, kv = rows["qwen2.5-32b"][1], rows["qwen2.5-32b"][2] / 1e3
    tp2, one = K.sessions_per_gpu(160, w, kv, 2048), K.sessions_per_gpu(80, rows["qwen2.5-1.5b"][1], rows["qwen2.5-1.5b"][2] / 1e3, 2048)
    present(f"leave for KV ({80 * 0.9 - w:.2f} GB of the usable {80 * 0.9:.0f})",
            f"the 32B holds {tp2:.1f} such sessions, {tp2 / 2:.1f} per GPU, and the per-GPU ratio is {one / (tp2 / 2):.0f}×")


def test_s1_soft_targets_beat_hard_labels_and_capacity(lang, teacher):
    C = lang.contexts()
    #       (hard labels, soft targets): rule accuracy and KL to the truth in the reference run; the hard-label runs
    #       land on the same model on every kernel measured, the soft-target runs within 0.05 and a factor of 2.8
    pins = {242: ((0.678, 3.472), (0.876, 0.617)), 605: ((0.917, 1.303), (0.992, 0.108))}
    for N in (242, 605):
        rng = np.random.default_rng(1)
        ctx = C[rng.integers(0, 121, N)]
        y, zt = lang.sample_next(ctx, rng), teacher.logits(ctx)
        hard, soft = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
        train(hard, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
        train(soft, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 400)
        h, s = E.vs_truth(hard, lang), E.vs_truth(soft, lang)
        assert s["rule_acc"] > h["rule_acc"] + 0.02 and s["kl"] < h["kl"] / 2      # the claim, on this host
        (ha, hk), (sa, sk) = pins[N]
        present(f"| {N} ({round(N / 121)}) | {near(h['rule_acc'], ha, 0.005):.3f} / {near(h['kl'], hk, 0.01):.3f} nats | "
                f"{near(s['rule_acc'], sa, 0.05):.3f} / {near(s['kl'], sk, factor=4):.3f} nats |")
        if N == 242:
            present(f"two examples per context gave {sa:.3f} rule accuracy with soft targets and {ha:.3f} with labels")
    r8, r16 = E.vs_truth(fit_language(lang, 8), lang), E.vs_truth(fit_language(lang, 16), lang)
    a8 = near(r8["rule_acc"], 0.934, 0.005)
    present(f"tops out at {a8:.3f} rule accuracy", f"where 16 units reach {near(r16['rule_acc'], 1.0, 0.005):.3f}",
            f"an error rate of {(1 - a8) * 100:.1f}% per context")


def test_s2_soft_targets_temperature_and_the_limit():
    for label, z, T in (("teacher, T = 1", Z, 1), ("teacher, T = 2", Z, 2), ("student, T = 1", V, 1)):
        present(f"| {label} | " + " | ".join(f"{x:.4f}" for x in L.softmax(z, T)[0]) + " |")
    present(f"KL(p ‖ q) = {L.kl(L.softmax(Z), L.softmax(V))[0]:.5f} nats at T = 1")
    g = L.kd(V, Z, 2.0, scale=False)[1][0]
    present(neg("(" + ", ".join(f"{x:.6f}" for x in g) + ")"))
    Ts = (1, 2, 4, 10, 100, 1000)
    present("| KL(p_T ‖ q_T) | " + " | ".join(f"{L.kd(V, Z, T, scale=False)[0]:.5f}" for T in Ts) + " |",
            "| T²·KL | " + " | ".join(f"{L.kd(V, Z, T)[0]:.5f}" for T in Ts) + " |")
    present(f"{L.logit_mse(V, Z)[0]:.5f} for the example", f"{L.label_noise(np.array([0.8, 0.1, 0.1]))[0]:.2f} per example")
    p = L.softmax(Z)
    vals = {b: L.gkd(V, p, b)[0] for b in (0.0, 0.1, 0.5, 0.9, 1.0)}
    present(f"{vals[0.0]:.6f} at β = 0, {vals[0.1]:.6f} at 0.1, {vals[0.5]:.6f} at 0.5, {vals[0.9]:.6f} at 0.9, "
            f"{vals[1.0]:.6f} at β = 1")


def test_s2_forward_against_reverse_kl():
    p = D.bimodal()
    f, j, r = D.fit_bump(p, "forward"), D.fit_bump(p, "jsd", 0.5), D.fit_bump(p, "reverse")
    for label, x, val in (("forward KL", f, f"{f['value']:.4f}"), ("JSD, β = 0.5", j, f"{j['value']:.4f}"),
                          ("reverse KL", r, f"{r['value']:.4f} = ln 2")):
        present(f"| {label} | {x['mu']:.1f}, {x['s']:.1f} | {val} | {x['mass_where_teacher_is_empty']:.3f} | "
                f"{x['teacher_mass_uncovered']:.2f} |")
    present(f"puts {f['mass_where_teacher_is_empty']:.1%} of its samples")
    assert D.fit_bump(p, "jsd", 0.6)["mu"] == 5.0 and D.fit_bump(p, "jsd", 0.7)["mu"] == 2.0
    present("flips from covering to seeking between β = 0.6 and β = 0.7",
            f"{round(f['mass_where_teacher_is_empty'] * 100)}% of the toy student's samples")


def test_s3_pipeline_bookkeeping_and_exposure_bias(lang, teacher, bench):
    for label, T, verified, unique in (("T = 0 (greedy)", 0.0, 160, 20), ("T = 1", 1.0, 16, 11)):
        _, st = seqkd.pipeline(teacher, lang, bench["prompts"], 8, 12, np.random.default_rng(1), T=T)
        assert st["generated"] == 160 and st["tokens_paid"] == 1920 and st["tokens_used"] == 12 * st["unique"]
        v, u = near(st["verified"], verified, 0 if T == 0 else 4), near(st["unique"], unique, 0 if T == 0 else 4)
        present(f"| {label} | {st['generated']} | {v} | {u} | {st['tokens_paid']:,} | {12 * u} |")
    present(f"(0.8)^12 = {0.8 ** 12:.3f}", f"about {12 / 0.8 ** 12:.0f} teacher tokens")
    greedy, kd = bench["greedy"], bench["kd"]
    sft = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(sft, greedy, 400)
    ctx, _ = teacher.positions(greedy)
    ent = lambda m: float(-(m.probs(ctx) * np.log(m.probs(ctx))).sum(1).mean())
    e_t, e_s = near(ent(teacher), 0.64, 0.02), near(ent(sft), 0.009, 0.005)
    present(f"Entropy (teacher {e_t:.2f})", f"entropy ({e_s:.3f} against the teacher's {e_t:.2f})")
    #       on the teacher's prefixes | own prefixes at 1, 4, 12 | own output right through 4, through 12 | entropy
    pins = {"SFT on the greedy text": ((1.0, 1.0, 0.999, 0.999, 0.999, 0.994, 0.009), (0.005,) * 7),
            "supervised KD on the greedy text (GKD λ = 0)": ((1.0, 1.0, 0.788, 0.774, 0.650, 0.190, 0.652),
                                                              (0.005, 0.005, 0.03, 0.06, 0.03, 0.04, 0.02))}
    ebs = {}
    for label, m in (("SFT on the greedy text", sft), ("supervised KD on the greedy text (GKD λ = 0)", kd)):
        eb = ebs[label] = seqkd.exposure_bias(m, lang, greedy, bench["prompts"], np.random.default_rng(5))
        o, a = eb["own_prefixes"], eb["all_right_so_far"]
        got = (eb["teacher_prefixes"].mean(), o[0], o[3], o[11], a[3], a[11], ent(m))
        present(f"| {label} | " + " | ".join(f"{near(x, p, tol):.3f}" for x, p, tol in zip(got, *pins[label])) + " |")
    k, t = ebs["supervised KD on the greedy text (GKD λ = 0)"], seqkd.exposure_bias(teacher, lang, greedy, bench["prompts"], np.random.default_rng(5))
    o, a, r, tr = k["own_prefixes"], k["all_right_so_far"], k["sampled_on_rule"], t["sampled_on_rule"]
    assert abs(r[0] - tr[0]) < 0.02 and r[3] < tr.min() - 0.1       # it slips more often than the teacher after position 1
    assert np.ptp(o[3:]) < 0.05 and np.all(np.diff(a) < 0)          # flat per position; compounding per output
    o3, o11, a3, a11 = near(o[3], 0.788, 0.03), near(o[11], 0.774, 0.06), near(a[3], 0.650, 0.03), near(a[11], 0.190, 0.04)
    present(f"its samples follow the rule {near(r[0], 0.792, 0.01):.3f} of the time at position 1, like the teacher's, then "
            f"{near(r[3], 0.636, 0.03):.3f} by position 4, where the teacher's stay between {near(tr.min(), 0.78, 0.01):.2f} "
            f"and {near(tr.max(), 0.81, 0.01):.2f}",
            f"per-position accuracy drops to {o3:.3f} by position 4 and then stays there ({o11:.3f} at position 12)",
            f"falls to {a3:.3f} by position 4 and {a11:.3f} by position 12, where the teacher's stays at "
            f"{near(t['all_right_so_far'][-1], 1.0, 0.005):.3f}", f"only {a11:.3f} of its outputs were right all the way to position 12")


def test_s3_and_s9_the_fixed_cost():
    s = K.SHAPES["qwen2.5-1.5b"]
    api = K.fixed_cost(100_000, 1, 2000, 9.00, s.params(), H100, 11, mfu=0.4)
    t_cost = K.serving(K.SHAPES["qwen2.5-32b"], H100, 11, 2048, 0.030, n_gpus=2)["usd_per_m"]
    own = K.fixed_cost(100_000, 1, 2000, t_cost, s.params(), H100, 11, mfu=0.4)
    present(f"that is ${api['generation_usd']:,.0f}", f"at §9's roofline cost, makes it ${own['generation_usd']:,.2f}",
            f"6·N·D = {api['train_flops'] / 1e18:.2f} × 10¹⁸ FLOPs", f"{api['gpu_hours']:.3f} GPU-hours on an H100 at 40% MFU, "
            f"${api['train_usd']:.2f} at $11/GPU-hour", f"{own['gpu_hours']:.3f} GPU-hours (${own['train_usd']:.2f})",
            f"The teacher's tokens are {own['generation_usd'] / own['total_usd']:.0%} of it self-hosted and "
            f"{api['generation_usd'] / api['total_usd']:.0%} through the API",
            f"${own['generation_usd']:,.2f} on the 32B self-hosted at TP = 2")
    present(f"(${own['generation_usd']:,.0f} on a self-hosted teacher, ${api['generation_usd']:,.0f} through an API")


def test_s4_on_policy_removes_exposure_bias(lang, teacher, bench):
    C = lang.contexts()
    #       own prefixes at position 4, at 12, rule accuracy over all 121 contexts: (reference run, tolerance) — the
    #       on-policy runs are the primer's least reproducible numbers after §7's small drafts (tests/pins.py)
    rows = [("(nothing: supervised KD only)", bench["kd"], ((0.788, 0.03), (0.774, 0.06), (0.198, 0.04)))]
    for lam, beta, label, pins in ((1.0, 0.0, "GKD λ = 1, β = 0 (forward)", ((0.994, 0.02), (0.984, 0.08), (0.942, 0.10))),
                                   (1.0, 0.5, "GKD λ = 1, β = 0.5", ((0.957, 0.05), (0.831, 0.15), (0.636, 0.15))),
                                   (1.0, 1.0, "GKD λ = 1, β = 1 (reverse)", ((0.930, 0.06), (0.754, 0.16), (0.504, 0.15)))):
        s = bench["kd"].copy()
        op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=lam, beta=beta, data=bench["greedy"], seed=1)
        rows.append((label, s, pins))
    got = {}
    for label, m, pins in rows:
        o = seqkd.own_accuracy(m, lang, bench["prompts"], 12, np.random.default_rng(5))
        got[label] = (o[3], o[11], E.vs_truth(m, lang)["rule_acc"])
        present(f"| {label} | " + " | ".join(f"{near(x, p, tol):.3f}" for x, (p, tol) in zip(got[label], pins)) + " |")
    kd_only, fwd, mid, rev = (got[label] for label, _, _ in rows)
    assert fwd[0] > 0.96 and fwd[1] > 0.90 and fwd[2] > kd_only[2] + 0.5   # on-policy data fixes it …
    assert all(f > m and f > r for f, m, r in zip(fwd, mid, rev))            # … forward KL fastest on every column,
    assert fwd[2] > rev[2] + 0.25                                            # reverse KL far behind (β = 0.5 sits between
    #                                                                          them in the reference run; the two overlap
    #                                                                          across kernels, so that is not asserted)
    present(f"on-policy training held {near(fwd[0], 0.994, 0.02):.3f}")
    sft = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(sft, bench["greedy"], 400)
    after, at_start = {}, {}
    #       at the start: rule accuracy, mass on the right token where wrong; after 300 GKD steps at β = 0, at β = 1
    starts = (("the KD student (§3)", bench["kd"], ((0.198, 0.04), (0.029, 0.02), (0.942, 0.10), (0.504, 0.15))),
              ("the SFT student (§3)", sft, ((0.190, 0.005), (0.007, 0.005), (0.950, 0.10), (0.372, 0.20))),
              ("a fresh 16-unit student", TinyLM(11, 16, 8, seed=1), ((0.074, 0.005), (0.083, 0.005), (0.992, 0.02), (0.479, 0.30))))
    for label, start, pins in starts:
        v = at_start[label] = E.vs_truth(start, lang)
        accs = []
        for beta in (0.0, 1.0):
            s = start.copy()
            op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=1.0, beta=beta, data=bench["greedy"], seed=1)
            after[(label, beta)] = E.vs_truth(s, lang)
            accs.append(after[(label, beta)]["rule_acc"])
        assert accs[1] < accs[0] - 0.25                             # reverse KL is slow from every start here
        cells = zip((v["rule_acc"], v["wrong_right_q"], accs[0], accs[1]), pins)
        present(f"| {label} | " + " | ".join(f"{near(x, p, tol):.3f}" for x, (p, tol) in cells) + " |")
    kd_q, sft_q = at_start["the KD student (§3)"]["wrong_right_q"], at_start["the SFT student (§3)"]["wrong_right_q"]
    present(f"they give the right token {near(kd_q, 0.029, 0.02):.3f} and {near(sft_q, 0.007, 0.005):.3f}, "
            f"so starting from them does not help here")
    f1 = after[("a fresh 16-unit student", 1.0)]
    present(f"it puts {near(f1['wrong_top_q'], 0.757, 0.2):.3f} on a wrong token and "
            f"{near(f1['wrong_right_q'], 0.034, 0.06):.3f} on the right one")
    p_t, v_wrong = np.array([0.8, 0.1, 0.0999, 0.0001]), np.array([0.0, 0.0, 0.0, 9.0])
    ratio = np.linalg.norm(L.gkd(v_wrong[None], p_t[None], 1.0)[1]) / np.linalg.norm(L.softmax(v_wrong) - p_t)
    present(f"the reverse-KL gradient is {ratio:.4f} of the forward one")


def test_s4_the_policy_gradient_identities():
    t5, s5, prompt = fit_language(ModLang(5, 0.2), 16, steps=500), TinyLM(5, 4, 3, seed=1), np.array([1, 2])
    _, _, seqs = op.exact_seq_kl(s5, t5, prompt, 3)
    g = op.exact_pg_grad(s5, t5, prompt, 3)
    dirs = {k: np.random.default_rng(0).normal(size=a.shape) for k, a in s5.p.items()}
    sp, sm = s5.copy(), s5.copy()
    for k in dirs:
        sp.p[k] += 1e-5 * dirs[k]
        sm.p[k] -= 1e-5 * dirs[k]
    fd = -(op.exact_seq_kl(sp, t5, prompt, 3)[0] - op.exact_seq_kl(sm, t5, prompt, 3)[0]) / 2e-5
    an = sum(float((g[k] * dirs[k]).sum()) for k in dirs)
    assert abs(fd - an) / abs(an) < 1e-8
    present(f"over all {len(seqs)} three-token continuations of a 5-token language", "to 9 digits")
    flat = lambda d: np.concatenate([d[k].ravel() for k in sorted(d)])
    ex_seq, ex_tok = flat(g), flat(op.exact_pg_grad(s5, t5, prompt, 3, per_token=True))
    rng, A, B = np.random.default_rng(0), [], []
    for _ in range(200):
        s = s5.sample(np.tile(prompt, (64, 1)), 3, rng)
        A.append(flat(op.pg_grad(s5, t5, s)))
        B.append(flat(op.pg_grad(s5, t5, s, per_token=True)))
    bias = np.linalg.norm(ex_tok - ex_seq) / np.linalg.norm(ex_seq)
    present(f"differs by {near(bias, 0.31, 0.02):.0%} of the gradient's norm",
            f"total variance {near(np.array(B).var(0).sum(), 1.86, 0.05):.2f} against "
            f"{near(np.array(A).var(0).sum(), 3.15, 0.05):.2f} per batch of 64")
    g = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False)
    gr = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False, reference_params=8e9)
    d = op.flops_per_prompt(8e9, 32e9, 4096, samples=16)
    present(f"that is {d['per_token'] / 1e9:.0f} GFLOP per student token, against {g['per_token'] / 1e9:.0f} for GRPO "
            f"({gr['per_token'] / 1e9:.0f} with a reference model's forward pass",
            "makes each on-policy token twice as dear",
            f"that is {d['total'] / 1e15:.2f} × 10¹⁵ FLOPs against {g['total'] / 1e15:.2f} × 10¹⁵ "
            f"({gr['total'] / 1e15:.2f} × 10¹⁵ with the reference)")
    assert d["per_token"] == 2 * g["per_token"] and d["per_token"] == 2 * 8e9 + 6 * 8e9 + 2 * 32e9


def test_s5_distilling_reasoning(think):
    """The reasoning toy's numbers stay exact: a 32-entry table policy contracts last-bit differences away, and its
    REINFORCE run and samples came out identical on every kernel and perturbation tools/host_sensitivity.py tried."""
    task, teacher = think
    rows = [("untrained student", R.LengthPolicy(32)), ("teacher (RL, 16,000 rollouts)", teacher)]
    L_, ok = R.traces(teacher, task, np.random.default_rng(1), 1000)
    student = R.distil(L_, ok)
    rows.append(("student, SFT on 1,000 traces", student))
    for label, pol in rows:
        e = pol.expected(task)
        present(f"| {label} | {e['accuracy']:.3f} | {e['length']:.2f} | {e['p90']} | {e['tokens_per_correct']:.2f} |")
    accs = []
    for n in (10, 30, 100, 300, 1000):
        Ln, okn = R.traces(teacher, task, np.random.default_rng(2), n)
        accs.append(R.distil(Ln, okn).expected(task)["accuracy"])
    present("reach " + ", ".join(f"{a:.3f}" for a in accs[:-1]) + f" and {accs[-1]:.3f}")
    for label, keep, ml in (("all (plain SeqKD)", "all", None), ("correct only (rejection sampling)", "correct", None),
                            ("correct and L ≤ 16", "correct", 16), ("correct and L ≤ 12", "correct", 12)):
        kept = int(((ok if keep == "correct" else np.ones_like(ok)) & (L_ <= (ml or 99)) & (L_ < 32)).sum())
        e = R.distil(L_, ok, keep=keep, max_len=ml).expected(task)
        present(f"| {label} | {kept} | {e['accuracy']:.3f} | {e['length']:.2f} | {e['tokens_per_correct']:.2f} |")
    rl = [R.reinforce(task, np.random.default_rng(3), n).expected(task)["accuracy"] for n in (1000, 4000, 16000)]
    present(f"SFT on 1,000 teacher traces reaches {student.expected(task)['accuracy']:.3f}; REINFORCE on the student "
            f"with 1,000 rollouts reaches {rl[0]:.3f}, with 4,000 {rl[1]:.3f}, with 16,000 {rl[2]:.3f}")
    weak = student.expected(task, q=0.05)
    present(f"copies the teacher's {weak['length']:.2f}-token thinking exactly and scores {weak['accuracy']:.3f}, "
            f"not {teacher.expected(task)['accuracy']:.3f}",
            f"is {task.optimal_length(0.01, q=0.05):.1f} tokens, not the {student.expected(task)['length']:.1f} it copied",
            f"its {teacher.expected(task)['length']:.1f} is where 16,000 rollouts left it",
            f"({student.expected(task)['length']:.1f} tokens against {teacher.expected(task)['length']:.1f} in the toy)")


def prune_vs_fresh(lang, teacher, steps, draws=range(1, 6), inits=range(1, 5)):
    """Rule accuracy of the pruned teacher and of fresh 16-unit students after `steps` KD steps on 242 contexts,
    over several data draws (draw 1 is notebook 01's worked example)."""
    C, pruned, fresh = lang.contexts(), [], []
    for d in draws:
        rng = np.random.default_rng(d)
        ctx = C[rng.integers(0, 121, 242)]
        lang.sample_next(ctx, rng)                                # the same draws as notebook 01
        zt = teacher.logits(ctx)
        for m, out in [(teacher.prune_width(ctx, 16), pruned)] + [(TinyLM(11, 16, 8, seed=s), fresh) for s in inits]:
            if steps:
                train(m, ctx, lambda z, i: L.kd(z, zt[i], 1.0), steps)
            out.append(E.vs_truth(m, lang)["rule_acc"])
    return np.array(pruned), np.array(fresh)


def test_s6_prune_then_distil(lang, teacher):
    runs = {n: prune_vs_fresh(lang, teacher, n) for n in (0, 20, 100)}
    (p20, f20), (p100, f100) = runs[20], runs[100]
    assert p20.min() > f20.max()                                  # a head start at every seed
    assert abs(p100.mean() - f100.mean()) < f100.std() + p100.std()   # no better destination: within seed noise
    #       mean (min–max) of rule accuracy after 0, 20 and 100 KD steps in the reference run; the mean of five
    #       draws lands within 0.02 of it on the kernels measured, a single draw within 0.05 (tests/pins.py)
    pins = {"pruned from the teacher": ((0.317, 0.289, 0.355), (0.678, 0.612, 0.727), (0.868, 0.843, 0.917)),
            "fresh": ((0.097, 0.074, 0.132), (0.278, 0.240, 0.331), (0.853, 0.818, 0.934))}
    for (label, cells), col in zip(pins.items(), (0, 1)):
        row = [f"{near(a.mean(), m, 0.04):.3f} ({near(a.min(), lo, 0.10):.3f}–{near(a.max(), hi, 0.10):.3f})"
               for (m, lo, hi), a in zip(cells, (runs[n][col] for n in (0, 20, 100)))]
        present(f"| {label} | " + " | ".join(row) + " |")


@pytest.fixture(scope="module")
def draft_world():
    base, dialect = ModLang(11, 0.2), ModLang(11, 0.2, skew=1.0)
    C = base.contexts()
    target = fit_language(dialect, 64)
    text = target.sample(C[np.random.default_rng(1).integers(0, 121, 500)], 12, np.random.default_rng(2))
    corpus = target.sample(C[np.random.default_rng(3).integers(0, 121, 2000)], 12, np.random.default_rng(4))
    PT = target.probs(C)

    def kd_draft(H):
        d = TinyLM(11, H, 8, seed=3)
        train(d, C, lambda z, i: L.soft_ce(z, PT[i]), 1500)
        return d

    sk = TinyLM(11, 16, 8, seed=3)
    seqkd.sft(sk, corpus, 1500, batch=512)
    return target, text, {"off": fit_language(base, 16, seed=3), "seqkd": sk}, {H: kd_draft(H) for H in (4, 8, 16, 32)}


def test_s7_the_formulas_on_the_serving_primers_numbers():
    P, Q = np.array([0.5, 0.3, 0.15, 0.05]), np.array([0.2, 0.2, 0.2, 0.4])
    a = Dr.acceptance_rate(P, Q)
    kb = Dr.best_k(a, 0.1)
    present(f"α = {a:.1f}, {Dr.expected_tokens(a, 4):.4f} tokens per pass at k = 4, and at c = 0.1 the best depth is "
            f"k = {kb} with a {Dr.speedup(a, kb, 0.1):.4f}× speedup",
            f"{Dr.greedy_acceptance(P, Q):.2f} against {a:.1f} on the primer's p and q",
            f"{Dr.vllm_view(a, 4)['draft_acceptance_rate']:.4f} at α = 0.6, k = 4")


def test_s7_distilled_against_off_the_shelf_and_size(draft_world):
    target, text, d, kd = draft_world
    c16 = Dr.draft_cost(kd[16].n_params, target.n_params)
    r = {key: Dr.acceptance_on_text(target, m, text) for key, m in (("off", d["off"]), ("seqkd", d["seqkd"]), ("kd16", kd[16]))}
    assert r["off"]["alpha"] < r["seqkd"]["alpha"] < r["kd16"]["alpha"]           # the claim, on this host
    cap = target.probs(target.positions(text)[0]).max(1).mean()                      # the target's mean top-1 probability
    assert all(x["greedy"] <= cap + 1e-9 for x in r.values())                        # caps greedy drafting, exactly
    #       α, greedy acceptance, KL(p ‖ q), speedup at k = 4: (reference run, tolerance); the KL within a factor
    pins = {"off": ("off-the-shelf (trained on the base language)", (0.890, 0.005), (0.800, 0.005), (0.147, 0.01), (1.86, 0.02)),
            "seqkd": ("SeqKD from the target (24,000 of its tokens)", (0.933, 0.08), (0.795, 0.03), (0.038, None), (2.03, 0.30)),
            "kd16": ("logit KD from the target", (0.988, 0.05), (0.800, 0.02), (0.010, None), (2.26, 0.25))}
    for key, (label, (a, ta), (g, tg), (k, tk), (s, ts)) in pins.items():
        x = r[key]
        kl = near(x["kl"], k, tk) if tk else near(x["kl"], k, factor=4)
        present(f"| {label} | {near(x['alpha'], a, ta):.3f} | {near(x['greedy'], g, tg):.3f} | {kl:.3f} | "
                f"{near(Dr.speedup(x['alpha'], 4, c16), s, ts):.2f}× |")
    present(f"k = 4, c = {c16:.3f}")
    cs = {H: Dr.draft_cost(m.n_params, target.n_params) for H, m in kd.items()}
    al = {H: Dr.acceptance_on_text(target, m, text)["alpha"] for H, m in kd.items()}
    assert al[4] < al[8] < al[16] < al[32]                                            # α rises with width …
    assert max(kd, key=lambda H: Dr.speedup(al[H], 4, cs[H])) == 16                   # … the speedup peaks at 16
    #       the 4- and 8-unit drafts are the primer's least reproducible numbers: too small to hold the target, they
    #       end 1,500 steps at different compromises on different kernels (α within 0.05 and 0.14 of the reference)
    pa = {4: (0.589, 0.12), 8: (0.722, 0.30), 16: (0.988, 0.05), 32: (0.995, 0.02)}
    ps = {4: (1.56, 0.35), 8: (1.72, 0.80), 16: (2.26, 0.25), 32: (1.60, 0.04)}
    present("| c | " + " | ".join(f"{cs[H]:.3f}" for H in kd) + " |",
            "| α (logit KD) | " + " | ".join(f"{near(al[H], *pa[H]):.3f}" for H in kd) + " |",
            "| speedup at k = 4 | " + " | ".join(f"{near(Dr.speedup(al[H], 4, cs[H]), *ps[H]):.2f}×" for H in kd) + " |")
    present(f"({near(r['off']['alpha'], 0.890, 0.005):.3f} → {near(al[16], 0.988, 0.05):.3f} in the toy)")
    c = K.SHAPES["qwen3-0.6b"].params() / K.SHAPES["qwen3-4b"].params()
    present(f"c ≈ {c:.3f} by weight bytes, so k = 4 gives {Dr.speedup(0.6, 4, c):.2f}×, {Dr.speedup(0.7, 4, c):.2f}× and "
            f"{Dr.speedup(0.8, 4, c):.2f}× at α = 0.6, 0.7 and 0.8")


def test_s8_measuring_a_student(lang, teacher, bench):
    C, prompts = lang.contexts(), bench["prompts"]
    onp = bench["kd"].copy()
    op.gkd_train(onp, teacher, prompts, 12, 300, lam=1.0, beta=0.0, data=bench["greedy"], seed=1)
    #       agreement: KL(teacher ‖ student) (within a factor for the on-policy student), top-1, top-3; then the
    #       counts right on common n = 2 (of 20), rare n = 2 and rare n = 8 (of 101) — (reference run, tolerance)
    students = (("KD on the teacher's text (§3)", "KD on the teacher's text", bench["kd"],
                 ((5.557, 0.2), (0.198, 0.04), (0.342, 0.06)), ((20, 0), (0, 2), (0, 2))),
                ("+ on-policy GKD (§4)", "+ on-policy GKD", onp,
                 ((0.185, None), (0.942, 0.10), (0.923, 0.35)), ((20, 0), (89, 22), (69, 38))),
                ("8 units, KD on everything (§1's capacity gap)", "8 units, KD on everything", fit_language(lang, 8),
                 ((0.322, 0.005), (0.934, 0.005), (0.826, 0.01)), ((20, 0), (86, 2), (53, 2))))
    zt = teacher.logits(C)
    common = set(map(tuple, prompts))
    rare = np.array([c for c in C if tuple(c) not in common])
    ok = lambda m, x, n: lang.verify(m.sample(x, n, None, 0.0))
    for label1, label2, s, ((kl_pin, kl_tol), agree_pin, top3_pin), count_pins in students:
        zs = s.logits(C)
        kl = near(E.kl(zt, zs), kl_pin, kl_tol) if kl_tol else near(E.kl(zt, zs), kl_pin, factor=4)
        present(f"| {label1} | {kl:.3f} | {near(E.argmax_agreement(zt, zs), *agree_pin):.3f} | "
                f"{near(E.topk_overlap(zt, zs, 3), *top3_pin):.3f} |")
        assert ok(s, prompts, 8).all()                                # perfect on the slice a quick eval would use
        counts = {(len(x), n): int(ok(s, x, n).sum()) for x, n in ((prompts, 2), (rare, 2), (rare, 8))}
        assert counts[(101, 8)] <= counts[(101, 2)]                   # the gap grows with the output's length
        cells = []
        for (m, n), (k_pin, k_tol) in zip(counts, count_pins):
            k = near(counts[(m, n)], k_pin, k_tol)
            lo, hi = E.wilson_interval(k, m)
            cells.append(f"{k}/{m} ({lo:.3f}–{hi:.3f})" if k == m else f"{k / m:.3f} ({lo:.3f}–{hi:.3f})")
        present(f"| {label2} | " + " | ".join(cells) + " |")
    lo, _ = E.wilson_interval(20, 20)
    present(f"anything tighter than {lo * 100:.0f}–100%")


def test_s8_a_student_that_beats_its_teacher(lang):
    C = lang.contexts()
    ctx = C[np.random.default_rng(7).integers(0, 121, 605)]
    y = lang.sample_next(ctx, np.random.default_rng(8))
    weak = TinyLM(11, 64, 8, seed=2)
    train(weak, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
    data = seqkd.teacher_data(weak, C, 20, 1, np.random.default_rng(9))
    filt, raw = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
    seqkd.sft(filt, seqkd.keep_verified(lang, data), 600)
    seqkd.sft(raw, data, 600)
    zw, zf, zr = weak.logits(C), filt.logits(C), raw.logits(C)
    acc = lambda m: E.vs_truth(m, lang)["rule_acc"]
    a_w, a_f, a_r = acc(weak), acc(filt), acc(raw)
    assert a_f > a_w > a_r and E.argmax_agreement(zw, zf) < E.argmax_agreement(zw, zr)   # the claim, on this host
    present(f"is right on {near(a_w, 0.950, 0.005):.3f} of contexts", f"is right on {near(a_f, 0.975, 0.005):.3f} — better than its teacher",
            f"(KL {near(E.kl(zw, zf), 1.59, 0.05):.2f}, top-1 {near(E.argmax_agreement(zw, zf), 0.942, 0.005):.3f})",
            f"({near(a_r, 0.934, 0.005):.3f} right, KL {near(E.kl(zw, zr), 0.164, 0.01):.3f}, "
            f"top-1 {near(E.argmax_agreement(zw, zr), 0.959, 0.005):.3f})")


def test_s9_serving_break_even_and_the_cascade():
    serve = {}
    for key, n, label in (("qwen2.5-32b", 1, "Qwen2.5-32B (teacher), one GPU"), ("qwen2.5-32b", 2, "Qwen2.5-32B (teacher), TP = 2"),
                          ("qwen2.5-1.5b", 1, "Qwen2.5-1.5B (student)"), ("qwen2.5-0.5b", 1, "Qwen2.5-0.5B (student)")):
        s = serve[(key, n)] = K.serving(K.SHAPES[key], H100, 11, 2048, 0.030, n_gpus=n)
        usd = f"${s['usd_per_m']:.3f}" if s["usd_per_m"] > 0.1 else f"${s['usd_per_m']:.4f}"
        present(f"| {label} | {n} | {s['batch']} | {s['step_s'] * 1e3:.2f} ms | {s['tok_s']:,.0f} | {usd} |")
    t1, t, s = serve[("qwen2.5-32b", 1)], serve[("qwen2.5-32b", 2)], serve[("qwen2.5-1.5b", 1)]
    t4 = K.serving(K.SHAPES["qwen2.5-32b"], H100, 11, 2048, 0.030, n_gpus=4)
    mt, ms = K.SHAPES["qwen2.5-32b"], K.SHAPES["qwen2.5-1.5b"]
    present(f"the 32B's weights leave {80 * 0.9 - K.weight_gb(mt.params()):.2f} GB of the usable 72 for KV, room for {t1['batch']}",
            f"and the {t1['usd_per_m'] / s['usd_per_m']:.0f}× it gives is an artefact of that",
            f"it runs batch {t['batch']} at ${t['usd_per_m']:.3f} per million, and the student is "
            f"{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token: {mt.params() / ms.params():.0f}× fewer weight bytes and "
            f"{mt.kv_bytes_per_token() / ms.kv_bytes_per_token():.0f}× less KV per token let it run "
            f"{s['batch'] / (t['batch'] / 2):.0f}× the batch per GPU",
            f"At TP = 4 the teacher reaches ${t4['usd_per_m']:.3f} and the ratio {t4['usd_per_m'] / s['usd_per_m']:.0f}×",
            f"batch 1 already takes {K.decode_step(mt, H100, 1, 2048) * 1e3:.1f} ms — while the 1.5B still runs batch "
            f"{K.serving(ms, H100, 11, 2048, 0.010)['batch']}",
            f"~{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token on the roofline than a 32B teacher served on two H100s "
            f"(the {t1['usd_per_m'] / s['usd_per_m']:.0f}× you get against one H100",
            f"a 1.5B student serves {s['batch'] / (t['batch'] / 2):.0f}× the batch per GPU under the same ITL and is "
            f"~{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token on the roofline ({t1['usd_per_m'] / s['usd_per_m']:.0f}× against one H100")
    assert K.serving(mt, H100, 11, 2048, 0.010)["batch"] == 0      # one H100 misses a 10 ms ITL at batch 1
    ll, H = K.SHAPES["llama-3.1-8b"], H100
    b = K.best_batch(ll, H, 2048, 0.010)
    step = K.decode_step(ll, H, b, 2048)
    present(f"batch {b}, {step * 1e3:.2f} ms, {b / step:,.0f} tokens/s, ${K.cost_per_million_tokens(11, b / step):.3f} per million")
    own = K.fixed_cost(100_000, 1, 2000, t["usd_per_m"], ms.params(), H, 11)
    api = K.fixed_cost(100_000, 1, 2000, 9.00, ms.params(), H, 11)
    saving = t["usd_per_m"] - s["usd_per_m"]
    b50, b5, b1 = (K.break_even(own["total_usd"], t["usd_per_m"], s["usd_per_m"], v) for v in (50e6, 5e6, 1e6))
    a50, a5 = (K.break_even(api["total_usd"], t["usd_per_m"], s["usd_per_m"], v) for v in (50e6, 5e6))
    present(f"${own['total_usd']:,.2f} ÷ (${t['usd_per_m']:.3f} − ${s['usd_per_m']:.3f} per million) is "
            f"{b50['tokens'] / 1e6:.1f} million tokens: {b50['days']:.1f} days at 50 million output tokens a day "
            f"(${b50['saving_per_day']:.2f} a day saved), {b5['days']:.1f} days at 5 million, {b1['days']:.1f} days at 1 million",
            f"makes it ${api['total_usd']:,.2f} ÷ ${saving:.3f}, {a50['tokens'] / 1e6:,.1f} million tokens: "
            f"{a50['days']:.1f} days at 50 million a day, {a5['days']:.1f} at 5 million",
            f"a saving of ~${saving:.2f} per million over the teacher on two H100s — {b5['days']:.0f} days with self-hosted "
            f"data, {a5['days']:.0f} with API-bought data")
    assert abs(b50["tokens"] - 2e8) / 2e8 < 0.2                    # ≈ the volume the teacher wrote for the student
    c_t, c_s = 500 * t["usd_per_m"] / 1e6, 500 * s["usd_per_m"] / 1e6
    gate = K.cascade(c_s, c_t, (0.95, 0.30), (0.97, 0.85), 0.3, catch=0.8, false_alarm=0.1)
    present(f"every false alarm is a full teacher call ({0.7 * 0.1 * c_t / gate['cost']:.0%} of this gate's bill)")
    a_s, a_t = (0.95, 0.30), (0.97, 0.85)
    for label, kw in (("teacher only", dict(catch=1.0, false_alarm=1.0, student_first=False)),
                      ("student only", dict(catch=0.0, false_alarm=0.0)),
                      ("student first, gate catches 80% of hard, 10% false alarms", dict(catch=0.8, false_alarm=0.1)),
                      ("a perfect router up front", dict(catch=1.0, false_alarm=0.0, student_first=False))):
        r = K.cascade(c_s, c_t, a_s, a_t, 0.3, **kw)
        present(f"| {label} | {r['to_teacher']:.2f} | {r['accuracy']:.3f} | {r['cost_per_correct'] * 1e3:.3f} |")
