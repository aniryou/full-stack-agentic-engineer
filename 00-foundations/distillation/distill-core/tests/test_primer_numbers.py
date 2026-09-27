"""Every computed number the topic's PRIMER.md quotes is recomputed here and must appear verbatim.

If a formula, a seed or an experiment changes, the primer fails this test until it is updated. Cited facts (TRL
and vLLM defaults, the R1, Qwen3, Minitron and EAGLE results, prices, the T4 memory predictions) are quoted and
marked (verify) in the primer; they are not checked here.
"""
from pathlib import Path

import numpy as np
import pytest

from distillcore import (ModLang, ThinkToy, TinyLM, cost as K, divergences as D, draft as Dr, eval as E,
                         losses as L, onpolicy as op, reasoning as R, seqkd, train)
from distillcore.tinylm import fit_language


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


def test_s1_soft_targets_beat_hard_labels_and_capacity(lang, teacher):
    C = lang.contexts()
    for N in (242, 605):
        rng = np.random.default_rng(1)
        ctx = C[rng.integers(0, 121, N)]
        y, zt = lang.sample_next(ctx, rng), teacher.logits(ctx)
        hard, soft = TinyLM(11, 16, 8, seed=1), TinyLM(11, 16, 8, seed=1)
        train(hard, ctx, lambda z, i: L.hard_ce(z, y[i]), 400)
        train(soft, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 400)
        h, s = E.vs_truth(hard, lang), E.vs_truth(soft, lang)
        present(f"| {N} ({round(N / 121)}) | {h['rule_acc']:.3f} / {h['kl']:.3f} nats | {s['rule_acc']:.3f} / {s['kl']:.3f} nats |")
        if N == 242:
            present(f"two examples per context gave {s['rule_acc']:.3f} rule accuracy with soft targets and {h['rule_acc']:.3f} with labels")
    r8, r16 = E.vs_truth(fit_language(lang, 8), lang), E.vs_truth(fit_language(lang, 16), lang)
    present(f"tops out at {r8['rule_acc']:.3f} rule accuracy", f"where 16 units reach {r16['rule_acc']:.3f}",
            f"an error rate of {(1 - r8['rule_acc']) * 100:.1f}% per context")


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
    for label, T in (("T = 0 (greedy)", 0.0), ("T = 1", 1.0)):
        _, st = seqkd.pipeline(teacher, lang, bench["prompts"], 8, 12, np.random.default_rng(1), T=T)
        present(f"| {label} | {st['generated']} | {st['verified']} | {st['unique']} | {st['tokens_paid']:,} | {st['tokens_used']} |")
    present(f"(0.8)^12 = {0.8 ** 12:.3f}", f"about {12 / 0.8 ** 12:.0f} teacher tokens")
    greedy, kd = bench["greedy"], bench["kd"]
    sft = TinyLM(11, 16, 8, seed=1)
    seqkd.sft(sft, greedy, 400)
    ctx, _ = teacher.positions(greedy)
    ent = lambda m: float(-(m.probs(ctx) * np.log(m.probs(ctx))).sum(1).mean())
    present(f"Entropy (teacher {ent(teacher):.2f})", f"the teacher's {ent(teacher):.2f})")
    for label, m in (("SFT on the greedy text", sft), ("supervised KD on the greedy text (GKD λ = 0)", kd)):
        eb = seqkd.exposure_bias(m, lang, greedy, bench["prompts"], np.random.default_rng(5))
        o = eb["own_prefixes"]
        present(f"| {label} | {eb['teacher_prefixes'].mean():.3f} | {o[0]:.3f} | {o[3]:.3f} | {o[11]:.3f} | {ent(m):.3f} |")
    present(f"entropy (0.009 against the teacher's {ent(teacher):.2f})".replace("0.009", f"{ent(sft):.3f}"))


def test_s3_and_s9_the_fixed_cost():
    s = K.SHAPES["qwen2.5-1.5b"]
    api = K.fixed_cost(100_000, 1, 2000, 9.00, s.params(), H100, 11, mfu=0.4)
    t_cost = K.serving(K.SHAPES["qwen2.5-32b"], H100, 11, 2048, 0.030)["usd_per_m"]
    own = K.fixed_cost(100_000, 1, 2000, t_cost, s.params(), H100, 11, mfu=0.4)
    present(f"that is ${api['generation_usd']:,.0f}", f"makes it ${own['generation_usd']:,.2f}",
            f"6·N·D = {api['train_flops'] / 1e18:.2f} × 10¹⁸ FLOPs", f"{api['gpu_hours']:.3f} GPU-hours on an H100 at 40% MFU, "
            f"${api['train_usd']:.2f} at $11/GPU-hour", f"{own['gpu_hours']:.3f} GPU-hours (${own['train_usd']:.2f})",
            f"The teacher's tokens are {own['generation_usd'] / own['total_usd']:.0%} of it")
    present(f"(${own['generation_usd']:,.0f} self-hosted, ${api['generation_usd']:,.0f} through an API")


def test_s4_on_policy_removes_exposure_bias(lang, teacher, bench):
    C = lang.contexts()
    rows = [("(nothing: supervised KD only)", bench["kd"])]
    for lam, beta, label in ((1.0, 0.0, "GKD λ = 1, β = 0 (forward)"), (1.0, 0.5, "GKD λ = 1, β = 0.5"),
                             (1.0, 1.0, "GKD λ = 1, β = 1 (reverse)")):
        s = bench["kd"].copy()
        op.gkd_train(s, teacher, bench["prompts"], 12, 300, lam=lam, beta=beta, data=bench["greedy"], seed=1)
        rows.append((label, s))
    for label, m in rows:
        o = seqkd.own_accuracy(m, lang, bench["prompts"], 12, np.random.default_rng(5))
        present(f"| {label} | {o[3]:.3f} | {o[11]:.3f} | {E.vs_truth(m, lang)['rule_acc']:.3f} |")
    o_fwd = seqkd.own_accuracy(rows[1][1], lang, bench["prompts"], 12, np.random.default_rng(5))
    present(f"on-policy training held {o_fwd[3]:.3f}")
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
    present(f"differs by {np.linalg.norm(ex_tok - ex_seq) / np.linalg.norm(ex_seq):.0%} of the gradient's norm",
            f"total variance {np.array(B).var(0).sum():.2f} against {np.array(A).var(0).sum():.2f} per batch of 64")
    g16 = op.flops_per_prompt(8e9, 32e9, 4096, samples=16, teacher_scores=False)["total"]
    d4 = op.flops_per_prompt(8e9, 32e9, 4096, samples=4)["total"]
    present(f"cost {d4 / 1e15:.2f} × 10¹⁵ FLOPs against {g16 / 1e15:.2f} × 10¹⁵")


def test_s5_distilling_reasoning(think):
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
            f"is {task.optimal_length(0.01, q=0.05):.1f} tokens, against the teacher's {task.optimal_length(0.01):.1f}",
            f"({student.expected(task)['length']:.1f} tokens against {teacher.expected(task)['length']:.1f} in the toy)")


def test_s6_prune_then_distil(lang, teacher):
    C = lang.contexts()
    rng = np.random.default_rng(1)
    ctx = C[rng.integers(0, 121, 242)]
    lang.sample_next(ctx, rng)                                    # the same draws as notebook 01
    zt = teacher.logits(ctx)
    pruned = teacher.prune_width(ctx, 16)
    before = E.vs_truth(pruned, lang)["rule_acc"]
    train(pruned, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 100)
    fresh = TinyLM(11, 16, 8, seed=1)
    train(fresh, ctx, lambda z, i: L.kd(z, zt[i], 1.0), 100)
    present(f"scores {before:.3f} rule accuracy; 100 KD steps from the parent take it to "
            f"{E.vs_truth(pruned, lang)['rule_acc']:.3f}, where a fresh 16-unit student after the same 100 steps "
            f"reaches {E.vs_truth(fresh, lang)['rule_acc']:.3f}")


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
    for label, m in (("off-the-shelf (trained on the base language)", d["off"]),
                     ("SeqKD from the target (24,000 of its tokens)", d["seqkd"]), ("logit KD from the target", kd[16])):
        r = Dr.acceptance_on_text(target, m, text)
        present(f"| {label} | {r['alpha']:.3f} | {r['greedy']:.3f} | {r['kl']:.3f} | {Dr.speedup(r['alpha'], 4, c16):.2f}× |")
    present(f"k = 4, c = {c16:.3f}")
    cs = {H: Dr.draft_cost(m.n_params, target.n_params) for H, m in kd.items()}
    al = {H: Dr.acceptance_on_text(target, m, text)["alpha"] for H, m in kd.items()}
    present("| c | " + " | ".join(f"{cs[H]:.3f}" for H in kd) + " |",
            "| α (logit KD) | " + " | ".join(f"{al[H]:.3f}" for H in kd) + " |",
            "| speedup at k = 4 | " + " | ".join(f"{Dr.speedup(al[H], 4, cs[H]):.2f}×" for H in kd) + " |")
    off, kd16 = Dr.acceptance_on_text(target, d["off"], text)["alpha"], al[16]
    present(f"({off:.3f} → {kd16:.3f} in the toy)")
    c = K.SHAPES["qwen3-0.6b"].params() / K.SHAPES["qwen3-4b"].params()
    present(f"c ≈ {c:.3f} by weight bytes, so k = 4 gives {Dr.speedup(0.6, 4, c):.2f}×, {Dr.speedup(0.7, 4, c):.2f}× and "
            f"{Dr.speedup(0.8, 4, c):.2f}× at α = 0.6, 0.7 and 0.8")


def test_s8_measuring_a_student(lang, teacher, bench):
    C, prompts = lang.contexts(), bench["prompts"]
    onp = bench["kd"].copy()
    op.gkd_train(onp, teacher, prompts, 12, 300, lam=1.0, beta=0.0, data=bench["greedy"], seed=1)
    students = (("KD on the teacher's text (§3)", "KD on the teacher's text", bench["kd"]),
                ("+ on-policy GKD (§4)", "+ on-policy GKD", onp),
                ("8 units, KD on everything (§1's capacity gap)", "8 units, KD on everything", fit_language(lang, 8)))
    zt = teacher.logits(C)
    common = set(map(tuple, prompts))
    rare = np.array([c for c in C if tuple(c) not in common])
    ok = lambda m, x, n: lang.verify(m.sample(x, n, None, 0.0))
    for label1, label2, s in students:
        zs = s.logits(C)
        present(f"| {label1} | {E.kl(zt, zs):.3f} | {E.argmax_agreement(zt, zs):.3f} | {E.topk_overlap(zt, zs, 3):.3f} |")
        cells = []
        for x, n in ((prompts, 2), (rare, 2), (rare, 8)):
            k, m = int(ok(s, x, n).sum()), len(x)
            lo, hi = E.wilson_interval(k, m)
            cells.append(f"{k}/{m} ({lo:.3f}–{hi:.3f})" if k == m else f"{k / m:.3f} ({lo:.3f}–{hi:.3f})")
        assert ok(s, prompts, 8).all()
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
    zw = weak.logits(C)
    acc = lambda m: E.vs_truth(m, lang)["rule_acc"]
    present(f"is right on {acc(weak):.3f} of contexts", f"is right on {acc(filt):.3f} — better than its teacher",
            f"(KL {E.kl(zw, filt.logits(C)):.2f}, top-1 {E.argmax_agreement(zw, filt.logits(C)):.3f})",
            f"({acc(raw):.3f} right, KL {E.kl(zw, raw.logits(C)):.3f}, top-1 {E.argmax_agreement(zw, raw.logits(C)):.3f})")


def test_s9_serving_break_even_and_the_cascade():
    serve = {}
    for key, label in (("qwen2.5-32b", "Qwen2.5-32B (teacher)"), ("qwen2.5-1.5b", "Qwen2.5-1.5B (student)"),
                       ("qwen2.5-0.5b", "Qwen2.5-0.5B (student)")):
        s = serve[key] = K.serving(K.SHAPES[key], H100, 11, 2048, 0.030)
        usd = f"${s['usd_per_m']:.3f}" if s["usd_per_m"] > 1 else f"${s['usd_per_m']:.4f}"
        present(f"| {label} | {s['batch']} | {s['step_s'] * 1e3:.2f} ms | {s['tok_s']:,.0f} | {usd} |")
    t, s = serve["qwen2.5-32b"], serve["qwen2.5-1.5b"]
    mt, ms = K.SHAPES["qwen2.5-32b"], K.SHAPES["qwen2.5-1.5b"]
    present(f"the student is {t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token: "
            f"{mt.params() / ms.params():.0f}× fewer weight bytes and {mt.kv_bytes_per_token() / ms.kv_bytes_per_token():.0f}× "
            f"less KV per token let it run {s['batch'] / t['batch']:.0f}× the batch",
            f"batch 1 already takes {K.decode_step(mt, H100, 1, 2048) * 1e3:.1f} ms — while the 1.5B still runs batch "
            f"{K.serving(ms, H100, 11, 2048, 0.010)['batch']}",
            f"~{t['usd_per_m'] / s['usd_per_m']:.0f}× cheaper per token on the roofline")
    ll, H = K.SHAPES["llama-3.1-8b"], H100
    b = K.best_batch(ll, H, 2048, 0.010)
    step = K.decode_step(ll, H, b, 2048)
    present(f"batch {b}, {step * 1e3:.2f} ms, {b / step:,.0f} tokens/s, ${K.cost_per_million_tokens(11, b / step):.3f} per million")
    fixed = K.fixed_cost(100_000, 1, 2000, t["usd_per_m"], ms.params(), H, 11)["total_usd"]
    b50, b5 = (K.break_even(fixed, t["usd_per_m"], s["usd_per_m"], v) for v in (50e6, 5e6))
    present(f"${fixed:,.2f} ÷ (${t['usd_per_m']:.3f} − ${s['usd_per_m']:.3f} per million) — {b50['days']:.1f} days at 50 "
            f"million output tokens a day (${b50['saving_per_day']:.2f} a day saved), {b5['days']:.1f} days at 5 million",
            f"a saving of ~${t['usd_per_m'] - s['usd_per_m']:.2f} per million — {b5['days']:.0f} days")
    present(f"{K.break_even(fixed, t['usd_per_m'], s['usd_per_m'], 1e6)['days']:.1f} days at 1 million")
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
