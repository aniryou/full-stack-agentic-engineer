"""Every computed number the topic's PRIMER.md quotes is recomputed here and must appear verbatim.

If a formula, a price row or the bundled sample changes, the primer fails this test until it is updated.
(Inputs -- 5,000 tokens, a 10 s timeout, $3.7 an hour, 19.3 ms -- are quoted, not computed.)
"""
from pathlib import Path

import pytest

from gwcore import cache, guardrails as G, keys as K, metering as M, providers as P, ratelimit as RL, routing as R
from gwcore.mcp_authz import as_metadata_urls, s256

PRIMER = " ".join((Path(__file__).resolve().parents[2] / "PRIMER.md").read_text(encoding="utf-8").split())


def present(*fragments):
    missing = [f for f in fragments if " ".join(f.split()) not in PRIMER]
    assert not missing, f"PRIMER.md no longer says: {missing}"


def test_s1_adapters_and_the_hop():
    u = {d: P.normalize_usage(d, P.PAYLOADS[d]["response"].get("usage") or P.PAYLOADS[d]["response"]["usageMetadata"])
         for d in ("openai", "anthropic", "gemini", "vllm")}
    assert all(x["prompt_tokens"] == 5000 and x["completion_tokens"] == 1550 for x in u.values())
    present("maps every one of them to 5,000 prompt and 1,550 completion tokens", "(2,300 + 2,700 + 0)", "(350 + 1,200)")
    a = R.chain_availability([0.995, 0.99], common_mode=0.0005)
    present(f"gives {a:.3%}".replace("%", " %"), f"= {a:.9f})")


def test_s2_chain_numbers():
    ind, cm = R.chain_availability([0.995, 0.99]), R.chain_availability([0.995, 0.99], 0.001)
    present(f"1 − 0.005 × 0.01 = {ind:.3%}".replace("%", " %"), f"0.999 × 0.99995 = {cm:.3%}".replace("%", " %"),
            f"({cm:.8f})", f"to {R.chain_availability([0.995, 0.99, 0.99]):.5%}".replace("%", " %"))
    c = {m: M.price_call(m, 5000, 350, 2700) for m in ("gemini-3.5-flash", "gpt-5.4-mini", "claude-haiku-4-5")}
    steps = lambda tf: [dict(p_ok=0.95, t_ok=0.6, t_fail=tf, cost_ok=c["gemini-3.5-flash"]),
                        dict(p_ok=0.99, t_ok=0.5, t_fail=0.15, cost_ok=c["gpt-5.4-mini"]),
                        dict(p_ok=0.99, t_ok=0.7, t_fail=0.15, cost_ok=c["claude-haiku-4-5"])]
    fast, slow = R.chain_cost(steps(0.15)), R.chain_cost(steps(10.0))
    present(f"| fast (a 503 in 0.15 s) | {fast['latency']:.3f} s | 5.0 × 10⁻⁶ | ${fast['cost']:.6f} |",
            f"| slow (a 10 s timeout) | {slow['latency']:.3f} s | 5.0 × 10⁻⁶ | ${slow['cost']:.6f} |",
            f"{R.chain_cost(steps(0.15)[:1])['latency']:.3f} s, but 5 % of requests fail",
            f"= {slow['latency'] - fast['latency']:.2f} s to the *mean*")
    assert fast["p_fail"] == pytest.approx(5.0e-6)


def test_s3_cache_numbers():
    present(f"costs ${M.price_call('gemini-3.5-flash', 5000, 350):.5f} uncached and "
            f"${M.price_call('gemini-3.5-flash', 5000, 350, 2700):.6f} with 2,700",
            f"{1 - M.price_call('gemini-3.5-flash', 5000, 350, 2700) / M.price_call('gemini-3.5-flash', 5000, 350):.0%} less"
            .replace("%", " %"))
    off = {r["threshold"]: r for r in cache.sweep_thresholds([0.6, 0.8, 0.9, 0.95], guard=False)}
    on = {r["threshold"]: r for r in cache.sweep_thresholds([0.6, 0.8, 0.9, 0.95], guard=True)}
    pct = lambda x: f"{100 * x:.1f} %"
    for th in (0.6, 0.8, 0.9, 0.95):
        present(f"| {th:.2f} | {pct(off[th]['hit_rate'])} | {pct(off[th]['false_hit_rate'])} | "
                f"{pct(on[th]['hit_rate'])} | {pct(on[th]['false_hit_rate'])} |")
    s, e = cache.load_sample(), cache.HashingEmbedder()
    counts = {k: sum(p["label"] == k for p in s["probes"]) for k in ("same", "near_miss", "unrelated", "uncacheable")}
    present(f"{len(s['seeds'])} cached questions, {counts['same']} paraphrases that should hit, {counts['near_miss']} near "
            f"misses that must not, {counts['unrelated']} unrelated, {counts['uncacheable']} uncacheable",
            f"over all {counts['same'] + counts['near_miss'] + counts['unrelated']} cacheable lookups")
    sim = lambda a, b: float(e.encode(a) @ e.encode(b))
    present(f"scores {sim('What were the Q3 2024 revenue figures?', 'What were the Q3 2025 revenue figures?'):.3f}",
            f"scores {sim('Is SSO included in the Pro plan?', 'Does the Pro plan include SSO?'):.3f}")
    assert on[0.6]["right"] == off[0.6]["right"] - 1                      # the shouted query the guard refuses


def test_s4_rate_limit_numbers():
    old, new = RL.lognormal_mean(300, 1), RL.lognormal_mean(1500, 1)
    present(f"(mean {old:.1f}, `ratelimit.lognormal_mean()`)", f"mean {new:,.0f}",
            f"= {1500 + new:,.0f} / {1500 + old:,.0f} = **{RL.overadmission_ratio(1500, old, new):.2f}×**")
    res = RL.compare_buckets()
    rows = [("per-request bucket, yesterday's outputs (median 300)", res["per-request, old outputs"], "{} of 600"),
            ("per-request bucket, thinking outputs (median 1,500)", res["per-request, thinking outputs"], "{} (every second)"),
            ("reserve prompt + 4,096, debit, reconcile", res["reserve prompt + 4,096"], "{}"),
            ("reserve prompt + the 16,384 cap, debit, reconcile", res["reserve prompt + cap 16,384"], "{}")]
    for name, r, over in rows:
        util = f"{r['utilisation']:.2f}" if r["utilisation"] > 0.9 else f"{r['utilisation']:.3f}"
        present(f"| {name} | {r['admitted']:,} | {r['peak_ratio']:.2f} | {over.format(r['seconds_over_limit'])} | {util} |")
    est, cap = res["reserve prompt + 4,096"], res["reserve prompt + cap 16,384"]
    present(f"pushes {res['per-request, thinking outputs']['peak_ratio']:.2f}× the limit",
            f"for {res['per-request, old outputs']['seconds_over_limit']} of 600 seconds",
            f"({cap['utilisation']:.1%} served)".replace("%", " %"), f"serves {est['utilisation']:.1%}".replace("%", " %"),
            f"{est['overrun_tokens']:,.0f} tokens overran")
    work = RL.workload(600, 12, 1500, 1500, 1.0, seed=0)
    sweep = {ro: RL.simulate(work, "reserve", limit=1_000_000, reserve_out=ro) for ro in (1024, 2048, 4096)}
    present(f"1,024 serves {sweep[1024]['utilisation']:.1%} of the limit but spends {sweep[1024]['seconds_over_limit']} "
            f"seconds over it, 2,048 serves {sweep[2048]['utilisation']:.1%} with {sweep[2048]['seconds_over_limit']} "
            f"seconds over, and 4,096 is the smallest with none".replace("%", " %"))
    assert sweep[4096]["seconds_over_limit"] == 0 < sweep[2048]["seconds_over_limit"]


def test_s5_metering_numbers():
    think, plain = M.price_call("gemini-3.5-flash", 5000, 1550, 2700), M.price_call("gemini-3.5-flash", 5000, 350, 2700)
    present(f"costs ${think:.6f} on gemini-3.5-flash, {think / plain:.2f}× the ${plain:.6f}")
    for model in ("gemini-3.5-flash", "claude-haiku-4-5", "gpt-5.4-mini", "gemini-3.5-flash-lite"):
        call = M.price_call(model, 5000, 350, 2700)
        present(f"| {model} | ${call:.7f}".rstrip("0") + f" | ${M.cost_per_million(model, 5000, 350, 2700):.4f} |")
    present(f"**${M.self_hosted_per_million(3.7, 6846.515877965477):.3f}**",
            f"(${M.self_hosted_per_million(3.7, 6846.515877965477, 0.6):.3f} at 60 %)",
            f"at $11 → ${M.self_hosted_per_million(11, 6846.515877965477):.3f}",
            f"at $0.70 → ${M.self_hosted_per_million(0.70, 294.3945617414261):.3f}")
    u = {"rag": {"prompt_tokens": 90_000_000, "completion_tokens": 1_000_000},
         "thinking": {"prompt_tokens": 5_000_000, "completion_tokens": 10_000_000}}
    tok, gpu = M.chargeback(u, 10_000, by="tokens"), M.chargeback(u, 10_000, by="gpu_seconds")
    present(f"| tokens | ${tok['rag']:,.2f} ({tok['rag'] / 100:.1f} %) | ${tok['thinking']:,.2f} |",
            f"| GPU-seconds | ${gpu['rag']:,.2f} ({gpu['rag'] / 100:.1f} %) | ${gpu['thinking']:,.2f} |",
            f"({M.chargeback.__kwdefaults__['decode_tps']:,} tokens/s)")


def test_s6_keys_numbers():
    salt = K.cache_salt("acme", b"gateway-salt-secret")
    assert len(salt) == 43 and K.valid_cache_salt(salt)
    lo, hi = K.svid_rotation_window(3600)
    present(f"{len(salt)} characters (`keys.cache_salt()`)", f"between {lo:,.0f} and {hi:,.0f} seconds before expiry, "
            f"{lo / 60:.0f}–{hi / 60:.0f} minutes")


def test_s7_guardrail_numbers():
    kw = dict(ttft=0.4, itl=0.02, out_tokens=300)
    e2e = 0.4 + 0.02 * 299
    present(f"(end to end {e2e:.2f} s)", f"adds {G.added_latency('inline_input', t_check=0.0193, **kw)[0] * 1e3:.1f} ms inline")
    par = G.added_latency("parallel_input", t_check=0.0924, ttft=0.05, itl=0.02, out_tokens=300)[0]
    present(f"adds {par * 1e3:.1f} ms if the first token is held, or lets "
            f"{G.exposed_tokens(t_check=0.0924, ttft=0.05, itl=0.02)} tokens through")
    for w, label in ((200, "+ 0.02 × 199 + 0.15"), (50, "+ 0.02 × 49 + 0.15")):
        present(f"{label} = + {G.added_latency('held_back', t_check=0.15, window=w, **kw)[0]:.2f} s")
    present(f"+ 0.02 × 299 + 0.15 = + {G.added_latency('final', t_check=0.15, **kw)[0]:.2f} s")
    present(f"is ${G.cost_per_1k_checks(3.7, 0.0924):.3f} per 1,000 checks, and batched 16 at a time "
            f"${G.cost_per_1k_checks(3.7, 0.0924, 16):.4f}",
            f"{G.false_block_rate(0.01, 26):.1%} of conversations".replace("%", " %"),
            f"at 0.1 %, {G.false_block_rate(0.001, 26):.2%}".replace("%", " %"))


def test_s8_mcp_numbers():
    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    present(f"verifier `{verifier}` → challenge `{s256(verifier)}`")
    urls = as_metadata_urls("https://auth.example.com/tenant1")
    for u in urls:
        present(f"(`{u.removeprefix('https://auth.example.com')}`)")
