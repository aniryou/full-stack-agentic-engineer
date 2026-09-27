# %% [markdown]
# # 03 · The semantic cache vs the prefix cache: what each saves, what each risks
#
# **Tier:** T0 — the gateway's exact and semantic caches are real (sqlite3 + numpy); the engine's prefix cache is
# **emulated** by the fake provider (16-token block hashes, the vLLM hit rule, `cache_salt` on the first block),
# and its TTFT saving is **simulated** from a per-token prefill time. T1: the same prompts against a real vLLM
# started with `--enable-prompt-tokens-details`, whose `cached_tokens` are measured.
#
# ## The one-minute version
#
# Four caches can sit on a request's path (PRIMER §3):
#
# | cache | where | a hit saves | the risk |
# |---|---|---|---|
# | exact response | gateway | the whole call: tokens, dollars, seconds | stale answers; leaks across tenants without a namespace |
# | semantic response | gateway | the whole call, for a *paraphrase* | **false hits**: a near miss gets another question's answer |
# | provider prompt cache | hosted API | ~90 % of the cached input's price (scaling primer §3.4) | none to correctness |
# | engine prefix cache | vLLM | prefill compute, so TTFT (serving-engine PRIMER §5) | a timing side channel between tenants: `cache_salt` |
#
# The response caches are only safe for answers that do not depend on who asks or when: deterministic
# (`temperature: 0`), no tools, not personal, not time-bound — and namespaced by tenant, alias and system prompt.
# A semantic cache embeds the question, finds its nearest cached neighbour and serves it above a similarity
# threshold; an exact **guard** on numbers, dates and codes stops the classic false hit ("order 1234" answered
# with order 1243's status). Where to set the threshold is not a guess: sweep it on labelled traffic and read hit
# rate against false-hit rate. The embedder here is the lexical hashing embedder of 07.4's `ragkit` — it finds
# rewordings that share words and misses those that do not (embeddings primer §15, "Embeddings elsewhere in
# agent systems"); a real embedder (T1) moves both curves.

# %%
import collections, json
from gwlab import client, env, report
from gwlab.fakes import FakeSpec
from gwlab.gateway import cache as C
from gwlab.gateway.adapters import Usage
from gwlab.gateway.metering import PRICES, cost_usd
from gwlab.stack import LocalStack

tiers = env.describe()
fakes = {"acme": FakeSpec(name="acme", ttft_s=0.02, prefill_s_per_token=0.0005, itl_s=0.003, output_tokens=20),
         "bolt": FakeSpec.anthropic("bolt", ttft_s=0.02, prefill_s_per_token=0.0005, itl_s=0.003, output_tokens=20)}
stack = LocalStack(fakes=fakes, overrides={"cache": {"threshold": 0.85}}).start()   # low, to show the guard at work
key_a, key_b = stack.issue_key("team-a"), stack.issue_key("team-b")
sample = C.load_sample()
print(C.SAMPLE_LABEL)
print("queries:", len(sample), "| by kind:", dict(collections.Counter(q["kind"] for q in sample)))
print("by class:", dict(collections.Counter(q["class"] for q in sample)), "| embedder:", C.EMBEDDER_LABEL)

# %% [markdown]
# ## Worked example: the threshold sweep
#
# Replay the sample through a fresh semantic cache at each threshold: a cacheable query is answered by its
# nearest cached query when the cosine clears the threshold (and, with the guard, the entities match); the answer
# is *correct* when both are in the same answer group. `reachable` is the share whose group was already cached —
# the most any threshold could serve.

# %%
thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
rows = {g: C.sweep(sample, thresholds, guard=g) for g in (False, True)}
print(f"{'threshold':>9} | {'hit rate':>8} {'false':>6} (no guard) | {'hit rate':>8} {'false':>6} (entity guard) | reachable")
for a, b in zip(rows[False], rows[True]):
    print(f"{a['threshold']:>9} | {a['hit_rate']:8.1%} {a['false_hit_rate']:6.1%}            | {b['hit_rate']:8.1%} {b['false_hit_rate']:6.1%}                | {b['reachable']:.1%}")

# %% [markdown]
# Read it like an engineer choosing a setting: below ~0.85 the cache "hits" a lot and a large share of those are
# wrong; the guard removes the false hits that differ only in a number or a code ("plan 4" vs "plan 8", "E1042"
# vs "E1043") but not "annual" vs "monthly" plans; at 0.9 almost nothing wrong gets through and the cache
# serves only near-verbatim rewordings. The gap between the hit rate at a safe threshold and `reachable` is what
# a better embedder could win — or lose, because it also brings new false hits.

# %% [markdown]
# ## Exercise 3.1 — the cache key
#
# Write `cache_key(tenant, alias, body)` returning `None` when the request must not be cached, and otherwise a
# SHA-256 hex digest over the **namespace** (tenant, alias, and the system prompt) and the **canonical request**
# (`model`, `messages`, `temperature`, `top_p`, the token caps, `tools`, `tool_choice`, `response_format`, `stop`,
# `seed`, `reasoning_effort`; not `stream`, `stream_options`, `user` or `metadata`). Not cacheable: `temperature`
# absent or not 0, any `tools`, `n > 1`, or a last user message that `C.classify_query` calls personal or
# time-sensitive. The check asserts your key agrees with the gateway's (same key exactly when the gateway's
# namespace and exact key agree) on 400 pairs of generated requests.

# %% exercise
import hashlib

def cache_key(tenant: str, alias: str, body: dict):
    ### BEGIN SOLUTION
    if body.get("temperature") != 0 or body.get("tools") or (body.get("n") or 1) != 1:
        return None
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    if users and C.classify_query(users[-1].get("content") or "") != "general":
        return None
    system = " ".join(m.get("content") or "" for m in body.get("messages", []) if m.get("role") in ("system", "developer"))
    keep = ("model", "messages", "temperature", "top_p", "max_completion_tokens", "max_tokens", "tools", "tool_choice",
            "response_format", "stop", "seed", "reasoning_effort")
    canon = json.dumps({k: body[k] for k in keep if k in body}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{tenant}|{alias}|{system}|{canon}".encode()).hexdigest()
    ### END SOLUTION

# %% check
import random
rng = random.Random(7)
qs = [q["query"] for q in sample]

def gen():
    b = {"model": "chat", "messages": ([{"role": "system", "content": rng.choice(["Be brief.", "Be kind."])}] if rng.random() < 0.7 else [])
         + [{"role": "user", "content": rng.choice(qs)}]}
    if rng.random() < 0.85:
        b["temperature"] = rng.choice([0, 0, 0, 0.7])
    for k, v in (("stream", True), ("user", "u1"), ("max_completion_tokens", 64), ("tools", [{"type": "function"}]), ("n", 2)):
        if rng.random() < 0.15:
            b[k] = v
    return rng.choice(["team-a", "team-b"]), rng.choice(["chat", "chat-cheap"]), b

def gateway_key(t, a, b):
    return (C.GatewayCache.namespace(t, a, b), C.GatewayCache.exact_key(b)) if C.cacheable(b)[0] else None

for _ in range(400):
    x, y = gen(), gen()
    if rng.random() < 0.3:                                    # a variant of the same request
        y = (x[0], x[1], {**x[2], "stream": not x[2].get("stream"), "user": "someone-else"})
    assert (cache_key(*x) is None) == (gateway_key(*x) is None), x
    if cache_key(*x) and cache_key(*y):
        assert (cache_key(*x) == cache_key(*y)) == (gateway_key(*x) == gateway_key(*y)), (x, y)
print("✅ one key per (tenant, alias, system prompt, canonical request); sampling, tools, personal and time-bound bypass")

# %% [markdown]
# ## Worked example: the caches over HTTP
#
# The same question from the same tenant twice (exact hit), a rewording that keeps the words (semantic hit), a
# near miss with another number (it clears this stack's threshold of 0.85 and is rejected by the guard), a
# personal question (bypassed), and another tenant (a separate namespace). A hit costs nothing in the ledger and
# returns in a few milliseconds. (0.85 is low on purpose, to show the guard; the sweep above argues for 0.9 or
# more on this sample.)

# %%
def ask(key, q, **kw):
    r = stack.chat(key, [{"role": "system", "content": "You answer product questions."}, {"role": "user", "content": q}],
                   temperature=0, **kw)
    d = stack.last_decision()["cache"]
    return r, f"{q[:46]:46s} -> {r.header('x-gwlab-cache'):8s} score {d['score']:.3f} {d['reason']:16s} {r.e2e_s * 1e3:6.1f} ms"

for k, q in ((key_a, "How many vCPUs does plan 4 include?"), (key_a, "How many vCPUs does plan 4 include?"),
             (key_a, "how many vCPUs does plan 4 include"), (key_a, "How many vCPUs does plan 8 include?"),
             (key_a, "What is the status of my order 1234?"), (key_b, "How many vCPUs does plan 4 include?")):
    print(("team-a " if k == key_a else "team-b ") + ask(k, q)[1])
print(report.ledger_table(stack.ledger(), last=6))

# %% [markdown]
# ## Exercise 3.2 — the entity guard
#
# Write `entity_guard(a, b)`: True when the two questions contain the same numbers, dates and codes (compare the
# multisets from `C.entities`, which lower-cases and drops hyphens so "E-1042" equals "e1042"). The check runs it
# on every pair of general queries in the sample that the hashing embedder scores at 0.8 or more and compares
# with the gateway's rule; then it measures what the guard buys at threshold 0.85.

# %% exercise
def entity_guard(a: str, b: str) -> bool:
    ### BEGIN SOLUTION
    return sorted(C.entities(a)) == sorted(C.entities(b))
    ### END SOLUTION

# %% check
emb = C.HashingEmbedder()
general = [q for q in sample if q["class"] == "general"]
close = [(x, y) for i, x in enumerate(general) for y in general[i + 1:]
         if float(emb.encode(x["query"]) @ emb.encode(y["query"])) >= 0.8]
blocked = [(x["query"], y["query"]) for x, y in close if not entity_guard(x["query"], y["query"])]
for x, y in close:
    assert entity_guard(x["query"], y["query"]) == (C.entities(x["query"]) == C.entities(y["query"]))
assert not entity_guard("How many vCPUs does plan 4 include?", "How many vCPUs does plan 8 include?")
assert entity_guard("What does error code E-1042 mean?", "meaning of error code e1042")
wrong_blocked = sum(1 for x, y in close if not entity_guard(x["query"], y["query"]) and x["group"] != y["group"])
at85 = {g: next(r for r in rows[g] if r["threshold"] == 0.85) for g in (False, True)}
print(f"{len(close)} close pairs; the guard separates {len(blocked)}, {wrong_blocked} of them rightly (different answers)")
print(f"✅ at threshold 0.85 the guard cuts false hits from {at85[False]['false_hits']} to {at85[True]['false_hits']} "
      f"(of {at85[True]['n']}); 'annual' vs 'monthly' plans has no number to catch")

# %% [markdown]
# ## Exercise 3.3 — choose the threshold from the data
#
# Write `choose_threshold(rows, max_false_hit_rate)`: among the sweep rows whose `false_hit_rate` is within the
# budget, the one with the highest `hit_rate` (ties: the higher threshold); `None` if no row qualifies. The check
# compares with a brute-force pick on a fine sweep for three budgets.

# %% exercise
def choose_threshold(rows: list, max_false_hit_rate: float):
    ### BEGIN SOLUTION
    ok = [r for r in rows if r["false_hit_rate"] <= max_false_hit_rate]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["hit_rate"], r["threshold"]))["threshold"]
    ### END SOLUTION

# %% check
fine = C.sweep(sample, [round(0.5 + 0.01 * i, 2) for i in range(50)], guard=True)
for budget in (0.0, 0.02, 0.2):
    best = max((r for r in fine if r["false_hit_rate"] <= budget), key=lambda r: (r["hit_rate"], r["threshold"]))
    assert choose_threshold(fine, budget) == best["threshold"], budget
t0 = choose_threshold(fine, 0.0)
r0 = next(r for r in fine if r["threshold"] == t0)
print(f"✅ zero false hits on this sample: threshold {t0} serves {r0['hit_rate']:.1%} of cacheable traffic "
      f"(of {r0['reachable']:.1%} reachable) — a sample of {r0['n']}, so measure again on your own traffic")

# %% [markdown]
# ## Worked example: the engine's prefix cache, and a salt per tenant
#
# A long system prompt shared by every request of an agent. The first request computes it; the second finds its
# full 16-token blocks in the engine's prefix cache (`usage.prompt_tokens_details.cached_tokens`) and its TTFT
# drops by the prefill it skipped (0.5 ms per token here, simulated). The gateway sends each tenant's
# `cache_salt` — an HMAC of the *verified* tenant — so another tenant with the same prompt starts cold: no
# shared blocks, no timing side channel.

# %%
system = "You are the support agent for a large company. Follow the policy exactly and cite the article you used. " * 6
def prefixed(key, q):
    r = stack.chat(key, [{"role": "system", "content": system}, {"role": "user", "content": q}], stream=True,
                   stream_options={"include_usage": True})
    return r.usage["prompt_tokens"], r.usage["prompt_tokens_details"]["cached_tokens"], r.ttft_s

for who, k, q in (("team-a", key_a, "first question"), ("team-a", key_a, "second question"),
                  ("team-a", key_a, "third question here"), ("team-b", key_b, "first question")):
    n, hit, ttft = prefixed(k, q)
    print(f"{who}: prompt {n} tokens, cached {hit:4d}, TTFT {ttft * 1e3:6.1f} ms [SIMULATED]")

# %% [markdown]
# ## Exercise 3.4 — a salt per tenant, valid for vLLM
#
# Write `tenant_salt(tenant, secret)`: HMAC-SHA256 of the tenant name keyed with the gateway's secret, base64url
# without padding (43 characters, 256 bits), and `valid_for_vllm(salt)`: vLLM 0.30.0's rule — a non-empty
# string of at most 128 characters containing none of `@`, `/`, `\` or NUL. The check compares with the gateway,
# sends your salt straight to the vLLM-shaped fake (which validates it the same way), and shows that the same
# prompt under two salts shares no blocks.

# %% exercise
import base64, hmac

def tenant_salt(tenant: str, secret: str) -> str:
    ### BEGIN SOLUTION
    mac = hmac.new(secret.encode(), tenant.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()
    ### END SOLUTION

def valid_for_vllm(salt) -> bool:
    ### BEGIN SOLUTION
    return isinstance(salt, str) and 1 <= len(salt) <= 128 and not any(c in salt for c in "@/\\\x00")
    ### END SOLUTION

# %% check
for t in ("team-a", "team-b", "tenant/with/slashes"):
    assert tenant_salt(t, "s3cret") == C.cache_salt(t, "s3cret") and len(tenant_salt(t, "s3cret")) == 43
for s in ("", "x" * 129, "a/b", "a@b", "a\\b", "a\x00b", None, 7, "x" * 128, "Zm9v-_"):
    assert valid_for_vllm(s) == C.valid_cache_salt(s), s
up = lambda salt: client.request("POST", stack.fake_url("acme") + "/v1/chat/completions",           # noqa: E731
                                 {"model": "fast-1", "cache_salt": salt, "messages": [{"role": "system", "content": system},
                                                                                     {"role": "user", "content": "salted"}]},
                                 {"Authorization": "Bearer acme-key-1"})
s1, s2 = tenant_salt("tenant-x", "k"), tenant_salt("tenant-y", "k")
up(s1)
assert up(s1).json["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
assert up(s2).json["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
assert up("bad/salt").status == 400
print("✅ same prompt, two tenants' salts: the second starts cold; an invalid salt is a 400 from vLLM, not a silent miss")

# %% [markdown]
# ## Exercise 3.5 — predict `cached_tokens`
#
# vLLM caches **full** blocks only (16 tokens), walks the prompt's block-hash chain until the first miss, and
# always computes the last prompt token (it needs its logits), so a hit is capped at `(n − 1) // 16` blocks.
# Write `expected_cached_tokens(prompt_tokens, shared_prefix_tokens, block=16)` for a prompt whose first
# `shared_prefix_tokens` tokens were already processed (under the same salt). The check builds prompts that share
# a varying prefix with a warmed prompt — lengths counted by the fake's tokenizer — and compares with what the
# fake reports. (The same rule is `expected_cached_tokens` in 04's `servelab`; at T1 vLLM reports the real one.)

# %% exercise
def expected_cached_tokens(prompt_tokens: int, shared_prefix_tokens: int, block: int = 16) -> int:
    ### BEGIN SOLUTION
    return min(shared_prefix_tokens // block, (prompt_tokens - 1) // block) * block
    ### END SOLUTION

# %% check
from gwlab.fakes import prompt_ids
base = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon ".split()
warm = [{"role": "user", "content": " ".join(base * 5)}]                    # 100 words (+1 role marker)
hdr = {"Authorization": "Bearer acme-key-1"}
client.request("POST", stack.fake_url("acme") + "/v1/chat/completions", {"model": "strong-1", "messages": warm}, hdr)
for keep, extra in ((100, ""), (100, " plus a tail"), (63, " different ending words"), (31, " x"), (15, " y"), (0, "fresh start")):
    content = " ".join((base * 5)[:keep]) + extra
    msgs = [{"role": "user", "content": content}]
    ids, warm_ids = prompt_ids(msgs), prompt_ids(warm)
    shared = next((i for i, (a, b) in enumerate(zip(ids, warm_ids)) if a != b), min(len(ids), len(warm_ids)))
    got = client.request("POST", stack.fake_url("acme") + "/v1/chat/completions", {"model": "strong-1", "messages": msgs}, hdr).json["usage"]
    pred = expected_cached_tokens(got["prompt_tokens"], shared)
    print(f"prompt {got['prompt_tokens']:3d}, shared prefix {shared:3d} -> cached {got['prompt_tokens_details']['cached_tokens']:3d} (predicted {pred})")
    assert got["prompt_tokens_details"]["cached_tokens"] == pred
print("✅ full blocks only, and never the last token: an identical 101-token prompt hits 96, not 101")

# %% [markdown]
# ## Worked example: what each hit is worth
#
# Take the scaling primer's call shape (5,000 input tokens, 350 output) at `gemini-3.5-flash` prices (verify,
# 2026-09-26). A prefix/prompt-cache hit on 2,700 of the input tokens bills them at 10 %; a semantic hit skips the
# call. The engine prefix cache is the self-hosted version of the first: it saves prefill time, not a line on a
# provider's bill.

# %%
p = PRICES["gemini-3.5-flash"]
full, prompt_cached = cost_usd(Usage(5000, 350), p), cost_usd(Usage(5000, 350, 2700), p)
print(f"no cache ${full:.5f} | prompt cache (2,700 cached) ${prompt_cached:.5f} ({1 - prompt_cached / full:.0%} saved) | "
      f"semantic hit $0 (100 % saved, on the {next(r for r in rows[True] if r['threshold'] == 0.9)['hit_rate']:.0%} of "
      "traffic it can safely serve here)")

# %% [markdown]
# ## T1: measured `cached_tokens` on a real vLLM
#
# With `GWLAB_VLLM_URL` set, the shared-prefix requests go through a gateway whose first target is vLLM; the
# engine reports `cached_tokens` itself (with `--enable-prompt-tokens-details`). Its tokenizer is the model's, so
# the check is the rule, not a number: a multiple of 16, at most `(n − 1) // 16 × 16`, and 0 under another
# tenant's salt.

# %%
if tiers["vllm_url"]:
    t1 = LocalStack(config="vllm", upstreams={"local": tiers["vllm_url"]}, fakes={"acme": fakes["acme"]}).start()
    ka, kb = t1.issue_key("team-a"), t1.issue_key("team-b")
    for who, k, q in (("team-a", ka, "first"), ("team-a", ka, "second"), ("team-b", kb, "first")):
        r = t1.chat(k, [{"role": "system", "content": system}, {"role": "user", "content": q}], stream=True,
                    stream_options={"include_usage": True}, max_completion_tokens=8)
        n, hit = r.usage["prompt_tokens"], (r.usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        print(f"MEASURED {who}: prompt {n}, cached {hit}, TTFT {r.ttft_s * 1e3:.1f} ms, served by {r.header('x-gwlab-target')}")
        assert hit is None or (hit % 16 == 0 and hit <= (n - 1) // 16 * 16)
    t1.stop()
else:
    print("T1 skipped: deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000")

# %%
stack.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We cache at two places for two reasons. The engine's prefix cache — and a hosted provider's
# prompt cache — is exact and safe: we lay prompts out with the stable part first, and we salt the engine's
# cache per tenant with an HMAC of the verified tenant so no tenant can time another's prompts. The gateway's
# response cache is for answers that do not depend on who asks or when: deterministic requests, no tools, not
# personal, not time-bound, namespaced by tenant, alias and system prompt. Its semantic half serves paraphrases
# above a threshold we chose from a sweep on labelled traffic — hit rate against false-hit rate — with an exact
# guard on numbers, dates and codes. A false hit is a wrong answer delivered confidently, so we would rather
# serve fewer hits."
#
# **Drill 1.** *The semantic cache answered one user with another user's order status. What went wrong?* — The
# question was personal, a class that must never be cached, and the key had no tenant or user namespace; a
# lexical near-match ("order 1234" vs "order 1243") cleared the threshold. Classify, namespace, guard entities,
# measure false hits before lowering the threshold (CURRICULUM cross-layer drill 17).
#
# **Drill 2.** *Should the semantic cache key include the system prompt?* — Yes: the same question under
# different instructions has a different correct answer. Here the namespace hashes the system prompt, and a
# prompt change invalidates by construction.
#
# **Drill 3.** *Why does a salt go on the first block only?* — Block hashes are chained, so salting the first
# block changes every block after it; the whole prefix is then private to the salt, and within a tenant the
# prefix cache works as before.
