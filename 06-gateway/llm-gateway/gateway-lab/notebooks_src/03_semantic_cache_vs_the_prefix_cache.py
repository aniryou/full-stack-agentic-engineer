# %% [markdown]
# # 03 · The semantic cache vs the prefix cache: what each saves, what each risks
#
# **Tier:** T0. The exact cache and the semantic cache of the gateway are real (sqlite3 + numpy). The fake provider
# **emulates** the prefix cache of the engine: 16-token block hashes, the vLLM hit rule, and `cache_salt` on the first
# block. The TTFT that this prefix cache saves is **simulated** from a prefill time per token. T1: the same prompts
# against a real vLLM started with `--enable-prompt-tokens-details`. Its `cached_tokens` are measured values.
#
# ## The one-minute version
#
# Four caches can be on the path of a request (PRIMER §3):
#
# | cache | where | a hit saves | the risk |
# |---|---|---|---|
# | exact response | gateway | the whole call: tokens, dollars, seconds | stale answers. Leaks across tenants without a namespace. |
# | semantic response | gateway | the whole call, for a *paraphrase* | **false hits**: a near miss gets the answer of another question |
# | provider prompt cache | hosted API | ~90 % of the price of the cached input (scaling primer §3.4) | no risk to correctness |
# | engine prefix cache | vLLM | prefill compute, thus TTFT (serving-engine PRIMER §5) | a timing side channel between tenants: `cache_salt` |
#
# The response caches are safe only for answers that do not depend on who asks or when. A gateway cannot find that from
# the text. Thus the **route declares** the class of each request in `metadata.cache_class` (PRIMER §3.2). The gateway
# caches only the classes on the allowlist. Shared classes (`faq`) go in a namespace per tenant, alias and system
# prompt. Per-user classes (`account`) also have the user in the namespace.
#
# The gateway caches only deterministic requests (`temperature: 0`) with no tools. A regex can still *veto* a declared
# shared class that looks personal. The regex never makes anything cacheable.
#
# A semantic cache embeds the question and finds its nearest cached neighbour. If the similarity is above a threshold,
# the cache serves that neighbour. An exact **guard** on numbers, dates and codes stops the classic false hit ("order
# 1234" answered with order 1243's status). The setting of the threshold is not a guess. Do a sweep of the threshold on labelled
# traffic. Then make sure that the live gateway does what the sweep predicted.
#
# The embedder is the lexical hashing embedder of 07.4's `ragkit` (embeddings primer §15, "Embeddings elsewhere in agent
# systems"). A real embedder (T1) moves both curves.

# %%
import collections, json
from gwlab import client, env, report, t1
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
print("cacheable classes:", stack.cfg.cache.classes, "| per user:", stack.cfg.cache.per_user_classes)

# %% [markdown]
# ## Why the class is declared, not inferred
#
# The gateway comes with a regex that sorts questions into general, personal and time-sensitive. The regex agrees with
# the labels of this sample, because the lab wrote the sample with the regex in mind. But the regex is still incorrect
# as soon as real users type.
#
# An FAQ that says "my" looks personal to the regex. The result is a missed hit, which does no harm. A personal
# question with nothing that a regex can see looks general. If the gateway caches it in a tenant-wide namespace, it
# serves the plan limit of one user to the next user.

# %%
for q in ("How do I cancel my subscription?", "What is my plan limit?", "Which plan am I on?"):
    print(f"{q:36s} the regex says {C.classify_query(q)!r}")
print(C.CLASSIFIER_LABEL)

# %% [markdown]
# ## Worked example: the threshold sweep
#
# Replay the sample through a new, empty semantic cache at each threshold. A cacheable query is a `general` query of the
# sample, declared `faq`. Its nearest cached query answers it when the cosine passes the threshold. With the guard, the
# entities must also match. The answer is *correct* when the two queries are in the same answer group.
#
# The counts:
#
# - `served`: every answer that the cache gave.
# - `false`: the incorrect answers.
# - `correct`: the other answers.
# - `reachable`: the share of queries whose group was already in the cache. This is the maximum that any threshold can
#   serve correctly.
#
# These rates are over all cacheable queries on the sample of this lab. The sweep of gateway-core reports correct hits
# over *paraphrases* on a different sample. Thus compare the method, not the thresholds.

# %%
thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
rows = {g: C.sweep(sample, thresholds, guard=g) for g in (False, True)}
print(f"{'threshold':>9} | {'served':>7} {'false':>6} {'correct':>7} (no guard) | {'served':>7} {'false':>6} {'correct':>7} (entity guard) | reachable")
for a, b in zip(rows[False], rows[True]):
    print(f"{a['threshold']:>9} | {a['served_rate']:7.1%} {a['false_hit_rate']:6.1%} {a['correct_rate']:7.1%}            | "
          f"{b['served_rate']:7.1%} {b['false_hit_rate']:6.1%} {b['correct_rate']:7.1%}                | {b['reachable']:.1%}")

# %% [markdown]
# Read the result as an engineer who selects a setting:
#
# - Below ~0.85, the cache serves a lot, and a large share of it is incorrect.
# - The guard removes the false hits that differ only in a number or a code ("plan 4" against "plan 8", "E1042"
#   against "E1043"). It does not remove "annual" against "monthly" plans.
# - At 0.9, almost no incorrect answer gets through, and the cache serves only paraphrases that are almost verbatim.
#
# The gap between the correct rate at a safe threshold and `reachable` is what a better embedder can win. A better
# embedder can also lose, because it also brings new false hits.

# %% [markdown]
# ## Exercise 3.1 — the cache key, from the declared class
#
# Write `cache_key(tenant, alias, body)`. Return `None` when the gateway must not cache this request. Else return a key.
# Two requests have equal keys exactly when the gateway treats them as the same cache entry. These requests are not
# cacheable:
#
# - `temperature` absent or not 0, any `tools`, or `n > 1`.
# - No `metadata.cache_class`.
# - A class that is not in `SHARED` or `PER_USER`.
# - A per-user class without `metadata.user`.
# - A shared class whose last user message `C.classify_query` does not call `"general"` (the deny-only guard).
#
# The key is a SHA-256 over two parts:
#
# - The namespace: tenant, alias, the system prompt, and **the user for a per-user class**.
# - The canonical request: `model`, `messages`, `temperature`, `top_p`, the token caps, `tools`, `tool_choice`,
#   `response_format`, `stop`, `seed`, `reasoning_effort`. It does not include `stream`, `stream_options`, `user` or
#   `metadata`.
#
# The check compares your key with the cache of the live gateway on 600 pairs of generated requests.

# %% exercise
import hashlib
SHARED, PER_USER = tuple(stack.cfg.cache.classes), tuple(stack.cfg.cache.per_user_classes)

def cache_key(tenant: str, alias: str, body: dict):
    ### BEGIN SOLUTION
    if body.get("temperature") != 0 or body.get("tools") or (body.get("n") or 1) != 1:
        return None
    meta = body.get("metadata") or {}
    cls, user = meta.get("cache_class"), meta.get("user")
    if cls in PER_USER:
        if not user:
            return None
    elif cls not in SHARED:
        return None
    else:
        users = [m for m in body.get("messages", []) if m.get("role") == "user"]
        if users and C.classify_query(users[-1].get("content") or "") != "general":
            return None
    system = " ".join(m.get("content") or "" for m in body.get("messages", []) if m.get("role") in ("system", "developer"))
    keep = ("model", "messages", "temperature", "top_p", "max_completion_tokens", "max_tokens", "tools", "tool_choice",
            "response_format", "stop", "seed", "reasoning_effort")
    canon = json.dumps({k: body[k] for k in keep if k in body}, sort_keys=True, separators=(",", ":"))
    who = json.dumps([tenant, alias, system, user if cls in PER_USER else None])
    return hashlib.sha256(f"{who}|{canon}".encode()).hexdigest()
    ### END SOLUTION

# %% check
import random
rng = random.Random(7)
qs = [q["query"] for q in sample] + ["What is my plan limit?", "How do I cancel my subscription?"]
gw_cache = stack.gateway.cache

def gen():
    b = {"model": "chat", "messages": ([{"role": "system", "content": rng.choice(["Be brief.", "Be kind."])}] if rng.random() < 0.7 else [])
         + [{"role": "user", "content": rng.choice(qs)}]}
    if rng.random() < 0.85:
        b["temperature"] = rng.choice([0, 0, 0, 0.7])
    meta = {}
    if rng.random() < 0.85:
        meta["cache_class"] = rng.choice(["faq", "faq", "account", "marketing"])
    if rng.random() < 0.6:
        meta["user"] = rng.choice(["alice", "bob"])
    if meta:
        b["metadata"] = meta
    for k, v in (("stream", True), ("user", "u1"), ("max_completion_tokens", 64), ("tools", [{"type": "function"}]), ("n", 2)):
        if rng.random() < 0.15:
            b[k] = v
    return rng.choice(["team-a", "team-b"]), rng.choice(["chat", "chat-cheap"]), b

def gateway_key(t, a, b):
    return (gw_cache.namespace(t, a, b), gw_cache.exact_key(b)) if gw_cache.cacheable(b)[0] else None

same = 0
for _ in range(600):
    x, y = gen(), gen()
    if rng.random() < 0.4:                                    # a variant of the same request, perhaps another user
        meta = {**(x[2].get("metadata") or {}), **({"user": rng.choice(["alice", "bob"])} if rng.random() < 0.5 else {})}
        y = (x[0], x[1], {**x[2], "stream": not x[2].get("stream"), "user": "someone-else", "metadata": meta})
    assert (cache_key(*x) is None) == (gateway_key(*x) is None), x
    if cache_key(*x) and cache_key(*y):
        assert (cache_key(*x) == cache_key(*y)) == (gateway_key(*x) == gateway_key(*y)), (x, y)
        same += cache_key(*x) == cache_key(*y)
print(f"✅ the gateway's cacheability and keys, {same} matching pairs among them: undeclared, sampled, tool-using and "
      "vetoed requests bypass; a per-user answer is keyed by its user")

# %% [markdown]
# ## Worked example: the caches over HTTP
#
# The next cell sends these requests:
#
# - The same declared-`faq` question from the same tenant two times (an exact hit).
# - A paraphrase that keeps the words (a semantic hit).
# - A near miss with another number. It passes the threshold of this stack, 0.85, and the guard rejects it.
# - A personal question that someone declared `faq` (the regex vetoes it).
# - A per-user question from alice and then from bob.
# - An undeclared question.
# - Another tenant (a separate namespace).
#
# A hit costs nothing in the ledger and returns in a few milliseconds. (0.85 is low on purpose, to show the guard. The
# sweep supports 0.9 or more on this sample.)

# %%
def ask(key, q, meta, **kw):
    r = stack.chat(key, [{"role": "system", "content": "You answer product questions."}, {"role": "user", "content": q}],
                   temperature=0, **({"metadata": meta} if meta else {}), **kw)
    d = stack.last_decision()["cache"]
    return r, f"{q[:40]:40s} {str(meta):44s} -> {r.header('x-gwlab-cache'):8s} {d['score']:.3f} {d['reason'][:40]:40s} {r.e2e_s * 1e3:5.1f} ms"

faq, alice, bob = {"cache_class": "faq"}, {"cache_class": "account", "user": "alice"}, {"cache_class": "account", "user": "bob"}
for k, q, meta in ((key_a, "How many vCPUs does plan 4 include?", faq), (key_a, "How many vCPUs does plan 4 include?", faq),
                   (key_a, "how many vCPUs does plan 4 include", faq), (key_a, "How many vCPUs does plan 8 include?", faq),
                   (key_a, "What is the status of my order 1234?", faq), (key_a, "What is my plan limit?", alice),
                   (key_a, "What is my plan limit?", alice), (key_a, "What is my plan limit?", bob),
                   (key_a, "How many vCPUs does plan 4 include?", None), (key_b, "How many vCPUs does plan 4 include?", faq)):
    print(("team-a " if k == key_a else "team-b ") + ask(k, q, meta)[1])
print(report.ledger_table(stack.ledger(), last=6))

# %% [markdown]
# ## Exercise 3.2 — does the live gateway do what the sweep predicted?
#
# The sweep is an offline model. The gateway is the thing that serves users. Replay the whole labelled sample, in
# order, through a new gateway at threshold 0.85 with the entity guard. Declare every query `faq`, as a careless route
# does. Thus the run also uses the regex veto. Calculate the score from the decision log of the gateway itself.
#
# Write `score(sample, decisions)`. Return `{"served", "false", "vetoed"}`:
#
# - `served`: the requests that the cache answered (exact or semantic).
# - `false`: the requests among them that got the answer of another group. The `cache.matched` of the decision is the
#   cached query that answered.
# - `vetoed`: the requests that the regex vetoed. Their `cache.reason` says so.
#
# The check compares your score with `C.sweep` at the same setting.

# %%
replay = LocalStack(fakes=fakes, overrides={"cache": {"threshold": 0.85, "entity_guard": True}}).start()
k_r = replay.issue_key("team-a")
for q in sample:
    replay.chat(k_r, [{"role": "system", "content": "You answer product questions."}, {"role": "user", "content": q["query"]}],
                temperature=0, metadata={"cache_class": "faq"})
decisions = replay.decisions()
replay.stop()
print(len(decisions), "decisions; one hit:", next(d["cache"] for d in decisions if d["cache"]["result"] == "semantic"))

# %% exercise
def score(sample, decisions):
    ### BEGIN SOLUTION
    group = {q["query"]: q["group"] for q in sample}
    served = [(q, d) for q, d in zip(sample, decisions) if d["cache"]["result"] in ("exact", "semantic")]
    return {"served": len(served), "false": sum(group[d["cache"]["matched"]] != q["group"] for q, d in served),
            "vetoed": sum("vetoed" in d["cache"]["reason"] for d in decisions)}
    ### END SOLUTION

# %% check
pred = C.sweep(sample, [0.85], guard=True)[0]
got = score(sample, decisions)
assert got == {"served": pred["hits"], "false": pred["false_hits"], "vetoed": pred["bypassed"]}, (got, pred)
print(f"✅ live: {got['served']} served from the cache, {got['false']} of them wrong, {got['vetoed']} vetoed by the regex — "
      f"exactly the sweep's {pred['hits']} / {pred['false_hits']} / {pred['bypassed']}: the offline sweep is a faithful "
      "model of this gateway, so a threshold chosen on it means what it says")

# %% [markdown]
# ## Worked example: the engine's prefix cache, and a salt per tenant
#
# Every request of an agent shares one long system prompt. The first request computes it. The second request finds the
# full 16-token blocks of the prompt in the prefix cache of the engine (`usage.prompt_tokens_details.cached_tokens`).
# Its TTFT decreases by the prefill that it skipped (0.5 ms per token here, simulated).
#
# The gateway sends the `cache_salt` of each tenant. The salt is an HMAC of the *verified* tenant under the secret of
# the gateway, with 43 characters. The gateway never takes it from the request. Thus another tenant with the same
# prompt starts cold.

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
print("salts:", {t: C.cache_salt(t, stack.cfg.salt_secret)[:12] + "..." for t in ("team-a", "team-b")},
      "| valid for vLLM:", all(C.valid_cache_salt(C.cache_salt(t, stack.cfg.salt_secret)) for t in ("team-a", "team-b")))

# %% [markdown]
# ## Exercise 3.3 — the side channel the salt closes
#
# Why use a salt at all, when the prefix cache never serves an incorrect answer? The reason is that its *speed* is
# observable. A tenant who can send a candidate prompt and measure the time to the first token learns if someone else
# already sent that prompt. This is a leak of the system prompt or document of another tenant, one guess at a time.
#
# Write `looks_cached(ttft_s, prompt_tokens, base_s, prefill_s_per_token)`. It is the test of the attacker. Return
# `True` when the measured TTFT is below the midpoint between a fully warm prefill (`base_s`) and a fully cold one
# ($\mathrm{base\_s} \:+$ $\mathrm{prompt\_tokens} \times \mathrm{prefill\_s\_per\_token}$).
#
# The check does the attack directly against the engine (the vLLM-shaped fake). A victim sends a prompt, and the
# attacker measures the time of the same prompt. The attacker does this first with no salt at all (a gateway that forgot
# it). Then the attacker does it under the own salt of each tenant (what this gateway sends).

# %% exercise
def looks_cached(ttft_s: float, prompt_tokens: int, base_s: float, prefill_s_per_token: float) -> bool:
    ### BEGIN SOLUTION
    return ttft_s < base_s + 0.5 * prompt_tokens * prefill_s_per_token
    ### END SOLUTION

# %% check
import uuid
spec = stack.fakes["acme"].spec
def probe(secret_doc, victim_salt, attacker_salt):
    body = lambda salt: {"model": "fast-1", "messages": [{"role": "system", "content": secret_doc},        # noqa: E731
                                                          {"role": "user", "content": "summarise"}],
                         "stream_options": {"include_usage": True}, **({"cache_salt": salt} if salt else {})}
    client.chat(stack.fake_url("acme"), "acme-key-1", body(victim_salt), stream=True)                       # the victim
    r = client.chat(stack.fake_url("acme"), "acme-key-1", body(attacker_salt), stream=True)                 # the attacker
    return looks_cached(r.ttft_s, r.usage["prompt_tokens"], spec.ttft_s, spec.prefill_s_per_token), r

doc = lambda: f"Contract {uuid.uuid4().hex}: the supplier delivers within thirty days of the order. " * 30   # noqa: E731
leak, r1 = probe(doc(), None, None)
salted, r2 = probe(doc(), C.cache_salt("victim", stack.cfg.salt_secret), C.cache_salt("attacker", stack.cfg.salt_secret))
assert leak and r1.usage["prompt_tokens_details"]["cached_tokens"] > 0
assert not salted and r2.usage["prompt_tokens_details"]["cached_tokens"] == 0
for _ in range(5):                                              # and the test is not fooled by a cold prompt either way
    assert not probe(doc(), "x", "y")[0] and probe(doc(), None, None)[0]
print(f"✅ unsalted: the attacker's TTFT {r1.ttft_s * 1e3:.0f} ms gives the victim's prompt away; salted per tenant: "
      f"{r2.ttft_s * 1e3:.0f} ms, cold, nothing to learn [SIMULATED timings, the same rule on a real engine]")

# %% [markdown]
# ## Exercise 3.4 — predict `cached_tokens`
#
# vLLM caches **full** blocks only (16 tokens). It goes along the block-hash chain of the prompt until the first miss.
# It always computes the last prompt token, because it needs its logits. Thus a hit has a limit of
# $\lfloor (n - 1)/16 \rfloor$ blocks. Write `expected_cached_tokens(prompt_tokens, shared_prefix_tokens, block=16)` for
# a prompt whose first `shared_prefix_tokens` tokens the engine already processed (under the same salt).
#
# The check builds prompts that share a prefix of different lengths with a warmed prompt. The tokenizer of the fake
# counts the lengths. The check compares your result with what the fake reports. (The same rule is
# `expected_cached_tokens` in 04's `servelab`. At T1, vLLM reports the real value.)

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
# Use the call shape of the scaling primer (5,000 input tokens, 350 output) at `gemini-3.5-flash` prices (verify,
# 2026-09-26). With a prefix/prompt-cache hit on 2,700 of the input tokens, the provider bills them at 10 %. A semantic
# hit skips the call. The engine prefix cache is the self-hosted version of the first hit. It saves prefill time, not a
# line on the bill of a provider.

# %%
p = PRICES["gemini-3.5-flash"]
full, prompt_cached = cost_usd(Usage(5000, 350), p), cost_usd(Usage(5000, 350, 2700), p)
print(f"no cache ${full:.5f} | prompt cache (2,700 cached) ${prompt_cached:.5f} ({1 - prompt_cached / full:.0%} saved) | "
      f"semantic hit $0 (100 % saved, on the {next(r for r in rows[True] if r['threshold'] == 0.9)['correct_rate']:.0%} of "
      "cacheable traffic it serves correctly here)")

# %% [markdown]
# ## T1: measured `cached_tokens` on a real vLLM
#
# When you set `GWLAB_VLLM_URL` and vLLM answers `/health`, the shared-prefix requests go through a gateway whose first
# target is vLLM. The engine itself reports `cached_tokens` (with `--enable-prompt-tokens-details`). Each run puts a new
# nonce in the system prompt. Thus blocks that an earlier run cached cannot hit. The tokenizer of vLLM is the tokenizer
# of the model. Thus the check asserts the rule, and only on the rows that vLLM served (`MEASURED`):
#
# - a multiple of 16,
# - at most $\lfloor (n - 1)/16 \rfloor \times 16$,
# - 0 for the first request of team-a,
# - more than 0 for its second request,
# - **0 under team-b's salt**.
#
# If the gateway does not send `cache_salt` any more, that last assertion fails.

# %%
if tiers["vllm_url"]:
    rows_t1 = t1.prefix_cache(tiers["vllm_url"], system, fallback=fakes["acme"])
    for who, tag, n, hit, ttft in rows_t1:
        print(f"{tag} {who}: prompt {n}, cached {hit}, TTFT {ttft * 1e3:.1f} ms")
    t1.check_prefix_rows(rows_t1)
else:
    print("T1 skipped: deploy/any-gpu/serve.sh, then export GWLAB_VLLM_URL=http://127.0.0.1:8000")

# %%
stack.stop()

# %% [markdown]
# ## In a design review
#
# **Two minutes:** "We cache at two places for two reasons. The prefix cache of the engine, and also the prompt cache
# of a hosted provider, is exact and safe. We put the stable part of each prompt first. We add a salt per tenant to the
# cache of the engine: an HMAC of the verified tenant. The reason is that the speed of the cache is observable. A prefix
# cache with no salt lets one tenant measure the time of the prompts of another tenant.
#
# "The response cache of the gateway is for answers that do not depend on who asks or when. The route declares which
# answers these are: `faq` shared in a tenant, `account` per user. The reason is that no classifier can read that from
# the text. A regex only vetoes. The namespace of every entry has the tenant, the alias and the system prompt.
#
# "Its semantic half serves paraphrases above a threshold that we selected from a sweep on labelled traffic. We made
# sure that the live gateway serves exactly what the sweep predicted. The semantic half has an exact guard on numbers,
# dates and codes. A false hit is an incorrect answer that the gateway gives with confidence. Thus we prefer to serve
# fewer hits."
#
# **Drill 1.** *The semantic cache answered one user with another user's order status. What went wrong?* The question
# was personal, a class that must never share an answer, and the key had no user namespace. A lexical near-match
# ("order 1234" against "order 1243") passed the threshold.
#
# Declare classes per route, and put the user in the key of the per-user classes. Put each tenant in its own namespace.
# Use a guard on entities. Measure the false hits before you decrease the threshold (CURRICULUM cross-layer drill 19).
#
# **Drill 2.** *Is it correct for the semantic cache key to include the system prompt?* Yes. The same question under
# different instructions has a different correct answer. Here the namespace hashes the system prompt. Thus a change to
# the prompt invalidates the old entries by construction.
#
# **Drill 3.** *Why does a salt go on the first block only?* Block hashes form a chain. Thus a salt on the first block
# changes every block after it. Then the whole prefix is private to the salt. In a tenant, the prefix cache works as
# before.
