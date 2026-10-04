# %% [markdown]
# # 03 · Exact and semantic caching
#
# **Tier:** T0. It uses only the CPU and no network, and it runs in a few seconds. The embedder is lexical (a hashing embedder, as in
# the `ragkit` of 07.4). It is the floor that a real embedder must beat, and the notebook labels it so. The semantic
# cache next to the prefix cache of vLLM, over HTTP, is `gateway-lab` notebook `03_semantic_cache_vs_the_prefix_cache`
# (T0 emulated, T1 with `cached_tokens` measured).
#
# ## The one-minute version
# Four caches sit on the path of an LLM request, and only two of them can be *incorrect*. The prompt cache of the
# provider and the prefix cache of the engine reuse computation. They save money or prefill time, and they never change
# an answer. The **exact** cache and the **semantic** cache of the gateway reuse an *answer*. Thus they are safe only
# under these conditions:
#
# - The route declares the class cacheable.
# - The key of the namespace is the **verified tenant**.
# - For the semantic cache, the threshold comes from labelled traffic, and a guard compares numbers, dates and codes
#   exactly. The reason is that an embedding cannot see that "Q3 2024" is not "Q3 2025".
#
# By the end, you can do these things:
#
# - make a safe exact key,
# - extract the entities that a guard compares,
# - select a threshold under a false-hit budget,
# - put a price on a false hit,
# - decide what a namespace must hold for shared classes and for per-user classes,
# - tell what the prompt cache of the provider saves instead.
#
# Primer: §3 *Caching at the gateway* (`../PRIMER.md`). See §6.4 for `cache_salt`. The prefix cache of the engine is
# in serving-engine PRIMER §5 (module 04.3). For the false positives of semantic caching, see §15 of the embeddings
# primer.

# %%
import numpy as np

from gwcore import api, cache, keys, metering
from gwcore.gateway import Gateway
from gwcore.providers import Clock, FakeProvider
from gwcore.routing import Router, Target

for cls in ("faq", "personal", "time_sensitive", None):
    print(f"cache_class={cls!s:15s} ->", cache.cacheable({"metadata": {"cache_class": cls}}))
print("faq with tools            ->", cache.cacheable({"metadata": {"cache_class": "faq"}, "tools": [{}]}))
req = {"model": "chat", "messages": [{"role": "user", "content": "What is SSO?"}], "temperature": 0}
print("key(acme)   ", cache.exact_key("acme", req)[:24])
print("key(globex) ", cache.exact_key("globex", req)[:24], " <- another tenant, another key")
print("key(stream) ", cache.exact_key("acme", {**req, "stream": True})[:24], " <- stream does not change the answer")

# %% [markdown]
# ## Worked example 1 — the exact cache in the front door
# The same question comes two times from one tenant. The cache replays the second answer at $0, and no provider sees the
# request. A second tenant asks the same thing and gets a miss, because its namespace is its own.

# %%
clock = Clock()
prov = FakeProvider("openai", clock)
ks = keys.KeyStore(seed=5)
acme, globex = ks.issue("acme"), ks.issue("globex")
gw = Gateway(keys=ks, router=Router({"chat": [Target("openai", "gpt-5.4-mini")]}), providers={"openai": prov},
             clock=clock, cache=cache.ExactCache(ttl=3600))
faq = api.chat_request("chat", [{"role": "user", "content": "What is SSO?"}], metadata={"cache_class": "faq"})
for who, k in (("acme", acme), ("acme", acme), ("globex", globex)):
    r = gw.handle(k, faq)
    print(f"{who:7s} cache={r.cache:4s} provider calls so far={prov.calls}")

# %% [markdown]
# ## Worked example 2 — what a lexical embedder sees
# The bundled sample has 15 cached questions and labelled probes. We want the paraphrases to hit. The near misses must
# not hit.

# %%
s, emb = cache.load_sample(), cache.HashingEmbedder()
seeds = {x["id"]: x["q"] for x in s["seeds"]}
print(emb.model_name)
show = ("How can I reset my password?", "Is SSO included in the Pro plan?", "How much is the Team plan per seat?",
        "What were the Q3 2024 revenue figures?", "What is the price of the Business plan per seat?",
        "Does the Pro plan include SCIM?", "What is the API rate limit on the Pro plan?")
for p in [p for p in s["probes"] if p["q"] in show]:
    sim = float(emb.encode(p["q"]) @ emb.encode(seeds[p["seed"]]))
    print(f"{p['label']:9s} {sim:.3f}  {p['q']!r:52s} vs {seeds[p['seed']]!r}")
rows = {g: cache.sweep_thresholds([0.6, 0.7, 0.8, 0.85, 0.9, 0.95], guard=g) for g in (False, True)}
print("\n  tau | hits  false (no guard) | hits  false (entity guard)")
for a, b in zip(rows[False], rows[True]):
    print(f" {a['threshold']:.2f} | {a['hit_rate']:5.1%} {a['false_hit_rate']:5.1%}         | {b['hit_rate']:5.1%} {b['false_hit_rate']:5.1%}")

# %% [markdown]
# On this embedder, the near misses are *nearer* to the cached question than the paraphrases. Thus no threshold gives
# many hits without false ones. The entity guard halves the false hits at low thresholds, but it cannot see "Business"
# against "Team".
#
# This is not an artefact that you can adjust away. A real embedder increases the paraphrase scores, but "the Team plan"
# and "the Business plan" stay near in any embedding. Thus the answer has three parts:
#
# - narrow cacheable classes,
# - a threshold that you select on *your* labelled traffic,
# - a guard or a verifier where a false hit is high-cost.
#
# ## Exercise 3.1 — an exact key that is safe
# Write `my_exact_key(namespace, request)`. It returns a SHA-256 hex digest over the namespace and **only** the fields
# that change the answer. These fields are model, messages, tools, tool_choice, response_format, temperature, top_p,
# max_completion_tokens, seed, reasoning_effort. Serialise them deterministically (sorted keys).

# %% exercise
import hashlib
import json


def my_exact_key(namespace, request):
    ### BEGIN SOLUTION
    fields = ("model", "messages", "tools", "tool_choice", "response_format", "temperature", "top_p",
              "max_completion_tokens", "seed", "reasoning_effort")
    body = {k: request[k] for k in fields if k in request}
    return hashlib.sha256(json.dumps([namespace, body], sort_keys=True).encode()).hexdigest()
    ### END SOLUTION

# %% check
base = {"model": "chat", "messages": [{"role": "user", "content": "What is SSO?"}], "temperature": 0}
same = [{**base, "stream": True}, {**base, "user": "u-17"}, {**base, "metadata": {"cache_class": "faq"}},
        dict(reversed(list(base.items())))]
differ = [{**base, "temperature": 1}, {**base, "model": "chat-pro"}, {**base, "tools": [{"type": "function"}]},
          {**base, "messages": [{"role": "user", "content": "What is SAML?"}]}, {**base, "reasoning_effort": "high"}]
k0 = my_exact_key("acme", base)
assert len(k0) == 64 and all(my_exact_key("acme", r) == k0 for r in same)
assert all(my_exact_key("acme", r) != k0 for r in differ) and my_exact_key("globex", base) != k0
print("✅ stream, user and metadata leave the key alone; the answer-changing fields and the tenant do not")

# %% [markdown]
# ## Exercise 3.2 — what the guard compares
# Write `my_entities(text)`. It returns the set of tokens that are **numbers** or **codes**, as whole words. A number is
# a digit, then any of digits and `. , / : -`. A code is an uppercase letter, then one or more uppercase letters or
# digits: `SSO`, `Q3`, `EUR`. Use `re`.

# %% exercise
import re


def my_entities(text):
    ### BEGIN SOLUTION
    return set(re.findall(r"\b(?:[A-Z][A-Z0-9]+|\d[\d.,/:-]*)\b", text))
    ### END SOLUTION

# %% check
texts = [x["q"] for x in s["seeds"]] + [p["q"] for p in s["probes"]]
assert all(my_entities(t) == set(cache.entities(t)) for t in texts)
blocked = [p["q"] for p in s["probes"] if p["label"] == "near_miss" and my_entities(p["q"]) != my_entities(seeds[p["seed"]])]
print(f"✅ the guard separates {len(blocked)} of 15 near misses:", blocked)

# %% [markdown]
# ## Exercise 3.3 — choose a threshold under a false-hit budget
# The budget of the route: at most **5 %** of cacheable lookups can get an incorrect answer from the cache. Examine the
# thresholds 0.50, 0.51, …, 1.00, with and without the guard. Select the `(tau, guard)` that maximises the hit rate in
# the budget. If there is a tie, prefer the higher threshold, then the guard on. Use `cache.sweep_thresholds`.

# %% exercise
grid = [round(0.5 + i / 100, 2) for i in range(51)]
### BEGIN SOLUTION
candidates = [(r["hit_rate"], r["threshold"], g) for g in (False, True)
              for r in cache.sweep_thresholds(grid, guard=g) if r["false_hit_rate"] <= 0.05]
best_hits, tau, guard = max(candidates)
### END SOLUTION

# %% check
table = {(g, r["threshold"]): r for g in (False, True) for r in cache.sweep_thresholds(grid, guard=g)}
ok = [(r["hit_rate"], th, g) for (g, th), r in table.items() if r["false_hit_rate"] <= 0.05]
assert (table[(guard, tau)]["hit_rate"], tau, guard) == max(ok)
print(f"✅ tau={tau} guard={guard}: {table[(guard, tau)]['hit_rate']:.1%} of paraphrases hit at "
      f"{table[(guard, tau)]['false_hit_rate']:.1%} false hits -- a small win, on a lexical embedder")

# %% [markdown]
# ## Exercise 3.4 — price a false hit
# At $\tau = 0.90$ with the guard, for each 44 cacheable lookups, the cache serves `right` correct answers and `wrong`
# incorrect answers (from `cache.sweep_thresholds`). A correct hit saves the §5.3 call on gemini-3.5-flash
# (`metering.price_call`, 5,000 in of which 2,700 cached, 350 out). Set `break_even` to the cost of one incorrect
# answer, in dollars, at which the cache saves exactly nothing.

# %% exercise
### BEGIN SOLUTION
r90 = cache.sweep_thresholds([0.9], guard=True)[0]
saving = metering.price_call("gemini-3.5-flash", 5000, 350, 2700)
break_even = r90["right"] * saving / r90["wrong"]
### END SOLUTION

# %% check
assert abs(break_even - 4 * 0.007005 / 2) < 1e-12
print(f"✅ break-even ${break_even:.5f} per wrong answer: any wrong answer that costs more than 1.4 cents "
      "(a support ticket, a refund, a wrong order status) makes this cache a loss")

# %% [markdown]
# ## Exercise 3.5 — which fields a namespace needs
# The `leaky` cache in the next cell is one semantic cache that every tenant and every user share. Its key is the text
# and nothing else. Thus the answer of one tenant reaches another tenant. The solution is a namespace, and its contents
# depend on the class that the route declared.
#
# A shared class (`faq`: "How do I reset my password?") has one correct answer for each **tenant**. Every user of that
# tenant can share it, and no other tenant can. A per-user class (`account`: "What is my plan limit?") has one correct
# answer for each **user**. If the users of the tenant share it, that is the incident of drill 1.
#
# Write `namespace(tenant, user, cache_class)`. It returns a string. For the classes in `PER_USER`, the namespace
# contains the tenant *and* the user. For every other class, it contains only the tenant. Tenant ids and user ids are
# arbitrary strings (they can contain `:` or `|`). Thus two different (tenant, user) pairs must never give the same
# namespace.

# %%
leaky = cache.SemanticCache(0.8)
leaky.store("shared", "How do I reset my password?", "acme: use the ACME SSO portal")
print("globex asks, leaky cache answers:", leaky.lookup("shared", "how do I reset my password")[0])
PER_USER = {"account"}

# %% exercise
def namespace(tenant, user, cache_class):
    ### BEGIN SOLUTION
    return json.dumps([cache_class, tenant, user if cache_class in PER_USER else None])
    ### END SOLUTION

# %% check
c = cache.SemanticCache(0.8)
put = lambda t, u, cls, q, a: c.store(namespace(t, u, cls), q, a)          # noqa: E731
get = lambda t, u, cls, q: c.lookup(namespace(t, u, cls), q)               # noqa: E731
put("acme", "alice", "faq", "How do I reset my password?", "acme: use the ACME SSO portal")
put("acme", "alice", "account", "What is my plan limit?", "alice: 10 seats")
assert get("acme", "bob", "faq", "how do I reset my password")[0].startswith("acme")   # shared inside the tenant
assert get("globex", "bob", "faq", "how do I reset my password") is None               # never across tenants
assert get("acme", "bob", "account", "what is my plan limit") is None                  # a per-user answer stays put
assert get("acme", "alice", "account", "what is my plan limit")[0] == "alice: 10 seats"
tricky = [("acme", "bob:x"), ("acme:bob", "x"), ("acme|bob", "x"), ("acme", "bob|x"), ("", "acmebob"), ("acmebob", "")]
assert len({namespace(t, u, "account") for t, u in tricky}) == len(tricky), "two identities share a namespace"
assert namespace("acme", "alice", "faq") != namespace("acme", "alice", "account")
print("✅ faq: one namespace per tenant; account: one per (tenant, user); no two identities collide")

# %% [markdown]
# ## Exercise 3.6 — what the provider's prompt cache saves instead
# The prompt cache of the provider is never incorrect. It bills cached input at ~10 % of the input price. Take
# the §5.3 call on **gpt-5.4-mini** (5,000 input tokens, 350 output). Use `metering.price_call` to set `uncached`,
# `cached` (2,700 of the input cached) and `saved_share` (the fraction of the uncached cost that the cache saves).

# %% exercise
### BEGIN SOLUTION
uncached = metering.price_call("gpt-5.4-mini", 5000, 350)
cached = metering.price_call("gpt-5.4-mini", 5000, 350, 2700)
saved_share = 1 - cached / uncached
### END SOLUTION

# %% check
assert abs(uncached - (5000 * 0.75 + 350 * 4.5) / 1e6) < 1e-12 and abs(cached - 0.0035025) < 1e-12
print(f"✅ ${uncached:.6f} -> ${cached:.6f}: prompt caching saves {saved_share:.0%} of this call with zero false hits; "
      f"a semantic hit saves 100 % of it and can be wrong")
salt = keys.cache_salt("acme", b"gateway-salt-secret")
print(f"   for a vLLM pool the engine's prefix cache is isolated per tenant with cache_salt={salt[:10]}... ({len(salt)} chars)")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "There are four caches on the path. The prompt cache of the provider and the prefix cache
# of the engine reuse computation. They decrease cost and TTFT, and they never change an answer. Thus we design the
# layout of the prompts for them, and we isolate the cache of the engine with a per-tenant `cache_salt`.
#
# "The exact cache and the semantic cache of the gateway reuse answers. Thus they serve only the classes that a route
# declares cacheable, in a namespace with the verified tenant as its key. They never serve personal or time-sensitive
# requests, or requests that use tools. The exact key hashes every field that changes the answer, and nothing else.
#
# "The semantic cache needs a threshold that we select on labelled traffic. On our sample, a lexical embedder ranks near
# misses above paraphrases. The entity guard halves the false hits, but it cannot see a changed word. At a 5 %
# false-hit budget, the cache hits a fifth of the paraphrases. If an incorrect answer costs more than about a cent and a
# half, that cache is a loss. Thus, for anything that matters, we cache narrowly, put a guard on entities and add a
# verifier."
#
# **Drill questions**
# 1. *The semantic cache answered one user with another user's order status. What went wrong?* A personal class was
#    cacheable, and the namespace had no tenant or user. Also, a near miss of one digit went above the threshold. Declare the
#    classes for each route. Make the namespace from the verified tenant. Put a guard on numbers and IDs. Measure the
#    false hits first.
# 2. *Why is `stream` not part of the exact key?* It changes the transport, not the answer. The gateway can send a
#    cached answer as a stream again. `temperature`, tools and `reasoning_effort` change the answer, and they are in
#    the key.
# 3. *Is the engine's prefix cache a semantic cache?* No. It is exact (chained block hashes over the token prefix). It
#    reuses K/V, not answers, and it cannot be incorrect. Its risk is a timing side channel between tenants, and
#    `cache_salt` closes that channel.
