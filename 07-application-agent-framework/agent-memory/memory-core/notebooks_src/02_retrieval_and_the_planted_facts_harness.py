# %% [markdown]
# # 02 · Retrieval and the planted-facts harness
#
# **Tier:** T0. It uses only a CPU, it needs no model and no network, and it takes less than ten seconds. A real
# embedder on the paraphrase subset is in `memory-lab` notebook `05_evaluate_forget_and_audit` (T1, optional).
#
# ## The one-minute version
# Retrieval decides which memories get to the prompt. Thus it decides what the agent "remembers". Similarity alone
# returns the memory that is most *on-topic*. The generative-agents score (Park et al. 2023) adds **recency** and
# **importance**, each normalised to [0, 1] over the candidates. The score has two forms that disagree:
#
# - The paper's form: all weights 1, and recency 0.995 per hour since last access (verify).
# - The reference code's form: weights 0.5 / 3 / 2, and a rank-based recency that gives the **oldest** memory the
#   largest term.
#
# Whatever the score is, the result goes into a **token budget** for each turn.
#
# You cannot adjust any of this without a benchmark. Thus the harness does these steps:
#
# - It plants facts about a user across sessions.
# - It changes one fact.
# - It puts a false claim in through a tool.
# - It asks questions in LongMemEval's and LoCoMo's task shapes: extraction, preference, multi-session, temporal,
#   knowledge update, abstention, adversarial.
#
# It reports accuracy with a Wilson interval, recall within the budget, stale answers and abstention. It also reports a
# paraphrase subset that a lexical embedder misses by design.
#
# Primer: §3 *Retrieval: similarity, recency and importance*, §4 *Measuring memory* (`../PRIMER.md`).

# %%
import math

import numpy as np

from memcore import (DAY, HOUR, HashingEmbedder, MemoryRecord, MemoryStore, Scope, Writer, build_store, evaluate,
                     generate, knee, minmax, pack, recall_vs_budget, recency, retrieve, score, summarize)

ALICE = Scope("acme", "alice")
emb = HashingEmbedder()

# %% [markdown]
# ## Worked example 1 — the embedder is lexical, on purpose
# memcore implements `ragkit`'s crc32 hashing embedder (07.4) again. Each `[a-z0-9]+` token adds 1 to bucket
# `crc32(token) % 1024`. Then the embedder L2-normalises the vector. Thus a dot product is a cosine. Two texts are
# similar if, and only if, they share tokens.

# %%
q = emb.encode("user lives in lisbon")
for other in ["the user moved to porto", "Where does the user live?", "user lives in lisbon"]:
    print(f"cos(user lives in lisbon, {other!r:32}) = {float(q @ emb.encode(other)):.4f}")

# %% [markdown]
# $0.2236 = 1/\sqrt{4 \cdot 5}$: one shared token (`user`) out of four and five. "Where does the user live?" is the same
# question that a person asks about the first text. But its score is not better than the score of the unrelated move,
# because `live` is not `lives`. That is the **paraphrase miss** that the harness measures in worked example 5. A real
# embedder closes most of it (T1).
#
# ## Worked example 2 — three memories, two scoring forms
# The three memories are A (last used 1 h ago, importance 2, cosine 0.80), B (24 h, 9, 0.50) and C (72 h, 5, 0.20).

# %%
now = 100 * HOUR
last = [now - h * HOUR for h in (1, 24, 72)]
imp, rel = [2, 9, 5], [0.80, 0.50, 0.20]
for form in ("paper", "code"):
    r = recency(last, now, form)
    print(f"{form:5}: raw recency {np.round(r, 4)} -> normalised {np.round(minmax(r), 4)}")
    print(f"       importance {np.round(minmax(imp), 4)}  relevance {np.round(minmax(rel), 4)}")
    print(f"       score A, B, C = {np.round(score(last, imp, rel, now, form), 4)}")

# %% [markdown]
# Both forms rank B, A, C here, but for different reasons. In the paper form, recency and relevance cancel between A
# and B, and the importance of B decides. In the code form, relevance has the weight 3 and importance has the weight 2.
# The recency term is $0.99^{\text{rank}}$ over the memories, sorted by last access, oldest first. This term gives
# **C, the stalest**, the full recency point.
#
# That inversion is not important on this example. It is important on a knowledge update. There, the stale fact is
# exactly the one that the term favours (the harness shows it after exercise 2.3). The paper's form comes from the
# paper alone (arXiv is not available from here: verify). The code form comes from
# `joonspk-research/generative_agents` (`retrieve.py`, read on 2026-09-26).
#
# ## Worked example 3 — pack into a budget, filter as of a date

# %%
store, w = MemoryStore(), None
w = Writer(store)
for day, city in [(0, "Lisbon"), (4, "Porto")]:
    w.write(MemoryRecord(f"The user's home city is {city}.", "semantic", ALICE, "user", key="home_city",
                         value=city, importance=6, created_at=day * DAY))
for day, text, key, value, importance in [(1, "The user's pet is cat.", "pet", "cat", 3),
                                          (2, "The user's allergy is peanuts.", "allergy", "peanuts", 9)]:
    w.write(MemoryRecord(text, "semantic", ALICE, "user", key=key, value=value, importance=importance,
                         created_at=day * DAY))
for budget in (15, 30, 60):
    got = retrieve(store, ALICE, "What is the user's home city?", now=6 * DAY, budget_tokens=budget, touch=False)
    print(f"budget {budget:3}: {got.tokens:3} tokens, {[r.value for r in got.records]}")
past = retrieve(store, ALICE, "What was the user's home city?", now=6 * DAY, as_of=2 * DAY, touch=False)
print("as of day 2:", [r.render() for r in past.records][:2])

# %% [markdown]
# A normal query does not see the superseded Lisbon fact, but an as-of query returns it. That is why an update closes a
# fact and does not delete it. Hybrid search (BM25 + vectors + RRF) is the topic of 07.4 (`ragkit.reference`), and it
# applies with no change. An ANN index such as `minifaiss`'s HNSW (M0 = 2M) gives a benefit only at tenant scale. The
# reason is that one user has tens to thousands of records in memory, and a flat scan of one partition is exact and
# fast.
#
# ## Worked example 4 — a planted-facts scenario

# %%
sc = generate(0)
for t, s, text, source in sc.turns[:4] + sc.turns[-6:]:
    print(f"day {t / DAY:4.2f} session {s} [{source:4}] {text}")
print()
for q in sc.questions:
    print(f"{q.qtype:16} {'(paraphrase) ' if q.paraphrase else ''}{q.text!r:48} -> {q.answer}")

# %% [markdown]
# The harness uses the shapes, not the data. The shapes are LongMemEval's question types and LoCoMo's adversarial
# category (unanswerable questions). The LongMemEval types are single-session user/assistant/preference,
# multi-session, temporal reasoning and knowledge update. An `_abs` id suffix marks abstention.
#
# LoCoMo's data is CC BY-NC 4.0, and the harness bundles none of it. The harness downloads nothing. Here the adversarial
# question asks about a fact that only a **tool result** asserted. The write path put that fact in quarantine. Thus
# the correct answer is "I don't know".
#
# **A fixture artefact, disclosed.** Look at the planted statements: "I live in Prague. (about my home city)". No real
# user adds that hint. The hint is there to give the template extractor and the lexical embedder something to use on
# raw episodes. It gives the embedder the words of the question ("home", "city").
#
# Thus the raw-episode numbers in this notebook make raw episodes look better than they are. `generate(hint=False)`
# removes the hints, and the cell after worked example 6 shows the difference. The hints do not change consolidated
# facts: their text is the template "The user's home city is …" in both cases.
#
# ## Worked example 5 — evaluate, with an interval

# %%
runs = [(sc, build_store(sc, "consolidated")) for sc in map(generate, range(30))]
res = summarize([o for sc, st in runs for o in evaluate(sc, st, 60, ("semantic", "procedural"))])
lo, hi = res["wilson"]
print(f"{res['n']} questions at a 60-token budget: accuracy {res['accuracy']:.1%} (95% Wilson {lo:.1%}-{hi:.1%}), "
      f"recall {res['recall']:.1%}, stale {res['stale']:.1%}, abstention {res['abstention']:.0%}")
for k, v in res["by_type"].items():
    print(f"   {k:26} {v:.0%}")
c_lo, c_hi = res["cluster"]
print(f"resampling users instead of questions (a cluster bootstrap): {c_lo:.1%}-{c_hi:.1%}")

# %% [markdown]
# All answers are correct except the preference paraphrase. "Where does the user like to sit on a plane?" shares no
# token with "The user's seat preference is aisle.", the stored fact. Every fact ties on relevance, and the fact with
# the lowest importance loses its place in the budget. The reader is a strict template reader. Thus each miss is a
# retrieval miss or a write-path miss. That is the purpose of a memory benchmark.
#
# The Wilson interval treats the 390 questions as 390 independent trials. They are not independent. The thirteen
# questions of each user share one store and one write path. Here, each of the 30 users misses the *same* question.
#
# If you resample users (the `cluster` interval), the interval becomes a point. Thus the uncertainty of this harness is
# in which question types it asks, not in which users. Compare designs for each question type (`by_type`). Read an
# interval as a statement about this generator, not about your users.
#
# ## Worked example 6 — recall against the budget, raw episodes against facts

# %%
for mode in ("consolidated", "episodes"):
    rows = recall_vs_budget(mode=mode)
    print(mode, "knee at", knee(rows), "tokens")
    for r in rows:
        print(f"   budget {r['budget']:4}: recall {r['recall']:6.1%}  accuracy {r['accuracy']:6.1%}  "
              f"tokens used {r['tokens']:5.1f}")

# %% [markdown]
# At 90 tokens, the whole profile fits, and facts get to full recall. But raw episodes (long, in the user's words,
# filler included) are at 40% with 120, and that is with the help of the slot hints:

# %%
for hint in (True, False):
    pairs = [(s, build_store(s, "episodes")) for s in (generate(i, hint=hint) for i in range(30))]
    r = summarize([o for s, st in pairs for o in evaluate(s, st, 60, ("episodic",))])
    print(f"raw episodes at 60 tokens, slot hints {'on ' if hint else 'off'}: recall {r['recall']:.1%}")

# %% [markdown]
# The curve of the episodes is the best case of the fixture. Look at the accuracy of the episodes at 15 tokens: 15.4%
# with **zero** recall. The two unanswerable questions of each user are "correct" for a reader that knows nothing.
# Report recall and abstention separately. If you do not, a memory that stores nothing looks like it works.
#
# ## Exercise 2.1 — the paper's score
# Write `paper_score(hours_since_access, importance, relevance)`. Min-max normalise each term over the candidates. If
# all values are equal, the term is 0.5. Use $\text{recency} = 0.995^{\text{hours}}$ and the weights 1, 1, 1.

# %% exercise
def paper_score(hours, importance, relevance):
    ### BEGIN SOLUTION
    def norm(x):
        x = np.asarray(x, dtype=float)
        span = x.max() - x.min()
        return np.full_like(x, 0.5) if span == 0 else (x - x.min()) / span
    return norm(0.995 ** np.asarray(hours, dtype=float)) + norm(importance) + norm(relevance)
    ### END SOLUTION

# %% check
rng = np.random.default_rng(0)
for _ in range(50):
    n = rng.integers(2, 8)
    h, i, r = rng.uniform(0, 500, n), rng.integers(1, 11, n), rng.uniform(-1, 1, n)
    np.testing.assert_allclose(paper_score(h, i, r), score(now - h * HOUR, i, r, now, "paper"), atol=1e-9)
np.testing.assert_allclose(paper_score([1, 24, 72], [2, 9, 5], [0.8, 0.5, 0.2]), [2.0, 2.1364, 0.4286], atol=5e-5)
print("✅ the paper form: A 2.0000, B 2.1364, C 0.4286")

# %% [markdown]
# ## Exercise 2.2 — the code's recency
# Write `code_recency(last_accessed)` the same way as the reference code. Sort by last access, **oldest first**.
# Give the $i$-th memory (from 1) the value $0.99^i$. Return the values in the order of the input.

# %% exercise
def code_recency(last_accessed):
    ### BEGIN SOLUTION
    order = sorted(range(len(last_accessed)), key=lambda k: last_accessed[k])
    out = [0.0] * len(last_accessed)
    for rank, k in enumerate(order, start=1):
        out[k] = 0.99 ** rank
    return out
    ### END SOLUTION

# %% check
for _ in range(50):
    t = list(rng.permutation(20)[: rng.integers(1, 10)] * 3600.0)
    np.testing.assert_allclose(code_recency(t), recency(t, max(t), "code"))
oldest = code_recency([now - 72 * HOUR, now - HOUR])
assert oldest[0] > oldest[1]
print("✅ the code form hands the stalest memory the largest recency term")

# %% [markdown]
# ## Exercise 2.3 — which one wins?
# There are two memories about the same slot. The **old** memory was last used 30 days ago, with importance 6 and
# cosine 0.6. The **new** memory was last used 2 hours ago, with importance 6 and cosine 0.6. With only these two candidates, which memory
# ranks first in each form? Set `first_paper` and `first_code` to `"old"`, `"new"` or `"tie"`.

# %% exercise
### BEGIN SOLUTION
first_paper, first_code = "new", "old"
### END SOLUTION

# %% check
s_p = score([now - 30 * DAY, now - 2 * HOUR], [6, 6], [0.6, 0.6], now, "paper")
s_c = score([now - 30 * DAY, now - 2 * HOUR], [6, 6], [0.6, 0.6], now, "code")
name = lambda s: "tie" if s[0] == s[1] else ("old" if s[0] > s[1] else "new")
assert (first_paper, first_code) == (name(s_p), name(s_c))
print(f"✅ paper {np.round(s_p, 3)}, code {np.round(s_c, 3)}: on a knowledge update the code form serves the stale fact")

# %% [markdown]
# The harness shows the damage at scale. On raw episodes at a 60-token budget, the code form answers 70% of the
# knowledge-update questions with the old value. The paper form answers 0% of them with the old value. When you remove
# the slot hints, the gap decreases, but it keeps its direction.

# %%
for hint in (True, False):
    eruns = [(sc, build_store(sc, "episodes")) for sc in (generate(i, hint=hint) for i in range(30))]
    for form in ("paper", "code"):
        r = summarize([o for sc, st in eruns for o in evaluate(sc, st, 60, ("episodic",), form)])
        print(f"hints {'on ' if hint else 'off'} {form:5}: recall {r['recall']:.1%}  stale answers {r['stale']:.1%}")

# %% [markdown]
# Read both columns. The *recall* of the code form is higher, because its relevance weight of 3 suits long episodes.
# But 70% of its knowledge-update answers are stale (27% without the hints). In this harness, a single aggregate selects
# the incorrect form. The metrics for each type (stale answers, abstention) find the problem.
#
# ## Exercise 2.4 — pack into the budget
# Write `pack_budget(records, budget)`. Go through the records in the given order (best first). Take each record whose
# `tokens` still fits. **Skip** a record that does not fit, and continue to look.

# %% exercise
def pack_budget(records, budget):
    ### BEGIN SOLUTION
    out, used = [], 0
    for r in records:
        if used + r.tokens <= budget:
            out.append(r)
            used += r.tokens
    return out
    ### END SOLUTION

# %% check
recs = [MemoryRecord("x" * n, "semantic", ALICE, "user") for n in (60, 200, 20, 90, 10)]
for b in (0, 10, 30, 35, 60, 200):
    assert pack_budget(recs, b) == pack(recs, b), b
print("✅ greedy with skip — a long record never blocks shorter ones behind it")

# %% [markdown]
# ## Exercise 2.5 — the Wilson interval
# Write `wilson(passes, n, z=1.96)` (07.2 notebook 08 §4). With $p = \text{passes}/n$:
#
# $$
# \text{centre} = \frac{p + z^2/2n}{1 + z^2/n}, \qquad
# \text{half-width} = \frac{z\sqrt{p(1-p)/n + z^2/4n^2}}{1 + z^2/n},
# $$
#
# Clip the result to [0, 1]. When $n = 0$, return `(0.0, 1.0)`.

# %% exercise
def wilson(passes, n, z=1.96):
    ### BEGIN SOLUTION
    if n == 0:
        return (0.0, 1.0)
    p, z2 = passes / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (max(0.0, centre - half), min(1.0, centre + half))
    ### END SOLUTION

# %% check
assert tuple(round(x, 4) for x in wilson(45, 50)) == (0.7864, 0.9565)
assert tuple(round(x, 4) for x in wilson(0, 20)) == (0.0, 0.1611)
assert wilson(0, 0) == (0.0, 1.0)
print(f"✅ 360/390 correct is {wilson(360, 390)[0]:.1%}-{wilson(360, 390)[1]:.1%}; "
      f"12/13 for one user is {wilson(12, 13)[0]:.0%}-{wilson(12, 13)[1]:.0%} - too wide to compare two designs")

# %% [markdown]
# ## Exercise 2.6 — find the knee
# Write `find_knee(rows, tol=0.02)`. It returns the smallest budget whose recall is within `tol` of the best recall of
# all budgets. Then select `my_budget` for the consolidated store of this harness. Give the reason in a comment.

# %% exercise
def find_knee(rows, tol=0.02):
    ### BEGIN SOLUTION
    best = max(r["recall"] for r in rows)
    return min(r["budget"] for r in rows if r["recall"] >= best - tol)
    ### END SOLUTION

### BEGIN SOLUTION
my_budget = 90        # full recall; 60 loses 9% of recall to save 14 tokens a turn
### END SOLUTION

# %% check
rows = recall_vs_budget(mode="consolidated")
assert find_knee(rows) == knee(rows) == 90 and find_knee(rows, tol=0.1) == knee(rows, tol=0.1)
assert my_budget in [r["budget"] for r in rows] and my_budget >= 45
print(f"✅ knee at {find_knee(rows)} tokens; you chose {my_budget}")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "Retrieval decides the quality of memory. Thus we score and we measure. Each candidate
# in the user's partition gets the generative-agents score: recency, importance and relevance, each min-max
# normalised. We use the paper's form: recency decreases with the hours since last use.
#
# "The form of the reference code ranks by access order, oldest first. In our harness, it served stale facts on 70% of knowledge updates over raw
# episodes (27% when we removed the slot hints of the harness). Thus we do not copy it.
#
# "We pack the ranked list into a token budget for each turn, greedily with skip. As-of queries can get to closed
# facts.
#
# "We measure on a planted-facts harness in LongMemEval's and LoCoMo's task shapes. We use their shapes, not their
# data. We report accuracy with a Wilson interval, recall within the budget, stale answers and abstention separately.
# We report them separately because a memory that stores nothing still gets the abstention questions correct. The knee of recall against
# budget sets the budget: 90 tokens here. A paraphrase subset tells us the gain that a real embedder must give over a
# lexical embedder."
#
# **Drill questions**
# 1. *Why add recency and importance to similarity?* Similarity finds the memory that is most on-topic. But the most
#    useful memory is often more recent (a knowledge update) or more important (an allergy) than the nearest match.
# 2. *Accuracy is 92% on 13 questions. Ship?* 12/13 is a 67–99% Wilson interval, and that is too wide. Run hundreds of
#    questions (390 here: 89–95%). Remember that they are not independent (resample users, compare per question type).
#    Read recall, stale answers and abstention separately.
# 3. *When do you need an ANN index for memory?* You need one when one partition is large. But the memory of one user
#    is small. Thus a flat scan of the partition is exact. HNSW (M0 = 2M in minifaiss) gives a benefit at tenant or
#    corpus scale.
