# %% [markdown]
# # 04 · Consolidation, forgetting and deletion
#
# **Tier:** T0 — CPU only, no network, a few seconds. The same job with a lease row and checkpoints on SQLite, a
# crash and a resume, and the GCP schedule printed (Cloud Run job on Cloud Scheduler) is `memory-lab` notebook
# `04_consolidation_as_a_scheduled_job`; a deletion checked on the bytes of the database file is its notebook
# `05_evaluate_forget_and_audit`.
#
# ## The one-minute version
# Raw episodes are cheap to write and expensive to read: long, repetitive, full of values that have since changed.
# **Consolidation** turns a window of episodes into facts, on a schedule, with rules rather than a summary: the
# highest-precedence source wins (human > user > tool > inferred), newer supersedes older **and the older fact is
# kept, closed** (`valid_to`, bi-temporal, as Graphiti closes an edge with `invalid_at`), and a weaker contradiction
# is flagged for review. It runs as a durable job: a deterministic run id, a lease, a checkpoint per slot, ids that
# make a re-applied slot overwrite rather than duplicate. **Forgetting** is four mechanisms — decay, TTL, the per-turn
# budget, a cap per scope — and none of them is **deletion**. Deleting a fact means finding every copy: the record,
# its vector and full-text postings, facts derived from it, anything that quotes it, cached prompt prefixes, logs,
# eval sets and backups. You prove a deletion by searching for the data afterwards.
#
# Primer: §7 *Consolidation, forgetting and deletion* (`../PRIMER.md`).

# %%
from memcore import (DAY, Budget, Consolidator, Crash, LeaseHeld, MemoryRecord, MemoryStore, PrefixCache, Scope,
                     Surfaces, Writer, build_store, cap, evaluate, expire, extract, generate, plan_key, propagate,
                     reflect, residue, retention, summarize, token_ids)

ALICE = Scope("acme", "alice")

# %% [markdown]
# ## Worked example 1 — the rules for one slot
# `plan_key` takes the statements about one slot in the window — `(time, value, source, evidence id)` — and an
# existing fact, and returns the facts to write (value, valid from, valid to, source, evidence) and the flags.

# %%
statements = [(1 * DAY, "Lisbon", "user", "e1"), (4 * DAY, "Porto", "user", "e2"), (5 * DAY, "porto", "user", "e3"),
              (6 * DAY, "Madrid", "inferred", "e4")]
facts, flags = plan_key(statements)
for value, vfrom, vto, source, evidence, _ in facts:
    print(f"{value:7} valid day {vfrom / DAY:g} to {'now' if vto is None else f'day {vto / DAY:g}'}  ({source}, evidence {evidence})")
print("flags:", [(st[1], st[2], why) for st, why in flags])
human = MemoryRecord("The user's home city is Oslo.", "semantic", ALICE, "human", key="home_city", value="Oslo")
print("with a human-entered fact on file:", plan_key(statements, human))

# %% [markdown]
# Porto twice is one fact with two pieces of evidence; Lisbon is closed at day 4, not deleted; the inferred Madrid
# is weaker than the user's statements and only flagged. With a human-entered Oslo on file, none of the user's
# statements is applied — they are all flagged for a person to look at. Precedence is a product decision; making it
# explicit is what lets you test it.
#
# ## Worked example 2 — a consolidation run

# %%
def seeded_store():
    store = MemoryStore()
    w = Writer(store)
    for day, text in [(0, "I live in Lisbon."), (1, "I prefer window seats."), (2, "I work at Acme."),
                      (3, "I moved to Porto."), (3.5, "My pet is a cat.")]:
        w.write(extract(f"Quick note. {text}", ALICE, at=day * DAY)[0])       # episodes only: the cheap write path
    w.write(extract("The user's employer is Evilcorp.", ALICE, source="tool", at=2.5 * DAY)[0])   # quarantined
    return store


store = seeded_store()
report = Consolidator(store).run(ALICE, 0, 7 * DAY, now=7 * DAY)
print(report.run_id, "|", report.status, "| facts written", len(report.written), "| of which closed",
      len(store.records(ALICE, status="superseded")), "| flagged", report.flagged)
for r in store.records(ALICE, status=None):
    if r.kind == "semantic":
        print(f"  {r.status:10} {r.render()}   evidence {r.provenance}")

# %% [markdown]
# The tool's Evilcorp episode was quarantined on write, so consolidation never read it: nothing a tool said is
# promoted without review. The run id is deterministic per (user, window) — `consolidate:acme:alice:day0-7` — the
# way lra-gcp primer §3.13 names a scheduled run (`weekly-review-2026-W37`), so a double fire of the schedule is one
# run.
#
# ## Worked example 3 — a crash, a lease, a resume

# %%
store = seeded_store()
job = Consolidator(store, lease_ttl=60)
try:
    job.run(ALICE, 0, 7 * DAY, now=7 * DAY, worker="w1", crash_after=2)
except Crash as e:
    print("crash:", e, "| checkpoint:", job.checkpoints[report.run_id]["done"])
try:
    job.run(ALICE, 0, 7 * DAY, now=7 * DAY + 30, worker="w2")
except LeaseHeld as e:
    print("30 s later, w2:", e)
again = job.run(ALICE, 0, 7 * DAY, now=7 * DAY + 61, worker="w2")
print("61 s later, w2:", again.status, "| skipped", again.skipped, "| semantic records:",
      len(store.records(ALICE, kind="semantic", status=None)))
print("the schedule fires twice:", job.run(ALICE, 0, 7 * DAY, now=8 * DAY).status)

# %% [markdown]
# The dead worker held the lease; nobody else could run the job until it expired (lra-gcp primer §3.3: leases, not
# locks — the reaper's job). The resumed run skipped the two checkpointed slots, and because each fact's id is
# derived from (run id, slot, position), a slot re-applied after a crash overwrites instead of duplicating. On GCP
# this is a Cloud Run job fired by Cloud Scheduler (the lab's `deploy/gcp/`).
#
# ## Worked example 4 — facts beat raw episodes at a fixed budget

# %%
runs = {mode: [(sc, build_store(sc, mode)) for sc in map(generate, range(30))] for mode in ("episodes", "consolidated")}
kinds = {"episodes": ("episodic",), "consolidated": ("semantic", "procedural")}
for mode, pairs in runs.items():
    s = summarize([o for sc, st in pairs for o in evaluate(sc, st, 60, kinds[mode])])
    print(f"{mode:12} at 60 tokens: recall {s['recall']:.1%}, accuracy {s['accuracy']:.1%}, tokens {s['tokens']:.1f}")

# %% [markdown]
# ## Worked example 5 — reflection, in brief
# Generative agents write *insights* when the importance of recent events sums past a trigger (150 in the reference
# code), each citing its evidence. The job has explicit exits, like lra-gcp primer §3.8's reflection loop: below the
# trigger, done, a maximum number of insights, budget exhausted.

# %%
store = MemoryStore()
w = Writer(store)
for i in range(30):
    w.write(extract(["I live in Lisbon.", "I work at Acme.", "I am learning Norwegian."][i % 3] + f" (note {i})",
                    ALICE, at=i * DAY / 3)[0])
print("importance of new episodes:", sum(e.importance for e in store.records(ALICE)))
insights, why = reflect(store, ALICE, now=10 * DAY)
print(why, [(r.text, len(r.provenance)) for r in insights])
print("with a budget of one model call:", reflect(store, ALICE, now=10 * DAY,
      budget=Budget({"memory_tokens": 0, "writes": 0, "llm_calls": 1, "usd": 0}))[1])

# %% [markdown]
# ## Worked example 6 — forgetting is not deleting

# %%
old = MemoryRecord("User said: parking at the office is a pain.", "episodic", ALICE, "user", importance=3, created_at=0)
key = MemoryRecord("The user's allergy is peanuts.", "semantic", ALICE, "user", importance=9, created_at=0)
for days in (0, 30, 90):
    print(f"day {days:3}: retention trivia {retention(old, days * DAY):.3f}, allergy {retention(key, days * DAY):.3f}")
store = MemoryStore()
for i in range(6):
    store.put(MemoryRecord(f"event {i}", "episodic", ALICE, "user", importance=i + 1, created_at=0,
                           ttl_s=7 * DAY if i < 2 else None))
print("TTL expired:", len(expire(store, ALICE, now=10 * DAY)), "| over the cap of 2:", len(cap(store, ALICE, 2, now=10 * DAY)),
      "| left:", [r.text for r in store.records(ALICE)])

# %% [markdown]
# Decay and the per-turn budget make a memory *unlikely to be seen*; TTL and caps remove records on a schedule. None
# of them answers "forget my address": that needs deletion with propagation.
#
# ## Worked example 7 — a deletion, followed to every copy

# %%
def world():
    store = MemoryStore()
    w = Writer(store)
    for day, text in [(0, "I live in Lisbon and I work at Acme."), (2, "I prefer window seats.")]:
        for rec in extract(text, ALICE, at=day * DAY):
            w.write(rec)
    store.put(MemoryRecord("The user cycles to work in Lisbon.", "semantic", ALICE, "inferred", created_at=3 * DAY))
    cache = PrefixCache(4)
    cache.serve(token_ids("system prompt + The user's home city is Lisbon."), salt="acme/alice")
    return Surfaces(store, cache, logs=["08:00 recall -> The user's home city is Lisbon.", "08:01 ok"],
                    eval_cases=[{"text": "Q: home city? A: Lisbon", "provenance": ()}],
                    backups=[{r.id: r.text for r in store.records(ALICE, status=None)}])


s = world()
print("before:", residue(s, ALICE, ["Lisbon"]))
rep = propagate(s, ALICE, key="home_city")
print(rep.checklist())

# %% [markdown]
# Three records went, not one: the fact, the episode it came from (it says "Lisbon" too) and an inferred insight that
# quotes it. The employer fact extracted from the same episode survives — it does not contain the data. The cached
# prefixes under the user's salt are evicted (a cached prefix cannot be edited), the log line is redacted, the eval
# case is dropped. The backup still holds three copies: backups are not rewritten; they expire on their retention
# schedule, or their encryption key is shredded. Say which, with a date, when you answer a deletion request.
#
# ## Exercise 4.1 — the consolidation rules
# Implement `plan(statements)` for statements `(time, value, source)` about one slot: keep only the statements from
# the highest-precedence source present; walk them in time order; a value equal (ignoring case) to the previous one
# extends it; a different one closes the previous fact at its own time and opens a new one. Return
# `[(value, valid_from, valid_to_or_None)]`, and the list of weaker statements whose value differs from the value
# that held at their time.

# %% exercise
TRUST = {"human": 3, "user": 2, "tool": 1, "inferred": 0}

def plan(statements):
    ### BEGIN SOLUTION
    best = max(TRUST[s] for _, _, s in statements)
    facts = []
    for t, v, s in sorted(x for x in statements if TRUST[x[2]] == best):
        if facts and facts[-1][0].lower() == v.lower():
            continue
        if facts:
            facts[-1][2] = t
        facts.append([v, t, None])
    def held(t):
        return next((f[0] for f in reversed(facts) if f[1] <= t), "")
    weak = [x for x in statements if TRUST[x[2]] < best and held(x[0]).lower() != x[1].lower()]
    return [tuple(f) for f in facts], weak
    ### END SOLUTION

# %% check
import random
rng = random.Random(0)
for _ in range(300):
    sts = [(rng.randrange(20) * DAY + i, rng.choice(["Lisbon", "lisbon", "Porto", "Oslo"]), rng.choice(list(TRUST)))
           for i in range(rng.randrange(1, 7))]
    want_facts, want_flags = plan_key([(t, v, s, f"e{i}") for i, (t, v, s) in enumerate(sts)])
    got_facts, got_weak = plan(sts)
    assert got_facts == [(f[0], f[1], f[2]) for f in want_facts], sts
    assert sorted(got_weak) == sorted((st[0], st[1], st[2]) for st, _ in want_flags), sts
print("✅ precedence first, then time; the old fact is closed, not dropped; weaker contradictions are flagged")

# %% [markdown]
# ## Exercise 4.2 — the lease rule
# Write `can_acquire(holder, worker, now)`: `holder` is `None` or `(worker_id, expires_at)`. A worker may take the
# lease if nobody holds it, if it already holds it (a heartbeat renews), or if the holder's lease has expired.

# %% exercise
def can_acquire(holder, worker, now):
    ### BEGIN SOLUTION
    return holder is None or holder[0] == worker or holder[1] <= now
    ### END SOLUTION

# %% check
for holder, worker, now in [(None, "w1", 0), (("w1", 60), "w1", 30), (("w1", 60), "w2", 30), (("w1", 60), "w2", 60),
                            (("w1", 60), "w2", 61)]:
    job = Consolidator(MemoryStore())
    if holder:
        job.leases["r"] = holder
    try:
        job.acquire("r", worker, now)
        real = True
    except LeaseHeld:
        real = False
    assert can_acquire(holder, worker, now) == real, (holder, worker, now)
print("✅ a lease expires - that is what lets a dead worker's job be picked up")

# %% [markdown]
# ## Exercise 4.3 — predict the retention
# `retention = importance / 10 × 0.5 ** (days since last use / 30)`. Predict, to three decimals, the retention of an
# importance-8 memory last used 45 days ago, and how many days an importance-4 memory takes to fall below 0.1.

# %% exercise
### BEGIN SOLUTION
r_45 = 0.283
days_below = 61          # 0.4 x 0.5 ** (d / 30) < 0.1  <=>  d > 60
### END SOLUTION

# %% check
m8 = MemoryRecord("x", "semantic", ALICE, "user", importance=8, created_at=0)
m4 = MemoryRecord("y", "semantic", ALICE, "user", importance=4, created_at=0)
assert abs(r_45 - retention(m8, 45 * DAY)) < 5e-4
assert retention(m4, (days_below - 1) * DAY) >= 0.1 > retention(m4, days_below * DAY)
print(f"✅ {retention(m8, 45 * DAY):.3f}; an importance-4 memory drops below 0.1 on day {days_below}")

# %% [markdown]
# ## Exercise 4.4 — what is derived from what
# Implement `derived(records, targets)`: the ids of `targets` plus every record whose `provenance` cites any id
# already in the set, repeated until nothing changes (facts derived from facts derived from the target).

# %% exercise
def derived(records, targets):
    ### BEGIN SOLUTION
    found = set(targets)
    while True:
        more = {r.id for r in records if r.id not in found and set(r.provenance) & found}
        if not more:
            return found
        found |= more
    ### END SOLUTION

# %% check
a = MemoryRecord("A", "episodic", ALICE, "user", id="a")
b = MemoryRecord("B", "semantic", ALICE, "inferred", id="b", provenance=("a",))
c = MemoryRecord("C", "semantic", ALICE, "inferred", id="c", provenance=("b", "z"))
d = MemoryRecord("D", "semantic", ALICE, "user", id="d", provenance=("q",))
assert derived([a, b, c, d], {"a"}) == {"a", "b", "c"} and derived([a, b, c, d], {"d"}) == {"d"}
print("✅ provenance is what makes 'delete what was derived from it' computable")

# %% [markdown]
# ## Exercise 4.5 — the naive deletion
# Delete the home-city fact the naive way — `store.delete` on the records whose `key` is `home_city`, nothing
# else — on a fresh `world()`. Then fill `left` with the number of copies of "Lisbon" each surface still holds, as
# `residue` would report it (predict first, then check with `residue`).

# %% exercise
s = world()
for r in s.store.find(ALICE, "home_city", status=None):
    s.store.delete(ALICE, r.id)
### BEGIN SOLUTION
left = {"records": 2, "vectors": 2, "fulltext_postings": 2, "prompt_cache_blocks": 2, "logs": 1, "eval_cases": 1,
        "backups": 3}
### END SOLUTION

# %% check
assert left == residue(s, ALICE, ["Lisbon"]), residue(s, ALICE, ["Lisbon"])
print(f"✅ the naive delete left {sum(left.values())} copies on {sum(v > 0 for v in left.values())} surfaces")

# %% [markdown]
# ## In a design review
# **The two-minute version.** "We write raw episodes cheaply on the hot path and consolidate them per user and window
# on a schedule. The job is rules, not a summary: the highest-precedence source wins, newer supersedes older and the
# older fact is kept with a `valid_to`, and weaker contradictions are flagged for review — tool output is never read
# because it is quarantined on write. It runs as a durable job: a deterministic run id per window so a double fire is
# one run, a lease so a dead worker's run is picked up after the TTL, a checkpoint per slot, ids that make a re-applied
# slot overwrite. At a 60-token budget consolidated facts give 91% recall against 18% for raw episodes. Forgetting is
# decay, TTL, the per-turn budget and a cap per scope — and deletion is separate: a deletion key and provenance let us
# delete the record, its vector and full-text postings, everything derived from it and everything quoting it, evict
# the user's cached prefixes, redact logs and drop eval cases; backups expire or their key is shredded, and we test
# deletion by searching every surface for the data."
#
# **Drill questions**
# 1. *A user asked to forget their address; a week later the agent quoted it. Where was it?* — In a copy the deletion
#    did not reach: a consolidated fact or insight derived from it, the raw episode, a full-text index, a WAL file, a
#    cached prompt prefix, a log, an eval set. Propagate by provenance and by content, evict caches, and test by
#    searching the bytes.
# 2. *Why keep the superseded fact instead of deleting it?* — As-of questions ("where did they live in March?"), audit
#    ("what did the agent believe when it acted?"), and undoing a wrong update. Deletion is for data the user asked
#    you to remove, not for facts that changed.
# 3. *The consolidation job crashed halfway. What must be true for a rerun to be safe?* — Same run id, a lease that
#    expires, checkpoints per slot, and deterministic ids so a re-applied slot overwrites rather than duplicates.
