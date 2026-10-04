# %% [markdown]
# # 04 · Consolidation, forgetting and deletion
#
# **Tier:** T0. It uses only a CPU, it needs no network, and it takes a few seconds. The same job is in `memory-lab`
# notebook `04_consolidation_as_a_scheduled_job`. That notebook keeps a lease row and checkpoints on SQLite. It shows
# a crash and a resume, and it prints the GCP schedule (Cloud Run job on Cloud Scheduler). The `memory-lab` notebook
# `05_evaluate_forget_and_audit` does a check of a deletion on the bytes of the database file.
#
# ## The one-minute version
# Raw episodes are low-cost to write and high-cost to read. They are long and repetitive, and they are full of values
# that changed since then.
#
# **Consolidation** changes a window of episodes into facts. It runs on a schedule, and it uses rules, not a summary:
#
# - The source with the highest precedence wins (human > user > tool > inferred).
# - A newer fact replaces an older fact, **and the job keeps the older fact, closed** (`valid_to`, bi-temporal, as
#   Graphiti closes an edge with `invalid_at`).
# - The job flags a weaker contradiction for review.
#
# Here, "newer" means valid time, not arrival order. Thus a backfilled window never lets an old value win.
#
# Consolidation runs as a durable job, with these parts:
#
# - a deterministic run id.
# - a lease that the job renews before every slot and examines before every write.
# - a checkpoint for each slot.
# - ids that make a re-applied slot overwrite the facts, not duplicate them.
#
# **Forgetting** is four mechanisms: decay, TTL, the per-turn budget and a cap for each scope. None of them is
# **deletion**. To delete a fact, you must find every copy of it. The copies are:
#
# - the record.
# - the vector and the full-text postings of the record.
# - the facts derived from the record.
# - anything that quotes the fact.
# - cached prompt prefixes.
# - logs.
# - eval sets.
# - backups.
#
# The proof of a deletion is a search for the data after the deletion.
#
# Primer: §7 *Consolidation, forgetting and deletion* (`../PRIMER.md`).

# %%
from memcore import (DAY, Budget, Consolidator, Crash, LeaseHeld, LeaseLost, MemoryRecord, MemoryStore, PrefixCache,
                     Salts, Scope, Surfaces, Writer, build_store, cap, evaluate, expire, extract, generate, plan_key,
                     propagate, reflect, residue, retention, summarize, token_ids)

ALICE = Scope("acme", "alice")

# %% [markdown]
# ## Worked example 1 — the rules for one slot
# `plan_key` takes two inputs: the statements about one slot in the window, as `(time, value, source, evidence id)`,
# and a fact that is already on file. It returns the facts to write (value, valid from, valid to, source, evidence)
# and the flags.

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
# The two Porto statements are one fact with two pieces of evidence. The job closes Lisbon at day 4, and it does not
# delete it. The inferred Madrid is weaker than the statements of the user, thus the job only flags it.
#
# With a human-entered Oslo on file, the job applies none of the statements of the user. It flags all of them for a
# person to look at. Precedence is a product decision. When you make it explicit, you can test it.
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
# The write path put the Evilcorp episode of the tool in quarantine, thus consolidation never read it. The job
# promotes nothing that a tool said without a review. The run id is deterministic for each (user, window):
# `consolidate:acme:alice:day0-7`. This is the same way that lra-gcp primer §3.13 names a scheduled run
# (`weekly-review-2026-W37`). Thus, when the schedule starts the job two times, the result is one run.
#
# Windows do not always run in order. Two examples are a backfill and a re-run after an outage. The store already
# has a fact on file. That fact joins the statements of the window at its own `valid_from`. Thus an older statement
# becomes closed history in front of it:

# %%
store = MemoryStore()
w = Writer(store)
w.write(extract("I moved to Porto.", ALICE, at=5 * DAY)[0])
w.write(extract("I live in Lisbon.", ALICE, at=1 * DAY)[0])
job = Consolidator(store)
job.run(ALICE, 4 * DAY, 7 * DAY, now=7 * DAY)          # this week's window first
job.run(ALICE, 0, 4 * DAY, now=8 * DAY)                # then last week's, late
for r in store.records(ALICE, status=None, kind="semantic"):
    print(f"  {r.status:10} {r.render()}")

# %% [markdown]
# Porto stays current. The job keeps Lisbon, valid from day 1 until day 5. If the job applies the statements in their
# order of arrival, Lisbon becomes current again, and the job closes Porto *before* Porto starts. The job does not use
# that order.
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
random_ids = Consolidator(seeded_store(), lease_ttl=60, derived_ids=False)
try:
    random_ids.run(ALICE, 0, 7 * DAY, now=7 * DAY, worker="w1", crash_after=2)
except Crash:
    pass
random_ids.run(ALICE, 0, 7 * DAY, now=7 * DAY + 61, worker="w2")
print("the same crash with random fact ids:", len(random_ids.store.records(ALICE, kind="semantic", status=None)),
      "semantic records")

# %% [markdown]
# w1 stopped in `home_city` **after it wrote its facts and before it wrote the checkpoint for the slot**. This is the
# worst place for a crash. The stopped worker held the lease. Nobody else was able to run the job until the lease
# expired (lra-gcp primer §3.3: leases, not locks, and the work of the reaper).
#
# The resumed run skipped the one slot with a checkpoint, and it applied `home_city` again. The job derives the id of
# each fact from (run id, slot, position). Thus the re-applied facts overwrote the half-applied facts. The result was
# five facts, as in a clean run. With random ids, the same crash leaves seven. On GCP, this is a Cloud Run job that
# Cloud Scheduler starts (the lab's `deploy/gcp/`).
#
# A lease covers a worker that stopped. A **slow** worker is worse, for example a worker with a long GC pause or a model
# call that hangs for minutes. It is worse because it still runs when its lease expires and another worker takes over.
# Thus the job renews its lease before every slot (a heartbeat). Before it writes, it makes sure that it still holds the
# lease (a fence). If it does not hold the lease, it stops:

# %%
job = Consolidator(seeded_store(), lease_ttl=60)
def meanwhile(i, t):
    if i == 1:                                      # w1 is 90 s into a 60 s lease: w2 is scheduled and takes over
        print(f"t+{t - 7 * DAY:.0f} s, w2:", job.run(ALICE, 0, 7 * DAY, now=t, worker="w2").status)
try:
    job.run(ALICE, 0, 7 * DAY, now=7 * DAY, worker="w1", step_s=90, between_slots=meanwhile)
except LeaseLost as e:
    print("w1:", e)
print("semantic records:", len(job.store.records(ALICE, kind="semantic", status=None)))

# %% [markdown]
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
# Generative agents write *insights* when the sum of the importance of recent events goes past a trigger (150 in the
# reference code). Each insight cites its evidence. The job has explicit exits, like the reflection loop of lra-gcp
# primer §3.8: below the trigger, done, a maximum number of insights, and budget exhausted.

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
# Decay and the per-turn budget make a memory *unlikely to appear*. TTL and caps remove records on a schedule. None of
# them answers "forget my address". That request needs deletion with propagation.
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
    cache, salts = PrefixCache(4), Salts(b"server-secret")
    cache.serve(token_ids("system prompt + The user's home city is Lisbon."), salt=salts.salt(ALICE.tenant))
    return Surfaces(store, cache, salts, logs=["08:00 recall -> The user's home city is Lisbon.", "08:01 ok"],
                    eval_cases=[{"text": "Q: home city? A: Lisbon", "provenance": ()}],
                    backups=[{r.id: r.text for r in store.records(ALICE, status=None)}])


s = world()
print("before:", residue(s, ALICE, ["Lisbon"]))
rep = propagate(s, ALICE, key="home_city")
print(rep.checklist())

# %% [markdown]
# The deletion removed three records, not one:
#
# - the fact.
# - the episode that the fact came from (the episode also says "Lisbon").
# - an inferred insight that quotes the fact.
#
# The deletion keeps the employer fact, extracted from the same episode, because that fact does not contain the
# data. But the report lists it for **review**, because it derives from a deleted record. The deletion redacts the
# log line and drops the eval case.
#
# Two surfaces are *pending*, not done. The backup still holds three copies. The deletion does not write the backups
# again. Backups expire on their retention schedule, or somebody shreds their encryption key.
#
# The prompt cache is the second pending surface. Nobody can edit a cached prefix. Also, vLLM cannot evict the blocks
# of one tenant. Its only tool, the dev-mode `/reset_prefix_cache`, clears the blocks of every tenant. Thus the
# deletion **rotated** the `cache_salt` of the tenant. No request can hit the two blocks again, and they leave GPU
# memory when LRU eviction uses them again.
#
# When you answer a deletion request, say how long each pending surface takes, with a date.
#
# The deletion matches values on word boundaries (`memcore.forget.mentions`). Thus, when you forget a pet "cat", the
# deletion never deletes a record about "education" or "Catalyst". But a content match finds only verbatim copies. A
# paraphrase goes past it. The review list is for that case.

# %%
store = MemoryStore()
w = Writer(store)
episode, fact = extract("I moved to Porto.", ALICE, at=0)
w.write(episode), w.write(fact)
para = store.put(MemoryRecord("The user lives in Portugal's second city.", "semantic", ALICE, "inferred",
                              created_at=DAY, provenance=(episode.id,)))
rep2 = propagate(Surfaces(store), ALICE, key="home_city")
print("deleted:", len(rep2.ids), "records | review:", [store.get(ALICE, i).text for i in rep2.review])

# %% [markdown]
# The insight paraphrases the fact and cites the deleted episode. No search for "Porto" finds it, but provenance finds
# it.
#
# ## Exercise 4.1 — the consolidation rules
# Implement `plan(statements)` for statements `(time, value, source)` about one slot. Use these rules:
#
# 1. Keep only the statements from the source with the highest precedence that is present.
# 2. Go through these statements in time order.
# 3. If a value is equal to the previous value (case-insensitive), extend the previous fact.
# 4. If a value is different, close the previous fact at the time of the new value. Then open a new fact.
#
# Return `[(value, valid_from, valid_to_or_None)]`. Also return the list of weaker statements whose value is different
# from the value that was valid at their time.

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
# ## Exercise 4.2 — the lease rules
# `holder` is `None` or `(worker_id, expires_at)`. Write `can_acquire(holder, worker, now)`. A worker can take the
# lease in three cases:
#
# - Nobody holds the lease.
# - The worker already holds the lease (a heartbeat renews it).
# - The lease of the holder is expired.
#
# Then write `may_continue(holder, worker)`. This is the fence that a worker examines during its run, before the
# writes of each slot. The worker can continue only if the store still records the lease as its own. An expired
# lease still counts as its own, if nobody took the lease. The worker must stop if nobody holds the lease,
# because another worker finished the run and released it. It must also stop if another worker holds the lease.

# %% exercise
def can_acquire(holder, worker, now):
    ### BEGIN SOLUTION
    return holder is None or holder[0] == worker or holder[1] <= now
    ### END SOLUTION

def may_continue(holder, worker):
    ### BEGIN SOLUTION
    return holder is not None and holder[0] == worker
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
for holder, worker in [(None, "w1"), (("w1", 60), "w1"), (("w1", 10), "w1"), (("w2", 200), "w1")]:
    job = Consolidator(MemoryStore())
    job.checkpoints["r"] = {"done": [], "status": "running"}
    if holder:
        job.leases["r"] = holder
    try:
        job.heartbeat("r", worker, 100)
        real = True
    except LeaseLost:
        real = False
    assert may_continue(holder, worker) == real, (holder, worker)
print("✅ a lease expires - that lets a dead worker's job be picked up; the fence stops a slow one that lost it")

# %% [markdown]
# ## Exercise 4.3 — predict the retention
#
# $$
# \text{retention} = \frac{\text{importance}}{10} \times 0.5^{\text{days since last use} / 30}.
# $$
#
# Predict the retention of an importance-8 memory that the agent last used 45 days ago, to three decimals. Then
# predict how many days an importance-4 memory takes to decrease below 0.1.

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
# Implement `derived(records, targets)`. It returns the ids of `targets` and every record whose `provenance` cites an
# id already in the set. Add those records to the set again and again, until the set does not change (facts derived
# from facts derived from the target).

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
# Delete the home-city fact in the simple way, on a new `world()`. Use `store.delete` on the records whose `key` is
# `home_city`, and do nothing else. Then fill `left` with the number of copies of "Lisbon" that each surface still
# holds. Count them in the same way as `residue`. Predict the numbers first. Then do a check with `residue`.

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
# **The two-minute version.** "We write raw episodes at low cost on the hot path. We run consolidation on them for each
# user and window, on a schedule.
#
# "The job is rules, not a summary. The source with the highest precedence wins. A newer fact replaces an older fact in
# valid time, thus a backfill never undoes a later change. The job keeps the older fact with a `valid_to`, and it flags
# weaker contradictions for review. The job never reads tool output, because the write path puts it in quarantine.
#
# "It runs as a durable job. A deterministic run id for each window makes two starts of the same window one run. A
# lease lets another worker take the run of a stopped worker after the TTL. A heartbeat and a fence stop a slow worker
# that lost its lease. The job writes a checkpoint for each slot, and its ids make a re-applied slot overwrite. At a
# 60-token budget, consolidated facts give 91% recall against 18% for raw episodes.
#
# "Forgetting is decay, TTL, the per-turn budget and a cap for each scope. Deletion is separate. With a deletion key
# and provenance, we delete the record, its vector and its full-text postings. We also delete everything derived from
# it, and everything that quotes it on word boundaries. We list for review what derives from a deleted record but does
# not quote it. We rotate the cache salt of the tenant, redact logs and drop eval cases.
#
# "Backups and unreachable cache blocks expire on a stated schedule. Our test of a deletion is a search of every
# surface for the data."
#
# **Drill questions**
# 1. *A user asked the agent to forget their address. A week later, the agent quoted it. Where was it?* It was in a
#    copy that the deletion did not reach. The possible copies are:
#
#     - a consolidated fact or insight derived from the address.
#     - the raw episode.
#     - a full-text index.
#     - a WAL file.
#     - a cached prompt prefix.
#     - a log.
#     - an eval set.
#     - a paraphrase that no content search finds.
#
#    Propagate the deletion by provenance and by content. Rotate the cache salt. Then do a test that searches the
#    bytes.
# 2. *Why keep the superseded fact, and not delete it?* There are three reasons. The first is as-of questions ("where
#    did they live in March?"). The second is audit ("what did the agent believe when it acted?"). The third is the
#    undo of an incorrect update. Deletion is for data that the user asked you to remove, not for facts that changed.
# 3. *The consolidation job crashed halfway. What must be true for a rerun to be safe?* The rerun must have the same
#    run id, a lease that expires, checkpoints for each slot, and deterministic ids. With these ids, a re-applied slot
#    overwrites the facts and does not duplicate them.
