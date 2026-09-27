"""Consolidation (rules, durability) and forgetting (decay, TTL, caps, deletion propagation)."""
import pytest

from memcore import (DAY, Budget, Consolidator, Crash, LeaseHeld, LeaseLost, MemoryRecord, MemoryStore, PrefixCache,
                     Salts, Scope, Surfaces, Writer, cap, expire, extract, plan_key, propagate, reflect, residue,
                     retention, token_ids)

U = Scope("acme", "alice")


def test_plan_newer_supersedes_older_and_keeps_it():
    facts, flags = plan_key([(1 * DAY, "Lisbon", "user", "e1"), (4 * DAY, "Porto", "user", "e2"),
                             (5 * DAY, "porto", "user", "e3")])
    assert [(f[0], f[1], f[2], f[4]) for f in facts] == [("Lisbon", DAY, 4 * DAY, ("e1",)),
                                                          ("Porto", 4 * DAY, None, ("e2", "e3"))]
    assert flags == []


def test_plan_precedence_and_weaker_contradiction_flagged():
    facts, flags = plan_key([(1 * DAY, "Lisbon", "human", "e1"), (4 * DAY, "Porto", "user", "e2")])
    assert [f[0] for f in facts] == ["Lisbon"] and flags[0][0][1] == "Porto"
    existing = MemoryRecord("The user's home city is Oslo.", "semantic", U, "human", key="home_city", value="Oslo")
    facts, flags = plan_key([(2 * DAY, "Porto", "user", "e3")], existing)
    assert facts == [] and "weaker than the human fact" in flags[0][1]


def seeded_store():
    store, w = MemoryStore(), Writer(MemoryStore())
    w = Writer(store)
    for day, text in [(0, "I live in Lisbon."), (1, "I prefer window seats."), (2, "I work at Acme."),
                      (3, "I moved to Porto."), (3.5, "My pet is a cat.")]:
        w.write(extract(text, U, at=day * DAY)[0])                      # episodes only: the background path
    w.write(extract("I work at Evilcorp.", U, source="tool", at=2.5 * DAY)[0])   # quarantined: never consolidated
    return store


def test_consolidation_run_writes_facts_and_closes_the_old_one():
    store = seeded_store()
    rep = Consolidator(store).run(U, 0, 7 * DAY, now=7 * DAY)
    active = {r.key: r.value for r in store.records(U, kind="semantic")}
    assert active == {"home_city": "Porto", "seat_preference": "window", "employer": "Acme", "pet": "cat"}
    closed = store.records(U, status="superseded")
    assert [(r.value, r.valid_from, r.valid_to) for r in closed] == [("Lisbon", 0, 3 * DAY)]
    assert len(rep.written) == 5 and rep.status == "done" and rep.run_id == "consolidate:acme:alice:day0-7"


def test_crash_lease_and_resume_write_each_fact_once():
    store = seeded_store()
    job = Consolidator(store, lease_ttl=60)
    with pytest.raises(Crash):                                           # dies in home_city: written, not checkpointed
        job.run(U, 0, 7 * DAY, now=7 * DAY, worker="w1", crash_after=2)
    assert job.checkpoints[job.run_id(U, 0, 7 * DAY)]["done"] == ["employer"]
    assert len(store.records(U, kind="semantic", status=None)) == 3      # employer + Lisbon (closed) + Porto
    with pytest.raises(LeaseHeld):                                       # w1 died holding the lease
        job.run(U, 0, 7 * DAY, now=7 * DAY + 30, worker="w2")
    rep = job.run(U, 0, 7 * DAY, now=7 * DAY + 61, worker="w2")          # the lease expired: w2 picks it up
    assert rep.skipped == ["employer"] and rep.status == "done"          # home_city is re-applied...
    assert len(store.records(U, kind="semantic", status=None)) == 5      # ...and overwrites: same as a clean run
    assert job.run(U, 0, 7 * DAY, now=8 * DAY).status == "already done"  # a double fire of the schedule


def test_random_fact_ids_would_duplicate_the_reapplied_slot():
    store = seeded_store()
    job = Consolidator(store, lease_ttl=60, derived_ids=False)
    with pytest.raises(Crash):
        job.run(U, 0, 7 * DAY, now=7 * DAY, worker="w1", crash_after=2)
    job.run(U, 0, 7 * DAY, now=7 * DAY + 61, worker="w2")
    assert len(store.records(U, kind="semantic", status=None)) == 7      # Lisbon and Porto twice


def test_a_slow_worker_that_lost_its_lease_stops_before_writing():
    store = seeded_store()
    job = Consolidator(store, lease_ttl=60)
    seen = []

    def meanwhile(i, t):                                                 # w1 stalls 90 s per slot (a GC pause, a
        if i == 1:                                                       # slow model call); at slot 1 its lease has
            seen.append(job.run(U, 0, 7 * DAY, now=t, worker="w2").status)   # expired and w2 takes the run
    with pytest.raises(LeaseLost):
        job.run(U, 0, 7 * DAY, now=7 * DAY, worker="w1", step_s=90, between_slots=meanwhile)
    assert seen == ["done"] and len(store.records(U, kind="semantic", status=None)) == 5
    fast = Consolidator(seeded_store(), lease_ttl=60)                    # nobody took it: the heartbeat renews it
    assert fast.run(U, 0, 7 * DAY, now=7 * DAY, worker="w1", step_s=90).status == "done"


def test_windows_consolidated_out_of_order_never_let_an_older_value_win():
    store, w = MemoryStore(), None
    w = Writer(store)
    w.write(extract("I moved to Porto.", U, at=5 * DAY)[0])
    w.write(extract("I live in Lisbon.", U, at=1 * DAY)[0])
    job = Consolidator(store)
    job.run(U, 4 * DAY, 7 * DAY, now=7 * DAY)                            # this week first...
    job.run(U, 0, 4 * DAY, now=8 * DAY)                                  # ...then a backfill of last week
    by = {r.value: r for r in store.records(U, kind="semantic", status=None)}
    assert (by["Porto"].status, by["Porto"].valid_to) == ("active", None)
    assert (by["Lisbon"].status, by["Lisbon"].valid_from, by["Lisbon"].valid_to) == ("superseded", 1 * DAY, 5 * DAY)
    facts, _ = plan_key([(1 * DAY, "Lisbon", "user", "e1")], by["Porto"])
    assert [(f[0], f[2]) for f in facts] == [("Lisbon", 5 * DAY), ("Porto", None)]


def test_reflection_trigger_and_exits():
    store, w = MemoryStore(), None
    w = Writer(store)
    for i in range(12):
        w.write(extract(f"Note {i}: I live in Lisbon.", U, at=i * DAY)[0])      # importance 6 each -> 72
    assert reflect(store, U, now=12 * DAY) == ([], "below_trigger")
    insights, why = reflect(store, U, now=12 * DAY, threshold=60)
    assert why == "done" and len(insights) == 1 and len(insights[0].provenance) == 12
    assert reflect(store, U, now=12 * DAY, threshold=60, max_insights=0)[1] == "max_insights"
    tight = Budget({"memory_tokens": 0, "writes": 0, "llm_calls": 0, "usd": 0})
    assert reflect(store, U, now=12 * DAY, threshold=60, budget=tight)[1] == "budget_exhausted"


def test_retention_decay_ttl_and_cap():
    r = MemoryRecord("x", "episodic", U, "user", importance=9, created_at=0)
    assert round(retention(r, now=60 * DAY), 4) == 0.225                 # 0.9 x 0.5 ** (60 / 30)
    store = MemoryStore()
    for i in range(5):
        store.put(MemoryRecord(f"event {i}", "episodic", U, "user", importance=i + 1, created_at=0,
                               ttl_s=DAY if i == 0 else None))
    assert len(expire(store, U, now=2 * DAY)) == 1
    gone = cap(store, U, 2, now=2 * DAY)
    assert len(gone) == 2 and sorted(r.importance for r in store.records(U)) == [4, 5]


def world():
    store = MemoryStore()
    w = Writer(store)
    for day, text in [(0, "I live in Lisbon and I work at Acme."), (2, "I prefer window seats.")]:
        for rec in extract(text, U, at=day * DAY):
            w.write(rec)
    insight = store.put(MemoryRecord("The user cycles to work in Lisbon.", "semantic", U, "inferred", created_at=3 * DAY))
    cache, salts = PrefixCache(4), Salts(b"server-secret")
    cache.serve(token_ids("system prompt + The user's home city is Lisbon."), salt=salts.salt(U.tenant))
    return insight, Surfaces(store, cache, salts, logs=["08:00 recall -> The user's home city is Lisbon.", "08:01 ok"],
                             eval_cases=[{"text": "Q: home city? A: Lisbon", "provenance": ()}],
                             backups=[{r.id: r.text for r in store.records(U, status=None)}])


def test_propagate_reaches_every_copy_and_reports_what_is_pending():
    insight, s = world()
    employer = s.store.find(U, "employer")[0]
    rep = propagate(s, U, key="home_city")
    assert rep.removed == {"records": 3, "vectors": 3, "fulltext_postings": 24, "derived_facts": 0,
                           "prompt_cache_blocks": 2, "logs": 1, "eval_cases": 1, "backups": 0}
    assert insight.id in rep.ids                                         # it quotes the value: it goes too
    assert {k: v for k, v in rep.residue.items() if v} == {"backups": 3, "prompt_cache_blocks": 2}
    assert s.cache.lookup(token_ids("system prompt + The user's home city is Lisbon."),
                          salt=s.salts.salt(U.tenant)) == 0              # unreachable under the rotated salt
    assert [r.key for r in s.store.records(U, kind="semantic")] == ["employer", "seat_preference"]
    assert rep.review == [employer.id]                                   # derived from the deleted episode: check it
    text = rep.checklist()
    assert text.count("pending") == 2 and "LEFT" not in text and s.logs[0].startswith("[redacted")


def test_values_match_on_word_boundaries_not_substrings():
    store = MemoryStore()
    w = Writer(store)
    for day, text in [(0, "My pet is a cat."), (1, "I work in education and communications."), (2, "I work at Catalyst.")]:
        for rec in extract(text, U, at=day * DAY):
            w.write(rec)
    rep = propagate(Surfaces(store), U, key="pet")
    left = sorted(r.text for r in store.records(U, status=None))
    assert rep.removed["records"] == 2 and left == ["The user's employer is Catalyst.",
                                                    "User said: I work at Catalyst.",
                                                    "User said: I work in education and communications."]


def test_a_paraphrase_is_found_by_provenance_not_by_content():
    store = MemoryStore()
    w = Writer(store)
    episode, fact = extract("I moved to Porto.", U, at=0)
    w.write(episode), w.write(fact)
    para = store.put(MemoryRecord("The user lives in Portugal's second city.", "semantic", U, "inferred",
                                  created_at=DAY, provenance=(episode.id,)))
    s = Surfaces(store)
    rep = propagate(s, U, key="home_city")
    assert episode.id in rep.ids and para.id not in rep.ids              # content matching cannot see it...
    assert rep.review == [para.id] and residue(s, U, ["Porto"])["records"] == 0   # ...residue() cannot either
