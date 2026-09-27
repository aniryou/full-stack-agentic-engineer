# %% [markdown]
# # 04 · Consolidation as a scheduled job: episodes become facts, and the job survives being killed halfway
#
# **Tier:** T0 — the job runs in this process on SQLite with a virtual clock; a crash is simulated at a chosen
# step and a second worker resumes it. **T3 (printed):** the same job as a Cloud Run job started by Cloud
# Scheduler — the commands are printed, never run from here (`python -m memlab gcp-commands`).
#
# ## The one-minute version
#
# Consolidation turns a user's window of raw episodes ("On 2026-09-18 the user said: I moved to Porto") into
# facts ("Home city: the user lives in Porto", superseding Lisbon, kept with a closed validity) — PRIMER §7
# "Consolidation, forgetting and deletion". It is a batch job that calls a model, so it must be a **durable
# run** (lra-gcp primer §3.3 "Leases and the reaper", §3.13 "Scheduled and event-triggered runs"):
#
# * a **deterministic run id** — `consolidate/<tenant>/<user>/<ISO week>` — so a duplicated or retried
#   trigger is the same run, and a finished run is a no-op; the schedule is weekly to match (a daily trigger with a
#   weekly id would do its work on one day in seven and do nothing on the other six);
# * a **lease** row so two workers never run it at once; a crashed worker's lease expires and the next takes
#   over;
# * a **checkpoint** per step, so a resume skips finished work — above all the model calls;
# * **idempotent effects**, so the step that died after writing and before checkpointing writes once when it
#   runs again (durable primer §3.2).
#
# Then the facts are resolved (newer supersedes older; precedence human > user > tool > inferred; a weaker
# contradiction is flagged), consolidated episodes get a TTL (forgetting by decay), and — when the window's
# importance sum crosses a threshold (150 in the generative-agents code) — one insight citing its evidence.
# Primer: [`../../PRIMER.md`](../../PRIMER.md).

# %%
import datetime as dt, os, sqlite3, tempfile
from memlab import consolidate as C
from memlab import harness as H
from memlab.consolidate import ConsolidationJob, LeaseHeld, SimulatedCrash
from memlab.extract import ts_of
from memlab.memory import resolve
from memlab.records import MemoryRecord
from memlab.store import SQLiteMemoryStore

class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t

WEEK = [("2026-09-14", "I live in Lisbon and my manager is Ana."), ("2026-09-15", "I work at Globex."),
        ("2026-09-16", "My dog is called Rex."), ("2026-09-17", "I took a trip to Rome."),
        ("2026-09-18", "I moved to Porto."), ("2026-09-19", "I started at Initech."),
        ("2026-09-20", "I took a trip to Kyoto."), ("2026-09-20", "I think I am allergic to peanuts.")]

def week_store(path):
    clock = Clock(ts_of("2026-09-21"))
    store = SQLiteMemoryStore(path, clock=clock)
    for d, text in WEEK:
        store.add(MemoryRecord("acme", "u1", f"On {d} the user said: {text}", kind="episodic", importance=25,
                               created_at=ts_of(d)))
    return store, clock

WORK = tempfile.mkdtemp(prefix="memlab-nb04-")
import atexit, shutil
atexit.register(shutil.rmtree, WORK, True)   # removed when the kernel exits, even if a cell stops early
START, END = ts_of("2026-09-14"), ts_of("2026-09-21")

# %% [markdown]
# ## Worked example: a clean run
#
# Eight episodes, two chunks of four (one model call each), a plan of actions, one step per action.

# %%
clean, clean_clock = week_store(os.path.join(WORK, "clean.db"))
rep = ConsolidationJob(clean, worker="A").run("acme", "u1", START, END)
print(rep.line())
print("steps:", " ".join(rep.steps_run), "| checkpoints left after the run:", clean.job_tables().steps(rep.run_id))
for r in clean.records("acme", "u1", kind="semantic"):
    print(f"  {r.status:10s} {r.text:78s} valid_to={'-' if r.valid_to is None else dt.date.fromtimestamp(r.valid_to)}")

# %% [markdown]
# Lisbon and Globex are **superseded, not deleted**: their validity closes when Porto and Initech begin
# (Graphiti keeps the same pair as `invalid_at` / `expired_at`), so "where did the user live on 15
# September?" still has an answer (`store.search(..., as_of=...)`). The hedged allergy ("I think…", confidence
# 0.6) was skipped: below the write policy's floor.
#
# ## Worked example: kill it at the worst moment, then resume
#
# Worker A dies in `apply:2` — *after* the step's write, *before* its checkpoint. Worker B is scheduled
# right away and finds the lease held; a minute later the lease has expired and B takes over.

# %%
store, clock = week_store(os.path.join(WORK, "crash.db"))
try:
    ConsolidationJob(store, worker="A", crash_at="apply:2", lease_ttl_s=60).run("acme", "u1", START, END)
except SimulatedCrash as e:
    print("A:", e)
try:
    ConsolidationJob(store, worker="B", lease_ttl_s=60).run("acme", "u1", START, END)
except LeaseHeld as e:
    print("B, at once:", e)
clock.t += 61
resumed = ConsolidationJob(store, worker="B", lease_ttl_s=60).run("acme", "u1", START, END)
print("B, after the lease expired:", resumed.line())
print("   skipped:", resumed.steps_skipped)
snap = lambda s: sorted((r.status, r.text, r.valid_to) for r in s.records("acme", "u1", kind="semantic"))
assert snap(store) == snap(clean)
print("same facts as the clean run:", snap(store) == snap(clean), "| a third trigger:",
      ConsolidationJob(store, worker="C").run("acme", "u1", START, END).line())

# %% [markdown]
# ## Exercise 4.1 — one run per user and week
#
# Write `my_run_id(tenant, user, window_end)`: `consolidate/<tenant>/<user>/<YYYY>-W<ww>` from the ISO calendar
# week of `window_end` (a unix time, UTC), the week zero-padded to two digits. A trigger that fires twice, or
# a retry an hour later, must produce the same id.

# %% exercise
def my_run_id(tenant, user, window_end):
    ### BEGIN SOLUTION
    y, w, _ = dt.datetime.fromtimestamp(window_end, dt.timezone.utc).isocalendar()
    return f"consolidate/{tenant}/{user}/{y}-W{w:02d}"
    ### END SOLUTION

# %% check
assert my_run_id("acme", "u1", END) == C.run_id("acme", "u1", END) == "consolidate/acme/u1/2026-W39"
assert my_run_id("acme", "u1", END + 3600) == my_run_id("acme", "u1", END)
assert my_run_id("acme", "u1", ts_of("2027-01-01")) == "consolidate/acme/u1/2026-W53"   # ISO years differ from calendar years
print("✅ the run id is a function of the work, not of the trigger:", my_run_id("acme", "u1", END))

# %% [markdown]
# The trigger has to fire as often as the id changes. `week_window(now)` gives the window a trigger at `now`
# consolidates — the previous ISO week, Monday 00:00 to Monday 00:00 UTC — so every firing in one week names the
# same run. If the schedule were daily, six of seven firings would find that run `done` and return:

# %%
fires = [ts_of("2026-09-21") + d * 86400 + 3 * 3600 + 17 * 60 for d in range(7)]       # 03:17 each day
ids = [C.run_id("acme", "u1", C.week_window(t)[1]) for t in fires]
print("a daily trigger's run ids this week:", sorted(set(ids)), "-> one run, six no-ops")
print("so the schedule is weekly:", next(c for c in C.gcp_commands() if "scheduler jobs create" in c).split("--schedule ")[1][:13])

# %% [markdown]
# ## Exercise 4.2 — resolve a new fact against what memory holds
#
# Write `my_resolve(existing, new)` returning `(action, other)` for a new semantic fact against the
# partition's records (only `active` ones of the same `slot` count):
#
# | case | action | `other` |
# |---|---|---|
# | same value already there (case-insensitive) | `"NOOP"` | that record |
# | nothing for the slot, or a multi-valued slot (`trip`) | `"ADD"` | `None` |
# | the current value's trust outranks the new one's (human > user > tool > inferred) | `"FLAG"` | the current record |
# | the new fact is *older* than the current one (a late episode) | `"ADD_HISTORY"` | the current record |
# | otherwise | `"UPDATE"` (the current record will be superseded) | the current record |
#
# "Current" is the active record of the slot with the latest `valid_from`. Use `records.outranks(a, b)` and
# `extract.SLOTS[slot].multi`.

# %% exercise
from memlab.records import outranks
from memlab.extract import SLOTS

def my_resolve(existing, new):
    ### BEGIN SOLUTION
    same = [r for r in existing if r.slot == new.slot and r.status == "active" and r.slot]
    for r in same:
        if (r.value or "").lower() == (new.value or "").lower():
            return "NOOP", r
    if not same or (new.slot in SLOTS and SLOTS[new.slot].multi):
        return "ADD", None
    current = max(same, key=lambda r: r.valid_from)
    if outranks(current.trust, new.trust):
        return "FLAG", current
    if new.valid_from < current.valid_from:
        return "ADD_HISTORY", current
    return "UPDATE", current
    ### END SOLUTION

# %% check
def fact(slot, value, t, trust="user", status="active"):
    return MemoryRecord("acme", "u1", f"{slot}={value}", slot=slot, value=value, trust=trust, valid_from=t,
                        created_at=t, status=status)
lisbon = fact("home_city", "Lisbon", 100)
cases = [([lisbon], fact("home_city", "lisbon", 200), "NOOP"), ([], fact("home_city", "Porto", 200), "ADD"),
         ([lisbon], fact("home_city", "Porto", 200), "UPDATE"), ([lisbon], fact("home_city", "Porto", 50), "ADD_HISTORY"),
         ([fact("home_city", "Lisbon", 100, "human")], fact("home_city", "Porto", 200), "FLAG"),
         ([lisbon], fact("home_city", "Porto", 200, "tool"), "FLAG"), ([fact("trip", "Rome", 1)], fact("trip", "Kyoto", 2), "ADD"),
         ([fact("home_city", "Lisbon", 100, status="superseded")], fact("home_city", "Porto", 200), "ADD")]
for existing, new, want in cases:
    got = my_resolve(existing, new)
    assert got[0] == want == resolve(existing, new)[0], (new.value, new.trust, want, got)
print("✅ my_resolve agrees with memory.resolve on", len(cases), "cases, including a flagged tool-sourced contradiction")

# %% [markdown]
# ## Exercise 4.3 — take the lease in one statement
#
# Write `acquire(con, run, holder, now, ttl)` for a table `job_leases(run_id PRIMARY KEY, holder, expires_at,
# attempts)`: in **one** SQL statement, insert the lease, or take it over if it has expired, or renew it if
# `holder` already has it — and do nothing if another holder's lease is still live. Return True when `holder`
# holds it afterwards. (SQLite's `INSERT … ON CONFLICT … DO UPDATE … WHERE`; Postgres has the same, see
# `store/pgvector.py` `ACQUIRE_LEASE`.) Two statements — a SELECT then an UPDATE — would race.

# %% exercise
def acquire(con, run, holder, now, ttl):
    ### BEGIN SOLUTION
    con.execute("INSERT INTO job_leases (run_id, holder, expires_at, attempts) VALUES (?, ?, ?, 1) "
                "ON CONFLICT (run_id) DO UPDATE SET holder = excluded.holder, expires_at = excluded.expires_at, "
                "attempts = job_leases.attempts + (job_leases.holder != excluded.holder) "
                "WHERE job_leases.expires_at <= ? OR job_leases.holder = excluded.holder",
                (run, holder, now + ttl, now))
    return con.execute("SELECT holder FROM job_leases WHERE run_id = ?", (run,)).fetchone()[0] == holder
    ### END SOLUTION

# %% check
con = sqlite3.connect(":memory:")
con.execute("CREATE TABLE job_leases (run_id TEXT PRIMARY KEY, holder TEXT NOT NULL, expires_at REAL NOT NULL, "
            "attempts INTEGER NOT NULL DEFAULT 1)")
assert acquire(con, "r1", "A", 0, 60)            # free
assert not acquire(con, "r1", "B", 30, 60)       # A's lease is live
assert acquire(con, "r1", "A", 50, 60)           # A renews (heartbeat): now expires at 110
assert not acquire(con, "r1", "B", 100, 60)      # still A's
assert acquire(con, "r1", "B", 111, 60)          # expired: B takes over
assert con.execute("SELECT attempts FROM job_leases").fetchone()[0] == 2
print("✅ one statement: take, renew, refuse, take over — attempts counts the takeovers")

# %% [markdown]
# ## Exercise 4.4 — what a resume costs
#
# The job makes one model call per chunk of episodes (`extract:0`, `extract:1`, …). Write
# `calls_on_resume(n_chunks, crash_at)`: how many model calls the *resumed* run makes when the first worker
# died in step `crash_at` (after its effect, before its checkpoint). Steps before the crash are checkpointed;
# the crashed step runs again.

# %% exercise
def calls_on_resume(n_chunks, crash_at):
    ### BEGIN SOLUTION
    if crash_at == "load":
        return n_chunks
    if crash_at.startswith("extract:"):
        return n_chunks - int(crash_at.split(":")[1])
    return 0
    ### END SOLUTION

# %% check
for crash in ("load", "extract:0", "extract:1", "resolve", "apply:3", "mark"):
    s, c = week_store(os.path.join(WORK, f"c-{crash.replace(':', '-')}.db"))
    try:
        ConsolidationJob(s, worker="A", crash_at=crash).run("acme", "u1", START, END)
    except SimulatedCrash:
        pass
    c.t += 61
    got = ConsolidationJob(s, worker="B").run("acme", "u1", START, END).model_calls
    assert got == calls_on_resume(2, crash), (crash, got)
    assert snap(s) == snap(clean), crash
print("✅ checkpoints make a resume pay only for the step that died; every resumed run ends in the clean run's state")

# %% [markdown]
# ## Worked example: does consolidation help recall at a fixed budget?
#
# The planted-facts harness (notebook 05) with memory written as raw episodes, then consolidated, at a
# budget of 64 memory tokens per turn. Reading the four rows: facts beat raw episodes because a fact is
# shorter and names its slot; raw episodes in the user's own words sometimes match a first-person question
# better (the paraphrase column); and keeping raw episodes retrievable next to the facts lets an **old
# episode** resurface a superseded value — the `stale` column.

# %%
ds = H.generate(7)
rows = {"raw episodes": H.run(ds, "implicit", write="episodes", budget_tokens=64),
        "episodes + consolidation": H.run(ds, "implicit", write="episodes", consolidate=True, budget_tokens=64),
        "  ... answer from facts only": H.run(ds, "implicit", write="episodes", consolidate=True, budget_tokens=64,
                                              kinds=("semantic",)),
        "facts written at the turn": H.run(ds, "implicit", budget_tokens=64)}
print(H.summary(rows))
stale = [r for r in rows["episodes + consolidation"].rows if r.verdict == "stale"]
print("the stale answer:", f"{stale[0].qid} said {stale[0].answer!r}" if stale else "none this time")

# %% [markdown]
# ## Worked example: forgetting by decay
#
# The job gave every consolidated episode a TTL of 30 days from consolidation. Thirty-one days later the
# episodes expire and are purged; the facts they produced remain, each citing its (now deleted) evidence id.

# %%
clean_clock.t = END + 31 * 86400
n = clean.expire()
print(f"expired {n} episodes; left:", {k: sum(1 for r in clean.records('acme', 'u1') if r.kind == k)
                                       for k in ("episodic", "semantic")})

# %% [markdown]
# ## T3, printed: a Cloud Run job on Cloud Scheduler
#
# The job is `python -m memlab consolidate` in the lab's image — by default over the previous ISO week — started
# every Monday at 03:17 UTC; Cloud Run jobs start `--tasks N` copies and each takes the partitions whose hash lands
# on its `CLOUD_RUN_TASK_INDEX` (verify). Cloud Scheduler calls the Cloud Run Admin API's `jobs/<job>:run` with an
# **OAuth** token (the target is a `googleapis.com` API; OIDC is for your own `run.app` service, like the lra-gcp
# reaper). On GCP the store is Postgres + pgvector (Cloud SQL, verify), never a SQLite file on an instance's
# ephemeral disk. The printed path is complete enough to work: the image is built and pushed, the job runs as its
# own service account that may read the DSN secret and connect to Cloud SQL (`--set-cloudsql-instances`), and the
# cleanup deletes everything the setup created (`deploy/gcp/README.md`).

# %%
for c in C.gcp_commands("my-project", "us-central1", tasks=4):
    print(c)
print("# cleanup")
print("\n".join(C.cleanup_commands("my-project", "us-central1")))
print("\nshard of 10 users across 4 tasks:",
      [len(C.shard([("acme", f"u{i}") for i in range(10)], k, 4)) for k in range(4)])
import shutil
shutil.rmtree(WORK, ignore_errors=True)    # this notebook's databases: gone

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "Consolidation runs weekly as a Cloud Run job per shard of users. Each (tenant, user, ISO week)
# is one durable run with a deterministic id, and the schedule fires once a week to match, so a double-fired
# schedule is a no-op and no firing is wasted; facts a user states mid-week reach memory at once through
# extraction after the turn, the job only distils the week's episodes. A lease row keeps two workers apart and
# expires when one dies; every step checkpoints, so a resume skips the model calls it already paid for; writes
# carry idempotency keys derived from the run, so the step that died after writing does not write twice.
#
# "Resolution is deterministic: newer supersedes older with the old validity closed, a weaker source never
# overwrites a stronger one — it is flagged — and low-confidence extractions are skipped. Consolidated episodes
# get a 30-day TTL. We measured recall at a fixed budget before and after, and chose to answer from facts once an
# episode is consolidated, because a raw episode can resurface a superseded value."
#
# **Drill 1.** *The scheduler fired twice at 03:17. What happens?* — Both triggers compute the same run id;
# one takes the lease, the other gets `LeaseHeld` (or, if the first finished, sees the run `done`) and exits.
#
# **Drill 1b.** *Someone changed the schedule to daily. What breaks?* — Nothing visibly: the first firing of the
# week does the work and the other six find the run `done`. That is the bug — a "daily" job that runs weekly. The
# run id and the window must change with the schedule (a per-day id and a one-day window).
#
# **Drill 2.** *A worker died after writing a fact but before checkpointing. Duplicate?* — No: the resumed step
# writes with the same idempotency key (run, slot, value, evidence) and the planned record id, so the store
# returns the existing row; the supersede is applied only if the old fact is still active.
#
# **Drill 3.** *Why OAuth, not OIDC, for Cloud Scheduler here?* — The target is the Cloud Run Admin API
# (`run.googleapis.com`), which takes OAuth access tokens; OIDC identity tokens are for invoking your own
# service's URL.
