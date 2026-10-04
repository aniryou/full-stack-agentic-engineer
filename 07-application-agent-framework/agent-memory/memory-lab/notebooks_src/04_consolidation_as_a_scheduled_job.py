# %% [markdown]
# # 04 · Consolidation as a scheduled job: episodes become facts, and the job survives being killed halfway
#
# **Tier:** T0. The job runs in this process on SQLite with a virtual clock. The notebook simulates a crash at a
# selected step, and a second worker resumes the job. **T3 (printed):** the same job as a Cloud Run job that
# Cloud Scheduler starts. The notebook prints the commands and never runs them from here
# (`python -m memlab gcp-commands`).
#
# ## The one-minute version
#
# Consolidation takes the raw episodes of a user in one window, for example "On 2026-09-18 the user said: I
# moved to Porto". It changes them into facts, for example "Home city: the user lives in Porto". The new fact
# supersedes Lisbon, and the store keeps Lisbon with a closed validity (PRIMER §7 "Consolidation, forgetting and deletion"). Consolidation is a batch job
# that calls a model. Thus it must be a **durable run** (lra-gcp primer §3.3 "Leases and the reaper", §3.13
# "Scheduled and event-triggered runs"):
#
# * a **deterministic run id**, `consolidate/<tenant>/<user>/<ISO week>`. Thus a duplicated or retried trigger
#   is the same run, and a finished run is a no-op. The schedule is weekly to match the id. With a daily
#   trigger and a weekly id, the job does its work on one day in seven and does nothing on the other six.
# * a **lease** row, so that two workers never run the job at the same time. The lease of a crashed worker
#   expires, and the next worker takes over.
# * a **checkpoint** per step, so that a resume skips finished work, above all the model calls.
# * **idempotent effects**. When the step that crashed after its write and before its checkpoint runs again,
#   the write still occurs only once (durable primer §3.2).
#
# Then the job resolves the facts:
#
# * A newer fact supersedes an older fact.
# * The precedence is human > user > tool > inferred.
# * The job marks a weaker contradiction with a flag.
#
# After that, consolidated episodes get a TTL (forgetting by decay). When the importance sum of the window
# crosses a threshold (150 in the generative-agents code), the job also writes one insight that cites its
# evidence. Primer: [`../../PRIMER.md`](../../PRIMER.md).

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
# This run has eight episodes, in two chunks of four (one model call each). Then it has a plan of actions, with
# one step per action.

# %%
clean, clean_clock = week_store(os.path.join(WORK, "clean.db"))
rep = ConsolidationJob(clean, worker="A").run("acme", "u1", START, END)
print(rep.line())
print("steps:", " ".join(rep.steps_run), "| checkpoints left after the run:", clean.job_tables().steps(rep.run_id))
for r in clean.records("acme", "u1", kind="semantic"):
    print(f"  {r.status:10s} {r.text:78s} valid_to={'-' if r.valid_to is None else dt.date.fromtimestamp(r.valid_to)}")

# %% [markdown]
# Lisbon and Globex are **superseded, not deleted**. Their validity closes when Porto and Initech begin
# (Graphiti keeps the same pair as `invalid_at` / `expired_at`). Thus the question "where did the user live on
# 15 September?" still has an answer (`store.search(..., as_of=...)`). The job skipped the hedged allergy ("I
# think…", confidence 0.6), because it is below the floor of the write policy.
#
# ## Worked example: kill it at the worst moment, then resume
#
# Worker A crashes in `apply:2`, *after* the write of the step and *before* its checkpoint. The notebook
# schedules worker B immediately, and B finds that A still holds the lease. One minute later, the lease has
# expired, and B takes over.

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
# Write `my_run_id(tenant, user, window_end)`. It returns `consolidate/<tenant>/<user>/<YYYY>-W<ww>` from the
# ISO calendar week of `window_end` (a unix time, UTC). Pad the week with zeros to two digits. A trigger that
# occurs two times, or a retry one hour later, must give the same id.

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
# The trigger must occur as often as the id changes. `week_window(now)` gives the window that a trigger at
# `now` works on: the previous ISO week, Monday 00:00 to Monday 00:00 UTC. Thus every trigger in one week names
# the same run. With a daily schedule, six of seven triggers find that run `done` and return:

# %%
fires = [ts_of("2026-09-21") + d * 86400 + 3 * 3600 + 17 * 60 for d in range(7)]       # 03:17 each day
ids = [C.run_id("acme", "u1", C.week_window(t)[1]) for t in fires]
print("a daily trigger's run ids this week:", sorted(set(ids)), "-> one run, six no-ops")
print("so the schedule is weekly:", next(c for c in C.gcp_commands() if "scheduler jobs create" in c).split("--schedule ")[1][:13])

# %% [markdown]
# ## Exercise 4.2 — resolve a new fact against what memory holds
#
# Write `my_resolve(existing, new)`. It compares a new semantic fact with the records of the partition and
# returns `(action, other)`. Only the `active` records of the same `slot` count:
#
# | case | action | `other` |
# |---|---|---|
# | same value already there (case-insensitive) | `"NOOP"` | that record |
# | nothing for the slot, or a multi-valued slot (`trip`) | `"ADD"` | `None` |
# | the current value's trust outranks the new one's (human > user > tool > inferred) | `"FLAG"` | the current record |
# | the new fact is *older* than the current one (a late episode) | `"ADD_HISTORY"` | the current record |
# | all other cases | `"UPDATE"` (the new fact will supersede the current record) | the current record |
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
# attempts)`. In **one** SQL statement, do one of these:
#
# * Insert the lease.
# * Take it over if it has expired.
# * Renew it if `holder` already has it.
# * Do nothing if the lease of another holder is still live.
#
# Return True when `holder` holds the lease after the statement. SQLite has
# `INSERT … ON CONFLICT … DO UPDATE … WHERE`, and Postgres has the same (see `store/pgvector.py` `ACQUIRE_LEASE`).
# With two statements (a SELECT, then an UPDATE), a race condition is possible.

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
# `calls_on_resume(n_chunks, crash_at)`. The first worker crashed in step `crash_at` (after its effect, before its
# checkpoint). The function returns the number of model calls that the *resumed* run makes. The steps before the
# crash have checkpoints. The crashed step runs again.

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
# This example runs the planted-facts harness (notebook 05) at a budget of 64 memory tokens per turn. The
# harness writes memory as raw episodes, and some rows then run consolidation on them. The four rows show this:
#
# * Facts are better than raw episodes, because a fact is shorter and names its slot.
# * Raw episodes in the user's own words sometimes match a first-person question better (the paraphrase
#   column).
# * If raw episodes stay retrievable next to the facts, an **old episode** can bring back a superseded value
#   (the `stale` column).

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
# The job gave every consolidated episode a TTL of 30 days from consolidation. Thirty-one days later, the
# episodes expire, and the store purges them. The facts that they produced stay. Each fact cites the id of its
# evidence. The store has now deleted that evidence.

# %%
clean_clock.t = END + 31 * 86400
n = clean.expire()
print(f"expired {n} episodes; left:", {k: sum(1 for r in clean.records('acme', 'u1') if r.kind == k)
                                       for k in ("episodic", "semantic")})

# %% [markdown]
# ## T3, printed: a Cloud Run job on Cloud Scheduler
#
# The job is `python -m memlab consolidate` in the image of the lab. By default, it works on the previous ISO
# week. It starts every Monday at 03:17 UTC. Cloud Run jobs start `--tasks N` copies. Each copy takes the
# partitions whose hash maps to its `CLOUD_RUN_TASK_INDEX` (verify). Cloud Scheduler calls `jobs/<job>:run` of
# the Cloud Run Admin API with an **OAuth** token, because the target is a `googleapis.com` API.
#
# OIDC is for your own `run.app` service, like the lra-gcp reaper. On GCP the store is Postgres + pgvector
# (Cloud SQL, verify), never a SQLite file on the ephemeral disk of an instance. The printed path is
# sufficiently complete to work:
#
# * The commands build and push the image.
# * The job runs as its own service account. That account has permission to read the DSN secret and to
#   connect to Cloud SQL (`--set-cloudsql-instances`).
# * The cleanup deletes everything that the setup created (`deploy/gcp/README.md`).

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
# **Two minutes.** "Consolidation runs weekly as a Cloud Run job per shard of users. Each (tenant, user, ISO
# week) is one durable run with a deterministic id. The schedule starts the job once a week to match. Thus a
# second trigger from the schedule is a no-op, and each weekly trigger has a new run to do.
#
# "Facts that a user states during the week reach memory immediately, through extraction after the turn. The
# job only distils the episodes of the week.
#
# "A lease row keeps two workers apart, and it expires when one worker stops. Every step writes a checkpoint,
# thus a resume skips the model calls that it already paid for. Writes carry idempotency keys derived from the
# run. Thus the step that crashed after its write does not write two times.
#
# "Resolution is deterministic. A newer fact supersedes an older fact, and the old validity closes. A weaker
# source never overwrites a stronger one, and the job marks the weaker fact with a flag. The job skips
# extractions with low confidence. Consolidated episodes get a 30-day TTL.
#
# "We measured recall at a fixed budget before and after. We decided to answer from facts after an episode is
# consolidated, because a raw episode can bring back a superseded value."
#
# **Drill 1.** *The scheduler sent the trigger two times at 03:17. What occurs?* Both triggers calculate the
# same run id. One takes the lease, and the other gets `LeaseHeld` (or, if the first finished, sees the run
# `done`) and exits.
#
# **Drill 1b.** *Someone changed the schedule to daily. What breaks?* Nothing breaks visibly: the first trigger
# of the week does the work, and the other six find the run `done`. That is the bug: a "daily" job that runs
# weekly. The run id and the window must change with the schedule (a per-day id and a one-day window).
#
# **Drill 2.** *A worker crashed after it wrote a fact but before its checkpoint. Is there a duplicate?* No:
# the resumed step writes with the same idempotency key (run, slot, value, evidence) and the planned record id.
# Thus the store returns the row that already exists. The job applies the supersede only if the old fact is
# still active.
#
# **Drill 3.** *Why OAuth, not OIDC, for Cloud Scheduler here?* The target is the Cloud Run Admin API
# (`run.googleapis.com`), and that API takes OAuth access tokens. OIDC identity tokens are for calls to the URL
# of your own service.
