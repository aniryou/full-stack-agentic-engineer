"""Consolidation as a durable run: a lease, checkpoints, a crash anywhere, the same end state."""
import pytest

from memlab import consolidate as C
from memlab.consolidate import ConsolidationJob, LeaseHeld, SimulatedCrash
from memlab.extract import ts_of
from memlab.records import MemoryRecord
from memlab.store import SQLiteMemoryStore

WEEK = [("2026-09-14", "I live in Lisbon and my manager is Ana."), ("2026-09-15", "I work at Globex."),
        ("2026-09-16", "My dog is called Rex."), ("2026-09-17", "I took a trip to Rome."),
        ("2026-09-18", "I moved to Porto."), ("2026-09-19", "I started at Initech."),
        ("2026-09-20", "I took a trip to Kyoto."), ("2026-09-20", "I think I am allergic to peanuts.")]
START, END = ts_of("2026-09-14"), ts_of("2026-09-21")


class Clock:
    def __init__(self):
        self.t = END

    def __call__(self):
        return self.t


def week(path):
    clock = Clock()
    s = SQLiteMemoryStore(path, clock=clock)
    for d, text in WEEK:
        s.add(MemoryRecord("acme", "u1", f"On {d} the user said: {text}", kind="episodic", importance=25,
                           created_at=ts_of(d)))
    return s, clock


def state(s):
    return sorted((r.status, r.text, r.valid_to) for r in s.records("acme", "u1", kind="semantic"))


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    s, _ = week(tmp_path_factory.mktemp("c") / "clean.db")
    rep = ConsolidationJob(s).run("acme", "u1", START, END)
    return s, rep


def test_clean_run(clean):
    s, rep = clean
    assert rep.status == "done" and rep.model_calls == 2 and rep.episodes == 8 and rep.insights == 1
    assert rep.actions == {"ADD": 6, "UPDATE": 2, "SKIP_LOW_CONFIDENCE": 1}
    by = {r.text: r for r in s.records("acme", "u1", kind="semantic")}
    assert by["Home city: the user lives in Lisbon."].status == "superseded"
    assert by["Home city: the user lives in Lisbon."].valid_to == ts_of("2026-09-18")
    assert by["Home city: the user lives in Porto."].status == "active"
    assert not any("allergic" in t for t in by)
    assert s.job_tables().steps(rep.run_id) == []                        # checkpoints cleared when done
    assert all(r.ttl_s is not None for r in s.records("acme", "u1", kind="episodic"))


@pytest.mark.parametrize("crash", ["load", "extract:0", "extract:1", "resolve", "apply:0", "apply:2", "apply:8",
                                   "reflect", "mark"])
def test_crash_anywhere_resume_same_state(tmp_path, clean, crash):
    s, clock = week(tmp_path / "x.db")
    with pytest.raises(SimulatedCrash):
        ConsolidationJob(s, worker="A", crash_at=crash).run("acme", "u1", START, END)
    with pytest.raises(LeaseHeld):
        ConsolidationJob(s, worker="B").run("acme", "u1", START, END)
    clock.t += 61
    rep = ConsolidationJob(s, worker="B").run("acme", "u1", START, END)
    want_calls = 2 if crash == "load" else 2 - int(crash.split(":")[1]) if crash.startswith("extract") else 0
    assert rep.status == "done" and rep.model_calls == want_calls
    assert state(s) == state(clean[0])


def test_a_duplicate_trigger_is_a_no_op(clean):
    s, rep = clean
    again = ConsolidationJob(s, worker="C").run("acme", "u1", START, END)
    assert again.status == "done" and again.steps_run == [] and again.steps_skipped == ["(run already done)"]


def test_run_id_and_sharding():
    assert C.run_id("acme", "u1", END) == "consolidate/acme/u1/2026-W39"
    assert C.run_id("acme", "u1", ts_of("2027-01-01")) == "consolidate/acme/u1/2026-W53"
    parts = [("acme", f"u{i}") for i in range(50)]
    shards = [C.shard(parts, k, 4) for k in range(4)]
    assert sorted(p for sh in shards for p in sh) == sorted(parts)


def test_lease_semantics(mem_store):
    jobs = mem_store.job_tables()
    assert jobs.acquire("r", "A", 0, 60) and not jobs.acquire("r", "B", 30, 60)
    assert jobs.acquire("r", "A", 50, 60) and not jobs.acquire("r", "B", 100, 60) and jobs.acquire("r", "B", 111, 60)
    assert jobs.lease("r") == ("B", 171, 2)


def test_ttl_expiry_after_consolidation(tmp_path):
    s, clock = week(tmp_path / "t.db")
    ConsolidationJob(s).run("acme", "u1", START, END)
    clock.t = END + 29 * 86400
    assert s.expire() == 0
    clock.t = END + 31 * 86400
    assert s.expire() == 8 and len(s.records("acme", "u1", kind="semantic")) == 9


def test_gcp_commands_use_oauth_and_the_v2_run_uri():
    cmds = C.gcp_commands("p", "europe-west1", tasks=4)
    joined = "\n".join(cmds)
    assert "--oauth-service-account-email" in joined and "oidc" not in joined.lower()
    assert "https://run.googleapis.com/v2/projects/p/locations/europe-west1/jobs/memlab-consolidate:run" in joined
    assert "--tasks 4" in joined and "--max-retries 3" in joined and "--http-method POST" in joined
    programs = {c.split(" | ")[-1].split()[0] for c in cmds + C.cleanup_commands()}
    assert programs == {"gcloud", "docker"}


def test_the_schedule_is_as_weekly_as_the_run_id():
    """A weekly run id with a daily trigger would make six of seven firings no-ops."""
    cron = next(c for c in C.gcp_commands() if "scheduler jobs create" in c).split('--schedule "')[1].split('"')[0]
    minute, hour, dom, month, dow = cron.split()
    assert (dom, month, dow) == ("*", "*", "1")                          # Mondays
    ids = {C.run_id("acme", "u1", C.week_window(ts_of("2026-09-21") + h * 3600)[1]) for h in range(0, 7 * 24, 5)}
    assert ids == {"consolidate/acme/u1/2026-W39"}                       # every trigger that week: one run
    start, end = C.week_window(ts_of("2026-09-23") + 3600)
    assert (start, end) == (START, END)                                  # the previous ISO week, Monday to Monday
    nxt = C.run_id("acme", "u1", C.week_window(ts_of("2026-09-28") + 3600)[1])
    assert nxt == "consolidate/acme/u1/2026-W40"


def test_a_late_episode_becomes_history_and_a_forget_clears_in_flight_checkpoints(tmp_path):
    s, clock = week(tmp_path / "late.db")
    ConsolidationJob(s).run("acme", "u1", START, END)
    # next week, an episode from before the move arrives late
    s.add(MemoryRecord("acme", "u1", "On 2026-09-10 the user said: I live in Madrid.", kind="episodic",
                       importance=25, created_at=END + 3600, valid_from=ts_of("2026-09-10")))
    clock.t = END + 7 * 86400
    ConsolidationJob(s).run("acme", "u1", END, END + 7 * 86400)
    madrid = [r for r in s.records("acme", "u1", kind="semantic") if "Madrid" in r.text]
    assert len(madrid) == 1 and madrid[0].status == "superseded"
    lisbon = [r for r in s.records("acme", "u1", kind="semantic") if "Lisbon" in r.text][0]
    assert lisbon.status == "superseded" and madrid[0].valid_to == lisbon.valid_from == ts_of("2026-09-14")
    as_of = {h.record.text for h in s.search("acme", "u1", "home city", 5, as_of=ts_of("2026-09-12"), kinds=("semantic",))}
    assert as_of == {"Home city: the user lives in Madrid."}
    # a crashed run holds extracted text in its checkpoints until a forget clears them
    s.add(MemoryRecord("acme", "u1", "On 2026-10-05 the user said: my address is 12 Rua das Flores.", kind="episodic",
                       created_at=END + 15 * 86400, deletion_key="acme/u1/address"))
    with pytest.raises(SimulatedCrash):
        ConsolidationJob(s, crash_at="resolve").run("acme", "u1", END + 14 * 86400, END + 21 * 86400)
    rep = s.forget("acme", "acme/u1/address", user="u1", needles=["Flores"])
    assert rep.counts["checkpoints"] >= 2 and rep.clean
