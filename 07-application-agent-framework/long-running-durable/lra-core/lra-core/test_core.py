"""One test per claim in core.py's docstring, plus the reaper's crash windows. Run: python -m pytest -q"""

import pytest

from core import Crash, Engine, Queue, Store, drain
from workflow import CALLS, STEPS


@pytest.fixture
def env():
    CALLS.clear()
    clock = [0.0]
    now = lambda: clock[0]
    engine = Engine(Store(), Queue(), dict(STEPS), clock=now, lease_ttl=60)
    return engine, now, clock


def to_review(engine, now):
    engine.start("r1", "draft", {"topic": "x"})
    assert drain(engine, engine.queue, now) == [("draft", "ok"), ("review", "waiting")]
    return engine.store.get("r1")


def test_happy_path_wait_then_resume(env):
    engine, now, clock = env
    run = to_review(engine, now)
    assert run["status"] == "WAITING" and run["lease"] is None and engine.queue.tasks == []   # sleeping is free
    clock[0] += 2 * 86400
    engine.resume("r1", "approval:r1", {"decision": "approve"})
    assert drain(engine, engine.queue, now) == [("publish", "ok"), ("notify", "succeeded")]
    assert engine.store.get("r1")["result"]["published"] is True
    assert [c[0] for c in CALLS] == ["model", "publish", "email"]


def test_I2_duplicate_delivery_is_stale(env):
    engine, now, _ = env
    engine.start("r1", "draft", {"topic": "x"})
    assert engine.execute("r1", "draft", 1) == "ok"
    assert engine.execute("r1", "draft", 1) == "stale"        # same (step, attempt) again
    assert engine.execute("r1", "review", 7) == "stale"       # wrong attempt
    assert len(CALLS) == 1


def test_retry_bumps_attempt_with_backoff(env):
    engine, now, clock = env
    boom = {"n": 0}

    def flaky(ctx):
        boom["n"] += 1
        if boom["n"] == 1:
            raise ConnectionError("503")
        return ("done", "ok")

    engine.steps["flaky"] = flaky
    engine.start("r2", "flaky", {})
    assert drain(engine, engine.queue, now) == [("flaky", "retry")]
    assert engine.store.get("r2")["attempts"] == {"flaky": 2} and engine.queue.tasks[0][0] == 2   # due in 2s
    clock[0] += 2
    assert drain(engine, engine.queue, now) == [("flaky", "succeeded")]


def test_permanent_failure_after_max_attempts(env):
    engine, now, clock = env
    engine.steps["bad"] = lambda ctx: 1 / 0
    engine.start("r3", "bad", {})
    for _ in range(3):
        drain(engine, engine.queue, now)
        clock[0] += 10
    run = engine.store.get("r3")
    assert run["status"] == "FAILED" and "after 3 attempts" in run["error"]


def test_I1_crash_after_checkpoint_before_enqueue_is_repaired_by_reaper(env):
    engine, now, clock = env
    engine.crash_before_enqueue = True
    engine.start("r1", "draft", {"topic": "x"})
    assert drain(engine, engine.queue, now) == [("draft", "CRASH")]
    run = engine.store.get("r1")
    assert run["step"] == "review" and run["lease"] is not None       # checkpoint landed; lease never released
    clock[0] += 30
    assert drain(engine, engine.queue, now) == [("draft", "stale")]   # Cloud Tasks redelivers -> already done
    assert engine.reap() == []                                         # lease still live
    clock[0] += 31
    assert engine.reap() == [("lease", "r1")]                          # expired -> re-drive the current step
    assert drain(engine, engine.queue, now) == [("review", "waiting")]


def test_I3_crash_after_effect_before_checkpoint_does_not_repeat_effect(env):
    engine, now, clock = env
    run = to_review(engine, now)
    engine.resume("r1", "approval:r1", {"decision": "approve"})

    real_publish, dead = engine.steps["publish"], {"once": True}

    def publish_then_die(ctx):
        out = real_publish(ctx)             # the effect happened ...
        if dead["once"]:
            dead["once"] = False
            raise Crash("died before the checkpoint")
        return out

    engine.steps["publish"] = publish_then_die
    assert drain(engine, engine.queue, now) == [("publish", "CRASH")]
    assert [c[0] for c in CALLS].count("publish") == 1
    clock[0] += 61                          # lease expired; redelivered task now executes
    assert drain(engine, engine.queue, now) == [("publish", "ok"), ("notify", "succeeded")]
    assert [c[0] for c in CALLS].count("publish") == 1              # effect_once skipped it


def test_resume_is_idempotent_and_key_scoped(env):
    engine, now, _ = env
    to_review(engine, now)
    assert engine.resume("r1", "approval:other", {"decision": "approve"}) is None
    assert engine.resume("r1", "approval:r1", {"decision": "approve"}) is not None
    assert engine.resume("r1", "approval:r1", {"decision": "approve"}) is None   # duplicate webhook
    assert len(engine.queue.tasks) == 1


def test_wait_timeout_via_reaper(env):
    engine, now, clock = env
    to_review(engine, now)
    clock[0] += 3 * 86400 + 1
    assert engine.reap() == [("timeout", "r1")]
    assert engine.store.get("r1")["status"] == "FAILED"


def test_lease_blocks_second_worker(env):
    engine, now, clock = env
    engine.start("r1", "draft", {"topic": "x"})
    run = engine.store.get("r1")
    run["lease"] = now() + 60                 # another replica is mid-step
    engine.store.save(run)
    assert engine.execute("r1", "draft", 1) == "busy"
    clock[0] += 61
    assert engine.execute("r1", "draft", 1) == "ok"


# ---- regression: a crash between a checkpoint that leaves no lease and its enqueue ----
# start(), resume() and the retry branch of execute() write a RUNNING run with no lease, then
# enqueue. Die in between and there is no task and no lease to expire: only the reaper's orphan
# rule (RUNNING, no lease, untouched for 2 x lease_ttl) re-drives the run.

def crash_on_next_push(queue):
    real = queue.push

    def push(*args, **kwargs):
        queue.push = real                     # die once; later pushes work
        raise Crash("died after the checkpoint, before the enqueue")

    queue.push = push


def test_crash_between_retry_checkpoint_and_enqueue_is_redriven_by_reaper(env):
    engine, now, clock = env
    boom = {"n": 0}

    def flaky(ctx):
        boom["n"] += 1
        if boom["n"] == 1:
            raise ConnectionError("503")
        return ("done", "ok")

    engine.steps["flaky"] = flaky
    engine.start("r2", "flaky", {})
    crash_on_next_push(engine.queue)
    assert drain(engine, engine.queue, now) == [("flaky", "CRASH")]
    run = engine.store.get("r2")
    assert run["status"] == "RUNNING" and run["lease"] is None and run["attempts"] == {"flaky": 2}
    clock[0] += 30
    assert drain(engine, engine.queue, now) == [("flaky", "stale")]    # redelivered attempt 1 is stale
    assert engine.queue.tasks == [] and engine.reap() == []            # nothing queued; not orphaned yet
    clock[0] += 2 * 60
    assert engine.reap() == [("orphan", "r2")]
    assert drain(engine, engine.queue, now) == [("flaky", "succeeded")]


def test_crash_between_resume_checkpoint_and_enqueue_is_redriven_by_reaper(env):
    engine, now, clock = env
    to_review(engine, now)
    crash_on_next_push(engine.queue)
    with pytest.raises(Crash):
        engine.resume("r1", "approval:r1", {"decision": "approve"})
    run = engine.store.get("r1")
    assert run["status"] == "RUNNING" and run["step"] == "publish" and run["lease"] is None
    assert engine.queue.tasks == [] and engine.resume("r1", "approval:r1", {"decision": "approve"}) is None
    clock[0] += 2 * 60
    assert engine.reap() == [("orphan", "r1")]
    assert drain(engine, engine.queue, now) == [("publish", "ok"), ("notify", "succeeded")]


def test_crash_between_start_checkpoint_and_enqueue_is_redriven_by_reaper(env):
    engine, now, clock = env
    crash_on_next_push(engine.queue)
    with pytest.raises(Crash):
        engine.start("r1", "draft", {"topic": "x"})
    engine.start("r1", "draft", {"topic": "x"})                        # idempotent start does not re-enqueue
    assert engine.queue.tasks == []
    clock[0] += 2 * 60
    assert engine.reap() == [("orphan", "r1")]
    assert drain(engine, engine.queue, now) == [("draft", "ok"), ("review", "waiting")]
