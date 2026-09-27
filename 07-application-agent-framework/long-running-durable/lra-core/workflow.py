"""workflow.py — a four-step long-running workflow: draft -> review (wait days) -> publish -> notify.

    python workflow.py           # start, park at review, "two days" pass, approve, publish
    python workflow.py kill      # process 1 publishes, then dies before the checkpoint (exit 137)
    python workflow.py resume    # process 2 finds the run in runs.json and finishes it: one publish

Steps take a `ctx` (state copy, input, effect) and return an outcome tuple. No framework:
the `model` and `cms` below are stand-ins for Gemini and an external system.
"""

import json

CALLS = []                                   # what the "external systems" saw (for tests/notebooks)


def model(prompt):                           # Gemini stands in for this
    CALLS.append(("model", prompt[:30]))
    return "Draft about " + prompt.split(":")[-1].strip()


def cms_publish(text):                       # an external API with real consequences
    CALLS.append(("publish", text))
    return {"post_id": f"post_{abs(hash(text)) % 10_000}"}


def draft(ctx):
    ctx.state["draft"] = model("Write a short post about: " + ctx.input["topic"])
    return ("next", "review")


def review(ctx):
    # Suspend. Nothing runs, nothing is billed, until POST /runs/{id}/events arrives with this key.
    return ("wait", f"approval:{ctx.run_id}", "publish")


def publish(ctx):
    decision = ctx.state["events"][f"approval:{ctx.run_id}"]
    if decision.get("decision") != "approve":
        return ("done", {"published": False, "reason": decision.get("comment", "rejected")})
    rec = ctx.effect("publish", lambda: cms_publish(ctx.state["draft"]))   # I3: at most once, ever
    ctx.state["post_id"] = rec["post_id"]
    return ("next", "notify")


def notify(ctx):
    ctx.effect("notify", lambda: CALLS.append(("email", ctx.state["post_id"])) or {"sent": True})
    return ("done", {"published": True, "post_id": ctx.state["post_id"]})


STEPS = {"draft": draft, "review": review, "publish": publish, "notify": notify}


RUNS_FILE = "runs.json"


def kill(path=RUNS_FILE):
    """Process 1: start a run, approve it, publish -- then die before the checkpoint (exit 137)."""
    import os
    import time
    from core import Engine, FileStore, Queue, drain

    if os.path.exists(path):
        os.remove(path)
    engine = Engine(FileStore(path), Queue(), dict(STEPS), clock=time.time)
    engine.start("run-1", "draft", {"topic": "durable agents"})
    drain(engine, engine.queue, time.time)                  # draft ok, review waiting
    engine.resume("run-1", "approval:run-1", {"decision": "approve", "by": "editor"})

    def publish_then_die(ctx):
        publish(ctx)                                        # the effect happened and its record is on disk ...
        print("published", CALLS[-1][1][:40], "-- now dying before the checkpoint")
        os._exit(137)                                       # ... then SIGKILL-style: no cleanup, lease never released

    engine.steps["publish"] = publish_then_die
    drain(engine, engine.queue, time.time)


def resume(path=RUNS_FILE, later_s=3600):
    """Process 2: a different process opens the same store an hour later; the reaper finishes the run."""
    import time
    from core import Engine, FileStore, Queue, drain

    now = lambda: time.time() + later_s                     # the dead worker's lease has long expired
    engine = Engine(FileStore(path), Queue(), dict(STEPS), clock=now)
    run = engine.store.get("run-1")
    print("found run-1:", run["status"], "at", run["step"], "| lease expired:", run["lease"] is not None and run["lease"] <= now())
    print("reaper:", engine.reap())
    print(drain(engine, engine.queue, now))
    run = engine.store.get("run-1")
    print("result:", run["status"], run["result"], "| publish calls in THIS process:", [c for c in CALLS if c[0] == "publish"])
    return run


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["kill"]:                            # python workflow.py kill; python workflow.py resume
        kill()
    if sys.argv[1:] == ["resume"]:
        resume()
        sys.exit(0)

    from core import Engine, Queue, Store, drain

    clock = [0.0]
    now = lambda: clock[0]                                  # a clock we control
    engine = Engine(Store(), Queue(), STEPS, clock=now)

    run = engine.start("run-1", "draft", {"topic": "durable agents"})
    print(drain(engine, engine.queue, now))                 # -> draft ok, review waiting
    print(json.dumps(engine.store.get("run-1"), indent=1, default=str))

    clock[0] += 2 * 86400                                   # two days later ...
    engine.resume("run-1", "approval:run-1", {"decision": "approve", "by": "editor"})
    print(drain(engine, engine.queue, now))                 # -> publish ok, notify done
    print(engine.store.get("run-1")["result"], CALLS)
