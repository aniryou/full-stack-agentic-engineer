"""workflow.py — a four-step long-running workflow: draft -> review (wait days) -> publish -> notify.

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


if __name__ == "__main__":
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
