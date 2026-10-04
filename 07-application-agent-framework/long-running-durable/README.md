# long-running-durable — agents that survive crashes, duplicate deliveries and multi-day waits

After this topic, you can take an agent that must wait and design it as a durable state machine. The agent can wait
for a tool, a person, the world or the clock. You can do these things:

- Give the invariants that keep the agent correct.
- Show where the design closes each crash window.
- Map the design onto a queue, a store and stateless compute on any cloud. Google Cloud is the full example, from
  end to end.

## Start here

1. Read [`PRIMER.md`](PRIMER.md) §0–§4 (45 min). These sections explain why a long-running agent is a different
   problem. They also describe the run as a state machine, and give the five invariants and the catalogue of patterns.
2. Run `cd lra-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q`. The 13 tests run in
   approximately one second and use only the standard library. Then run
   `python3 workflow.py kill; python3 workflow.py resume`. The output shows that one process crashes after it
   publishes. Then a second process completes the run and does not publish again.
3. Open [`lra-core/notebooks/01_durable_loop.ipynb`](lra-core/notebooks/01_durable_loop.ipynb). Then go to
   [`lra-gcp`](lra-gcp/README.md), the full engine.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost. T3 is the Google Cloud deployment, and it is optional. The T1/T2
tiers of the repository do not apply, because no part of the topic uses a GPU. Model keys are optional, because each
notebook runs a scripted model.* Each notebook has a blank in `notebooks/` and a worked answer with the same name in
`solutions/`. The blank stops at its first exercise.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`PRIMER.md`](PRIMER.md) | Explain the five invariants, eight patterns and three reference architectures. Give the Google Cloud limits that decide a design. Calculate approximate counts of wake-ups, writes and tokens. Answer the design drills in §11. | 2 h | T0 |
| [`lra-core`](lra-core/README.md) | Build the durable loop from nothing in ~250 lines of standard library. The loop has the store, the named-task queue, the lease, checkpoint-before-enqueue, the effect records and the reaper. Break the loop at each crash window. One of the crashes is a real two-process crash. There are 3 notebooks. | 2 h | T0 (T3 optional: one Cloud Run service) |
| [`lra-gcp`](lra-gcp/README.md) | Run the full engine. Offline, it needs only pydantic. It has fan-out/fan-in, human-in-the-loop, sagas, reflection, budgets, workflow versions, scheduled ticks, long-running tools and a tool loop in which the model selects the tool. It uses Firestore, Cloud Tasks, Pub/Sub, Cloud Run, Cloud Workflows and Terraform. The ADK 2 and Mistral paths are optional. There are 6 notebooks. | 6 h (8 h with the optional paths) | T0, T3 |

The topic puts three earlier lineages into one. Each part comes from one lineage:

- The Google Cloud edition had a standard-library core and an ADK-based lab. From it came this primer, its drills,
  the ADK 2 ticket-queue workflow, the scheduled and long-running-tool patterns and the two-process crash.
- From the Mistral edition came the Mistral adapter, the Mistral Workflows version and their tests.
- The engine, its tests and its notebooks come from the `lra` lineage.

## Run it

```bash
cd lra-core
python3 -m pip install -r requirements.txt   # pytest and Jupyter's runner; core.py itself is standard library
python3 -m pytest -q                         # 13 tests, ~1 s
python3 tools/run_notebooks.py solutions     # the worked answers run clean

cd ../lra-gcp
python3 -m pip install -e ".[dev,services]"  # pydantic, FastAPI for the services, Jupyter's runner
python3 -m pytest -q                         # 73 tests: 66 pass, 7 skip without the optional extras, ~15 s
python3 scripts/local_demo.py                # fan-out -> crash -> reaper -> 3-day wait -> approval -> saga rollback
```

## How it fits

Read about the agent loop first ([`agent-core`](../agent-fundamentals/agent-core/README.md), module 07.1). This topic
makes that loop work over long periods of time. Notebook 03 of the platform lab
([`gcp-agent-platform-lab`](../agent-fundamentals/gcp-agent-platform-lab/README.md)) is about state and checkpoints
in one session. For anything that waits, it refers to this topic.

Layer 06 ([`06-gateway`](../../06-gateway/README.md)) is the adjacent layer. It sets limits on the turns and on the
authority of the agent. Module 07.3 of the curriculum ([`CURRICULUM.md`](../../CURRICULUM.md)) sets the path and the
drills.

## Caveats

- At T0, the store, the queue and the clock are in-memory twins with the same semantics. A script takes the place of
  the model. Thus the notebooks show simulated costs and latencies. Unit tests run the Google Cloud adapters against
  fake clients. The Terraform passes validation, but no one applied it to a real project during the work on this topic.
- The ADK 2 path (`pip install -e ".[adk]"`, approximately 220 MB) and the Mistral Workflows path (Python 3.12–3.14,
  `mistralai-workflows`) are optional, and they are off by default. Without them, their tests and notebook 04 skip
  (measured 2026-09-26, verify).
- The product facts are about ADK 2, the Gemini and Mistral model names, and the limits of Cloud Run, Cloud Tasks,
  Workflows and Pub/Sub. These facts have the date September 2026 and the tag (verify). At its end, the primer gives a
  list of facts to examine again.
- The prose of this topic is in ASD-STE100 (Simplified Technical English). This topic was the first to use the
  style, as an experiment. The other layers followed. The style has short sentences, the active voice, the
  imperative for instructions and one meaning for each word. The brief is
  [STE100-STYLE.md](../../tools/orchestration/STE100-STYLE.md) and the linter is
  [ste_lint.py](../../tools/orchestration/ste_lint.py). The root documents keep their usual style.
