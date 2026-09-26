# agent-core — build the agent loop yourself, then swap in a real model

The smallest honest agent — **a fake model, a tool, and the loop**: after it you can explain termination, tool
dispatch, a step budget, an approval gate and structured tool errors from code you wrote, and show that the loop does
not care which provider answers.

## Start here

1. Read [`agentcore/fake_llm.py`](agentcore/fake_llm.py), [`tools.py`](agentcore/tools.py) and
   [`agent.py`](agentcore/agent.py), in that order (about 30 min; ~330 lines, standard library only).
2. `python3 -m pip install -r requirements.txt && python3 -m pytest -q` — 32 tests, ~20 s.
3. Open [`notebooks/01_the_agent_loop.ipynb`](notebooks/01_the_agent_loop.ipynb) and fill in the exercises.

## What you get

*T0 = a laptop or Colab CPU, free.* Nothing here needs a GPU, a cloud account or a key.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| `notebooks/01_the_agent_loop` | build the loop — termination, tool dispatch, the budget — then meet the packaged `Agent` | 50 min | T0 |
| `notebooks/02_tools` | write tool contracts: schema, validation, structured errors, an idempotent write; watch the loop recover from a not-found error | 50 min | T0 |
| `notebooks/03_state_and_control` | keep multi-turn memory, react to results, detect duplicate calls, reason about the step budget | 50 min | T0 |
| `notebooks/04_mini_support_agent` | build a small bank support agent: every fact from a tool, a card block gated by human approval, out-of-scope work escalated as a case (the bank agent of `gcp-agent-platform-lab`'s notebook 14, shrunk) | 60 min | T0 |
| `notebooks/05_going_live_on_mistral` | swap `FakeLLM` for a provider adapter: what it translates, a routing rule for the cheapest model that clears the bar, a guarded live call | 45 min | T0 (a key adds the live call) |
| [`docs/MISTRAL.md`](docs/MISTRAL.md) | the adapter's reference: shapes, the client, models and prices (dated), deployment posture | 15 min | — |

Solutions are in `solutions/`. Each exercise has a check cell that prints ✅ when you get it right.

**The provider path.** The optional `mistral` extra (`pip install -e ".[mistral]"`) adds only the `mistralai` client
that [`agentcore/mistral_llm.py`](agentcore/mistral_llm.py) uses for a live call; the adapter's tests and all of
notebook 05 except its last cell run without the extra, a key or a network, and without them that cell prints a
labelled stop instead of failing.

## Run it

```bash
cd agent-core
python3 -m pip install -r requirements.txt              # pytest and Jupyter; the library itself needs nothing
python3 -m pytest -q                                     # 32 tests, ~20 s (one skips without the mistral extra)
python3 -m jupyterlab notebooks                          # do the exercises
```

The library is standard library only:

```python
from agentcore import Agent, FakeLLM, tool, call

@tool
def get_time(city: str) -> dict:
    """Return the current time in a city."""
    return {"city": city, "time": "16:00"}

llm = FakeLLM([call("get_time", city="Singapore"), "It's 4pm in Singapore."])
result = Agent(llm, tools=[get_time]).run("what time is it in SG?")
print(result.text)          # It's 4pm in Singapore.
print(result.transcript())  # see every step the loop took
```

Going live is one line — the same `Agent`, another model object:

```bash
python3 -m pip install -e ".[mistral]"
export MISTRAL_API_KEY=...                               # billed per token
```

```python
from agentcore.mistral_llm import MistralLLM
agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[get_time])
```

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format with `### BEGIN SOLUTION`
blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants (a no-op when nothing changed)
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

## The library

| File | Lines | What it teaches |
|------|-------|-----------------|
| `agentcore/fake_llm.py` | ~115 | a tool-calling model returns *text* or *tool calls*; drive it with a script or a policy |
| `agentcore/tools.py` | ~130 | a tool is a contract: schema from the signature, arguments validated, results structured |
| `agentcore/agent.py` | ~90 | the loop: call the model → run tools → append results → repeat, with a step budget and a human-approval gate |
| `agentcore/mistral_llm.py` | ~150 | *optional* provider adapter: the same loop against Mistral's API; pure converters plus a thin client wrapper |

There is no async, no pydantic, no framework — just the shape.

## When you outgrow this

Reach for [`gcp-agent-platform-lab`](../gcp-agent-platform-lab/README.md) when you want async and parallel tool
execution, MCP servers with a policy gateway, OAuth identity propagation, evaluation gates and OpenTelemetry-style
tracing — the same concepts, much more machinery, all built on this loop. Running a `run_code` tool safely — no
credentials, no network by default, a budget for every resource — is the
[sandboxed-execution](../../sandboxed-execution/README.md) topic, which reuses this loop's tool contract. Module 07.1
in [`CURRICULUM.md`](../../../CURRICULUM.md).

## Caveats

- `FakeLLM` is scripted: it shows the loop's mechanics, not a model's judgement. Notebook 05's live call is the only
  place a real model answers, and its output varies run to run.
- Mistral model aliases, prices and the `mistralai` client's import path move; they are dated 2026-09-19 to 2026-09-26
  in [`docs/MISTRAL.md`](docs/MISTRAL.md) `(verify)`.
- MIT licensed.
