# agent-core — build the agent loop yourself, then swap in a real model

This is the smallest honest agent. It has **a fake model, a tool, and the loop**. After this lab, you can explain
these things from your own code: termination, tool dispatch, a step budget, an approval gate and structured tool
errors. You can also show that the loop does not care which provider answers.

## Start here

1. Read [`agentcore/fake_llm.py`](agentcore/fake_llm.py), [`tools.py`](agentcore/tools.py) and
   [`agent.py`](agentcore/agent.py), in that sequence. This takes about 30 min. The files have ~330 lines and use
   only the standard library.
2. Run `python3 -m pip install -r requirements.txt && python3 -m pytest -q`. There are 32 tests. They take ~20 s.
3. Open [`notebooks/01_the_agent_loop.ipynb`](notebooks/01_the_agent_loop.ipynb). Fill in the exercises.

## What you get

*T0 is a laptop or a Colab CPU, at no cost.* Nothing here needs a GPU, a cloud account or a key.

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| `notebooks/01_the_agent_loop` | Build the loop: termination, tool dispatch and the budget. Then use the packaged `Agent`. | 50 min | T0 |
| `notebooks/02_tools` | Write tool contracts: a schema, validation, structured errors and an idempotent write. Look at how the loop recovers from a not-found error. | 50 min | T0 |
| `notebooks/03_state_and_control` | Keep multi-turn memory, react to results and find duplicate calls. Think about the step budget. | 50 min | T0 |
| `notebooks/04_mini_support_agent` | Build a small bank support agent. Every fact comes from a tool. A card block must have a person's approval. The agent escalates out-of-scope work as a case. This is the bank agent of `gcp-agent-platform-lab`'s notebook 14, in a smaller form. | 60 min | T0 |
| `notebooks/05_going_live_on_mistral` | Replace `FakeLLM` with a provider adapter. Learn what the adapter translates. Write a routing rule that selects the lowest-cost model that meets the bar. Make a guarded live call. | 45 min | T0 (a key adds the live call) |
| [`docs/MISTRAL.md`](docs/MISTRAL.md) | The reference of the adapter: shapes, the client, models and prices (dated) and the deployment posture. | 15 min | — |

The solutions are in `solutions/`. Each exercise has a check cell. The check cell prints ✅ when your answer is
correct.

**The provider path.** The optional `mistral` extra (`pip install -e ".[mistral]"`) adds only the `mistralai` client.
[`agentcore/mistral_llm.py`](agentcore/mistral_llm.py) uses this client for a live call. The tests of the adapter run
without the extra, a key or a network. All of notebook 05 also runs without them, except its last cell. Without
them, that cell prints a labelled stop and does not fail.

## Run it

```bash
cd agent-core
python3 -m pip install -r requirements.txt              # pytest and Jupyter; the library itself needs nothing
python3 -m pytest -q                                     # 32 tests, ~20 s (one skips without the mistral extra)
python3 -m jupyterlab notebooks                          # do the exercises
```

The library uses only the standard library:

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

To go live, change one line. Use the same `Agent` with a different model object:

```bash
python3 -m pip install -e ".[mistral]"
export MISTRAL_API_KEY=...                               # billed per token
```

```python
from agentcore.mistral_llm import MistralLLM
agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[get_time])
```

The builder makes `notebooks/` and `solutions/` from `notebooks_src/*.py`. The sources are in percent format with
`### BEGIN SOLUTION` blocks. Edit the sources, then run these commands:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants (a no-op when nothing changed)
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

## The library

| File | Lines | What it teaches |
|------|-------|-----------------|
| `agentcore/fake_llm.py` | ~115 | A tool-calling model returns *text* or *tool calls*. A script or a policy controls it. |
| `agentcore/tools.py` | ~130 | A tool is a contract. The schema comes from the signature, the tool validates the arguments, and the tool gives structured results. |
| `agentcore/agent.py` | ~90 | The loop calls the model, runs the tools, appends the results and does it again. It has a step budget and a human-approval gate. |
| `agentcore/mistral_llm.py` | ~150 | An *optional* provider adapter. It runs the same loop against Mistral's API. It has pure converters and a thin wrapper around the client. |

There is no async, no pydantic and no framework. There is only the shape.

## When you outgrow this

Use [`gcp-agent-platform-lab`](../gcp-agent-platform-lab/README.md) for async and parallel tool execution, MCP
servers with a policy gateway, OAuth identity propagation, evaluation gates and OpenTelemetry-style tracing.
It has the same concepts and much more machinery, and all of it uses this loop.

The [sandboxed-execution](../../sandboxed-execution/README.md) topic shows how to run a `run_code` tool safely. In that
topic, the tool gets no credentials and, by default, no network. It also has a budget for every resource. That topic
uses the tool contract of this loop again. This lab is module 07.1 in [`CURRICULUM.md`](../../../CURRICULUM.md).

## Caveats

- `FakeLLM` uses a script. It shows the mechanics of the loop, not the judgement of a model. Notebook 05's live call is
  the only place where a real model answers, and its output is different from run to run.
- The Mistral model aliases, the prices and the import path of the `mistralai` client change. Their dates in
  [`docs/MISTRAL.md`](docs/MISTRAL.md) are 2026-09-19 to 2026-09-26 `(verify)`.
- MIT licensed.
