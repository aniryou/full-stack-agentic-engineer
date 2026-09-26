# agent-core

The smallest honest agent: **a fake model, a tool, and the loop** — the one idea the
rest of the stack keeps circling back to. Pure standard library, synchronous, about 200
lines you can read in a sitting, plus four fill-in notebooks.

This is the *starter*. The full `gcp-agent-platform-lab` (async, parallel tools, MCP,
OAuth, evals, tracing) is the **step-up** for later — same concepts, much more
machinery. Learn it here first.

## Quick start

```bash
cd agent-core
python3 -m pip install -r requirements.txt   # only to run the notebooks/tests
python3 -m pytest -q                          # 11 tests, ~0.1s
python3 -m jupyterlab notebooks               # do the exercises
```

The library itself needs **nothing installed** — it is standard library only:

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

## The whole library (three files)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `agentcore/fake_llm.py` | ~110 | a tool-calling model returns *text* or *tool calls*; drive it with a script or a policy |
| `agentcore/tools.py` | ~90 | a tool is a contract: schema from the signature, arguments validated, results structured |
| `agentcore/agent.py` | ~90 | the loop: call the model → run tools → append results → repeat, with a step budget and a human-approval gate |

Read them in that order. There is no async, no pydantic, no framework — just the shape.

## The notebooks

Each has worked examples, then exercises with `# YOUR CODE HERE` and a check cell that
prints ✅ when you get it right. Solutions are in `solutions/`.

1. **`01_the_agent_loop`** — build the loop yourself (termination, tool dispatch, the loop, the budget), then meet the packaged `Agent`.
2. **`02_tools`** — tool contracts: schema, validation, structured errors, an idempotent write, and watching the loop recover from a not-found error.
3. **`03_state_and_control`** — multi-turn memory, a policy that reacts to results, duplicate-call detection, reasoning about the step budget.
4. **`04_mini_support_agent`** — a small bank support agent: every fact from a tool, a card block gated by human approval, out-of-scope work escalated as a case. The primer's §8.1 scenario, shrunk.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format
with `### BEGIN SOLUTION` blocks). Edit the sources, then:

```bash
python3 tools/build_notebooks.py                        # rebuild both variants
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks must stop at the first exercise
```

## When you outgrow this

Reach for `gcp-agent-platform-lab` when you want to see: async and parallel tool
execution, MCP servers with a policy gateway, OAuth identity propagation, evaluation
gates, and OpenTelemetry-style tracing. Everything there is built on the loop you learn
here. Running a `run_code` tool safely — no credentials, no network by default, a budget for
every resource — is the [sandboxed-execution](../../sandboxed-execution/README.md) topic, which
reuses this loop's tool contract. MIT licensed.
