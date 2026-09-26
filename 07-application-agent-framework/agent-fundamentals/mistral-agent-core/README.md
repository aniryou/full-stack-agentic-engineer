# mistral-agent-core

The smallest honest agent: **a fake model, a tool, and the loop** — plus a one-line
swap to a real **Mistral** model. Pure standard library at the core, synchronous, about
200 lines you can read in a sitting, four fill-in notebooks, and a fifth that takes you
live on Mistral.

The core loop is provider-agnostic — that is the point. `FakeLLM` runs it offline;
`MistralLLM` runs it against `la Plateforme`; nothing else changes. The full-featured
version (async, parallel tools, MCP, OAuth, evals, tracing) is the separate
`gcp-agent-platform-lab` — the **step-up** for later. Learn it here first.

## Quick start (offline, no key)

```bash
cd mistral-agent-core
python3 -m pip install -r requirements.txt   # only to run the notebooks/tests
python3 -m pytest -q                          # 19 tests, ~0.1s
python3 -m jupyterlab notebooks               # do the exercises
```

The core library needs **nothing installed** — it is standard library only:

```python
from agentcore import Agent, FakeLLM, tool, call

@tool
def get_time(city: str) -> dict:
    """Return the current time in a city."""
    return {"city": city, "time": "16:00"}

llm = FakeLLM([call("get_time", city="Singapore"), "It's 4pm in Singapore."])
print(Agent(llm, tools=[get_time]).run("what time is it in SG?").text)
```

## Going live on Mistral

```bash
pip install mistralai
export MISTRAL_API_KEY=...          # from https://console.mistral.ai
```

```python
from agentcore import Agent, tool
from agentcore.mistral_llm import MistralLLM

agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[get_time])
print(agent.run("what time is it in Singapore?").text)
```

Same `Agent`, real model. The adapter (`agentcore/mistral_llm.py`) is the *only*
Mistral-specific code — it translates our tool schemas and messages into Mistral's
function-calling shapes and back. Models, pricing and the deployment story are in
[`docs/MISTRAL.md`](docs/MISTRAL.md).

## The core library (three files, provider-agnostic)

| File | Lines | What it teaches |
|------|-------|-----------------|
| `agentcore/fake_llm.py` | ~110 | a tool-calling model returns *text* or *tool calls*; drive it with a script or a policy |
| `agentcore/tools.py` | ~130 | a tool is a contract: schema from the signature, arguments validated, results structured |
| `agentcore/agent.py` | ~90 | the loop: call the model → run tools → append results → repeat, with a step budget and a human-approval gate |
| `agentcore/mistral_llm.py` | ~90 | *optional* adapter: the same loop against Mistral's API |

Read the first three in order — there is no async, no pydantic, no framework, just the
shape. The Mistral adapter is a small translation layer on top.

## The notebooks

Each has worked examples, then exercises with `# YOUR CODE HERE` and a check cell that
prints ✅ when you get it right. Solutions are in `solutions/`.

1. **`01_the_agent_loop`** — build the loop yourself, then meet the packaged `Agent`.
2. **`02_tools`** — tool contracts: schema, validation, structured errors, an idempotent write.
3. **`03_state_and_control`** — multi-turn memory, a policy that reacts to results, duplicate-call detection, the budget.
4. **`04_mini_support_agent`** — a small bank support agent: every fact from a tool, a card block gated by human approval, escalation as a case.
5. **`05_going_live_on_mistral`** — the swap to Mistral: what the adapter translates, choosing a model, and a guarded live call.

## Regenerating notebooks

`notebooks/` and `solutions/` are generated from `notebooks_src/*.py` (percent format
with `### BEGIN SOLUTION` blocks):

```bash
python3 tools/build_notebooks.py
python3 tools/run_notebooks.py solutions                # solutions must run clean
python3 tools/run_notebooks.py notebooks --expect-fail  # blanks stop at the first exercise
```

## When you outgrow this

`gcp-agent-platform-lab` is the production version: async and parallel tool execution, MCP
servers with a policy gateway, OAuth identity propagation, evaluation gates, and
tracing. Everything there is built on the loop you learn here. MIT licensed.
