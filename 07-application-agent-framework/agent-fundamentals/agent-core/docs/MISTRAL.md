# Running agent-core on Mistral

The whole lab runs offline with `FakeLLM`. When you want a real model, the only
Mistral-specific code is `agentcore/mistral_llm.py`. It is a provider adapter of about 150
lines, and it connects to the same `Agent`. Notebook `05_going_live_on_mistral` goes
through it. All of that notebook runs offline (T0), except the last cell.

```bash
pip install -e ".[mistral]"         # the optional extra: the mistralai client (2.x)
export MISTRAL_API_KEY=...          # from https://console.mistral.ai; live calls are billed per token
```

Without the extra or the key, `MistralLLM(...)` raises `MistralUnavailable`. The
exception says which part you do not have. Notebook 05 catches it and prints a labelled
stop. For tests, pass as `client=` any object that has `.chat.complete(**kwargs)`. Then you
need no SDK and no key (`tests/test_mistral_adapter.py` does this).

```python
from agentcore import Agent, tool
from agentcore.mistral_llm import MistralLLM

@tool
def get_balance(account_id: str) -> dict:
    """Return the balance for an account."""
    return {"account_id": account_id, "balance": 1234.5}

agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[get_balance],
              instruction="You are a bank assistant. Use tools for every fact.")
print(agent.run("what's my balance on a1?").text)
```

This is the same `Agent` as in notebooks 01–04. Only the model object is different.

## What the adapter translates

Mistral's chat API uses the usual function-calling shape. Thus the adapter is small:

| Direction | Our shape | Mistral shape |
|-----------|-----------|---------------|
| tools out | `{"name","description","parameters"}` | `{"type":"function","function":{...}}` |
| assistant tool call out | `args` as a **dict** | `function.arguments` as a **JSON string** |
| reply in | `Response(text, tool_calls)` | `response.choices[0].message.content` / `.tool_calls` |

In the 2.x SDK, the client is `from mistralai.client import Mistral; client = Mistral(api_key=...)`.
In 1.x, it is `from mistralai import Mistral`. The adapter tries both. The call is
`client.chat.complete(model=..., messages=..., tools=..., tool_choice="auto")`.
`MistralLLM.request(messages, tools)` returns exactly those keyword arguments. It
sends nothing.

`tool_choice` accepts `"auto"`, `"any"` (the model must call a tool), or `"none"`. The
surface of the SDK changes. If a call fails, read https://docs.mistral.ai.

## Choosing a model (list prices 2026-09-19, verify)

The prices are USD per million tokens, input / output. The repo pins them in
`06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/mistral.py` (dated
2026-09-19). The rows with the mark † come from the pricing page of the provider on the same
date. No test pins them.

Aliases such as `mistral-large-latest` move to newer releases.
Before you use an alias, find what it points at. Before you use a price, make sure that
it is correct.

| Model string | Use for | Input / output ($ per 1M) | Weights |
|--------------|---------|---------------------------|---------|
| `mistral-large-latest` (Large 3, 675B MoE, 41B active) | The most difficult tasks, **agents / tool use**, long context | $0.50 / $1.50 | **open (Apache-2.0)** |
| `mistral-medium-latest` (Medium 3.5, 128B dense) | Frontier-class agentic work and code tasks | $1.50 / $7.50 | Published (modified MIT). Above $20 M of monthly revenue, a commercial licence is necessary (verify). Thus notebook 05's snapshot counts it as not self-hostable. |
| `mistral-small-latest` (Small 4, 119B MoE, 6.5B active) | Cost-effective general work and tool use | $0.15 / $0.60 | **open (Apache-2.0)** |
| `ministral-14b-2512` / `ministral-8b-2512` / `ministral-3b-2512` | On-device use, low-cost high-volume work | $0.20 / $0.20, $0.15 / $0.15, $0.10 / $0.10 | **open (Apache-2.0)** |
| `magistral-medium-latest` † | Step-by-step **reasoning** | $2 / $5 | API |
| `codestral-latest` † | Code generation, autocomplete | $0.30 / $0.90 | API |
| `pixtral-latest`, `voxtral-*`, `mistral-ocr-latest` † | Vision, audio, document extraction | — | API |

Cached input costs 10 % of the input price. The Batch API costs half the price (2026-09-19, verify).

The general rule for an agent has three parts:

- For the tool loop, use the lowest-cost model that passes your evals. Exercise 5.3 turns
  this rule into code.
- Use a reasoning model only where the reasoning is worth its cost.
- When the data must not go out of your own environment, use an open-weight model that you
  host yourself.

**Deployment posture.** The place where the model can run is a design input, not an
afterthought. The open-weight models can run anywhere. Thus the same agent can call the
managed API (`la Plateforme`), run on weights deployed in your own VPC, or run fully on-prem.
When the data is regulated (banks, health, public sector) or must stay in one
jurisdiction, that choice can decide the model.

In a design review, name the deployment
options together with the model choice. A managed API gives speed. A private/VPC deployment
gives control. Self-hosted open weights give data residency. Then say which option the
requirements select, and why.

## The rest of the stack (worth a sentence each)

- **la Plateforme**: Mistral's API platform and console (keys, usage, fine-tuning).
- **Le Chat**: the assistant product for end users. It is useful as a reference UX, not as
  a build target.
- **Agents / Conversations API**: Mistral's higher-level agent runtime, with built-in
  tools and persistent conversations. This lab builds the loop by hand, so that you
  understand what that API does for you. When the hand-built loop is not sufficient for
  you, look at that API.
- **Structured outputs / JSON mode**: for tools whose results a program uses.

This lab stays at the level of the loop, by intention. The step up to production
(async, parallel tools, MCP, OAuth, evals, tracing) is the
[`gcp-agent-platform-lab`](../../gcp-agent-platform-lab/README.md) next to it in this repo.
The same adapter idea applies there.
