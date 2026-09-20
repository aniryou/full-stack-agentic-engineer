# Running on Mistral

The whole lab runs offline with `FakeLLM`. When you want a real model, the only
Mistral-specific code is `agentcore/mistral_llm.py` — a ~90-line adapter that plugs
into the same `Agent`.

```bash
pip install mistralai
export MISTRAL_API_KEY=...          # from https://console.mistral.ai
```

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

That is the same `Agent` from notebooks 01–04 — only the model object changed.

## What the adapter translates

Mistral's chat API follows the familiar function-calling shape, so the adapter is small:

| Direction | Our shape | Mistral shape |
|-----------|-----------|---------------|
| tools out | `{"name","description","parameters"}` | `{"type":"function","function":{...}}` |
| assistant tool call out | `args` as a **dict** | `function.arguments` as a **JSON string** |
| reply in | `Response(text, tool_calls)` | `response.choices[0].message.content` / `.tool_calls` |

The client is `from mistralai import Mistral; client = Mistral(api_key=...)`, and the
call is `client.chat.complete(model=..., messages=..., tools=..., tool_choice="auto")`.
`tool_choice` accepts `"auto"`, `"any"` (force a tool), or `"none"`. The SDK surface
moves — check https://docs.mistral.ai if a call fails.

## Choosing a model (September 2026 — verify before quoting)

Prices are per million tokens and illustrative; confirm on the pricing page.

| Model string | Use for | ~Input / Output | Context | Deployment |
|--------------|---------|-----------------|---------|------------|
| `mistral-large-latest` | flagship; hardest tasks and **agents / tool use** | $2 / $6 | 128K | API |
| `mistral-medium-latest` | frontier-class agentic + coding, cheaper | $1.50 / $7.50 | 256K | API |
| `mistral-small-latest` | cost-effective general work | $0.15 / $0.60 | 128K | **open-weight (Apache-2.0)** |
| `magistral-medium-latest` | step-by-step **reasoning** | $2 / $5 | 128K | API |
| `codestral-latest` | code generation / autocomplete | $0.30 / $0.90 | 256K | API |
| `ministral-8b-latest` / `ministral-3b-latest` | on-device, cheap high-volume | ~$0.04 / $0.10 (3B) | 128K | open-weight options |
| `open-mistral-nemo` | budget open workhorse | ~$0.02 / $0.04 | 128K | **open-weight (Apache-2.0)** |
| `pixtral-latest` | vision / document understanding | — | — | API |
| `voxtral-*`, `mistral-ocr-latest` | audio, document extraction | — | — | API |

Rule of thumb for an agent: `mistral-large` for the hard multi-step tool loop, route
routine turns to `mistral-small`, reach for `magistral` only where reasoning earns its
cost, and offer a self-hosted `small` / `ministral` when data cannot leave the
customer's environment.

**Deployment posture.** Mistral's differentiator is not only model quality but where the model can run. The
small models are open-weight, so the same agent can run on `la Plateforme`, in a
customer's VPC, or fully on-prem. For regulated APAC accounts (banking, public sector,
telco) that data-sovereignty story is often the reason Mistral is chosen. When you design an
agent, name the deployment options alongside the model choice: managed
API for speed, private/VPC for control, self-hosted open weights for sovereignty — and
say which you would pick for a given customer and why.

## The rest of the stack (worth a sentence each)

- **la Plateforme** — Mistral's API platform and console (keys, usage, fine-tuning).
- **Le Chat** — the end-user assistant product; useful as a reference UX, not a build target.
- **Agents / Conversations API** — Mistral's higher-level agent runtime with built-in
  tools and persistent conversations; this lab builds the loop by hand so you understand
  what that API does for you. When you outgrow the hand-built loop, that is where to look.
- **Structured outputs / JSON mode** — for tools whose results a program consumes.

This lab stays deliberately at the level of the loop. The production step-up
(async, parallel tools, MCP, OAuth, evals, tracing) is the `gcp-agent-platform-lab` repo;
the same adapter idea applies there.
