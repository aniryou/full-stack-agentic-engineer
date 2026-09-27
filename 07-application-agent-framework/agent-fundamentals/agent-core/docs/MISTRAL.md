# Running agent-core on Mistral

The whole lab runs offline with `FakeLLM`. When you want a real model, the only
Mistral-specific code is `agentcore/mistral_llm.py` — a provider adapter of about 150
lines that plugs into the same `Agent`. Notebook `05_going_live_on_mistral` walks
through it; everything in it except the last cell runs offline (T0).

```bash
pip install -e ".[mistral]"         # the optional extra: the mistralai client (2.x)
export MISTRAL_API_KEY=...          # from https://console.mistral.ai; live calls are billed per token
```

Without the extra or the key, `MistralLLM(...)` raises `MistralUnavailable` naming what
is missing; notebook 05 catches it and prints a labelled stop. For tests, pass any
object with `.chat.complete(**kwargs)` as `client=` — no SDK and no key needed
(`tests/test_mistral_adapter.py` does).

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

The client is `from mistralai.client import Mistral; client = Mistral(api_key=...)` in
the 2.x SDK (`from mistralai import Mistral` in 1.x; the adapter tries both), and the
call is `client.chat.complete(model=..., messages=..., tools=..., tool_choice="auto")`.
`MistralLLM.request(messages, tools)` returns exactly those keyword arguments without
sending anything.
`tool_choice` accepts `"auto"`, `"any"` (force a tool), or `"none"`. The SDK surface
moves — check https://docs.mistral.ai if a call fails.

## Choosing a model (list prices 2026-09-19, verify)

Prices are USD per million tokens, input / output, as the repo pins them in
`06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/mistral.py` (dated
2026-09-19); rows marked † are from the provider's pricing page on the same date and are
not pinned by a test. Aliases such as `mistral-large-latest` move to newer releases:
confirm what an alias points at, and the price, before relying on either.

| Model string | Use for | Input / output ($ per 1M) | Weights |
|--------------|---------|---------------------------|---------|
| `mistral-large-latest` (Large 3, 675B MoE, 41B active) | hardest tasks and **agents / tool use**, long context | $0.50 / $1.50 | **open (Apache-2.0)** |
| `mistral-medium-latest` (Medium 3.5, 128B dense) | frontier-class agentic work and coding | $1.50 / $7.50 | published (modified MIT: a commercial licence above $20 M of monthly revenue, verify), so notebook 05's snapshot counts it as not self-hostable |
| `mistral-small-latest` (Small 4, 119B MoE, 6.5B active) | cost-effective general work and tool use | $0.15 / $0.60 | **open (Apache-2.0)** |
| `ministral-14b-2512` / `ministral-8b-2512` / `ministral-3b-2512` | on-device, cheap high-volume | $0.20 / $0.20, $0.15 / $0.15, $0.10 / $0.10 | **open (Apache-2.0)** |
| `magistral-medium-latest` † | step-by-step **reasoning** | $2 / $5 | API |
| `codestral-latest` † | code generation / autocomplete | $0.30 / $0.90 | API |
| `pixtral-latest`, `voxtral-*`, `mistral-ocr-latest` † | vision, audio, document extraction | — | API |

Cached input is billed at 10 % of input; the Batch API is half price (2026-09-19, verify).

Rule of thumb for an agent: the cheapest model that passes your evals for the tool loop
(exercise 5.3 turns that into code), a reasoning model only where reasoning earns its
cost, and an open-weight model you host yourself when the data must not leave your own
environment.

**Deployment posture.** Where the model can run is a design input, not an afterthought.
The open-weight models can run anywhere, so the same agent can call the managed API
(`la Plateforme`), run on weights deployed in your own VPC, or run fully on-prem. When the
data is regulated (banking, health, public sector) or must stay in one jurisdiction, that
choice can decide the model. In a design review, name the deployment options alongside the
model choice — managed API for speed, private/VPC for control, self-hosted open weights for
data residency — and say which one the requirements pick and why.

## The rest of the stack (worth a sentence each)

- **la Plateforme** — Mistral's API platform and console (keys, usage, fine-tuning).
- **Le Chat** — the end-user assistant product; useful as a reference UX, not a build target.
- **Agents / Conversations API** — Mistral's higher-level agent runtime with built-in
  tools and persistent conversations; this lab builds the loop by hand so you understand
  what that API does for you. When you outgrow the hand-built loop, that is where to look.
- **Structured outputs / JSON mode** — for tools whose results a program consumes.

This lab stays deliberately at the level of the loop. The production step-up
(async, parallel tools, MCP, OAuth, evals, tracing) is the
[`gcp-agent-platform-lab`](../../gcp-agent-platform-lab/README.md) next to it in this repo;
the same adapter idea applies there.
