# Using a real Gemini model

Everything in the lab runs against `agentlab.llm.FakeLLM`. When you want to see real model behaviour,
`agentlab.llm.gemini.GeminiLLM` implements the same `LLM` protocol (`generate` and `stream`, tool
schemas in, `ModelResponse` out), so any notebook cell that builds an agent can swap the model:

```python
from agentlab.llm.gemini import GeminiLLM
llm = GeminiLLM(model="gemini-3-flash")          # GOOGLE_API_KEY in the environment, or Vertex env vars
agent = LlmAgent("assistant", llm, "You are a bank assistant.", tools=[get_balance])
```

Install the extra first: `pip install -e ".[gemini]"`.

## Verify before relying on it

The adapter follows the `google-genai` 1.x surface as documented in mid-2026:

- `genai.Client()` picks up `GOOGLE_API_KEY`, or `GOOGLE_GENAI_USE_VERTEXAI=true` with
  `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` for Vertex.
- `client.aio.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(...))`.
- Tools are passed as `types.Tool(function_declarations=[types.FunctionDeclaration(name, description, parameters)])`;
  the SDK's automatic function calling is disabled because the lab's loop executes tools itself.
- Function calls come back as `part.function_call` (name, args); results are sent back as
  `Part.from_function_response(name, response)` in a user turn.
- Usage comes from `response.usage_metadata` (`prompt_token_count`, `candidates_token_count`,
  `cached_content_token_count`, `thoughts_token_count`).

Field names move between SDK versions. If a call fails, compare the adapter with the current SDK
reference (https://googleapis.github.io/python-genai/) and adjust; the rest of the lab does not
depend on this file.

## What changes when the model is real

- **Non-determinism.** The notebooks' checks assume scripted behaviour. With a real model, run the
  evaluation notebooks (08) with `n_runs ≥ 3` and reason about pass rates, not single passes.
- **Thinking models and function calling.** Gemini 3.x thinking models return *thought signatures*
  alongside function calls that must be round-tripped in multi-step flows; the adapter passes
  through what the SDK returns, but if you build your own message conversion, keep those parts.
- **Cost.** Attach `observability.Tracer` and a `PriceTable` with the current prices before running
  anything at volume; the lab's default prices are illustrative.
- **Safety.** The indirect-injection demo in notebook 11 uses a deliberately obedient fake. A real
  model is harder to inject but not immune; the defences (allowlists, provenance-tagged data blocks,
  screening, confirmation for irreversible actions) are what actually hold.

## Going further: ADK

Once the lab's concepts are comfortable, rebuild one notebook on Google's Agent Development Kit —
`docs/LAB_TO_ADK.md` maps the names one to one and lists the seven steps.
