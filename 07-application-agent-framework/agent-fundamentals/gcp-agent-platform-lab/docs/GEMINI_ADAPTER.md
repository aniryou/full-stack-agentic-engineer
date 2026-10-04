# Using a real Gemini model

Everything in the lab runs against `agentlab.llm.FakeLLM`. When you want to see the behaviour of a real
model, use `agentlab.llm.gemini.GeminiLLM`. It implements the same `LLM` protocol (`generate` and `stream`,
tool schemas in, `ModelResponse` out). Thus any notebook cell that builds an agent can replace the model:

```python
from agentlab.llm.gemini import GeminiLLM
llm = GeminiLLM(model="gemini-3-flash")          # model name as of September 2026 (verify); GOOGLE_API_KEY or Vertex env vars
agent = LlmAgent("assistant", llm, "You are a bank assistant.", tools=[get_balance])
```

Install the extra first: `pip install -e ".[gemini]"`.

## Verify before relying on it

The author wrote the adapter against the `google-genai` 1.x surface, as the documentation of mid-2026
described it. The 2.x line of the SDK came onto PyPI on 2026-05-07 (2.25.0 on 2026-09-22). The `[gemini]`
extra sets no upper limit on the version. Thus, compare these calls with the version that you install
(verify):

- `genai.Client()` reads `GOOGLE_API_KEY`. For Vertex, it reads `GOOGLE_GENAI_USE_VERTEXAI=true` with
  `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`.
- `client.aio.models.generate_content(model=..., contents=..., config=types.GenerateContentConfig(...))`.
- The adapter passes the tools as `types.Tool(function_declarations=[types.FunctionDeclaration(name, description, parameters)])`.
  It disables the automatic function calling of the SDK, because the loop of the lab executes the tools itself.
- Function calls come back as `part.function_call` (name, args). The adapter sends the results back as
  `Part.from_function_response(name, response)` in a user turn.
- Usage comes from `response.usage_metadata` (`prompt_token_count`, `candidates_token_count`,
  `cached_content_token_count`, `thoughts_token_count`).

Field names change between SDK versions. If a call fails, compare the adapter with the current SDK
reference (https://googleapis.github.io/python-genai/). Then adjust the adapter. The rest of the lab
does not depend on this file.

## What changes when the model is real

- **Non-determinism.** The checks of the notebooks assume scripted behaviour. With a real model, run
  the evaluation notebooks (08) with `n_runs ≥ 3`. Then think about pass rates, not single passes.
- **Thinking models and function calling.** Gemini 3.x thinking models return *thought signatures*
  together with function calls. In multi-step flows, the next request must send these signatures back to the model. The
  adapter passes through what the SDK returns. But if you build your own message conversion, keep
  those parts.
- **Cost.** Before you run anything at volume, attach `observability.Tracer` and a `PriceTable` with
  the current prices. The default prices of the lab are illustrative.
- **Safety.** The indirect-injection demo in notebook 11 uses a fake that obeys, by intention. A real
  model is more difficult to inject, but it is not immune. The defences are the things that actually
  hold: allowlists, provenance-tagged data blocks, screening and confirmation for irreversible actions.

## Going further: ADK

When you are comfortable with the concepts of the lab, rebuild one notebook on Google's Agent
Development Kit. `docs/LAB_TO_ADK.md` gives a one-to-one mapping of the names and lists the seven steps.
