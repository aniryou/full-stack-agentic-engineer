"""agentcore — the smallest honest agent: a fake model, a tool, and the loop.

    from agentcore import Agent, tool, FakeLLM, call, calls, text

Three core files, ~200 lines, pure standard library, synchronous. Read them in
this order: fake_llm.py, tools.py, agent.py. The loop is provider-agnostic — it
only needs something with ``.generate(messages, tools) -> Response``.

To run against a real Mistral model, use the optional adapter (needs
``pip install mistralai`` and ``MISTRAL_API_KEY``):

    from agentcore.mistral_llm import MistralLLM
    agent = Agent(MistralLLM(model="mistral-large-latest"), tools=[...])

When you want the production version (async, parallel tools, MCP, OAuth, evals,
tracing), that is the separate `gcp-agent-platform-lab` repository — this is the
concept it is built on.
"""
from .agent import Agent, Result
from .fake_llm import FakeLLM, Response, ToolCall, call, calls, text
from .tools import Tool, ToolError, tool, tool_message

__all__ = [
    "Agent", "Result",
    "FakeLLM", "Response", "ToolCall", "call", "calls", "text",
    "Tool", "ToolError", "tool", "tool_message",
]

# MistralLLM is intentionally not imported here: keeping it out means the core
# package never depends on the `mistralai` SDK. Import it explicitly when needed:
#     from agentcore.mistral_llm import MistralLLM
