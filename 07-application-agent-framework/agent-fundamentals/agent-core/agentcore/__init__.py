"""agentcore — the smallest honest agent: a fake model, a tool, and the loop.

    from agentcore import Agent, tool, FakeLLM, call, calls, text

Three files, ~200 lines, pure standard library, synchronous. Read them in this
order: fake_llm.py, tools.py, agent.py. When you want the production version
(async, parallel tools, MCP, OAuth, evals, tracing), that is the separate
`gcp-agent-platform-lab` repository — this is the concept it is built on.
"""
from .agent import Agent, Result
from .fake_llm import FakeLLM, Response, ToolCall, call, calls, text
from .tools import Tool, ToolError, tool, tool_message

__all__ = [
    "Agent", "Result",
    "FakeLLM", "Response", "ToolCall", "call", "calls", "text",
    "Tool", "ToolError", "tool", "tool_message",
]
