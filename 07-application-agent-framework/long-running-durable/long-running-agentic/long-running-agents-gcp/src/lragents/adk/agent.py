"""Model-backed ADK agent with a long-running tool (for `adk web` / Cloud Run).

Contrast with ``nightly_workflow.py``: here the *model* decides when to join the
queue. ``LongRunningFunctionTool`` tells ADK the tool returns a handle to work
that is still in progress, so with ``ResumabilityConfig`` the invocation pauses
after the call and can be resumed later with a ``FunctionResponse`` carrying the
same function-call id. ``before_tool_callback`` is a code guard that re-verifies
availability before any purchase — a prompt cannot guarantee that.

Run locally:   adk web src/lragents/adk --session_service_uri sqlite+aiosqlite:///sessions.db
Needs: GOOGLE_GENAI_USE_VERTEXAI=1, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from google.adk.agents import Agent
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.tools import LongRunningFunctionTool, ToolContext

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

_QUEUE: dict[str, int] = {}
_AVAILABLE: dict[str, int] = {"ams-sat": 100, "ams-tue": 100}
_ORDERS: dict[str, dict[str, Any]] = {}


def search_events(city: str = "") -> dict:
    """List tour dates, optionally filtered by city."""
    events = [{"id": "ams-tue", "city": "Amsterdam", "weekday": "Tuesday", "price": 90},
              {"id": "ams-sat", "city": "Amsterdam", "weekday": "Saturday", "price": 110}]
    return {"events": [e for e in events if not city or e["city"].lower() == city.lower()]}


def join_queue(event_id: str, tool_context: ToolContext) -> dict:
    """Join the purchase queue for an event. Returns immediately with a ticket; the
    queue keeps moving while the agent is paused. (Long-running.)"""
    ticket = tool_context.state.get("ticket")
    if not ticket:                                        # idempotent: one ticket per session
        ticket = f"q_{uuid.uuid4().hex[:8]}"
        _QUEUE[ticket] = 14203
        tool_context.state["ticket"] = ticket
        tool_context.state["event_id"] = event_id
    return {"ticket": ticket, "position": _QUEUE[ticket], "status": "pending"}


def check_queue(tool_context: ToolContext) -> dict:
    """Current queue position for this session's ticket. (Synchronous — do NOT mark long-running.)"""
    ticket = tool_context.state.get("ticket")
    return {"ticket": ticket, "position": _QUEUE.get(ticket, -1), "ready": _QUEUE.get(ticket, -1) == 0}


def purchase(event_id: str, seats: int, tool_context: ToolContext) -> dict:
    """Buy seats. Carries an idempotency key so a retried call cannot double-charge."""
    key = f"{tool_context.session.id}:{event_id}:purchase" if getattr(tool_context, "session", None) else f"{event_id}:purchase"
    if key in _ORDERS:
        return {**_ORDERS[key], "replayed": True}
    order = {"order_id": f"ord_{uuid.uuid4().hex[:8]}", "event_id": event_id, "seats": seats}
    _ORDERS[key] = order
    _AVAILABLE[event_id] -= seats
    return order


def refresh_before_purchase(tool, args, tool_context):
    """Code guard (before_tool_callback): re-read inventory before a mutating call."""
    if tool.name != "purchase":
        return None
    left = _AVAILABLE.get(args.get("event_id"), 0)
    if left < int(args.get("seats", 1)):
        return {"error": "sold_out", "available": left, "message": "re-fetch availability and pick again"}
    return None


root_agent = Agent(
    name="concert",
    model=MODEL,
    instruction=("You help plan concert tickets. Join the queue as early as possible with join_queue; "
                 "use check_queue to see if it is our turn; only then purchase. Never purchase twice."),
    tools=[search_events, LongRunningFunctionTool(func=join_queue), check_queue, purchase],
    before_tool_callback=refresh_before_purchase,
)

app = App(
    name="concert",
    root_agent=root_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(compaction_interval=10, overlap_size=1),
)
