"""Run the ADK workflow locally: start -> interrupt at review -> resume.

Uses ``DatabaseSessionService`` (SQLite) so you can kill the process after
the interrupt and resume from a fresh one — the same shape as production
with ``VertexAiSessionService`` on Agent Engine.

    python examples/adk_agent_engine/run_local.py            # start, pause at review
    python examples/adk_agent_engine/run_local.py approve    # resume the paused invocation
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions.database_session_service import DatabaseSessionService  # noqa: E402
from google.genai import types  # noqa: E402

from workflow_agent import app, review_response, state_summary  # noqa: E402

DB = Path(__file__).resolve().parent / "sessions.db"
BOOKMARK = Path(__file__).resolve().parent / ".paused.json"
USER = "anil"


async def start(goal: str) -> None:
    svc = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{DB}")
    runner = Runner(app=app, session_service=svc)
    session = await svc.create_session(app_name=app.name, user_id=USER)
    invocation_id = None
    async for ev in runner.run_async(
        user_id=USER, session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text=goal)])
    ):
        invocation_id = ev.invocation_id
        if ev.long_running_tool_ids:
            print(f"⏸  paused for human review (interrupt ids={sorted(ev.long_running_tool_ids)})")
    session = await svc.get_session(app_name=app.name, user_id=USER, session_id=session.id)
    print("state:", state_summary(dict(session.state)))
    BOOKMARK.write_text(json.dumps({"session_id": session.id, "invocation_id": invocation_id}))
    print(f"bookmark written to {BOOKMARK.name}; run again with 'approve' or 'reject' to resume")


async def resume(decision: str) -> None:
    bm = json.loads(BOOKMARK.read_text())
    svc = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{DB}")
    runner = Runner(app=app, session_service=svc)
    async for ev in runner.run_async(
        user_id=USER,
        session_id=bm["session_id"],
        invocation_id=bm["invocation_id"],           # continue the *paused* invocation
        new_message=review_response(decision, by="reviewer@example.com"),
    ):
        if ev.actions and ev.actions.state_delta:
            print("state delta:", ev.actions.state_delta)
    session = await svc.get_session(app_name=app.name, user_id=USER, session_id=bm["session_id"])
    print("final state:", state_summary(dict(session.state)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"approve", "reject"}:
        asyncio.run(resume(sys.argv[1]))
    else:
        asyncio.run(start(" ".join(sys.argv[1:]) or "durable execution for AI agents"))
