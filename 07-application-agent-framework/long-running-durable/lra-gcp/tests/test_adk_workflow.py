"""ADK 2 Workflow example: pause at the review interrupt, resume, publish. Skipped without google-adk."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

adk = pytest.importorskip("google.adk")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "adk_agent_engine"))


def test_adk_workflow_interrupts_and_resumes() -> None:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from workflow_agent import REVIEW_CALL_ID, app, review_response

    svc = InMemorySessionService()
    runner = Runner(app=app, session_service=svc)

    async def scenario():
        s = await svc.create_session(app_name=app.name, user_id="u")
        inv, interrupts = None, set()
        async for ev in runner.run_async(user_id="u", session_id=s.id, new_message=types.Content(role="user", parts=[types.Part(text="durable agents")])):
            inv = ev.invocation_id
            interrupts |= set(ev.long_running_tool_ids or [])
        paused = await svc.get_session(app_name=app.name, user_id="u", session_id=s.id)
        assert interrupts == {REVIEW_CALL_ID}
        assert paused.state["iteration"] == 1 and paused.state["exit_reason"] == "threshold" and "published" not in paused.state

        async for _ in runner.run_async(user_id="u", session_id=s.id, invocation_id=inv, new_message=review_response("approve", by="qa")):
            pass
        done = await svc.get_session(app_name=app.name, user_id="u", session_id=s.id)
        assert done.state["published"] is True and done.state["post_id"].startswith("post_")

    asyncio.run(scenario())
