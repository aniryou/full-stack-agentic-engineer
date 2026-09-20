"""Helpers to drive an ADK ``Runner`` turn-by-turn, including confirmation and auth round-trips.

The front-end's responsibilities are modelled here explicitly because they are part of the
security design:

* it authenticates the user and seeds the session with the user's identity and granted scopes
  (or a delegated token) — ``seed_session``;
* it renders ``adk_request_confirmation`` with the *actual* tool call (not the model's prose)
  and sends back a ``ToolConfirmation`` — ``confirm``;
* it handles ``adk_request_credential`` by redirecting the user to the consent URI, finalising
  the consent with the broker, and resuming — ``resume_after_auth``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types

from .policy.adk_plugin import STATE_ACCESS_TOKEN, STATE_SCOPES, STATE_USER

REQUEST_CONFIRMATION = "adk_request_confirmation"
REQUEST_CREDENTIAL = "adk_request_credential"


@dataclass
class PendingConfirmation:
    request_id: str  # id of the adk_request_confirmation function call
    original_call: dict[str, Any]  # {"name":..., "args":..., "id":...}
    hint: str | None
    payload: Any


@dataclass
class PendingAuth:
    request_id: str
    auth_config: dict[
        str, Any
    ]  # serialised AuthConfig (contains exchanged_auth_credential.oauth2.auth_uri)

    @property
    def auth_uri(self) -> str | None:
        return self.auth_config.get("exchangedAuthCredential", {}).get("oauth2", {}).get(
            "authUri"
        ) or self.auth_config.get("exchanged_auth_credential", {}).get("oauth2", {}).get("auth_uri")

    @property
    def consent_nonce(self) -> str | None:
        return self.auth_config.get("exchangedAuthCredential", {}).get("oauth2", {}).get(
            "nonce"
        ) or self.auth_config.get("exchanged_auth_credential", {}).get("oauth2", {}).get("nonce")


@dataclass
class TurnResult:
    events: list[Event] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_responses: list[dict[str, Any]] = field(default_factory=list)
    pending_confirmations: list[PendingConfirmation] = field(default_factory=list)
    pending_auth: list[PendingAuth] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return self.texts[-1] if self.texts else ""

    def summary(self) -> str:
        lines = []
        for c in self.tool_calls:
            lines.append(f"→ call {c['name']}({c['args']})")
        for r in self.tool_responses:
            lines.append(f"← {r['name']}: {r['response']}")
        for p in self.pending_confirmations:
            lines.append(
                f"⏸ confirmation requested for {p.original_call.get('name')} {p.original_call.get('args')} — {p.hint}"
            )
        for a in self.pending_auth:
            lines.append(f"⏸ credential requested — consent at {a.auth_uri}")
        if self.final_text:
            lines.append(f"model: {self.final_text}")
        return "\n".join(lines)


async def seed_session(
    runner: Runner,
    *,
    user_id: str,
    session_id: str,
    user: dict[str, Any] | None = None,
    scopes: list[str] | None = None,
    access_token: str | None = None,
) -> None:
    """Create the session with the front-end-verified identity context."""
    state: dict[str, Any] = {}
    if user:
        state[STATE_USER] = user
    if scopes is not None:
        state[STATE_SCOPES] = list(scopes)
    if access_token:
        state[STATE_ACCESS_TOKEN] = access_token
    await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id, state=state
    )


def _collect(events: list[Event]) -> TurnResult:
    result = TurnResult(events=events)
    for ev in events:
        if not ev.content or not ev.content.parts:
            continue
        for part in ev.content.parts:
            if part.text and ev.author != "user":
                result.texts.append(part.text)
            if part.function_call is not None:
                fc = part.function_call
                if fc.name == REQUEST_CONFIRMATION:
                    args = fc.args or {}
                    tc = args.get("toolConfirmation", {})
                    result.pending_confirmations.append(
                        PendingConfirmation(
                            request_id=fc.id,
                            original_call=args.get("originalFunctionCall", {}),
                            hint=tc.get("hint"),
                            payload=tc.get("payload"),
                        )
                    )
                elif fc.name == REQUEST_CREDENTIAL:
                    args = fc.args or {}
                    result.pending_auth.append(
                        PendingAuth(
                            request_id=fc.id,
                            auth_config=args.get("authConfig", args.get("auth_config", {})),
                        )
                    )
                else:
                    result.tool_calls.append(
                        {"name": fc.name, "args": dict(fc.args or {}), "id": fc.id}
                    )
            if part.function_response is not None:
                fr = part.function_response
                result.tool_responses.append(
                    {"name": fr.name, "response": fr.response, "id": fr.id}
                )
    return result


async def run_turn(
    runner: Runner, *, user_id: str, session_id: str, message: str | types.Content
) -> TurnResult:
    content = (
        message
        if isinstance(message, types.Content)
        else types.Content(role="user", parts=[types.Part(text=message)])
    )
    events: list[Event] = []
    async for ev in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        events.append(ev)
    return _collect(events)


async def confirm(
    runner: Runner,
    *,
    user_id: str,
    session_id: str,
    pending: PendingConfirmation,
    confirmed: bool,
    payload: Any = None,
) -> TurnResult:
    """Answer an ``adk_request_confirmation``; ADK re-runs the original tool with the decision."""
    tc = ToolConfirmation(
        hint=pending.hint,
        confirmed=confirmed,
        payload=payload if payload is not None else pending.payload,
    )
    content = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=pending.request_id,
                    name=REQUEST_CONFIRMATION,
                    response=tc.model_dump(by_alias=True, exclude_none=True),
                )
            )
        ],
    )
    return await run_turn(runner, user_id=user_id, session_id=session_id, message=content)


async def resume_after_auth(
    runner: Runner, *, user_id: str, session_id: str, pending: PendingAuth
) -> TurnResult:
    """Resume after the user completed consent; the broker now returns the credential."""
    content = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=pending.request_id, name=REQUEST_CREDENTIAL, response=pending.auth_config
                )
            )
        ],
    )
    return await run_turn(runner, user_id=user_id, session_id=session_id, message=content)
