"""``SecurityPlugin`` — the runtime policy enforcement point, as an ADK plugin.

Register it once on the ``Runner`` and it applies to every agent and every tool in the run:

* ``before_model_callback``  screens the user's latest message (Model Armor shape); blocked
  prompts never reach the model — the callback returns an ``LlmResponse`` instead.
* ``after_model_callback``   screens the model's text for sensitive data / malicious URIs.
* ``before_tool_callback``   resolves the authority context (who is acting, for whom, with
  which scopes), evaluates the deny-by-default policy, tracks per-invocation budgets, and
  either denies (returns a dict → the tool never runs), asks for human confirmation
  (``tool_context.request_confirmation`` → ADK emits ``adk_request_confirmation``), or allows.
* ``after_tool_callback``    tags results with provenance and sanitises untrusted text.
* ``on_tool_error_callback`` records failures so they are auditable.

Every step emits a structured :class:`~agentsec.audit.AuditEvent` with the dual identity.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from ..audit.log import AuditLog, args_digest
from ..guardrails.sanitize import Provenance, TrustLevel, wrap_untrusted
from ..guardrails.screening import LocalScreener, Screener
from ..identity.delegation import AuthorityContext, AuthorityMode
from ..identity.principals import AgentIdentity, UserPrincipal
from ..identity.tokens import TokenIssuer
from ..secrets.redaction import redact
from .engine import Decision, Effect, PolicyEngine, ToolCallRequest
from .model import Tier

STATE_USER = "agentsec:user"  # {"subject":..., "email":..., "tenant":...}
STATE_SCOPES = "agentsec:scopes"  # ["tickets:read", ...]
STATE_ACCESS_TOKEN = "agentsec:access_token"  # a delegated token verified by the resolver
COUNTER_TOOL_CALLS = "temp:agentsec_tool_calls"
COUNTER_DESTRUCTIVE = "temp:agentsec_destructive_calls"

AuthorityResolver = Callable[[ReadonlyContext], AuthorityContext]


class SessionAuthorityResolver:
    """Derive the authority context from session state seeded by the (trusted) front-end.

    Two shapes are supported:

    * ``state["agentsec:access_token"]`` — a delegated token (from the STS / Auth Manager),
      verified against ``issuer`` with the expected audience; ``act.sub`` must be this agent.
    * ``state["agentsec:user"]`` + ``state["agentsec:scopes"]`` — the front-end already
      authenticated the user and records subject/email/tenant and granted scopes.

    With neither present the agent runs under its **own** authority with no user scopes.
    """

    def __init__(
        self,
        agent: AgentIdentity,
        *,
        issuer: TokenIssuer | None = None,
        audience: str | None = None,
    ):
        self.agent = agent
        self.issuer = issuer
        self.audience = audience

    def __call__(self, ctx: ReadonlyContext) -> AuthorityContext:
        state = ctx.state
        token = state.get(STATE_ACCESS_TOKEN)
        if token and self.issuer:
            claims = self.issuer.verify(token, audience=self.audience, allow_unbound=True)
            authority = AuthorityContext.from_claims(claims, agent=self.agent)
            if authority.mode is AuthorityMode.DELEGATED and authority.agent != self.agent:
                raise PermissionError("delegated token names a different actor agent")
            return authority
        user = state.get(STATE_USER)
        scopes = set(state.get(STATE_SCOPES, []) or [])
        if user:
            principal = UserPrincipal(
                subject=user.get("subject") or ctx.user_id,
                email=user.get("email"),
                tenant=user.get("tenant"),
                groups=tuple(user.get("groups", ())),
            )
            return AuthorityContext.delegated(self.agent, principal, scopes)
        return AuthorityContext.own(self.agent, scopes)


def _latest_user_text(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents or []):
        if content.role == "user":
            texts = [p.text for p in (content.parts or []) if getattr(p, "text", None)]
            if texts:
                return "\n".join(texts)
            return ""  # a function response, not user text
    return ""


def _response_text(llm_response: LlmResponse) -> str:
    if not llm_response.content or not llm_response.content.parts:
        return ""
    return "\n".join(p.text for p in llm_response.content.parts if getattr(p, "text", None))


class SecurityPlugin(BasePlugin):
    def __init__(
        self,
        *,
        engine: PolicyEngine,
        authority_resolver: AuthorityResolver,
        audit: AuditLog | None = None,
        screener: Screener | None = None,
        screen_prompts: bool = True,
        screen_responses: bool = True,
        name: str = "agentsec_security",
    ):
        super().__init__(name=name)
        self.engine = engine
        self.resolve_authority = authority_resolver
        self.audit = audit or AuditLog()
        self.screener = screener or LocalScreener()
        self.screen_prompts = screen_prompts
        self.screen_responses = screen_responses
        self._started: dict[str, float] = {}
        # Authority is resolved ONCE per invocation, from the session state as it was when the
        # run started, and pinned for the rest of that invocation. Nothing the model or a tool
        # does mid-run (including writing to session state) can change who is acting.
        self._authority_by_invocation: dict[str, AuthorityContext] = {}

    # ---- run boundary ----------------------------------------------------------------
    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        try:
            authority = self.resolve_authority(ReadonlyContext(invocation_context))
        except Exception as e:  # fail closed: a run with unverifiable authority does not start
            self.audit.record(
                event_type="run.authority",
                agent="unknown",
                authority="unknown",
                decision="deny",
                reasons=[f"{type(e).__name__}: {e}"],
                invocation_id=invocation_context.invocation_id,
            )
            return types.Content(
                role="model",
                parts=[
                    types.Part(
                        text="This session's identity context could not be verified; refusing to run."
                    )
                ],
            )
        self._authority_by_invocation[invocation_context.invocation_id] = authority
        return None

    async def after_run_callback(self, *, invocation_context: InvocationContext) -> None:
        self._authority_by_invocation.pop(invocation_context.invocation_id, None)

    # ---- helpers ---------------------------------------------------------------------
    def _authority(self, ctx: ReadonlyContext) -> AuthorityContext:
        pinned = self._authority_by_invocation.get(getattr(ctx, "invocation_id", None) or "")
        if pinned is not None:
            return pinned
        return self.resolve_authority(ctx)

    def _emit(self, ctx: ReadonlyContext, authority: AuthorityContext, **fields: Any) -> None:
        ids = authority.audit_identities()
        session = getattr(ctx, "session", None)
        self.audit.record(
            agent=ids["agent"],
            user=ids["user"],
            authority=ids["authority"],
            invocation_id=getattr(ctx, "invocation_id", None),
            session_id=getattr(session, "id", None),
            **fields,
        )

    # ---- model boundary ----------------------------------------------------------------
    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        if not self.screen_prompts:
            return None
        text = _latest_user_text(llm_request)
        if not text:
            return None
        result = self.screener.screen_prompt(text)
        authority = self._authority(callback_context)
        self._emit(
            callback_context,
            authority,
            event_type="model.screen",
            decision="blocked" if result.blocked else ("flagged" if result.matched else "ok"),
            reasons=[f.detail for f in result.findings],
            extra={"direction": "prompt", "findings": result.summary()},
        )
        if result.blocked:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=f"I can't process that request: it was blocked by input screening ({result.summary()})."
                        )
                    ],
                ),
                custom_metadata={"agentsec": {"blocked": True, "stage": "prompt"}},
            )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if not self.screen_responses or llm_response.partial:
            return None
        text = _response_text(llm_response)
        if not text:
            return None
        result = self.screener.screen_response(text)
        if not result.matched:
            return None
        authority = self._authority(callback_context)
        self._emit(
            callback_context,
            authority,
            event_type="model.screen",
            decision="blocked" if result.blocked else "flagged",
            reasons=[f.detail for f in result.findings],
            extra={"direction": "response", "findings": result.summary()},
        )
        if result.blocked:
            return LlmResponse(
                content=types.Content(
                    role="model", parts=[types.Part(text="[response withheld by output screening]")]
                ),
                custom_metadata={
                    "agentsec": {"blocked": True, "stage": "response", "findings": result.summary()}
                },
            )
        return None

    # ---- tool boundary ------------------------------------------------------------------
    async def before_tool_callback(
        self, *, tool: BaseTool, tool_args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        authority = self._authority(tool_context)
        counters = {
            "tool_calls": int(tool_context.state.get(COUNTER_TOOL_CALLS, 0) or 0),
            "destructive_calls": int(tool_context.state.get(COUNTER_DESTRUCTIVE, 0) or 0),
        }
        confirmed_by = None
        confirmation = tool_context.tool_confirmation
        if confirmation is not None:
            if confirmation.confirmed:
                confirmed_by = (
                    (authority.user.email or authority.user.subject)
                    if authority.user
                    else tool_context.user_id
                )
            else:
                decision = Decision(
                    Effect.DENY,
                    ["human rejected the confirmation"],
                    self.engine.tool_policy(tool.name),
                )
                return self._deny(tool, tool_args, tool_context, authority, decision)

        req = ToolCallRequest(
            tool=tool.name,
            args=tool_args,
            authority=authority,
            invocation_id=tool_context.invocation_id,
            counters=counters,
            confirmed_by=confirmed_by,
        )
        decision = self.engine.evaluate(req)

        if decision.effect is Effect.DENY:
            return self._deny(tool, tool_args, tool_context, authority, decision)

        if decision.effect is Effect.CONFIRM:
            payload = {
                "tool": tool.name,
                "args": redact(tool_args),
                "authority": authority.audit_identities(),
                "tier": decision.tool_policy.tier.value if decision.tool_policy else None,
                "reasons": decision.reasons,
            }
            tool_context.request_confirmation(hint=decision.hint, payload=payload)
            self._emit(
                tool_context,
                authority,
                event_type="tool.decision",
                tool=tool.name,
                decision="confirm",
                reasons=decision.reasons,
                args_hash=args_digest(tool_args),
                args_redacted=redact(tool_args),
            )
            return {
                "status": "confirmation_required",
                "tool": tool.name,
                "hint": decision.hint,
                "payload": payload,
            }

        # ALLOW: account budgets and record the decision; the tool now runs.
        tool_context.state[COUNTER_TOOL_CALLS] = counters["tool_calls"] + 1
        if decision.tool_policy and decision.tool_policy.tier is Tier.DESTRUCTIVE:
            tool_context.state[COUNTER_DESTRUCTIVE] = counters["destructive_calls"] + 1
        self._started[tool_context.function_call_id or tool.name] = time.perf_counter()
        self._emit(
            tool_context,
            authority,
            event_type="tool.decision",
            tool=tool.name,
            decision="allow",
            reasons=decision.reasons,
            approver=confirmed_by,
            args_hash=args_digest(tool_args),
            args_redacted=redact(tool_args),
        )
        return None

    def _deny(
        self,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        authority: AuthorityContext,
        decision: Decision,
    ) -> dict[str, Any]:
        self._emit(
            tool_context,
            authority,
            event_type="tool.decision",
            tool=tool.name,
            decision="deny",
            reasons=decision.reasons,
            args_hash=args_digest(tool_args),
            args_redacted=redact(tool_args),
        )
        return {"error": "policy_denied", "tool": tool.name, "reasons": decision.reasons}

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        authority = self._authority(tool_context)
        started = self._started.pop(tool_context.function_call_id or tool.name, None)
        latency = (time.perf_counter() - started) * 1000 if started else None
        tp = self.engine.tool_policy(tool.name)
        trust = TrustLevel.INTERNAL if tp and tp.trust == "internal" else TrustLevel.EXTERNAL
        provenance = Provenance(
            source=f"tool:{tool.name}", trust=trust, retrieved_by=authority.agent.spiffe_id
        )

        wrapped: dict[str, Any] | None = None
        if (
            isinstance(result, dict)
            and "error" not in result
            and result.get("status") != "confirmation_required"
        ):
            for key in ("content", "text", "body", "result"):
                value = result.get(key)
                if isinstance(value, str) and value:
                    wrapped = dict(result)
                    wrapped[key] = wrap_untrusted(value, provenance)
                    wrapped["provenance"] = provenance.tag()
                    break
                if key == "content" and isinstance(value, list):
                    # MCP CallToolResult shape: content blocks; wrap every text block.
                    blocks = []
                    changed = False
                    for block in value:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        ):
                            blocks.append(
                                {**block, "text": wrap_untrusted(block["text"], provenance)}
                            )
                            changed = True
                        else:
                            blocks.append(block)
                    if changed:
                        wrapped = dict(result)
                        wrapped["content"] = blocks
                        wrapped["provenance"] = provenance.tag()
                    break
        if isinstance(result, dict) and result.get("status") == "confirmation_required":
            outcome = "pending"
        elif isinstance(result, dict) and (
            "error" in result
            or result.get("isError")
            or (
                isinstance(result.get("structuredContent"), dict)
                and "error" in result["structuredContent"]
            )
        ):
            outcome = "error"
        else:
            outcome = "ok"
        self._emit(
            tool_context,
            authority,
            event_type="tool.result",
            tool=tool.name,
            decision=outcome,
            result_hash=args_digest(result),
            provenance=[provenance.tag()],
            latency_ms=latency,
        )
        return wrapped

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        authority = self._authority(tool_context)
        self._emit(
            tool_context,
            authority,
            event_type="tool.error",
            tool=tool.name,
            decision="error",
            reasons=[f"{type(error).__name__}: {error}"],
            args_hash=args_digest(tool_args),
        )
        return {"error": "tool_failed", "tool": tool.name, "detail": type(error).__name__}
