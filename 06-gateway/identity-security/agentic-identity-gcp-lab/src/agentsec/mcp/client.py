"""ADK-side MCP client wiring: the right credential, for the right audience, per request.

Two ways to attach credentials to an ``McpToolset``:

1. **Broker-managed (production on Google Cloud)** — ``auth_scheme=GcpAuthProviderScheme(...)``
   and a registered ``GcpAuthProvider``: ADK asks Auth Manager for the user's 3-legged token
   (or the agent's 2-legged/API-key credential) on each call and injects it. Locally the same
   scheme is served by :func:`agentsec.identity.make_local_gcp_auth_provider`.

2. **Explicit token exchange (this module)** — a ``header_provider`` that mints a token *for
   this MCP server's audience* from the session's delegated authority, via the STS. This is the
   pattern you use when the tool server is your own resource server and you want RFC 8707
   audience binding and scope narrowing per toolset.

Both keep raw credentials out of prompts and session state: the token is produced at call time
from a broker/STS and handed straight to the transport.
"""

from __future__ import annotations

from collections.abc import Callable

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams

from ..identity.principals import AgentIdentity
from ..identity.tokens import TokenIssuer
from ..policy.adk_plugin import STATE_ACCESS_TOKEN, STATE_SCOPES, STATE_USER

TokenMinter = Callable[[ReadonlyContext], str | None]


def delegated_token_minter(
    issuer: TokenIssuer,
    *,
    agent: AgentIdentity,
    audience: str,
    scopes: list[str],
    front_end_audience: str = "https://app.acme.example",
) -> TokenMinter:
    """Mint a delegated, audience-bound token for the MCP server from the session's authority.

    If the session carries a delegated access token, it is *exchanged* (RFC 8693) for one
    narrowed to ``audience`` and ``scopes`` — never forwarded. If it carries only the
    front-end-verified user record, a subject token is minted for that user first (what the
    front-end's IdP would have produced), then exchanged with the agent as actor.
    """

    def mint(ctx: ReadonlyContext) -> str | None:
        state = ctx.state
        subject_token = state.get(STATE_ACCESS_TOKEN)
        granted = set(state.get(STATE_SCOPES, []) or [])
        requested = [s for s in scopes if s in granted] if granted else []
        if not subject_token:
            user = state.get(STATE_USER)
            if not user:
                return None  # own authority: caller should use an agent token instead
            subject_token = issuer.mint(
                subject=user.get("subject") or ctx.user_id,
                audience=front_end_audience,
                scope=sorted(granted),
                ttl=120,
                extra={k: v for k, v in user.items() if k in {"email", "tenant", "groups"}},
            )
        actor_token = issuer.mint(
            subject=agent.spiffe_id, audience=issuer.issuer, ttl=60, extra={"authority": "own"}
        )
        exchanged = issuer.exchange(
            subject_token=subject_token,
            actor_token=actor_token,
            actor_token_audience=issuer.issuer,
            audience=audience,
            scope=requested,
            ttl=120,
        )
        return exchanged["access_token"]

    return mint


def make_mcp_toolset(
    url: str,
    *,
    token_minter: TokenMinter,
    tool_filter: list[str] | None = None,
    tool_name_prefix: str | None = "tickets",
    timeout: float = 10.0,
) -> McpToolset:
    """An ``McpToolset`` whose Authorization header is minted per request from session authority."""

    async def header_provider(ctx: ReadonlyContext) -> dict[str, str]:
        token = token_minter(ctx)
        return {"Authorization": f"Bearer {token}"} if token else {}

    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(url=url, timeout=timeout),
        header_provider=header_provider,
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )
