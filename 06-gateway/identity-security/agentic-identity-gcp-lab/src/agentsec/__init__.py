"""agentsec — identity & security reference implementation for agentic systems on Google Cloud.

Layout (mirrors docs/primer.md):

- identity/   principals (users, agents as SPIFFE identities, principal sets), certificates,
              tokens (local STS: RFC 8693 exchange, cert-bound tokens, DPoP), delegation context,
              credential access boundaries, and an Auth-Manager-compatible credential broker.
- policy/     tool policy model, deny-by-default engine, and the ADK plugin that enforces it.
- guardrails/ prompt/response screening (Model Armor shape) and untrusted-content sanitisation.
- secrets/    secret store abstraction (local + Secret Manager) and log redaction.
- audit/      structured audit events with dual identity (user + agent).
- mcp/        an MCP server that is a proper OAuth 2.1 resource server, and a client factory.
- a2a/        Agent Card security schemes, signing, and inbound verification.
- agents/     the reference ADK agent, its tools, and a scripted LLM for offline runs.

Everything runs offline by default (profile="local"). Set AGENTSEC_PROFILE=gcp to bind to
Google Cloud services (Agent Identity, Auth Manager, Secret Manager, Model Armor).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentsec")
except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
    __version__ = "0.0.0"

__all__ = ["__version__"]
