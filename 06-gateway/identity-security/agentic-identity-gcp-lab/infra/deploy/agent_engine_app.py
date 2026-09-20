"""Agent Engine entrypoint for the agentsec support agent.

Two deployment paths import this module:

* **Python SDK** (``infra/scripts/deploy_agent_engine.py``) — ``app`` is cloudpickled on your
  machine and unpickled inside Agent Engine. Module-level side effects (registering the Auth
  Manager provider with ADK) therefore also run in :meth:`AgentsecApp.set_up`, which the
  runtime calls after unpickling.
* **Terraform source-based deploy** (``infra/terraform/reasoning_engine.tf``,
  ``entrypoint_module = "deploy.agent_engine_app"``, ``entrypoint_object = "app"``) — the
  runtime imports this module directly; the import-time bootstrap is enough.

What is wired here, and where the primer explains it:

* ``identity_type = AGENT_IDENTITY`` is a *deployment* property (Terraform / deploy script),
  not code — the runtime issues the SPIFFE certificate; the code never sees a key (§3.3).
* ``CredentialManager.register_auth_provider(GcpAuthProvider())`` lets ADK fetch
  user-delegated tokens from Auth Manager for tools declared with ``GcpAuthProviderScheme``
  (§3.2, §3.5). The CRM tool from ``agentsec.agents.root_agent`` is such a tool.
* ``SecurityPlugin`` is the runtime policy enforcement point (§4.1): deny-by-default tool
  policy, argument envelopes, confirmation, Model Armor screening, dual-identity audit.

Environment (set by Terraform ``local.agent_env`` / the deploy script):
  AGENTSEC_PROFILE=gcp, AGENTSEC_PROJECT_ID, AGENTSEC_PROJECT_NUMBER, AGENTSEC_ORG_ID,
  AGENTSEC_LOCATION, AGENTSEC_MODEL, AGENTSEC_AUTH_PROVIDER (full resource name or short ID),
  AGENTSEC_AUTH_PROVIDER_LOCATION, AGENTSEC_MODEL_ARMOR_TEMPLATE, AGENTSEC_MCP_URL,
  AGENTSEC_AUDIT_LOG, AGENTSEC_POLICY, AGENTSEC_AGENT_ENGINE_ID (known only after deploy),
  AGENTSEC_CONTINUE_URI.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("agentsec.deploy")

# --------------------------------------------------------------------------------------------
# Paths: work both from the repo (infra/deploy/agent_engine_app.py, package under src/) and
# from the flattened source archive built by build_agent_source.sh (deploy/, agentsec/,
# policies/ side by side).
# --------------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ARCHIVE_ROOT = HERE.parent
REPO_ROOT = ARCHIVE_ROOT.parent

for candidate in (ARCHIVE_ROOT, REPO_ROOT / "src"):
    if (candidate / "agentsec").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _resolve_policy_path() -> Path:
    raw = os.environ.get("AGENTSEC_POLICY", "policies/support-agent.yaml")
    path = Path(raw)
    if path.is_absolute():
        return path
    for base in (ARCHIVE_ROOT, REPO_ROOT, Path.cwd()):
        if (base / path).exists():
            return base / path
    return ARCHIVE_ROOT / path


def _auth_provider_name() -> str:
    """Accept either the full resource name or the short ID (Settings' convention)."""
    raw = os.environ["AGENTSEC_AUTH_PROVIDER"]
    if "/" in raw:
        return raw
    project = os.environ.get("AGENTSEC_PROJECT_ID", "")
    location = os.environ.get(
        "AGENTSEC_AUTH_PROVIDER_LOCATION", os.environ.get("AGENTSEC_LOCATION", "global")
    )
    return f"projects/{project}/locations/{location}/authProviders/{raw}"


_BOOTSTRAPPED = False


def _bootstrap_runtime() -> None:
    """Idempotent, runs at import time and again inside the runtime (``set_up``)."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    os.environ.setdefault("AGENTSEC_PROFILE", "gcp")
    os.environ["AGENTSEC_POLICY"] = str(_resolve_policy_path())

    # ADK <-> Auth Manager: the class-level registry lives in *this* process, so it must be
    # (re)done wherever the agent actually runs.
    from google.adk.auth.credential_manager import CredentialManager
    from google.adk.integrations.agent_identity import GcpAuthProvider

    CredentialManager.register_auth_provider(GcpAuthProvider())
    _BOOTSTRAPPED = True


_bootstrap_runtime()

from agentsec.agents.root_agent import build_support_agent, make_crm_lookup_tool  # noqa: E402
from agentsec.audit.log import AuditLog, CloudLoggingSink, JsonLinesSink  # noqa: E402
from agentsec.config import Settings  # noqa: E402
from agentsec.guardrails.screening import LocalScreener, ModelArmorScreener  # noqa: E402
from agentsec.identity.principals import AgentIdentity  # noqa: E402
from agentsec.policy.adk_plugin import SecurityPlugin, SessionAuthorityResolver  # noqa: E402
from agentsec.policy.engine import PolicyEngine  # noqa: E402
from agentsec.policy.model import Policy  # noqa: E402


class LazyAuthorityResolver:
    """Resolve the agent's own identity from the *runtime* environment on first use.

    The reasoning-engine ID is not known when the app is pickled (SDK path). Agent Engine
    exposes it at runtime; we accept AGENTSEC_AGENT_ENGINE_ID first and fall back to
    GOOGLE_CLOUD_AGENT_ENGINE_ID.
    VERIFY: the exact runtime env var name Agent Engine sets for the engine ID.
    """

    def __init__(self) -> None:
        self._resolver: SessionAuthorityResolver | None = None

    def _build(self) -> SessionAuthorityResolver:
        engine_id = os.environ.get("AGENTSEC_AGENT_ENGINE_ID") or os.environ.get(
            "GOOGLE_CLOUD_AGENT_ENGINE_ID"
        )
        if engine_id:
            os.environ["AGENTSEC_AGENT_ENGINE_ID"] = engine_id
        settings = Settings.from_env()
        identity: AgentIdentity = settings.agent_identity()
        log.info("agentsec: running as %s", identity.iam_principal)
        return SessionAuthorityResolver(identity)

    def __call__(self, ctx: Any):
        if self._resolver is None:
            self._resolver = self._build()
        return self._resolver(ctx)


def build_security_plugin() -> SecurityPlugin:
    settings = Settings.from_env()
    policy = Policy.from_yaml(settings.policy_path)
    screener = (
        ModelArmorScreener(settings.model_armor_template)
        if settings.model_armor_template
        else LocalScreener()
    )
    audit = AuditLog(sinks=[CloudLoggingSink(settings.audit_log_name), JsonLinesSink()])
    return SecurityPlugin(
        engine=PolicyEngine(policy),
        authority_resolver=LazyAuthorityResolver(),
        audit=audit,
        screener=screener,
    )


def build_agent():
    return build_support_agent(
        model=os.environ.get("AGENTSEC_MODEL", "gemini-2.5-flash"),
        extra_tools=[
            make_crm_lookup_tool(
                _auth_provider_name(),
                continue_uri=os.environ.get(
                    "AGENTSEC_CONTINUE_URI", "https://app.acme.example/validateUserId"
                ),
            )
        ],
    )


# --------------------------------------------------------------------------------------------
# AdkApp construction — defensive about the SDK version.
# --------------------------------------------------------------------------------------------
try:  # google-cloud-aiplatform >= 1.95 / 2.x
    from vertexai.agent_engines import AdkApp as _AdkApp
except ImportError:  # older layout
    from vertexai.preview.reasoning_engines import AdkApp as _AdkApp  # type: ignore[no-redef]


class AgentsecApp(_AdkApp):  # type: ignore[misc,valid-type]
    """AdkApp that re-runs the runtime bootstrap after unpickling (SDK deployment path)."""

    def set_up(self) -> None:  # called by Agent Engine in the runtime
        _bootstrap_runtime()
        parent = getattr(super(), "set_up", None)
        if callable(parent):
            parent()


def _attach_plugin_as_callbacks(agent: Any, plugin: SecurityPlugin) -> None:
    """Last-resort fallback for AdkApp versions without ``plugins``: agent-level callbacks.

    Loses plugin-only hooks (on_user_message, before/after_run) but keeps the PEP on model
    and tool calls. Tool callbacks differ in one keyword (``args`` vs ``tool_args``).
    """

    async def before_tool(*, tool, args, tool_context):
        return await plugin.before_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_context
        )

    async def after_tool(*, tool, args, tool_context, tool_response):
        return await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_context, result=tool_response
        )

    async def on_tool_error(*, tool, args, tool_context, error):
        return await plugin.on_tool_error_callback(
            tool=tool, tool_args=args, tool_context=tool_context, error=error
        )

    agent.before_model_callback = [plugin.before_model_callback]
    agent.after_model_callback = [plugin.after_model_callback]
    agent.before_tool_callback = [before_tool]
    agent.after_tool_callback = [after_tool]
    agent.on_tool_error_callback = [on_tool_error]


def build_app() -> Any:
    agent = build_agent()
    plugin = build_security_plugin()
    params = inspect.signature(_AdkApp.__init__).parameters
    kwargs: dict[str, Any] = {}
    if "enable_tracing" in params:
        kwargs["enable_tracing"] = True

    if "plugins" in params:
        # VERIFY: `plugins=` on AdkApp — present in recent google-cloud-aiplatform releases;
        # checked at runtime via the signature so older SDKs fall through to the next branch.
        return AgentsecApp(agent=agent, plugins=[plugin], **kwargs)

    if "app" in params:  # some releases accept an ADK App (which carries plugins)
        from google.adk.apps import App

        return AgentsecApp(
            app=App(name="agentsec_support", root_agent=agent, plugins=[plugin]), **kwargs
        )

    log.warning(
        "AdkApp has no plugins parameter; attaching SecurityPlugin as agent callbacks (degraded)"
    )
    _attach_plugin_as_callbacks(agent, plugin)
    return AgentsecApp(agent=agent, **kwargs)


app = build_app()
