# Verified research notes (5 Sep 2026) — identity & security for agentic systems on GCP

## Product landscape (Google Cloud)
- 22 Apr 2026: Google combined the Vertex AI agent capabilities into **Gemini Enterprise Agent Platform** ("Agent Platform").
    - Components:
        - Agent Studio, ADK, Agent Garden, Workspaces.
        - Agent Runtime (Agent Engine resource type `reasoningEngines`), Agent Sandbox, Agent Memory Bank, Agent Sessions.
        - Agent Identity, Agent Registry, Agent Gateway, Agent Anomaly Detection, Agent Threat Detection, Agent Security Dashboard (SCC).
        - Agent Simulation, Agent Evaluation, Agent Observability, Agent Optimizer.
    - Source: https://cloud.google.com/blog/products/ai-machine-learning/introducing-gemini-enterprise-agent-platform
- **Agent Identity** (IAM) is SPIFFE-based.
    - Each agent has an X.509 cert (24h validity, auto-rotated). The tokens are cryptographically bound to the cert.
    - By default, the identity is not shareable. Nobody can impersonate it. It has no long-lived keys.
    - Agent Identity became GA on 22 Apr 2026 (from a third-party timeline). Auth Manager and the APIs became GA on 22 Aug 2026.
      The Org Policy custom constraints and the VPC-SC integration became GA on 14 Aug 2026.
    - The SPIFFE ID is `spiffe://TRUST_DOMAIN/resources/SERVICE/RESOURCE_PATH`. The IAM principal is `principal://TRUST_DOMAIN/resources/SERVICE/RESOURCE_PATH`.
    - The trust domain is `agents.global.org-ORG_ID.system.id.goog` (org) or `agents.global.project-PROJECT_NUMBER.system.id.goog` (no org).
    - An example principal: `principal://agents.global.org-123456789012.system.id.goog/resources/aiplatform/projects/9876543210/locations/us-central1/reasoningEngines/my-test-agent`
    - The principalSet forms are `principalSet://TRUST_DOMAIN/attribute.platformContainer/aiplatform/projects/PROJECT_NUMBER` (all agents in a project)
      and `principalSet://TRUST_DOMAIN/attribute.platform/aiplatform` (all agents on the platform in the org).
    - The supported runtimes are Agent Runtime, Gemini Enterprise and Cloud Run. The policy types are allow, deny, Principal Access Boundary and VPC-SC ingress/egress rules.
    - VPC-SC: add `agentidentity.googleapis.com` and `agentidentitycredentials.googleapis.com` to the perimeters. Use the restricted VIP.
    - A limitation: you cannot grant legacy bucket roles to an agent identity. The recommended default roles are `roles/aiplatform.expressUser`, `roles/serviceusage.serviceUsageConsumer` and `roles/browser`.
    - Sources: https://docs.cloud.google.com/iam/docs/agent-identity-overview, https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/runtime/agent-identity,
      https://arnav.au/2026/08/26/gcp-agent-identity-auth-manager-and-apis-what-ga-changes/
- The credential acquisition table:
    - User-delegated: 3-legged OAuth (external tools).
    - The agent's own: cloud identity (GCP services), 2-legged OAuth, API key, HTTP basic (not recommended).
- Token security:
    - Google APIs use mTLS and cert-bound tokens. Traffic across Agent Gateway also uses DPoP (RFC 9449). The result is "double-bound" credentials.
    - The default Google-managed Context-Aware Access policy makes it impossible to replay bound tokens. The opt-out env var (strongly discouraged) is
      `GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES=False`.
    - Source: https://docs.cloud.google.com/access-context-manager/docs/caa-agent-security, https://docs.cloud.google.com/iam/docs/auth-agent-own-identity
- Deploy with identity:
    - Python: `client.agent_engines.create(agent=AdkApp(agent), config={"identity_type": types.IdentityType.AGENT_IDENTITY, ...})` with
      `vertexai.Client(project, location, http_options=dict(api_version="v1beta1"))`.
    - `.agent_engine_config.json` contains `{ "identity_type": "AGENT_IDENTITY" }`.
    - `agents-cli deploy --agent-identity`.
    - An example of the requirements list: `"google-cloud-aiplatform[agent_engines,adk]"`, `"google-adk[agent-identity,mcp]>=2.7.1"`.
- Cloud Run: deploy an agent with `gcloud beta run deploy SERVICE --image=... --functional-type=agent --identity-type=agent-identity`.
    - For an MCP server, use `--functional-type=mcp-server --identity-type=agent-identity|service-account`.
    - Cloud Run supports jobs for agents. Only services can be MCP servers.
    - A migration from an SA to an agent identity gives a NEW principal. The new principal inherits no permissions (use Policy Analyzer).
    - Cloud Run registers the agents and MCP servers automatically in Agent Registry, under `/agents` and `/mcpServers`.
    - Source: https://docs.cloud.google.com/run/docs/ai/agent-platform-features
- **Auth Manager** (Agent Identity auth manager) is a centralized credential vault and broker for outbound tool auth. It supports 3LO (user-delegated), 2LO and API key.
    - The resource is `projects/PROJECT_ID/locations/LOCATION/authProviders/NAME`. The callback is `https://agentidentitycredentials.googleapis.com/v1/projects/.../authProviders/NAME/oauthcallback`.
    - gcloud: `gcloud agent-identity auth-providers create NAME --project --location --three-legged-oauth-client-id ... --three-legged-oauth-client-secret ... --three-legged-oauth-authorization-url ... --three-legged-oauth-token-url ...`
    - IAM: the agent principal gets `roles/agentidentity.user` on the auth provider. The admin roles are `roles/agentidentity.admin|editor`.
      The agent also needs `roles/aiplatform.user` and `roles/serviceusage.serviceUsageConsumer`.
    - API: `agentidentitycredentials.retrieveCredentials`. To finalize, send `POST https://agentidentitycredentials.googleapis.com/v1/{authProviderName}/credentials:finalize` with `{userId, userIdValidationState, consentNonce}`.
    - ADK: `CredentialManager.register_auth_provider(GcpAuthProvider())`, `GcpAuthProviderScheme(name=..., scopes=[...], continue_uri=...)`, `McpToolset(connection_params=StreamableHTTPConnectionParams(url=...), auth_scheme=scheme)`.
      Also `AuthenticatedFunctionTool(func=..., auth_config=AuthConfig(auth_scheme=GcpAuthProviderScheme(...)))` with a `credential: AuthCredential` param. The token is in `credential.http.credentials.token`.
    - The consent flow:
        1. The agent emits an `adk_request_credential` function call with `authorization_uri` and `consent_nonce`.
        2. The frontend does a redirect.
        3. The `/validateUserId` endpoint finalizes the consent.
        4. Resume with a FunctionResponse named `adk_request_credential`.
    - There are two paths for secrets:
        1. The direct ADK path: the token goes back to the agent process.
        2. The Agent Gateway path: the auth manager encrypts the end-user creds, and only the gateway decrypts them. The agent never sees the raw credential.
    - Source: https://docs.cloud.google.com/iam/docs/auth-with-3lo-v2
- ADK 2.8.0 internals (verified from the installed source):
    - `google.adk.integrations.agent_identity.GcpAuthProvider` uses `google.cloud.agentidentitycredentials_v1.AuthProviderCredentialsServiceClient.retrieve_credentials(RetrieveCredentialsRequest(auth_provider, user_id, scopes, continue_uri))`.
    - The response oneof: success{header, token} | pending | uri_consent_required{authorization_uri, consent_nonce} | consent_rejected. The env var `AGENT_IDENTITY_CREDENTIALS_TARGET_HOST` overrides the endpoint.
    - `BaseAuthProvider.get_auth_credential(auth_config, context)`, `CredentialManager.register_auth_provider(provider)`. The experimental feature flags are PLUGGABLE_AUTH and AUTHENTICATED_FUNCTION_TOOL.
    - `McpToolset(connection_params, tool_filter, tool_name_prefix, auth_scheme, auth_credential, require_confirmation, header_provider(ReadonlyContext)->headers, ...)`.
    - The `BasePlugin` hooks: before/after_agent, before/after_model, before/after_tool, on_tool_error, on_model_error, on_user_message, on_event, before/after_run.
    - `ToolContext.request_confirmation(hint, payload)`, `tool_context.tool_confirmation` (ToolConfirmation{hint, confirmed, payload}), and `FunctionTool(func, require_confirmation=bool|callable)`.
    - `adk_request_credential` == REQUEST_EUC_FUNCTION_CALL_NAME. There is also REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.
- **Agent Gateway** is a networking component.
    - It has two modes: ingress (from the client to the agent) and egress (from the agent to anywhere).
    - It uses mTLS and DPoP. IAP enforces IAM for each SPIFFE id on registered resources. The permission is `iap.resources.egressViaIAP`.
    - It applies Model Armor on ingress/egress. It gives "semantic governance".
    - It supports all HTTP traffic, MCP and A2A included. It parses MCP attributes.
    - You must register the resources in Agent Registry (≤5,000 per gateway).
    - The commands are `gcloud network-services agent-gateways`. The API resource is Network Services v1 `agentGateways`.
    - The limitations: the gateway itself has no VPC-SC support. It needs publicly trusted CA certs. For Gemini Enterprise, the gateway supports egress only.
    - Source: https://docs.cloud.google.com/gemini-enterprise-agent-platform/govern/gateways/agent-gateway-overview
- **VPC Service Controls** (27 Jun 2026 blog):
    - Ingress and egress rules can contain agent identities (a single principal or a principalSet).
    - Conditions can use MCP attributes: `mcp.toolName`, `mcp.method`, `mcp.tool.isReadOnly`.
    - When Agent Platform is a protected service, it blocks public internet access automatically.
    - Source: https://cloud.google.com/blog/products/identity-security/securing-agentic-ai-whats-new-in-vpc-service-controls
- **Model Armor** with Vertex AI:
    - A per-request `modelArmorConfig{promptTemplateName, responseTemplateName}` on generateContent.
    - The project floor settings:
      `gcloud model-armor floorsettings update --full-uri=projects/PROJECT_ID/locations/global/floorSetting --add-integrated-services=VERTEX_AI`.
    - The modes are INSPECT_ONLY (default) and INSPECT_AND_BLOCK. The filters are RAI, prompt injection and jailbreak, Sensitive Data Protection, and malicious URI.
    - The precedence: request template > floor > Gemini built-in safety.
    - A limitation: Model Armor does not sanitize file/PDF prompts. The docs list a limited set of regions for the integration (europe-west1/2/3, asia-southeast1, asia-south1).
    - Source: https://docs.cloud.google.com/model-armor/model-armor-vertex-integration
- IAP for agents: you bind IAM allow/deny policies on Agent Registry service instances (MCP servers, destination agents, endpoints). Agent Gateway enforces them.
  Source: https://docs.cloud.google.com/iap/docs/agent-overview

## Standards
- **MCP authorization** (spec rev 2025-11-25, with the changes of the 2026-07-28 revision noted):
    - It uses OAuth 2.1 (draft-ietf-oauth-v2-1-13). The MCP server is the resource server.
    - The server MUST implement Protected Resource Metadata (RFC 9728). The server MUST advertise the metadata through `WWW-Authenticate: Bearer resource_metadata="..."` on 401 and/or `/.well-known/oauth-protected-resource[/path]`.
    - Clients MUST use PRM for AS discovery. AS metadata discovery uses RFC 8414 or OIDC discovery.
    - Clients MUST implement PKCE S256. They MUST refuse if `code_challenge_methods_supported` is absent.
    - The `resource` param of Resource Indicators (RFC 8707) MUST be in the authorization and token requests (the canonical server URI). Servers MUST validate the audience.
    - Servers MUST NOT accept or transit other tokens (the specification forbids token passthrough).
    - Confused deputy: proxies with static client IDs MUST get user consent for each dynamically registered client.
    - Client registration: Client ID Metadata Documents (SHOULD, `client_id_metadata_document_supported`), pre-registration, and DCR (MAY).
      The 2026-07-28 revision deprecates DCR in favor of CIMD.
    - If the token does not have the necessary scope, the server returns 403 `insufficient_scope`, and a step-up flow follows. Tokens MUST NOT go in query strings.
    - Redirect URIs are localhost or HTTPS. Tokens are short-lived. Public clients use refresh rotation.
    - The 2026-07-28 changes:
        - Stateless operation (no sessions/initialize).
        - `server/discover`.
        - The MRTR pattern.
        - The tasks extension.
        - `iss` validation (RFC 9207).
        - Credentials keyed by issuer.
        - `Mcp-Method`/`Mcp-Name` headers.
        - DCR deprecated.
    - Sources: https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization, https://modelcontextprotocol.io/specification/2026-07-28/changelog
- **A2A v1.0.0**:
    - The Agent Card has `securitySchemes` and `security`.
    - The scheme types are APIKeySecurityScheme, HTTPAuthSecurityScheme, OAuth2SecurityScheme, OpenIdConnectSecurityScheme and MutualTlsSecurityScheme.
    - `TASK_STATE_AUTH_REQUIRED` is for in-task auth. There is an authenticated extended card.
    - All the bindings (JSON-RPC/gRPC/HTTP+JSON) need TLS. `AgentCardSignature` (JWS) holds the signature of the card.
    - Source: https://a2a-protocol.org/latest/specification/
- **OWASP Top 10 for Agentic Applications 2026** (published 9 Dec 2025):
    - ASI01 Agent Goal Hijack.
    - ASI02 Tool Misuse & Exploitation.
    - ASI03 Identity & Privilege Abuse.
    - ASI04 Agentic Supply Chain Vulnerabilities.
    - ASI05 Unexpected Code Execution (RCE).
    - ASI06 Memory & Context Poisoning.
    - ASI07 Insecure Inter-Agent Communication.
    - ASI08 Cascading Failures.
    - ASI09 Human-Agent Trust Exploitation.
    - ASI10 Rogue Agents.
    - Source: https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/, https://cycode.com/blog/owasp-top-10-agentic-applications/
- Google, "An Introduction to Google's Approach for Secure AI Agents" (Jun 2025). Its principles:
    - Well-defined human controllers.
    - Limited agent powers.
    - Observable actions and planning.
    - Hybrid defense-in-depth: deterministic runtime policy enforcement and reasoning-based defenses.
- The RFCs:
    - RFC 8693 OAuth token exchange (`urn:ietf:params:oauth:grant-type:token-exchange`, `subject_token`, `actor_token`, the `act` claim).
    - RFC 9449 DPoP (`htm`, `htu`, `jti`, `iat`, `ath`, and separately `cnf.jkt`).
    - RFC 8705 mTLS-bound tokens (`cnf.x5t#S256`).
    - RFC 9728 PRM.
    - RFC 8707 resource indicators.
    - RFC 8414 AS metadata.
    - RFC 7591 DCR.
    - RFC 9068 JWT access tokens.
- The GCP features with a long history:
    - Workload Identity Federation: external identities go to an STS token exchange (`sts.googleapis.com`), then to an optional SA impersonation.
    - Service account impersonation (`generateAccessToken`, short-lived).
    - Credential Access Boundaries / downscoped tokens (`google.auth.downscoped.Credentials`, Cloud Storage only).
    - IAM Conditions (CEL).
    - Deny policies.
    - Principal Access Boundary policies.
    - Org Policy custom constraints.
    - Secret Manager.
    - CMEK.
    - Cloud Audit Logs (Admin Activity / Data Access).
    - Cloud KMS.

## Versions available (PyPI, 5 Sep 2026)
- google-adk 2.8.0 (extras: agent-identity, mcp).
- mcp 1.27.0: compatible with ADK 2.8. The mcp 2.x versions conflict with ADK 2.8.
- a2a-sdk 1.1.2.
- google-cloud-aiplatform 2.1.0.
- The Terraform 1.13.3 binary works in the sandbox.
