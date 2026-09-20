# Verified research notes (5 Sep 2026) — identity & security for agentic systems on GCP

## Product landscape (Google Cloud)
- 22 Apr 2026: Vertex AI agent capabilities consolidated as **Gemini Enterprise Agent Platform** ("Agent Platform").
  Components: Agent Studio, ADK, Agent Garden, Workspaces; Agent Runtime (Agent Engine resource type `reasoningEngines`),
  Agent Sandbox, Agent Memory Bank, Agent Sessions; Agent Identity, Agent Registry, Agent Gateway, Agent Anomaly Detection,
  Agent Threat Detection, Agent Security Dashboard (SCC); Agent Simulation, Agent Evaluation, Agent Observability, Agent Optimizer.
  Source: https://cloud.google.com/blog/products/ai-machine-learning/introducing-gemini-enterprise-agent-platform
- **Agent Identity** (IAM): SPIFFE-based, per-agent X.509 cert (24h validity, auto-rotated), tokens cryptographically bound to cert;
  not shareable by default, cannot be impersonated, no long-lived keys. GA 22 Apr 2026 (per third-party timeline); Auth Manager + APIs GA 22 Aug 2026;
  Org Policy custom constraints + VPC-SC integration GA 14 Aug 2026.
  SPIFFE ID: `spiffe://TRUST_DOMAIN/resources/SERVICE/RESOURCE_PATH`; IAM principal: `principal://TRUST_DOMAIN/resources/SERVICE/RESOURCE_PATH`.
  Trust domain: `agents.global.org-ORG_ID.system.id.goog` (org) or `agents.global.project-PROJECT_NUMBER.system.id.goog` (no org).
  Example principal: `principal://agents.global.org-123456789012.system.id.goog/resources/aiplatform/projects/9876543210/locations/us-central1/reasoningEngines/my-test-agent`
  principalSet forms: `principalSet://TRUST_DOMAIN/attribute.platformContainer/aiplatform/projects/PROJECT_NUMBER` (all agents in a project),
  `principalSet://TRUST_DOMAIN/attribute.platform/aiplatform` (all agents on the platform in org).
  Supported runtimes: Agent Runtime, Gemini Enterprise, Cloud Run. Policy types: allow, deny, Principal Access Boundary, VPC-SC ingress/egress rules.
  VPC-SC: add `agentidentity.googleapis.com` and `agentidentitycredentials.googleapis.com` to perimeters; use restricted VIP.
  Limitation: cannot grant legacy bucket roles. Recommended default roles: `roles/aiplatform.expressUser`, `roles/serviceusage.serviceUsageConsumer`, `roles/browser`.
  Sources: https://docs.cloud.google.com/iam/docs/agent-identity-overview ; https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/runtime/agent-identity ;
  https://arnav.au/2026/08/26/gcp-agent-identity-auth-manager-and-apis-what-ga-changes/
- Credential acquisition table: user-delegated 3-legged OAuth (external tools); agent's own: cloud identity (GCP services), 2-legged OAuth, API key, HTTP basic (not recommended).
- Token security: mTLS + cert-bound tokens for Google APIs; across Agent Gateway also DPoP (RFC 9449) → "double-bound" credentials.
  Default Google-managed Context-Aware Access policy makes bound tokens unreplayable. Opt-out env var (strongly discouraged):
  `GOOGLE_API_PREVENT_AGENT_TOKEN_SHARING_FOR_GCP_SERVICES=False`.
  Source: https://docs.cloud.google.com/access-context-manager/docs/caa-agent-security ; https://docs.cloud.google.com/iam/docs/auth-agent-own-identity
- Deploy with identity: Python `client.agent_engines.create(agent=AdkApp(agent), config={"identity_type": types.IdentityType.AGENT_IDENTITY, ...})` with
  `vertexai.Client(project, location, http_options=dict(api_version="v1beta1"))`; `.agent_engine_config.json` → `{ "identity_type": "AGENT_IDENTITY" }`; `agents-cli deploy --agent-identity`.
  Requirements list example: `"google-cloud-aiplatform[agent_engines,adk]"`, `"google-adk[agent-identity,mcp]>=2.7.1"`.
- Cloud Run: `gcloud beta run deploy SERVICE --image=... --functional-type=agent --identity-type=agent-identity`;
  MCP server: `--functional-type=mcp-server --identity-type=agent-identity|service-account`; jobs supported for agents; only services can be MCP servers.
  Migration from SA to agent identity = NEW principal, no inherited permissions (use Policy Analyzer). Auto-registered in Agent Registry under `/agents` and `/mcpServers`.
  Source: https://docs.cloud.google.com/run/docs/ai/agent-platform-features
- **Auth Manager** (Agent Identity auth manager): centralized credential vault/broker for outbound tool auth: 3LO (user-delegated), 2LO, API key.
  Resource: `projects/PROJECT_ID/locations/LOCATION/authProviders/NAME`. Callback: `https://agentidentitycredentials.googleapis.com/v1/projects/.../authProviders/NAME/oauthcallback`.
  gcloud: `gcloud agent-identity auth-providers create NAME --project --location --three-legged-oauth-client-id ... --three-legged-oauth-client-secret ... --three-legged-oauth-authorization-url ... --three-legged-oauth-token-url ...`
  IAM: `roles/agentidentity.user` bound to agent principal on the auth provider; admin roles `roles/agentidentity.admin|editor`. Agent also needs `roles/aiplatform.user`, `roles/serviceusage.serviceUsageConsumer`.
  API: `agentidentitycredentials.retrieveCredentials`; finalize `POST https://agentidentitycredentials.googleapis.com/v1/{authProviderName}/credentials:finalize` with `{userId, userIdValidationState, consentNonce}`.
  ADK: `CredentialManager.register_auth_provider(GcpAuthProvider())`; `GcpAuthProviderScheme(name=..., scopes=[...], continue_uri=...)`; `McpToolset(connection_params=StreamableHTTPConnectionParams(url=...), auth_scheme=scheme)`;
  `AuthenticatedFunctionTool(func=..., auth_config=AuthConfig(auth_scheme=GcpAuthProviderScheme(...)))` with `credential: AuthCredential` param → `credential.http.credentials.token`.
  Consent flow: agent emits `adk_request_credential` function call with `authorization_uri` + `consent_nonce`; frontend redirects; `/validateUserId` endpoint finalizes; resume with FunctionResponse named `adk_request_credential`.
  Two secret-handling paths: (1) direct ADK path — token returned to agent process; (2) Agent Gateway path — end-user creds encrypted by auth manager and decrypted only at gateway (agent never sees raw credential).
  Source: https://docs.cloud.google.com/iam/docs/auth-with-3lo-v2
- ADK 2.8.0 internals (verified from installed source): `google.adk.integrations.agent_identity.GcpAuthProvider` uses `google.cloud.agentidentitycredentials_v1.AuthProviderCredentialsServiceClient.retrieve_credentials(RetrieveCredentialsRequest(auth_provider, user_id, scopes, continue_uri))`;
  response oneof: success{header, token} | pending | uri_consent_required{authorization_uri, consent_nonce} | consent_rejected. Env `AGENT_IDENTITY_CREDENTIALS_TARGET_HOST` overrides endpoint.
  `BaseAuthProvider.get_auth_credential(auth_config, context)`; `CredentialManager.register_auth_provider(provider)`; feature flags experimental: PLUGGABLE_AUTH, AUTHENTICATED_FUNCTION_TOOL.
  `McpToolset(connection_params, tool_filter, tool_name_prefix, auth_scheme, auth_credential, require_confirmation, header_provider(ReadonlyContext)->headers, ...)`.
  `BasePlugin` hooks: before/after_agent, before/after_model, before/after_tool, on_tool_error, on_model_error, on_user_message, on_event, before/after_run.
  `ToolContext.request_confirmation(hint, payload)`, `tool_context.tool_confirmation` (ToolConfirmation{hint, confirmed, payload}); `FunctionTool(func, require_confirmation=bool|callable)`.
  `adk_request_credential` == REQUEST_EUC_FUNCTION_CALL_NAME; also REQUEST_CONFIRMATION_FUNCTION_CALL_NAME.
- **Agent Gateway**: networking component; ingress (client→agent) and egress (agent→anywhere) modes; mTLS + DPoP; IAP enforces IAM per SPIFFE id on registered resources; permission `iap.resources.egressViaIAP`;
  Model Armor on ingress/egress; "semantic governance"; supports all HTTP traffic incl. MCP and A2A; parses MCP attributes; resources must be registered in Agent Registry (≤5,000 per gateway);
  `gcloud network-services agent-gateways`; Network Services v1 `agentGateways`. Limitations: no VPC-SC support for gateway itself, publicly trusted CA certs required, Gemini Enterprise egress-only.
  Source: https://docs.cloud.google.com/gemini-enterprise-agent-platform/govern/gateways/agent-gateway-overview
- **VPC Service Controls** (27 Jun 2026 blog): agent identities in ingress/egress rules (single principal or principalSet);
  MCP attributes in conditions: `mcp.toolName`, `mcp.method`, `mcp.tool.isReadOnly`; Agent Platform as protected service auto-blocks public internet access.
  Source: https://cloud.google.com/blog/products/identity-security/securing-agentic-ai-whats-new-in-vpc-service-controls
- **Model Armor** ↔ Vertex AI: per-request `modelArmorConfig{promptTemplateName, responseTemplateName}` on generateContent; project floor settings:
  `gcloud model-armor floorsettings update --full-uri=projects/PROJECT_ID/locations/global/floorSetting --add-integrated-services=VERTEX_AI`;
  modes INSPECT_ONLY (default) / INSPECT_AND_BLOCK; filters: RAI, prompt injection & jailbreak, Sensitive Data Protection, malicious URI;
  precedence: request template > floor > Gemini built-in safety; limitation: file/PDF prompts not sanitized; limited regions listed (europe-west1/2/3, asia-southeast1, asia-south1) for the integration.
  Source: https://docs.cloud.google.com/model-armor/model-armor-vertex-integration
- IAP for agents: IAM allow/deny policies bound on Agent Registry service instances (MCP servers, destination agents, endpoints), enforced at Agent Gateway.
  Source: https://docs.cloud.google.com/iap/docs/agent-overview

## Standards
- **MCP authorization** (spec rev 2025-11-25; 2026-07-28 revision changes noted): OAuth 2.1 (draft-ietf-oauth-v2-1-13); MCP server = resource server; MUST implement Protected Resource Metadata (RFC 9728),
  advertise via `WWW-Authenticate: Bearer resource_metadata="..."` on 401 and/or `/.well-known/oauth-protected-resource[/path]`; clients MUST use PRM for AS discovery;
  AS metadata discovery RFC 8414 / OIDC discovery; clients MUST implement PKCE S256 and refuse if `code_challenge_methods_supported` absent;
  Resource Indicators RFC 8707 `resource` param MUST be in authorization + token requests (canonical server URI); servers MUST validate audience;
  MUST NOT accept/transit other tokens (token passthrough forbidden); confused-deputy: proxies with static client IDs MUST obtain user consent per dynamically registered client;
  Client registration: Client ID Metadata Documents (SHOULD; `client_id_metadata_document_supported`), pre-registration, DCR (MAY; deprecated in 2026-07-28 in favor of CIMD);
  403 `insufficient_scope` + step-up flow; tokens MUST NOT go in query strings; redirect URIs localhost or HTTPS; short-lived tokens; refresh rotation for public clients.
  2026-07-28 changes: stateless (no sessions/initialize), `server/discover`, MRTR pattern, tasks extension, `iss` validation (RFC 9207), credentials keyed by issuer, `Mcp-Method`/`Mcp-Name` headers, DCR deprecated.
  Sources: https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization ; https://modelcontextprotocol.io/specification/2026-07-28/changelog
- **A2A v1.0.0**: Agent Card `securitySchemes` + `security`; scheme types APIKeySecurityScheme, HTTPAuthSecurityScheme, OAuth2SecurityScheme, OpenIdConnectSecurityScheme, MutualTlsSecurityScheme;
  `TASK_STATE_AUTH_REQUIRED` for in-task auth; authenticated extended card; bindings JSON-RPC/gRPC/HTTP+JSON all require TLS; `AgentCardSignature` (JWS) for card signing.
  Source: https://a2a-protocol.org/latest/specification/
- **OWASP Top 10 for Agentic Applications 2026** (published 9 Dec 2025): ASI01 Agent Goal Hijack; ASI02 Tool Misuse & Exploitation; ASI03 Identity & Privilege Abuse;
  ASI04 Agentic Supply Chain Vulnerabilities; ASI05 Unexpected Code Execution (RCE); ASI06 Memory & Context Poisoning; ASI07 Insecure Inter-Agent Communication;
  ASI08 Cascading Failures; ASI09 Human-Agent Trust Exploitation; ASI10 Rogue Agents.
  Source: https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/ ; https://cycode.com/blog/owasp-top-10-agentic-applications/
- Google "An Introduction to Google's Approach for Secure AI Agents" (Jun 2025): principles — well-defined human controllers; limited agent powers; observable actions & planning; hybrid defense-in-depth (deterministic runtime policy enforcement + reasoning-based defenses).
- RFC 8693 OAuth token exchange (`urn:ietf:params:oauth:grant-type:token-exchange`, `subject_token`, `actor_token`, `act` claim), RFC 9449 DPoP (`htm`, `htu`, `jti`, `iat`, `ath`; `cnf.jkt`), RFC 8705 mTLS-bound tokens (`cnf.x5t#S256`),
  RFC 9728 PRM, RFC 8707 resource indicators, RFC 8414 AS metadata, RFC 7591 DCR, RFC 9068 JWT access tokens.
- GCP long-standing: Workload Identity Federation (external identities → STS `sts.googleapis.com` token exchange → optional SA impersonation), service account impersonation (`generateAccessToken`, short-lived),
  Credential Access Boundaries / downscoped tokens (`google.auth.downscoped.Credentials`, Cloud Storage only), IAM Conditions (CEL), deny policies, Principal Access Boundary policies, Org Policy custom constraints,
  Secret Manager, CMEK, Cloud Audit Logs (Admin Activity / Data Access), Cloud KMS.

## Versions available (PyPI, 5 Sep 2026)
google-adk 2.8.0 (extras: agent-identity, mcp); mcp 1.27.0 (compatible with ADK 2.8; mcp 2.x conflicts); a2a-sdk 1.1.2; google-cloud-aiplatform 2.1.0; Terraform 1.13.3 binary works in sandbox.
