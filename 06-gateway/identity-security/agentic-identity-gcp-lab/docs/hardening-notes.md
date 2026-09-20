# Hardening notes and known limitations

A short record of the security review done on the reference implementation before release, and of
what the local profile deliberately does *not* reproduce. Useful for the "what would you change for
production?" follow-up in a design review.

## Findings fixed during review

| Area | Finding | Fix |
|---|---|---|
| Runtime PEP | Authority (who is acting, for whom, with which scopes) was re-derived from session state on every callback. A tool with `tool_context` — or any code path that writes `agentsec:*` keys — could escalate its own scopes mid-run. | `SecurityPlugin.before_run_callback` resolves the authority **once per invocation** from the state as it was when the run started and pins it; later callbacks use the pinned value. A run whose authority cannot be verified is refused before the model runs (`run.authority` audit event). Test: `test_authority_is_pinned_for_the_invocation`, `test_unverifiable_authority_refuses_to_run`. |
| MCP server | The DPoP replay cache (`jti` set) grew without bound. | Cache is `jti → first-seen`, pruned past `proof_max_age` (default 300 s, the same window `DPoP.verify` enforces) and bounded in size. |
| MCP server | The upstream "no passthrough" example minted the upstream token with the *agent* as subject, which blurred the point. | The MCP server now obtains the upstream token under its **own** identity (`sub` = the server's SPIFFE ID) with the agent in `act` and the user in `on_behalf_of` — a fresh delegation hop, never the inbound token. |
| Audit | Raw tool arguments were stored in in-memory audit events (redaction only applied on serialisation). | Arguments are redacted at record time (`redact(tool_args)`), so notebooks and sinks never see raw secrets. |
| Broker | The consent URL advertised `code_challenge_method=S256` without a real PKCE exchange. | Removed; a comment states that PKCE is performed by the broker (Auth Manager `enable_pkce`) and never surfaces to the agent. |

## Deliberate simplifications of the local profile

- **Sender-constraining on the ADK → MCP path.** Locally the `McpToolset` sends audience-bound, short-lived, user-delegated *bearer* tokens (RFC 8707 audience + scope narrowing). DPoP is demonstrated with raw HTTP in notebook 06 rather than through the ADK client, because a DPoP proof must be unique per HTTP request and ADK's `header_provider` is evaluated per session, not per request. On Google Cloud this binding is done for you by Agent Identity (certificate-bound tokens) and Agent Gateway (mTLS + DPoP).
- **One STS for everyone.** `TokenIssuer` plays the IdP, the STS, the Auth Manager's token minter and the upstream authorization server. In production those are separate trust domains with separate keys; the *checks* the code performs (issuer, audience, scope, `cnf`, `act`) are the same.
- **Certificate binding is emulated.** `LocalRuntimeCA` issues real X.509 certificates with SPIFFE SANs and tokens carry `cnf.x5t#S256`, but no TLS handshake happens locally; the verifier is handed the "presented" thumbprint explicitly. Agent Identity performs the mTLS handshake at the metadata server / gateway.
- **`LocalScreener` is a regex stand-in for Model Armor.** It exists so the plugin's control flow (block before the model, withhold a response, fence tool output) is testable offline; it is not a detector you should ship. `ModelArmorScreener` is the production drop-in.
- **`LocalAuthManager` keeps consent in memory.** Auth Manager stores end-user credentials in a Google-managed vault with regional residency; here the "vault" is a dict and the tokens it hands out are minted, not stored refresh tokens.
- **Policy `egress_hosts` accepts suffixes.** `.googleapis.com` is used in the sample policy for brevity; in a real policy list the specific hosts a tool needs.
- **Budgets live in ADK `temp:` state.** They are invocation-scoped and reset per run; production budgets (spend, rate) belong in a shared store keyed by agent principal and user.
- **A2A card verification trusts a caller-supplied key set.** Real deployments resolve the signing key from the agent's registry entry or a JWKS endpoint pinned to the card's `kid`; `verify_agent_card` deliberately takes the key map as a parameter so that resolution is explicit.

## Things worth citing as done right

- Deny-by-default tool policy with explicit tiers, principals matched exactly the way IAM matches `principal://` and `principalSet://` members (no substring matching).
- Every verification path is specific about *why* it failed (`InvalidAudience`, `BindingMismatch`, `InsufficientScope`, `ReplayDetected`) so policy and audit can act on the reason.
- Human confirmation payloads carry the tool, the exact arguments and the authority — never the model's description of what it intends.
- Tool results are fenced with provenance before the model sees them, and the audit record keeps the tag.
- Secrets are opaque values (`SecretValue`) whose `repr` never reveals them and which refuse to be pickled.
