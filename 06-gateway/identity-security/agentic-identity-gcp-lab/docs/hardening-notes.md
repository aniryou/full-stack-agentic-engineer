# Hardening notes and known limitations

This file is a short record of two things. The first is the security review of the reference
implementation before release. The second is what the local profile does *not* reproduce, on
purpose. Use it for the "what would you change for production?" follow-up in a design review.

## Findings fixed during review

| Area | Finding | Fix |
|---|---|---|
| Runtime PEP | The code calculated the authority again from the session state on each callback. The authority is who acts, for whom, and with which scopes. A tool with `tool_context`, or any code path that writes `agentsec:*` keys, was able to escalate its own scopes in the middle of a run. | `SecurityPlugin.before_run_callback` resolves the authority **once per invocation**, from the state at the start of the run, and pins it. Later callbacks use the pinned value. If the plugin cannot verify the authority of a run, it refuses the run before the model runs (`run.authority` audit event). Test: `test_authority_is_pinned_for_the_invocation`, `test_unverifiable_authority_refuses_to_run`. |
| MCP server | The DPoP replay cache (`jti` set) grew with no limit. | The cache is `jti → first-seen`. The code prunes the entries older than `proof_max_age` (default 300 s, the same window that `DPoP.verify` enforces). The cache also has a size limit. |
| MCP server | The upstream "no passthrough" example minted the upstream token with the *agent* as the subject. This made the point of the example less clear. | The MCP server now gets the upstream token under its **own** identity (`sub` = the server's SPIFFE ID). The agent is in `act` and the user is in `on_behalf_of`. The upstream token is a new delegation hop, never the inbound token. |
| Audit | The in-memory audit events stored the raw tool arguments. Redaction occurred only at serialisation. | The code redacts the arguments when it records the event (`redact(tool_args)`). Thus notebooks and sinks never see raw secrets. |
| Broker | The consent URL advertised `code_challenge_method=S256`, but no real PKCE exchange occurred. | The parameter is gone from the URL. A comment states that the broker does PKCE (Auth Manager `enable_pkce`). PKCE is never visible to the agent. |

## Deliberate simplifications of the local profile

- **Sender-constraining on the ADK → MCP path.** In the local profile, the `McpToolset` sends audience-bound, short-lived, user-delegated *bearer* tokens (an RFC 8707 audience and narrowed scopes). Notebook 06 shows DPoP with raw HTTP, not through the ADK client. This is because a DPoP proof must be unique for each HTTP request. But ADK evaluates its `header_provider` once for each session, not for each request. On Google Cloud, Agent Identity (certificate-bound tokens) and Agent Gateway (mTLS + DPoP) do this binding for you.
- **One STS for everyone.** `TokenIssuer` acts as the IdP, the STS, the token minter of Auth Manager and the upstream authorization server. In production, these are different trust domains with different keys. But the *checks* that the code does (issuer, audience, scope, `cnf`, `act`) are the same.
- **Certificate binding is emulated.** `LocalRuntimeCA` issues real X.509 certificates with SPIFFE SANs, and the tokens carry `cnf.x5t#S256`. But no TLS handshake occurs in the local profile. The code gives the "presented" thumbprint to the verifier explicitly. Agent Identity does the mTLS handshake at the metadata server or the gateway.
- **`LocalScreener` is a regex stand-in for Model Armor.** It exists so that you can test the control flow of the plugin offline (block before the model, hold back a response, fence tool output). Do not ship it as a detector. `ModelArmorScreener` is the drop-in replacement for production.
- **`LocalAuthManager` keeps consent in memory.** Auth Manager stores end-user credentials in a Google-managed vault with regional residency. Here, the "vault" is a dict. The tokens that it gives out are minted tokens, not stored refresh tokens.
- **Policy `egress_hosts` accepts suffixes.** The sample policy uses `.googleapis.com` to keep the policy short. In a real policy, list the specific hosts that a tool needs.
- **Budgets live in ADK `temp:` state.** They are invocation-scoped, and they reset for each run. The correct place for production budgets (spend, rate) is a shared store. The key of that store is the agent principal and the user.
- **A2A card verification trusts a caller-supplied key set.** Real deployments resolve the signing key from the registry entry of the agent, or from a JWKS endpoint pinned to the `kid` of the card. `verify_agent_card` takes the key map as a parameter on purpose, so that the resolution is explicit.

## Things worth citing as done right

- A deny-by-default tool policy with explicit tiers. The policy matches principals exactly as IAM matches `principal://` and `principalSet://` members (no substring match).
- Every verification path tells *why* it failed (`InvalidAudience`, `BindingMismatch`, `InsufficientScope`, `ReplayDetected`). Thus policy and audit can act on the reason.
- Human confirmation payloads carry the tool, the exact arguments and the authority. They never carry the model's description of its intent.
- The code fences tool results with provenance before the model sees them. The audit record keeps the tag.
- Secrets are opaque values (`SecretValue`). Their `repr` never shows them, and they refuse any attempt to pickle them.
