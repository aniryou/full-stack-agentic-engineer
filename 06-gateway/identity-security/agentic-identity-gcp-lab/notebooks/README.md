# Notebooks — hands-on companions to `docs/primer.md`

Nine worked notebooks, each with a fill-in-the-blank **practice** twin and a completed **solution**.
Everything runs offline against the local twins in `src/agentsec/` (`LocalRuntimeCA`, `TokenIssuer`,
`LocalAuthManager`, `LocalScreener`, the tickets MCP server in a background thread, and the ADK loop
driven by `ScriptedLlm`) — no Google Cloud project or API key is needed. Every notebook ends with a
"In one sentence" line: the 30-second version of the idea it demonstrates.

## Map: notebook → primer section

| Notebook | Primer | What you demonstrate |
|---|---|---|
| `01_agent_identity_and_principals` | §3.1–§3.3 | SPIFFE IDs and `principal://` members; `principalSet` matching on **exact** segments; runtime CA certificate with the SPIFFE SAN; certificate-bound token (`cnf.x5t#S256`); replay from another certificate failing with `BindingMismatch`; own vs delegated `AuthorityContext` and `audit_identities()` |
| `02_delegation_and_token_exchange` | §3.5, §4.3 | RFC 8693 exchange with the `act` claim; scope narrowing; nested `act.act` chains for sub-agents; audience/expiry/scope failures; DPoP proofs, `ath` binding and `jti` replay; Credential Access Boundary JSON and local evaluation |
| `03_auth_manager_broker` | §4.3, §5 | 3LO / 2LO / API-key providers; IAM binding on the provider; `retrieveCredentials` outcomes; consent + `finalize`; an access log attributable to agent **and** user; the ADK path `crm_lookup` → `adk_request_credential` → finalize → `resume_after_auth` |
| `04_policy_enforcement_point` | §4.1, §4.2, §4.4 | `policies/support-agent.yaml` evaluated request by request (unknown tool, principal, authority, scopes, constraints, egress, budgets, confirmation and the `unless` envelope); `dry_run` of a plan; the ADK loop with `LocalStack`: default deny, auto-allow inside the envelope, confirmation approve/reject, `audit.timeline()` |
| `05_prompt_injection_and_guardrails` | §6 | `LocalScreener` (Model Armor shape) on injection / SDP / malicious URIs; the poisoned KB article; a hijacked model contained by deterministic controls; a blocked prompt that never reaches the model (`stack.llm.requests == []`); `EgressPolicy` SSRF cases; what Model Armor floor settings and templates do on GCP |
| `06_mcp_resource_server` | §7.1 | Protected Resource Metadata and the 401 challenge; wrong audience → 401; missing scope → 403; `tools/list` annotations; read vs write scope on `tools/call`; a DPoP-required server (bearer rejected, proof accepted, replay rejected); ADK `McpToolset` with delegated audience-bound tokens; row-level filtering by subject; the separate upstream token (no passthrough) |
| `07_a2a_agent_cards` | §7.2 | Building, signing and verifying an Agent Card; tamper detection; required scopes; `token_for_peer` + `authorize_inbound` with the hop chain; a forwarded user token rejected; a second hop |
| `08_audit_and_governance` | §9, §4.5 | Querying the `AuditLog` (by agent, denials, approvals with approver, dual identity); a policy kill switch effective without redeploy; revoking Auth Manager consent; a burst-detection heuristic over destructive attempts |
| `09_code_evaluation_drills` | §11.2 | Eight "spot the bug" snippets — token passthrough, `aud` not checked, allow-unknown-tools, confirmation UI showing the model's summary, secret in `tool_context.state`, substring egress match, DPoP verifier without `jti` tracking, `principalSet` prefix match — each with the fix and a runnable proof |

## The practice / solution workflow

1. **Read the worked notebook** (`NN_*.ipynb`) top to bottom and run it. Each cell states the security
   idea, runs it, and asserts the property it claims.
2. **Do the practice notebook** (`practice/NN_*_practice.ipynb`). It keeps the narrative but replaces the
   key lines with `____` blanks (an argument, a method name, an expected value) or a
   `raise NotImplementedError("fill me")` in a function body. Every exercise ends with `assert` checks —
   if the cell runs silently, you got it right. Practice notebooks will not run until the blanks are filled.
3. **Compare with the solution** (`solutions/NN_*_solution.ipynb`) — the completed practice notebook.
   Every solution executes cleanly end to end.
4. **Say it out loud.** The last cell of every notebook is the one-minute version; the notebooks exist so
   that you can *show* each claim (replay fails, audience mismatch is rejected, the hijacked model is
   contained) rather than describe it.

## Running

```bash
pip install -e ".[dev]"              # from the repository root
python3 -m ipykernel install --user --name python3   # only if the kernel is missing
jupyter lab notebooks/
```

Conventions inside the notebooks:

* The first cell always calls `quiet_logs()` (ADK, uvicorn and httpx are chatty) and, where the demo
  CRM data is used, `reset_demo_state()`. Run notebooks top to bottom; restart the kernel before a
  re-run, since demo state (`REFUNDS`, `OUTBOX`, MCP `TICKETS`) is module-level.
* Top-level `await` is used for the ADK loop (`run_turn`, `confirm`, `resume_after_auth`).
* Notebook 06 starts the tickets MCP server on a free localhost port in a background thread and stops it
  in the last cell; if a cell fails midway, run the `server.stop()` cell before restarting.
* Notebook 05 contains a guarded `ModelArmorScreener(template)` cell (`RUN_ON_GCP = False`) showing the
  one-line swap from the local screener to Model Armor on a real project.

To verify that the worked and solution notebooks still execute (what CI would run):

```bash
mkdir -p /tmp/nb-out
for nb in notebooks/*.ipynb notebooks/solutions/*.ipynb; do
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 \
    --output "/tmp/nb-out/$(basename "$nb")" "$nb" || echo "FAILED: $nb"
done
```

The repository copies are stored without outputs; keep it that way
(`jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb`) so diffs stay reviewable.
