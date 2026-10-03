# Notebooks — hands-on companions to `docs/primer.md`

This folder has nine worked notebooks. Each worked notebook has a fill-in-the-blank **practice** twin
beside it. Each also has a completed **solution** in [`../solutions/`](../solutions/), under the name
of the practice notebook.

Everything runs offline against the local twins in `src/agentsec/`. These twins are `LocalRuntimeCA`,
`TokenIssuer`, `LocalAuthManager` and `LocalScreener`. They also include the tickets MCP server in a
background thread and the ADK loop that `ScriptedLlm` drives. A Google Cloud project and an API key are
not necessary. Every notebook ends with a "In one sentence" line. This line is the 30-second version of
the idea that the notebook shows.

## Map: notebook → primer section

| Notebook | Primer | What you demonstrate |
|---|---|---|
| `01_agent_identity_and_principals` | §3.1–§3.3 | SPIFFE IDs and `principal://` members. A `principalSet` match on **exact** segments. A runtime CA certificate with the SPIFFE SAN. A certificate-bound token (`cnf.x5t#S256`). A replay from another certificate that fails with `BindingMismatch`. The own and the delegated `AuthorityContext`, and `audit_identities()`. |
| `02_delegation_and_token_exchange` | §3.5, §4.3 | RFC 8693 exchange with the `act` claim. Scope narrowing. Nested `act.act` chains for sub-agents. Audience, expiry and scope failures. DPoP proofs, `ath` binding and `jti` replay. Credential Access Boundary JSON and its local evaluation. |
| `03_auth_manager_broker` | §4.3, §5 | 3LO, 2LO and API-key providers. The IAM binding on the provider. The `retrieveCredentials` outcomes. Consent and `finalize`. An access log that identifies the agent **and** the user. The ADK path: `crm_lookup`, then `adk_request_credential`, then finalize, then `resume_after_auth`. |
| `04_policy_enforcement_point` | §4.1, §4.2, §4.4 | `policies/support-agent.yaml`, evaluated request by request. The evaluation covers unknown tool, principal, authority, scopes, constraints, egress, budgets, confirmation and the `unless` envelope. A `dry_run` of a plan. The ADK loop with `LocalStack`: default deny, auto-allow inside the envelope, confirmation approve/reject, `audit.timeline()`. |
| `05_prompt_injection_and_guardrails` | §6 | `LocalScreener` (Model Armor shape) on injection, SDP and malicious URIs. The poisoned KB article. A hijacked model that deterministic controls contain. A blocked prompt that never reaches the model (`stack.llm.requests == []`). `EgressPolicy` SSRF cases. What Model Armor floor settings and templates do on GCP. |
| `06_mcp_resource_server` | §7.1 | Protected Resource Metadata and the 401 challenge. An incorrect audience gets 401. A request without the necessary scope gets 403. `tools/list` annotations. Read scope against write scope on `tools/call`. A server where DPoP is necessary: it rejects the bearer token, accepts the proof and rejects the replay. ADK `McpToolset` with delegated audience-bound tokens. Row-level filters by subject. The separate upstream token (no passthrough). |
| `07_a2a_agent_cards` | §7.2 | An Agent Card that you build, sign and verify. Tamper detection. Necessary scopes. `token_for_peer` and `authorize_inbound` with the hop chain. The rejection of a forwarded user token. A second hop. |
| `08_audit_and_governance` | §9, §4.5 | Queries on the `AuditLog`: by agent, denials, approvals with approver, dual identity. A policy kill switch that takes effect without a redeploy. The revocation of Auth Manager consent. A burst-detection heuristic over destructive attempts. |
| `09_code_evaluation_drills` | §11.2 | Eight "spot the bug" snippets. The bugs are token passthrough, `aud` not checked, allow-unknown-tools and a confirmation UI that shows the model's summary. The other bugs are a secret in `tool_context.state`, a substring egress match, a DPoP verifier that keeps no record of `jti` values and a `principalSet` prefix match. Each snippet has the correction and a runnable proof. |

## The practice / solution workflow

1. **Read the worked notebook** (`NN_*.ipynb`) from top to bottom. Then run it. Each cell states the
   security idea, runs it, and asserts the property that it claims.
2. **Do the practice notebook** (`NN_*_practice.ipynb`, in this folder). It keeps the narrative. But it
   replaces the key lines with `____` blanks (an argument, a method name, an expected value) or with a
   `raise NotImplementedError("fill me")` in a function body. Every exercise ends with `assert` checks.
   If the cell runs with no output, your answer is correct. Practice notebooks do not run until you
   fill the blanks.
3. **Compare with the solution** (`../solutions/NN_*_practice.ipynb`, the same file name). The solution
   is the completed practice notebook. Every solution runs from end to end with no error.
4. **Say it out loud.** The last cell of every notebook is the one-minute version. The notebooks exist
   so that you can *show* each claim, and you do not only describe it. For example, the replay fails,
   the check rejects the audience mismatch, and the controls contain the hijacked model.

## Running

```bash
pip install -e ".[dev]"              # from the repository root
python3 -m ipykernel install --user --name python3   # only if the kernel is missing
jupyter lab notebooks/
```

Conventions inside the notebooks:

* The first cell always calls `quiet_logs()`, because ADK, uvicorn and httpx write many log lines.
  Where a notebook uses the demo CRM data, the first cell also calls `reset_demo_state()`. Run the
  notebooks from top to bottom. Restart the kernel before a re-run, because the demo state
  (`REFUNDS`, `OUTBOX`, MCP `TICKETS`) is module-level.
* The notebooks use top-level `await` for the ADK loop (`run_turn`, `confirm`, `resume_after_auth`).
* Notebook 06 starts the tickets MCP server on a free localhost port in a background thread. It stops
  the server in the last cell. If a cell fails before the last cell, run the `server.stop()` cell
  before you restart.
* Notebook 05 contains a guarded `ModelArmorScreener(template)` cell (`RUN_ON_GCP = False`). This cell
  shows the one-line change from the local screener to Model Armor on a real project.

To make sure that the worked and solution notebooks still run, use these commands from the lab
folder. The manual `solutions` job of the root [`tests` workflow](../../../../.github/workflows/tests.yml)
runs the same commands:

```bash
mkdir -p /tmp/nb-out/notebooks /tmp/nb-out/solutions
for nb in $(ls notebooks/*.ipynb | grep -v '_practice\.ipynb$') solutions/*.ipynb; do
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=180 \
    --output "/tmp/nb-out/$nb" "$nb" || echo "FAILED: $nb"
done
```

The repository keeps its copies without outputs. Keep them that way
(`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb solutions/*.ipynb`), so that the diffs
stay easy to review.
