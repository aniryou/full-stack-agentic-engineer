# identity-security — give an agent its own identity and keep its authority bounded

After this topic you can say who an agent is, whose authority it acts under, what stops it from widening a user's
grant, where policy is enforced when the model is hijacked, and how every decision is audited with both identities —
and show each of those in code that runs on a laptop.

## Start here

1. Read the [identity primer](agentic-identity-gcp-lab/docs/primer.md) §0 (the mental model in one page) and §3
   (principals, own vs delegated authority), about 40 min.
2. `cd agentic-identity-core && python3 -m pip install -r requirements.txt && python3 agentsec_core.py` — a second:
   the five moves end to end (identity, authority, policy, resource, audit) with the audit timeline; then
   `python3 -m pytest -q` (13 tests).
3. Open the core's [`core_walkthrough.ipynb`](agentic-identity-core/core_walkthrough.ipynb), then do
   `core_practice.ipynb`; then step up to the Google Cloud lab's notebooks 01–09.

## What you get

*Tiers: T0 = laptop or Colab CPU, free (no GPU, no key, no cloud account); T3 = the Google Cloud deployment, optional.*
Times are rough; the core and the GCP lab together are module 06.6 in [`CURRICULUM.md`](../../CURRICULUM.md).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`agentic-identity-core/`](agentic-identity-core/README.md) | the five moves in one ~300-line file: an agent principal, a delegated token naming user and agent (RFC 8693 token exchange with `may_act`), deny-by-default policy with human confirmation, a tool server that authorizes by the verified subject, one audit event per decision; walkthrough, practice and solution notebooks | 1–2 h | T0 |
| [`agentic-identity-gcp-lab/`](agentic-identity-gcp-lab/README.md) | each move in production shape: SPIFFE principals and certificate-bound tokens, DPoP, a credential broker with a consent round-trip, the policy as a plugin on a real agent runner, prompt-injection screening and provenance fencing, an MCP server as an OAuth 2.1 resource server, signed A2A agent cards, audit and governance; the [primer](agentic-identity-gcp-lab/docs/primer.md) (13 sections, design drills), 9 notebooks with practice and solutions, 41 tests; Terraform for Google Cloud | ~10 h at T0 (+ optional T3) | T0 (T3 optional) |
| [`agentic-identity-core-mistral/`](agentic-identity-core-mistral/README.md) | the same core on Mistral's platform: a real model doing function calling (with an offline scripted twin), Mistral's moderation classifier as the screener, and where each control lives when the platform gives you the model and connectors but not the identity plane | 1–2 h | T0 (a `MISTRAL_API_KEY` adds the live model) |

## Run it

```bash
cd agentic-identity-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q    # 13 tests, ~1 s
cd ../agentic-identity-gcp-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q     # 41 tests, ~6 s, offline
agentsec demo                                                                                    # policy, confirmation, audit trail
cd ../agentic-identity-core-mistral && python3 -m pip install -r requirements.txt && python3 -m pytest -q   # 16 offline tests
```

Open the notebooks in JupyterLab (`python3 -m pip install jupyterlab`) or from the Colab links in the
[layer README](../README.md#run-in-colab). The GCP lab's deploy path is in its
[`docs/deploy.md`](agentic-identity-gcp-lab/docs/deploy.md).

## How it fits

Needs nothing from the layers below: this is the gateway's control plane, and it runs on local fakes. It pairs with
[`scaling-admission-cost/`](../scaling-admission-cost/README.md) (how much an agent may run; this topic says what it
may do). It leads to layer 07: the agent loop whose tool calls the policy gates
([`agent-fundamentals`](../../07-application-agent-framework/README.md)), and
[sandboxed execution](../../07-application-agent-framework/sandboxed-execution/README.md), which starts from the
primer's §6.2 (code execution) and keeps secrets out of the sandbox with the same token discipline. In the
[curriculum's spiral](../../CURRICULUM.md#31-why-this-order) it is step 24, after the scaling lab.

## Caveats

- At T0 the identity plane is local fakes with the same semantics (a local CA, token issuer and credential broker);
  the GCP lab binds to Google Cloud's agent identity, credential broker, agent runtime and screening products only
  when you set `AGENTSEC_PROFILE=gcp` and apply its Terraform. Those products are a September 2026 snapshot, several
  went GA in 2026; the primer's §13 Verify list says what to re-check (verify).
- The Mistral variant's connector scopes, guardrail fields and model names were checked in September 2026 (verify).
- Not covered yet: the MCP client-side authorization flow (discovery → PKCE → code → token), DPoP nonces, the SPIFFE
  Workload API and SVID rotation ([`CURRICULUM.md`](../../CURRICULUM.md) §2).
