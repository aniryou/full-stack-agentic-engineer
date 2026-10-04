# identity-security — give an agent its own identity and keep its authority bounded

After this topic, you can explain these things about an agent:

- Who the agent is.
- Whose authority the agent acts under.
- What stops the agent if it tries to make a user's grant wider.
- Where the system enforces the policy when someone hijacks the model.
- How the system records every decision in the audit, with both identities.

You can also show each of these things in code that runs on a laptop.

## Start here

1. Read §0 and §3 of the [identity primer](agentic-identity-gcp-lab/docs/primer.md), about 40 min. In §0, you
   get the mental model in one page. In §3, you read about principals, and about own authority against delegated
   authority.
2. Run `cd agentic-identity-core && python3 -m pip install -r requirements.txt && python3 agentsec_core.py`. It takes
   a second. It shows the five moves from end to end, with the audit timeline. The five moves are identity,
   authority, policy, resource and audit. Then run `python3 -m pytest -q`. There are 32 tests. 3 of them skip
   without the optional Mistral client or a key.
3. Open the core's [`notebooks/core_walkthrough.ipynb`](agentic-identity-core/notebooks/core_walkthrough.ipynb). Then
   do `notebooks/core_practice.ipynb`. Then continue to the next level: notebooks 01–09 of the Google Cloud lab.

## What you get

*Tiers: T0 is a laptop or a Colab CPU, at no cost (no GPU, no key, no cloud account). T3 is the Google Cloud
deployment, and it is optional.* The times are approximate. The core and the GCP lab together are module 06.6 in
[`CURRICULUM.md`](../../CURRICULUM.md).

| Path | You will be able to… | Time | Tier |
|---|---|---|---|
| [`agentic-identity-core/`](agentic-identity-core/README.md) | Show the five moves in one file. The file has an agent principal and a delegated token that names the user and the agent (RFC 8693 token exchange with `may_act`). It also has a deny-by-default policy with human confirmation, a tool server that authorizes by the verified subject, and one audit event for each decision. There are walkthrough, practice and solution notebooks. | 1–2 h | T0 |
| The same core on Mistral's platform: `agentic-identity-core/agentsec_core_mistral.py` and its `core_mistral_*` notebooks | Put the five moves, imported unchanged, behind a real model that does function calling (with an offline scripted twin). Use Mistral's moderation classifier as the screener, and use per-agent keys. Show where each control lives when the platform gives you the model and connectors but not the identity plane. | +1 h | T0 (the notebooks use the optional client in `requirements-mistral.txt`, and a `MISTRAL_API_KEY` adds the live model) |
| [`agentic-identity-gcp-lab/`](agentic-identity-gcp-lab/README.md) | Build each move in production shape. The lab has SPIFFE principals and certificate-bound tokens, DPoP, and a credential broker with a consent round-trip. It also has the policy as a plugin on a real agent runner. It has prompt-injection screening and provenance fencing, an MCP server as an OAuth 2.1 resource server, signed A2A agent cards, audit and governance. It has the [primer](agentic-identity-gcp-lab/docs/primer.md) (13 sections, design drills), 9 notebooks with practice and solutions, and 41 tests. It has Terraform for Google Cloud. | ~10 h at T0 (+ optional T3) | T0 (T3 optional) |

## Run it

```bash
cd agentic-identity-core && python3 -m pip install -r requirements.txt && python3 -m pytest -q    # 32 tests, ~2 s
cd ../agentic-identity-gcp-lab && python3 -m pip install -e ".[dev]" && python3 -m pytest -q     # 41 tests, ~6 s, offline
agentsec demo                                                                                    # policy, confirmation, audit trail
cd ../agentic-identity-core && python3 -m pip install -r requirements-mistral.txt                # optional: the Mistral path's client
```

Open the notebooks in JupyterLab (`python3 -m pip install jupyterlab`) or from the Colab links in the
[layer README](../README.md#run-in-colab). The deploy path of the GCP lab is in its
[`docs/deploy.md`](agentic-identity-gcp-lab/docs/deploy.md).

## How it fits

This topic needs nothing from the layers below it. It is the control plane of the gateway, and it runs on local
fakes. It pairs with [`scaling-admission-cost/`](../scaling-admission-cost/README.md). That topic says how much an
agent can run, and this topic says what the agent can do.

This topic leads to layer 07. There, the policy permits or stops each tool call of the agent loop
([`agent-fundamentals`](../../07-application-agent-framework/README.md)). This topic also leads to
[sandboxed execution](../../07-application-agent-framework/sandboxed-execution/README.md). That topic starts from
§6.2 of the primer (code execution). Sandboxed execution keeps secrets out of the sandbox with the same token
discipline. In the [curriculum's spiral](../../CURRICULUM.md#31-why-this-order), this topic is step 25, after the scaling lab.

## Caveats

- At T0, the identity plane is local fakes with the same semantics: a local CA, token issuer and credential broker.
  The GCP lab connects to the agent identity, credential broker, agent runtime and screening products of Google
  Cloud. It does this only when you set `AGENTSEC_PROFILE=gcp` and apply its Terraform. The facts about these
  products are a September 2026 snapshot. Several of the products went GA in 2026. The Verify list in §13 of the primer says what to examine again (verify).
- The connector scopes, guardrail fields and model names of the Mistral path are from a check in September 2026
  (verify).
- [`llm-gateway`](../llm-gateway/README.md) (module 06.7) covers the MCP client-side authorization flow (discovery,
  then PKCE, then code, then token), DPoP nonces, the SPIFFE Workload API and SVID rotation. Its primer covers the
  flow and the nonces in §8. It covers the Workload API and rotation in brief in §6.5.
