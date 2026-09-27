# Facts — 07 · agent-memory (researched 2026-09-26)

Source of truth for the `agent-memory` primer, `memory-core` (`memcore`) and `memory-lab` (`memlab`). Each fact names its source; `$SP/ref/<repo>` = a shallow
clone of 2026-09-26 (§0); **(unverified)** = not found in a source, with the reason; environment facts are in `FACTS.md` (structure-plan builds). **Corrections to
the brief:** mem0 is ADD-only (§5); Graphiti has no `valid_to` (§6); Letta's Python server is archived (§5b); OTel GenAI has memory operations (§15); the code's
generative-agents recency gives the *oldest* node the largest value (§2); vLLM's `cache_salt` is ≤ 128 characters (§11).

## 0. Clones read (all `git clone --depth 1`)
| ref dir | upstream | commit, date | version / licence |
|---|---|---|---|
| `generative_agents` | joonspk-research/generative_agents | fe05a71, 2023-08-11 | Apache-2.0 |
| `LongMemEval` | xiaowu0162/LongMemEval | 9e0b455, 2026-05-11 | code MIT (`LICENSE`) |
| `locomo` | snap-research/locomo | 3eb6f2c, 2024-08-12 | **CC BY-NC 4.0** (`LICENSE.txt`, whole repo) |
| `mem0` | mem0ai/mem0 | 94c3fe9, 2026-09-25 | `mem0ai` 2.2.1 (`pyproject.toml`), Apache-2.0 |
| `letta`, `letta-archive`, `letta-code` | letta-ai/letta (main, `archive` branch), letta-ai/letta-code | 5bcdd17; 56ba9c2 (0.16.8); 31887d9 (0.33.2) | Apache-2.0 |
| `graphiti` | getzep/graphiti | ba4a9cb, 2026-09-25 | `graphiti-core` 0.30.2, Apache-2.0 |
| `zep` | getzep/zep | 495bf72, 2026-09-11 | Apache-2.0 |
| `langmem` | langchain-ai/langmem | 9d033b4, 2026-09-08 | 0.0.30, MIT |
| `adk-python` | google/adk-python | 044a1ec, 2026-09-25 | `__version__ = "2.10.0"` (`src/google/adk/version.py`), Apache-2.0 |
| `cognee` | topoteretes/cognee | eb90d03, 2026-09-24 | 1.6.1, Apache-2.0 |
| `pgvector`, `pgvector-python` | pgvector/pgvector, pgvector/pgvector-python | 7db2345; 60739df | extension 0.8.6 (`vector.control`), PostgreSQL License; python 0.5.0, MIT |
| `vllm` | vllm-project/vllm | ced6857 = **tag v0.30.0** | Apache-2.0 |
| `sqlite` | sqlite/sqlite (sparse `src`, `ext/fts5`) | 2acb2ea (VERSION 3.54.0) | public domain |
| `bedrock-agentcore-sdk-python`, `anthropic-sdk-python` | aws/…, anthropics/… | c7423e5 (1.23.1); 4421d56 | Apache-2.0; MIT |
| `semantic-conventions-genai` | open-telemetry/semantic-conventions-genai | e57c543, 2026-09-24 | Apache-2.0 |
| `AgentPoison`, `cloud-run-samples` | BillChan226/AgentPoison, GoogleCloudPlatform/cloud-run-samples | 7236bf4; cd2eb7e | MIT; Apache-2.0 |

## 1. Repo numbers the core reproduces (all recomputed here on 2026-09-26)
- **`ragkit.embed.HashingEmbedder(dim=1024)`** (`07-application-agent-framework/retrieval-rag/rag-from-scratch/ragkit/embed.py`): per token of
  `ragkit.corpus.tokenize` (`_TOKEN_RE = re.compile(r"[a-z0-9]+")` on `text.lower()`), `v[zlib.crc32(tok.encode("utf-8")) % self.dim] += 1.0`, then `v / norm` (zero
  vector stays zero). `model_name = FALLBACK_LABEL = "hashing embedder (T0 fallback; not semantic)"`; forced by `RAGKIT_EMBEDDER=hashing`; the real default is
  `_DEFAULT_MODEL = "all-MiniLM-L6-v2"`. `encode(str)` returns a **1-D** `(1024,)` float32 vector; `encode(list)` returns `(n, 1024)`. Pinned values: buckets
  `user→585, lives→606, in→590, lisbon→181, the→486, moved→501, to→708, porto→130`; cos("user lives in lisbon", "the user moved to porto") = 1/√(4·5) = **0.2236**;
  cos("user lives in lisbon", "Where does the user live?") = **0.2236** (only `user` shared — `live` ≠ `lives`: the paraphrase miss by design).
- **`ragkit.reference.reciprocal_rank_fusion(rankings, k=60)`**: `scores[doc_id] += 1.0 / (k + rank + 1)`, `rank` from 0. `BM25(corpus_tokens, k1=1.5, b=0.75)`,
  `mmr(query_vec, cand_vecs, cand_ids, lambda_=0.7, k=5)` (`ragkit/reference.py`).
- **`expected_cached_tokens(prev, new, block_size=16)`** — defined only in the solution of exercise 4.1 in
  `04-inference-engine/serving-engine/vllm-serving-lab/notebooks_src/04_prefix_caching_for_agents.py` (lines 98–106), not in `servelab`:
  ```python
  common = 0
  for x, y in zip(prev, new):
      if x != y: break
      common += 1
  blocks = min(common // block_size, (len(prev) - 1) // block_size, (len(new) - 1) // block_size)
  return blocks * block_size
  ```
  Rules: only full blocks of the common prefix; no more than `prev` left cached; never the last token of `new`. Asserts: identical 64-token prompts → **48**;
  `range(40)` then `range(40)+[7]*30` → **32**; `[1]*20+[2]*20` vs `[1]*20+[3]*20`, B = 8 → **16**.
- **`minengine.kv`** (`04-inference-engine/serving-engine/mini-engine-core/minengine/kv.py`): `hash_block(parent, tokens, extra=None)` = `sha256(repr((parent,
  tuple(int(t) for t in tokens), extra)))`; `block_hashes(tokens, block_size, extra=None, hash_fn=hash_block)` names full blocks only; `KVCacheManager(num_blocks,
  block_size=16, enable_prefix_caching=True, ...)`; `.lookup(tokens, extra=None)` walks `range((len(tokens) - 1) // B)` and stops at the first miss. Note: minengine
  passes `extra` into every block's hash; vLLM puts `cache_salt` into the first block only — equivalent, because each hash chains its parent (§11).
- **`minengine.perf.step_cost(gpu, llm, chunks, flop_eff=0.6, bw_eff=0.8, overhead_s=0.002)`**, chunks `[(start, n)]`; one 2,000-token prefill cold `[(0, 2000)]` vs
  `[(1800, 200)]` (ms, SIMULATED roofline): L4 `qwen2.5-1.5b` **78.7** (compute) vs **15.1** (memory); L4 `llama-3.1-8b` **401.0** vs **65.6**; H100-SXM
  `qwen2.5-1.5b` **11.4** vs **3.2**; H100-SXM `llama-3.1-8b` **50.8** vs **7.7**. `GPUS["L4"] = GPU("L4", 121e12, 300e9, 24e9, fp8=True)`, `GPUS["H100-SXM"] =
  GPU("H100-SXM", 989e12, 3.35e12, 80e9, fp8=True)`.
- **`capacity.ttft_s(active_b, prompt_tokens, gpu, dtype="fp8", mfu=0.5, model=None)`** (`00-foundations/gpu-capacity-planning/capacity.py`): `ttft_s(24, 2000,
  GPUS["H100"])` = **0.0970 s**.
- **`scalelab.capacity.cost_per_call(model, input_tokens, output_tokens, cached_tokens=0)`**
  (`06-gateway/scaling-admission-cost/agentic-scaling-lab/scalelab/capacity.py`): `(uncached*inp + cached*cached_rate + out*out_rate)/1e6`;
  `PRICES["gemini-3.5-flash"] = (1.5, 9.0, 0.15, 675, 6)`; `cost_per_call("gemini-3.5-flash", 5000, 350, 2700)` = (2300·1.5 + 2700·0.15 + 350·9.0)/1e6 = 7005/1e6 =
  **$0.007005** (scaling primer §3.4 "Cost per conversation"; prices of 5 Sep 2026, verify).
- **`agentlab.estimation.calc.token_cost(in_tokens, out_tokens, price, cached_share=0.0, batch=False)`**; `PRICES["gemini-3-flash"] = Price(input=0.5,
  cached_input=0.05, output=3.0, long_context=None, long_context_threshold=200000)` (illustrative, verify); `token_cost(5000, 350, PRICES["gemini-3-flash"],
  cached_share=0.54)` = (2300·0.5 + 2700·0.05 + 350·3.0)/1e6 = **$0.002335**. `calc.py` and `llm/types.py` load standalone by path
  (`importlib.util.spec_from_file_location`); `evals/gate.py` needs pydantic.
- **`wilson_interval(passes, n, z=1.96)`** (`agentlab/evals/gate.py`, restate — do not import): `n == 0 → (0.0, 1.0)`; `centre = (p + z²/2n)/(1 + z²/n)`, `half =
  z·sqrt(p(1−p)/n + z²/4n²)/(1 + z²/n)`, clipped to [0, 1]. Pins: `(45, 50)` → **(0.7864, 0.9565)**; `(0, 20)` → **(0.0, 0.1611)**; `(20, 20)` → (0.8389, 1.0).
  `half_width_rule_of_thumb(n) = 1/√n`.
- **`count_tokens(text)`** (`agentlab/llm/types.py`): `0` if empty else `max(1, len(text) // 4)`.
- **`ContextBuilder`** (`agentlab/agents/context.py`): `max_recent_turns=12`, `memory_provider: Callable[[Session], list[str]] | None` ("per-user facts, injected
  after the static prefix"), `memory_messages()` → one system message `"# Known about this user\n- fact..."`, `cacheable_prefix_tokens(session)`, module-level
  `context_report(messages)`. `agentlab/llm/fake.py` `PrefixCache` "simulates implicit context caching" by hashing message prefixes.
- **Audit**: `06-gateway/identity-security/agentic-identity-gcp-lab/src/agentsec/audit/log.py` `args_digest(args)` = `sha256(canonical JSON).hexdigest()[:16]`;
  `AuditEvent.event_type` comment lists `tool.decision | tool.result | model.screen | credential.retrieve | egress | confirmation` — `memory.write` / `memory.read` /
  `memory.forget` are new values. Standalone mirror to copy: `07-application-agent-framework/sandboxed-execution/sandbox-core/sandboxcore/audit.py`
  (`sandbox.decision | sandbox.result | egress | confirmation`); `sandboxcore/agent.py` has `ScriptedLLM`, `SandboxAgent`, `injection_scenarios(policy,
  trap=("127.0.0.1", 9))`.
- **agent-core** (`agentcore/agent.py`): `Agent(llm, tools=None, instruction="You are a helpful assistant.", max_steps=6)`, `.run(user_message, history=None,
  on_confirm=None)` → `Result`; `agentcore/tools.py` `@tool` / `@tool(confirm=True)`, `ToolError(message, kind="tool_error", hint=None)`; `agentcore/fake_llm.py`
  `FakeLLM`, `text()`, `call()`, `calls()`. Notebook `notebooks/03_state_and_control.ipynb`.
- **Durable** (on `main` today; c1-durable moves them — cite post-move paths per SPEC line 474): `long-running-durable/00_primer.md` (→ `PRIMER.md`) §3.2
  "Idempotency — effectively-once, not exactly-once", §3.3 "Exclusivity — leases, not locks", §3.4 "Boundedness — budgets are code", §3.5 "Context hygiene — every
  fact has a shelf life"; `lra/lra-gcp/docs/primer.md` (→ `lra-gcp/docs/primer.md`) §3.3 "Leases and the reaper (crash recovery)", §3.8 "Reflection
  (evaluator–optimizer) as durable steps", §3.9, §3.10 "Memory and context management", §3.13 "Scheduled and event-triggered runs"; `lra-core/lra-core/core.py` (→
  `lra-core/core.py`) `Engine(store, queue, steps, clock=time.time, lease_ttl=60, max_attempts=3)`, `Store.effect_once(key, fn)`, `Engine.reap()`. Scheduler
  patterns: `lra/lra-gcp/infra/ terraform/main.tf` `google_cloud_scheduler_job.reaper` (`*/2 * * * *`, `oidc_token` with `audience` = the worker's own run.app URI)
  and `lra-core/lra-core/gcp/deploy.sh` (`gcloud scheduler jobs create http lra-reap --schedule="*/2 * * * *" ... --oidc-service-account-email`).
- **Section titles verified**: vector-databases primer "### 8. What the database layer adds", "### 9. Metadata filtering — the hardest "easy" problem", "### 11.
  Scale-out: multi-tenancy, sharding, replication, storage tiers", "### 17. Use-case patterns" (its "Agent memory" bullet), "### 18. Operating a vector database in
  production", "### 19. Common pitfalls"; embeddings primer "## 15. Retrieval pipelines for RAG and agents" (Memory bullet cites Park et al., 2023; Poisoning bullet
  cites PoisonedRAG, Zou et al., 2024); identity primer "## 2. Threat model" (ASI06 row), "### 3.5 Delegation mechanics", "### 6.1 The untrusted-content boundary",
  "## 8. Data boundaries, network, and tenancy", "## 9. Observability, audit, and governance"; serving-engine PRIMER "## 5. Prefix caching" (60-request simulation:
  hit rate 84%, TTFT p50 13 ms vs 74 ms, SIMULATED); vllm-internals "### 4.1 Object model", "### 4.3 Block hashes: a chain over the prefix"; scaling primer "###
  1.4", "### 3.4 Cost per conversation", "### 5.5 Context engineering for scale" (385 k vs 184 k input tokens over 20 turns; cache minimum 4,096 tokens on Gemini
  3.x, 6,144 on 3.7/3.8 Flash and 3.1 Pro), "### 5.9 State stores".
- `minifaiss` (`retrieval-rag/vector_stores/minifaiss/hnsw.py`): `IndexHNSWFlat(d, M=16, ...)`, `M0 = 2 * M`; no delete/update.
- Serving lab `deploy/any-gpu/serve.sh`: `MODEL` default `Qwen/Qwen2.5-0.5B-Instruct`, `IMAGE` default `vllm/vllm-openai:v0.30.0`, `EXTRA_ARGS`, `DRY_RUN=1` prints
  only; its README row: `--enable-prompt-tokens-details` "adds `usage.prompt_tokens_details.cached_tokens`".

## 2. Generative agents: retrieval and reflection (`$SP/ref/generative_agents/reverie/backend_server/persona/`)
- Paper: Park et al., "Generative Agents: Interactive Simulacra of Human Behavior", UIST '23, arXiv 2304.03442 (`README.md` bibtex).
- **`cognitive_modules/retrieve.py` `new_retrieve(persona, focal_points, n_count=30)`** — quoted:
  ```python
  nodes = [[i.last_accessed, i] for i in persona.a_mem.seq_event + persona.a_mem.seq_thought if "idle" not in i.embedding_key]
  nodes = sorted(nodes, key=lambda x: x[0])          # ascending last_accessed: OLDEST first
  recency_out = normalize_dict_floats(extract_recency(persona, nodes), 0, 1)
  importance_out = normalize_dict_floats(extract_importance(persona, nodes), 0, 1)
  relevance_out = normalize_dict_floats(extract_relevance(persona, nodes, focal_pt), 0, 1)
  gw = [0.5, 3, 2]                                   # comments: "# gw = [1, 1, 1]", "# gw = [1, 2, 1]"
  master_out[key] = (persona.scratch.recency_w*recency_out[key]*gw[0]
                   + persona.scratch.relevance_w*relevance_out[key]*gw[1]
                   + persona.scratch.importance_w*importance_out[key]*gw[2])
  ...top n_count...;  for n in master_nodes: n.last_accessed = persona.scratch.curr_time
  ```
  `extract_recency`: `recency_vals = [persona.scratch.recency_decay ** i for i in range(1, len(nodes) + 1)]`, assigned in list order. **Settled:** recency is
  `recency_decay ** rank` over *all* nodes sorted by `last_accessed` ascending, so the **oldest-accessed node gets 0.99¹ (the largest) and the most recently accessed
  gets 0.99ᴺ (the smallest)** — the code's recency term favours stale memories (an inversion of the paper's intent). It is rank-based, not time-based: an hour and a
  year apart score alike. `extract_importance` = `node.poignancy`; `extract_relevance` = `cos_sim(node_embedding, get_embedding(focal_pt))`
  (`dot(a,b)/(norm(a)*norm(b))`; embeddings `text-embedding-ada-002`, `prompt_template/gpt_structure.py` `get_embedding`). Normalisation is over the whole memory
  stream, not a candidate set. `normalize_dict_floats(d, lo, hi)`: `(v − min)·(hi − lo)/range + lo`; if `range == 0` every value becomes `(hi − lo)/2` (0.5 for [0,
  1]). Retrieval also rewrites `last_accessed` of what it returns (reads are writes).
- **`memory_structures/scratch.py` defaults**: `recency_w = 1`, `relevance_w = 1`, `importance_w = 1`, `recency_decay = 0.99`, `importance_trigger_max = 150`,
  `importance_trigger_curr = importance_trigger_max`, `importance_ele_n = 0`, `thought_count = 5`, `vision_r = 4`, `att_bandwidth = 3`, `retention = 5`. Effective
  code weights = `gw` = recency 0.5, relevance 3, importance 2.
- **`memory_structures/associative_memory.py` `ConceptNode`**: `node_id, node_count, type_count, type` ("thought / event / chat"), `depth, created, expiration,
  last_accessed = created, subject, predicate, object, description, embedding_key, poignancy, keywords, filling` (`filling` holds evidence node ids for thoughts).
  Stream lists `seq_event`, `seq_thought`, `seq_chat`.
- **Importance**: `prompt_template/v3_ChatGPT/poignancy_event_v1.txt`: "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10
  is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy…". Idle events score 1 (`reflect.py generate_poig_score`).
- **Reflection** (`cognitive_modules/reflect.py`): `perceive.py` does `importance_trigger_curr -= event_poignancy; importance_ele_n += 1` per new event;
  `reflection_trigger` fires when `importance_trigger_curr <= 0` (i.e. new-event importance summed past **150**); `run_reflect`: `generate_focal_points(persona, 3)`
  over the **last `importance_ele_n` nodes** (the events since the last reflection, sorted by `last_accessed` — not "the 100 most recent"); `new_retrieve` (30 nodes
  per focal point); `generate_insights_and_evidence(persona, nodes, 5)` (prompt `v2/insight_and_evidence_v1.txt`: "What 5 high-level insights can you infer… (example
  format: insight (because of 1, 5, 3))"); each thought is stored with `expiration = curr_time + timedelta(days=30)`, its own poignancy and `evidence` = cited node
  ids; `reset_reflection_counter` restores 150 and zeroes `importance_ele_n`. Focal prompt `v2/generate_focal_pt_v1.txt`: "…what are 3 most salient high-level
  questions we can answer about the subjects in the statements?"
- **Paper form (unverified — only the paper states it; arxiv is blocked)**: score = α·recency + β·importance + γ·relevance with α = β = γ = 1; recency =
  0.995^(sandbox hours since last retrieval); each term min-max normalised to [0, 1]; reflection on importance sum > 150; "100 most recent records" → 3 questions → 5
  insights with evidence. The core implements this form by default and the code's (rank recency, oldest-first, `gw = [0.5, 3, 2]`, decay 0.99) as an option.
- **Worked example (computed; use as the hand test)**. Three memories (hours since last access, importance, cosine): A (1, 2, 0.80), B (24, 9, 0.50), C (72, 5,
  0.20). Normalised importance A 0, B 1, C 0.4286; relevance A 1, B 0.5, C 0.
  - Paper form: raw recency 0.995¹ = 0.9950, 0.995²⁴ = 0.8867, 0.995⁷² = 0.6970 → normalised A 1.0, B 0.6364, C 0; scores **A 2.0000, B 2.1364, C 0.4286** → B, A, C.
  - Code form: order oldest-first C, B, A → raw 0.99, 0.9801, 0.9703 → normalised **C 1.0, B 0.4975, A 0.0**; scores 0.5·R + 3·L + 2·I = **A 3.0000, B 3.7487, C
    1.3571** → B, A, C. With recency reversed (newest largest) A 3.5, B 3.7513, C 0.8571: same order here, but the code hands C its recency bonus.
- Licence: Apache-2.0 (`LICENSE`). Last commit 2023-08-11 (unmaintained).

## 3. LongMemEval (`$SP/ref/LongMemEval/README.md`, `src/evaluation/`)
- Wu, Wang, Yu, Zhang, Chang, Yu, "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory", arXiv 2410.10813, accepted at **ICLR 2025** (README
  news "[2025/02]"). 500 questions; **five abilities**: Information Extraction, Multi-Session Reasoning, Knowledge Updates, Temporal Reasoning, Abstention.
- **`question_type`** ∈ `single-session-user`, `single-session-assistant`, `single-session-preference`, `temporal-reasoning`, `knowledge-update`, `multi-session`.
  **Abstention: `question_id` ends with `_abs`** (`print_qa_metrics.py`: `if '_abs' in entry['question_id']`).
- Item fields: `question_id`, `question_type`, `question`, `answer`, `question_date`, `haystack_session_ids`, `haystack_dates`, `haystack_sessions` (list of
  sessions; each a list of `{"role", "content"}` turns; evidence turns carry `has_answer: true`), `answer_session_ids`.
- Variants: `longmemeval_s` ≈ 115k tokens (~40 sessions) per item (Llama 3 tokens); `longmemeval_m` ≈ 500 sessions per item; `longmemeval_oracle` = evidence sessions
  only. Current files are `longmemeval_s_cleaned.json` / `longmemeval_m_cleaned.json` (cleaned 2025/09) on HF `xiaowu0162/longmemeval-cleaned`. **LongMemEval-V2**
  (agentic memory) announced 2026/05.
- Judging (`evaluate_qa.py`): an LLM judge (`gpt-4o-2024-08-06` by default; `temperature 0`, `max_tokens 10`), `label = 'yes' in response.lower()`, one prompt per
  type: exact containment for `single-session-user`/`-assistant`/`multi-session`; temporal forgives off-by-one day counts; knowledge-update accepts old + updated
  info "as long as the updated answer is the required answer"; preference uses a rubric ("does not need to reflect all the points"); abstention asks whether the
  model "correctly identifies the question as unanswerable".
- Licence: code **MIT** (`LICENSE`, © 2024 Di Wu). Dataset licence on the HF card: (unverified — huggingface.co blocked; recalled as MIT).
- Per-type counts (unverified, paper table from memory): 70 single-session-user, 56 -assistant, 30 -preference, 133 multi-session, 78 knowledge-update, 133
  temporal-reasoning; 30 of the 500 are `_abs`. The harness copies shapes only; **nothing is downloaded**.

## 4. LoCoMo (`$SP/ref/locomo`)
- Maharana, Lee, Tulyakov, Bansal, Barbieri, Fang, "Evaluating Very Long-Term Conversational Memory of LLM Agents", **ACL 2024** (`README.MD` title), arXiv
  2402.17753. Tasks: QA, event summarisation, multimodal dialogue generation.
- **Licence: CC BY-NC 4.0** (`LICENSE.txt`, "Attribution-NonCommercial 4.0 International"; covers `data/locomo10.json`). Non-commercial → the repo bundles **none**
  of it; the harness copies the task shape only.
- `data/locomo10.json` = **10 conversations** (a subset of the 50 released on arXiv in March 2024). Computed here: sessions per conversation 19–32 (mean **27.2**);
  turns 369–689 (mean **588.2**); tokens by `len(text)//4` 10,896–22,434 (mean **≈ 18,169**). Keys per sample: `qa`, `conversation` (`session_<n>`,
  `session_<n>_date_time`, `speaker_a`, `speaker_b`; turns have `speaker`, `dia_id`, `text`), `event_summary`, `observation`, `session_summary`, `sample_id`.
- **QA categories — numeric only in the JSON; names from the evaluation code** (`task_eval/evaluation.py` lines 203–223, `gpt_utils.py` 243–253) plus evidence counts
  computed here: **1 = multi-hop** (partial F1 over comma-split sub-answers, `f1()`; 98% have >1 evidence turn; 282 Qs), **2 = temporal** (prompt appends "Use DATE
  of CONVERSATION to answer with an approximate date."; 321), **3 = open-domain / commonsense inference** (answer truncated at the first `;`; 96), **4 = single-hop**
  (841), **5 = adversarial** (unanswerable; scored 1 if the output contains "no information available" or "not mentioned"; 444 of 446 carry `adversarial_answer` and
  no `answer`). Total **1,986**. The code comment "single-hop, temporal, open-domain" over `[2, 3, 4]` is in a different order from the ids — do not map names by
  that comment; papers disagree on the numbering.
- Metric: token F1 (`f1_score`, with normalisation and Porter stemming) for 2/3/4, partial F1 for 1, a refusal check for 5; retrieval recall when `evidence` present.

## 5. mem0 (`$SP/ref/mem0`, `mem0ai` 2.2.1)
- **The ADD/UPDATE/DELETE/NONE write path is gone.** `docs/changelog/sdk.mdx` v2.0.0 (2026-04-14): "Single-Pass Extraction: Replaced 2-LLM-call pipeline with
  additive extraction using `ADDITIVE_EXTRACTION_PROMPT`. Memories accumulate via `linked_memory_ids`: no more UPDATE/DELETE events".
  `docs/migration/oss-v2-to-v3.mdx`: "`add()` events | Returns `ADD`, `UPDATE`, `DELETE` | Returns `ADD` only"; claims LoCoMo 71.4 → 91.6, LongMemEval 67.8 → 93.4
  (vendor numbers). `docs/core-concepts/memory-evaluation.mdx`: knowledge update (93.6) "remains the hardest category for an additive, ADD-only architecture".
- The **old** prompt still ships unused: `mem0/configs/prompts.py` `DEFAULT_UPDATE_MEMORY_PROMPT` — "You can perform four operations: (1) add into the memory, (2)
  update the memory, (3) delete from the memory, and (4) no change… ADD / UPDATE / DELETE / **NONE**" (the event is `"NONE"`, not NOOP); `FACT_RETRIEVAL_PROMPT` is
  the old extractor. Only `tests/configs/test_prompts.py` references them. So the primer states "extract → compare → ADD/UPDATE/DELETE/NONE" as **mem0 v1.x (before
  2.0.0, April 2026)** and the current mem0 as ADD-only.
- `Memory.add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, timestamp=None, expiration_date=None, infer=True, memory_type=None, prompt=None)`
  (`mem0/memory/main.py` line 760). One of the ids required; `infer=False` stores each non-system message raw with `"event": "ADD"`. Returns `{"results": [{"id",
  "memory", "event": "ADD", ...}]}`.
- v3 pipeline (`_add_to_vector_store`): last **10** messages of the session scope (`self.db.get_last_messages(session_scope, limit=10)`); existing memories = one
  vector search of the **whole new message text** with `top_k=10` (not per fact); UUIDs mapped to `"0"`, `"1"`… ("anti-hallucination"); one LLM call with
  `response_format={"type": "json_object"}`; dedup by `hashlib.md5(text).hexdigest()` against existing and in-batch hashes; payload `data`, `text_lemmatized` (BM25),
  `hash`, `created_at`, `updated_at`, `attributed_to`; history rows `event="ADD"`; then entity linking.
- `search(query, *, top_k=20, filters=None, threshold=0.1, rerank=False, show_expired=False, ...)`: ids go **inside `filters`** (`filters={"user_id": ...}`) —
  top-level `user_id` raises `ValueError`. Also `get(memory_id)`, `get_all(...)`, `update(memory_id, data)`, `delete(memory_id)`, `delete_all(user_id=None,
  agent_id=None, run_id=None)` (≥ 1 filter, else "use the `reset()` method"), `history(memory_id)`, `reset()`.
- **What `delete` leaves behind** (`_delete_memory`): `self.vector_store.delete(vector_id=memory_id)` then `self.db.add_history(memory_id, prev_value, None,
  "DELETE", ..., is_deleted=1)` — the **deleted text survives as `old_memory` in the SQLite `history` table**; `update` likewise stores `prev_value`. The `messages`
  table keeps raw turns (rolling 10 per `session_scope`). Default path `history_db_path = os.path.join(mem0_dir, "history.db")` (`mem0/configs/base.py`). Schema
  (`mem0/memory/storage.py`): `history(id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role)`; `messages(id,
  session_scope, role, content, name, created_at)`. Client `delete(..., delete_linked=False)` removes superseded linked memories transitively (sdk changelog, PR
  #5270).
- Licence Apache-2.0. Graph memory removed from OSS in 2.0.0 (Platform only).

## 5b. Letta / MemGPT
- `letta-ai/letta` **main is now a README**: "The current source code lives in `letta-ai/letta-code`"; the Python server is the **`archive` branch** ("the retired
  Letta V1 API server"), `pyproject.toml` version **0.16.8**. letta-code (TypeScript, npm `@letta-ai/letta-code`, 0.33.2) uses a `memory` tool and
  `memory_apply_patch` over a git-backed memory filesystem (grep of `src/`). Cite Letta concepts from the archive with that caveat.
- `letta/schemas/block.py`: `Block` fields `value: str`, `limit: int = CORE_MEMORY_BLOCK_CHAR_LIMIT`, `label` ("e.g. 'human', 'persona'"), `read_only: bool = False`.
  `letta/constants.py`: `CORE_MEMORY_BLOCK_CHAR_LIMIT = 100000`, `CORE_MEMORY_PERSONA_CHAR_LIMIT = 20000`, `CORE_MEMORY_HUMAN_CHAR_LIMIT = 20000` — limits are
  **characters**, not tokens.
- Tool sets (`constants.py`): `BASE_TOOLS = [send_message, "conversation_search", "archival_memory_insert", "archival_memory_search"]` (the two archival tools also
  in `DEPRECATED_LETTA_TOOLS`); `BASE_MEMORY_TOOLS = ["core_memory_append", "core_memory_replace", "memory", "memory_apply_patch"]`; `BASE_MEMORY_TOOLS_V2 =
  ["memory_replace", "memory_insert"]`; `BASE_MEMORY_TOOLS_V3 = ["memory"]`; `BASE_SLEEPTIME_TOOLS = ["memory_replace", "memory_insert", "memory_rethink",
  "memory_finish_edits"]`.
- Signatures (`letta/functions/function_sets/base.py`): `core_memory_append(agent_state, label, content)`, `core_memory_replace(agent_state, label, old_content,
  new_content)`, `memory_replace(agent_state, label, old_string, new_string)`, `memory_insert(agent_state, label, new_string, insert_line=-1)`,
  `memory_rethink(agent_state, label, new_memory)`, `archival_memory_insert(self, content, tags=None)`, `memory(agent_state, command, path=None, file_text=None,
  description=None, old_string=None, new_string=None, insert_line=None, insert_text=None, old_path=None, new_path=None)` with commands `create`, `str_replace`,
  `insert`, `delete`, `rename`.
- Overflow: `SUMMARIZATION_TRIGGER_MULTIPLIER = 0.9` ("Summarization triggers when step usage > context_window * 0.9"); `services/summarizer/thresholds.py
  get_compaction_trigger_threshold` returns `int(context_window * 0.9)` (its docstring says others use 100% — the code does not). `letta/settings.py`
  (`letta_summarizer_` env prefix): `mode = PARTIAL_EVICT_MESSAGE_BUFFER`, `message_buffer_limit = 60`, `message_buffer_min = 15`,
  `partial_evict_summarizer_percentage = 0.30` ("of message count, not token count"), `memory_warning_threshold = 0.75`, `send_memory_warning_message = False`,
  `desired_memory_token_pressure = 0.3`.
- Sleep-time: `AgentState.enable_sleeptime` — "If set to True, memory management will move to a background agent thread" (`letta/schemas/agent.py`). MemGPT paper's
  "warning near 70%, flush at 100%" is (unverified; paper only).

## 6. Graphiti (`$SP/ref/graphiti`, `graphiti-core` 0.30.2) and Zep
- `graphiti_core/edges.py` `EntityEdge`: `name`, `fact`, `fact_embedding`, `episodes` (episode ids), **`valid_at`** ("when the fact became true"), **`invalid_at`**
  ("when the fact stopped being true"), **`expired_at`** ("when the node was invalidated" — system time), `reference_time`, `attributes`; base `Edge` has `group_id`
  ("partition of the graph") and `created_at`. Bi-temporal = valid time (`valid_at`/`invalid_at`) + transaction time (`created_at`/`expired_at`). **There is no
  `valid_to` field**: `memcore` may name its field `valid_to` but must say it plays Graphiti's `invalid_at`, and keep a separate system-time
  `expired_at`/`superseded_at`.
- `graphiti_core/utils/maintenance/edge_operations.py` `resolve_edge_contradictions(resolved_edge, invalidation_candidates)`: skip candidates whose validity does not
  overlap the new edge; if `edge.valid_at < resolved_edge.valid_at` then `edge.invalid_at = resolved_edge.valid_at` and `edge.expired_at = edge.expired_at or
  utc_now()` — the old fact is **kept, closed, not deleted**. Timestamps come from an LLM call (`_extract_edge_timestamps`) relative to the episode's time.
- `Graphiti.add_episode(name, episode_body, source_description, reference_time, source=EpisodeType.message, group_id=None, uuid=None, update_communities=False,
  entity_types=None, ...)` (`graphiti_core/graphiti.py` line 1043).
- Search (`graphiti_core/search/search_config.py`): methods `cosine_similarity`, `bm25`, `breadth_first_search`; rerankers `rrf` (`reciprocal_rank_fusion`),
  `node_distance`, `episode_mentions`, `mmr`, `cross_encoder`; edge default reranker `rrf`. Recipes (`search_config_recipes.py`):
  `COMBINED_HYBRID_SEARCH_RRF|_MMR|_CROSS_ENCODER`, `EDGE_HYBRID_SEARCH_RRF|_MMR|_NODE_DISTANCE| _EPISODE_MENTIONS|_CROSS_ENCODER`, `NODE_HYBRID_SEARCH_*`,
  `COMMUNITY_HYBRID_SEARCH_*`. Graphiti's `rrf(results, rank_const=1)` scores `1 / (i + rank_const)` with `i` from 0 — effectively **k = 1**, not ragkit's k = 60.
- Zep: `getzep/zep` README — "Zep Community Edition is no longer supported. Its code has been moved to the `legacy/` folder"; the maintained server is Zep Cloud
  (managed). Graphiti is Zep's open-source engine.

## 7. LangMem (`$SP/ref/langmem`, 0.0.30, MIT)
- `langmem/knowledge/tools.py`: `create_manage_memory_tool(namespace, *, instructions=..., schema=str, actions_permitted=("create", "update", "delete"), store=None,
  name="manage_memory")` → tool `manage_memory(content=None, action="create", *, id=None)`; `create_search_memory_tool(namespace, *, instructions=..., store=None,
  response_format="content", name="search_memory")` → tool `search_memory(query, limit=10, offset=0, filter=None)`. Namespaces are tuples with runtime placeholders,
  e.g. `("memories", "{langgraph_user_id}")`, filled from the run config (`NamespaceTemplate`).
- `langmem/knowledge/extraction.py`: `create_memory_manager(model, /, *, schemas=(Memory,), instructions=..., enable_inserts=True, enable_updates=True,
  enable_deletes=False)`; `create_memory_store_manager(model, /, *, schemas=None, ..., enable_inserts=True, enable_deletes=False, query_model=None, query_limit=5,
  namespace=("memories", "{langgraph_user_id}"), store=None, phases=None)` — deletes are **off by default** in both.
- `langmem/reflection.py`: `ReflectionExecutor(...)` returns a `LocalReflectionExecutor` or `RemoteReflectionExecutor`; `submit(payload, ..., after_seconds=0,
  thread_id=...)`; a new submit for the same `thread_id` cancels the pending one (`existing.cancel_event.set(); existing.future.cancel()`) — a debounce: consolidate
  after the conversation goes quiet.
- `docs/docs/concepts/conceptual_guide.md`: types Semantic ("Facts & Knowledge"; profile or collection), Episodic ("Past Experiences"; few-shot examples, summaries),
  Procedural ("System Behavior"; prompt rules); formation "Active" (hot path: latency Higher, update Immediate) vs "Background" (latency None, Delayed,
  "Between/After Calls"); "Recall should combine similarity with 'importance'… and the memory's 'strength'… how recently/frequently it was used".

## 8. ADK memory and Vertex AI Agent Engine Memory Bank (`$SP/ref/adk-python`, ADK 2.10.0, Apache-2.0)
- `src/google/adk/memory/base_memory_service.py` `BaseMemoryService`: abstract `async add_session_to_memory(self, session)` ("A session may be added multiple times
  during its lifetime"); `async add_events_to_memory(self, *, app_name, user_id, events, session_id=None, custom_metadata=None)` (a delta; default raises
  `NotImplementedError`); `async add_memory(self, *, app_name, user_id, memories, custom_metadata=None)`; abstract `async search_memory(self, *, app_name, user_id,
  query) -> SearchMemoryResponse` (`memories: list[MemoryEntry]`). **Scope = `(app_name, user_id)`**.
- `InMemoryMemoryService.search_memory`: keyword match — counts query words found in each event's words, keeps events with ≥ 1 match, sorts by match count, returns
  the top `_MAX_SEARCH_RESULTS = 10`. Not semantic. `_sqlite_memory_service.py` (private): table `memory_events` + external-content FTS5 `memory_events_fts USING
  fts5(search_text, content='memory_events', content_rowid='id', tokenize='unicode61')` with insert/delete/update triggers, ranked `ORDER BY bm25(memory_events_fts),
  timestamp DESC`; no secure-delete.
- `tools/load_memory_tool.py`: `load_memory(query: str, tool_context)` — **the model calls it** with its own query (explicit). `tools/preload_memory_tool.py`
  `PreloadMemoryTool` (`preload_memory_tool`): "automatically executed for each llm_request, and it won't be called by the model"; the query is **the text parts of
  the current user message** (`tool_context.user_content`); results wrapped in `<PAST_CONVERSATIONS>…</PAST_CONVERSATIONS>` and inserted by
  `llm_request._insert_transient_user_content`, which places it "at the current-turn boundary… before the latest ordinary user batch… prevents the request-scoped
  content from entering a reusable system/history prefix" (`models/llm_request.py`) — i.e. ADK already uses the **memory-at-the-tail** layout of PRIMER §5.
- `VertexAiMemoryBankService`: writes via `agent_engines.memories.ingest_events` by default, or `memories.generate` when `custom_metadata` carries generate-only keys
  (`ttl` — an alias of `revision_ttl` —, `revision_ttl`, `revision_expire_time`, `disable_consolidation`, `disable_memory_revisions`, `metadata`,
  `wait_for_completion`…); `add_memory` uses `memories.create`, or `generate` with `direct_memories_source` when `enable_consolidation` (≤ 5 direct memories per
  call); every call has `scope={'app_name': app_name, 'user_id': user_id}`; `search_memory` → `memories.retrieve(name='reasoningEngines/<id>', scope=...,
  similarity_search_params={'search_query': query})` (no explicit top-k passed). `generation_trigger_config`, e.g. `{"generation_rule": {"idle_duration": "60s"}}`.
  Regions, quotas, prices: (verify) — not in the SDK.

## 9. cognee (`$SP/ref/cognee`, 1.6.1, Apache-2.0) — one table row
- Top-level API (`cognee/__init__.py`): `add`, `cognify`, `memify` (`memify(extraction_tasks=None, enrichment_tasks=None, data=None, dataset=...)`), `search` with
  `SearchType` (`SUMMARIES`, `CHUNKS`, `RAG_COMPLETION`, `HYBRID_COMPLETION`, `GRAPH_COMPLETION`, `TEMPORAL`, `CYPHER`, `CHUNKS_LEXICAL`, …), `forget`, `prune`.
  Deletion: `cognee.forget(*, data_id=None, dataset=None, dataset_id=None, everything=False, memory_only=False)` ("Unified deletion command"; `memory_only=True`
  removes graph + vector, keeps raw files); `cognee.delete(data_id, dataset_id, mode="soft")` is deprecated since 0.3.9 (`datasets.delete_data`; "Don't use 'hard',
  it is dangerous").

## 10. pgvector (`$SP/ref/pgvector`, 0.8.6; `pgvector-python` 0.5.0)
- Current extension **0.8.6** (2026-07-29, `CHANGELOG.md`; 0.8.7 unreleased). Iterative index scans added in **0.8.0** (2024-10-30).
- Types (README): `vector` stores up to 16,000 dims (`4 * dims + 8` bytes); **indexable up to 2,000** (`vector`), 4,000 (`halfvec`, `2 * dims + 8` bytes), 64,000
  (`bit`); `sparsevec` 16,000 non-zeros. A 384-d vector = 4·384 + 8 = 1,544 bytes.
- Operators: `<->` L2, `<#>` **negative** inner product ("Postgres only supports ASC order index scans on operators"), `<=>` cosine distance, `<+>` L1, `<~>`
  Hamming, `<%>` Jaccard. Opclasses e.g. `vector_l2_ops`, `vector_cosine_ops`.
- HNSW: `m` = 16, `ef_construction` = 64 (`WITH (m = 16, ef_construction = 64)`); `hnsw.ef_search` = 40 (range 1..1000); `hnsw.iterative_scan` = `off` (default) |
  `strict_order` | `relaxed_order` (`src/hnsw.c`); `hnsw.max_scan_tuples` = 20,000; `hnsw.scan_mem_multiplier` = 1 (× `work_mem`). README: "filtering is applied
  *after* the index is scanned. If a condition matches 10% of rows, with HNSW and the default `hnsw.ef_search` of 40, only 4 rows will match on average" — per-tenant
  partitions or iterative scans.
- Delete/vacuum: deleted rows are dead tuples; "There may be even less results due to dead tuples"; "Vacuuming can take a while for HNSW indexes. Speed it up by
  reindexing first" (`REINDEX INDEX CONCURRENTLY index_name; VACUUM table_name;`). Postgres has no `secure_delete`: bytes of deleted tuples can persist in
  heap/index/WAL pages until reused or `VACUUM FULL` (unverified in pgvector sources — standard Postgres behaviour).
- Docker tags: `pgvector/pgvector:pg18-trixie` (README's pull example), `0.8.6-pg18-trixie`, `pg18`/`0.8.6-pg18` (bookworm), same for pg17…pg13; e.g.
  **`pgvector/pgvector:0.8.6-pg17`**. Licence: PostgreSQL License.
- psycopg 3: `conn.execute('CREATE EXTENSION IF NOT EXISTS vector')` then `from pgvector.psycopg import register_vector; register_vector(conn)`
  (`register_vector_async` for async) — register after the extension exists.

## 11. vLLM v0.30.0 (`$SP/ref/vllm`, tag v0.30.0)
- `--enable-prompt-tokens-details` (`vllm/entrypoints/launchers/cli_args.py`: `enable_prompt_tokens_details: bool = False`, "enable prompt_tokens_details in usage")
  → `usage.prompt_tokens_details` = `PromptTokenUsageInfo(cached_tokens, created_cache_tokens, multimodal_tokens)` (`vllm/entrypoints/serve/engine/protocol.py`).
  Without the flag there is no `cached_tokens`.
- `cache_salt` request field (chat, completions, responses): `Field(default=None, min_length=1, max_length=1024)` with the description "the prefix cache will be
  salted… The salt should be random, protected from access by 3rd parties, and long enough to be unpredictable (e.g., 43 characters base64-encoded, corresponding to
  256 bit)" — but `validate_cache_salt` (`vllm/entrypoints/generate/base/ protocol.py`) enforces **≤ 128 characters** (`_MAX_CACHE_SALT_LENGTH = 128`) and forbids
  `@`, `/`, `\`, NUL. It enters **only the first block's** extra keys: `cache_salt_keys = [request.cache_salt] if (start_token_idx == 0 and request.cache_salt) else
  []` (`vllm/v1/core/kv_cache_utils.py` line 621) — later blocks differ through the hash chain (vllm-internals §4.3).
- Block size: `CacheConfig.DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`, applied when `block_size` is unset (`vllm/config/cache.py`); some backends/hybrid models choose
  otherwise — read it from `vllm:cache_config_info`.
- Tool calls (`docs/features/tool_calling.md`): `--enable-auto-tool-choice` ("mandatory") + `--tool-call-parser <name>`; **Qwen2.5 → `--tool-call-parser hermes`**
  ("the chat template… has already included support for the Hermes-style tool use"); Qwen3-Coder → `qwen3_xml`. Qwen3 (non-Coder) with `hermes` + `--reasoning-parser
  qwen3`: (unverified in the docs; Qwen3's template emits Hermes-style `<tool_call>` — see `facts/rl-and-thinking-models.md` for the reasoning parser).
- Embeddings: `--runner {auto,generate,pooling,draft}` (`RunnerType`), `--convert {auto,none,embed,classify}`; **no `--task` flag** in `vllm/engine/arg_utils.py`.
  Auto-detection (`vllm/config/model.py _get_default_runner_type`): a Sentence-Transformers `modules.json` pooling config → `pooling`; else registry; else arch
  suffix (`*Model` → pooling/embed, `*ForCausalLM` → generate). `BertModel` → `BertEmbeddingModel` (registry). vLLM tests use
  `sentence-transformers/all-MiniLM-L6-v2` (`architectures=["BertModel"]`, `runner_type="pooling"`), `BAAI/bge-small-en-v1.5` (`architecture="BertModel"`,
  `tests/models/language/pooling_mteb_test/test_baai.py`, `enable_test=False`) and `Qwen/Qwen3-Embedding-0.6B` (`Qwen3ForCausalLM`, `runner_type="pooling"`,
  `seq_pooling_type="LAST"`, `is_prefix_caching_supported=False` for BERT-type encoders). Endpoint `POST /v1/embeddings`. Safe command: `vllm serve
  BAAI/bge-small-en-v1.5 --runner pooling` (explicit, silences the "Resolved `--runner auto`" log).

## 12. SQLite (Python 3.11.15, SQLite 3.45.1 here; compile options `ENABLE_FTS5`, `SECURE_DELETE`)
- Re-measured 2026-09-26 (50 rows, one containing a unique term; counts = occurrences of the term's bytes in `db` + `-wal`):
  | setup | before | after DELETE | after step | after `VACUUM` |
  |---|---|---|---|---|
  | regular FTS5 table | 3 | **2** | — | **2** |
  | FTS5 `secure-delete` = 1 (set before inserts) | 3 | **0** | — | 0 |
  | regular + `INSERT INTO fts(fts) VALUES('optimize')` | 3 | 2 | optimize → 0 | 0 |
  | external content (`content='m'`) + `'rebuild'` | 2 | 2 | rebuild → 0 | 0 |
  | WAL, regular | 3 | **5** (old pages in `-wal`) | `wal_checkpoint(TRUNCATE)` → 2 | 2 |
  | WAL, `secure-delete` | 3 | 3 (all in `-wal`) | checkpoint → **0** | 0 |
  | WAL, optimize | 3 | 5 | ckpt 2 → optimize 2 → ckpt **0** | 0 |
  Lesson: `DELETE` + `VACUUM` does not remove the term from a regular FTS5 index; the WAL holds old pages until a checkpoint.
- Sources: FTS5 option parsed in `ext/fts5/fts5_config.c` (`sqlite3_stricmp(zKey, "secure-delete")`, `pConfig->bSecureDelete`), set with `INSERT INTO fts(fts, rank)
  VALUES('secure-delete', 1)`; test `ext/fts5/test/fts5secure.test` dated "2023 Feb 17". Release that added it: **3.42.0 (2023-05-16)** (unverified — sqlite.org
  blocked; consistent with the test date). Page-level `PRAGMA secure_delete`: default **off** unless built with `SQLITE_SECURE_DELETE` (`src/btree.c`: `#if
  defined(SQLITE_SECURE_DELETE) pBt->btsFlags |= BTS_SECURE_DELETE; #elif defined(SQLITE_FAST_SECURE_DELETE)` …); values `ON/OFF/FAST` (`src/pragma.c`).
  `SQLITE_CHECKPOINT_TRUNCATE` "Like RESTART but also truncate WAL".
- `bm25()` (`ext/fts5/fts5_aux.c`): `k1 = 1.2`, `b = 0.75` (ragkit's `BM25` uses k1 = 1.5); `idf = log((nRow − nHit + 0.5)/(nHit + 0.5))`, **floored at 1e-6** when ≤
  0; returns `-1.0 * score` → **negative, lower is better**, `ORDER BY bm25(fts)` ascending. On tiny corpora where a term is in ≥ half the rows the IDF floor makes
  scores ≈ −1e-6 (measured: −1.419e-06 vs −1e-06).
- Colab: its Python's SQLite version is (unverified); if < 3.42 the `secure-delete` insert raises `sqlite3.OperationalError` → fall back to `optimize`/`rebuild` +
  checkpoint + `VACUUM`. Feature-detect with a throwaway in-memory FTS5 table, never by version string alone.

## 13. Cloud Run jobs on Cloud Scheduler (for `deploy/gcp/README.md`; no Terraform, no prices)
- `gcloud run jobs create <job> --set-env-vars K=V,... --max-retries 10 --tasks 5` and `gcloud run jobs execute <job> --wait`
  (`$SP/ref/cloud-run-samples/jobs-shell/README.md`). `gcloud run jobs deploy` (source/image deploy) and `--task-timeout`: (unverified, not in the samples read).
- Scheduler → job: HTTP target **`POST https://run.googleapis.com/v2/projects/<p>/locations/<r>/jobs/<job>:run`** with an **OAuth** token (`oauth_token {
  service_account_email = ... }`) — terraform-docs-samples `run/jobs_execute_jobs_on_schedule/main.tf` (fetched via raw.githubusercontent.com); gcloud form
  (GoogleCloudPlatform samples, e.g. `jobs-demos/invoice-processing-pipeline/README.md`) uses the v1 URI
  `https://<r>-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/<p>/jobs/<job>:run --http-method POST --oauth-service-account-email <sa>`. OAuth (not OIDC)
  because the target is a `*.googleapis.com` API; OIDC is for your own service (lra-gcp's reaper → `.../internal/reap`). Roles in that sample: `roles/run.developer`
  on the project for the invoker SA plus `roles/iam.serviceAccountTokenCreator`; the least-privilege `roles/run.invoker` on the job (which carries `run.jobs.run`) is
  (unverified).

## 14. Managed memory services (PRIMER §9 verify table; dated 2026-09-26, no prices)
| service | what the source shows | source |
|---|---|---|
| Vertex AI Agent Engine Memory Bank | scope `{app_name, user_id}`; `ingest_events` / `generate` / `create` / `retrieve(similarity_search_params)`; revisions, `revision_ttl`, consolidation toggle | ADK `vertex_ai_memory_bank_service.py` |
| Amazon Bedrock AgentCore Memory | `StrategyType`: `semanticMemoryStrategy`, `summaryMemoryStrategy`, `userPreferenceMemoryStrategy`, `episodicMemoryStrategy`, `customMemoryStrategy`; short-term events with `event_expiry_days: int = 90` (`eventExpiryDuration`) | `src/bedrock_agentcore/memory/constants.py`, `client.py` |
| Anthropic memory tool | `{"type": "memory_20250818", "name": "memory"}` (in `types/memory_tool_20250818_param.py`, not only beta); commands `view`, `create`, `str_replace`, `insert`, `delete`, `rename`; the **client** stores files; the SDK helper confines paths to `/memories` (rejects symlink escape) | `anthropic-sdk-python` `types/beta/beta_memory_tool_20250818_*_command.py`, `lib/tools/_beta_builtin_memory_tool.py` |
| mem0 Platform · Zep Cloud · Letta Cloud | mem0: ADD-only v3 API `/v3/memories/add/`, graph memory Platform-only; Zep CE deprecated to `legacy/`; Letta OSS moved to letta-code | mem0 `docs/`; `zep/README.md`; `letta/README.md` |

## 15. OpenTelemetry GenAI conventions
- Memory **does now have conventions**, in the new repo `open-telemetry/semantic-conventions-genai` (unreleased — `changelog.d/140.enhancement.md` "Add GenAI memory
  operation span and attributes…"; `stability: development`): span type `gen_ai.memory.client`, name `{gen_ai.operation.name}`; `gen_ai.operation.name` ∈
  `create_memory_store`, `search_memory`, `create_memory`, `update_memory`, `upsert_memory`, `delete_memory`, `delete_memory_store`; attributes
  `gen_ai.memory.store.id`, `gen_ai.memory.record.id` (absent on `delete_memory` may mean "delete all"), `gen_ai.memory.record.count`, `gen_ai.memory.query.text` and
  `gen_ai.memory.records` (both opt-in, "may contain sensitive information") (`model/gen-ai/registry.yaml` 661–816, `spans.yaml` 404–452). Also an operation
  `retrieval` and `gen_ai.data_source.id`.
- The 07.2 names (`agentlab/observability/tracing.py`) exist unchanged: `gen_ai.usage.input_tokens`, `gen_ai.usage.cache_read.input_tokens` (also `cache_write`), `gen_ai.tool.name`. Recommendation: emit
  `gen_ai.operation.name = search_memory|create_memory|delete_memory` + `gen_ai.memory.*`, marked "development (verify)", and keep the audit's
  `memory.write|read|forget` event types for the `AuditEvent`.

## 16. Memory poisoning sources
- **AgentPoison** — Chen, Xiang, Xiao, Song, Li, "AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases", NeurIPS 2024 (`AgentPoison/README.md`
  bibtex; arXiv 2407.12784; MIT). Finding: an optimised backdoor trigger in the query retrieves a few poisoned memory/knowledge entries and steers the agent (metrics
  ASR-r, ASR-a, ASR-t, ACC); headline rates (≈ 80% ASR at < 0.1% poison rate) are (unverified — paper only).
- **MINJA** — "Memory Injection Attacks on LLM Agents via Query-Only Interaction", arXiv 2503.03704, 2025 (repo `dsh3n77/MINJA` description;
  `ai-fwd/minja-toy-harness`): an attacker with only normal user queries induces the agent to write malicious records that later users' queries retrieve. Authors,
  venue, rates: (unverified).
- **PoisonedRAG** — Zou et al., 2024, already cited in `embeddings-lab/docs/primer.md` §15: poisoned passages written to the corpus are retrieved for target queries.
  Venue (USENIX Security 2025): (unverified).
- OWASP ASI06 "Memory & Context Poisoning" row: identity primer §2 (controls "Provenance tags on stored content; per-user/per-tenant memory isolation; write-gating
  to memory | Screening before persistence").

## 17. Model and tool ids to use
| tier | id | size / shape | licence | vLLM v0.30.0 |
|---|---|---|---|---|
| T0 | scripted template extractor/answerer (`memcore`) | — | repo MIT | — |
| T0 | `ragkit`-style crc32 hashing embedder, dim 1024 | 4 KiB per float32 vector | repo | — |
| T0 | SQLite 3.45.1 (stdlib) + numpy; FTS5 | — | public domain | — |
| T0 + Docker | `pgvector/pgvector:0.8.6-pg17` (or `:pg18`) | — | PostgreSQL | — |
| T1 chat | `Qwen/Qwen2.5-1.5B-Instruct` — 1.54 B params (`minengine` LLMS: 28 layers, 12 heads, 2 KV heads, head_dim 128, vocab 151,936×1,536, tied); BF16 weights 1.54e9·2 ≈ **3.1 GB**; KV/token = 2·28·2·128·2 B = **28,672 B (28 KiB)** | fits a T4/L4 | Apache-2.0 (unverified — HF blocked) | `--enable-auto-tool-choice --tool-call-parser hermes` (docs) |
| T1 chat | `Qwen/Qwen3-1.7B` — 1,720,574,976 params, 3.44 GB BF16, 112 KiB KV/token (from `facts/rl-and-thinking-models.md`) | thinking on by default | Apache-2.0 (verify) | `hermes` + `--reasoning-parser qwen3` (verify) |
| T1 small | `Qwen/Qwen2.5-0.5B-Instruct` (the serving lab's `serve.sh` default) | — | (verify) | `hermes` |
| T1 embed | `BAAI/bge-small-en-v1.5` — `BertModel`, 384-d, ≈ 33 M params (≈ 133 MB fp32), 512 tokens | — | MIT (unverified) | pooling (tests list it) |
| T1 embed | `sentence-transformers/all-MiniLM-L6-v2` — `BertModel`, 384-d, ≈ 22.7 M params, ~90 MB (rag-from-scratch README "(verify)") | ragkit's default | Apache-2.0 (unverified) | pooling (vLLM test) |
| T1 embed alt | `Qwen/Qwen3-Embedding-0.6B` — `Qwen3ForCausalLM`, LAST pooling, 1,024-d (unverified) | — | Apache-2.0 (unverified) | pooling via `modules.json` |
Nothing is loaded at T0; T1 ids are `(verify)` in notebooks. Serve one model per vLLM process (chat and embedder are two processes).

## 18. Pitfalls
1. **Generative-agents recency**: the code sorts by `last_accessed` ascending and gives rank 1 the largest `0.99**1`, so the *oldest* node gets the highest recency;
   it is rank-based, uses weights `[0.5, 3, 2]`, and normalises over the whole stream. The paper form (0.995 per hour, weights 1) is unverified here. Test both; name
   them `score(..., form="paper")` / `form="code"`.
2. **Reflection input**: the code reflects over the `importance_ele_n` events since the last reflection, not "the 100 most recent".
3. **mem0 is ADD-only since 2.0.0 (2026-04-14)**; the four-op prompt's no-op event is `"NONE"`; describe ADD/UPDATE/DELETE/NONE as mem0 v1. `memcore.write` can keep
   ADD/UPDATE/NOOP as its own design — cite mem0 accurately.
4. **mem0 `delete` keeps the text** in `history.old_memory` and `messages` — a "deleted" memory is not deleted. Same trap as FTS5/WAL.
5. **LoCoMo numbering**: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial — inferred from the eval code and evidence counts; the code comment
   lists names in another order. **CC BY-NC 4.0**: bundle nothing.
6. LongMemEval abstention is an `_abs` suffix on `question_id`, not a `question_type`; judge = LLM yes/no per type.
7. **FTS5 keeps deleted terms** (regular and external-content) until `optimize`/`rebuild` or `secure-delete`; **WAL** keeps old pages until `PRAGMA
   wal_checkpoint(TRUNCATE)`; `VACUUM` alone does not clean the FTS5 index. Test deletion by grepping the file *and* `-wal`.
8. **`bm25()` is negative**, lower is better; k1 = 1.2 (not ragkit's 1.5); IDF floors at 1e-6 on tiny corpora. RRF: ragkit k = 60, Graphiti effectively k = 1 — say
   which.
9. **`cache_salt`**: secret, random, per tenant, **≤ 128 chars** (the schema's 1024 is not the real limit), no `@ / \ NUL`; only enters the first block. A shared
   salt across tenants = cross-tenant prefix hits.
10. `cached_tokens` appears only with `--enable-prompt-tokens-details`; no `--task` flag in v0.30.0 — use `--runner pooling`.
11. **`PreloadMemoryTool` runs on every LLM request** (including tool-call continuations), querying with the current user message's text; `load_memory` runs only
    when the model asks. ADK inserts preloaded memory at the turn boundary, not in the prefix.
12. **Graphiti has no `valid_to`** — `valid_at`/`invalid_at` (valid time) and `created_at`/`expired_at` (system time); contradictions close the old edge, never
    delete it.
13. Letta: Python server archived; block `limit` counts **characters** (default 100,000); compaction at 0.9 of the window in code.
14. **Cloud Scheduler → Cloud Run job uses OAuth** (`--oauth-service-account-email`), because the target is `run.googleapis.com`; OIDC is for your own `*.run.app`
    service (lra-gcp's reaper).
15. pgvector filters apply after the HNSW scan (ef_search 40 → ~4 rows at 10% selectivity): partition by tenant/user or enable `hnsw.iterative_scan`; `vector`
    indexes ≤ 2,000 dims; `<#>` is negative inner product.
16. SPEC §6b's opening list says torch is installed — **not in this build** (FACTS "structure-plan builds"); no weights, no network in tests;
    `ragkit.HashingEmbedder.encode(str)` returns 1-D.
17. OTel memory conventions exist but are unreleased/`development` — label them "(verify)"; `gen_ai.memory.query.text`/`records` are opt-in (sensitive).
18. Colab's SQLite may predate 3.42 → feature-detect FTS5 `secure-delete`.

## 19. Unverified (collected)
Generative-agents paper form (0.995/hour, weights 1, "100 most recent"); LongMemEval per-type counts and HF dataset licence; SQLite 3.42.0 as the FTS5
`secure-delete` release; Colab's SQLite version; Postgres residue after `DELETE`/`VACUUM`; `gcloud run jobs deploy` and `--task-timeout`; `roles/run.invoker`
sufficing for `jobs.run`; Qwen3 tool-call parser; licences and sizes of Qwen2.5-1.5B-Instruct, bge-small-en-v1.5, all-MiniLM-L6-v2, Qwen3-Embedding-0.6B (HF
blocked); MemGPT's 70%/100% thresholds; AgentPoison headline rates; MINJA authors/venue; PoisonedRAG venue; managed-service regions/quotas/prices.
