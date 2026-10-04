# %% [markdown]
# # 05 · Evaluate, forget and audit: planted facts, the recall knee, a deletion checked on disk
#
# **Tier:** T0. The planted-facts harness runs the agent against a scripted model and the hashing embedder.
# The notebook computes every number, and each number is deterministic and reproducible. No number is a model
# measurement. To make sure of the deletion, the notebook searches the database files for the bytes. **T1:** when `MEMLAB_EMBED_URL` points
# at a real embedder (vLLM `--runner pooling`, `deploy/any-gpu/`), the last section uses it to measure the
# paraphrase subset.
#
# ## The one-minute version
#
# **Measure** memory as the long-term memory benchmarks do (PRIMER §4 "Measuring memory: planted facts
# across sessions"). Do these steps:
#
# 1. Plant facts across sessions of small talk.
# 2. Change one fact halfway.
# 3. Ask about the facts later, in new sessions. The questions have the question shapes of LongMemEval
#    (single-session, preference, multi-session, temporal, knowledge update, abstention). They also have the
#    adversarial shape of LoCoMo.
# 4. Grade every answer `correct`, `stale`, `abstained`, `hallucinated` or `wrong`. Give the Wilson intervals
#    (07.2 notebook 08 §4) and the tokens and calls that each answer cost.
#
# The notebook downloads nothing (LoCoMo is CC BY-NC 4.0). The generator uses a seed. After you grade the
# answers, select the per-turn memory budget at the **knee** of recall against tokens (PRIMER §5).
#
# **Forget** a fact in all the locations where it went (PRIMER §7):
#
# * the record, its vector and its FTS row
# * the facts *derived* from it
# * the idempotency table
# * the prefix cache of the engine. vLLM can reset this cache only as a whole. To forget the data of one
#   tenant, you rotate the cache salt of that tenant.
# * the eval set
#
# Then search the files to prove the deletion. The audit log holds only hashes. Thus it is not one more copy.
# Backups expire on their own schedule.
#
# **Audit** every read, write and forget with both identities (PRIMER §8, identity primer §9). The primer is
# [`../../PRIMER.md`](../../PRIMER.md).

# %%
import json, os, tempfile, urllib.request
from memlab import env, harness as H
from memlab.audit import AuditLog, read_json_lines
from memlab.consolidate import ConsolidationJob
from memlab.deletion import DeletionReport, residue
from memlab.embedders import get_embedder, OpenAIEmbeddings
from memlab.extract import ts_of
from memlab.fakeserver import FakeLLMServer
from memlab.llm import ChatClient
from memlab.memory import LocalMemory
from memlab.records import MemoryRecord
from memlab.report import harness_markdown
from memlab.service import MemoryClient, MemoryService, TokenVerifier
from memlab.store import SQLiteMemoryStore

print(env.banner())
ds = H.generate(seed=7)
print(len(ds.questions), "questions:", ds.counts())
print("u1's third session:", ds.haystacks["u1"][2].turns)
print("a few questions:", [(q.id, q.category, q.text, q.answer) for q in ds.questions[:3] + ds.questions[11:14]])

# %% [markdown]
# ## Worked example: five ways to remember, one benchmark
#
# The comparison has five modes:
#
# * `none`: no memory. This is the floor.
# * `full_history`: every past session is in the prompt. This is the ceiling on recall and on tokens.
# * the three agent modes of notebook 02: each has a 128-token memory budget.

# %%
results = H.compare_modes(ds, budget_tokens=128)
print(H.summary(results))
print()
print(results["implicit"].table())

# %% [markdown]
# Read the result as in a design review. `full_history` answers every question. But it carries every past
# session on every turn, and it grows without a limit.
#
# `implicit` retrieval misses the **preference**, because a recipe question shares no words with "Diet: the
# user is vegan". It also misses the **paraphrases**, because the hashing embedder is lexical. `tools` also
# misses the preference, because the model never thought to ask. The pinned profile carries the preference
# with no request. Its cost is its tokens and a second model call.
#
# Abstention and the adversarial questions are correct in all modes, because the scripted answerer never
# invents an answer. A real model can invent an answer. That is why these questions are in the benchmark.
#
# This harness is not the harness of memory-core (PRIMER §4, §6). This harness has other users, 84 questions, a
# 128-token budget, an answerer that understands paraphrases, and a scripted model that recalls on first-person
# questions. Thus its ranking of the modes is different from the table in PRIMER §6. Neither ranking is
# evidence about a real model, because in both harnesses a script makes the choices of the model. The method
# is what transfers: the same questions, graded the same way, for every design that you consider.
#
# ## Exercise 5.1 — grade an answer
#
# Write `my_grade(q, answer)`. An answer *abstains* when it contains "don't know", "not mentioned" or "no
# information" (case-insensitive). Apply these rules in this order:
#
# 1. If the question has no answer (`q.answer is None`), the correct move is to abstain. Return `"correct"`
#    for an abstention and `"hallucinated"` for any other answer.
# 2. If the expected answer appears in the answer, return `"correct"`.
# 3. If the pre-update value (`q.stale`) appears in it, return `"stale"`.
# 4. If the answer abstained, return `"abstained"`.
# 5. Else, return `"wrong"`.

# %% exercise
def my_grade(q, answer):
    ### BEGIN SOLUTION
    a = answer.lower()
    abstained = any(s in a for s in ("don't know", "not mentioned", "no information"))
    if q.answer is None:
        return "correct" if abstained else "hallucinated"
    if q.answer.lower() in a:
        return "correct"
    if q.stale and q.stale.lower() in a:
        return "stale"
    return "abstained" if abstained else "wrong"
    ### END SOLUTION

# %% check
upd = next(q for q in ds.questions if q.category == "knowledge-update")
unans = next(q for q in ds.questions if q.id.endswith("_abs"))
assert my_grade(upd, upd.answer + ".") == "correct" and my_grade(upd, f"{upd.stale}.") == "stale"
assert my_grade(upd, "I don't know.") == "abstained" and my_grade(upd, "Mars.") == "wrong"
assert my_grade(unans, "I don't know.") == "correct" and my_grade(unans, "Peanuts.") == "hallucinated"
qs = {q.id: q for q in ds.questions}
for r in results.values():
    assert all(my_grade(qs[row.qid], row.answer) == row.verdict for row in r.rows)
print("✅ my_grade agrees with the harness on", sum(len(r.rows) for r in results.values()), "graded answers")

# %% [markdown]
# ## Worked example: recall versus the per-turn budget
#
# The agent writes memory as raw episodes. This is the harder case: the records are longer, and there are more
# of them. `k` is large. Thus only the token budget decides what fits. Recall means that the evidence for an
# answerable question was in the prompt.

# %%
curve = H.recall_vs_budget(ds, write="episodes")
print(" budget  recall  accuracy  memory tokens/turn")
for p in curve:
    print(f" {p['budget']:6d}  {p['recall']:6.1%}  {p['accuracy']:8.1%}  {p['memory_tokens']:8.0f}")

# %% [markdown]
# ## Exercise 5.2 — find the knee
#
# Write `my_knee(points, key="recall", frac=0.95)`. It returns the point with the smallest `budget` whose
# `key` reaches `frac` of the best value on the curve. After this point, more tokens buy almost nothing. Also,
# you pay for every token on every turn, and the tokens can cost prefix-cache hits (notebook 03).
#
# The knee of this lab is *relative*: 95% of the best. The `knee(tol=0.02)` of memory-core is *absolute*:
# within 2 points of the best. The two knees select the same budget when recall saturates near 100%. They can
# be different on a flat curve. The Glossary of the PRIMER names both. Say which knee you used.

# %% exercise
def my_knee(points, key="recall", frac=0.95):
    ### BEGIN SOLUTION
    best = max(p[key] for p in points)
    return next(p for p in sorted(points, key=lambda p: p["budget"]) if p[key] >= frac * best)
    ### END SOLUTION

# %% check
toy = [{"budget": 0, "recall": 0.1}, {"budget": 64, "recall": 0.8}, {"budget": 128, "recall": 0.9},
       {"budget": 256, "recall": 0.92}]
assert my_knee(toy)["budget"] == 128 and my_knee(toy, frac=0.85)["budget"] == 64
k = my_knee(curve)
assert k == H.knee(curve) and k["budget"] < max(p["budget"] for p in curve)
print(f"✅ knee at {k['budget']} tokens: recall {k['recall']:.1%} of a best {max(p['recall'] for p in curve):.1%}, "
      f"{k['memory_tokens']:.0f} memory tokens per turn instead of {curve[-1]['memory_tokens']:.0f}")

# %% [markdown]
# ## Worked example: a user asks to be forgotten
#
# The next cell makes every copy that PRIMER §7 lists. User `acme/u1` told the assistant their home address.
# The address went:
#
# 1. into an **episode**, through the memory service. The episode has the deletion key `acme/u1/address`.
#    The write sends an Idempotency-Key, and the key is a row in the **idempotency** table of the service.
# 2. into a **fact** that consolidation derived from the episode, and into a weekly **summary** whose
#    provenance cites the episode. The summary has a different deletion key. Thus only provenance links it.
# 3. into a prompt that an inference engine served. The **prefix cache** of the engine holds those blocks.
# 4. into the **eval set**, as a golden case built from a transcript (07.2 notebook 08 §9).
# 5. into the **audit log**, as hashes. This is by design.

# %%
WORK = tempfile.mkdtemp(prefix="memlab-nb05-")
import atexit, shutil
atexit.register(shutil.rmtree, WORK, True)   # removed when the kernel exits, even if a cell stops early
DB = os.path.join(WORK, "memory.db")
store = SQLiteMemoryStore(DB, fts_secure_delete=False)       # the harder case: FTS5 keeps terms until optimize
verifier = TokenVerifier(os.urandom(32))
audit = AuditLog(os.path.join(WORK, "audit.jsonl"))
svc = MemoryService(store, verifier, audit=audit)
URL = svc.start()
me = MemoryClient(URL, verifier.mint("acme", "u1"))
ADDRESS = "12 Rua das Flores"
status, ep = me.write(f"On 2026-09-10 the user said: my address is {ADDRESS}.", "turn-s3-2", kind="episodic",
                      deletion_key="acme/u1/address", session="s3")
for i in range(20):
    me.write(f"On 2026-09-1{i % 10} the user said: note {i} about lunch and the weather.", kind="episodic")
store.add(MemoryRecord("acme", "u1", f"Summary of week 37: the user shared their home address ({ADDRESS}) and "
                       "planned a trip.", source="consolidation", trust="user", provenance=[ep["id"]],
                       deletion_key="acme/u1/summary-2026-W37"))
job = ConsolidationJob(store, worker="weekly")
print(job.run("acme", "u1", ts_of("2026-09-01"), ts_of("2026-12-01")).line())
eval_set = [{"id": "g1", "input": "What is my address?", "expected": ADDRESS, "user": "acme/u1"},
            {"id": "g2", "input": "What is my employer?", "expected": "Globex", "user": "acme/u1"}]
fake = FakeLLMServer(enable_prompt_tokens_details=True)
LLM = fake.start()
_, found = me.search("what is my address", k=3)
ChatClient(LLM).generate([{"role": "system", "content": "Memory: " + " ".join(i["text"] for i in found["items"])},
                          {"role": "user", "content": "Where should the parcel go?"}])
print("on disk:", residue(DB, [ADDRESS, "Flores"]), "| cached blocks in the engine:", len(fake.cache.blocks))

# %% [markdown]
# ## Exercise 5.3 — forget it everywhere, and prove it
#
# Write `forget_everywhere(client, llm_url, eval_set, needle)`. It returns a `DeletionReport`. The `counts` of
# the report tell how many copies each surface held:
#
# 1. Send `DELETE /v1/memories?subject=address` through the service (`client.forget("address")`). Its report
#    has the records, vectors, FTS rows, derived records (by provenance) and idempotency rows that it removed.
#    Start from `report["counts"]`.
# 2. Clear the prefix cache of the engine. vLLM v0.30.0 cannot evict the blocks of one tenant or of one user. Its
#    only tool is the dev-mode `POST {llm_url}/reset_prefix_cache`. The server must run with
#    `VLLM_SERVER_DEV_MODE=1` (verify). The endpoint clears the cache of **every** tenant and answers only
#    `{"success": bool}`. The value is false while active requests hold blocks. Thus examine it.
#
#    Call the endpoint. Make sure that `success` is true. Then record `prompt_cache` as `cached_blocks`, the count that
#    the caller passes in. The fake server knows its own blocks (SIMULATED). A real vLLM does not report them.
#
#    In production, it is better to rotate the `cache_salt` of the tenant (notebook 03, exercise 3.5, with an
#    epoch in the HMAC input). The rotation makes the blocks of that tenant unreachable at once, and LRU evicts
#    them over time. The
#    caches of all other tenants stay warm. The reset is the runbook step of the operator for the residue.
# 3. Remove every eval-set case that contains the needle. Change the list in place. Record the count as
#    `eval_sets`.
# 4. Count the audit lines that contain the needle, as `audit_log`. The expected count is 0, because the log
#    holds hashes.
# 5. Put `"backups"` in `not_reachable`. Backups expire on their retention schedule, and the deletion policy
#    must state that schedule.

# %% exercise
def forget_everywhere(client, llm_url, eval_set, needle, audit_path, cached_blocks):
    ### BEGIN SOLUTION
    status, body = client.forget("address")
    r = body["report"]
    rep = DeletionReport(r["deletion_key"], r["tenant"], counts=dict(r["counts"]), steps=list(r["steps"]),
                         deleted_ids=list(r["deleted_ids"]))
    req = urllib.request.Request(llm_url.rstrip("/") + "/reset_prefix_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        if not json.loads(resp.read()).get("success"):
            raise RuntimeError("the engine refused the reset (blocks still held): retry")
    rep.counts["prompt_cache"] = cached_blocks
    hits = [c for c in eval_set if needle in json.dumps(c)]
    for c in hits:
        eval_set.remove(c)
    rep.counts["eval_sets"] = len(hits)
    rep.counts["audit_log"] = sum(needle in line for line in open(audit_path))
    rep.not_reachable.append("backups: expire on their retention schedule")
    return rep
    ### END SOLUTION

# %% check
held = len(fake.cache.blocks)                    # the fake's own state (SIMULATED): vLLM does not report this
rep = forget_everywhere(me, LLM, eval_set, ADDRESS, audit.path, held)
rep.residue = residue(DB, [ADDRESS, "Flores"])
print(rep.table())
assert rep.counts["records"] >= 2 and rep.counts["derived"] >= 1 and rep.counts["idempotency"] == 1, rep.counts
assert rep.counts["prompt_cache"] == held > 0 and len(fake.cache.blocks) == 0
assert rep.counts["eval_sets"] == 1 and [c["id"] for c in eval_set] == ["g2"] and rep.counts["audit_log"] == 0
assert rep.clean, rep.residue
assert all(ADDRESS not in i["text"] for i in me.search("what is my home address", k=5)[1]["items"])
print("✅ every copy counted and removed; the address occurs 0 times in the database file and its WAL")

# %% [markdown]
# ## Exercise 5.4 — read the audit trail
#
# Write `audit_counts(events)`. From the audit JSON lines, it returns a dict `{(tenant, user, event_type):
# count}` and the number of events whose `decision` is `deny`. A forget request appears as exactly one
# `memory.forget`.

# %% exercise
def audit_counts(events):
    ### BEGIN SOLUTION
    counts, denied = {}, 0
    for e in events:
        key = (e["tenant"], e["user"], e["event_type"])
        counts[key] = counts.get(key, 0) + 1
        denied += e.get("decision") == "deny"
    return counts, denied
    ### END SOLUTION

# %% check
events = read_json_lines(audit.path)
counts, denied = audit_counts(events)
assert counts[("acme", "u1", "memory.forget")] == 1 and counts[("acme", "u1", "memory.write")] >= 21
assert denied == sum(e.decision == "deny" for e in audit.events)
assert all(set(e) >= {"agent", "authority", "user", "tenant", "args_hash"} for e in events)
print("✅", {f"{k[2]}": v for k, v in counts.items()}, "| denied:", denied)
svc.stop(); fake.stop()
store.close()
import shutil
shutil.rmtree(WORK, ignore_errors=True)    # the database, its WAL and the audit log of this notebook: gone

# %% [markdown]
# ## T1: a real embedder on the paraphrase subset
#
# The paraphrase questions ask about the same facts in other words ("Which firm pays my salary?"). The hashing
# embedder misses them by design. You expect that a semantic embedder does not miss them. When you set
# `MEMLAB_EMBED_URL`, this cell runs the harness again with that embedder. This is a measurement on your
# embedder, not a simulation.

# %%
emb = get_embedder()
base = H.run(ds, "implicit", budget_tokens=128)
print("hashing embedder (computed): standard %.0f%%, paraphrase %.0f%%" % tuple(100 * x for x in H.paraphrase_gap(base)))
if isinstance(emb, OpenAIEmbeddings):
    label = "SIMULATED" if env.is_simulated(emb.url) else "MEASURED"
    real = H.run(ds, "implicit", budget_tokens=128, embedder=emb, label=emb.label)
    print(f"[{label}] {emb.label}: standard %.0f%%, paraphrase %.0f%%" % tuple(100 * x for x in H.paraphrase_gap(real)))
else:
    print("No MEMLAB_EMBED_URL: to measure a real embedder (T1), from the repo root on a GPU box:")
    print("   MODEL=BAAI/bge-small-en-v1.5 PORT=8001 MAX_MODEL_LEN=512 GPU_MEM_UTIL=0.2 EXTRA_ARGS='--runner pooling' "
          "04-inference-engine/serving-engine/vllm-serving-lab/deploy/any-gpu/serve.sh")
    print("   export MEMLAB_EMBED_URL=http://127.0.0.1:8001")
print()
print(harness_markdown({"implicit (hashing)": base}, "computed: scripted model, hashing embedder, seed 7"))

# %% [markdown]
# ## In a design review
#
# **Two minutes.** "We evaluate memory with a planted-facts harness that uses a seed. It uses the question
# shapes of LongMemEval and LoCoMo. It has 84 questions across seven categories. We grade each answer correct,
# stale, abstained or hallucinated. We report Wilson intervals and the tokens and model calls per answer.
#
# "The harness tells us where each design fails. Retrieval before every turn misses preferences. Tools miss
# what the model does not ask for. A pinned profile costs tokens and a call.
#
# "We set the per-turn memory budget at the knee of recall against tokens. Deletion is a checklist with
# counts. It has a deletion key on every record and provenance to reach derived facts. It covers the
# idempotency rows of the service and the prefix cache of the engine. For that cache, we rotate the salt of
# the tenant. A full reset is the step of the operator, because vLLM cannot evict the blocks of one tenant.
#
# "The checklist also covers the eval set and a purge of FTS and WAL. Then we search the files for the bytes.
# Audit logs carry hashes, not memory. Backups expire, and the policy says after how long."
#
# **Drill 1.** *A user asked us to forget their address, and a week later the assistant quoted it. Where was
# it?* It was in a copy that the delete did not reach. The copy can be a consolidated summary derived from the
# address (no provenance link), or the FTS index or WAL (no purge). It can also be the prefix cache of the engine, a
# log or an eval set.
#
# Deletion keys, provenance and a byte-level check are the solution.
#
# **Drill 2.** *Abstention is 100% here. Can we drop those questions?* No, because the scripted answerer
# never invents an answer. With a real model, the abstention and false-premise questions are where hallucination
# becomes visible. These questions stay in the golden set.
#
# **Drill 3.** *Why does the paraphrase subset exist if T0 always fails it?* It measures the contribution of
# the embedder: the same facts, the same answerer, different words. The gap between the two columns is the
# argument for a semantic embedder, and the test of it.
