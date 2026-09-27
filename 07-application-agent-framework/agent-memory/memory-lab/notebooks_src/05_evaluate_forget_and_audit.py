# %% [markdown]
# # 05 · Evaluate, forget and audit: planted facts, the recall knee, a deletion checked on disk
#
# **Tier:** T0 — the planted-facts harness runs the agent against a scripted model and the hashing embedder;
# every number is computed, deterministic and reproducible (not a model measurement). The deletion is checked
# by searching the database files for the bytes. **T1:** with `MEMLAB_EMBED_URL` pointing at a real embedder
# (vLLM `--runner pooling`, `deploy/any-gpu/`), the last section measures the paraphrase subset with it.
#
# ## The one-minute version
#
# **Measure** memory the way the long-term memory benchmarks do (PRIMER §4 "Measuring memory: planted facts
# across sessions"): plant facts across sessions of chatter, change one halfway, ask later in fresh sessions —
# LongMemEval's question shapes (single-session, preference, multi-session, temporal, knowledge update,
# abstention) and LoCoMo's adversarial one — and grade every answer `correct`, `stale`, `abstained`,
# `hallucinated` or `wrong`, with Wilson intervals (07.2 notebook 08 §4) and the tokens and calls each answer
# cost. Nothing is downloaded (LoCoMo is CC BY-NC 4.0); the generator is seeded. Then pick the per-turn
# memory budget at the **knee** of recall versus tokens (PRIMER §5).
#
# **Forget** a fact everywhere it went (PRIMER §7): the record, its vector and FTS row, the facts *derived* from
# it, the idempotency table, the engine's prefix cache (which vLLM can only reset whole — per tenant you rotate the
# cache salt), the eval set — and prove it by searching the files; the audit log holds hashes only, so it is not
# one more copy; backups age out on their own schedule.
#
# **Audit** every read, write and forget with both identities (PRIMER §8; identity primer §9). Primer:
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
# `none` (no memory: the floor), `full_history` (every past session in the prompt: the ceiling on recall and
# on tokens), and the three agent modes of notebook 02 with a 128-token memory budget.

# %%
results = H.compare_modes(ds, budget_tokens=128)
print(H.summary(results))
print()
print(results["implicit"].table())

# %% [markdown]
# Read it as a design review would. `full_history` answers everything but carries every past session on every
# turn (and grows without bound). `implicit` retrieval misses the **preference** (a recipe question shares no
# words with "Diet: the user is vegan") and the **paraphrases** (the hashing embedder is lexical); `tools`
# misses the preference too — the model never thought to ask. The pinned profile carries the preference
# without being asked, for its tokens and a second model call. Abstention and the adversarial questions
# are right everywhere because the scripted answerer never invents — a real model is not so polite, which is
# why they are in the benchmark.
#
# This is not memory-core's harness (PRIMER §4, §6): other users, 84 questions, a 128-token budget, an answerer that
# understands paraphrases and a scripted model that recalls on first-person questions. So its ranking of the modes
# differs from PRIMER §6's table, and neither ranking is evidence about a real model — both script the model's
# choices. What transfers is the method: the same questions, graded the same way, for every design you consider.
#
# ## Exercise 5.1 — grade an answer
#
# Write `my_grade(q, answer)`: an answer *abstains* when it contains "don't know", "not mentioned" or "no
# information" (case-insensitive). If the question has no answer (`q.answer is None`) the right move is to
# abstain (`"correct"`), anything else is `"hallucinated"`. Otherwise: `"correct"` if the expected answer
# appears in it, `"stale"` if the pre-update value (`q.stale`) does, `"abstained"` if it abstained, else
# `"wrong"`.

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
# Memory written as raw episodes (the harder case: longer records, more of them), `k` large so only the token
# budget decides what fits. Recall = the evidence for an answerable question was in the prompt.

# %%
curve = H.recall_vs_budget(ds, write="episodes")
print(" budget  recall  accuracy  memory tokens/turn")
for p in curve:
    print(f" {p['budget']:6d}  {p['recall']:6.1%}  {p['accuracy']:8.1%}  {p['memory_tokens']:8.0f}")

# %% [markdown]
# ## Exercise 5.2 — find the knee
#
# Write `my_knee(points, key="recall", frac=0.95)`: the point with the smallest `budget` whose `key` reaches
# `frac` of the best value on the curve. Past it, more tokens buy almost nothing — and every token is paid on
# every turn, and (notebook 03) can cost prefix-cache hits too. (This lab's knee is *relative* — 95% of the best;
# memory-core's `knee(tol=0.02)` is *absolute* — within 2 points of the best. They pick the same budget when recall
# saturates near 100% and can differ on a flat curve; PRIMER's Glossary names both. Say which one you used.)

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
# Set the scene with every copy PRIMER §7 lists. User `acme/u1` told the assistant their home address; it went
#
# 1. into an **episode** (tagged with the deletion key `acme/u1/address`) through the memory service, with an
#    Idempotency-Key — a row in the service's **idempotency** table;
# 2. into a **fact** consolidation derived from it, and a weekly **summary** whose provenance cites the episode
#    (a different deletion key: only provenance links it);
# 3. into a prompt an inference engine served — its **prefix cache** holds those blocks;
# 4. into the **eval set**, as a golden case built from a transcript (07.2 notebook 08 §9);
# 5. into the **audit log** — as hashes, by design.

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
# Write `forget_everywhere(client, llm_url, eval_set, needle)` returning a `DeletionReport` whose `counts` say
# how many copies each surface held:
#
# 1. `DELETE /v1/memories?subject=address` through the service (`client.forget("address")`): its report has the
#    records, vectors, FTS rows, derived records (by provenance) and idempotency rows it removed — start from
#    `report["counts"]`;
# 2. the engine's prefix cache. vLLM v0.30.0 cannot evict one tenant's or one user's blocks: its only tool is the
#    dev-mode `POST {llm_url}/reset_prefix_cache` (the server must run with `VLLM_SERVER_DEV_MODE=1`, verify), which
#    clears **every** tenant's cache and answers only `{"success": bool}` — false while running requests hold
#    blocks, so check it. Call it, require `success`, and record `prompt_cache` as `cached_blocks`, the count the
#    caller passes in: the fake server knows its own blocks (SIMULATED); a real vLLM does not report them. In
#    production you would rather rotate the tenant's `cache_salt` (notebook 03, exercise 3.5, with an epoch in the
#    HMAC input): its blocks become unreachable at once and age out under LRU, without cooling every other tenant's
#    cache — the reset is the operator's runbook step for the residue;
# 3. remove every eval-set case that contains the needle (mutate the list) — `eval_sets`;
# 4. count audit lines that contain the needle — `audit_log` (should be 0: the log holds hashes);
# 5. put `"backups"` in `not_reachable`: they expire on their retention schedule, which the deletion policy
#    must state.

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
# Write `audit_counts(events)`: from the audit JSON lines, a dict `{(tenant, user, event_type): count}` and the
# number of events whose `decision` is `deny`. A forget request should appear as exactly one `memory.forget`.

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
# The paraphrase questions ask the same facts in other words ("Which firm pays my salary?"). The hashing
# embedder misses them by design; a semantic embedder should not. With `MEMLAB_EMBED_URL` set this cell
# re-runs the harness with it — measured on your embedder, not simulated.

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
# **Two minutes.** "We evaluate memory with a seeded planted-facts harness in LongMemEval's and LoCoMo's
# question shapes: 84 questions across seven categories, graded correct, stale, abstained or hallucinated,
# reported with Wilson intervals and the tokens and model calls per answer. It tells us where each design
# fails — retrieval before every turn misses preferences, tools miss what the model does not ask for, a
# pinned profile costs tokens and a call. We set the per-turn memory budget at the knee of recall versus
# tokens. Forgetting is a checklist with counts: a deletion key on every record, provenance to reach derived
# facts, the service's idempotency rows, the engine's prefix cache (the tenant's salt rotated; a full reset as the
# operator's step, since vLLM cannot evict one tenant's blocks), the eval set, a purge of FTS and WAL — then we
# search the files for the bytes. Audit logs carry hashes, not memory; backups age out, and the policy says
# how long."
#
# **Drill 1.** *A user asked us to forget their address; a week later the assistant quoted it. Where was it?* —
# In a copy the delete did not reach: a consolidated summary derived from it (no provenance link), the FTS
# index or WAL (no purge), the engine's prefix cache, a log or an eval set. Deletion keys plus provenance plus
# a byte-level check are the fix.
#
# **Drill 2.** *Abstention is 100% here. Can we drop those questions?* — No: the scripted answerer never
# invents. With a real model abstention and false-premise questions are where hallucination shows up; they
# stay in the golden set.
#
# **Drill 3.** *Why does the paraphrase subset exist if T0 always fails it?* — It measures what the embedder
# contributes: the same facts, the same answerer, different words. The gap between the two columns is the
# case for (and the test of) a semantic embedder.
