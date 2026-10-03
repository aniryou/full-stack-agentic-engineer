# STE100-STYLE.md — the long-running-durable topic in ASD-STE100

ASD-STE100, Simplified Technical English ("STE"), is a controlled language for technical documentation. It has a
set of writing rules and a dictionary of approved general words, each with one meaning. Technical names and
technical verbs of the domain are permitted in addition. The current issue is Issue 8 (2021). The dictionary is the
property of ASD and is not reproduced here; this file paraphrases the rules from the standard and records the
choices made for one topic.

**The experiment (2026-10-03):** the prose of `07-application-agent-framework/long-running-durable/` (the topic
primer and README, the two lab READMEs, the lab's docs and the Markdown cells of its nine notebook pairs) is
rewritten in STE. Code, notebooks' code cells, tests and deploy files are unchanged. This file is the brief the
rewrite followed; `tools/orchestration/ste_lint.py` checks the rules a program can check. The rest of the repository
keeps the writing rules in `CONTRIBUTING.md`.

---

## 1. What does not change

These are the hard constraints of the rewrite. A rewrite that breaks one of them is wrong, whatever its STE score.

1. **Facts.** Every number, unit, limit, date, product name, version, model id, price and `(verify)` tag stays as it
   is. "60 min" stays "60 min"; "roughly 1 h–24 h" keeps its hedge, because the hedge is a fact.
2. **Code.** Fenced code blocks, Mermaid diagrams, inline code spans, file paths, commands, identifiers, HTTP
   status codes and the comments inside code blocks are code. Do not touch them.
3. **Headings.** Every heading stays verbatim, the H1 included, in Markdown files and in notebooks: anchors, the
   site navigation (`mkdocs.yml` is generated from README H1s) and the notebook titles depend on them.
4. **Links.** Every link target stays verbatim. The link text can be rewritten.
5. **Structure.** The order of sections; tables with the same columns and rows; lists with the same items in the
   same order; callouts; `<details>` blocks; horizontal rules; the Colab cell. A long sentence that lists three or
   more items can become a vertical list, and a paragraph can be divided into two. Never remove an item or a row.
6. **Mathematics.** `$...$` and `$$` blocks stay verbatim.
7. **Quotations.** A quoted prompt or a quoted ask (the drill prompts in quotation marks, an instruction given to a
   model) is a quotation. Keep it verbatim.
8. **Meaning.** Add no claim, remove none, weaken none, strengthen none. When a sentence cannot be put in STE
   without a loss of meaning, keep the meaning, break the STE rule, and say so in your report.
9. **The reader.** An engineer who explains the design in a design review (CLAUDE.md, 2026-09-20). STE speaks to
   the reader as "you" and gives instructions in the imperative. Never a role, an employer or a customer.
10. **The repository's writing rules** still apply (`CONTRIBUTING.md`, "Writing"): mathematics is TeX, paragraphs
    stay under about 150 words, callouts are bold-lead blockquotes or GitHub alerts, no emojis, no marketing
    adjectives, `(verify)` on dated product facts. The "In a design review" and "The one-minute version" callouts
    keep their lead-ins.

## 2. The writing rules

### Words

- **W1. Three sources of words only.** (a) An STE approved general word, with its approved meaning and part of
  speech. (b) A technical name: a noun or an adjective that names a part, a product, a service, a computer term, a
  status or a mathematical term (§3.2 lists this topic's). (c) A technical verb: a verb for a computer process
  (§3.3). Treat a word that is not in a basic 1,000-word English vocabulary as unapproved, unless it is a technical
  name or verb of this topic.
- **W2. One word, one meaning.** "Follow" means "come after", never "obey": "obey the procedure", "do the steps".
  "Close" is a verb; the adjective is "near". "Fall" is "move down"; a quantity "decreases". "Check" and "test" are
  nouns: "do a check", "make sure that", "examine". "Right" is a direction; the opposite of "incorrect" is
  "correct".
- **W3. A technical name is a noun or an adjective, never a verb.** "Write a checkpoint", not "checkpoint the run".
  "Take the lease", not "lease the run". "Write the decision to the journal", not "journal the decision". "Put the
  task in the queue" or "enqueue the task" (enqueue is a technical verb), not "queue the task".
- **W4. One name for one thing**, everywhere (§3.4 fixes the choices).
- **W5. No "should", "may", "might", "could", "would", "shall", "ought".** "Must" for an obligation. "Can" for a
  possibility or a permission. "Will" only for the future. "It is possible that" for a chance.
- **W6. No contractions. No "etc.", "e.g.", "i.e.", "vs.", "&".** Write "for example", "that is", "and",
  "against", or list the items.
- **W7. Prefer the short, everyday word.** §3.1 lists the replacements the standard makes most often.

### Noun phrases

- **N1. No cluster of more than three nouns.** Break it with "of", "for", "in", or rewrite: "the state machine of
  the run", not "the agent run state machine". A hyphenated compound counts as one word: "fan-in counter",
  "long-running agent", "write-ahead intent", "at-least-once delivery".
- **N2. Use articles** (a, an, the) and demonstratives (this, these) where English permits them.

### Verbs

- **V1. Approved verb forms only:** the infinitive, the imperative, the simple present, the simple past, the past
  participle as an adjective, and the future with "will".
- **V2. No -ing forms**, as verbs or as nouns: "the worker does a poll", not "polling"; "a wait", not "waiting";
  "when the engine schedules the task", not "scheduling". Exceptions: an -ing word that is part of a technical name
  (long-running, Cloud Logging, logging, monitoring, tracing, streaming, routing, ordering key, pricing, batching,
  caching, sharding, training, embedding, fine-tuning, billing, mapping) and the approved nouns "warning", "thing",
  "nothing", "something", "everything", "anything", "during", "morning", "string", "setting".
- **V3. Active voice.** "The engine writes the checkpoint", not "the checkpoint is written". In descriptive text
  the passive is permitted only when the agent of the action is unknown or unimportant and the active form would be
  unclear; in a procedure, never. A state is not a passive: "the run is stuck", "the lease is expired" (a past
  participle used as an adjective) are correct.
- **V4. Tenses.** The present for what the system does. The past for what occurred in an example or a test.
  "Will" for the future only.
- **V5. One instruction per sentence, in the imperative:** "Run the tests. Then open the notebook." Two actions in
  one sentence only when they occur at the same time.

### Sentences

- **S1. One topic per sentence.**
- **S2. At most 20 words in a procedure** (an instruction, a numbered step, a "Run it" line) **and 25 in
  description.** A hyphenated compound, a number, an inline code span, a link and a formula count as one word each.
- **S3. Do not omit words to make a sentence short.** Keep the articles, the subject, "that" and "which". A
  telegraphic fragment ("crash after side effect → memo + key") becomes full sentences.
- **S4. No dashes to replace words. No arrows in prose.** A colon introduces a list. Parentheses hold a
  cross-reference or a short clarification, not an aside with its own topic: make that a sentence.
- **S5. A condition first, then the instruction, with a comma:** "If the lease is held, return 503."
- **S6. A vertical list or a table** for a sequence and for three or more parallel items.

### Paragraphs

- **P1. One topic per paragraph, the topic sentence first.**
- **P2. At most six sentences per paragraph.**
- **P3. Show the relation between sentences** with connecting words: "then", "after that", "also", "but",
  "because", "thus", "if", "when", "before", "after", "until". Not "however" (but), "therefore" (thus), "in
  addition" (also).

### Safety instructions and callouts

- **C1.** A callout about a loss (a double charge, lost data, a run that is stuck) is a caution; a callout with
  information is a note. Keep the repository's forms: `> [!CAUTION]`, `> [!WARNING]`, `> [!NOTE]`, or a blockquote
  with a bold lead-in such as `> **Pitfall.**`, `> **Verify.**`, `> **In a design review**`.
- **C2.** A caution starts with a command, then gives the reason: "Do not put the budget in the prompt. The model
  has no counter."

### Layout and punctuation

- **L1.** Keep the Markdown as it is (tables, lists, headings, blockquotes, `<details>`), within the permissions of
  §1 item 5.
- **L2.** A semicolon chain becomes separate sentences. In a table cell, a semicolon-separated list of fragments
  becomes short sentences or a comma list.
- **L3.** Numbers, units, ranges (10–600 s), dates and the `(verify)` tag stay as in the original.

## 3. Vocabulary for this topic

### 3.1 The usual replacements

The standard replaces these words. Use the replacement; when you are not sure that a word is approved, use the
shorter everyday word that keeps the meaning.

| Not approved | Use |
|---|---|
| accomplish, achieve, carry out, perform | do |
| additional, further | more |
| additionally, in addition, furthermore | also |
| allow, enable | let, permit |
| alter, modify | change |
| amount | quantity |
| appropriate | correct, applicable |
| assure, ensure, verify (as a verb; the `(verify)` tag is a tag) | make sure |
| attempt (as a verb; "attempt" the noun is a technical name here) | try |
| avoid | prevent, do not |
| big, huge | large |
| choose, pick | select |
| commence, initiate | start |
| comprise, consist of | contain, have |
| consequently, therefore, hence | thus |
| demonstrate, indicate | show |
| determine | find, calculate, decide |
| due to | because of |
| employ, utilize | use |
| enough | sufficient |
| establish | make, find |
| fix (verb or noun) | repair; a correction, a solution |
| happen | occur |
| however | but |
| in order to | to |
| little, tiny | small |
| locate | find |
| obtain, acquire | get |
| prior to | before |
| proceed | continue |
| provide | supply, give |
| quick, quickly | fast |
| regarding | about |
| remain | stay |
| require, required | must, necessary |
| retain | keep |
| subsequently | then |
| terminate | stop, end |
| very, quite, rather, essentially, basically, simply | (remove) |
| whether | if |
| whilst | while |
| wrong | incorrect |

Approved words you will use often: make sure, necessary, possible, permit, let, cause, occur, thus, but, also,
then, because, if, when, until, before, after, during, sufficient, approximately, correct, incorrect, applicable,
available, different, same, other, each, all, some, many, much, more, most, less, large, small, long, short, fast,
slow, high, low, full, empty, safe, start, stop, continue, do, make, get, give, put, send, receive, read, write,
find, show, keep, hold, move, go, stay, wait, use, have, know, think, try, want, look, tell, count, calculate,
measure, examine, monitor, record, remove, replace, repair, release, set, select, change, increase, decrease,
divide, add, compare, connect, open, close, install, operate, prevent, protect.

### 3.2 Technical names (nouns and adjectives)

Use these as they are, always with the same meaning:

- The engine: agent, long-running agent, run, child run, parent run, step, task, named task, stale task, attempt,
  worker, replica, instance, store, document, run document, queue, lease, lease TTL, heartbeat, checkpoint,
  journal, history, effect, effect record, idempotency key, intent, write-ahead intent, reaper, tick, wake-up,
  callback, webhook, event, gate, approval, token, budget, deadline, saga, compensation, fan-out, fan-in, planner,
  aggregator, critic, reflection loop, workflow, workflow version, version (the compare-and-set token), state,
  state machine, invariant (I1, I2, I3), crash, crash window, retry, backoff, timeout, dead-letter topic,
  transaction, compare-and-set (CAS), outbox, poll, poll deadline, trigger, triggerer, tool, tool call, side effect,
  model, prompt, context, session, artifact, memory.
- Delivery: at-least-once delivery, effectively-once, exactly-once, duplicate delivery, late delivery.
- Statuses and record kinds, as code: `PENDING`, `RUNNING`, `WAITING`, `WAITING_HUMAN`, `WAITING_EVENT`,
  `SUCCEEDED`, `FAILED`, `COMPENSATING`, `COMPENSATED`, `STARTED`, `HUMAN`.
- Products and standards: Cloud Run (service, job, instance, revision), Cloud Tasks, Pub/Sub, Firestore, Cloud
  Scheduler, Cloud Workflows, Eventarc, Cloud Logging, Cloud Trace, Cloud Monitoring, Secret Manager, IAM, IAP,
  Cloud SQL, AlloyDB, Spanner, Memorystore, Cloud Storage (GCS), BigQuery, Cloud Build, Document AI, Vertex AI,
  Gemini, Agent Engine, Agent Runtime, Memory Bank, ADK 2, LangGraph, Mistral, Mistral Workflows, Temporal,
  Terraform, Docker, SQLite, FastAPI, Python, JSON, OIDC, OAuth, OpenTelemetry, HTTP, the HTTP status codes.
- The repository: tier (T0, T3), notebook, blank, solution, exercise, design review, design drill, primer, core,
  lab, module 07.3.

### 3.3 Technical verbs (computer processes, permitted as verbs)

enqueue, dequeue, deploy, log, ack (acknowledge), retry, resume, replay, record, poll (also "do a poll"), schedule,
scale, crash, fail, approve, reject, cancel, release (a lease), expire, commit, roll back, call, return, raise (an
exception), run (a program, a test, a notebook), start, stop, spawn (a child run), re-drive (the reaper starts a
stuck step again), compensate, wake (a run; the `/wake` endpoint).

### 3.4 Choices for this topic

| Instead of | Write |
|---|---|
| a human (the actor) | a person; keep "human-in-the-loop", "human gate" and `HUMAN` as names |
| park the run, the run sleeps, the run dozes | the run waits (status `WAITING`); a wait costs nothing |
| the worker dies | the worker stops, the worker crashes |
| double-fire, fire twice | sends the tick two times |
| wedge the run | block the run |
| burn retries | use retries |
| swallow an error | hide an error |
| kick off, spin up | start |
| the happy path | the path with no failure |
| gotcha, pitfall (in prose) | a trap; `> **Pitfall.**` stays as a callout lead-in |
| bump the attempt | increase the attempt |
| memoise, memo | record the result; the record |
| flaky | intermittent (a step that fails some of the time) |
| stuck | stuck (a technical adjective here: a run with no task and no lease) |
| hand-rolled | written by hand |
| plumbing | the infrastructure code |
| naive loop | a loop with no journal |
| re-ask the model | ask the model again |
| trivial (a quantity) | small |
| cheap, expensive (a cost) | low-cost, high-cost; or give the cost |

## 4. Examples

Each example keeps every fact of the original.

**Original.** A chat agent lives inside a request: seconds long, purely reactive, all state in RAM. A
long-running agent has to wait — for a queue, a human, a batch job, a clock — for minutes to weeks, on
infrastructure where any process can be killed at any line.

**STE.** A chat agent lives in one request. The request is seconds long, the agent only reacts, and all of its
state is in RAM. A long-running agent must wait for a queue, a person, a batch job or a clock. The wait can be
minutes or weeks. The infrastructure can stop any process at any line.

**Original.** The agent must not *hold a process* across any of these. A Cloud Run instance can be scaled to
zero, preempted, redeployed, or OOM-killed; a Cloud Run **service** request times out at 60 minutes; a **job** task
at up to 7 days. Holding memory across a wait is not a design, it is a bet.

**STE.** The agent must not keep a process during a wait. The platform can scale a Cloud Run instance to zero,
preempt it, deploy it again, or stop it when it has no memory. A Cloud Run **service** request stops after 60
minutes. A **job** task stops after 7 days at the most. A design that keeps memory during a wait is a bet, not a
design.

**Original.** Leases *expire*, which is the difference from a lock: a dead worker cannot wedge a run forever.

**STE.** A lease expires. This is the difference between a lease and a lock. A dead worker cannot block a run
forever.

**Original (a procedure).** Check Cloud Run logs for the run ids; raise `--timeout`, lower `--concurrency`, or
split the step. The reaper will re-drive once fixed.

**STE.** Examine the Cloud Run logs for the run ids. Increase `--timeout`, decrease `--concurrency`, or divide the
step. When the worker is repaired, the reaper re-drives the runs.

**Original (a callout).** Never put them in the prompt.

**STE.** Do not put the limits in the prompt. The model has no counter and no cost meter.

**Original (telegraphic).** Failure modes → mitigations: crash after side effect (memo + key) · duplicate
delivery (journal index guard + named tasks) · zombie worker (lease TTL) · runaway (budget).

**STE.** Failure modes and their mitigations:

- A crash after the side effect: the effect record and the idempotency key.
- A duplicate delivery: the journal index guard and the named task.
- A zombie worker: the lease TTL.
- A runaway loop: the budget.

## 5. How to check

```bash
python3 tools/orchestration/ste_lint.py <file.md|notebook.ipynb|dir> ...   # findings, then a summary
python3 tools/orchestration/ste_lint.py --summary <paths>                  # the summary only
python3 tools/orchestration/ste_lint.py --json <paths>                     # machine-readable
```

Errors: a sentence over 25 words, a paragraph over six sentences, a modal verb from W5, a contraction, a dash or an
arrow in prose (S4). Warnings: an -ing word outside the exceptions of V2, a probable passive, a semicolon, an
abbreviation from W6, a word from the table in §3.1, a numbered step over 20 words. Code, headings, link targets and
mathematics are skipped. The target of the rewrite: zero errors, and every warning repaired or justified as a
technical name.

The linter is a heuristic. It cannot see a noun cluster, a word used with an unapproved meaning, or a lost fact.
Those need a reader: the rewrite had each file verified by an agent that compared it with the original, sentence by
sentence, and was told to refute the rewrite.
