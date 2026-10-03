# Bundled samples — what each file is and where it came from

Every file here is **illustrative**. It shows a format, and it lets the T0 paths run offline. No file is a
measurement of a real model. Generate the files again with the code of this lab (the tests make sure that they are
current).

| File | Format | Made by | Used in |
|---|---|---|---|
| `traces_thinker_illustrative.jsonl` | one `distillab.traces.Trace` per line: question, `reasoning` (vLLM 0.30.0's response field), content, verifier result, reasoning and completion token counts | the fake thinking teacher (`FakeTeacher("thinker")`, **simulated**). It answered 40 generated training problems, with 4 traces each, seeds 0–3. | notebook 03 (T0) |
| `chat_completion_logprobs.json` | a `/v1/chat/completions` response with `logprobs` and `top_logprobs: 2`, cut by `max_tokens` (`finish_reason: "length"`) | the fake teacher (`FakeTeacher("teacher")`, **simulated**). The fake teacher invents its log-probabilities and `alt` tokens. | notebook 02 (the response shape) |
| `spec_decode_metrics_before.txt`, `spec_decode_metrics_after.txt` | vLLM v0.30.0's four spec-decode counters in Prometheus text (`vllm:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`, `..._num_accepted_tokens_total`, `..._num_accepted_tokens_per_pos_total{position}`) | `distillab.draft.simulate_counters` at a per-token acceptance of 0.7 and k = 4 over 20,000 drafts. The data is **synthetic**. It shows how to read the counters, not what any draft gets. | notebook 04 (T0) |

The recorded curves of the small run (`../tinylm_recorded_run.json`) are a real run of
`distillab.tinylm.train.run()` on a CPU. Their label names that machine. The lab shows them only when torch is
absent.
