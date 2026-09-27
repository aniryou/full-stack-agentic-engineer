# Bundled samples — what each file is and where it came from

Every file here is **illustrative**: it shows a format and lets the T0 paths run offline. None is a
measurement of a real model. Regenerate them with this lab's code (the tests check that they are current).

| File | Format | Made by | Used in |
|---|---|---|---|
| `traces_thinker_illustrative.jsonl` | one `distillab.traces.Trace` per line: question, `reasoning` (vLLM 0.30.0's response field), content, verifier result, reasoning and completion token counts | the fake thinking teacher (`FakeTeacher("thinker")`, **simulated**) answering 40 generated training problems, 4 traces each, seeds 0–3 | notebook 03 (T0) |
| `chat_completion_logprobs.json` | a `/v1/chat/completions` response with `logprobs` and `top_logprobs: 2`, cut by `max_tokens` (`finish_reason: "length"`) | the fake teacher (`FakeTeacher("teacher")`, **simulated**); its log-probabilities and `alt` tokens are made up | notebook 02 (the response shape) |
| `spec_decode_metrics_before.txt`, `spec_decode_metrics_after.txt` | vLLM v0.30.0's four spec-decode counters in Prometheus text (`vllm:spec_decode_num_drafts_total`, `..._num_draft_tokens_total`, `..._num_accepted_tokens_total`, `..._num_accepted_tokens_per_pos_total{position}`) | `distillab.draft.simulate_counters` at a per-token acceptance of 0.7 and k = 4 over 20,000 drafts — **synthetic**, to show how the counters read, not what any draft achieves | notebook 04 (T0) |

The tiny run's recorded curves (`../tinylm_recorded_run.json`) are a real run of `distillab.tinylm.train.run()`
on a CPU, labelled with that machine; shown only when torch is absent.
