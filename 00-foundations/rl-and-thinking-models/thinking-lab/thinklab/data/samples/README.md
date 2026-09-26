# Sample outputs (illustrative)

Every file here is **sample output in the documented format (illustrative)**: the field names, nesting and
markers follow vLLM v0.30.0 (`vllm/entrypoints/openai/chat_completion/protocol.py`, `docs/features/reasoning_outputs.md`),
SGLang (`docs/advanced_features/separate_reasoning.mdx`) and the gpt-oss Harmony format; the *text and numbers*
were written for this lab, not recorded from a GPU. The notebooks parse them at T0 and parse the real thing at T1.

| File | What it shows |
|---|---|
| `vllm_chat_qwen3_thinking.json` | `vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3`: `message.reasoning` + `message.content`, `usage.completion_tokens_details.reasoning_tokens` |
| `vllm_chat_qwen3_nothinking.json` | the same request with `chat_template_kwargs: {"enable_thinking": false}`: `reasoning: null` |
| `vllm_chat_truncated.json` | `max_tokens: 64` on a question that needs more thinking: `content: null`, `finish_reason: "length"` |
| `sglang_chat_reasoning_content.json` | the SGLang / DeepSeek-API spelling: `message.reasoning_content` |
| `vllm_stream_qwen3.sse` | a streamed response: role chunk, `delta.reasoning` chunks, `delta.content` chunks, usage, `[DONE]` |
| `deepseek_r1_raw.txt` | raw R1-distill text as `/v1/completions` returns it: no opening `<think>` (the template put it in the prompt) |
| `gptoss_harmony_raw.txt` | raw gpt-oss text: `analysis` and `final` channels |
| `sim_eval_records.jsonl` | per-sample outcomes of the lab's **simulated** model on the generated eval set (regenerate: `python -m thinklab.thinking.recorded`) |
