"""gwlab — an LLM gateway you can run on a laptop: one OpenAI-compatible front door for many models.

The gateway (`gwlab.gateway`) decides whether a request runs and which model, provider or pool serves it;
it holds the provider keys, meters tokens as they stream, caches what is safe to cache and writes one ledger
row and a set of GenAI spans per request. The fake providers (`gwlab.fakes`) stand in for hosted APIs in two
dialects (OpenAI chat completions and Anthropic Messages) so every notebook runs offline (T0); point the same
gateway at a real `vllm serve` and the numbers become measurements (T1).

Concepts: ../PRIMER.md (the llm-gateway primer). This lab never imports the core package (gwcore).
"""
__version__ = "0.1.0"
