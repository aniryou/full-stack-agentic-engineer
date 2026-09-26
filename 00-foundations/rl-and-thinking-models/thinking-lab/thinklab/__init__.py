"""thinklab — thinking models and RL post-training, hands on: from a tiny transformer that learns to use
a scratchpad under GRPO, to a real thinking model on one GPU, to the serving shape its long outputs create.

    tinyrl/      GRPO on a tiny transformer (torch, lazy): SFT warm-up, verifier, curves
    thinking/    an OpenAI-compatible client for thinking models: switches, budgets, best-of-n, eval set
    parsers      reasoning-block parsers (qwen3, deepseek_r1, gpt-oss Harmony; streaming)
    templates    Qwen3-style chat rendering: why dropped thinking ends the prefix-cache hit
    fakemodel    a simulated thinking model: heavy-tailed thinking, accuracy that grows with it
    engine       a timing emulator of a vLLM-style engine for long outputs (simulated)
    fakeserver   a fake vLLM with a reasoning parser (T0 target; every number simulated)
    workload     output-length distributions → the serving shape; open-loop load; modes compared
    rollout      the bookkeeping of an RL step whose rollouts come from an engine (+ the T1 vLLM path)
    metrics      vLLM's Prometheus metric names, a writer and a parser
    report       tables, text charts, labelled JSON/Markdown reports
    env          tier detection (THINKLAB_URL, GPU, torch, vLLM, docker)

Concepts: ``../PRIMER.md``. Nothing here imports the topic's core (``rl-core``) — the lab stands alone.
"""
__version__ = "0.1.0"
