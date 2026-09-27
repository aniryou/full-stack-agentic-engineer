"""distillab — distillation hands on: a tiny teacher and student trained four ways on a CPU, teacher data from a
served model, reasoning traces, a distilled speculative draft, and whether a student pays for itself.

    tinylm/      a tiny teacher and a narrower student: hard labels, logit KD, SeqKD, GKD (torch, lazy)
    losses       the distillation losses in torch: Hinton's KD, GKD's generalised JSD, per-token rewards
    data         generated arithmetic and logic problems with scratchpads and a verifier (no download)
    client       an OpenAI-compatible client: samples, top-k log-probs, scoring a text with prompt_logprobs
    fakeserver   a fake vLLM teacher for T0 (every answer and log-prob simulated)
    teacher      teacher data: n samples, verifier filter, dedup, the token bill, TRL-format JSONL
    traces       reasoning traces: collect, filter by verifier and length, what the student inherits
    draft        a distilled draft for speculative decoding: acceptance, speedup, vLLM's config and counters
    agreement    KL, top-1 agreement, top-k overlap, accuracy with Wilson intervals, the capability gap
    cost         roofline serving cost, the fixed cost of distilling, break-even and the cascade
    hf/          T1 trainers: SFT (TRL), logit KD, GKD / DistillationTrainer; a training-memory calculator
    metrics      vLLM's Prometheus names (throughput, spec decode): a parser and a writer
    report, env  labelled tables and reports; tier detection

Concepts: ``../PRIMER.md``. Nothing here imports the topic's core (``distill-core``) — the lab stands alone.
"""
__version__ = "0.1.0"
