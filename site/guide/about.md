# About

## Who wrote it

The material in this repository was written by Claude, Anthropic's model, working through Claude Code (Claude Opus
and Claude Fable models). The repository owner directed the work: chose the topics and the order, set the rules the
content follows, reviewed it, and decided what was kept. This site is built from the same files.

## How claims are kept honest

Every technical claim is meant to be one of three things:

- **Computed by code in the repo.** Worked numbers in the primers come from the labs' functions, and the page names
  the function.
- **Pinned by a test.** The cores and labs ship test suites that fix those numbers, so a change that moves one fails.
- **Marked `(verify)` with a date.** Product names, versions, prices and availability change month to month. Each is
  marked, and each primer ends with a Verify list saying what to re-check. Treat an unmarked number that is not
  computed as a mistake worth reporting.

Output that does not come from real hardware is labelled: simulator output says "simulated"; bundled tool output
says "sample output in the documented format (illustrative)".

## How the September 2026 layers were reviewed

The topics added to layers 01–05 in September 2026 each went through the same review before being merged:

1. **Three adversarial reviewers**, each with one lens: *concepts* (is it correct, current and precise),
   *runnability* (do the commands, tests and notebooks run as written, at the tier they claim), and *pedagogy*
   (does a reader learn it in this order, are the exercises and checks fair).
2. **A fixer** that checks each finding against the code and the sources before changing anything.
3. **An independent validator** that re-runs the tests, notebooks and link checks after the fixes.

Older material in layers 00, 04, 06 and 07 predates this process; it was edited for consistency but not put through
the same review.

## Reporting an error

Open an issue on [GitHub](https://github.com/aniryou/full-stack-agentic-engineer/issues) with the page, the claim,
and what you think is right, ideally with a source or a command that shows it.

## Licences

There is no repository-wide licence; each lab carries its own.

| Licence | Labs |
|---|---|
| MIT | `roofline-core`, `gpu-bench-lab` (01); `cuda-nccl-core`, `cuda-nccl-lab` (02); `k8s-gpu-core`, `k8s-gpu-lab` (03); `mini-engine-core`, `vllm-serving-lab` (04); `orchestrator-core`, `inference-gateway-lab` (05); `agentic-scaling-lab`, `agentic-scaling-lab-mistral` (06); `agent-core`, `gcp-agent-platform-lab`, `mistral-agent-core` (07) |
| Apache 2.0 | `agentic-identity-gcp-lab` (06); `lra-gcp` (07) |
| No licence file | everything else, including the primers outside those labs; open an issue before reusing it |

Third-party names and products are trademarks of their owners and are mentioned only to explain how they work.
