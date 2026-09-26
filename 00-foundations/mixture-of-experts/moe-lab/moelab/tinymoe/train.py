"""train.py — train the tiny MoE three ways and watch the router: collapse, then balance.

One idea: nothing in next-token loss asks the router to spread tokens. With top-1 routing the
expert that happens to be chosen early gets the gradient, gets better, gets chosen more — the
rich get richer, and some experts starve (their parameters are memory you pay for and never use).
Two fixes, both run here: the Switch auxiliary loss (alpha x E x sum f_e P_e, which only pushes the
*probabilities* P), and DeepSeek-V3's auxiliary-loss-free bias (after every step, raise the bias of
under-loaded experts and lower it for over-loaded ones by a fixed step; the bias only picks experts,
it never weights them — so it cannot bend the model's outputs the way a loss term can).

CPU minutes at most: the defaults train in seconds per run. ``record()`` writes the curves that
ship in ``fixtures/tinymoe_curves.json`` for machines without torch.

    python -m moelab.tinymoe --steps 400 --seeds 0 1 2              # print the three runs
    python -m moelab.tinymoe --record                               # rewrite the bundled curves
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .curves import load_stats, specialisation
from .data import ToyTask
from .model import TinyMoETransformer, router_z_loss, switch_aux_loss

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tinymoe_curves.json"


@dataclass
class TrainConfig:
    balance: str = "none"          # "none" | "aux" | "bias"
    steps: int = 400
    batch: int = 32
    lr: float = 3e-3
    seed: int = 0
    n_experts: int = 8
    top_k: int = 1
    d_model: int = 32
    expert_ff: int = 16
    aux_alpha: float = 0.1         # larger than the usual 0.01: a 300-step run has little time
    bias_rate: float = 0.01        # DeepSeek-V3 used 0.001 over a far longer run (verify)
    z_coef: float = 0.0            # router z-loss weight (1e-3 is the usual starting point)
    log_every: int = 10
    eval_batch: int = 256


@dataclass
class Run:
    config: dict
    steps: list = field(default_factory=list)        # logged step numbers
    ce: list = field(default_factory=list)           # next-token cross-entropy (nats)
    aux: list = field(default_factory=list)          # Switch loss value (logged even when unused)
    load: list = field(default_factory=list)         # per-expert share of assignments over the window
    domain_expert: list = field(default_factory=list)  # final [domains, experts] assignment counts
    token_expert: list = field(default_factory=list)   # final [content tokens, experts] assignment counts
    seconds: float = 0.0
    bayes_ce: float = 0.0

    @property
    def final_load(self) -> np.ndarray:
        return np.asarray(self.load[-1])

    def summary(self) -> str:
        s = load_stats(self.final_load)
        by_dom, by_tok = specialisation(self.domain_expert), specialisation(self.token_expert)
        return (f"{self.config['balance']:5s} seed {self.config['seed']}: ce {self.ce[-1]:.3f} "
                f"(floor {self.bayes_ce:.3f})  max/mean load {s['max_over_mean']:.2f}  dead experts {s['dead']}  "
                f"expert explained by domain {by_dom:.2f}, by token {by_tok:.2f}  [{self.seconds:.1f} s]")


def build(task: ToyTask, cfg: TrainConfig) -> TinyMoETransformer:
    torch.manual_seed(cfg.seed)
    return TinyMoETransformer(task.vocab, task.seq_len, cfg.d_model, 1, cfg.n_experts, cfg.top_k, cfg.expert_ff)


def train(cfg: TrainConfig | None = None, task: ToyTask | None = None, **overrides) -> tuple[Run, TinyMoETransformer]:
    """Train one run; returns the recorded curves and the model."""
    cfg = cfg or TrainConfig(**overrides)
    task = task or ToyTask()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)          # tiny matmuls: one thread is fastest, and immune to a busy machine
    try:
        return _train(cfg, task)
    finally:
        torch.set_num_threads(threads)


def _train(cfg: TrainConfig, task: ToyTask) -> tuple[Run, TinyMoETransformer]:
    rng = np.random.default_rng(cfg.seed)
    model = build(task, cfg)
    moe = model.moes[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    run = Run(asdict(cfg), bayes_ce=task.bayes_loss())
    window = torch.zeros(cfg.n_experts)
    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        x, _ = task.batch(rng, cfg.batch)
        x = torch.from_numpy(x)
        logits = model(x)
        # predict content token t+1 from position t, for t >= 1 (position 0 is the tag)
        ce = F.cross_entropy(logits[:, 1:-1].reshape(-1, logits.shape[-1]), x[:, 2:].reshape(-1))
        r = moe.last
        aux = switch_aux_loss(r["logits"], r["indices"], cfg.n_experts)
        loss = ce
        if cfg.balance == "aux":
            loss = loss + cfg.aux_alpha * aux
        if cfg.z_coef:
            loss = loss + cfg.z_coef * router_z_loss(r["logits"])
        opt.zero_grad()
        loss.backward()
        opt.step()
        counts = r["counts"].float()
        if cfg.balance == "bias":                                  # aux-loss-free: sign of the error
            with torch.no_grad():
                moe.router.bias += cfg.bias_rate * torch.sign(counts.mean() - counts)
        window += counts
        if step % cfg.log_every == 0:
            run.steps.append(step)
            run.ce.append(ce.item())
            run.aux.append(aux.item())
            run.load.append((window / window.sum()).tolist())
            window.zero_()
    run.seconds = time.perf_counter() - t0
    by_domain, by_token = routing_counts(model, task, cfg.eval_batch, cfg.seed + 1000)
    run.domain_expert, run.token_expert = by_domain.tolist(), by_token.tolist()
    return run, model


@torch.no_grad()
def routing_counts(model: TinyMoETransformer, task: ToyTask, n: int = 256, seed: int = 1):
    """How often content tokens pick each expert (all k slots), tallied two ways:
    ``[domains, experts]`` by the sequence's domain and ``[content tokens, experts]`` by the token itself."""
    x, dom = task.batch(np.random.default_rng(seed), n)
    model(torch.from_numpy(x))
    moe = model.moes[0]
    idx = moe.last["indices"].reshape(n, task.seq_len, -1)[:, 1:, :].numpy()   # skip the tag position
    tok = np.repeat((x[:, 1:] - task.n_domains)[..., None], idx.shape[-1], axis=-1)
    dom3 = np.broadcast_to(dom[:, None, None], idx.shape)
    by_domain = np.zeros((task.n_domains, moe.n_experts), dtype=np.int64)
    by_token = np.zeros((task.n_content, moe.n_experts), dtype=np.int64)
    np.add.at(by_domain, (dom3.ravel(), idx.ravel()), 1)
    np.add.at(by_token, (tok.ravel(), idx.ravel()), 1)
    return by_domain, by_token


def record(seeds=(0, 1, 2), balances=("none", "aux", "bias"), path: Path = FIXTURE, **kw) -> dict:
    """Train every (balance, seed) and write the curves bundled for the numpy-only path."""
    runs = []
    for b in balances:
        for s in seeds:
            run, _ = train(TrainConfig(balance=b, seed=s, **kw))
            print(run.summary())
            runs.append(asdict(run))
    doc = {"_label": (f"recorded by `python -m moelab.tinymoe --record` with torch {torch.__version__} on CPU "
                      "(real output of this code on the toy task; illustrative of the phenomenon, not of any real model)"),
           "task": asdict(ToyTask()), "runs": runs}
    path.write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m moelab.tinymoe", description=__doc__.split("\n\n")[0])
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--balance", nargs="+", default=["none", "aux", "bias"])
    p.add_argument("--record", action="store_true", help="rewrite fixtures/tinymoe_curves.json (seeds 0 1 2)")
    a = p.parse_args(argv)
    if a.record:
        record(steps=a.steps)
        return 0
    for b in a.balance:
        for s in a.seeds:
            run, _ = train(TrainConfig(balance=b, seed=s, steps=a.steps))
            print(run.summary())
    return 0
