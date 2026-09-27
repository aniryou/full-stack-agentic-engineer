"""The tiny teacher and students with torch: sampler/scorer consistency, a short run of every method, the
exposure-bias and draft measurements, and the full default run — its claims, and on the recording's torch build
an exact match with the bundled run (skipped when torch is absent)."""
import json
import random
from importlib import resources

import pytest

torch = pytest.importorskip("torch")

from distillab import draft as DR  # noqa: E402
from distillab.tinylm import train as T  # noqa: E402
from distillab.tinylm.model import TinyLM, completion_logits, sample, token_logprobs  # noqa: E402
from distillab.tinylm.task import EOS, PAD, SumTask  # noqa: E402


def test_sampler_logprobs_equal_scorer_logprobs():
    torch.manual_seed(0)
    task = SumTask(4, 5)
    m = TinyLM(15, 32, 1, 2, task.seq_len)
    P = torch.tensor([task.sample(random.Random(i)).prompt for i in range(6)])
    comps, lp, mask = sample(m, P, task.max_completion, 1.0, torch.Generator().manual_seed(0))
    assert torch.allclose(lp * mask, token_logprobs(m, P, comps) * mask, atol=1e-5)
    assert completion_logits(m, P, comps).shape == (6, task.max_completion, 15)
    for row, mm in zip(comps.tolist(), mask.tolist()):
        n = int(sum(mm))
        assert all(t == PAD for t in row[n:]) and (n == len(row) or row[n - 1] == EOS)


QUICK = dict(k=4, teacher_steps=60, teacher_batch=64, student_steps=20, student_batch=32, n_labelled=64,
             eval_every=10, eval_prompts=32, methods=("hard", "kd", "seqkd", "gkd", "seqkd_all"))


def test_a_short_run_of_every_method():
    run = T.run(T.DistillConfig(**QUICK), log=lambda *a: None, keep_models=True)
    assert run["source"] == "measured" and set(run["students"]) == set(QUICK["methods"])
    assert run["students"]["gkd"]["onpolicy_batches"] > 0 and run["students"]["kd"]["onpolicy_batches"] == 0
    for s in run["students"].values():
        assert [c["step"] for c in s["curve"]] == [0, 10, 20]
        f = s["final"]
        assert 0 <= f["accuracy"] <= 1 and 4 <= f["length"] <= 8 and f["accept_greedy"] <= 1 and f["kl"] >= 0
    assert run["teacher"]["eval"]["agree"] == 1.0 and run["teacher"]["eval"]["kl"] == 0.0
    sq = run["data"]["seqkd"]
    assert sq["kept"] <= sq["generated"] == QUICK["n_labelled"]
    models, task = run["models"], run["task"]
    rows = T.exposure(models["kd"], models["teacher"], task, run["eval_problems"], run["teacher_seqs"],
                      torch.Generator().manual_seed(0))
    assert len(rows) == task.max_completion and rows[0]["n own"] == QUICK["eval_prompts"]
    acc = DR.tiny_acceptance(models["kd"], models["teacher"], run["teacher_seqs"])
    assert len(acc) == task.max_completion and all(0 <= r["alpha_greedy"] <= 1 and 0 <= r["alpha"] <= 1 for r in acc)
    with pytest.raises(ValueError):
        T.train_student("nope", models["teacher"], task, T.DistillConfig(**QUICK), [], {}, [], run["teacher_seqs"])


@pytest.fixture(scope="module")
def full_run():
    """The default DistillConfig, as notebook 01 and `python -m distillab tinylm` run it (~1.5 min on a CPU)."""
    return T.run(T.DistillConfig(), log=lambda *a: None)


@pytest.mark.slow
def test_the_default_run_separates_the_four_students(full_run):
    """The lab's claims, on a live run: every distilled student copies the teacher better than hard labels; the
    verifier-filtered SeqKD student inherits the teacher's scratchpad and beats hard labels on accuracy; GKD is
    the closest copy on the student's own samples."""
    s = {m: v["final"] for m, v in full_run["students"].items()}
    t = full_run["teacher"]["eval"]
    for m in ("kd", "seqkd", "gkd"):
        assert s[m]["agree"] > s["hard"]["agree"] and s[m]["kl"] < s["hard"]["kl"] and s[m]["accept"] > s["hard"]["accept"]
        assert s[m]["full"] > s["hard"]["full"] + 0.2 and s[m]["length"] > s["hard"]["length"] + 2
    assert s["seqkd"]["accuracy"] > s["hard"]["accuracy"] + 0.3 and s["kd"]["accuracy"] > s["hard"]["accuracy"] + 0.2
    assert s["gkd"]["rkl"] < s["kd"]["rkl"] and s["gkd"]["kl"] == min(x["kl"] for x in s.values())
    assert full_run["data"]["seqkd"]["full_after"] > full_run["data"]["seqkd"]["full_before"]
    assert s["hard"]["full"] < 0.4 < t["full"]


@pytest.mark.slow
def test_the_recorded_run_is_this_code_on_this_torch(full_run):
    """The bundled run (the no-torch fallback) must be what this code produces. Floating-point results can differ
    across torch builds and thread counts, so the exact comparison runs only where the recording was made with the
    same torch and threads; elsewhere the claims above are the guard."""
    rec = json.loads(resources.files("distillab.assets").joinpath("tinylm_recorded_run.json").read_text())
    if f"torch {torch.__version__} / {full_run['config']['threads']} threads" not in rec["machine"]:
        pytest.skip(f"recorded with {rec['machine']}; this is torch {torch.__version__}")
    fresh = json.loads(json.dumps({k: full_run[k] for k in ("teacher", "students", "data", "config", "params")}))
    assert {k: rec[k] for k in fresh} == fresh
