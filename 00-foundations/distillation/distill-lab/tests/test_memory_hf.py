"""Training memory (the fact sheet's §8 table) and the T1 trainers' configuration builders — no TRL needed."""
import subprocess
import sys
from pathlib import Path

import pytest

from distillab.cost import load_config
from distillab.hf import gkd, kd, memory as H, sft

LAB = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name,total,lora_all,lora_qv", [
    ("qwen2.5-0.5b-instruct", 494_032_768, 8_798_208, 1_081_344),
    ("qwen3-0.6b", 596_049_920, 10_092_544, 2_293_760),
    ("qwen2.5-1.5b-instruct", 1_543_714_304, 18_464_768, 2_179_072),
    ("qwen3-1.7b", 1_720_574_976, 17_432_576, 3_211_264),
])
def test_param_and_lora_counts(name, total, lora_all, lora_qv):
    cfg = load_config(name)
    assert H.param_count(cfg).total == total
    assert H.lora_params(cfg, 16) == lora_all and H.lora_params(cfg, 16, ["q_proj", "v_proj"]) == lora_qv


def test_plans_on_a_t4():
    s, t = load_config("qwen2.5-0.5b-instruct"), load_config("qwen2.5-1.5b-instruct")
    full = H.plan(s, gpu="T4", batch=4, seq=512)
    assert full["fits"] and full["rows"][0]["GB"] == pytest.approx(7.905, abs=1e-3)
    kd_full = H.plan(s, gpu="T4", batch=4, seq=512, teacher=t)
    assert not kd_full["fits"]                                         # 17.4 GB: the logits sink it
    assert H.plan(s, gpu="T4", regime="lora", batch=2, seq=512, teacher=t, chunk=256)["fits"]
    assert H.logits_bytes(s, 1, 4096, copies=1) / 1e9 == pytest.approx(2.49, abs=0.01)
    with pytest.raises(ValueError):
        H.plan(s, gpu="T4", regime="pure_bf16")                        # a T4 has no bf16
    with pytest.raises(ValueError):
        H.plan(s, gpu="L4", teacher=load_config("qwen2.5-7b-instruct"))  # 151,936 vs 152,064 tokens


def test_trainer_configs_follow_the_card():
    t4 = sft.sft_kwargs(out="o", gpu="T4")
    assert (t4["fp16"], t4["bf16"], t4["gradient_checkpointing"]) == (True, False, True) and t4["learning_rate"] == 2e-5
    assert sft.sft_kwargs(out="o", gpu="L4", lora=True)["bf16"] and sft.lora_kwargs(16)["target_modules"] == "all-linear"
    g = gkd.gkd_kwargs(out="o", teacher="Qwen/Qwen2.5-1.5B-Instruct")
    assert (g["lmbda"], g["beta"], g["temperature"], g["fp16"]) == (0.5, 0.5, 0.9, True)
    assert g["teacher_model_init_kwargs"] == {"dtype": "float16"}                 # a 16-bit frozen teacher on a T4
    assert gkd.GKD_DEFAULTS["temperature"] == 0.9 and gkd.DISTILLATION_DEFAULTS["beta"] == 1.0
    with pytest.raises(ValueError):
        gkd.gkd_kwargs(out="o", teacher="t", lmbda=1.5)
    with pytest.raises(ValueError):
        gkd.check_vocab(load_config("qwen2.5-7b-instruct"), load_config("qwen2.5-0.5b-instruct"))
    assert kd.check_pair(load_config("qwen2.5-1.5b-instruct"), load_config("qwen2.5-0.5b-instruct")) == []
    assert kd.check_pair(load_config("qwen3-1.7b"), load_config("qwen2.5-0.5b-instruct"))   # families differ


@pytest.mark.parametrize("mod", ["sft", "kd", "gkd"])
def test_trainer_clis_dry_run_without_the_gpu_stack(mod):
    args = {"sft": ["--data", "d.jsonl", "--out", "o"], "kd": [], "gkd": ["--data", "d.jsonl", "--out", "o"]}[mod]
    r = subprocess.run([sys.executable, "-m", f"distillab.hf.{mod}", *args, "--dry-run"], cwd=LAB,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "Config" in r.stdout


class _StubTokenizer:
    """apply_chat_template as a word splitter: 4.x returns ids, 5.x a dict with input_ids (both handled)."""

    def __init__(self, v5: bool):
        self.v5 = v5

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True):
        words = " ".join(f"<{m['role']}> {m['content']}" for m in msgs).split() + (["<assistant>"] if add_generation_prompt else [])
        ids = [len(w) for w in words]
        return {"input_ids": ids} if self.v5 else ids


@pytest.mark.parametrize("v5", [False, True])
def test_kd_encode_masks_the_prompt(v5):
    row = {"messages": [{"role": "user", "content": "what is two plus two"}, {"role": "assistant", "content": "four ok"}]}
    enc = kd.encode(_StubTokenizer(v5), row, max_length=64)
    n_prompt = len("<user> what is two plus two <assistant>".split())
    assert enc["labels"][:n_prompt] == [-100] * n_prompt and enc["labels"][n_prompt:] == enc["input_ids"][n_prompt:]
    assert len(enc["input_ids"]) == n_prompt + 2
    assert kd.encode(_StubTokenizer(v5), row, max_length=3)["labels"] == [-100] * 3
