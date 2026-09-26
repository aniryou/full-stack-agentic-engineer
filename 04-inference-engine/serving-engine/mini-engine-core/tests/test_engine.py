"""The step loop end to end: same tokens as the dense reference, stop conditions, aborts, traces."""
import pytest

from minengine import EOS, Engine, SamplingParams, TinyLM, encode
from minengine.sampler import log_softmax

MODEL = TinyLM()
PROMPTS = ["The engine runs a loop. ", "When two prompts start with the same text", "A"]


def test_batched_paged_engine_matches_dense_greedy():
    eng = Engine(MODEL, num_blocks=64, block_size=4, max_num_batched_tokens=16, max_num_seqs=2)
    outs = eng.generate(PROMPTS, SamplingParams(max_tokens=12, temperature=0))
    assert [o.token_ids for o in outs] == [MODEL.generate_dense(encode(p), 12) for p in PROMPTS]
    assert all(o.finish_reason == "finished_length_capped" for o in outs)
    assert eng.kv.num_free_blocks == 64


def test_prefix_caching_changes_cost_not_tokens():
    system = "The engine runs a loop. Each step it picks the requests to run. "
    prompts = [system + q for q in ("Why?", "How often?", "What next?")]
    runs = {}
    for caching in (True, False):
        eng = Engine(MODEL, num_blocks=64, block_size=4, max_num_batched_tokens=32, enable_prefix_caching=caching)
        outs = [eng.generate([p], SamplingParams(max_tokens=8, temperature=0))[0] for p in prompts]
        runs[caching] = outs
    assert [o.token_ids for o in runs[True]] == [o.token_ids for o in runs[False]]
    assert [o.num_cached_tokens for o in runs[True]] == [0, 64, 64]          # 16 full blocks of the system prompt
    assert all(o.num_cached_tokens == 0 for o in runs[False])


def test_a_burst_sharing_a_prefix_hits_within_one_step():
    """Requests admitted in the SAME step share the prefix a previous one is computing right then
    (blocks are published at scheduling time, as in vLLM), and the K/V they read is exact."""
    system = "The engine runs a loop. Each step it picks the requests to run. "
    prompts = [system + q for q in ("Why?", "How often?", "What next?")]
    eng = Engine(MODEL, num_blocks=64, block_size=4, max_num_batched_tokens=256)
    outs = eng.generate(prompts, SamplingParams(max_tokens=6, temperature=0))
    assert eng.history[0].num_batched_tokens == 68 + 10 + 10                   # all three in step 1
    assert [o.num_cached_tokens for o in outs] == [0, 64, 64]
    for p, o in zip(prompts, outs):
        ids = encode(p)
        ref = MODEL.forward_dense(ids + o.token_ids)
        for i, ((lp, _), t) in enumerate(zip(o.logprobs, o.token_ids)):
            assert abs(lp - log_softmax(ref[len(ids) - 1 + i])[t]) < 1e-9          # paged + shared == dense


def test_stop_conditions():
    eng = Engine(MODEL, num_blocks=64, block_size=4)
    greedy = MODEL.generate_dense(encode("The engine "), 20)
    text = bytes(greedy).decode()
    stop = text[5:7]
    o = eng.generate(["The engine "], SamplingParams(max_tokens=20, temperature=0, stop=(stop,)))[0]
    assert o.text == text[:text.index(stop)] and o.finish_reason == "finished_stopped"
    o = eng.generate(["The engine "], SamplingParams(max_tokens=20, temperature=0, stop_token_ids=(greedy[3],)))[0]
    assert o.token_ids == greedy[:greedy.index(greedy[3]) + 1]
    sampled = eng.generate(["The engine runs."] * 8, [SamplingParams(max_tokens=200, seed=s) for s in range(8)])
    assert any(o.token_ids[-1] == EOS and o.finish_reason == "finished_stopped" for o in sampled)
    ignore = eng.generate(["x"], SamplingParams(max_tokens=30, ignore_eos=True, seed=3))[0]
    assert len(ignore.token_ids) == 30


def test_abort_frees_blocks_and_trace_records_steps():
    eng = Engine(MODEL, num_blocks=16, block_size=4, max_num_batched_tokens=8)
    rid = eng.add_request("The engine runs a loop.", SamplingParams(max_tokens=50))
    eng.step()
    assert eng.history[-1].batch[0][1] == "chunk" and "kv 2/16" in eng.trace()
    eng.abort(rid)
    assert eng.kv.num_free_blocks == 16 and eng.output(rid).finish_reason == "finished_aborted"


def test_a_prompt_that_cannot_fit_is_rejected_at_the_door():
    eng = Engine(MODEL, num_blocks=2, block_size=4)                 # max_model_len defaults to the 8-token cache
    with pytest.raises(ValueError):
        eng.add_request("far too long for two blocks")
