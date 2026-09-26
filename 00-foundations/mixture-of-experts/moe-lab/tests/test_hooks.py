"""Router traces: vLLM's wire format, the analysis, placement, and the hook adapters for each router family."""
import http.server
import json
import threading

import numpy as np
import pytest

from moelab import env, hooks


def test_routed_experts_round_trip():
    ids = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    b64 = hooks.encode_routed_experts(ids)
    back = hooks.decode_routed_experts(b64)
    assert back.dtype == np.uint8 and (back == ids).all()


def test_fixture_has_vllm_shapes_and_is_labelled():
    ts = hooks.load_fixture()
    assert "illustrative" in ts.label and ts.n_experts == 64 and ts.top_k == 8
    assert ts.domains() == ["chinese", "code", "math", "prose"]
    for t in ts.traces:
        assert t.ids.shape[1:] == (16, 8) and t.ids.max() < 64 and t.source == env.ILLUSTRATIVE
        assert all(len(set(row)) == 8 for row in t.ids[:, 0, :].tolist())        # k distinct experts


def test_utilisation_balancedness_and_hot_experts():
    ids = np.array([[[0, 1]], [[0, 2]], [[0, 3]]])                              # 3 tokens, 1 layer, top-2
    u = hooks.utilisation(ids, 4)
    assert u.tolist() == [[3, 1, 1, 1]]
    assert hooks.balancedness(u)[0] == pytest.approx(1.5 / 3)
    assert hooks.hot_experts(u, 1) == [[(0, 0.5)]]


def test_js_divergence_bounds():
    assert hooks.js_divergence([1, 1], [2, 2]) == pytest.approx(0.0)
    assert hooks.js_divergence([1, 0], [0, 1]) == pytest.approx(1.0)
    ts = hooks.load_fixture()
    div = hooks.domain_divergence(ts)
    assert div.shape == (16,) and (div > 0).all()


def test_placement_and_rank_loads():
    assert hooks.placement(8, 2).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert hooks.placement(8, 2, "round_robin").tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    with pytest.raises(ValueError):
        hooks.placement(6, 4)
    r = hooks.rank_loads(np.array([[4, 0, 1, 1], [1, 1, 1, 1]]), 2)
    assert r.tolist() == [[4, 2], [2, 2]]


def test_rank_imbalance_grows_with_ep_on_the_fixture():
    ts = hooks.load_fixture()
    u = hooks.utilisation(ts.stacked(), ts.n_experts)
    imb = {ep: (lambda r: (r.max(1) / r.mean(1)).mean())(hooks.rank_loads(u, ep)) for ep in (2, 16)}
    assert imb[16] > imb[2] > 1.0


torch = pytest.importorskip("torch") if env.has_torch() else None


def _router(name, layout):
    nn = torch.nn

    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k, self.lin = 2, nn.Linear(8, 6, bias=False)

        def forward(self, x):
            logits = self.lin(x)
            idx = logits.topk(2, -1).indices
            w = logits.softmax(-1).gather(-1, idx)
            if layout == "hf":
                return logits, w, idx
            if layout == "granite":
                return idx, w, logits
            s = torch.full_like(logits, float("-inf")).scatter(1, idx, logits.gather(1, idx)).sigmoid()
            return s, logits
    R.__name__ = name
    return R()


@pytest.mark.skipif(torch is None, reason="torch not installed (T0 numpy path)")
@pytest.mark.parametrize("name,layout", [("OlmoeTopKRouter", "hf"), ("MixtralTopKRouter", "hf"),
                                         ("DeepseekV3TopkRouter", "hf"), ("GraniteMoeTopKRouter", "granite"),
                                         ("Llama4Router", "llama4")])
def test_recorder_reads_each_router_family(name, layout):
    torch.manual_seed(0)
    parent = torch.nn.ModuleList([_router(name, layout), _router(name, layout)])
    rec = hooks.RouterRecorder(parent, top_k=2)
    x = torch.randn(5, 8)
    for r in parent:
        r(x)
    ids = rec.pop()
    rec.remove()
    assert ids.shape == (5, 2, 2)
    for l, r in enumerate(parent):
        want = np.sort(r.lin(x).topk(2, -1).indices.numpy(), 1)
        assert (np.sort(ids[:, l], 1) == want).all()
    assert rec.handles == []


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_recorder_needs_routers():
    with pytest.raises(ValueError, match="no router"):
        hooks.RouterRecorder(torch.nn.Linear(2, 2))


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_unknown_router_layout_is_an_error():
    class OddRouter(torch.nn.Module):
        def forward(self, x):
            return x
    with pytest.raises(TypeError, match="unknown router output layout"):
        hooks.indices_from_output(OddRouter(), torch.zeros(2))


def test_text_histogram():
    h = hooks.text_histogram(np.array([6, 2, 0, 0]), width=6).splitlines()
    assert h[0].startswith("expert   0 ######") and "75.0%" in h[0] and "3.0x fair" in h[0]
    assert len(hooks.text_histogram(np.arange(10), top=3).splitlines()) == 3


def test_served_ids_resolve_to_the_catalogue():
    from moelab import configs
    for served, key in (("allenai/OLMoE-1B-7B-0924-Instruct", "olmoe-1b-7b"), ("allenai/OLMoE-1B-7B-0924", "olmoe-1b-7b"),
                        ("Qwen/Qwen1.5-MoE-A2.7B-Chat", "qwen1.5-moe-a2.7b"),
                        ("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4", "qwen1.5-moe-a2.7b"),
                        ("Qwen/Qwen3-30B-A3B-FP8", "qwen3-30b-a3b")):
        assert configs.by_hf_id(served) is configs.get(key)
    assert configs.by_hf_id("my-served-name") is None


def test_expert_layout_comes_from_the_catalogue_or_fails_loudly(monkeypatch):
    monkeypatch.delenv("MOELAB_EXPERTS", raising=False)
    monkeypatch.delenv("MOELAB_TOPK", raising=False)
    assert hooks.expert_layout("Qwen/Qwen1.5-MoE-A2.7B-Chat") == (60, 4)     # the lab's GKE model, not OLMoE's 64
    with pytest.raises(ValueError, match="MOELAB_EXPERTS"):
        hooks.expert_layout("my-served-name")
    with pytest.raises(ValueError, match="not an MoE"):
        hooks.expert_layout("Qwen/Qwen2.5-1.5B-Instruct")                      # a dense model has no router
    monkeypatch.setenv("MOELAB_EXPERTS", "32")
    assert hooks.expert_layout("my-served-name") == (32, None)


class _Routed(http.server.BaseHTTPRequestHandler):
    """A stand-in `vllm serve --enable-return-routed-experts`: every chat choice carries a top-4 routing
    over 60 experts (Qwen1.5-MoE-A2.7B's layout), 24 layers, 9 tokens."""
    IDS = np.random.default_rng(0).integers(0, 60, (9, 24, 4)).astype(np.uint8)

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["routed_experts_prompt_start"] == 0
        out = json.dumps({"choices": [{"message": {"content": "x"},
                                       "routed_experts": hooks.encode_routed_experts(self.IDS)}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_capture_vllm_infers_the_layout_from_the_served_model(monkeypatch):
    monkeypatch.delenv("MOELAB_EXPERTS", raising=False)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Routed)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        ts = hooks.capture_vllm(url, "Qwen/Qwen1.5-MoE-A2.7B-Chat", prompts={"code": ["a", "b"]})
        with pytest.raises(ValueError, match="top-"):
            hooks.capture_vllm(url, "Qwen/Qwen1.5-MoE-A2.7B-Chat", 60, 8, prompts={"code": ["a"]})
        with pytest.raises(ValueError, match="MOELAB_EXPERTS"):
            hooks.capture_vllm(url, "my-served-name", prompts={"code": ["a"]})
    finally:
        srv.shutdown()
    assert (ts.n_experts, ts.top_k, len(ts.traces)) == (60, 4, 2) and ts.label == env.MEASURED
    assert hooks.utilisation(ts.stacked(), ts.n_experts).shape == (24, 60)      # no phantom experts 60-63
