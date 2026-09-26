"""Which experts a step touches, and what it streams: layer 01's PRIMER §3.6 reproduced, plus the kernel layout."""
import http.server
import json
import threading
import time

import numpy as np
import pytest

from moelab import configs as C
from moelab import stream as S

MIX, Q3, L8, H200 = C.get("mixtral-8x7b"), C.get("qwen3-30b-a3b"), C.get("llama-3.1-8b"), C.gpu("H200")


def test_roofline_primer_3_6_table_reproduced():
    """Mixtral on an H200 at 1K context: experts/layer, GB and ms per step (roofline PRIMER §3.6)."""
    want = {1: (2.00, 25.6, 5.34, 8.0), 4: (5.47, 65.1, 13.57, 29.1), 16: (7.92, 94.4, 19.66, 82.4),
            64: (8.00, 101.7, 21.20, 125.9)}
    for b, (e, gb, ms, q) in want.items():
        s = S.decode_step(MIX, H200, b, 1024)
        assert round(S.experts_touched(8, 2, b), 2) == e
        assert round(s.bytes / 1e9, 1) == gb and round(s.time * 1e3, 2) == ms and s.bound == "memory"
        assert round(S.experts_touched(128, 8, b), 1) == q
    one = S.decode_step(MIX, H200, 1, 1024)
    assert one.weight_bytes == 25_497_182_208 and one.kv_bytes == 134_348_800        # fact sheet §1


def test_crossover_batches_and_the_total_over_active_rule():
    x = [S.decode_crossover_batch(m, H200, 0) for m in (L8, MIX, Q3)]
    assert x == [207, 754, 2055]
    for m, xm in ((MIX, x[1]), (Q3, x[2])):
        ratio = (m.total_params() - m.vocab * m.d_model) / (m.active_params() - m.vocab * m.d_model)
        assert xm / x[0] == pytest.approx(ratio, rel=0.01)


def test_closed_form_is_exact_for_k_distinct_experts():
    for E, k, T in ((8, 2, 3), (64, 8, 16), (128, 8, 32)):
        mc, _ = S.touched_mc(E, k, T, trials=400)
        assert mc == pytest.approx(S.experts_touched(E, k, T), rel=0.02)
    assert S.experts_touched(0, 0, 10) == 1.0


def test_skew_touches_fewer_and_concentrates_load():
    t_u, hot_u = S.touched_mc(64, 8, 16, s=0.0, trials=100)
    t_z, hot_z = S.touched_mc(64, 8, 16, s=1.2, trials=100)
    assert t_z < t_u and hot_z > hot_u
    p = S.zipf_popularity(64, 1.0)
    assert p.sum() == pytest.approx(1.0) and p.max() / p.min() == pytest.approx(64.0)


def test_sample_topk_is_k_distinct():
    ids = S.sample_topk(np.full(16, 1 / 16), 4, 200, np.random.default_rng(0))
    assert ids.shape == (200, 4) and all(len(set(r)) == 4 for r in ids.tolist())


def test_dense_model_streams_everything_at_any_batch():
    q15 = C.get("qwen2.5-1.5b")
    a, b = S.decode_step(q15, C.gpu("L4"), 1, 0), S.decode_step(q15, C.gpu("L4"), 64, 0)
    assert b.weight_bytes - a.weight_bytes == 63 * q15.d_model * 2          # only the embedding gather grows
    moe = C.get("olmoe-1b-7b")
    assert S.decode_step(moe, C.gpu("L4"), 64, 0).weight_bytes / S.decode_step(moe, C.gpu("L4"), 1, 0).weight_bytes > 4


def test_align_block_size_matches_vllm_docstring():
    s, e, n = S.align_block_size([[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], 4, 4)
    assert s.tolist() == [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12] and n == 16
    assert e.tolist() == [1, 2, 3, 4]


def test_padding_waste_falls_with_batch():
    w = [S.padding_waste(b, 8, 64, 16, trials=10) for b in (1, 32, 512)]
    assert w[0] == pytest.approx(1 - 8 / (8 * 16)) and w[0] > w[1] > w[2]


def test_simulation_and_calibration_round_trip():
    truth = S.SimParams(mem_eff=0.55, compute_eff=0.5, overhead_s=3e-3)
    m, g = C.get("qwen2.5-1.5b"), C.gpu("T4")
    pts = [(S.decode_step(m, g, b, 256).t_memory, S.simulate_itl(m, g, b, 256, truth)) for b in (1, 4, 16)]
    eff, ovh = S.fit_efficiency(pts)
    assert eff == pytest.approx(0.55) and ovh == pytest.approx(3e-3)


class _SSE(http.server.BaseHTTPRequestHandler):
    """A stand-in OpenAI completions server: N token chunks, GAP seconds apart, then usage and [DONE]."""
    N, GAP = 6, 0.02

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["stream"] and body["ignore_eos"] and body["max_tokens"] == self.N
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(self.N):
            self.wfile.write(f'data: {json.dumps({"choices": [{"text": f"t{i}"}]})}\n\n'.encode())
            self.wfile.flush()
            time.sleep(self.GAP)
        usage = {"choices": [], "usage": {"completion_tokens": self.N}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())

    def log_message(self, *a):
        pass


def test_measure_itl_against_a_local_stream():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SSE)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        p = S.measure_itl(url, "m", 2, n_requests=4, output_tokens=_SSE.N)
    finally:
        srv.shutdown()
    assert p.errors == 0 and p.output_tokens == 4 * _SSE.N
    assert len(p.itl_s) == 4 * (_SSE.N - 1) and len(p.ttft_s) == 4
    assert 15 < p.median_itl_ms < 80 and p.label == "measured"
