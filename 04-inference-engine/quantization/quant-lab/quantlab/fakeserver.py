"""fakeserver.py — a fake vLLM whose speed follows the quantization scheme (T0, simulated).

One idea: to practise measuring schemes you need a server that answers like ``vllm serve`` and
slows down or speeds up the way a real engine would when its weights, GEMM precision or KV dtype
change. This one runs :class:`quantlab.bench.Engine` in real time — every step sleeps for the
roofline step time of the chosen :class:`quantlab.bench.Profile` — behind ``/v1/completions``
(SSE streaming with ``stream_options.include_usage``), ``/v1/models``, ``/health``, ``/version``
and a small ``/metrics`` with vLLM's metric names. Prompts are counted in whitespace-separated
words; output text is filler. Everything it reports is **simulated**: ``/version`` says
``"simulated": true`` and responses carry ``x-quantlab-simulated: true``. Standard library only.

    with FakeServer(bench.profile("qwen2.5-1.5b-instruct", "L4", "w4a16")) as url: ...
    python -m quantlab fake --model qwen2.5-1.5b-instruct --gpu L4 --scheme fp8 --port 8000
"""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .bench import Engine, Profile, Seq


class FakeServer:
    def __init__(self, prof: Profile, model: str = "quantlab-fake", host: str = "127.0.0.1", port: int = 0,
                 max_num_seqs: int = 256, max_num_batched_tokens: int = 2048, time_scale: float = 1.0):
        self.prof, self.model, self.time_scale = prof, model, time_scale
        self.engine = Engine(prof, max_num_seqs, max_num_batched_tokens)
        self.cv = threading.Condition()
        self.streams: dict = {}
        self.counters = {"prompt": 0, "generation": 0, "success": 0}
        self.t0 = time.perf_counter()
        self._stop = False
        self.httpd = ThreadingHTTPServer((host, port), self._handler())
        self.httpd.daemon_threads = True
        self.url = f"http://{host}:{self.httpd.server_address[1]}"

    # -- the engine loop -------------------------------------------------------------------------
    def _now(self) -> float:
        return time.perf_counter() - self.t0

    def _loop(self):
        while True:
            with self.cv:
                while not self._stop and not self.engine.has_work():
                    self.cv.wait()
                if self._stop:
                    return
                plan = self.engine.schedule()
            if not plan.decode and not plan.prefill:
                time.sleep(0.001)
                continue
            time.sleep(plan.time * self.time_scale)
            with self.cv:
                for s, _, done in self.engine.commit(plan, self._now()):
                    self.counters["generation"] += 1
                    if done:
                        self.counters["success"] += 1
                    q = self.streams.get(s.rid)
                    if q is not None:
                        q.put(done)

    def submit(self, prompt_len: int, output_len: int):
        q: queue.Queue = queue.Queue()
        with self.cv:
            s = Seq(uuid.uuid4().int & 0xFFFFFFFF, self._now(), max(1, prompt_len), max(1, output_len))
            self.engine.add(s)
            self.streams[s.rid] = q
            self.counters["prompt"] += s.prompt_len
            self.cv.notify()
        return s, q

    def metrics_text(self) -> str:
        e = self.engine
        used = 1 - e.free_blocks / max(1, self.prof.num_blocks)
        lab = f'model_name="{self.model}",engine="0"'
        return "\n".join([
            "# quantlab fake server: SIMULATED values with vLLM's metric names",
            f"vllm:num_requests_running{{{lab}}} {len(e.running)}",
            f"vllm:num_requests_waiting{{{lab}}} {len(e.waiting)}",
            f"vllm:kv_cache_usage_perc{{{lab}}} {used:.6f}",
            f"vllm:prompt_tokens_total{{{lab}}} {self.counters['prompt']}",
            f"vllm:generation_tokens_total{{{lab}}} {self.counters['generation']}",
            f'vllm:request_success_total{{{lab},finished_reason="length"}} {self.counters["success"]}',
            f'vllm:cache_config_info{{block_size="{self.prof.block_size}",cache_dtype="{self.prof.kv_cache_dtype}",'
            f'num_gpu_blocks="{self.prof.num_blocks}",simulated="true"}} 1.0', ""])

    # -- HTTP ------------------------------------------------------------------------------------
    def _handler(self):
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):          # quiet
                pass

            def _json(self, obj, status=200):
                body = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("x-quantlab-simulated", "true")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/health":
                    self._json({})
                elif self.path == "/version":
                    self._json({"version": "quantlab-fakeserver", "simulated": True, "scheme": srv.prof.scheme})
                elif self.path == "/v1/models":
                    self._json({"object": "list", "data": [{"id": srv.model, "object": "model",
                                                             "owned_by": "quantlab (simulated)"}]})
                elif self.path == "/metrics":
                    body = srv.metrics_text().encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; version=0.0.4")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                if self.path != "/v1/completions":
                    return self._json({"error": "only /v1/completions"}, 404)
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = req.get("prompt", "")
                prompt = prompt[0] if isinstance(prompt, list) else prompt
                n_prompt, n_out = len(str(prompt).split()), int(req.get("max_tokens") or 16)
                try:
                    seq, q = srv.submit(n_prompt, n_out)
                except ValueError as e:
                    return self._json({"object": "error", "message": str(e)}, 400)
                rid = "cmpl-" + uuid.uuid4().hex[:12]
                usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out}
                if not req.get("stream"):
                    while not q.get():
                        pass
                    srv.streams.pop(seq.rid, None)
                    return self._json({"id": rid, "object": "text_completion", "model": srv.model, "usage": usage,
                                       "choices": [{"index": 0, "text": " tok" * n_out, "finish_reason": "length"}]})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("x-quantlab-simulated", "true")
                self.end_headers()
                try:
                    while True:
                        done = q.get()
                        chunk = {"id": rid, "object": "text_completion", "model": srv.model,
                                 "choices": [{"index": 0, "text": " tok", "finish_reason": "length" if done else None}]}
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                        self.wfile.flush()
                        if done:
                            break
                    if (req.get("stream_options") or {}).get("include_usage"):
                        self.wfile.write(b"data: " + json.dumps({"id": rid, "choices": [], "usage": usage}).encode() + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    srv.streams.pop(seq.rid, None)
        return H

    # -- lifecycle -------------------------------------------------------------------------------
    def start(self) -> str:
        threading.Thread(target=self._loop, name="quantlab-engine", daemon=True).start()
        threading.Thread(target=self.httpd.serve_forever, name="quantlab-http", daemon=True).start()
        return self.url

    def stop(self) -> None:
        with self.cv:
            self._stop = True
            self.cv.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
