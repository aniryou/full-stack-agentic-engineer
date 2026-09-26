# fixtures — what the notebooks read when there is no GPU

Two kinds of bundled data, labelled wherever they are printed:

| File | Kind | Made by | Read by |
|---|---|---|---|
| `tinymoe_curves.json` | **recorded**: real output of `moelab.tinymoe` on a CPU (torch 2.14), 3 balancing modes × seeds 0–2 | `python -m moelab.tinymoe --record` | notebook 01 without torch |
| `router_traces_olmoe.json` | **sample output in the documented format (illustrative)**: chat responses shaped like `vllm serve --enable-return-routed-experts` returns them (`choices[0].routed_experts` = base64 `.npy`, `(tokens − 1, 16 layers, top-8)`), routing drawn from an invented skewed model | `tools/make_fixtures.py` | notebooks 02 and 04, `python -m moelab trace` |
| `bench_serve_olmoe_2xT4_{tp,tp_ep,dp_ep}_c{1,16}.txt` | **illustrative**: the `Serving Benchmark Result` block of vLLM v0.30.0's `vllm bench serve`, numbers from `moelab.ep`'s simulation | `tools/make_fixtures.py` | notebook 04, `python -m moelab bench-parse` |
| `vllm_startup_olmoe_t4_offload.log`, `vllm_startup_olmoe_l4.log` | **illustrative**: `vllm serve` start-up lines in v0.30.0's wording (capacity lines, the fused-MoE default-config warning), numbers from `moelab.offload.fit` | `tools/make_fixtures.py` | notebook 05 |

None of the illustrative files is a measurement: they exist so the parsers and exercises have the
right shapes to read, and they agree with the lab's models because they were made from them. The
recorded curves are real, but only for the toy task. Replace any of them with your own run — the
notebooks and `python -m moelab` read yours the same way.
