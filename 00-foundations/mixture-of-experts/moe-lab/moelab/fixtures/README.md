# fixtures — what the notebooks read when there is no GPU

The lab has two kinds of bundled data. The data has a label wherever the lab prints it:

| File | Kind | Made by | Read by |
|---|---|---|---|
| `tinymoe_curves.json` | **recorded**: the real output of `moelab.tinymoe` on a CPU (torch 2.14), for 3 load-balance modes × seeds 0–2 | `python -m moelab.tinymoe --record` | notebook 01 without torch |
| `router_traces_olmoe.json` | **sample output in the documented format (illustrative)**: chat responses with the shape that `vllm serve --enable-return-routed-experts` returns (`choices[0].routed_experts` = base64 `.npy`, `(tokens − 1, 16 layers, top-8)`). The script samples the routing from an invented skewed model. | `tools/make_fixtures.py` | notebooks 02 and 04, `python -m moelab trace` |
| `bench_serve_olmoe_2xT4_{tp,tp_ep,dp_ep}_c{1,16}.txt` | **illustrative**: the `Serving Benchmark Result` block of `vllm bench serve` in vLLM v0.30.0. The numbers come from the simulation in `moelab.ep`. | `tools/make_fixtures.py` | notebook 04, `python -m moelab bench-parse` |
| `vllm_startup_olmoe_t4_offload.log`, `vllm_startup_olmoe_l4.log` | **illustrative**: the start-up lines of `vllm serve` with the same words that v0.30.0 prints (capacity lines, the fused-MoE default-config warning). The numbers come from `moelab.offload.fit`. | `tools/make_fixtures.py` | notebook 05 |

None of the illustrative files is a measurement. They exist to give the parsers and the exercises the
correct shapes to read. The illustrative files agree with the models of the lab because the lab made the files from
these models. The recorded curves are real, but only for the toy task. Replace any of these files with your
own run. The notebooks and `python -m moelab` read your files in the same way.
