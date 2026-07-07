# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

CPU-first benchmark policy: maintained publication claims are CPU-first and must be source-backed
by the EMEL-owned runtime lane plus an isolated reference lane. GPU or browser-target backend rows
are historical inventory or future backend scaffolding unless a milestone section explicitly names
them as maintained evidence.

| Benchmark | emel.cpp ns/op | emel.cpp tokens/s | Reference | Reference ns/op | Reference tokens/s | Comparison |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `batch/planner_equal` | 1356.670 |  | llama.cpp | 7694.580 |  | 0.176x |
| `batch/planner_seq` | 1586.670 |  | llama.cpp | 3282.500 |  | 0.483x |
| `batch/planner_simple` | 542.090 |  | llama.cpp | 2811.250 |  | 0.193x |
| `decode_wavefront/ggml_batch1` | 4700.420 |  | llama.cpp | 5067.500 |  | 0.928x |
| `decode_wavefront/ggml_batch4` | 6955.000 |  | llama.cpp | 7505.420 |  | 0.927x |
| `decode_wavefront/ggml_batch8` | 9588.750 |  | llama.cpp | 16819.170 |  | 0.570x |
| `diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono` | 682069791.000 |  | onnx.sortformer.v2_1 | 543529041.000 |  | 1.255x (+25.5%), exact_match |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 14246.250 |  | llama.cpp | 18857.500 |  | 0.755x |
| `gbnf/rule_parser_basic` | 465.410 |  | llama.cpp | 401.660 |  | 1.159x |
| `gbnf/rule_parser_complex` | 3256.250 |  | llama.cpp | 2029.580 |  | 1.604x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 383847833.000 |  | llama.cpp | 53832541.000 |  | 7.130x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 475178250.000 |  | llama.cpp | 113868791.000 |  | 4.173x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 1390099791.000 |  | llama.cpp | 801318292.000 |  | 1.735x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 12673042083.000 |  | llama.cpp | 8252899167.000 |  | 1.536x |
| `generation/preloaded_request/lfm2_5_230m_q8_0_prompt_hello_max_tokens_1` | 27806125.000 |  | llama.cpp | 10074333.000 |  | 2.760x |
| `generation/preloaded_request/lfm2_5_230m_q8_0_prompt_hello_max_tokens_100` | 300885792.000 |  | llama.cpp | 294963750.000 |  | 1.020x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 102294166.000 |  | llama.cpp | 136357709.000 |  | 0.750x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 206045666.000 |  | llama.cpp | 254174875.000 |  | 0.811x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1189609833.000 |  | llama.cpp | 1498063125.000 |  | 0.794x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 15148360708.000 |  | llama.cpp | 18977629917.000 |  | 0.798x |
| `kernel/aarch64/op_add` | 106.670 |  | llama.cpp | 5229.580 |  | 0.020x |
| `kernel/aarch64/op_cos` | 1607.500 |  | llama.cpp | 6027.090 |  | 0.267x |
| `kernel/aarch64/op_div` | 133.750 |  | llama.cpp | 4705.410 |  | 0.028x |
| `kernel/aarch64/op_dup` | 98.750 |  | llama.cpp | 4643.340 |  | 0.021x |
| `kernel/aarch64/op_log` | 1651.250 |  | llama.cpp | 6709.580 |  | 0.246x |
| `kernel/aarch64/op_mul` | 175.000 |  | llama.cpp | 4704.170 |  | 0.037x |
| `kernel/aarch64/op_mul_mat` | 2651.670 |  | llama.cpp | 10277.920 |  | 0.258x |
| `kernel/aarch64/op_sin` | 1246.670 |  | llama.cpp | 6465.420 |  | 0.193x |
| `kernel/aarch64/op_soft_max` | 1662.910 |  | llama.cpp | 5377.500 |  | 0.309x |
| `kernel/aarch64/op_sqr` | 59.590 |  | llama.cpp | 4720.420 |  | 0.013x |
| `kernel/aarch64/op_sqrt` | 138.330 |  | llama.cpp | 5143.750 |  | 0.027x |
| `kernel/aarch64/op_sub` | 104.580 |  | llama.cpp | 4989.580 |  | 0.021x |
| `kernel/aarch64/op_unary_exp` | 1485.410 |  | llama.cpp | 6478.340 |  | 0.229x |
| `kernel/aarch64/op_unary_neg` | 150.420 |  | llama.cpp | 4185.420 |  | 0.036x |
| `kernel/aarch64/op_unary_relu` | 137.500 |  | llama.cpp | 6232.090 |  | 0.022x |
| `logits/sampler_raw/vocab_128000` | 17480.410 |  | llama.cpp | 18770.410 |  | 0.931x |
| `logits/sampler_raw/vocab_256000` | 39341.660 |  | llama.cpp | 41833.330 |  | 0.940x |
| `logits/sampler_raw/vocab_32000` | 4567.080 |  | llama.cpp | 5489.580 |  | 0.832x |
| `logits/sampler_sml/vocab_128000` | 19444.170 |  | llama.cpp | 17710.830 |  | 1.098x |
| `logits/sampler_sml/vocab_256000` | 36249.170 |  | llama.cpp | 33192.920 |  | 1.092x |
| `logits/sampler_sml/vocab_32000` | 5593.330 |  | llama.cpp | 3810.830 |  | 1.468x |
| `logits/validator_raw/vocab_128000` | 93867.080 |  | llama.cpp | 100561.660 |  | 0.933x |
| `logits/validator_raw/vocab_256000` | 185163.750 |  | llama.cpp | 194978.330 |  | 0.950x |
| `logits/validator_raw/vocab_32000` | 24218.330 |  | llama.cpp | 26358.330 |  | 0.919x |
| `logits/validator_sml/vocab_128000` | 102629.170 |  | llama.cpp | 108406.660 |  | 0.947x |
| `logits/validator_sml/vocab_256000` | 197461.250 |  | llama.cpp | 218344.580 |  | 0.904x |
| `logits/validator_sml/vocab_32000` | 24590.000 |  | llama.cpp | 26821.250 |  | 0.917x |
| `memory/hybrid_full` | 420.420 |  | llama.cpp | 39208.750 |  | 0.011x |
| `memory/kv_full` | 120.830 |  | llama.cpp | 38020.000 |  | 0.003x |
| `memory/recurrent_full` | 131.250 |  | llama.cpp | 5452.920 |  | 0.024x |
| `parallel_matmul/ggml_gemm8_f32` | 157395.840 |  | llama.cpp | 276495.830 |  | 0.569x |
| `parallel_matmul/ggml_gemv_f32` | 256679.170 |  | llama.cpp | 37782.500 |  | 6.794x |
| `parallel_matmul/ggml_gemv_q4_k` | 18047.920 |  | llama.cpp | 14390.420 |  | 1.254x |
| `parallel_matmul/ggml_gemv_q6_k` | 24174.580 |  | llama.cpp | 25298.750 |  | 0.956x |
| `parallel_matmul/ggml_gemv_q8_0` | 19610.410 |  | llama.cpp | 16083.750 |  | 1.219x |
| `text/encoders/bpe_long` | 76.250 |  | llama.cpp | 71.670 |  | 1.064x |
| `text/encoders/bpe_short` | 58.750 |  | llama.cpp | 57.080 |  | 1.029x |
| `text/encoders/fallback_long` | 2313.340 |  | llama.cpp | 2577.080 |  | 0.898x |
| `text/encoders/fallback_short` | 61.250 |  | llama.cpp | 63.330 |  | 0.967x |
| `text/encoders/plamo2_long` | 7390.830 |  | llama.cpp | 8116.250 |  | 0.911x |
| `text/encoders/plamo2_short` | 212.500 |  | llama.cpp | 249.580 |  | 0.851x |
| `text/encoders/rwkv_long` | 877482.910 |  | llama.cpp | 891017.090 |  | 0.985x |
| `text/encoders/rwkv_short` | 59084.170 |  | llama.cpp | 59851.250 |  | 0.987x |
| `text/encoders/spm_long` | 3835491.660 |  | llama.cpp | 3783557.500 |  | 1.014x |
| `text/encoders/spm_short` | 1491.670 |  | llama.cpp | 1491.250 |  | 1.000x |
| `text/encoders/ugm_long` | 1472599.170 |  | llama.cpp | 1476967.910 |  | 0.997x |
| `text/encoders/ugm_short` | 853.750 |  | llama.cpp | 790.420 |  | 1.080x |
| `text/encoders/wpm_long` | 34315.830 |  | llama.cpp | 34540.000 |  | 0.994x |
| `text/encoders/wpm_short` | 664.160 |  | llama.cpp | 697.500 |  | 0.952x |
| `text/jinja/formatter_long` | 60.830 |  | llama.cpp | 296972.500 |  | 0.000x |
| `text/jinja/formatter_short` | 17.500 |  | llama.cpp | 5235.000 |  | 0.003x |
| `text/jinja/parser_long` | 65622.920 |  | llama.cpp | 56120.000 |  | 1.169x |
| `text/jinja/parser_short` | 908.330 |  | llama.cpp | 641.660 |  | 1.416x |
| `tokenizer/full_bpe_long` | 14560.410 |  | llama.cpp | 14782.910 |  | 0.985x |
| `tokenizer/full_bpe_short` | 289.170 |  | llama.cpp | 303.330 |  | 0.953x |
| `tokenizer/full_plamo2_long` | 13682.920 |  | llama.cpp | 13488.750 |  | 1.014x |
| `tokenizer/full_plamo2_short` | 2175.000 |  | llama.cpp | 2201.660 |  | 0.988x |
| `tokenizer/full_rwkv_long` | 905961.250 |  | llama.cpp | 908680.410 |  | 0.997x |
| `tokenizer/full_rwkv_short` | 60504.160 |  | llama.cpp | 60279.590 |  | 1.004x |
| `tokenizer/full_spm_long` | 3733154.170 |  | llama.cpp | 3828705.420 |  | 0.975x |
| `tokenizer/full_spm_short` | 1677.920 |  | llama.cpp | 1687.500 |  | 0.994x |
| `tokenizer/full_ugm_long` | 1490826.250 |  | llama.cpp | 1462667.920 |  | 1.019x |
| `tokenizer/full_ugm_short` | 2710.000 |  | llama.cpp | 2672.920 |  | 1.014x |
| `tokenizer/full_wpm_long` | 36643.330 |  | llama.cpp | 36127.500 |  | 1.014x |
| `tokenizer/full_wpm_short` | 2709.590 |  | llama.cpp | 2585.420 |  | 1.048x |
| `tokenizer/preprocessor_bpe_long` | 3705.410 |  | llama.cpp | 5596.660 |  | 0.662x |
| `tokenizer/preprocessor_bpe_short` | 122.500 |  | llama.cpp | 1790.410 |  | 0.068x |
| `tokenizer/preprocessor_plamo2_long` | 4527.500 |  | llama.cpp | 7969.160 |  | 0.568x |
| `tokenizer/preprocessor_plamo2_short` | 2554.580 |  | llama.cpp | 6100.000 |  | 0.419x |
| `tokenizer/preprocessor_rwkv_long` | 4417.500 |  | llama.cpp | 8057.920 |  | 0.548x |
| `tokenizer/preprocessor_rwkv_short` | 2773.750 |  | llama.cpp | 6091.660 |  | 0.455x |
| `tokenizer/preprocessor_spm_long` | 4482.920 |  | llama.cpp | 7759.170 |  | 0.578x |
| `tokenizer/preprocessor_spm_short` | 2660.840 |  | llama.cpp | 6047.500 |  | 0.440x |
| `tokenizer/preprocessor_ugm_long` | 4717.500 |  | llama.cpp | 7812.500 |  | 0.604x |
| `tokenizer/preprocessor_ugm_short` | 2716.670 |  | llama.cpp | 5710.000 |  | 0.476x |
| `tokenizer/preprocessor_wpm_long` | 4505.000 |  | llama.cpp | 7905.000 |  | 0.570x |
| `tokenizer/preprocessor_wpm_short` | 2766.670 |  | llama.cpp | 5929.170 |  | 0.467x |

## Generation Threading

- Generated generation rows in this snapshot use `applies_to=generated_generation_rows emel_thread_count=8 reference_thread_count=8 emel_thread_contract=emel_parallel_matmul_lanes=8 reference_thread_contract=llama.cpp_n_threads=8,n_threads_batch=8`.
- Rows listed under Preserved Rows keep their previous snapshot provenance and are not covered by this generation threading line.

## Preserved Rows

Some published rows were intentionally preserved from earlier snapshots because this refresh host could not regenerate that lane:

- `diarization_sortformer` compare rows: `source=previous_onnx_snapshot`, `reason=generic_compare_refresh_emitted_recorded_baseline_without_reference_timing`.
- `generation` fixture `Qwen3-0.6B-Q8_0.gguf` rows: `source=previous_snapshot`, `reason=fixture_absent_on_2026_07_06_threaded_cpu_refresh`.

## Sortformer Diarization Baseline

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- Maintained benchmark suite: `diarization_sortformer`.
- Supported maintained model: `tests/models/diar_streaming_sortformer_4spk-v2.1.gguf` (`diar_streaming_sortformer_4spk_v2_1_gguf`).
- Canonical proof fixture: `tests/fixtures/diarization/ami_en2002b_mix_headset_137.00_152.04_16khz_mono.wav` (`ami_en2002b_mix_headset_137.00_152.04_16khz_mono`).
- Maintained fixture provenance: `source=ami ihm/test path=EN2002b.Mix-Headset.wav window=137.00-152.04s proof_status=maintained_loader_real_audio`
- Input contract: real `15.04` second mono `16 kHz` audio, maintained GGUF loader, maintained request frontend (`1504 x 128` log-mel features), maintained pre-encode stack (`188 x 512` encoder frames), then the maintained executor/probability/segment pipeline.
- Current exact output: `output_dim=17` segments, `output_checksum=4249677247906920305`.
- Benchmark reference note: `proof_status=onnxruntime_cpu thread_contract=intra_op=1 inter_op=1 execution_mode=sequential benchmark_config=iterations:1,runs:15,warmup_iterations:1,warmup_runs:3 actual_providers=CPUExecutionProvider feature_contract=emel_maintained_features onnx_model=build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx onnx_sha256=5df5e883c8dae4e0ecba77739f3db38997c2ae57153de2583d625afb6abb2be0 output_contract=onnx_output_probabilities onnx_repo=ooobo/diar_streaming_sortformer_4spk-v2.1-onnx`
- Current publication caveat: the ONNX row is the CPU single-thread benchmark reference. PyTorch/NeMo remains the independent parity reference, and the ONNX row is only a closeout target after it exact-matches that parity lane.
- Table row: the primary steady-state case is included in the benchmark table above with its real reference timing and proof status.
- Publication workflow: the deterministic compare artifacts now live under `scripts/bench_diarization_compare.sh`, which writes `raw/emel.jsonl`, `raw/reference.jsonl`, optional `raw/onnx_reference.jsonl`, and `compare_summary.json` using `diarization_compare/v1` / `diarization_compare_summary/v1`.
- PyTorch/NeMo parity reference lane: when supplied with `--pytorch-reference-model`, the workflow emits a distinct `pytorch.nemo.sortformer.v2_1` backend using the official model-card `SortformerEncLabelModel.diarize(..., include_tensor_outputs=True)` path on the maintained WAV fixture. The reproducible environment is synced with `uv` from `tools/bench/diarization_pytorch_reference_requirements.txt` via `scripts/setup_diarization_pytorch_ref_env.sh`. Timing excludes one-time model load.
- ONNX benchmark reference lane: when supplied with `--onnx-reference-model`, the workflow emits a distinct `onnx.sortformer.v2_1` backend using ONNX Runtime CPU single-thread (`intra_op=1`, `inter_op=1`, `ORT_SEQUENTIAL`) and the maintained feature tensor from the same fixture. Generated records include `actual_providers` from ONNX Runtime. Missing ONNX dependencies or artifacts are hard failures, not recorded-baseline fallbacks. ONNX output must be cross-checked against the PyTorch parity lane before it is treated as a correctness target.
- Optimization contract: one-time setup costs are secondary evidence. The primary benchmark claim remains the steady-state maintained pipeline lane on exact hardware.
- Rejected or deferred candidates:
  - Tool-local synthetic contracts/PCM fixtures and reference-lane dependencies remain rejected.
  - Full transformer dense/matmul kernelization remains future kernel-contract work.
