# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

ARM-first benchmark policy: maintained publication claims are ARM/AArch64 first and must be
source-backed by the EMEL-owned runtime lane plus an isolated reference lane. Non-ARM, GPU, or
browser-target backend rows are historical inventory or future backend scaffolding unless a
milestone section explicitly names them as maintained evidence.

| Benchmark | emel.cpp ns/op | Reference | Reference ns/op | Comparison |
| --- | ---: | --- | ---: | --- |
| `batch/planner_equal` | 2042.333 | llama.cpp | 6278.708 | 0.325x |
| `batch/planner_seq` | 2451.542 | llama.cpp | 2673.333 | 0.917x |
| `batch/planner_simple` | 1213.625 | llama.cpp | 2256.459 | 0.538x |
| `diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono` | 682069791.000 | onnx.sortformer.v2_1 | 543529041.000 | 1.255x (+25.5%), exact_match |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15031.000 | llama.cpp | 25210.250 | 0.596x |
| `gbnf/rule_parser_basic` | 476.875 | llama.cpp | 257.375 | 1.853x |
| `gbnf/rule_parser_complex` | 3282.625 | llama.cpp | 1479.458 | 2.219x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 402703625.000 | llama.cpp | 308659250.000 | 1.305x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 592400791.000 | llama.cpp | 538869584.000 | 1.099x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 2471608417.000 | llama.cpp | 2900662541.000 | 0.852x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 22331522750.000 | llama.cpp | 27152140167.000 | 0.822x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 102294166.000 | llama.cpp | 136357709.000 | 0.750x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 206045666.000 | llama.cpp | 254174875.000 | 0.811x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1189609833.000 | llama.cpp | 1498063125.000 | 0.794x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 15148360708.000 | llama.cpp | 18977629917.000 | 0.798x |
| `kernel/aarch64/op_add` | 118.625 | llama.cpp | 6258.416 | 0.019x |
| `kernel/aarch64/op_cos` | 1723.208 | llama.cpp | 6794.000 | 0.254x |
| `kernel/aarch64/op_div` | 110.958 | llama.cpp | 4979.625 | 0.022x |
| `kernel/aarch64/op_dup` | 103.166 | llama.cpp | 4995.792 | 0.021x |
| `kernel/aarch64/op_log` | 1873.666 | llama.cpp | 6701.333 | 0.280x |
| `kernel/aarch64/op_mul` | 111.834 | llama.cpp | 6293.917 | 0.018x |
| `kernel/aarch64/op_mul_mat` | 2834.208 | llama.cpp | 18014.208 | 0.157x |
| `kernel/aarch64/op_sin` | 1316.167 | llama.cpp | 6108.000 | 0.215x |
| `kernel/aarch64/op_soft_max` | 1073.833 | llama.cpp | 5972.833 | 0.180x |
| `kernel/aarch64/op_sqr` | 103.416 | llama.cpp | 4939.250 | 0.021x |
| `kernel/aarch64/op_sqrt` | 144.750 | llama.cpp | 5108.375 | 0.028x |
| `kernel/aarch64/op_sub` | 118.625 | llama.cpp | 6595.209 | 0.018x |
| `kernel/aarch64/op_unary_exp` | 1432.166 | llama.cpp | 6451.166 | 0.222x |
| `kernel/aarch64/op_unary_neg` | 125.833 | llama.cpp | 5438.875 | 0.023x |
| `kernel/aarch64/op_unary_relu` | 139.709 | llama.cpp | 5285.041 | 0.026x |
| `logits/sampler_raw/vocab_128000` | 19439.792 | llama.cpp | 19685.958 | 0.987x |
| `logits/sampler_raw/vocab_256000` | 37210.000 | llama.cpp | 36543.292 | 1.018x |
| `logits/sampler_raw/vocab_32000` | 4796.041 | llama.cpp | 5427.334 | 0.884x |
| `logits/sampler_sml/vocab_128000` | 19775.917 | llama.cpp | 18578.417 | 1.064x |
| `logits/sampler_sml/vocab_256000` | 33379.292 | llama.cpp | 23628.959 | 1.413x |
| `logits/sampler_sml/vocab_32000` | 4402.542 | llama.cpp | 4797.750 | 0.918x |
| `logits/validator_raw/vocab_128000` | 92666.375 | llama.cpp | 90186.917 | 1.027x |
| `logits/validator_raw/vocab_256000` | 185233.000 | llama.cpp | 175983.250 | 1.053x |
| `logits/validator_raw/vocab_32000` | 23632.000 | llama.cpp | 23353.958 | 1.012x |
| `logits/validator_sml/vocab_128000` | 97867.333 | llama.cpp | 95745.125 | 1.022x |
| `logits/validator_sml/vocab_256000` | 200314.208 | llama.cpp | 193214.375 | 1.037x |
| `logits/validator_sml/vocab_32000` | 24274.209 | llama.cpp | 23702.209 | 1.024x |
| `memory/hybrid_full` | 444.458 | llama.cpp | 34307.000 | 0.013x |
| `memory/kv_full` | 131.833 | llama.cpp | 33199.750 | 0.004x |
| `memory/recurrent_full` | 148.625 | llama.cpp | 4460.167 | 0.033x |
| `text/encoders/bpe_long` | 65.417 | llama.cpp | 66.333 | 0.986x |
| `text/encoders/bpe_short` | 58.416 | llama.cpp | 57.000 | 1.025x |
| `text/encoders/fallback_long` | 2432.500 | llama.cpp | 2504.917 | 0.971x |
| `text/encoders/fallback_short` | 61.541 | llama.cpp | 63.708 | 0.966x |
| `text/encoders/plamo2_long` | 7505.625 | llama.cpp | 7621.416 | 0.985x |
| `text/encoders/plamo2_short` | 197.541 | llama.cpp | 204.083 | 0.968x |
| `text/encoders/rwkv_long` | 817830.666 | llama.cpp | 817051.958 | 1.001x |
| `text/encoders/rwkv_short` | 56197.916 | llama.cpp | 55188.041 | 1.018x |
| `text/encoders/spm_long` | 3571985.041 | llama.cpp | 3530273.833 | 1.012x |
| `text/encoders/spm_short` | 1345.542 | llama.cpp | 1296.458 | 1.038x |
| `text/encoders/ugm_long` | 1365791.166 | llama.cpp | 1345215.917 | 1.015x |
| `text/encoders/ugm_short` | 711.459 | llama.cpp | 728.083 | 0.977x |
| `text/encoders/wpm_long` | 30303.667 | llama.cpp | 30499.167 | 0.994x |
| `text/encoders/wpm_short` | 579.958 | llama.cpp | 546.333 | 1.062x |
| `text/jinja/formatter_long` | 60.541 | llama.cpp | 217732.916 | 0.000x |
| `text/jinja/formatter_short` | 15.583 | llama.cpp | 3876.583 | 0.004x |
| `text/jinja/parser_long` | 65722.792 | llama.cpp | 49943.250 | 1.316x |
| `text/jinja/parser_short` | 982.792 | llama.cpp | 505.125 | 1.946x |
| `tokenizer/full_bpe_long` | 13085.167 | llama.cpp | 13487.333 | 0.970x |
| `tokenizer/full_bpe_short` | 304.333 | llama.cpp | 310.458 | 0.980x |
| `tokenizer/full_plamo2_long` | 12613.167 | llama.cpp | 12076.209 | 1.044x |
| `tokenizer/full_plamo2_short` | 1963.167 | llama.cpp | 1886.250 | 1.041x |
| `tokenizer/full_rwkv_long` | 829591.750 | llama.cpp | 811812.250 | 1.022x |
| `tokenizer/full_rwkv_short` | 55552.208 | llama.cpp | 54091.042 | 1.027x |
| `tokenizer/full_spm_long` | 3602667.042 | llama.cpp | 3540245.417 | 1.018x |
| `tokenizer/full_spm_short` | 1455.375 | llama.cpp | 1453.000 | 1.002x |
| `tokenizer/full_ugm_long` | 1365972.708 | llama.cpp | 1349505.083 | 1.012x |
| `tokenizer/full_ugm_short` | 2427.667 | llama.cpp | 2406.584 | 1.009x |
| `tokenizer/full_wpm_long` | 33084.125 | llama.cpp | 32833.125 | 1.008x |
| `tokenizer/full_wpm_short` | 2178.833 | llama.cpp | 2255.458 | 0.966x |
| `tokenizer/preprocessor_bpe_long` | 3278.208 | llama.cpp | 5181.875 | 0.633x |
| `tokenizer/preprocessor_bpe_short` | 132.417 | llama.cpp | 1737.375 | 0.076x |
| `tokenizer/preprocessor_plamo2_long` | 4133.541 | llama.cpp | 7169.542 | 0.577x |
| `tokenizer/preprocessor_plamo2_short` | 2506.708 | llama.cpp | 5110.208 | 0.491x |
| `tokenizer/preprocessor_rwkv_long` | 4179.666 | llama.cpp | 6981.542 | 0.599x |
| `tokenizer/preprocessor_rwkv_short` | 2475.209 | llama.cpp | 5352.417 | 0.462x |
| `tokenizer/preprocessor_spm_long` | 3996.833 | llama.cpp | 7463.042 | 0.536x |
| `tokenizer/preprocessor_spm_short` | 2610.333 | llama.cpp | 4721.791 | 0.553x |
| `tokenizer/preprocessor_ugm_long` | 4256.458 | llama.cpp | 7343.500 | 0.580x |
| `tokenizer/preprocessor_ugm_short` | 2493.334 | llama.cpp | 4917.500 | 0.507x |
| `tokenizer/preprocessor_wpm_long` | 3994.042 | llama.cpp | 7210.958 | 0.554x |
| `tokenizer/preprocessor_wpm_short` | 2533.833 | llama.cpp | 5622.666 | 0.451x |

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
