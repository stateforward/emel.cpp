# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=3 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch_latest ref=660b1b4bdc6fedc18e8c3d87a945ffb51f91c547`
- `generation_architecture: lfm2`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 roles=system,user tools=none add_generation_prompt=true enable_thinking=false keep_past_thinking=false bos=<|startoftext|>`
- `generation_flash_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 flash_dispatch_calls=174 optimized_flash_dispatch_calls=174 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 402703625.000 ns/op, llama.cpp 308659250.000 ns/op, ratio=1.305x`

- The compare table below keeps additive generation rows for all maintained supported fixtures; this evidence block stays tied to the current maintained publication case.

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=0 packed_q8_0_dispatch_calls=0 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q4_dispatch_calls=656 shared_q4_dispatch_calls=0 optimized_q6_dispatch_calls=81 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical Liquid workload stayed on the approved runtime contract with explicit dense-f32-by-contract stages, native quantized dispatch on the maintained path, and no disallowed fallback or explicit no-claim branch on the supported path.

## Generation Stage Probes

- These are single-run benchmark-local probes. Full-request totals are exact, and the emitted EMEL prompt metadata records the resolved prefill contract, prompt-token count, and planner step size used to interpret the split.
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel_prefill_contract=actor_public_generate emel_prompt_tokens=29 emel_prefill_step_size=0 emel_total_ns=401208666 emel_conditioning_ns=44334 emel_prefill_ns=0 emel_first_decode_ns=0 emel_steady_decode_ns=0 emel_unattributed_ns=401164332 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=362143250 reference_conditioning_ns=51041 reference_prefill_ns=283958875 reference_first_decode_ns=25933958 reference_steady_decode_ns=0 reference_unattributed_ns=52199376 reference_prefill_linear_probe_ns=274011293 reference_prefill_attention_probe_ns=4593291 reference_prefill_misc_probe_ns=9820713`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10 emel_prefill_contract=actor_public_generate emel_prompt_tokens=29 emel_prefill_step_size=0 emel_total_ns=585425459 emel_conditioning_ns=34417 emel_prefill_ns=0 emel_first_decode_ns=0 emel_steady_decode_ns=0 emel_unattributed_ns=585391042 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=549035708 reference_conditioning_ns=45625 reference_prefill_ns=286408292 reference_first_decode_ns=26096250 reference_steady_decode_ns=234375917 reference_unattributed_ns=2109624 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100 emel_prefill_contract=actor_public_generate emel_prompt_tokens=29 emel_prefill_step_size=0 emel_total_ns=2510461208 emel_conditioning_ns=34459 emel_prefill_ns=0 emel_first_decode_ns=0 emel_steady_decode_ns=0 emel_unattributed_ns=2510426749 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=2946222875 reference_conditioning_ns=50834 reference_prefill_ns=282047500 reference_first_decode_ns=26033708 reference_steady_decode_ns=2568681366 reference_unattributed_ns=69409467 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000 emel_prefill_contract=actor_public_generate emel_prompt_tokens=29 emel_prefill_step_size=0 emel_total_ns=22151000417 emel_conditioning_ns=39250 emel_prefill_ns=0 emel_first_decode_ns=0 emel_steady_decode_ns=0 emel_unattributed_ns=22150961167 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=27833934458 reference_conditioning_ns=48250 reference_prefill_ns=283769084 reference_first_decode_ns=26542292 reference_steady_decode_ns=26975231173 reference_unattributed_ns=548343659 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`

## Preserved ARM Flash Baseline

- Preserved baseline artifact: `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`
- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- Note: this preserved ARM flash baseline remains tied to the archived Llama canonical slice and is not directly compared against the current maintained publication because the benchmark case identity changed explicitly.

## Sortformer Diarization Baseline

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- Maintained benchmark suite: `diarization_sortformer`.
- Supported maintained model: `tests/models/diar_streaming_sortformer_4spk-v2.1.gguf` (`diar_streaming_sortformer_4spk_v2_1_gguf`).
- Canonical proof fixture: `tests/fixtures/diarization/ami_en2002b_mix_headset_137.00_152.04_16khz_mono.wav` (`ami_en2002b_mix_headset_137.00_152.04_16khz_mono`).
- Maintained fixture provenance: `source=ami ihm/test path=EN2002b.Mix-Headset.wav window=137.00-152.04s proof_status=maintained_loader_real_audio`
- Input contract: real `15.04` second mono `16 kHz` audio, maintained GGUF loader, maintained request frontend (`1504 x 128` log-mel features), maintained pre-encode stack (`188 x 512` encoder frames), then the maintained executor/probability/segment pipeline.
- Self-recorded regression snapshot: `output_dim=17` segments, `output_checksum=4249677247906920305`.
- Recorded lane note: `source=ami ihm/test path=EN2002b.Mix-Headset.wav window=137.00-152.04s proof_status=recorded_maintained_baseline`
- Correctness caveat: the recorded lane was produced by EMEL and is not an independent parity oracle.
- Primary steady-state case: `diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono emel.cpp 1155218681.458 ns/op, reference-baseline 0.416 ns/op, proof_status=baseline_matched`
- Profile case: `diarization/sortformer/profile_ami_en2002b_mix_headset_137.00_152.04_16khz_mono emel.cpp 1192734833.000 ns/op, reference-baseline 0.000 ns/op, proof_status=measurement_only`
- Publication workflow: the deterministic compare artifacts now live under `scripts/bench_diarization_compare.sh`, which writes `raw/emel.jsonl`, `raw/reference.jsonl`, optional `raw/onnx_reference.jsonl`, and `compare_summary.json` using `diarization_compare/v1` / `diarization_compare_summary/v1`.
- PyTorch/NeMo parity reference lane: when supplied with `--pytorch-reference-model`, the workflow emits a distinct `pytorch.nemo.sortformer.v2_1` backend using the official model-card `SortformerEncLabelModel.diarize(..., include_tensor_outputs=True)` path on the maintained WAV fixture. The reproducible environment is synced with `uv` from `tools/bench/diarization_pytorch_reference_requirements.txt` via `scripts/setup_diarization_pytorch_ref_env.sh`. Timing excludes one-time model load.
- ONNX benchmark reference lane: when supplied with `--onnx-reference-model`, the workflow emits a distinct `onnx.sortformer.v2_1` backend using ONNX Runtime CPU single-thread (`intra_op=1`, `inter_op=1`, `ORT_SEQUENTIAL`) and the maintained feature tensor from the same fixture. Generated records include `actual_providers` from ONNX Runtime. Missing ONNX dependencies or artifacts are hard failures, not recorded-baseline fallbacks. ONNX output must be cross-checked against the PyTorch parity lane before it is treated as a correctness target.
- Optimization contract: one-time setup costs are secondary evidence. The primary benchmark claim remains the steady-state maintained pipeline lane on exact hardware.
- Rejected or deferred candidates:
  - Tool-local synthetic contracts/PCM fixtures and reference-lane dependencies remain rejected.
  - Full transformer dense/matmul kernelization remains future kernel-contract work.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2042.333 | 6278.708 | 0.325x |
| `batch/planner_seq` | 2451.542 | 2673.333 | 0.917x |
| `batch/planner_simple` | 1213.625 | 2256.459 | 0.538x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15031.000 | 25210.250 | 0.596x |
| `gbnf/rule_parser_basic` | 476.875 | 257.375 | 1.853x |
| `gbnf/rule_parser_complex` | 3282.625 | 1479.458 | 2.219x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 402703625.000 | 308659250.000 | 1.305x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 592400791.000 | 538869584.000 | 1.099x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 2471608417.000 | 2900662541.000 | 0.852x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 22331522750.000 | 27152140167.000 | 0.822x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 102294166.000 | 136357709.000 | 0.750x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 206045666.000 | 254174875.000 | 0.811x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1189609833.000 | 1498063125.000 | 0.794x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 15148360708.000 | 18977629917.000 | 0.798x |
| `kernel/aarch64/op_add` | 118.625 | 6258.416 | 0.019x |
| `kernel/aarch64/op_cos` | 1723.208 | 6794.000 | 0.254x |
| `kernel/aarch64/op_div` | 110.958 | 4979.625 | 0.022x |
| `kernel/aarch64/op_dup` | 103.166 | 4995.792 | 0.021x |
| `kernel/aarch64/op_log` | 1873.666 | 6701.333 | 0.280x |
| `kernel/aarch64/op_mul` | 111.834 | 6293.917 | 0.018x |
| `kernel/aarch64/op_mul_mat` | 2834.208 | 18014.208 | 0.157x |
| `kernel/aarch64/op_sin` | 1316.167 | 6108.000 | 0.215x |
| `kernel/aarch64/op_soft_max` | 1073.833 | 5972.833 | 0.180x |
| `kernel/aarch64/op_sqr` | 103.416 | 4939.250 | 0.021x |
| `kernel/aarch64/op_sqrt` | 144.750 | 5108.375 | 0.028x |
| `kernel/aarch64/op_sub` | 118.625 | 6595.209 | 0.018x |
| `kernel/aarch64/op_unary_exp` | 1432.166 | 6451.166 | 0.222x |
| `kernel/aarch64/op_unary_neg` | 125.833 | 5438.875 | 0.023x |
| `kernel/aarch64/op_unary_relu` | 139.709 | 5285.041 | 0.026x |
| `logits/sampler_raw/vocab_128000` | 19439.792 | 19685.958 | 0.987x |
| `logits/sampler_raw/vocab_256000` | 37210.000 | 36543.292 | 1.018x |
| `logits/sampler_raw/vocab_32000` | 4796.041 | 5427.334 | 0.884x |
| `logits/sampler_sml/vocab_128000` | 19775.917 | 18578.417 | 1.064x |
| `logits/sampler_sml/vocab_256000` | 33379.292 | 23628.959 | 1.413x |
| `logits/sampler_sml/vocab_32000` | 4402.542 | 4797.750 | 0.918x |
| `logits/validator_raw/vocab_128000` | 92666.375 | 90186.917 | 1.027x |
| `logits/validator_raw/vocab_256000` | 185233.000 | 175983.250 | 1.053x |
| `logits/validator_raw/vocab_32000` | 23632.000 | 23353.958 | 1.012x |
| `logits/validator_sml/vocab_128000` | 97867.333 | 95745.125 | 1.022x |
| `logits/validator_sml/vocab_256000` | 200314.208 | 193214.375 | 1.037x |
| `logits/validator_sml/vocab_32000` | 24274.209 | 23702.209 | 1.024x |
| `memory/hybrid_full` | 444.458 | 34307.000 | 0.013x |
| `memory/kv_full` | 131.833 | 33199.750 | 0.004x |
| `memory/recurrent_full` | 148.625 | 4460.167 | 0.033x |
| `text/encoders/bpe_long` | 65.417 | 66.333 | 0.986x |
| `text/encoders/bpe_short` | 58.416 | 57.000 | 1.025x |
| `text/encoders/fallback_long` | 2432.500 | 2504.917 | 0.971x |
| `text/encoders/fallback_short` | 61.541 | 63.708 | 0.966x |
| `text/encoders/plamo2_long` | 7505.625 | 7621.416 | 0.985x |
| `text/encoders/plamo2_short` | 197.541 | 204.083 | 0.968x |
| `text/encoders/rwkv_long` | 817830.666 | 817051.958 | 1.001x |
| `text/encoders/rwkv_short` | 56197.916 | 55188.041 | 1.018x |
| `text/encoders/spm_long` | 3571985.041 | 3530273.833 | 1.012x |
| `text/encoders/spm_short` | 1345.542 | 1296.458 | 1.038x |
| `text/encoders/ugm_long` | 1365791.166 | 1345215.917 | 1.015x |
| `text/encoders/ugm_short` | 711.459 | 728.083 | 0.977x |
| `text/encoders/wpm_long` | 30303.667 | 30499.167 | 0.994x |
| `text/encoders/wpm_short` | 579.958 | 546.333 | 1.062x |
| `text/jinja/formatter_long` | 60.541 | 217732.916 | 0.000x |
| `text/jinja/formatter_short` | 15.583 | 3876.583 | 0.004x |
| `text/jinja/parser_long` | 65722.792 | 49943.250 | 1.316x |
| `text/jinja/parser_short` | 982.792 | 505.125 | 1.946x |
| `tokenizer/full_bpe_long` | 13085.167 | 13487.333 | 0.970x |
| `tokenizer/full_bpe_short` | 304.333 | 310.458 | 0.980x |
| `tokenizer/full_plamo2_long` | 12613.167 | 12076.209 | 1.044x |
| `tokenizer/full_plamo2_short` | 1963.167 | 1886.250 | 1.041x |
| `tokenizer/full_rwkv_long` | 829591.750 | 811812.250 | 1.022x |
| `tokenizer/full_rwkv_short` | 55552.208 | 54091.042 | 1.027x |
| `tokenizer/full_spm_long` | 3602667.042 | 3540245.417 | 1.018x |
| `tokenizer/full_spm_short` | 1455.375 | 1453.000 | 1.002x |
| `tokenizer/full_ugm_long` | 1365972.708 | 1349505.083 | 1.012x |
| `tokenizer/full_ugm_short` | 2427.667 | 2406.584 | 1.009x |
| `tokenizer/full_wpm_long` | 33084.125 | 32833.125 | 1.008x |
| `tokenizer/full_wpm_short` | 2178.833 | 2255.458 | 0.966x |
| `tokenizer/preprocessor_bpe_long` | 3278.208 | 5181.875 | 0.633x |
| `tokenizer/preprocessor_bpe_short` | 132.417 | 1737.375 | 0.076x |
| `tokenizer/preprocessor_plamo2_long` | 4133.541 | 7169.542 | 0.577x |
| `tokenizer/preprocessor_plamo2_short` | 2506.708 | 5110.208 | 0.491x |
| `tokenizer/preprocessor_rwkv_long` | 4179.666 | 6981.542 | 0.599x |
| `tokenizer/preprocessor_rwkv_short` | 2475.209 | 5352.417 | 0.462x |
| `tokenizer/preprocessor_spm_long` | 3996.833 | 7463.042 | 0.536x |
| `tokenizer/preprocessor_spm_short` | 2610.333 | 4721.791 | 0.553x |
| `tokenizer/preprocessor_ugm_long` | 4256.458 | 7343.500 | 0.580x |
| `tokenizer/preprocessor_ugm_short` | 2493.334 | 4917.500 | 0.507x |
| `tokenizer/preprocessor_wpm_long` | 3994.042 | 7210.958 | 0.554x |
| `tokenizer/preprocessor_wpm_short` | 2533.833 | 5622.666 | 0.451x |
