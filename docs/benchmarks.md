# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

ARM-first benchmark policy: maintained publication claims are ARM/AArch64 first and must be
source-backed by the EMEL-owned runtime lane plus an isolated reference lane. Non-ARM, GPU, or
browser-target backend rows are historical inventory or future backend scaffolding unless a
milestone section explicitly names them as maintained evidence.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=3 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch_latest ref=5d2b52d80d9f375a6e81d07e212d047d8ee4f76e`
- `generation_architecture: lfm2`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 roles=system,user tools=none add_generation_prompt=true enable_thinking=false keep_past_thinking=false bos=<|startoftext|>`
- `generation_flash_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 flash_dispatch_calls=174 optimized_flash_dispatch_calls=174 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 815738250.000 ns/op, llama.cpp 628570250.000 ns/op, ratio=1.298x`

- The compare table below keeps additive generation rows for all maintained supported fixtures; this evidence block stays tied to the current maintained publication case.

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=0 packed_q8_0_dispatch_calls=0 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q4_dispatch_calls=656 shared_q4_dispatch_calls=0 optimized_q6_dispatch_calls=81 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical Liquid workload stayed on the approved runtime contract with explicit dense-f32-by-contract stages, native quantized dispatch on the maintained path, and no disallowed fallback or explicit no-claim branch on the supported path.

## Generation Stage Probes

- These are single-run benchmark-local probes. Full-request totals are exact, and the emitted EMEL prompt metadata records the resolved prefill contract, prompt-token count, and planner step size used to interpret the split.
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=956783000 emel_conditioning_ns=37917 emel_prefill_ns=738580208 emel_first_decode_ns=49785667 emel_steady_decode_ns=0 emel_unattributed_ns=168379208 emel_prefill_linear_probe_ns=683206247 emel_prefill_attention_probe_ns=8412574 emel_prefill_misc_probe_ns=46849622 reference_total_ns=652146292 reference_conditioning_ns=47125 reference_prefill_ns=518128083 reference_first_decode_ns=119345334 reference_steady_decode_ns=0 reference_unattributed_ns=14625750 reference_prefill_linear_probe_ns=751843870 reference_prefill_attention_probe_ns=1706125 reference_prefill_misc_probe_ns=16154961`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=1174051875 emel_conditioning_ns=36916 emel_prefill_ns=1202064083 emel_first_decode_ns=21403541 emel_steady_decode_ns=506591916 emel_unattributed_ns=0 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=1422409333 reference_conditioning_ns=46542 reference_prefill_ns=845748916 reference_first_decode_ns=61273250 reference_steady_decode_ns=468782500 reference_unattributed_ns=46558125 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=5379202208 emel_conditioning_ns=37542 emel_prefill_ns=1133095334 emel_first_decode_ns=90226209 emel_steady_decode_ns=5087114495 emel_unattributed_ns=0 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=5493854542 reference_conditioning_ns=140792 reference_prefill_ns=658225792 reference_first_decode_ns=45921250 reference_steady_decode_ns=5095874961 reference_unattributed_ns=0 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=47540722208 emel_conditioning_ns=37208 emel_prefill_ns=760644958 emel_first_decode_ns=21622500 emel_steady_decode_ns=43800367965 emel_unattributed_ns=2958049577 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=54319884292 reference_conditioning_ns=138208 reference_prefill_ns=372361750 reference_first_decode_ns=27112542 reference_steady_decode_ns=49638215669 reference_unattributed_ns=4282056123 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`

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
- Current exact output: `output_dim=17` segments, `output_checksum=4249677247906920305`.
- Benchmark reference note: `proof_status=onnxruntime_cpu thread_contract=intra_op=1 inter_op=1 execution_mode=sequential benchmark_config=iterations:1,runs:15,warmup_iterations:1,warmup_runs:3 actual_providers=CPUExecutionProvider feature_contract=emel_maintained_features onnx_model=build/onnx_ref/diar_streaming_sortformer_4spk-v2.1.onnx onnx_sha256=5df5e883c8dae4e0ecba77739f3db38997c2ae57153de2583d625afb6abb2be0 output_contract=onnx_output_probabilities onnx_repo=ooobo/diar_streaming_sortformer_4spk-v2.1-onnx`
- Current publication caveat: the ONNX row is the CPU single-thread benchmark reference. PyTorch/NeMo remains the independent parity reference, and the ONNX row is only a closeout target after it exact-matches that parity lane.
- Primary steady-state case: `diarization/sortformer/ami_en2002b_mix_headset_137.00_152.04_16khz_mono emel.cpp 1352780166.000 ns/op, onnx.sortformer.v2_1 1920646958.000 ns/op, proof_status=onnx_cpu_single_thread_exact_match`
- Profile case: `diarization/sortformer/profile_ami_en2002b_mix_headset_137.00_152.04_16khz_mono emel.cpp 1413927958.000 ns/op, reference-baseline 0.000 ns/op, proof_status=measurement_only`
- Publication workflow: the deterministic compare artifacts now live under `scripts/bench_diarization_compare.sh`, which writes `raw/emel.jsonl`, `raw/reference.jsonl`, optional `raw/onnx_reference.jsonl`, and `compare_summary.json` using `diarization_compare/v1` / `diarization_compare_summary/v1`.
- PyTorch/NeMo parity reference lane: when supplied with `--pytorch-reference-model`, the workflow emits a distinct `pytorch.nemo.sortformer.v2_1` backend using the official model-card `SortformerEncLabelModel.diarize(..., include_tensor_outputs=True)` path on the maintained WAV fixture. The reproducible environment is synced with `uv` from `tools/bench/diarization_pytorch_reference_requirements.txt` via `scripts/setup_diarization_pytorch_ref_env.sh`. Timing excludes one-time model load.
- ONNX benchmark reference lane: when supplied with `--onnx-reference-model`, the workflow emits a distinct `onnx.sortformer.v2_1` backend using ONNX Runtime CPU single-thread (`intra_op=1`, `inter_op=1`, `ORT_SEQUENTIAL`) and the maintained feature tensor from the same fixture. Generated records include `actual_providers` from ONNX Runtime. Missing ONNX dependencies or artifacts are hard failures, not recorded-baseline fallbacks. ONNX output must be cross-checked against the PyTorch parity lane before it is treated as a correctness target.
- Optimization contract: one-time setup costs are secondary evidence. The primary benchmark claim remains the steady-state maintained pipeline lane on exact hardware.
- Rejected or deferred candidates:
  - Tool-local synthetic contracts/PCM fixtures and reference-lane dependencies remain rejected.
  - Full transformer dense/matmul kernelization remains future kernel-contract work.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2252.958 | 6425.750 | 0.351x |
| `batch/planner_seq` | 2494.584 | 2896.959 | 0.861x |
| `batch/planner_simple` | 2456.000 | 2274.000 | 1.080x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 28642.917 | 41855.000 | 0.684x |
| `gbnf/rule_parser_basic` | 488.625 | 271.125 | 1.802x |
| `gbnf/rule_parser_complex` | 3298.458 | 1503.750 | 2.193x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 815738250.000 | 628570250.000 | 1.298x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 1784670500.000 | 1078314000.000 | 1.655x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 5102306792.000 | 5223846959.000 | 0.977x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 47774724667.000 | 56893067125.000 | 0.840x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 131225292.000 | 247880000.000 | 0.529x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 504214333.000 | 636819042.000 | 0.792x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 3379409541.000 | 2782825667.000 | 1.214x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 33315734042.000 | 36915570125.000 | 0.902x |
| `kernel/aarch64/op_add` | 110.584 | 6455.708 | 0.017x |
| `kernel/aarch64/op_cos` | 7024.917 | 8709.875 | 0.807x |
| `kernel/aarch64/op_div` | 110.667 | 5873.250 | 0.019x |
| `kernel/aarch64/op_dup` | 104.375 | 5788.291 | 0.018x |
| `kernel/aarch64/op_log` | 1888.166 | 10266.417 | 0.184x |
| `kernel/aarch64/op_mul` | 112.375 | 10176.375 | 0.011x |
| `kernel/aarch64/op_mul_mat` | 4994.583 | 12349.834 | 0.404x |
| `kernel/aarch64/op_sin` | 1352.792 | 7073.000 | 0.191x |
| `kernel/aarch64/op_soft_max` | 1053.417 | 6543.542 | 0.161x |
| `kernel/aarch64/op_sqr` | 103.583 | 5934.833 | 0.017x |
| `kernel/aarch64/op_sqrt` | 143.333 | 7189.667 | 0.020x |
| `kernel/aarch64/op_sub` | 110.625 | 7212.792 | 0.015x |
| `kernel/aarch64/op_unary_exp` | 1369.333 | 14917.292 | 0.092x |
| `kernel/aarch64/op_unary_neg` | 118.791 | 5713.875 | 0.021x |
| `kernel/aarch64/op_unary_relu` | 137.792 | 5880.708 | 0.023x |
| `logits/sampler_raw/vocab_128000` | 20031.792 | 32190.542 | 0.622x |
| `logits/sampler_raw/vocab_256000` | 42020.625 | 49786.709 | 0.844x |
| `logits/sampler_raw/vocab_32000` | 5812.125 | 5866.500 | 0.991x |
| `logits/sampler_sml/vocab_128000` | 33250.583 | 19545.125 | 1.701x |
| `logits/sampler_sml/vocab_256000` | 40009.166 | 94785.916 | 0.422x |
| `logits/sampler_sml/vocab_32000` | 5510.667 | 5846.541 | 0.943x |
| `logits/validator_raw/vocab_128000` | 172149.125 | 128547.833 | 1.339x |
| `logits/validator_raw/vocab_256000` | 398246.792 | 443420.458 | 0.898x |
| `logits/validator_raw/vocab_32000` | 28950.250 | 24548.708 | 1.179x |
| `logits/validator_sml/vocab_128000` | 187875.458 | 161780.583 | 1.161x |
| `logits/validator_sml/vocab_256000` | 384760.875 | 471802.625 | 0.816x |
| `logits/validator_sml/vocab_32000` | 40494.458 | 43212.958 | 0.937x |
| `memory/hybrid_full` | 438.375 | 47120.250 | 0.009x |
| `memory/kv_full` | 135.041 | 34738.917 | 0.004x |
| `memory/recurrent_full` | 137.916 | 4462.625 | 0.031x |
| `text/encoders/bpe_long` | 65.625 | 63.708 | 1.030x |
| `text/encoders/bpe_short` | 58.833 | 59.125 | 0.995x |
| `text/encoders/fallback_long` | 2485.250 | 2462.500 | 1.009x |
| `text/encoders/fallback_short` | 65.667 | 65.583 | 1.001x |
| `text/encoders/plamo2_long` | 7791.459 | 9806.625 | 0.795x |
| `text/encoders/plamo2_short` | 212.417 | 209.042 | 1.016x |
| `text/encoders/rwkv_long` | 1711080.625 | 2087617.250 | 0.820x |
| `text/encoders/rwkv_short` | 61692.083 | 190896.834 | 0.323x |
| `text/encoders/spm_long` | 8421898.791 | 7629752.000 | 1.104x |
| `text/encoders/spm_short` | 1388.000 | 1367.250 | 1.015x |
| `text/encoders/ugm_long` | 2611299.166 | 3041259.416 | 0.859x |
| `text/encoders/ugm_short` | 779.625 | 736.625 | 1.058x |
| `text/encoders/wpm_long` | 40910.625 | 42601.292 | 0.960x |
| `text/encoders/wpm_short` | 669.083 | 571.375 | 1.171x |
| `text/jinja/formatter_long` | 62.583 | 433776.125 | 0.000x |
| `text/jinja/formatter_short` | 16.208 | 7724.584 | 0.002x |
| `text/jinja/parser_long` | 67889.875 | 62403.458 | 1.088x |
| `text/jinja/parser_short` | 962.042 | 1090.542 | 0.882x |
| `tokenizer/full_bpe_long` | 13994.792 | 14157.708 | 0.988x |
| `tokenizer/full_bpe_short` | 321.083 | 307.000 | 1.046x |
| `tokenizer/full_plamo2_long` | 14876.292 | 17710.125 | 0.840x |
| `tokenizer/full_plamo2_short` | 2147.541 | 1964.250 | 1.093x |
| `tokenizer/full_rwkv_long` | 2004227.125 | 1651740.750 | 1.213x |
| `tokenizer/full_rwkv_short` | 69940.042 | 82096.875 | 0.852x |
| `tokenizer/full_spm_long` | 6817666.666 | 7975885.250 | 0.855x |
| `tokenizer/full_spm_short` | 1575.500 | 3048.500 | 0.517x |
| `tokenizer/full_ugm_long` | 2848087.125 | 2802872.291 | 1.016x |
| `tokenizer/full_ugm_short` | 2493.500 | 2465.125 | 1.012x |
| `tokenizer/full_wpm_long` | 34054.792 | 89538.541 | 0.380x |
| `tokenizer/full_wpm_short` | 2356.542 | 2273.958 | 1.036x |
| `tokenizer/preprocessor_bpe_long` | 3489.917 | 31531.458 | 0.111x |
| `tokenizer/preprocessor_bpe_short` | 124.708 | 1850.750 | 0.067x |
| `tokenizer/preprocessor_plamo2_long` | 4216.084 | 7092.042 | 0.594x |
| `tokenizer/preprocessor_plamo2_short` | 3169.209 | 5163.416 | 0.614x |
| `tokenizer/preprocessor_rwkv_long` | 4147.625 | 7447.125 | 0.557x |
| `tokenizer/preprocessor_rwkv_short` | 3086.083 | 5580.667 | 0.553x |
| `tokenizer/preprocessor_spm_long` | 4003.625 | 7190.500 | 0.557x |
| `tokenizer/preprocessor_spm_short` | 2736.167 | 7335.625 | 0.373x |
| `tokenizer/preprocessor_ugm_long` | 5260.917 | 7661.583 | 0.687x |
| `tokenizer/preprocessor_ugm_short` | 3107.916 | 5687.541 | 0.546x |
| `tokenizer/preprocessor_wpm_long` | 5484.708 | 7731.333 | 0.709x |
| `tokenizer/preprocessor_wpm_short` | 2541.709 | 5781.583 | 0.440x |
