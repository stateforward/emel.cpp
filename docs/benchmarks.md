# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=3 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch_latest ref=223373742bc1bd48e37b22192d1302f54d6f14bc`
- `generation_architecture: lfm2`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 roles=system,user tools=none add_generation_prompt=true enable_thinking=false keep_past_thinking=false bos=<|startoftext|>`
- `generation_flash_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 flash_dispatch_calls=174 optimized_flash_dispatch_calls=174 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 408512500.000 ns/op, llama.cpp 307468375.000 ns/op, ratio=1.329x`

- The compare table below keeps additive generation rows for all maintained supported fixtures; this evidence block stays tied to the current maintained publication case.

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=0 packed_q8_0_dispatch_calls=0 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q4_dispatch_calls=656 shared_q4_dispatch_calls=0 optimized_q6_dispatch_calls=81 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical Liquid workload stayed on the approved runtime contract with explicit dense-f32-by-contract stages, native quantized dispatch on the maintained path, and no disallowed fallback or explicit no-claim branch on the supported path.

## Generation Stage Probes

- These are single-run benchmark-local probes. Full-request totals are exact, and the emitted EMEL prompt metadata records the resolved prefill contract, prompt-token count, and planner step size used to interpret the split.
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=399902000 emel_conditioning_ns=45708 emel_prefill_ns=399719250 emel_first_decode_ns=20421625 emel_steady_decode_ns=0 emel_unattributed_ns=0 emel_prefill_linear_probe_ns=375726243 emel_prefill_attention_probe_ns=1558795 emel_prefill_misc_probe_ns=22340162 reference_total_ns=328129042 reference_conditioning_ns=56208 reference_prefill_ns=274797750 reference_first_decode_ns=24240875 reference_steady_decode_ns=0 reference_unattributed_ns=29034209 reference_prefill_linear_probe_ns=271048330 reference_prefill_attention_probe_ns=1663790 reference_prefill_misc_probe_ns=9047584`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=593653292 emel_conditioning_ns=39375 emel_prefill_ns=397155167 emel_first_decode_ns=20787875 emel_steady_decode_ns=188504209 emel_unattributed_ns=0 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=524524625 reference_conditioning_ns=56500 reference_prefill_ns=277471959 reference_first_decode_ns=24427333 reference_steady_decode_ns=224066749 reference_unattributed_ns=0 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=2472134625 emel_conditioning_ns=37083 emel_prefill_ns=403671375 emel_first_decode_ns=20894666 emel_steady_decode_ns=2069237001 emel_unattributed_ns=0 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=2744493166 reference_conditioning_ns=44083 reference_prefill_ns=276621791 reference_first_decode_ns=24145917 reference_steady_decode_ns=2429306455 reference_unattributed_ns=14374920 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`
- `generation_stage_probe: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000 emel_prefill_contract=flash_preselected_chunk8_q8_k emel_prompt_tokens=29 emel_prefill_step_size=8 emel_total_ns=22737997083 emel_conditioning_ns=35833 emel_prefill_ns=395689334 emel_first_decode_ns=20213042 emel_steady_decode_ns=21873018554 emel_unattributed_ns=449040320 emel_prefill_linear_probe_ns=0 emel_prefill_attention_probe_ns=0 emel_prefill_misc_probe_ns=0 reference_total_ns=25804161458 reference_conditioning_ns=59083 reference_prefill_ns=276097500 reference_first_decode_ns=24156459 reference_steady_decode_ns=25851391736 reference_unattributed_ns=0 reference_prefill_linear_probe_ns=0 reference_prefill_attention_probe_ns=0 reference_prefill_misc_probe_ns=0`

## Preserved ARM Flash Baseline

- Preserved baseline artifact: `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`
- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- Note: this preserved ARM flash baseline remains tied to the archived Llama canonical slice and is not directly compared against the current maintained publication because the benchmark case identity changed explicitly.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 1899.542 | 6059.042 | 0.314x |
| `batch/planner_seq` | 2192.667 | 3211.125 | 0.683x |
| `batch/planner_simple` | 984.125 | 2442.334 | 0.403x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15231.250 | 18612.625 | 0.818x |
| `gbnf/rule_parser_basic` | 481.125 | 272.458 | 1.766x |
| `gbnf/rule_parser_complex` | 3246.417 | 1522.625 | 2.132x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 408512500.000 | 307468375.000 | 1.329x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 589380666.000 | 523875709.000 | 1.125x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 2471907584.000 | 2842218125.000 | 0.870x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 22368668542.000 | 25874351417.000 | 0.865x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 110291083.000 | 123507708.000 | 0.893x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 203857542.000 | 219353333.000 | 0.929x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1229578375.000 | 1261727625.000 | 0.975x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 15577386000.000 | 16990208667.000 | 0.917x |
| `kernel/aarch64/op_add` | 115.167 | 6302.958 | 0.018x |
| `kernel/aarch64/op_cos` | 1645.875 | 6281.125 | 0.262x |
| `kernel/aarch64/op_div` | 114.458 | 4799.917 | 0.024x |
| `kernel/aarch64/op_dup` | 106.792 | 4575.083 | 0.023x |
| `kernel/aarch64/op_log` | 1862.708 | 6937.292 | 0.269x |
| `kernel/aarch64/op_mul` | 121.916 | 5915.333 | 0.021x |
| `kernel/aarch64/op_mul_mat` | 4931.750 | 11146.666 | 0.442x |
| `kernel/aarch64/op_sin` | 1298.416 | 6179.958 | 0.210x |
| `kernel/aarch64/op_soft_max` | 1019.084 | 5772.875 | 0.177x |
| `kernel/aarch64/op_sqr` | 105.000 | 4976.334 | 0.021x |
| `kernel/aarch64/op_sqrt` | 153.208 | 4706.791 | 0.033x |
| `kernel/aarch64/op_sub` | 110.334 | 6405.917 | 0.017x |
| `kernel/aarch64/op_unary_exp` | 1410.042 | 5635.083 | 0.250x |
| `kernel/aarch64/op_unary_neg` | 118.917 | 4843.458 | 0.025x |
| `kernel/aarch64/op_unary_relu` | 132.708 | 5172.667 | 0.026x |
| `logits/sampler_raw/vocab_128000` | 20448.250 | 18286.584 | 1.118x |
| `logits/sampler_raw/vocab_256000` | 38871.208 | 39891.667 | 0.974x |
| `logits/sampler_raw/vocab_32000` | 4706.125 | 5408.750 | 0.870x |
| `logits/sampler_sml/vocab_128000` | 16550.666 | 16180.417 | 1.023x |
| `logits/sampler_sml/vocab_256000` | 29869.333 | 26783.083 | 1.115x |
| `logits/sampler_sml/vocab_32000` | 4375.958 | 4157.916 | 1.052x |
| `logits/validator_raw/vocab_128000` | 91645.042 | 91257.000 | 1.004x |
| `logits/validator_raw/vocab_256000` | 184238.167 | 180452.625 | 1.021x |
| `logits/validator_raw/vocab_32000` | 23894.208 | 23857.084 | 1.002x |
| `logits/validator_sml/vocab_128000` | 99734.208 | 100179.083 | 0.996x |
| `logits/validator_sml/vocab_256000` | 197700.584 | 200569.083 | 0.986x |
| `logits/validator_sml/vocab_32000` | 24562.167 | 24648.084 | 0.997x |
| `memory/hybrid_full` | 434.917 | 36054.000 | 0.012x |
| `memory/kv_full` | 125.917 | 36519.250 | 0.003x |
| `memory/recurrent_full` | 139.916 | 6720.833 | 0.021x |
| `text/encoders/bpe_long` | 60.500 | 60.667 | 0.997x |
| `text/encoders/bpe_short` | 66.000 | 56.333 | 1.172x |
| `text/encoders/fallback_long` | 2457.583 | 2461.583 | 0.998x |
| `text/encoders/fallback_short` | 62.625 | 64.041 | 0.978x |
| `text/encoders/plamo2_long` | 7544.958 | 7705.167 | 0.979x |
| `text/encoders/plamo2_short` | 210.750 | 244.083 | 0.863x |
| `text/encoders/rwkv_long` | 839708.875 | 833583.667 | 1.007x |
| `text/encoders/rwkv_short` | 56620.500 | 56107.458 | 1.009x |
| `text/encoders/spm_long` | 3577368.416 | 3565931.542 | 1.003x |
| `text/encoders/spm_short` | 1321.333 | 1267.000 | 1.043x |
| `text/encoders/ugm_long` | 1490079.792 | 1489678.583 | 1.000x |
| `text/encoders/ugm_short` | 757.542 | 778.667 | 0.973x |
| `text/encoders/wpm_long` | 30564.458 | 30743.583 | 0.994x |
| `text/encoders/wpm_short` | 531.708 | 552.333 | 0.963x |
| `text/jinja/formatter_long` | 65.667 | 239787.000 | 0.000x |
| `text/jinja/formatter_short` | 15.958 | 4022.333 | 0.004x |
| `text/jinja/parser_long` | 65216.000 | 50247.292 | 1.298x |
| `text/jinja/parser_short` | 938.333 | 501.292 | 1.872x |
| `tokenizer/full_bpe_long` | 13455.833 | 13899.792 | 0.968x |
| `tokenizer/full_bpe_short` | 332.541 | 320.166 | 1.039x |
| `tokenizer/full_plamo2_long` | 12750.208 | 12621.000 | 1.010x |
| `tokenizer/full_plamo2_short` | 2477.000 | 1967.041 | 1.259x |
| `tokenizer/full_rwkv_long` | 833512.042 | 830968.792 | 1.003x |
| `tokenizer/full_rwkv_short` | 55883.916 | 55148.333 | 1.013x |
| `tokenizer/full_spm_long` | 3577063.041 | 3578127.416 | 1.000x |
| `tokenizer/full_spm_short` | 1414.417 | 1489.542 | 0.950x |
| `tokenizer/full_ugm_long` | 1489834.542 | 1489502.458 | 1.000x |
| `tokenizer/full_ugm_short` | 2441.583 | 2474.416 | 0.987x |
| `tokenizer/full_wpm_long` | 32083.333 | 32329.709 | 0.992x |
| `tokenizer/full_wpm_short` | 2337.166 | 2161.375 | 1.081x |
| `tokenizer/preprocessor_bpe_long` | 3399.625 | 5282.459 | 0.644x |
| `tokenizer/preprocessor_bpe_short` | 128.958 | 1721.708 | 0.075x |
| `tokenizer/preprocessor_plamo2_long` | 4149.541 | 7122.458 | 0.583x |
| `tokenizer/preprocessor_plamo2_short` | 2667.834 | 4600.125 | 0.580x |
| `tokenizer/preprocessor_rwkv_long` | 4034.166 | 7117.667 | 0.567x |
| `tokenizer/preprocessor_rwkv_short` | 2521.875 | 5207.916 | 0.484x |
| `tokenizer/preprocessor_spm_long` | 4162.958 | 6702.333 | 0.621x |
| `tokenizer/preprocessor_spm_short` | 2514.000 | 4829.083 | 0.521x |
| `tokenizer/preprocessor_ugm_long` | 4174.750 | 7107.875 | 0.587x |
| `tokenizer/preprocessor_ugm_short` | 2617.458 | 4693.792 | 0.558x |
| `tokenizer/preprocessor_wpm_long` | 4767.834 | 7016.500 | 0.680x |
| `tokenizer/preprocessor_wpm_short` | 2427.166 | 4889.042 | 0.496x |
