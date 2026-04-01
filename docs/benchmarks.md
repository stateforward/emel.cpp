# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=3 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch_latest ref=6de97b9d3ef7e5160083d3fecd98775ce959684e`
- `generation_architecture: lfm2`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 roles=system,user tools=none add_generation_prompt=true enable_thinking=false keep_past_thinking=false bos=<|startoftext|>`
- `generation_flash_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 flash_dispatch_calls=174 optimized_flash_dispatch_calls=174 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 551534459.000 ns/op, llama.cpp 479304833.000 ns/op, ratio=1.151x`

- The compare table below keeps additive generation rows for all maintained supported fixtures; this evidence block stays tied to the current maintained publication case.

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=0 packed_q8_0_dispatch_calls=0 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q4_dispatch_calls=2378 shared_q4_dispatch_calls=0 optimized_q6_dispatch_calls=291 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical Liquid workload stayed on the approved runtime contract with explicit dense-f32-by-contract stages, native quantized dispatch on the maintained path, and no disallowed fallback or explicit no-claim branch on the supported path.

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
| `batch/planner_equal` | 2242.667 | 5967.333 | 0.376x |
| `batch/planner_seq` | 2286.958 | 2592.333 | 0.882x |
| `batch/planner_simple` | 1078.667 | 2276.375 | 0.474x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15151.250 | 18013.542 | 0.841x |
| `gbnf/rule_parser_basic` | 492.292 | 264.167 | 1.864x |
| `gbnf/rule_parser_complex` | 3572.375 | 1483.792 | 2.408x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 551534459.000 | 479304833.000 | 1.151x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 752454042.000 | 697544250.000 | 1.079x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 2750994250.000 | 2919427208.000 | 0.942x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 23924315208.000 | 26278029792.000 | 0.910x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 106205417.000 | 474239500.000 | 0.224x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 225224917.000 | 561747208.000 | 0.401x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1293557041.000 | 1637228291.000 | 0.790x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 16310503125.000 | 17867865542.000 | 0.913x |
| `kernel/aarch64/op_add` | 105.166 | 5445.250 | 0.019x |
| `kernel/aarch64/op_cos` | 1687.375 | 6151.875 | 0.274x |
| `kernel/aarch64/op_div` | 107.666 | 4713.334 | 0.023x |
| `kernel/aarch64/op_dup` | 96.417 | 4270.375 | 0.023x |
| `kernel/aarch64/op_log` | 1864.500 | 6391.333 | 0.292x |
| `kernel/aarch64/op_mul` | 100.875 | 5818.416 | 0.017x |
| `kernel/aarch64/op_mul_mat` | 5001.209 | 10482.500 | 0.477x |
| `kernel/aarch64/op_sin` | 1335.250 | 6641.792 | 0.201x |
| `kernel/aarch64/op_soft_max` | 1063.708 | 5269.208 | 0.202x |
| `kernel/aarch64/op_sqr` | 99.333 | 4357.792 | 0.023x |
| `kernel/aarch64/op_sqrt` | 141.375 | 4750.167 | 0.030x |
| `kernel/aarch64/op_sub` | 105.292 | 5313.417 | 0.020x |
| `kernel/aarch64/op_unary_exp` | 1388.375 | 5561.667 | 0.250x |
| `kernel/aarch64/op_unary_neg` | 118.833 | 4458.875 | 0.027x |
| `kernel/aarch64/op_unary_relu` | 123.250 | 4508.625 | 0.027x |
| `logits/sampler_raw/vocab_128000` | 19368.666 | 18601.375 | 1.041x |
| `logits/sampler_raw/vocab_256000` | 37103.459 | 37338.917 | 0.994x |
| `logits/sampler_raw/vocab_32000` | 4981.042 | 4617.834 | 1.079x |
| `logits/sampler_sml/vocab_128000` | 13868.250 | 16316.542 | 0.850x |
| `logits/sampler_sml/vocab_256000` | 27685.750 | 28269.500 | 0.979x |
| `logits/sampler_sml/vocab_32000` | 4923.333 | 3953.791 | 1.245x |
| `logits/validator_raw/vocab_128000` | 91561.250 | 92555.875 | 0.989x |
| `logits/validator_raw/vocab_256000` | 182239.542 | 190859.375 | 0.955x |
| `logits/validator_raw/vocab_32000` | 24161.750 | 23851.292 | 1.013x |
| `logits/validator_sml/vocab_128000` | 99482.625 | 104814.833 | 0.949x |
| `logits/validator_sml/vocab_256000` | 201383.250 | 206358.000 | 0.976x |
| `logits/validator_sml/vocab_32000` | 24464.834 | 24733.625 | 0.989x |
| `memory/hybrid_full` | 452.000 | 34762.750 | 0.013x |
| `memory/kv_full` | 133.042 | 34091.000 | 0.004x |
| `memory/recurrent_full` | 148.625 | 4525.125 | 0.033x |
| `text/encoders/bpe_long` | 59.041 | 63.167 | 0.935x |
| `text/encoders/bpe_short` | 55.625 | 57.625 | 0.965x |
| `text/encoders/fallback_long` | 2416.500 | 2463.333 | 0.981x |
| `text/encoders/fallback_short` | 61.375 | 63.208 | 0.971x |
| `text/encoders/plamo2_long` | 7708.625 | 7844.458 | 0.983x |
| `text/encoders/plamo2_short` | 201.291 | 213.542 | 0.943x |
| `text/encoders/rwkv_long` | 839176.458 | 834787.500 | 1.005x |
| `text/encoders/rwkv_short` | 56454.208 | 56607.041 | 0.997x |
| `text/encoders/spm_long` | 3647285.292 | 3686635.667 | 0.989x |
| `text/encoders/spm_short` | 1273.084 | 1314.959 | 0.968x |
| `text/encoders/ugm_long` | 1423627.500 | 1390346.792 | 1.024x |
| `text/encoders/ugm_short` | 758.666 | 704.958 | 1.076x |
| `text/encoders/wpm_long` | 30616.750 | 30958.958 | 0.989x |
| `text/encoders/wpm_short` | 545.333 | 561.042 | 0.972x |
| `text/jinja/formatter_long` | 62.750 | 227751.750 | 0.000x |
| `text/jinja/formatter_short` | 17.500 | 3842.750 | 0.005x |
| `text/jinja/parser_long` | 75088.292 | 49688.167 | 1.511x |
| `text/jinja/parser_short` | 948.166 | 488.166 | 1.942x |
| `tokenizer/full_bpe_long` | 13505.250 | 13338.875 | 1.012x |
| `tokenizer/full_bpe_short` | 337.083 | 334.417 | 1.008x |
| `tokenizer/full_plamo2_long` | 12625.542 | 12557.625 | 1.005x |
| `tokenizer/full_plamo2_short` | 1952.875 | 2010.375 | 0.971x |
| `tokenizer/full_rwkv_long` | 841057.375 | 849232.542 | 0.990x |
| `tokenizer/full_rwkv_short` | 55593.875 | 54680.958 | 1.017x |
| `tokenizer/full_spm_long` | 3650841.750 | 3639192.250 | 1.003x |
| `tokenizer/full_spm_short` | 1525.667 | 1608.833 | 0.948x |
| `tokenizer/full_ugm_long` | 1400342.834 | 1398849.542 | 1.001x |
| `tokenizer/full_ugm_short` | 2533.833 | 2573.791 | 0.984x |
| `tokenizer/full_wpm_long` | 32367.334 | 32560.625 | 0.994x |
| `tokenizer/full_wpm_short` | 2265.042 | 2188.291 | 1.035x |
| `tokenizer/preprocessor_bpe_long` | 3420.083 | 5405.125 | 0.633x |
| `tokenizer/preprocessor_bpe_short` | 118.417 | 1811.084 | 0.065x |
| `tokenizer/preprocessor_plamo2_long` | 4102.791 | 6729.625 | 0.610x |
| `tokenizer/preprocessor_plamo2_short` | 2529.792 | 5091.792 | 0.497x |
| `tokenizer/preprocessor_rwkv_long` | 4085.208 | 7010.250 | 0.583x |
| `tokenizer/preprocessor_rwkv_short` | 2456.917 | 5034.333 | 0.488x |
| `tokenizer/preprocessor_spm_long` | 4106.458 | 7027.583 | 0.584x |
| `tokenizer/preprocessor_spm_short` | 2557.834 | 4893.541 | 0.523x |
| `tokenizer/preprocessor_ugm_long` | 4273.875 | 6745.750 | 0.634x |
| `tokenizer/preprocessor_ugm_short` | 2546.000 | 4739.458 | 0.537x |
| `tokenizer/preprocessor_wpm_long` | 4062.250 | 7238.166 | 0.561x |
| `tokenizer/preprocessor_wpm_short` | 2550.708 | 4920.291 | 0.518x |
