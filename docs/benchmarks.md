# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=3 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch_latest ref=c30e012253dd9e322c8e3424f808a5c74ecc46bf`
- `generation_architecture: lfm2`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 roles=system,user tools=none add_generation_prompt=true enable_thinking=false keep_past_thinking=false bos=<|startoftext|>`
- `generation_flash_evidence: case=generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 flash_dispatch_calls=174 optimized_flash_dispatch_calls=174 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 553167542.000 ns/op, llama.cpp 425307333.000 ns/op, ratio=1.301x`

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
| `batch/planner_equal` | 2093.459 | 6092.542 | 0.344x |
| `batch/planner_seq` | 2204.250 | 2682.417 | 0.822x |
| `batch/planner_simple` | 1044.917 | 2350.792 | 0.444x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15590.959 | 18403.833 | 0.847x |
| `gbnf/rule_parser_basic` | 492.084 | 258.208 | 1.906x |
| `gbnf/rule_parser_complex` | 3326.375 | 1504.667 | 2.211x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1` | 553167542.000 | 425307333.000 | 1.301x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10` | 761947667.000 | 641950458.000 | 1.187x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100` | 2761975917.000 | 2866691333.000 | 0.963x |
| `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000` | 24305002584.000 | 26216561042.000 | 0.927x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 105067334.000 | 453270208.000 | 0.232x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 208598458.000 | 551506167.000 | 0.378x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1267793875.000 | 1660124042.000 | 0.764x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 16285814958.000 | 17461994750.000 | 0.933x |
| `kernel/aarch64/op_add` | 106.833 | 5664.208 | 0.019x |
| `kernel/aarch64/op_cos` | 1641.125 | 5961.083 | 0.275x |
| `kernel/aarch64/op_div` | 106.125 | 4669.750 | 0.023x |
| `kernel/aarch64/op_dup` | 101.708 | 4406.291 | 0.023x |
| `kernel/aarch64/op_log` | 1871.958 | 6450.709 | 0.290x |
| `kernel/aarch64/op_mul` | 107.208 | 5835.000 | 0.018x |
| `kernel/aarch64/op_mul_mat` | 4885.583 | 11157.500 | 0.438x |
| `kernel/aarch64/op_sin` | 1332.625 | 5894.083 | 0.226x |
| `kernel/aarch64/op_soft_max` | 1018.833 | 5089.667 | 0.200x |
| `kernel/aarch64/op_sqr` | 98.167 | 4726.417 | 0.021x |
| `kernel/aarch64/op_sqrt` | 148.333 | 5144.458 | 0.029x |
| `kernel/aarch64/op_sub` | 111.042 | 5575.458 | 0.020x |
| `kernel/aarch64/op_unary_exp` | 1324.125 | 5774.666 | 0.229x |
| `kernel/aarch64/op_unary_neg` | 107.625 | 4852.333 | 0.022x |
| `kernel/aarch64/op_unary_relu` | 119.791 | 4883.167 | 0.025x |
| `logits/sampler_raw/vocab_128000` | 20732.459 | 18031.125 | 1.150x |
| `logits/sampler_raw/vocab_256000` | 38113.541 | 35831.167 | 1.064x |
| `logits/sampler_raw/vocab_32000` | 5243.333 | 4698.334 | 1.116x |
| `logits/sampler_sml/vocab_128000` | 18808.458 | 15460.834 | 1.217x |
| `logits/sampler_sml/vocab_256000` | 33696.833 | 32699.959 | 1.030x |
| `logits/sampler_sml/vocab_32000` | 4142.625 | 3795.792 | 1.091x |
| `logits/validator_raw/vocab_128000` | 92382.208 | 93279.167 | 0.990x |
| `logits/validator_raw/vocab_256000` | 182450.417 | 188601.167 | 0.967x |
| `logits/validator_raw/vocab_32000` | 23973.416 | 23963.417 | 1.000x |
| `logits/validator_sml/vocab_128000` | 100197.250 | 107877.334 | 0.929x |
| `logits/validator_sml/vocab_256000` | 206512.625 | 200835.291 | 1.028x |
| `logits/validator_sml/vocab_32000` | 25190.542 | 24861.500 | 1.013x |
| `memory/hybrid_full` | 438.875 | 34632.541 | 0.013x |
| `memory/kv_full` | 130.750 | 34247.500 | 0.004x |
| `memory/recurrent_full` | 137.375 | 4620.917 | 0.030x |
| `text/encoders/bpe_long` | 62.042 | 62.083 | 0.999x |
| `text/encoders/bpe_short` | 57.500 | 57.417 | 1.001x |
| `text/encoders/fallback_long` | 2455.959 | 2464.834 | 0.996x |
| `text/encoders/fallback_short` | 64.917 | 63.833 | 1.017x |
| `text/encoders/plamo2_long` | 7634.875 | 7717.125 | 0.989x |
| `text/encoders/plamo2_short` | 204.292 | 286.584 | 0.713x |
| `text/encoders/rwkv_long` | 839227.625 | 852561.875 | 0.984x |
| `text/encoders/rwkv_short` | 57440.208 | 56920.833 | 1.009x |
| `text/encoders/spm_long` | 3636845.625 | 3610133.208 | 1.007x |
| `text/encoders/spm_short` | 1274.250 | 1297.583 | 0.982x |
| `text/encoders/ugm_long` | 1448205.292 | 1380766.125 | 1.049x |
| `text/encoders/ugm_short` | 716.083 | 761.042 | 0.941x |
| `text/encoders/wpm_long` | 31340.333 | 30902.667 | 1.014x |
| `text/encoders/wpm_short` | 583.541 | 557.083 | 1.047x |
| `text/jinja/formatter_long` | 63.084 | 225073.375 | 0.000x |
| `text/jinja/formatter_short` | 16.500 | 3874.167 | 0.004x |
| `text/jinja/parser_long` | 65902.333 | 50337.333 | 1.309x |
| `text/jinja/parser_short` | 921.041 | 551.500 | 1.670x |
| `tokenizer/full_bpe_long` | 13619.666 | 13866.666 | 0.982x |
| `tokenizer/full_bpe_short` | 299.416 | 312.417 | 0.958x |
| `tokenizer/full_plamo2_long` | 12601.250 | 12507.250 | 1.008x |
| `tokenizer/full_plamo2_short` | 2029.041 | 1957.125 | 1.037x |
| `tokenizer/full_rwkv_long` | 835313.084 | 836846.833 | 0.998x |
| `tokenizer/full_rwkv_short` | 55224.542 | 55429.583 | 0.996x |
| `tokenizer/full_spm_long` | 3722605.209 | 3593684.500 | 1.036x |
| `tokenizer/full_spm_short` | 1422.542 | 1534.417 | 0.927x |
| `tokenizer/full_ugm_long` | 1482746.792 | 1407638.916 | 1.053x |
| `tokenizer/full_ugm_short` | 2509.625 | 2418.708 | 1.038x |
| `tokenizer/full_wpm_long` | 32867.708 | 32119.250 | 1.023x |
| `tokenizer/full_wpm_short` | 2309.333 | 2299.625 | 1.004x |
| `tokenizer/preprocessor_bpe_long` | 3533.875 | 5237.083 | 0.675x |
| `tokenizer/preprocessor_bpe_short` | 128.958 | 1653.333 | 0.078x |
| `tokenizer/preprocessor_plamo2_long` | 4251.333 | 7073.417 | 0.601x |
| `tokenizer/preprocessor_plamo2_short` | 2480.542 | 4800.875 | 0.517x |
| `tokenizer/preprocessor_rwkv_long` | 4156.875 | 7812.083 | 0.532x |
| `tokenizer/preprocessor_rwkv_short` | 2376.500 | 5149.125 | 0.462x |
| `tokenizer/preprocessor_spm_long` | 4083.125 | 6830.584 | 0.598x |
| `tokenizer/preprocessor_spm_short` | 2619.958 | 5178.917 | 0.506x |
| `tokenizer/preprocessor_ugm_long` | 4263.375 | 7027.625 | 0.607x |
| `tokenizer/preprocessor_ugm_short` | 2450.542 | 4736.916 | 0.517x |
| `tokenizer/preprocessor_wpm_long` | 4189.666 | 6780.625 | 0.618x |
| `tokenizer/preprocessor_wpm_short` | 2434.208 | 5081.292 | 0.479x |
