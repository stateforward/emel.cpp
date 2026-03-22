# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Flash Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- Preserved baseline artifact: `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`
- `reference_impl: source=cmake_fetch ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `generation_flash_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 flash_dispatch_calls=2 optimized_flash_dispatch_calls=2 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 emel.cpp 6133750.000 ns/op, llama.cpp 3028833.000 ns/op, ratio=2.025x`

## Preserved ARM Flash Baseline Comparison

- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- `current_emel_ns=6133750.000`
- `current_reference_ns=3028833.000`
- `current_ratio=2.025x`
- `speedup=1.140x`
- `latency_drop_pct=12.3`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2022.166 | 6127.709 | 0.330x |
| `batch/planner_seq` | 5207.375 | 2673.833 | 1.948x |
| `batch/planner_simple` | 1039.625 | 2254.750 | 0.461x |
| `gbnf/rule_parser_basic` | 474.708 | 275.125 | 1.725x |
| `gbnf/rule_parser_complex` | 3197.459 | 1490.667 | 2.145x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 6133750.000 | 3028833.000 | 2.025x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_8` | 48180042.000 | 13764250.000 | 3.500x |
| `kernel/aarch64/op_add` | 95.667 | 5206.208 | 0.018x |
| `kernel/aarch64/op_cos` | 1624.375 | 6107.459 | 0.266x |
| `kernel/aarch64/op_div` | 97.125 | 4670.083 | 0.021x |
| `kernel/aarch64/op_dup` | 90.125 | 4171.000 | 0.022x |
| `kernel/aarch64/op_log` | 1812.875 | 6397.542 | 0.283x |
| `kernel/aarch64/op_mul` | 96.166 | 5075.708 | 0.019x |
| `kernel/aarch64/op_mul_mat` | 4552.250 | 11229.000 | 0.405x |
| `kernel/aarch64/op_sin` | 1327.292 | 5821.000 | 0.228x |
| `kernel/aarch64/op_soft_max` | 2084.916 | 5462.541 | 0.382x |
| `kernel/aarch64/op_sqr` | 93.916 | 4457.792 | 0.021x |
| `kernel/aarch64/op_sqrt` | 150.458 | 4666.916 | 0.032x |
| `kernel/aarch64/op_sub` | 95.958 | 5347.333 | 0.018x |
| `kernel/aarch64/op_unary_exp` | 1351.292 | 5804.667 | 0.233x |
| `kernel/aarch64/op_unary_neg` | 103.875 | 4898.250 | 0.021x |
| `kernel/aarch64/op_unary_relu` | 112.625 | 4503.417 | 0.025x |
| `logits/sampler_raw/vocab_128000` | 18524.375 | 18980.083 | 0.976x |
| `logits/sampler_raw/vocab_256000` | 35138.584 | 36542.208 | 0.962x |
| `logits/sampler_raw/vocab_32000` | 4224.417 | 4569.084 | 0.925x |
| `logits/sampler_sml/vocab_128000` | 20955.334 | 12469.917 | 1.680x |
| `logits/sampler_sml/vocab_256000` | 29856.917 | 36650.417 | 0.815x |
| `logits/sampler_sml/vocab_32000` | 5363.750 | 3480.666 | 1.541x |
| `logits/validator_raw/vocab_128000` | 87302.083 | 90881.125 | 0.961x |
| `logits/validator_raw/vocab_256000` | 173066.333 | 176304.542 | 0.982x |
| `logits/validator_raw/vocab_32000` | 23303.875 | 23827.333 | 0.978x |
| `logits/validator_sml/vocab_128000` | 95700.083 | 95671.417 | 1.000x |
| `logits/validator_sml/vocab_256000` | 193896.542 | 198291.334 | 0.978x |
| `logits/validator_sml/vocab_32000` | 23366.167 | 23862.250 | 0.979x |
| `memory/hybrid_full` | 411.875 | 34131.958 | 0.012x |
| `memory/kv_full` | 130.292 | 34144.541 | 0.004x |
| `memory/recurrent_full` | 131.125 | 4471.125 | 0.029x |
| `text/encoders/bpe_long` | 60.958 | 60.958 | 1.000x |
| `text/encoders/bpe_short` | 58.625 | 54.500 | 1.076x |
| `text/encoders/fallback_long` | 2404.209 | 2360.667 | 1.018x |
| `text/encoders/fallback_short` | 61.583 | 60.666 | 1.015x |
| `text/encoders/plamo2_long` | 7489.250 | 7309.958 | 1.025x |
| `text/encoders/plamo2_short` | 207.791 | 196.167 | 1.059x |
| `text/encoders/rwkv_long` | 809187.958 | 802334.917 | 1.009x |
| `text/encoders/rwkv_short` | 56249.125 | 56623.333 | 0.993x |
| `text/encoders/spm_long` | 3492274.334 | 3470289.000 | 1.006x |
| `text/encoders/spm_short` | 1305.000 | 1249.500 | 1.044x |
| `text/encoders/ugm_long` | 1346845.208 | 1334135.916 | 1.010x |
| `text/encoders/ugm_short` | 775.042 | 704.541 | 1.100x |
| `text/encoders/wpm_long` | 31070.958 | 29370.958 | 1.058x |
| `text/encoders/wpm_short` | 521.792 | 522.250 | 0.999x |
| `text/jinja/formatter_long` | 59.416 | 214883.459 | 0.000x |
| `text/jinja/formatter_short` | 15.708 | 3514.542 | 0.004x |
| `text/jinja/parser_long` | 61929.917 | 48137.000 | 1.287x |
| `text/jinja/parser_short` | 926.292 | 478.458 | 1.936x |
| `tokenizer/full_bpe_long` | 13392.666 | 13023.500 | 1.028x |
| `tokenizer/full_bpe_short` | 317.041 | 304.500 | 1.041x |
| `tokenizer/full_plamo2_long` | 12472.416 | 12018.208 | 1.038x |
| `tokenizer/full_plamo2_short` | 2363.292 | 1856.666 | 1.273x |
| `tokenizer/full_rwkv_long` | 817616.791 | 808931.834 | 1.011x |
| `tokenizer/full_rwkv_short` | 53800.500 | 53376.000 | 1.008x |
| `tokenizer/full_spm_long` | 3551888.333 | 3477092.667 | 1.022x |
| `tokenizer/full_spm_short` | 1430.667 | 1432.875 | 0.998x |
| `tokenizer/full_ugm_long` | 1347004.375 | 1332779.333 | 1.011x |
| `tokenizer/full_ugm_short` | 2353.083 | 2343.291 | 1.004x |
| `tokenizer/full_wpm_long` | 31900.750 | 31048.625 | 1.027x |
| `tokenizer/full_wpm_short` | 2623.375 | 2204.292 | 1.190x |
| `tokenizer/preprocessor_bpe_long` | 3518.333 | 5751.791 | 0.612x |
| `tokenizer/preprocessor_bpe_short` | 140.541 | 1772.917 | 0.079x |
| `tokenizer/preprocessor_plamo2_long` | 3887.625 | 5305.541 | 0.733x |
| `tokenizer/preprocessor_plamo2_short` | 2337.000 | 3491.458 | 0.669x |
| `tokenizer/preprocessor_rwkv_long` | 3898.250 | 5445.292 | 0.716x |
| `tokenizer/preprocessor_rwkv_short` | 2405.833 | 3613.458 | 0.666x |
| `tokenizer/preprocessor_spm_long` | 3995.042 | 5584.041 | 0.715x |
| `tokenizer/preprocessor_spm_short` | 2352.167 | 3777.500 | 0.623x |
| `tokenizer/preprocessor_ugm_long` | 4120.417 | 5969.166 | 0.690x |
| `tokenizer/preprocessor_ugm_short` | 2300.416 | 3644.167 | 0.631x |
| `tokenizer/preprocessor_wpm_long` | 4043.333 | 5635.333 | 0.717x |
| `tokenizer/preprocessor_wpm_short` | 2453.791 | 3577.167 | 0.686x |
