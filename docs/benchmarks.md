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
- Current compare row: `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 emel.cpp 1965166.000 ns/op, llama.cpp 3108542.000 ns/op, ratio=0.632x`

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_quantized_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 optimized_q2_dispatch_calls=8 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=6 shared_q3_dispatch_calls=0 optimized_q6_dispatch_calls=1 shared_q6_dispatch_calls=0`

## Preserved ARM Flash Baseline Comparison

- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- `current_emel_ns=1965166.000`
- `current_reference_ns=3108542.000`
- `current_ratio=0.632x`
- `speedup=3.560x`
- `latency_drop_pct=71.9`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2060.542 | 6052.750 | 0.340x |
| `batch/planner_seq` | 2246.334 | 2634.583 | 0.853x |
| `batch/planner_simple` | 1092.375 | 2253.042 | 0.485x |
| `gbnf/rule_parser_basic` | 478.959 | 265.083 | 1.807x |
| `gbnf/rule_parser_complex` | 3332.709 | 1520.291 | 2.192x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 1965166.000 | 3108542.000 | 0.632x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_10` | 18053042.000 | 17255375.000 | 1.046x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_100` | 186626375.000 | 160084041.000 | 1.166x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1000` | 2304267791.000 | 1757807292.000 | 1.311x |
| `kernel/aarch64/op_add` | 98.833 | 5680.542 | 0.017x |
| `kernel/aarch64/op_cos` | 1667.208 | 6354.584 | 0.262x |
| `kernel/aarch64/op_div` | 95.917 | 4907.000 | 0.020x |
| `kernel/aarch64/op_dup` | 94.583 | 4778.292 | 0.020x |
| `kernel/aarch64/op_log` | 1864.375 | 6420.750 | 0.290x |
| `kernel/aarch64/op_mul` | 99.917 | 5626.125 | 0.018x |
| `kernel/aarch64/op_mul_mat` | 4488.833 | 10614.875 | 0.423x |
| `kernel/aarch64/op_sin` | 1441.083 | 5985.000 | 0.241x |
| `kernel/aarch64/op_soft_max` | 2152.917 | 5237.958 | 0.411x |
| `kernel/aarch64/op_sqr` | 88.667 | 4645.583 | 0.019x |
| `kernel/aarch64/op_sqrt` | 135.500 | 4790.125 | 0.028x |
| `kernel/aarch64/op_sub` | 97.917 | 5583.750 | 0.018x |
| `kernel/aarch64/op_unary_exp` | 1369.000 | 6035.458 | 0.227x |
| `kernel/aarch64/op_unary_neg` | 110.917 | 4460.583 | 0.025x |
| `kernel/aarch64/op_unary_relu` | 115.541 | 4696.834 | 0.025x |
| `logits/sampler_raw/vocab_128000` | 20165.000 | 18051.375 | 1.117x |
| `logits/sampler_raw/vocab_256000` | 36234.292 | 35704.542 | 1.015x |
| `logits/sampler_raw/vocab_32000` | 5240.833 | 5482.875 | 0.956x |
| `logits/sampler_sml/vocab_128000` | 18051.792 | 18254.208 | 0.989x |
| `logits/sampler_sml/vocab_256000` | 30457.917 | 29930.459 | 1.018x |
| `logits/sampler_sml/vocab_32000` | 5612.833 | 3654.000 | 1.536x |
| `logits/validator_raw/vocab_128000` | 91342.292 | 91368.834 | 1.000x |
| `logits/validator_raw/vocab_256000` | 181699.167 | 182396.208 | 0.996x |
| `logits/validator_raw/vocab_32000` | 24112.375 | 23864.750 | 1.010x |
| `logits/validator_sml/vocab_128000` | 100541.916 | 99212.292 | 1.013x |
| `logits/validator_sml/vocab_256000` | 198689.625 | 199663.666 | 0.995x |
| `logits/validator_sml/vocab_32000` | 24625.000 | 25515.583 | 0.965x |
| `memory/hybrid_full` | 451.750 | 34322.833 | 0.013x |
| `memory/kv_full` | 147.917 | 33932.542 | 0.004x |
| `memory/recurrent_full` | 144.916 | 4597.458 | 0.032x |
| `text/encoders/bpe_long` | 62.542 | 66.334 | 0.943x |
| `text/encoders/bpe_short` | 56.958 | 56.458 | 1.009x |
| `text/encoders/fallback_long` | 2460.333 | 2544.416 | 0.967x |
| `text/encoders/fallback_short` | 62.458 | 64.375 | 0.970x |
| `text/encoders/plamo2_long` | 7671.000 | 7837.833 | 0.979x |
| `text/encoders/plamo2_short` | 217.791 | 214.292 | 1.016x |
| `text/encoders/rwkv_long` | 828999.125 | 826518.625 | 1.003x |
| `text/encoders/rwkv_short` | 56646.625 | 55524.417 | 1.020x |
| `text/encoders/spm_long` | 3652794.750 | 3633286.458 | 1.005x |
| `text/encoders/spm_short` | 1331.750 | 1355.167 | 0.983x |
| `text/encoders/ugm_long` | 1373127.000 | 1363214.166 | 1.007x |
| `text/encoders/ugm_short` | 725.083 | 750.000 | 0.967x |
| `text/encoders/wpm_long` | 30464.166 | 30717.458 | 0.992x |
| `text/encoders/wpm_short` | 568.791 | 535.542 | 1.062x |
| `text/jinja/formatter_long` | 66.709 | 225120.125 | 0.000x |
| `text/jinja/formatter_short` | 15.917 | 3759.000 | 0.004x |
| `text/jinja/parser_long` | 64745.708 | 49772.125 | 1.301x |
| `text/jinja/parser_short` | 963.041 | 513.667 | 1.875x |
| `tokenizer/full_bpe_long` | 13389.625 | 13863.500 | 0.966x |
| `tokenizer/full_bpe_short` | 319.250 | 299.333 | 1.067x |
| `tokenizer/full_plamo2_long` | 12742.625 | 12652.708 | 1.007x |
| `tokenizer/full_plamo2_short` | 1875.791 | 1931.083 | 0.971x |
| `tokenizer/full_rwkv_long` | 841213.000 | 828549.042 | 1.015x |
| `tokenizer/full_rwkv_short` | 55114.458 | 55301.125 | 0.997x |
| `tokenizer/full_spm_long` | 3627671.250 | 3669710.708 | 0.989x |
| `tokenizer/full_spm_short` | 1482.667 | 1539.875 | 0.963x |
| `tokenizer/full_ugm_long` | 1377975.583 | 1387376.833 | 0.993x |
| `tokenizer/full_ugm_short` | 2393.500 | 2368.666 | 1.010x |
| `tokenizer/full_wpm_long` | 32302.666 | 31863.000 | 1.014x |
| `tokenizer/full_wpm_short` | 2155.250 | 2253.667 | 0.956x |
| `tokenizer/preprocessor_bpe_long` | 3355.500 | 5143.625 | 0.652x |
| `tokenizer/preprocessor_bpe_short` | 132.500 | 1650.625 | 0.080x |
| `tokenizer/preprocessor_plamo2_long` | 4213.625 | 5386.125 | 0.782x |
| `tokenizer/preprocessor_plamo2_short` | 2545.417 | 3591.166 | 0.709x |
| `tokenizer/preprocessor_rwkv_long` | 4181.459 | 5408.917 | 0.773x |
| `tokenizer/preprocessor_rwkv_short` | 2350.542 | 3550.875 | 0.662x |
| `tokenizer/preprocessor_spm_long` | 4125.542 | 5425.000 | 0.760x |
| `tokenizer/preprocessor_spm_short` | 2471.834 | 3661.208 | 0.675x |
| `tokenizer/preprocessor_ugm_long` | 4378.042 | 5535.333 | 0.791x |
| `tokenizer/preprocessor_ugm_short` | 2391.292 | 3582.375 | 0.668x |
| `tokenizer/preprocessor_wpm_long` | 4192.833 | 5543.750 | 0.756x |
| `tokenizer/preprocessor_wpm_short` | 2516.791 | 3447.917 | 0.730x |
