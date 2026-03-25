# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Flash Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- Preserved baseline artifact: `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`
- `reference_impl: source=cmake_fetch ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `generation_flash_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 flash_dispatch_calls=0 optimized_flash_dispatch_calls=0 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 emel.cpp 4135459.000 ns/op, llama.cpp 3194500.000 ns/op, ratio=1.295x`

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
- `current_emel_ns=4135459.000`
- `current_reference_ns=3194500.000`
- `current_ratio=1.295x`
- `speedup=1.692x`
- `latency_drop_pct=40.9`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 1993.125 | 6422.541 | 0.310x |
| `batch/planner_seq` | 2178.167 | 2702.042 | 0.806x |
| `batch/planner_simple` | 1037.417 | 2400.042 | 0.432x |
| `gbnf/rule_parser_basic` | 504.000 | 292.125 | 1.725x |
| `gbnf/rule_parser_complex` | 3290.250 | 1527.250 | 2.154x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 4135459.000 | 3194500.000 | 1.295x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_10` | 40107167.000 | 17093958.000 | 2.346x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_100` | 406015458.000 | 161191500.000 | 2.519x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1000` | 5289484000.000 | 1773140875.000 | 2.983x |
| `kernel/aarch64/op_add` | 100.792 | 5482.166 | 0.018x |
| `kernel/aarch64/op_cos` | 1676.666 | 6081.250 | 0.276x |
| `kernel/aarch64/op_div` | 110.000 | 4501.375 | 0.024x |
| `kernel/aarch64/op_dup` | 95.042 | 4495.875 | 0.021x |
| `kernel/aarch64/op_log` | 1878.292 | 6531.041 | 0.288x |
| `kernel/aarch64/op_mul` | 111.584 | 5553.000 | 0.020x |
| `kernel/aarch64/op_mul_mat` | 4539.958 | 10807.209 | 0.420x |
| `kernel/aarch64/op_sin` | 1307.625 | 5916.500 | 0.221x |
| `kernel/aarch64/op_soft_max` | 1021.083 | 5275.541 | 0.194x |
| `kernel/aarch64/op_sqr` | 101.250 | 4657.292 | 0.022x |
| `kernel/aarch64/op_sqrt` | 147.916 | 4851.000 | 0.030x |
| `kernel/aarch64/op_sub` | 100.958 | 5751.333 | 0.018x |
| `kernel/aarch64/op_unary_exp` | 1380.958 | 5946.750 | 0.232x |
| `kernel/aarch64/op_unary_neg` | 103.500 | 4797.583 | 0.022x |
| `kernel/aarch64/op_unary_relu` | 115.750 | 4815.167 | 0.024x |
| `logits/sampler_raw/vocab_128000` | 20166.458 | 18443.625 | 1.093x |
| `logits/sampler_raw/vocab_256000` | 38104.417 | 38633.083 | 0.986x |
| `logits/sampler_raw/vocab_32000` | 4329.000 | 5566.625 | 0.778x |
| `logits/sampler_sml/vocab_128000` | 17653.834 | 18641.875 | 0.947x |
| `logits/sampler_sml/vocab_256000` | 30503.000 | 32141.167 | 0.949x |
| `logits/sampler_sml/vocab_32000` | 4977.084 | 3517.084 | 1.415x |
| `logits/validator_raw/vocab_128000` | 163340.583 | 162545.084 | 1.005x |
| `logits/validator_raw/vocab_256000` | 336142.125 | 326633.750 | 1.029x |
| `logits/validator_raw/vocab_32000` | 41081.708 | 40939.917 | 1.003x |
| `logits/validator_sml/vocab_128000` | 100361.625 | 98715.750 | 1.017x |
| `logits/validator_sml/vocab_256000` | 198997.333 | 198059.125 | 1.005x |
| `logits/validator_sml/vocab_32000` | 24616.167 | 24530.167 | 1.004x |
| `memory/hybrid_full` | 450.167 | 34554.084 | 0.013x |
| `memory/kv_full` | 134.083 | 33809.916 | 0.004x |
| `memory/recurrent_full` | 143.791 | 4627.750 | 0.031x |
| `text/encoders/bpe_long` | 62.333 | 63.000 | 0.989x |
| `text/encoders/bpe_short` | 57.166 | 56.333 | 1.015x |
| `text/encoders/fallback_long` | 2466.709 | 2420.792 | 1.019x |
| `text/encoders/fallback_short` | 66.500 | 63.000 | 1.056x |
| `text/encoders/plamo2_long` | 7536.542 | 7768.625 | 0.970x |
| `text/encoders/plamo2_short` | 204.458 | 204.250 | 1.001x |
| `text/encoders/rwkv_long` | 839269.583 | 850578.875 | 0.987x |
| `text/encoders/rwkv_short` | 57024.584 | 56939.417 | 1.001x |
| `text/encoders/spm_long` | 3664208.708 | 3678366.209 | 0.996x |
| `text/encoders/spm_short` | 1270.583 | 1418.000 | 0.896x |
| `text/encoders/ugm_long` | 1384060.375 | 1386069.583 | 0.999x |
| `text/encoders/ugm_short` | 743.125 | 730.208 | 1.018x |
| `text/encoders/wpm_long` | 30683.125 | 30725.375 | 0.999x |
| `text/encoders/wpm_short` | 608.042 | 548.542 | 1.108x |
| `text/jinja/formatter_long` | 59.291 | 222734.917 | 0.000x |
| `text/jinja/formatter_short` | 15.750 | 3829.750 | 0.004x |
| `text/jinja/parser_long` | 66000.667 | 50167.333 | 1.316x |
| `text/jinja/parser_short` | 943.167 | 508.500 | 1.855x |
| `tokenizer/full_bpe_long` | 13773.333 | 13852.291 | 0.994x |
| `tokenizer/full_bpe_short` | 329.375 | 305.417 | 1.078x |
| `tokenizer/full_plamo2_long` | 12623.583 | 12369.166 | 1.021x |
| `tokenizer/full_plamo2_short` | 2001.500 | 1955.833 | 1.023x |
| `tokenizer/full_rwkv_long` | 838232.833 | 833388.625 | 1.006x |
| `tokenizer/full_rwkv_short` | 55368.125 | 55387.334 | 1.000x |
| `tokenizer/full_spm_long` | 3653798.583 | 3680401.125 | 0.993x |
| `tokenizer/full_spm_short` | 1466.000 | 1456.375 | 1.007x |
| `tokenizer/full_ugm_long` | 1381470.625 | 1419978.250 | 0.973x |
| `tokenizer/full_ugm_short` | 2452.709 | 2479.833 | 0.989x |
| `tokenizer/full_wpm_long` | 32604.750 | 32727.083 | 0.996x |
| `tokenizer/full_wpm_short` | 2262.084 | 2321.292 | 0.974x |
| `tokenizer/preprocessor_bpe_long` | 3374.708 | 5376.792 | 0.628x |
| `tokenizer/preprocessor_bpe_short` | 145.959 | 1717.916 | 0.085x |
| `tokenizer/preprocessor_plamo2_long` | 4170.083 | 5634.708 | 0.740x |
| `tokenizer/preprocessor_plamo2_short` | 2388.334 | 3732.917 | 0.640x |
| `tokenizer/preprocessor_rwkv_long` | 4037.083 | 5455.000 | 0.740x |
| `tokenizer/preprocessor_rwkv_short` | 2491.166 | 3652.958 | 0.682x |
| `tokenizer/preprocessor_spm_long` | 3932.583 | 5568.458 | 0.706x |
| `tokenizer/preprocessor_spm_short` | 2502.584 | 3591.375 | 0.697x |
| `tokenizer/preprocessor_ugm_long` | 4356.416 | 5735.834 | 0.760x |
| `tokenizer/preprocessor_ugm_short` | 2399.417 | 3626.792 | 0.662x |
| `tokenizer/preprocessor_wpm_long` | 4255.708 | 5513.375 | 0.772x |
| `tokenizer/preprocessor_wpm_short` | 2475.209 | 3676.458 | 0.673x |
