# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Generation Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `benchmark_config: iterations=1000 runs=3 warmup_iterations=100 warmup_runs=1 generation_iterations=1 generation_runs=1 generation_warmup_iterations=0 generation_warmup_runs=0`
- `reference_impl: source=cmake_fetch ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `generation_formatter_contract: source=tokenizer.chat_template support=supported_contract shape=structured_chat_messages_v1 tools=none add_generation_prompt=true enable_thinking=false`
- `generation_flash_evidence: case=generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 flash_dispatch_calls=644 optimized_flash_dispatch_calls=644 shared_flash_dispatch_calls=0 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 emel.cpp 515899791.000 ns/op, llama.cpp 143516959.000 ns/op, ratio=3.595x`

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=4509 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q6_dispatch_calls=0 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical Qwen3 workload stayed on the approved runtime contract with native q8_0 projection and output dispatch, explicit dense-f32-by-contract token embedding and per-head Q/K RMS norm vectors, and no disallowed fallback or explicit no-claim branch on the supported path.

## Preserved ARM Flash Baseline

- Preserved baseline artifact: `snapshots/bench/generation_pre_arm_flash_optimized_baseline.txt`
- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- Note: this preserved ARM flash baseline remains tied to the archived Llama canonical slice and is not directly compared against the current canonical Qwen3 publication because the benchmark case identity changed explicitly.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2115.458 | 6326.583 | 0.334x |
| `batch/planner_seq` | 2482.917 | 2629.666 | 0.944x |
| `batch/planner_simple` | 1325.917 | 2376.666 | 0.558x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15210.417 | 18019.959 | 0.844x |
| `gbnf/rule_parser_basic` | 466.208 | 282.375 | 1.651x |
| `gbnf/rule_parser_complex` | 3269.375 | 1555.583 | 2.102x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 515899791.000 | 143516959.000 | 3.595x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 776823125.000 | 205432667.000 | 3.781x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 3458358792.000 | 1235299291.000 | 2.800x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 34363718333.000 | 16759728625.000 | 2.050x |
| `kernel/aarch64/op_add` | 105.625 | 5373.000 | 0.020x |
| `kernel/aarch64/op_cos` | 1683.000 | 5771.375 | 0.292x |
| `kernel/aarch64/op_div` | 104.750 | 3993.833 | 0.026x |
| `kernel/aarch64/op_dup` | 99.792 | 4326.709 | 0.023x |
| `kernel/aarch64/op_log` | 1800.833 | 5856.166 | 0.308x |
| `kernel/aarch64/op_mul` | 104.334 | 4907.458 | 0.021x |
| `kernel/aarch64/op_mul_mat` | 4744.625 | 10313.625 | 0.460x |
| `kernel/aarch64/op_sin` | 1250.083 | 5685.125 | 0.220x |
| `kernel/aarch64/op_soft_max` | 1117.209 | 4713.625 | 0.237x |
| `kernel/aarch64/op_sqr` | 99.333 | 4290.667 | 0.023x |
| `kernel/aarch64/op_sqrt` | 141.375 | 4297.333 | 0.033x |
| `kernel/aarch64/op_sub` | 104.125 | 5059.208 | 0.021x |
| `kernel/aarch64/op_unary_exp` | 1360.333 | 5388.958 | 0.252x |
| `kernel/aarch64/op_unary_neg` | 112.792 | 4192.000 | 0.027x |
| `kernel/aarch64/op_unary_relu` | 135.750 | 4064.083 | 0.033x |
| `logits/sampler_raw/vocab_128000` | 18292.125 | 18981.667 | 0.964x |
| `logits/sampler_raw/vocab_256000` | 38324.709 | 37072.583 | 1.034x |
| `logits/sampler_raw/vocab_32000` | 4936.417 | 4936.875 | 1.000x |
| `logits/sampler_sml/vocab_128000` | 19061.917 | 15673.041 | 1.216x |
| `logits/sampler_sml/vocab_256000` | 36498.125 | 28775.041 | 1.268x |
| `logits/sampler_sml/vocab_32000` | 3881.417 | 4749.541 | 0.817x |
| `logits/validator_raw/vocab_128000` | 92176.416 | 94883.208 | 0.971x |
| `logits/validator_raw/vocab_256000` | 182519.792 | 186385.917 | 0.979x |
| `logits/validator_raw/vocab_32000` | 24233.459 | 23775.375 | 1.019x |
| `logits/validator_sml/vocab_128000` | 99610.916 | 99357.750 | 1.003x |
| `logits/validator_sml/vocab_256000` | 197688.708 | 198425.292 | 0.996x |
| `logits/validator_sml/vocab_32000` | 24430.125 | 23875.875 | 1.023x |
| `memory/hybrid_full` | 444.417 | 35242.958 | 0.013x |
| `memory/kv_full` | 127.125 | 34051.125 | 0.004x |
| `memory/recurrent_full` | 138.416 | 4579.458 | 0.030x |
| `text/encoders/bpe_long` | 62.750 | 63.375 | 0.990x |
| `text/encoders/bpe_short` | 58.041 | 59.083 | 0.982x |
| `text/encoders/fallback_long` | 2467.833 | 2418.250 | 1.021x |
| `text/encoders/fallback_short` | 63.542 | 64.542 | 0.985x |
| `text/encoders/plamo2_long` | 7851.250 | 7687.625 | 1.021x |
| `text/encoders/plamo2_short` | 205.333 | 195.250 | 1.052x |
| `text/encoders/rwkv_long` | 830978.375 | 830529.125 | 1.001x |
| `text/encoders/rwkv_short` | 56269.584 | 56271.834 | 1.000x |
| `text/encoders/spm_long` | 3625718.542 | 3633464.292 | 0.998x |
| `text/encoders/spm_short` | 1322.166 | 1317.458 | 1.004x |
| `text/encoders/ugm_long` | 1376099.958 | 1387376.125 | 0.992x |
| `text/encoders/ugm_short` | 736.541 | 735.458 | 1.001x |
| `text/encoders/wpm_long` | 30432.834 | 30476.792 | 0.999x |
| `text/encoders/wpm_short` | 542.583 | 542.500 | 1.000x |
| `text/jinja/formatter_long` | 60.791 | 230714.792 | 0.000x |
| `text/jinja/formatter_short` | 15.708 | 3806.208 | 0.004x |
| `text/jinja/parser_long` | 64516.958 | 49564.667 | 1.302x |
| `text/jinja/parser_short` | 911.292 | 500.125 | 1.822x |
| `tokenizer/full_bpe_long` | 13493.500 | 13460.542 | 1.002x |
| `tokenizer/full_bpe_short` | 306.625 | 312.208 | 0.982x |
| `tokenizer/full_plamo2_long` | 12710.666 | 12403.209 | 1.025x |
| `tokenizer/full_plamo2_short` | 1959.833 | 1984.542 | 0.988x |
| `tokenizer/full_rwkv_long` | 833077.417 | 832764.333 | 1.000x |
| `tokenizer/full_rwkv_short` | 54928.667 | 55174.333 | 0.996x |
| `tokenizer/full_spm_long` | 3620218.292 | 3630952.833 | 0.997x |
| `tokenizer/full_spm_short` | 1461.541 | 1468.166 | 0.995x |
| `tokenizer/full_ugm_long` | 1380726.916 | 1382819.584 | 0.998x |
| `tokenizer/full_ugm_short` | 2446.333 | 2454.959 | 0.996x |
| `tokenizer/full_wpm_long` | 32386.083 | 32393.458 | 1.000x |
| `tokenizer/full_wpm_short` | 2253.084 | 2256.625 | 0.998x |
| `tokenizer/preprocessor_bpe_long` | 3471.541 | 5274.833 | 0.658x |
| `tokenizer/preprocessor_bpe_short` | 142.625 | 1712.667 | 0.083x |
| `tokenizer/preprocessor_plamo2_long` | 4147.292 | 5377.792 | 0.771x |
| `tokenizer/preprocessor_plamo2_short` | 2490.792 | 3589.375 | 0.694x |
| `tokenizer/preprocessor_rwkv_long` | 4157.875 | 5386.958 | 0.772x |
| `tokenizer/preprocessor_rwkv_short` | 2477.417 | 3594.208 | 0.689x |
| `tokenizer/preprocessor_spm_long` | 4152.583 | 5422.959 | 0.766x |
| `tokenizer/preprocessor_spm_short` | 2480.083 | 3993.166 | 0.621x |
| `tokenizer/preprocessor_ugm_long` | 4298.500 | 5519.209 | 0.779x |
| `tokenizer/preprocessor_ugm_short` | 2473.750 | 3600.166 | 0.687x |
| `tokenizer/preprocessor_wpm_long` | 4134.375 | 5343.542 | 0.774x |
| `tokenizer/preprocessor_wpm_short` | 2464.709 | 3483.958 | 0.707x |
