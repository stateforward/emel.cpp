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
- Current compare row: `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 emel.cpp 121724417.000 ns/op, llama.cpp 117844834.000 ns/op, ratio=1.033x`

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=6 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1 native_q8_0_dispatch_calls=0 packed_q8_0_dispatch_calls=1569 optimized_q2_dispatch_calls=0 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=0 shared_q3_dispatch_calls=0 optimized_q6_dispatch_calls=0 shared_q6_dispatch_calls=0`

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
| `batch/planner_equal` | 1950.416 | 6303.792 | 0.309x |
| `batch/planner_seq` | 2142.250 | 2748.541 | 0.779x |
| `batch/planner_simple` | 1046.917 | 2401.833 | 0.436x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15283.208 | 17509.709 | 0.873x |
| `gbnf/rule_parser_basic` | 495.458 | 262.875 | 1.885x |
| `gbnf/rule_parser_complex` | 3561.583 | 1427.125 | 2.496x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1` | 121724417.000 | 117844834.000 | 1.033x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_10` | 217225500.000 | 200427166.000 | 1.084x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_100` | 1298558625.000 | 1248997625.000 | 1.040x |
| `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1000` | 16027638250.000 | 17552014417.000 | 0.913x |
| `kernel/aarch64/op_add` | 107.166 | 5082.959 | 0.021x |
| `kernel/aarch64/op_cos` | 1680.750 | 18600.459 | 0.090x |
| `kernel/aarch64/op_div` | 111.083 | 4734.042 | 0.023x |
| `kernel/aarch64/op_dup` | 98.959 | 4355.375 | 0.023x |
| `kernel/aarch64/op_log` | 1880.208 | 6239.542 | 0.301x |
| `kernel/aarch64/op_mul` | 106.000 | 5195.208 | 0.020x |
| `kernel/aarch64/op_mul_mat` | 4778.541 | 10684.750 | 0.447x |
| `kernel/aarch64/op_sin` | 1315.834 | 5754.792 | 0.229x |
| `kernel/aarch64/op_soft_max` | 1058.000 | 5542.500 | 0.191x |
| `kernel/aarch64/op_sqr` | 99.083 | 4178.625 | 0.024x |
| `kernel/aarch64/op_sqrt` | 141.458 | 4472.708 | 0.032x |
| `kernel/aarch64/op_sub` | 110.667 | 5364.208 | 0.021x |
| `kernel/aarch64/op_unary_exp` | 20139.958 | 5539.334 | 3.636x |
| `kernel/aarch64/op_unary_neg` | 115.459 | 4438.000 | 0.026x |
| `kernel/aarch64/op_unary_relu` | 130.250 | 4354.000 | 0.030x |
| `logits/sampler_raw/vocab_128000` | 17597.167 | 19383.208 | 0.908x |
| `logits/sampler_raw/vocab_256000` | 37710.500 | 36900.375 | 1.022x |
| `logits/sampler_raw/vocab_32000` | 4723.417 | 4521.875 | 1.045x |
| `logits/sampler_sml/vocab_128000` | 17973.750 | 13914.542 | 1.292x |
| `logits/sampler_sml/vocab_256000` | 29899.083 | 29478.083 | 1.014x |
| `logits/sampler_sml/vocab_32000` | 3860.167 | 5635.208 | 0.685x |
| `logits/validator_raw/vocab_128000` | 90945.791 | 93147.041 | 0.976x |
| `logits/validator_raw/vocab_256000` | 182388.708 | 183596.083 | 0.993x |
| `logits/validator_raw/vocab_32000` | 24099.583 | 24349.916 | 0.990x |
| `logits/validator_sml/vocab_128000` | 100008.667 | 100125.792 | 0.999x |
| `logits/validator_sml/vocab_256000` | 199924.167 | 197569.916 | 1.012x |
| `logits/validator_sml/vocab_32000` | 24022.583 | 24585.917 | 0.977x |
| `memory/hybrid_full` | 436.833 | 35442.667 | 0.012x |
| `memory/kv_full` | 137.625 | 33617.041 | 0.004x |
| `memory/recurrent_full` | 135.916 | 4630.292 | 0.029x |
| `text/encoders/bpe_long` | 62.833 | 61.459 | 1.022x |
| `text/encoders/bpe_short` | 65.292 | 55.583 | 1.175x |
| `text/encoders/fallback_long` | 2512.583 | 2512.292 | 1.000x |
| `text/encoders/fallback_short` | 63.667 | 63.250 | 1.007x |
| `text/encoders/plamo2_long` | 7676.250 | 7609.750 | 1.009x |
| `text/encoders/plamo2_short` | 205.417 | 210.875 | 0.974x |
| `text/encoders/rwkv_long` | 835941.209 | 829637.250 | 1.008x |
| `text/encoders/rwkv_short` | 56820.875 | 56432.500 | 1.007x |
| `text/encoders/spm_long` | 3719743.667 | 3683735.375 | 1.010x |
| `text/encoders/spm_short` | 1368.458 | 1343.417 | 1.019x |
| `text/encoders/ugm_long` | 1422752.125 | 1409380.750 | 1.009x |
| `text/encoders/ugm_short` | 747.792 | 744.125 | 1.005x |
| `text/encoders/wpm_long` | 30331.125 | 30951.625 | 0.980x |
| `text/encoders/wpm_short` | 549.834 | 550.917 | 0.998x |
| `text/jinja/formatter_long` | 71.042 | 226554.000 | 0.000x |
| `text/jinja/formatter_short` | 20.000 | 3800.750 | 0.005x |
| `text/jinja/parser_long` | 66910.541 | 50401.375 | 1.328x |
| `text/jinja/parser_short` | 936.958 | 504.917 | 1.856x |
| `tokenizer/full_bpe_long` | 13639.375 | 13601.083 | 1.003x |
| `tokenizer/full_bpe_short` | 301.250 | 323.125 | 0.932x |
| `tokenizer/full_plamo2_long` | 12671.708 | 12460.417 | 1.017x |
| `tokenizer/full_plamo2_short` | 1971.334 | 1935.000 | 1.019x |
| `tokenizer/full_rwkv_long` | 863185.292 | 837551.458 | 1.031x |
| `tokenizer/full_rwkv_short` | 55442.625 | 55213.875 | 1.004x |
| `tokenizer/full_spm_long` | 3673130.208 | 3717953.416 | 0.988x |
| `tokenizer/full_spm_short` | 1536.875 | 1490.875 | 1.031x |
| `tokenizer/full_ugm_long` | 1395803.583 | 1428754.583 | 0.977x |
| `tokenizer/full_ugm_short` | 2466.833 | 2473.709 | 0.997x |
| `tokenizer/full_wpm_long` | 32173.667 | 32897.125 | 0.978x |
| `tokenizer/full_wpm_short` | 2394.917 | 2318.917 | 1.033x |
| `tokenizer/preprocessor_bpe_long` | 10416.625 | 5261.750 | 1.980x |
| `tokenizer/preprocessor_bpe_short` | 127.583 | 1728.958 | 0.074x |
| `tokenizer/preprocessor_plamo2_long` | 4090.083 | 5542.542 | 0.738x |
| `tokenizer/preprocessor_plamo2_short` | 2487.250 | 3586.959 | 0.693x |
| `tokenizer/preprocessor_rwkv_long` | 4008.167 | 5420.459 | 0.739x |
| `tokenizer/preprocessor_rwkv_short` | 2533.584 | 3653.250 | 0.694x |
| `tokenizer/preprocessor_spm_long` | 4171.667 | 5648.584 | 0.739x |
| `tokenizer/preprocessor_spm_short` | 2484.250 | 3592.041 | 0.692x |
| `tokenizer/preprocessor_ugm_long` | 4170.083 | 5656.875 | 0.737x |
| `tokenizer/preprocessor_ugm_short` | 2497.667 | 3614.917 | 0.691x |
| `tokenizer/preprocessor_wpm_long` | 4154.542 | 5614.042 | 0.740x |
| `tokenizer/preprocessor_wpm_short` | 2373.333 | 3590.334 | 0.661x |
