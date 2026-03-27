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
- Current compare row: `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 emel.cpp 1375791.000 ns/op, llama.cpp 3006209.000 ns/op, ratio=0.458x`

## Current Quantized Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- `generation_runtime_contract: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 native_quantized=8 approved_dense_f32_by_contract=4 disallowed_fallback=0 explicit_no_claim=0`
- `generation_quantized_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 optimized_q2_dispatch_calls=8 shared_q2_dispatch_calls=0 optimized_q3_dispatch_calls=6 shared_q3_dispatch_calls=0 optimized_q6_dispatch_calls=1 shared_q6_dispatch_calls=0`

- Contract summary: the maintained canonical workload stayed on the approved runtime contract with native quantized matmul stages plus approved dense-f32-by-contract token-embedding and norm-vector seams; there was no disallowed fallback and no explicit no-claim branch on the supported path.

## Preserved ARM Flash Baseline Comparison

- `source_commit=3a5a4ee692912429a6d666bb709ec5934ef5655f`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=6995375.000`
- `baseline_reference_ns=5146125.000`
- `baseline_ratio=1.359x`
- `current_emel_ns=1375791.000`
- `current_reference_ns=3006209.000`
- `current_ratio=0.458x`
- `speedup=5.085x`
- `latency_drop_pct=80.3`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2200.708 | 6221.209 | 0.354x |
| `batch/planner_seq` | 2625.333 | 2669.042 | 0.984x |
| `batch/planner_simple` | 1025.875 | 2270.375 | 0.452x |
| `flash_attention/aarch64/op_flash_attn_ext_decode_like` | 15758.792 | 18112.875 | 0.870x |
| `gbnf/rule_parser_basic` | 483.750 | 277.584 | 1.743x |
| `gbnf/rule_parser_complex` | 3355.375 | 1479.375 | 2.268x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 1375791.000 | 3006209.000 | 0.458x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_10` | 13788250.000 | 16811000.000 | 0.820x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_100` | 139125333.000 | 156398708.000 | 0.890x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1000` | 1542089417.000 | 1750086292.000 | 0.881x |
| `kernel/aarch64/op_add` | 108.416 | 5349.084 | 0.020x |
| `kernel/aarch64/op_cos` | 1727.583 | 6766.541 | 0.255x |
| `kernel/aarch64/op_div` | 109.125 | 4936.375 | 0.022x |
| `kernel/aarch64/op_dup` | 102.333 | 4381.542 | 0.023x |
| `kernel/aarch64/op_log` | 1867.250 | 6488.375 | 0.288x |
| `kernel/aarch64/op_mul` | 107.916 | 5572.333 | 0.019x |
| `kernel/aarch64/op_mul_mat` | 4714.250 | 11052.583 | 0.427x |
| `kernel/aarch64/op_sin` | 1327.417 | 5753.417 | 0.231x |
| `kernel/aarch64/op_soft_max` | 1053.375 | 5389.875 | 0.195x |
| `kernel/aarch64/op_sqr` | 99.125 | 4931.583 | 0.020x |
| `kernel/aarch64/op_sqrt` | 143.292 | 5002.875 | 0.029x |
| `kernel/aarch64/op_sub` | 112.584 | 5528.625 | 0.020x |
| `kernel/aarch64/op_unary_exp` | 1326.625 | 5964.500 | 0.222x |
| `kernel/aarch64/op_unary_neg` | 119.250 | 4713.000 | 0.025x |
| `kernel/aarch64/op_unary_relu` | 132.708 | 4458.209 | 0.030x |
| `logits/sampler_raw/vocab_128000` | 19767.291 | 20513.250 | 0.964x |
| `logits/sampler_raw/vocab_256000` | 38920.125 | 36743.375 | 1.059x |
| `logits/sampler_raw/vocab_32000` | 5158.834 | 4961.917 | 1.040x |
| `logits/sampler_sml/vocab_128000` | 17178.875 | 14146.750 | 1.214x |
| `logits/sampler_sml/vocab_256000` | 32049.667 | 30232.750 | 1.060x |
| `logits/sampler_sml/vocab_32000` | 3624.958 | 4063.792 | 0.892x |
| `logits/validator_raw/vocab_128000` | 94824.750 | 92297.750 | 1.027x |
| `logits/validator_raw/vocab_256000` | 191876.666 | 188562.666 | 1.018x |
| `logits/validator_raw/vocab_32000` | 24527.792 | 24094.917 | 1.018x |
| `logits/validator_sml/vocab_128000` | 104854.209 | 99870.000 | 1.050x |
| `logits/validator_sml/vocab_256000` | 205330.000 | 197989.750 | 1.037x |
| `logits/validator_sml/vocab_32000` | 25097.917 | 24151.583 | 1.039x |
| `memory/hybrid_full` | 439.459 | 34456.292 | 0.013x |
| `memory/kv_full` | 145.333 | 33753.958 | 0.004x |
| `memory/recurrent_full` | 143.500 | 4596.000 | 0.031x |
| `text/encoders/bpe_long` | 63.208 | 63.292 | 0.999x |
| `text/encoders/bpe_short` | 57.833 | 57.208 | 1.011x |
| `text/encoders/fallback_long` | 2512.750 | 2488.417 | 1.010x |
| `text/encoders/fallback_short` | 59.875 | 63.625 | 0.941x |
| `text/encoders/plamo2_long` | 7658.500 | 7666.208 | 0.999x |
| `text/encoders/plamo2_short` | 207.541 | 202.375 | 1.026x |
| `text/encoders/rwkv_long` | 835311.792 | 834922.541 | 1.000x |
| `text/encoders/rwkv_short` | 56386.750 | 56691.292 | 0.995x |
| `text/encoders/spm_long` | 3640918.125 | 3626919.458 | 1.004x |
| `text/encoders/spm_short` | 1451.875 | 1306.875 | 1.111x |
| `text/encoders/ugm_long` | 1381454.291 | 1392619.875 | 0.992x |
| `text/encoders/ugm_short` | 702.375 | 747.041 | 0.940x |
| `text/encoders/wpm_long` | 30510.750 | 30731.708 | 0.993x |
| `text/encoders/wpm_short` | 529.875 | 518.000 | 1.023x |
| `text/jinja/formatter_long` | 60.375 | 227094.000 | 0.000x |
| `text/jinja/formatter_short` | 16.292 | 3829.000 | 0.004x |
| `text/jinja/parser_long` | 65053.917 | 48712.166 | 1.335x |
| `text/jinja/parser_short` | 920.875 | 489.333 | 1.882x |
| `tokenizer/full_bpe_long` | 13513.500 | 13494.417 | 1.001x |
| `tokenizer/full_bpe_short` | 328.250 | 309.334 | 1.061x |
| `tokenizer/full_plamo2_long` | 12516.750 | 12624.542 | 0.991x |
| `tokenizer/full_plamo2_short` | 1970.666 | 2480.333 | 0.795x |
| `tokenizer/full_rwkv_long` | 836135.750 | 832237.958 | 1.005x |
| `tokenizer/full_rwkv_short` | 55617.333 | 55375.667 | 1.004x |
| `tokenizer/full_spm_long` | 3625536.333 | 3626327.042 | 1.000x |
| `tokenizer/full_spm_short` | 1530.750 | 1511.416 | 1.013x |
| `tokenizer/full_ugm_long` | 1382505.750 | 1383240.875 | 0.999x |
| `tokenizer/full_ugm_short` | 2441.000 | 2417.333 | 1.010x |
| `tokenizer/full_wpm_long` | 32302.042 | 32673.208 | 0.989x |
| `tokenizer/full_wpm_short` | 2190.667 | 2253.292 | 0.972x |
| `tokenizer/preprocessor_bpe_long` | 3584.167 | 5378.458 | 0.666x |
| `tokenizer/preprocessor_bpe_short` | 127.292 | 1712.833 | 0.074x |
| `tokenizer/preprocessor_plamo2_long` | 5472.084 | 5384.292 | 1.016x |
| `tokenizer/preprocessor_plamo2_short` | 2584.459 | 3684.583 | 0.701x |
| `tokenizer/preprocessor_rwkv_long` | 5926.500 | 5467.167 | 1.084x |
| `tokenizer/preprocessor_rwkv_short` | 2468.875 | 3590.291 | 0.688x |
| `tokenizer/preprocessor_spm_long` | 4167.833 | 5443.708 | 0.766x |
| `tokenizer/preprocessor_spm_short` | 2520.917 | 3800.625 | 0.663x |
| `tokenizer/preprocessor_ugm_long` | 4361.750 | 5660.292 | 0.771x |
| `tokenizer/preprocessor_ugm_short` | 2550.709 | 3618.625 | 0.705x |
| `tokenizer/preprocessor_wpm_long` | 4138.875 | 5489.458 | 0.754x |
| `tokenizer/preprocessor_wpm_short` | 2461.334 | 3669.167 | 0.671x |
