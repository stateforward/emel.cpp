# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

## Current Flash Evidence

- Source snapshot: `snapshots/bench/benchmarks_compare.txt`
- Preserved baseline artifact: `snapshots/bench/generation_pre_flash_baseline.txt`
- `reference_impl: source=cmake_fetch ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `generation_flash_evidence: case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 flash_dispatch_calls=2 emel_decode_calls=0 emel_logits_calls=0 reference_decode_calls=0 reference_logits_calls=0`
- Current compare row: `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 emel.cpp 6995375.000 ns/op, llama.cpp 5146125.000 ns/op, ratio=1.359x`

## Pre-Flash Baseline Comparison

- `source_commit=2acd4fe^`
- `baseline_ref=ecbcb7ea9d3303097519723b264a8b5f1e977028`
- `case=generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `baseline_emel_ns=63837917.000`
- `baseline_reference_ns=10451583.000`
- `baseline_ratio=6.108x`
- `current_emel_ns=6995375.000`
- `current_reference_ns=5146125.000`
- `current_ratio=1.359x`
- `speedup=9.126x`
- `latency_drop_pct=89.0`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2208.000 | 10208.000 | 0.216x |
| `batch/planner_seq` | 2458.000 | 5167.000 | 0.476x |
| `batch/planner_simple` | 1250.000 | 5625.000 | 0.222x |
| `gbnf/rule_parser_basic` | 709.000 | 3042.000 | 0.233x |
| `gbnf/rule_parser_complex` | 4000.000 | 5750.000 | 0.696x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 6995375.000 | 5146125.000 | 1.359x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_8` | 51492917.000 | 15213333.000 | 3.385x |
| `kernel/aarch64/op_add` | 709.000 | 7250.000 | 0.098x |
| `kernel/aarch64/op_cos` | 1709.000 | 7000.000 | 0.244x |
| `kernel/aarch64/op_div` | 125.000 | 7375.000 | 0.017x |
| `kernel/aarch64/op_dup` | 333.000 | 6000.000 | 0.056x |
| `kernel/aarch64/op_log` | 2167.000 | 6750.000 | 0.321x |
| `kernel/aarch64/op_mul` | 291.000 | 8292.000 | 0.035x |
| `kernel/aarch64/op_mul_mat` | 7791.000 | 10917.000 | 0.714x |
| `kernel/aarch64/op_sin` | 1375.000 | 8959.000 | 0.153x |
| `kernel/aarch64/op_soft_max` | 9917.000 | 5750.000 | 1.725x |
| `kernel/aarch64/op_sqr` | 208.000 | 7583.000 | 0.027x |
| `kernel/aarch64/op_sqrt` | 7916.000 | 9333.000 | 0.848x |
| `kernel/aarch64/op_sub` | 11958.000 | 8041.000 | 1.487x |
| `kernel/aarch64/op_unary_exp` | 1625.000 | 7500.000 | 0.217x |
| `kernel/aarch64/op_unary_neg` | 7750.000 | 12917.000 | 0.600x |
| `kernel/aarch64/op_unary_relu` | 167.000 | 6833.000 | 0.024x |
| `logits/sampler_raw/vocab_128000` | 19000.000 | 20417.000 | 0.931x |
| `logits/sampler_raw/vocab_256000` | 41292.000 | 40375.000 | 1.023x |
| `logits/sampler_raw/vocab_32000` | 4708.000 | 7042.000 | 0.669x |
| `logits/sampler_sml/vocab_128000` | 19000.000 | 22708.000 | 0.837x |
| `logits/sampler_sml/vocab_256000` | 44042.000 | 40875.000 | 1.077x |
| `logits/sampler_sml/vocab_32000` | 5917.000 | 5958.000 | 0.993x |
| `logits/validator_raw/vocab_128000` | 94500.000 | 99833.000 | 0.947x |
| `logits/validator_raw/vocab_256000` | 195083.000 | 229084.000 | 0.852x |
| `logits/validator_raw/vocab_32000` | 23417.000 | 29625.000 | 0.790x |
| `logits/validator_sml/vocab_128000` | 94459.000 | 94500.000 | 1.000x |
| `logits/validator_sml/vocab_256000` | 187458.000 | 223667.000 | 0.838x |
| `logits/validator_sml/vocab_32000` | 22875.000 | 38917.000 | 0.588x |
| `memory/hybrid_full` | 1917.000 | 49750.000 | 0.039x |
| `memory/kv_full` | 583.000 | 39583.000 | 0.015x |
| `memory/recurrent_full` | 9666.000 | 8667.000 | 1.115x |
| `text/encoders/bpe_long` | 83.000 | 84.000 | 0.988x |
| `text/encoders/bpe_short` | 42.000 | 84.000 | 0.500x |
| `text/encoders/fallback_long` | 2250.000 | 2459.000 | 0.915x |
| `text/encoders/fallback_short` | 84.000 | 83.000 | 1.012x |
| `text/encoders/plamo2_long` | 7042.000 | 7500.000 | 0.939x |
| `text/encoders/plamo2_short` | 333.000 | 334.000 | 0.997x |
| `text/encoders/rwkv_long` | 836458.000 | 841167.000 | 0.994x |
| `text/encoders/rwkv_short` | 51209.000 | 63416.000 | 0.808x |
| `text/encoders/spm_long` | 3572834.000 | 3581458.000 | 0.998x |
| `text/encoders/spm_short` | 1458.000 | 1458.000 | 1.000x |
| `text/encoders/ugm_long` | 1382500.000 | 1355125.000 | 1.020x |
| `text/encoders/ugm_short` | 834.000 | 791.000 | 1.054x |
| `text/encoders/wpm_long` | 28125.000 | 30792.000 | 0.913x |
| `text/encoders/wpm_short` | 667.000 | 708.000 | 0.942x |
| `text/jinja/formatter_long` | 125.000 | 385375.000 | 0.000x |
| `text/jinja/formatter_short` | 167.000 | 558542.000 | 0.000x |
| `text/jinja/parser_long` | 66042.000 | 57125.000 | 1.156x |
| `text/jinja/parser_short` | 2750.000 | 1208.000 | 2.276x |
| `tokenizer/full_bpe_long` | 12417.000 | 13292.000 | 0.934x |
| `tokenizer/full_bpe_short` | 500.000 | 458.000 | 1.092x |
| `tokenizer/full_plamo2_long` | 11375.000 | 12125.000 | 0.938x |
| `tokenizer/full_plamo2_short` | 2416.000 | 2667.000 | 0.906x |
| `tokenizer/full_rwkv_long` | 792958.000 | 801958.000 | 0.989x |
| `tokenizer/full_rwkv_short` | 50875.000 | 62417.000 | 0.815x |
| `tokenizer/full_spm_long` | 3416041.000 | 3476791.000 | 0.983x |
| `tokenizer/full_spm_short` | 1792.000 | 1667.000 | 1.075x |
| `tokenizer/full_ugm_long` | 1406042.000 | 1412291.000 | 0.996x |
| `tokenizer/full_ugm_short` | 3041.000 | 3083.000 | 0.986x |
| `tokenizer/full_wpm_long` | 29625.000 | 31458.000 | 0.942x |
| `tokenizer/full_wpm_short` | 2958.000 | 2834.000 | 1.044x |
| `tokenizer/preprocessor_bpe_long` | 3125.000 | 5833.000 | 0.536x |
| `tokenizer/preprocessor_bpe_short` | 500.000 | 3583.000 | 0.140x |
| `tokenizer/preprocessor_plamo2_long` | 5875.000 | 6667.000 | 0.881x |
| `tokenizer/preprocessor_plamo2_short` | 3000.000 | 3916.000 | 0.766x |
| `tokenizer/preprocessor_rwkv_long` | 4709.000 | 5667.000 | 0.831x |
| `tokenizer/preprocessor_rwkv_short` | 3083.000 | 4125.000 | 0.747x |
| `tokenizer/preprocessor_spm_long` | 4334.000 | 7625.000 | 0.568x |
| `tokenizer/preprocessor_spm_short` | 2917.000 | 6208.000 | 0.470x |
| `tokenizer/preprocessor_ugm_long` | 4708.000 | 6792.000 | 0.693x |
| `tokenizer/preprocessor_ugm_short` | 2917.000 | 5000.000 | 0.583x |
| `tokenizer/preprocessor_wpm_long` | 4708.000 | 5916.000 | 0.796x |
| `tokenizer/preprocessor_wpm_short` | 2875.000 | 4083.000 | 0.704x |
