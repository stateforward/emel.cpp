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
- Current compare row: `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 emel.cpp 4949333.000 ns/op, llama.cpp 5688875.000 ns/op, ratio=0.870x`

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
- `current_emel_ns=4949333.000`
- `current_reference_ns=5688875.000`
- `current_ratio=0.870x`
- `speedup=1.413x`
- `latency_drop_pct=29.2`

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2417.000 | 8166.000 | 0.296x |
| `batch/planner_seq` | 2666.000 | 3666.000 | 0.727x |
| `batch/planner_simple` | 1459.000 | 7125.000 | 0.205x |
| `gbnf/rule_parser_basic` | 750.000 | 2958.000 | 0.254x |
| `gbnf/rule_parser_complex` | 4125.000 | 4959.000 | 0.832x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 4949333.000 | 5688875.000 | 0.870x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_10` | 41555334.000 | 17504083.000 | 2.374x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_100` | 434737166.000 | 160401167.000 | 2.710x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1000` | 5400116583.000 | 1796303041.000 | 3.006x |
| `kernel/aarch64/op_add` | 542.000 | 7500.000 | 0.072x |
| `kernel/aarch64/op_cos` | 1709.000 | 7916.000 | 0.216x |
| `kernel/aarch64/op_div` | 250.000 | 38750.000 | 0.006x |
| `kernel/aarch64/op_dup` | 500.000 | 5625.000 | 0.089x |
| `kernel/aarch64/op_log` | 2208.000 | 7042.000 | 0.314x |
| `kernel/aarch64/op_mul` | 292.000 | 5792.000 | 0.050x |
| `kernel/aarch64/op_mul_mat` | 9833.000 | 11583.000 | 0.849x |
| `kernel/aarch64/op_sin` | 1375.000 | 6500.000 | 0.212x |
| `kernel/aarch64/op_soft_max` | 9500.000 | 6792.000 | 1.399x |
| `kernel/aarch64/op_sqr` | 8416.000 | 5875.000 | 1.433x |
| `kernel/aarch64/op_sqrt` | 375.000 | 5750.000 | 0.065x |
| `kernel/aarch64/op_sub` | 9458.000 | 5958.000 | 1.587x |
| `kernel/aarch64/op_unary_exp` | 1667.000 | 6208.000 | 0.269x |
| `kernel/aarch64/op_unary_neg` | 8834.000 | 6583.000 | 1.342x |
| `kernel/aarch64/op_unary_relu` | 167.000 | 5500.000 | 0.030x |
| `logits/sampler_raw/vocab_128000` | 18250.000 | 30792.000 | 0.593x |
| `logits/sampler_raw/vocab_256000` | 44166.000 | 44916.000 | 0.983x |
| `logits/sampler_raw/vocab_32000` | 5917.000 | 3792.000 | 1.560x |
| `logits/sampler_sml/vocab_128000` | 15917.000 | 22750.000 | 0.700x |
| `logits/sampler_sml/vocab_256000` | 34708.000 | 48083.000 | 0.722x |
| `logits/sampler_sml/vocab_32000` | 5125.000 | 6333.000 | 0.809x |
| `logits/validator_raw/vocab_128000` | 96667.000 | 96750.000 | 0.999x |
| `logits/validator_raw/vocab_256000` | 204791.000 | 191959.000 | 1.067x |
| `logits/validator_raw/vocab_32000` | 23459.000 | 23041.000 | 1.018x |
| `logits/validator_sml/vocab_128000` | 96709.000 | 96875.000 | 0.998x |
| `logits/validator_sml/vocab_256000` | 191958.000 | 202666.000 | 0.947x |
| `logits/validator_sml/vocab_32000` | 23833.000 | 22125.000 | 1.077x |
| `memory/hybrid_full` | 1125.000 | 39333.000 | 0.029x |
| `memory/kv_full` | 17292.000 | 38584.000 | 0.448x |
| `memory/recurrent_full` | 625.000 | 6375.000 | 0.098x |
| `text/encoders/bpe_long` | 83.000 | 83.000 | 1.000x |
| `text/encoders/bpe_short` | 83.000 | 84.000 | 0.988x |
| `text/encoders/fallback_long` | 2583.000 | 2542.000 | 1.016x |
| `text/encoders/fallback_short` | 84.000 | 83.000 | 1.012x |
| `text/encoders/plamo2_long` | 9541.000 | 7708.000 | 1.238x |
| `text/encoders/plamo2_short` | 375.000 | 333.000 | 1.126x |
| `text/encoders/rwkv_long` | 855208.000 | 842458.000 | 1.015x |
| `text/encoders/rwkv_short` | 60417.000 | 55958.000 | 1.080x |
| `text/encoders/spm_long` | 3642291.000 | 3414500.000 | 1.067x |
| `text/encoders/spm_short` | 1584.000 | 1417.000 | 1.118x |
| `text/encoders/ugm_long` | 1400375.000 | 1389458.000 | 1.008x |
| `text/encoders/ugm_short` | 791.000 | 834.000 | 0.948x |
| `text/encoders/wpm_long` | 35459.000 | 30750.000 | 1.153x |
| `text/encoders/wpm_short` | 708.000 | 708.000 | 1.000x |
| `text/jinja/formatter_long` | 125.000 | 248750.000 | 0.001x |
| `text/jinja/formatter_short` | 167.000 | 481542.000 | 0.000x |
| `text/jinja/parser_long` | 65500.000 | 49375.000 | 1.327x |
| `text/jinja/parser_short` | 2500.000 | 875.000 | 2.857x |
| `tokenizer/full_bpe_long` | 13084.000 | 13875.000 | 0.943x |
| `tokenizer/full_bpe_short` | 459.000 | 500.000 | 0.918x |
| `tokenizer/full_plamo2_long` | 15791.000 | 12792.000 | 1.234x |
| `tokenizer/full_plamo2_short` | 2625.000 | 2709.000 | 0.969x |
| `tokenizer/full_rwkv_long` | 827917.000 | 841750.000 | 0.984x |
| `tokenizer/full_rwkv_short` | 55333.000 | 55250.000 | 1.002x |
| `tokenizer/full_spm_long` | 3581791.000 | 3425666.000 | 1.046x |
| `tokenizer/full_spm_short` | 1792.000 | 1583.000 | 1.132x |
| `tokenizer/full_ugm_long` | 1358250.000 | 1374458.000 | 0.988x |
| `tokenizer/full_ugm_short` | 3250.000 | 4000.000 | 0.812x |
| `tokenizer/full_wpm_long` | 33042.000 | 32875.000 | 1.005x |
| `tokenizer/full_wpm_short` | 3083.000 | 3250.000 | 0.949x |
| `tokenizer/preprocessor_bpe_long` | 4208.000 | 5791.000 | 0.727x |
| `tokenizer/preprocessor_bpe_short` | 333.000 | 3625.000 | 0.092x |
| `tokenizer/preprocessor_plamo2_long` | 4458.000 | 7083.000 | 0.629x |
| `tokenizer/preprocessor_plamo2_short` | 3000.000 | 5083.000 | 0.590x |
| `tokenizer/preprocessor_rwkv_long` | 4500.000 | 7166.000 | 0.628x |
| `tokenizer/preprocessor_rwkv_short` | 2916.000 | 5208.000 | 0.560x |
| `tokenizer/preprocessor_spm_long` | 4500.000 | 8167.000 | 0.551x |
| `tokenizer/preprocessor_spm_short` | 3875.000 | 6667.000 | 0.581x |
| `tokenizer/preprocessor_ugm_long` | 4542.000 | 7459.000 | 0.609x |
| `tokenizer/preprocessor_ugm_short` | 2625.000 | 6042.000 | 0.434x |
| `tokenizer/preprocessor_wpm_long` | 4625.000 | 7000.000 | 0.661x |
| `tokenizer/preprocessor_wpm_short` | 2875.000 | 5167.000 | 0.556x |
