# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 1914.162 | 8509.350 | 0.225x |
| `batch/planner_seq` | 1771.867 | 3837.858 | 0.462x |
| `batch/planner_simple` | 1102.600 | 3480.183 | 0.317x |
| `gbnf/rule_parser_basic` | 255.033 | 509.908 | 0.500x |
| `gbnf/rule_parser_complex` | 2137.992 | 2502.092 | 0.854x |
| `kernel/aarch64/op_add` | 92.075 | 4993.925 | 0.018x |
| `kernel/aarch64/op_cos` | 1695.575 | 5819.554 | 0.291x |
| `kernel/aarch64/op_div` | 91.921 | 4147.679 | 0.022x |
| `kernel/aarch64/op_dup` | 89.721 | 4035.817 | 0.022x |
| `kernel/aarch64/op_log` | 1841.329 | 5724.712 | 0.322x |
| `kernel/aarch64/op_mul` | 91.275 | 4986.517 | 0.018x |
| `kernel/aarch64/op_mul_mat` | 4609.500 | 10211.246 | 0.451x |
| `kernel/aarch64/op_sin` | 1290.792 | 5297.721 | 0.244x |
| `kernel/aarch64/op_soft_max` | 2671.783 | 4716.729 | 0.566x |
| `kernel/aarch64/op_sqr` | 88.829 | 4018.213 | 0.022x |
| `kernel/aarch64/op_sqrt` | 143.512 | 4049.696 | 0.035x |
| `kernel/aarch64/op_sub` | 88.371 | 4973.954 | 0.018x |
| `kernel/aarch64/op_unary_exp` | 1311.688 | 5463.533 | 0.240x |
| `kernel/aarch64/op_unary_neg` | 89.646 | 3991.562 | 0.022x |
| `kernel/aarch64/op_unary_relu` | 90.733 | 4041.067 | 0.022x |
| `logits/sampler_raw/vocab_128000` | 19411.192 | 17715.379 | 1.096x |
| `logits/sampler_raw/vocab_256000` | 39433.942 | 36102.583 | 1.092x |
| `logits/sampler_raw/vocab_32000` | 4940.271 | 4715.096 | 1.048x |
| `logits/sampler_sml/vocab_128000` | 14892.267 | 14896.858 | 1.000x |
| `logits/sampler_sml/vocab_256000` | 32773.429 | 34911.417 | 0.939x |
| `logits/sampler_sml/vocab_32000` | 4146.125 | 4343.358 | 0.955x |
| `logits/validator_raw/vocab_128000` | 89360.583 | 87803.812 | 1.018x |
| `logits/validator_raw/vocab_256000` | 177996.733 | 175681.950 | 1.013x |
| `logits/validator_raw/vocab_32000` | 23643.392 | 23191.487 | 1.019x |
| `logits/validator_sml/vocab_128000` | 97684.042 | 96452.829 | 1.013x |
| `logits/validator_sml/vocab_256000` | 194364.033 | 194215.342 | 1.001x |
| `logits/validator_sml/vocab_32000` | 24360.554 | 23703.929 | 1.028x |
| `memory/hybrid_full` | 392.375 | 37552.908 | 0.010x |
| `memory/kv_full` | 99.042 | 35730.542 | 0.003x |
| `memory/recurrent_full` | 111.883 | 5469.400 | 0.020x |
| `text/encoders/bpe_long` | 36.383 | 36.817 | 0.988x |
| `text/encoders/bpe_short` | 35.179 | 38.308 | 0.918x |
| `text/encoders/fallback_long` | 2433.396 | 2429.300 | 1.002x |
| `text/encoders/fallback_short` | 47.817 | 46.042 | 1.039x |
| `text/encoders/plamo2_long` | 4846.517 | 4850.354 | 0.999x |
| `text/encoders/plamo2_short` | 108.521 | 102.588 | 1.058x |
| `text/encoders/rwkv_long` | 4602.983 | 4581.512 | 1.005x |
| `text/encoders/rwkv_short` | 2634.875 | 2652.379 | 0.993x |
| `text/encoders/spm_long` | 12609.517 | 12076.792 | 1.044x |
| `text/encoders/spm_short` | 201.842 | 198.750 | 1.016x |
| `text/encoders/ugm_long` | 8014.363 | 8006.896 | 1.001x |
| `text/encoders/ugm_short` | 131.696 | 130.004 | 1.013x |
| `text/encoders/wpm_long` | 26881.250 | 25872.704 | 1.039x |
| `text/encoders/wpm_short` | 518.579 | 530.850 | 0.977x |
| `text/jinja/formatter_long` | 61.046 | 405189.104 | 0.000x |
| `text/jinja/formatter_short` | 14.008 | 6275.858 | 0.002x |
| `text/jinja/parser_long` | 48445.537 | 54558.404 | 0.888x |
| `text/jinja/parser_short` | 1082.000 | 669.046 | 1.617x |
| `tokenizer/full_bpe_long` | 9423.121 | 9396.950 | 1.003x |
| `tokenizer/full_bpe_short` | 207.958 | 205.671 | 1.011x |
| `tokenizer/full_plamo2_long` | 9896.721 | 9657.438 | 1.025x |
| `tokenizer/full_plamo2_short` | 1744.612 | 1724.917 | 1.011x |
| `tokenizer/full_rwkv_long` | 3481.021 | 3457.188 | 1.007x |
| `tokenizer/full_rwkv_short` | 2097.375 | 2052.317 | 1.022x |
| `tokenizer/full_spm_long` | 13368.117 | 13457.521 | 0.993x |
| `tokenizer/full_spm_short` | 289.850 | 287.092 | 1.010x |
| `tokenizer/full_ugm_long` | 9706.896 | 9650.829 | 1.006x |
| `tokenizer/full_ugm_short` | 1741.371 | 2122.100 | 0.821x |
| `tokenizer/full_wpm_long` | 27606.900 | 27721.588 | 0.996x |
| `tokenizer/full_wpm_short` | 2164.846 | 2146.154 | 1.009x |
| `tokenizer/preprocessor_bpe_long` | 2804.700 | 5050.296 | 0.555x |
| `tokenizer/preprocessor_bpe_short` | 82.121 | 1711.450 | 0.048x |
| `tokenizer/preprocessor_plamo2_long` | 3040.642 | 4339.342 | 0.701x |
| `tokenizer/preprocessor_plamo2_short` | 2373.262 | 3418.700 | 0.694x |
| `tokenizer/preprocessor_rwkv_long` | 3058.175 | 4482.637 | 0.682x |
| `tokenizer/preprocessor_rwkv_short` | 2389.096 | 3412.058 | 0.700x |
| `tokenizer/preprocessor_spm_long` | 3063.608 | 4318.142 | 0.709x |
| `tokenizer/preprocessor_spm_short` | 2386.796 | 3404.767 | 0.701x |
| `tokenizer/preprocessor_ugm_long` | 3148.338 | 4404.400 | 0.715x |
| `tokenizer/preprocessor_ugm_short` | 2382.367 | 3418.375 | 0.697x |
| `tokenizer/preprocessor_wpm_long` | 3068.100 | 4371.492 | 0.702x |
| `tokenizer/preprocessor_wpm_short` | 2379.254 | 3391.992 | 0.701x |
