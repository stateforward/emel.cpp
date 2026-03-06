# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2118.209 | 8819.625 | 0.240x |
| `batch/planner_seq` | 2415.000 | 3955.625 | 0.611x |
| `batch/planner_simple` | 1288.375 | 3550.250 | 0.363x |
| `gbnf/rule_parser_basic` | 488.709 | 500.417 | 0.977x |
| `gbnf/rule_parser_complex` | 3341.584 | 2556.625 | 1.307x |
| `kernel/aarch64/op_add` | 97.042 | 5402.375 | 0.018x |
| `kernel/aarch64/op_cos` | 1673.875 | 5981.209 | 0.280x |
| `kernel/aarch64/op_div` | 94.667 | 4337.250 | 0.022x |
| `kernel/aarch64/op_dup` | 93.125 | 4470.666 | 0.021x |
| `kernel/aarch64/op_log` | 1847.875 | 6028.375 | 0.307x |
| `kernel/aarch64/op_mul` | 94.917 | 5094.000 | 0.019x |
| `kernel/aarch64/op_mul_mat` | 4612.916 | 10541.083 | 0.438x |
| `kernel/aarch64/op_sin` | 1302.583 | 5466.125 | 0.238x |
| `kernel/aarch64/op_soft_max` | 2127.541 | 4986.583 | 0.427x |
| `kernel/aarch64/op_sqr` | 89.166 | 4615.583 | 0.019x |
| `kernel/aarch64/op_sqrt` | 159.250 | 4502.000 | 0.035x |
| `kernel/aarch64/op_sub` | 105.041 | 5300.667 | 0.020x |
| `kernel/aarch64/op_unary_exp` | 1287.416 | 5662.750 | 0.227x |
| `kernel/aarch64/op_unary_neg` | 92.000 | 4548.125 | 0.020x |
| `kernel/aarch64/op_unary_relu` | 91.583 | 4402.000 | 0.021x |
| `logits/sampler_raw/vocab_128000` | 18198.292 | 19903.166 | 0.914x |
| `logits/sampler_raw/vocab_256000` | 37163.542 | 37863.000 | 0.982x |
| `logits/sampler_raw/vocab_32000` | 4314.083 | 5266.416 | 0.819x |
| `logits/sampler_sml/vocab_128000` | 15645.000 | 16619.416 | 0.941x |
| `logits/sampler_sml/vocab_256000` | 31608.250 | 36052.792 | 0.877x |
| `logits/sampler_sml/vocab_32000` | 4203.042 | 4521.166 | 0.930x |
| `logits/validator_raw/vocab_128000` | 91549.042 | 91699.917 | 0.998x |
| `logits/validator_raw/vocab_256000` | 181266.750 | 181318.667 | 1.000x |
| `logits/validator_raw/vocab_32000` | 24441.833 | 24148.834 | 1.012x |
| `logits/validator_sml/vocab_128000` | 103882.959 | 99569.292 | 1.043x |
| `logits/validator_sml/vocab_256000` | 197125.750 | 196076.791 | 1.005x |
| `logits/validator_sml/vocab_32000` | 24381.584 | 24261.709 | 1.005x |
| `memory/hybrid_full` | 460.291 | 38673.375 | 0.012x |
| `memory/kv_full` | 133.500 | 37028.000 | 0.004x |
| `memory/recurrent_full` | 151.375 | 5674.917 | 0.027x |
| `text/encoders/bpe_long` | 64.458 | 64.708 | 0.996x |
| `text/encoders/bpe_short` | 57.917 | 60.125 | 0.963x |
| `text/encoders/fallback_long` | 2501.292 | 2497.125 | 1.002x |
| `text/encoders/fallback_short` | 63.500 | 64.875 | 0.979x |
| `text/encoders/plamo2_long` | 7480.042 | 7749.459 | 0.965x |
| `text/encoders/plamo2_short` | 224.750 | 205.500 | 1.094x |
| `text/encoders/rwkv_long` | 832029.875 | 829474.083 | 1.003x |
| `text/encoders/rwkv_short` | 56237.375 | 56070.208 | 1.003x |
| `text/encoders/spm_long` | 3606311.333 | 3610968.791 | 0.999x |
| `text/encoders/spm_short` | 1284.750 | 1347.500 | 0.953x |
| `text/encoders/ugm_long` | 1372310.333 | 1372264.667 | 1.000x |
| `text/encoders/ugm_short` | 745.709 | 721.166 | 1.034x |
| `text/encoders/wpm_long` | 30396.375 | 30763.584 | 0.988x |
| `text/encoders/wpm_short` | 610.291 | 617.208 | 0.989x |
| `text/jinja/formatter_long` | 61.875 | 415277.375 | 0.000x |
| `text/jinja/formatter_short` | 16.750 | 6741.833 | 0.002x |
| `text/jinja/parser_long` | 67714.959 | 55710.750 | 1.215x |
| `text/jinja/parser_short` | 962.125 | 634.375 | 1.517x |
| `tokenizer/full_bpe_long` | 13396.667 | 13712.916 | 0.977x |
| `tokenizer/full_bpe_short` | 295.333 | 321.084 | 0.920x |
| `tokenizer/full_plamo2_long` | 12623.584 | 12354.959 | 1.022x |
| `tokenizer/full_plamo2_short` | 1959.500 | 1954.041 | 1.003x |
| `tokenizer/full_rwkv_long` | 828026.500 | 833893.958 | 0.993x |
| `tokenizer/full_rwkv_short` | 55338.917 | 55265.791 | 1.001x |
| `tokenizer/full_spm_long` | 3616312.167 | 3609337.833 | 1.002x |
| `tokenizer/full_spm_short` | 1455.458 | 1455.583 | 1.000x |
| `tokenizer/full_ugm_long` | 1378101.959 | 1378477.459 | 1.000x |
| `tokenizer/full_ugm_short` | 2441.291 | 2435.416 | 1.002x |
| `tokenizer/full_wpm_long` | 32383.291 | 32134.792 | 1.008x |
| `tokenizer/full_wpm_short` | 2371.333 | 2383.208 | 0.995x |
| `tokenizer/preprocessor_bpe_long` | 3391.917 | 5285.375 | 0.642x |
| `tokenizer/preprocessor_bpe_short` | 125.417 | 1743.541 | 0.072x |
| `tokenizer/preprocessor_plamo2_long` | 4089.208 | 5547.459 | 0.737x |
| `tokenizer/preprocessor_plamo2_short` | 2427.250 | 3579.250 | 0.678x |
| `tokenizer/preprocessor_rwkv_long` | 4088.458 | 5509.833 | 0.742x |
| `tokenizer/preprocessor_rwkv_short` | 2465.500 | 3541.583 | 0.696x |
| `tokenizer/preprocessor_spm_long` | 4116.792 | 5397.875 | 0.763x |
| `tokenizer/preprocessor_spm_short` | 2488.292 | 3512.250 | 0.708x |
| `tokenizer/preprocessor_ugm_long` | 4212.041 | 5767.708 | 0.730x |
| `tokenizer/preprocessor_ugm_short` | 2498.209 | 3701.083 | 0.675x |
| `tokenizer/preprocessor_wpm_long` | 4130.833 | 5446.917 | 0.758x |
| `tokenizer/preprocessor_wpm_short` | 2526.625 | 3654.208 | 0.691x |
