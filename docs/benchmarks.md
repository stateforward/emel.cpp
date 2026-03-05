# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 939186.833 | 8692.417 | 108.047x |
| `batch/planner_seq` | 3657264.416 | 3837.959 | 952.919x |
| `batch/planner_simple` | 1284.584 | 3602.000 | 0.357x |
| `gbnf/rule_parser_basic` | 2771.709 | 455.291 | 6.088x |
| `gbnf/rule_parser_complex` | 73000.333 | 2478.042 | 29.459x |
| `kernel/aarch64/op_add` | 100.417 | 5267.250 | 0.019x |
| `kernel/aarch64/op_cos` | 1795.500 | 5821.667 | 0.308x |
| `kernel/aarch64/op_div` | 98.375 | 4783.625 | 0.021x |
| `kernel/aarch64/op_dup` | 91.916 | 4228.833 | 0.022x |
| `kernel/aarch64/op_log` | 2105.042 | 6519.542 | 0.323x |
| `kernel/aarch64/op_mul` | 101.667 | 5172.500 | 0.020x |
| `kernel/aarch64/op_mul_mat` | 4914.584 | 10503.542 | 0.468x |
| `kernel/aarch64/op_sin` | 1551.917 | 5880.000 | 0.264x |
| `kernel/aarch64/op_soft_max` | 2981.917 | 5499.375 | 0.542x |
| `kernel/aarch64/op_sqr` | 91.000 | 4428.208 | 0.021x |
| `kernel/aarch64/op_sqrt` | 143.666 | 4686.375 | 0.031x |
| `kernel/aarch64/op_sub` | 91.458 | 5398.917 | 0.017x |
| `kernel/aarch64/op_unary_exp` | 1311.250 | 5855.375 | 0.224x |
| `kernel/aarch64/op_unary_neg` | 90.000 | 4515.375 | 0.020x |
| `kernel/aarch64/op_unary_relu` | 94.583 | 4562.041 | 0.021x |
| `logits/sampler_raw/vocab_128000` | 24341.833 | 18744.958 | 1.299x |
| `logits/sampler_raw/vocab_256000` | 35257.833 | 35719.625 | 0.987x |
| `logits/sampler_raw/vocab_32000` | 4933.417 | 5106.333 | 0.966x |
| `logits/sampler_sml/vocab_128000` | 16054.625 | 16349.666 | 0.982x |
| `logits/sampler_sml/vocab_256000` | 33690.792 | 27527.209 | 1.224x |
| `logits/sampler_sml/vocab_32000` | 4291.666 | 4106.875 | 1.045x |
| `logits/validator_raw/vocab_128000` | 90702.083 | 88182.583 | 1.029x |
| `logits/validator_raw/vocab_256000` | 192301.708 | 176266.208 | 1.091x |
| `logits/validator_raw/vocab_32000` | 23929.792 | 23373.000 | 1.024x |
| `logits/validator_sml/vocab_128000` | 113048.708 | 96825.250 | 1.168x |
| `logits/validator_sml/vocab_256000` | 199162.333 | 190301.666 | 1.047x |
| `logits/validator_sml/vocab_32000` | 24686.542 | 23527.083 | 1.049x |
| `memory/hybrid_full` | 449.666 | 37575.042 | 0.012x |
| `memory/kv_full` | 127.708 | 36796.250 | 0.003x |
| `memory/recurrent_full` | 145.417 | 5488.125 | 0.026x |
| `text/encoders/bpe_long` | 62.125 | 59.500 | 1.044x |
| `text/encoders/bpe_short` | 61.166 | 56.792 | 1.077x |
| `text/encoders/fallback_long` | 2362.666 | 2714.000 | 0.871x |
| `text/encoders/fallback_short` | 62.041 | 65.542 | 0.947x |
| `text/encoders/plamo2_long` | 7414.167 | 8680.458 | 0.854x |
| `text/encoders/plamo2_short` | 206.458 | 197.291 | 1.046x |
| `text/encoders/rwkv_long` | 808757.708 | 826598.292 | 0.978x |
| `text/encoders/rwkv_short` | 55517.584 | 56073.250 | 0.990x |
| `text/encoders/spm_long` | 3517185.917 | 3583048.250 | 0.982x |
| `text/encoders/spm_short` | 1289.708 | 1275.958 | 1.011x |
| `text/encoders/ugm_long` | 1359503.542 | 1389902.250 | 0.978x |
| `text/encoders/ugm_short` | 708.459 | 738.958 | 0.959x |
| `text/encoders/wpm_long` | 29647.708 | 30952.750 | 0.958x |
| `text/encoders/wpm_short` | 580.417 | 585.833 | 0.991x |
| `text/jinja/formatter_long` | 62.666 | 405529.125 | 0.000x |
| `text/jinja/formatter_short` | 16.208 | 6655.708 | 0.002x |
| `text/jinja/parser_long` | 189800.417 | 55849.458 | 3.398x |
| `text/jinja/parser_short` | 2228.208 | 660.625 | 3.373x |
| `tokenizer/full_bpe_long` | 13145.375 | 14264.333 | 0.922x |
| `tokenizer/full_bpe_short` | 319.375 | 306.542 | 1.042x |
| `tokenizer/full_plamo2_long` | 12418.000 | 12462.000 | 0.996x |
| `tokenizer/full_plamo2_short` | 2026.375 | 1903.416 | 1.065x |
| `tokenizer/full_rwkv_long` | 814398.250 | 814529.208 | 1.000x |
| `tokenizer/full_rwkv_short` | 54591.125 | 54274.542 | 1.006x |
| `tokenizer/full_spm_long` | 3509957.875 | 3563597.917 | 0.985x |
| `tokenizer/full_spm_short` | 1436.333 | 1495.250 | 0.961x |
| `tokenizer/full_ugm_long` | 1361935.792 | 1348696.458 | 1.010x |
| `tokenizer/full_ugm_short` | 2444.750 | 2365.791 | 1.033x |
| `tokenizer/full_wpm_long` | 31507.875 | 31614.542 | 0.997x |
| `tokenizer/full_wpm_short` | 2254.542 | 2244.708 | 1.004x |
| `tokenizer/preprocessor_bpe_long` | 3358.042 | 5341.625 | 0.629x |
| `tokenizer/preprocessor_bpe_short` | 134.125 | 1727.208 | 0.078x |
| `tokenizer/preprocessor_plamo2_long` | 3991.750 | 5544.500 | 0.720x |
| `tokenizer/preprocessor_plamo2_short` | 2412.292 | 3613.083 | 0.668x |
| `tokenizer/preprocessor_rwkv_long` | 4216.417 | 5511.833 | 0.765x |
| `tokenizer/preprocessor_rwkv_short` | 3026.209 | 3572.750 | 0.847x |
| `tokenizer/preprocessor_spm_long` | 5179.917 | 5299.041 | 0.978x |
| `tokenizer/preprocessor_spm_short` | 2459.750 | 3744.958 | 0.657x |
| `tokenizer/preprocessor_ugm_long` | 5050.041 | 5589.458 | 0.903x |
| `tokenizer/preprocessor_ugm_short` | 3144.458 | 3573.167 | 0.880x |
| `tokenizer/preprocessor_wpm_long` | 5034.084 | 5417.125 | 0.929x |
| `tokenizer/preprocessor_wpm_short` | 3096.416 | 3470.541 | 0.892x |
