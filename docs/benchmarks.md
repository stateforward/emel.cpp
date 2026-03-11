# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/planner_equal` | 2042.000 | 10625.000 | 0.192x |
| `batch/planner_seq` | 2209.000 | 4042.000 | 0.547x |
| `batch/planner_simple` | 1209.000 | 6292.000 | 0.192x |
| `gbnf/rule_parser_basic` | 625.000 | 2750.000 | 0.227x |
| `gbnf/rule_parser_complex` | 3583.000 | 4125.000 | 0.869x |
| `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` | 63837917.000 | 10451583.000 | 6.108x |
| `kernel/aarch64/op_add` | 1542.000 | 6667.000 | 0.231x |
| `kernel/aarch64/op_cos` | 2208.000 | 5875.000 | 0.376x |
| `kernel/aarch64/op_div` | 541.000 | 6458.000 | 0.084x |
| `kernel/aarch64/op_dup` | 250.000 | 5084.000 | 0.049x |
| `kernel/aarch64/op_log` | 3167.000 | 5791.000 | 0.547x |
| `kernel/aarch64/op_mul` | 416.000 | 5917.000 | 0.070x |
| `kernel/aarch64/op_mul_mat` | 11167.000 | 10583.000 | 1.055x |
| `kernel/aarch64/op_sin` | 1542.000 | 5417.000 | 0.285x |
| `kernel/aarch64/op_soft_max` | 2500.000 | 4875.000 | 0.513x |
| `kernel/aarch64/op_sqr` | 417.000 | 4958.000 | 0.084x |
| `kernel/aarch64/op_sqrt` | 292.000 | 4417.000 | 0.066x |
| `kernel/aarch64/op_sub` | 500.000 | 14000.000 | 0.036x |
| `kernel/aarch64/op_unary_exp` | 2666.000 | 6042.000 | 0.441x |
| `kernel/aarch64/op_unary_neg` | 834.000 | 6000.000 | 0.139x |
| `kernel/aarch64/op_unary_relu` | 209.000 | 4167.000 | 0.050x |
| `logits/sampler_raw/vocab_128000` | 21250.000 | 17167.000 | 1.238x |
| `logits/sampler_raw/vocab_256000` | 27417.000 | 45916.000 | 0.597x |
| `logits/sampler_raw/vocab_32000` | 4542.000 | 5417.000 | 0.838x |
| `logits/sampler_sml/vocab_128000` | 14250.000 | 14125.000 | 1.009x |
| `logits/sampler_sml/vocab_256000` | 27875.000 | 42625.000 | 0.654x |
| `logits/sampler_sml/vocab_32000` | 5708.000 | 4958.000 | 1.151x |
| `logits/validator_raw/vocab_128000` | 92708.000 | 92875.000 | 0.998x |
| `logits/validator_raw/vocab_256000` | 185458.000 | 178958.000 | 1.036x |
| `logits/validator_raw/vocab_32000` | 22792.000 | 21458.000 | 1.062x |
| `logits/validator_sml/vocab_128000` | 92791.000 | 93417.000 | 0.993x |
| `logits/validator_sml/vocab_256000` | 185791.000 | 179083.000 | 1.037x |
| `logits/validator_sml/vocab_32000` | 22083.000 | 22334.000 | 0.989x |
| `memory/hybrid_full` | 1583.000 | 40083.000 | 0.039x |
| `memory/kv_full` | 1750.000 | 37667.000 | 0.046x |
| `memory/recurrent_full` | 1208.000 | 6625.000 | 0.182x |
| `text/encoders/bpe_long` | 83.000 | 84.000 | 0.988x |
| `text/encoders/bpe_short` | 84.000 | 125.000 | 0.672x |
| `text/encoders/fallback_long` | 2625.000 | 2667.000 | 0.984x |
| `text/encoders/fallback_short` | 83.000 | 83.000 | 1.000x |
| `text/encoders/plamo2_long` | 7834.000 | 7792.000 | 1.005x |
| `text/encoders/plamo2_short` | 333.000 | 334.000 | 0.997x |
| `text/encoders/rwkv_long` | 835417.000 | 736541.000 | 1.134x |
| `text/encoders/rwkv_short` | 56667.000 | 50000.000 | 1.133x |
| `text/encoders/spm_long` | 3393791.000 | 3427500.000 | 0.990x |
| `text/encoders/spm_short` | 1375.000 | 1417.000 | 0.970x |
| `text/encoders/ugm_long` | 1343459.000 | 1467625.000 | 0.915x |
| `text/encoders/ugm_short` | 875.000 | 1875.000 | 0.467x |
| `text/encoders/wpm_long` | 29625.000 | 30750.000 | 0.963x |
| `text/encoders/wpm_short` | 708.000 | 666.000 | 1.063x |
| `text/jinja/formatter_long` | 125.000 | 225041.000 | 0.001x |
| `text/jinja/formatter_short` | 208.000 | 38125.000 | 0.005x |
| `text/jinja/parser_long` | 60542.000 | 46083.000 | 1.314x |
| `text/jinja/parser_short` | 6458.000 | 750.000 | 8.611x |
| `tokenizer/full_bpe_long` | 14125.000 | 13917.000 | 1.015x |
| `tokenizer/full_bpe_short` | 583.000 | 500.000 | 1.166x |
| `tokenizer/full_plamo2_long` | 15833.000 | 12542.000 | 1.262x |
| `tokenizer/full_plamo2_short` | 3250.000 | 2500.000 | 1.300x |
| `tokenizer/full_rwkv_long` | 843750.000 | 824959.000 | 1.023x |
| `tokenizer/full_rwkv_short` | 56375.000 | 55667.000 | 1.013x |
| `tokenizer/full_spm_long` | 3564042.000 | 3571084.000 | 0.998x |
| `tokenizer/full_spm_short` | 2458.000 | 1792.000 | 1.372x |
| `tokenizer/full_ugm_long` | 1392125.000 | 1433750.000 | 0.971x |
| `tokenizer/full_ugm_short` | 3334.000 | 3500.000 | 0.953x |
| `tokenizer/full_wpm_long` | 33667.000 | 32833.000 | 1.025x |
| `tokenizer/full_wpm_short` | 3333.000 | 3083.000 | 1.081x |
| `tokenizer/preprocessor_bpe_long` | 3375.000 | 5208.000 | 0.648x |
| `tokenizer/preprocessor_bpe_short` | 333.000 | 3792.000 | 0.088x |
| `tokenizer/preprocessor_plamo2_long` | 4625.000 | 6667.000 | 0.694x |
| `tokenizer/preprocessor_plamo2_short` | 2833.000 | 4667.000 | 0.607x |
| `tokenizer/preprocessor_rwkv_long` | 4542.000 | 6916.000 | 0.657x |
| `tokenizer/preprocessor_rwkv_short` | 2792.000 | 4917.000 | 0.568x |
| `tokenizer/preprocessor_spm_long` | 5959.000 | 7625.000 | 0.782x |
| `tokenizer/preprocessor_spm_short` | 2917.000 | 6500.000 | 0.449x |
| `tokenizer/preprocessor_ugm_long` | 4750.000 | 7209.000 | 0.659x |
| `tokenizer/preprocessor_ugm_short` | 2625.000 | 6458.000 | 0.406x |
| `tokenizer/preprocessor_wpm_long` | 4500.000 | 6916.000 | 0.651x |
| `tokenizer/preprocessor_wpm_short` | 2709.000 | 5083.000 | 0.533x |
