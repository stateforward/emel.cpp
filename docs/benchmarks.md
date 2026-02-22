# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1626.933 | 6278.408 | 0.259x |
| `batch/splitter_seq` | 1319.379 | 2638.238 | 0.500x |
| `batch/splitter_simple` | 738.408 | 2273.875 | 0.325x |
| `buffer/allocator_alloc_graph` | 16.671 | 55.083 | 0.303x |
| `buffer/allocator_full` | 37.625 | 252.400 | 0.149x |
| `buffer/allocator_reserve_n` | 19.971 | 442.804 | 0.045x |
| `jinja/parser_long` | 30502.542 | 49796.596 | 0.613x |
| `jinja/parser_short` | 388.525 | 491.550 | 0.790x |
| `jinja/renderer_long` | 89658.308 | 227931.921 | 0.393x |
| `jinja/renderer_short` | 1427.583 | 3803.167 | 0.375x |
| `memory/coordinator_recurrent_full` | 3895.246 | 5590.212 | 0.697x |
| `tokenizer/full_bpe_long` | 6621.133 | 7004.667 | 0.945x |
| `tokenizer/full_bpe_short` | 163.496 | 157.471 | 1.038x |
| `tokenizer/full_plamo2_long` | 10211.054 | 10239.642 | 0.997x |
| `tokenizer/full_plamo2_short` | 2205.075 | 1822.450 | 1.210x |
| `tokenizer/full_rwkv_long` | 2418.412 | 2436.733 | 0.992x |
| `tokenizer/full_rwkv_short` | 1854.350 | 2193.179 | 0.846x |
| `tokenizer/full_spm_long` | 9995.317 | 10792.767 | 0.926x |
| `tokenizer/full_spm_short` | 187.167 | 191.354 | 0.978x |
| `tokenizer/full_ugm_long` | 8868.146 | 8974.592 | 0.988x |
| `tokenizer/full_ugm_short` | 1738.117 | 2098.412 | 0.828x |
| `tokenizer/full_wpm_long` | 25314.525 | 25538.029 | 0.991x |
| `tokenizer/full_wpm_short` | 2077.092 | 2376.600 | 0.874x |
| `tokenizer/preprocessor_bpe_long` | 2776.758 | 5373.312 | 0.517x |
| `tokenizer/preprocessor_bpe_short` | 78.850 | 1747.050 | 0.045x |
| `tokenizer/preprocessor_plamo2_long` | 3082.279 | 4788.679 | 0.644x |
| `tokenizer/preprocessor_plamo2_short` | 2386.262 | 3548.504 | 0.672x |
| `tokenizer/preprocessor_rwkv_long` | 2972.246 | 4580.996 | 0.649x |
| `tokenizer/preprocessor_rwkv_short` | 2305.317 | 3535.229 | 0.652x |
| `tokenizer/preprocessor_spm_long` | 3046.325 | 4598.229 | 0.662x |
| `tokenizer/preprocessor_spm_short` | 2361.629 | 3762.438 | 0.628x |
| `tokenizer/preprocessor_ugm_long` | 3027.463 | 4692.613 | 0.645x |
| `tokenizer/preprocessor_ugm_short` | 2348.642 | 3552.613 | 0.661x |
| `tokenizer/preprocessor_wpm_long` | 2952.042 | 4562.908 | 0.647x |
| `tokenizer/preprocessor_wpm_short` | 2307.729 | 3534.338 | 0.653x |
