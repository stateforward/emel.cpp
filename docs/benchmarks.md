# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1571.208 | 6402.871 | 0.245x |
| `batch/splitter_seq` | 1432.146 | 2653.508 | 0.540x |
| `batch/splitter_simple` | 742.508 | 2275.896 | 0.326x |
| `buffer/allocator_alloc_graph` | 17.113 | 54.221 | 0.316x |
| `buffer/allocator_full` | 39.987 | 251.492 | 0.159x |
| `buffer/allocator_reserve_n` | 20.696 | 426.308 | 0.049x |
| `jinja/parser_long` | 31008.346 | 50653.367 | 0.612x |
| `jinja/parser_short` | 408.267 | 499.417 | 0.817x |
| `jinja/renderer_long` | 91826.917 | 232903.558 | 0.394x |
| `jinja/renderer_short` | 1429.925 | 3899.846 | 0.367x |
| `memory/coordinator_recurrent_full` | 3818.421 | 5427.163 | 0.704x |
| `tokenizer/preprocessor_bpe_long` | 15788.421 | 16948.079 | 0.932x |
| `tokenizer/preprocessor_bpe_short` | 498.583 | 727.483 | 0.685x |
| `tokenizer/preprocessor_plamo2_long` | 3074.358 | 4663.188 | 0.659x |
| `tokenizer/preprocessor_plamo2_short` | 2439.450 | 3594.912 | 0.679x |
| `tokenizer/preprocessor_rwkv_long` | 3167.096 | 4548.988 | 0.696x |
| `tokenizer/preprocessor_rwkv_short` | 2448.875 | 3436.583 | 0.713x |
| `tokenizer/preprocessor_spm_long` | 3041.883 | 4731.308 | 0.643x |
| `tokenizer/preprocessor_spm_short` | 2355.250 | 3643.054 | 0.647x |
| `tokenizer/preprocessor_ugm_long` | 3117.054 | 4778.796 | 0.652x |
| `tokenizer/preprocessor_ugm_short` | 2346.454 | 3618.200 | 0.649x |
| `tokenizer/preprocessor_wpm_long` | 3025.596 | 4807.304 | 0.629x |
| `tokenizer/preprocessor_wpm_short` | 2352.137 | 3642.429 | 0.646x |
