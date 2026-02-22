# Benchmarks

Source: `snapshots/bench/benchmarks_compare.txt`

Note: While EMEL is modular and easy to bench in isolation, llama.cpp code is very
entangled. These microbenches aim for apples-to-apples comparisons but likely
are not. True benchmarks will be end-to-end once the system is complete.

| Benchmark | emel.cpp ns/op | llama.cpp ns/op | ratio |
| --- | ---: | ---: | ---: |
| `batch/splitter_equal` | 1659.904 | 6553.800 | 0.253x |
| `batch/splitter_seq` | 1454.075 | 2737.262 | 0.531x |
| `batch/splitter_simple` | 723.458 | 2305.613 | 0.314x |
| `buffer/allocator_alloc_graph` | 17.746 | 56.062 | 0.317x |
| `buffer/allocator_full` | 39.671 | 255.167 | 0.155x |
| `buffer/allocator_reserve_n` | 21.671 | 440.821 | 0.049x |
| `jinja/parser_long` | 31719.267 | 49052.071 | 0.647x |
| `jinja/parser_short` | 400.621 | 510.283 | 0.785x |
| `jinja/renderer_long` | 91641.196 | 225327.517 | 0.407x |
| `jinja/renderer_short` | 1470.879 | 3814.088 | 0.386x |
| `memory/coordinator_recurrent_full` | 3710.733 | 5472.954 | 0.678x |
| `tokenizer/preprocessor_bpe_long` | 15887.842 | 16836.250 | 0.944x |
| `tokenizer/preprocessor_bpe_short` | 499.233 | 717.212 | 0.696x |
| `tokenizer/preprocessor_rwkv_long` | 3153.838 | 4604.458 | 0.685x |
| `tokenizer/preprocessor_rwkv_short` | 2444.925 | 3527.350 | 0.693x |
| `tokenizer/preprocessor_spm_long` | 3128.225 | 4381.325 | 0.714x |
| `tokenizer/preprocessor_spm_short` | 2422.354 | 3405.854 | 0.711x |
| `tokenizer/preprocessor_ugm_long` | 3195.704 | 4491.029 | 0.712x |
| `tokenizer/preprocessor_ugm_short` | 2421.363 | 3530.558 | 0.686x |
| `tokenizer/preprocessor_wpm_long` | 3157.667 | 4646.071 | 0.680x |
| `tokenizer/preprocessor_wpm_short` | 2432.958 | 3595.879 | 0.677x |
